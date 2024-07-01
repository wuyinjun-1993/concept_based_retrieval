import torch
from typing import List
import pickle

class TinyTable:
    def __init__(self, num_tables: int, hash_range: int, num_elements: int, hashes: torch.Tensor, device):
        self._device = device
        self._hash_range = hash_range
        self._num_elements = num_elements
        self._num_tables = num_tables
        self._table_start = self._num_tables * (self._hash_range + 1)
        self._index = torch.zeros((self._table_start + self._num_elements * self._num_tables,), dtype=torch.int32, device=device)

        for table in range(num_tables):
            temp_buckets = [[] for _ in range(hash_range)]
            for vec_id in range(num_elements):
                hash_value = hashes[vec_id * num_tables + table].item()
                temp_buckets[hash_value].append(vec_id)

            table_offsets_start = table * (self._hash_range + 1)
            self._index[table_offsets_start + 1:table_offsets_start + self._hash_range + 1] = torch.cumsum(torch.tensor([len(temp_buckets[i]) for i in range(hash_range)], device=device), dim=0)

            current_offset = self._table_start + self._num_elements * table
            for bucket in range(hash_range):
                end_offset = current_offset + len(temp_buckets[bucket])
                self._index[current_offset:end_offset] = torch.tensor(temp_buckets[bucket], dtype=torch.int32, device=device)
                current_offset = end_offset

    def query_by_count(self, hashes: torch.Tensor, hash_offset: int, counts: torch.Tensor):
        for table in range(self._num_tables):
            hash_value = hashes[hash_offset + table].item()
            start_offset = self._index[(self._hash_range + 1) * table + hash_value].item()
            end_offset = self._index[(self._hash_range + 1) * table + hash_value + 1].item()
            table_offset = self._table_start + table * self._num_elements
            counts.index_add_(0, self._index[table_offset + start_offset:table_offset + end_offset], torch.ones(end_offset - start_offset, dtype=counts.dtype, device=self._device))

    def num_tables(self) -> int:
        return self._num_tables

    def num_elements(self) -> int:
        return self._num_elements

class SparseRandomProjection:
    def __init__(self, input_dim: int, srps_per_table: int, num_tables: int, device):
        self._device = device
        self._num_tables = num_tables
        self._srps_per_table = srps_per_table
        self._total_num_srps = srps_per_table * num_tables
        self._dim = input_dim
        self._sample_size = int(torch.ceil(torch.tensor(self._dim * 0.3, device=device)).item())
        assert srps_per_table < 32

        a = torch.arange(self._dim, device=device)

        self._random_bits = torch.zeros(self._total_num_srps * self._sample_size, dtype=torch.int16, device=device)
        self._hash_indices = torch.zeros(self._total_num_srps * self._sample_size, dtype=torch.int32, device=device)

        for i in range(self._total_num_srps):
            a = a[torch.randperm(a.size(0))]
            self._hash_indices[i * self._sample_size:(i + 1) * self._sample_size] = torch.sort(a[:self._sample_size]).values
            self._random_bits[i * self._sample_size:(i + 1) * self._sample_size] = (torch.randint(0, 2, (self._sample_size,), device=device) * 2 - 1)
        del a

    def hash_single_dense(self, values: torch.Tensor, dim: int, output: torch.Tensor):
        assert values.size(0) == dim
        hash_indices = self._hash_indices.view(self._num_tables, self._srps_per_table, self._sample_size)
        random_bits = self._random_bits.view(self._num_tables, self._srps_per_table, self._sample_size)
        gathered_values = values[hash_indices]
        products = gathered_values * random_bits
        sums = torch.sum(products, dim=2)
        binary_values = (sums > 0).int()
        powers_of_two = 2 ** torch.arange(self._srps_per_table, device=values.device)
        table_sums = torch.sum(binary_values * powers_of_two, dim=1)
        output[:] = table_sums

    def num_tables(self) -> int:
        return self._num_tables

    def range(self) -> int:
        return 1 << self._srps_per_table

class MaxFlash:
    def __init__(self, num_tables: int, hash_range: int, num_elements: int, hashes: torch.Tensor, device):
        self._hashtable = TinyTable(num_tables, hash_range, num_elements, hashes, device)

    def get_score(self, query_hashes: torch.Tensor, num_elements: int, count_buffer: torch.Tensor, collision_count_to_sim: torch.Tensor) -> float:
        results = torch.zeros(num_elements, dtype=torch.int32, device=query_hashes.device)
        assert len(count_buffer) >= self._hashtable.num_elements()
        parallel_count_buffer = torch.zeros((num_elements, self._hashtable.num_elements()), dtype=torch.int32, device=query_hashes.device)
        hash_offsets = torch.arange(num_elements, device=query_hashes.device) * self._hashtable.num_tables()
        for vec_id in range(num_elements):
            self._hashtable.query_by_count(query_hashes, vec_id * self._hashtable.num_tables(), parallel_count_buffer[vec_id])
        max_counts = torch.max(parallel_count_buffer, dim=1)[0]
        results[:] = max_counts
        sum_sim = torch.sum(collision_count_to_sim[results]).item()
        return sum_sim

class MaxFlashArray:
    def __init__(self, function, hashes_per_table: int, max_doc_size: int, device):
        self._device = device
        self._max_allowable_doc_size = max_doc_size
        self._hash_function = function
        self._maxflash_array = []
        self._collision_count_to_sim = torch.zeros(self._hash_function.num_tables() + 1, dtype=torch.float32, device=device)

        for collision_count in range(self._collision_count_to_sim.size(0)):
            table_collision_probability = float(collision_count) / self._hash_function.num_tables()
            if table_collision_probability > 0:
                self._collision_count_to_sim[collision_count] = torch.exp(torch.log(torch.tensor(table_collision_probability, device=device)) / hashes_per_table)
            else:
                self._collision_count_to_sim[collision_count] = 0.0

    def add_document(self, batch: torch.Tensor) -> int:
        num_vectors = batch.size(0)
        num_elements = min(num_vectors, self._max_allowable_doc_size)
        hashes = self.hash(batch)
        self._maxflash_array.append(MaxFlash(self._hash_function.num_tables(), self._hash_function.range(), num_elements, hashes, self._device))
        return len(self._maxflash_array) - 1

    def get_document_scores(self, query: torch.Tensor, documents_to_query: torch.Tensor):
        hashes = self.hash(query)
        num_vectors_in_query = query.size(0)
        num_docs_to_query = documents_to_query.size(0)

        buffers = torch.zeros((num_docs_to_query, self._max_allowable_doc_size), dtype=torch.int32, device=query.device)
        results = torch.zeros(num_docs_to_query, dtype=torch.float32, device=query.device)

        for i, flash_index in enumerate(documents_to_query):
            score = self._maxflash_array[flash_index].get_score(hashes, num_vectors_in_query, buffers[i], self._collision_count_to_sim)
            results[i] = score / num_vectors_in_query

        return results

    def hash(self, batch: torch.Tensor) -> torch.Tensor:
        num_vectors, dim = batch.size()
        output = torch.zeros((num_vectors, self._hash_function.num_tables()), dtype=torch.int32, device=batch.device)

        for i in range(num_vectors):
            self._hash_function.hash_single_dense(batch[i], dim, output[i])

        return output.view(-1)

class DocRetrieval:
    def __init__(self, hashes_per_table: int, num_tables: int, dense_input_dimension: int, centroids: torch.Tensor, max_doc_size: int, device: str):
        if (device == 'cuda'):
          self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
          self._device = torch.device("cpu")
        self._dense_dim = dense_input_dimension
        self._nprobe_query = 2
        self._largest_internal_id = 0
        self._num_centroids = centroids.shape[0]
        self._centroid_id_to_internal_id = [torch.empty(0, dtype=torch.int32, device=self._device) for _ in range(self._num_centroids)]
        self._internal_id_to_doc_id: List[str] = []

        if dense_input_dimension == 0 or num_tables == 0 or hashes_per_table == 0:
            raise ValueError("The dense dimension, number of tables, and hashes per table must all be greater than 0.")
        if self._num_centroids == 0:
            raise ValueError("Must pass in at least one centroid, found 0.")
        if centroids.shape[1] != self._dense_dim:
            raise ValueError("The centroids array must have dimension equal to dense_dim.")

        self._nprobe_query = min(len(centroids), self._nprobe_query)

        self._document_array = MaxFlashArray(SparseRandomProjection(dense_input_dimension, hashes_per_table, num_tables, self._device), hashes_per_table, max_doc_size, self._device)
        self._centroids = centroids.T.to(self._device)

    def add_doc(self, doc_embeddings: torch.Tensor, doc_id: str) -> bool:
        centroid_ids = self.getNearestCentroids(doc_embeddings, 1)
        return self.add_doc_with_centroids(doc_embeddings, doc_id, centroid_ids)

    def add_doc_with_centroids(self, doc_embeddings: torch.Tensor, doc_id: str, doc_centroid_ids: torch.Tensor) -> bool:
        internal_id = self._document_array.add_document(doc_embeddings.to(self._device))
        self._largest_internal_id = max(self._largest_internal_id, internal_id)

        for centroid_id in doc_centroid_ids:
            self._centroid_id_to_internal_id[centroid_id] = torch.cat((self._centroid_id_to_internal_id[centroid_id], torch.tensor([internal_id], dtype=torch.int32, device=self._device)))

        if internal_id >= len(self._internal_id_to_doc_id):
            self._internal_id_to_doc_id.extend([None] * (internal_id + 1 - len(self._internal_id_to_doc_id)))
        self._internal_id_to_doc_id[internal_id] = doc_id

        return True

    def query(self, query_embeddings: torch.Tensor, top_k: int, num_to_rerank: int):
        centroid_ids = self.getNearestCentroids(query_embeddings, self._nprobe_query)
        return self.query_with_centroids(query_embeddings, centroid_ids, top_k, num_to_rerank)

    def query_with_centroids(self, embeddings: torch.Tensor, centroid_ids: torch.Tensor, top_k: int, num_to_rerank: int):
        num_vectors_in_query = embeddings.shape[0]
        dense_dim = embeddings.shape[1]
        if dense_dim != self._dense_dim:
            raise ValueError("Invalid row dimension")
        if num_vectors_in_query == 0:
            raise ValueError("Need at least one query vector but found 0")
        if top_k == 0:
            raise ValueError("The passed in top_k must be at least 1, was 0")
        if top_k > num_to_rerank:
            raise ValueError("The passed in top_k must be <= the passed in num_to_rerank")

        top_k_internal_ids = self.frequencyCountCentroidBuckets(centroid_ids, num_to_rerank)
        top_k_internal_ids = torch.unique(top_k_internal_ids)
        reranked = self.rankDocuments(embeddings, top_k_internal_ids)

        result_size = min(len(reranked), top_k)
        result = [self._internal_id_to_doc_id[reranked[i]] for i in range(result_size)]

        return result

    def rankDocuments(self, query_embeddings: torch.Tensor, internal_ids_to_rerank: torch.Tensor):
        document_scores = self._document_array.get_document_scores(query_embeddings.to(self._device), internal_ids_to_rerank.to(self._device))
        sorted_indices = torch.argsort(document_scores, descending=True).to(internal_ids_to_rerank.device)
        return internal_ids_to_rerank[sorted_indices]

    def getNearestCentroids(self, batch: torch.Tensor, nprobe: int):
        eigen_result = torch.matmul(batch.to(self._device), self._centroids)
        nearest_centroids = torch.topk(eigen_result, nprobe, dim=1).indices.view(-1)
        return torch.unique(nearest_centroids)

    def frequencyCountCentroidBuckets(self, centroid_ids: torch.Tensor, num_to_rerank: int):
        count_buffer = torch.zeros(self._largest_internal_id + 1, dtype=torch.int32, device=centroid_ids.device)
        for centroid_id in centroid_ids:
            count_buffer.index_add_(0, self._centroid_id_to_internal_id[centroid_id], torch.ones_like(self._centroid_id_to_internal_id[centroid_id], dtype=torch.int32))
        top_counts, top_indices = torch.topk(count_buffer, num_to_rerank)
        return top_indices

    def serialize_to_file(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def deserialize_from_file(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)