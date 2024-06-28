import numpy as np
from typing import TypeVar, List, Tuple
import random
from concurrent.futures import ThreadPoolExecutor
import pickle
import heapq
from heapq import heappush, heappop

LABEL_T = TypeVar('LABEL_T', np.uint8, np.uint16, np.uint32)

class TinyTable:
    def __init__(self, num_tables: int, hash_range: int, num_elements: LABEL_T, hashes: np.ndarray):
        self._hash_range = hash_range
        self._num_elements = num_elements
        self._num_tables = num_tables
        self._table_start = self._num_tables * (self._hash_range + 1)
        self._index = np.zeros((self._table_start + self._num_elements * self._num_tables,), dtype=np.int32)

        for table in range(num_tables):
            # Generate inverted index from hashes to vec_ids
            temp_buckets = [[] for _ in range(hash_range)]

            for vec_id in range(num_elements):
                hash_value = hashes[vec_id * num_tables + table]
                temp_buckets[hash_value].append(vec_id)

            # Populate bucket start and end offsets
            table_offsets_start = table * (self._hash_range + 1)
            self._index[table_offsets_start + 1:table_offsets_start + self._hash_range + 1] = np.cumsum([len(temp_buckets[i]) for i in range(hash_range)])

            # Populate hashes into table itself
            current_offset = self._table_start + self._num_elements * table
            for bucket in range(hash_range):
                end_offset = current_offset + len(temp_buckets[bucket])
                self._index[current_offset:end_offset] = temp_buckets[bucket]
                current_offset = end_offset

    def query_by_count(self, hashes: np.ndarray, hash_offset: int, counts: np.ndarray):
        for table in range(self._num_tables):
            hash_value = hashes[hash_offset + table]
            start_offset = self._index[(self._hash_range + 1) * table + hash_value]
            end_offset = self._index[(self._hash_range + 1) * table + hash_value + 1]
            table_offset = self._table_start + table * self._num_elements
            np.add.at(counts, self._index[table_offset + start_offset:table_offset + end_offset], 1)

    def num_tables(self) -> int:
        return self._num_tables

    def num_elements(self) -> LABEL_T:
        return self._num_elements

def remove_duplicates(v: np.ndarray) -> np.ndarray:
    return np.unique(v)

def min_heap_pairs_to_descending(min_heap):
    result = []
    while min_heap:
        # heapq.heappop returns the smallest element
        result.append(heapq.heappop(min_heap)[1])

    result.reverse()
    return np.array(result)

def argmax(input: np.ndarray, top_k: int) -> np.ndarray:
    # Identifies the indices of the largest top_k elements in an array.
    min_heap: List[Tuple[float, int]] = []
    for i in range(len(input)):
        if len(min_heap) < top_k:
            heappush(min_heap, (input[i], i))
        elif input[i] > min_heap[0][0]:
            heappop(min_heap)
            heappush(min_heap, (input[i], i))
    return min_heap_pairs_to_descending(min_heap)

def argsort_descending(to_argsort: np.ndarray) -> np.ndarray:
    # Perform argsort and then reverse the result to get descending order
    return np.argsort(to_argsort)[::-1]

class SparseRandomProjection:
    def __init__(self, input_dim: int, srps_per_table: int, num_tables: int, seed: int = None):
        self._num_tables = num_tables
        self._srps_per_table = srps_per_table
        self._total_num_srps = srps_per_table * num_tables
        self._dim = input_dim
        self._sample_size = int(np.ceil(self._dim * 0.3))

        if seed is not None:
            random.seed(seed)
        assert srps_per_table < 32

        a = np.arange(self._dim)

        self._random_bits = np.zeros(self._total_num_srps * self._sample_size, dtype=np.int16)
        self._hash_indices = np.zeros(self._total_num_srps * self._sample_size, dtype=np.uint32)

        for i in range(self._total_num_srps):
            random.shuffle(a)  # Shuffle the array 'a'
            self._hash_indices[i * self._sample_size:(i + 1) * self._sample_size] = np.sort(a[:self._sample_size])
            self._random_bits[i * self._sample_size:(i + 1) * self._sample_size] = ((np.random.randint(0, 2, self._sample_size) * 2) - 1)
        del a

    def hash_single_dense(self, values: np.ndarray, dim: int, output: np.ndarray):
        assert values.size == dim

        for table in range(self._num_tables):
          table_sum = 0
          for srp in range(self._srps_per_table):
            # Corrected slices to include srp in the calculation
            start_index = table * self._srps_per_table * self._sample_size + srp * self._sample_size
            end_index = start_index + self._sample_size

            bit_indices = self._hash_indices[start_index:end_index]
            bits = self._random_bits[start_index:end_index]

            s = np.sum(bits * values[bit_indices])
            to_add = (s > 0) << srp
            table_sum += to_add
          output[table] = table_sum

    def num_tables(self) -> int:
        return self._num_tables

    def range(self) -> int:
        return 1 << self._srps_per_table

class MaxFlash:
    def __init__(self, num_tables: int, hash_range: int, num_elements: LABEL_T, hashes: np.ndarray):
        self._hashtable = TinyTable(num_tables, hash_range, num_elements, hashes)

    def get_score(self, query_hashes: np.ndarray, num_elements: int,
                  count_buffer: np.ndarray, collision_count_to_sim: np.ndarray) -> float:
        results = np.zeros(num_elements, dtype=np.uint32)

        assert len(count_buffer) >= self._hashtable.num_elements()

        for vec_id in range(num_elements):
            count_buffer[:self._hashtable.num_elements()] = 0

            self._hashtable.query_by_count(query_hashes, vec_id * self._hashtable.num_tables(), count_buffer)
            max_count = np.max(count_buffer)

            results[vec_id] = max_count

        sum_sim = np.sum(collision_count_to_sim[results])
        return sum_sim

class MaxFlashArray:
    def __init__(self, function: SparseRandomProjection, hashes_per_table: int, max_doc_size: int):
        self._max_allowable_doc_size = min(max_doc_size, np.iinfo(np.uint32).max)
        self._hash_function = function
        self._maxflash_array = []
        self._collision_count_to_sim = np.zeros(self._hash_function.num_tables() + 1, dtype=np.float32)

        for collision_count in range(self._collision_count_to_sim.size):
            table_collision_probability = float(collision_count) / self._hash_function.num_tables()
            if table_collision_probability > 0:
                self._collision_count_to_sim[collision_count] = np.exp(np.log(table_collision_probability) / hashes_per_table)
            else:
                self._collision_count_to_sim[collision_count] = 0.0

    def add_document(self, batch: np.ndarray) -> int:
        num_vectors = batch.shape[0]
        num_elements = min(num_vectors, self._max_allowable_doc_size)
        hashes = self.hash(batch)
        self._maxflash_array.append(MaxFlash(self._hash_function.num_tables(), self._hash_function.range(), num_elements, hashes))
        return len(self._maxflash_array) - 1

    def get_document_scores(self, query: np.ndarray, documents_to_query: np.ndarray):
        hashes = self.hash(query)
        num_vectors_in_query = query.shape[0]

        def compute_score(i):
            flash_index = documents_to_query[i]
            buffer = np.zeros(self._max_allowable_doc_size, dtype=np.uint32)
            score = self._maxflash_array[flash_index].get_score(hashes, num_vectors_in_query, buffer, self._collision_count_to_sim)
            return score / num_vectors_in_query

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(compute_score, range(len(documents_to_query))))

        return np.array(results)

    def hash(self, batch: np.ndarray) -> np.ndarray:
        num_vectors, dim = batch.shape
        output = np.zeros(num_vectors * self._hash_function.num_tables(), dtype=np.int32)

        def compute_hash(i):
            data = batch[i]
            start_index = i * self._hash_function.num_tables()
            end_index = (i + 1) * self._hash_function.num_tables()
            self._hash_function.hash_single_dense(data, dim, output[start_index:end_index])

        with ThreadPoolExecutor() as executor:
            list(executor.map(compute_hash, range(num_vectors)))

        return output

class DocRetrieval:
    def __init__(self, hashes_per_table: int, num_tables: int, dense_input_dimension: int, centroids: np.ndarray):
        self._dense_dim = dense_input_dimension
        self._nprobe_query = 2
        self._largest_internal_id = 0
        self._num_centroids = centroids.shape[0]
        self._centroid_id_to_internal_id = [np.empty(0, dtype=np.int32) for _ in range(self._num_centroids)]
        self._internal_id_to_doc_id: List[str] = []

        if dense_input_dimension == 0 or num_tables == 0 or hashes_per_table == 0:
            raise ValueError("The dense dimension, number of tables, and hashes per table must all be greater than 0.")
        if self._num_centroids == 0:
            raise ValueError("Must pass in at least one centroid, found 0.")
        if centroids.shape[1] != self._dense_dim:
            raise ValueError("The centroids array must have dimension equal to dense_dim.")

        self._nprobe_query = min(len(centroids), self._nprobe_query)

        self._document_array = MaxFlashArray(SparseRandomProjection(dense_input_dimension, hashes_per_table, num_tables), hashes_per_table, np.iinfo(np.uint32).max)
        self._centroids = np.transpose(centroids)

    def add_doc(self, doc_embeddings: np.ndarray, doc_id: str) -> bool:
        centroid_ids = self.getNearestCentroids(doc_embeddings, 1)
        return self.add_doc_with_centroids(doc_embeddings, doc_id, centroid_ids)

    def add_doc_with_centroids(self, doc_embeddings: np.ndarray, doc_id: str, doc_centroid_ids: np.ndarray) -> bool:
        internal_id = self._document_array.add_document(doc_embeddings)
        self._largest_internal_id = max(self._largest_internal_id, internal_id)

        for centroid_id in doc_centroid_ids:
            self._centroid_id_to_internal_id[centroid_id] = np.append(self._centroid_id_to_internal_id[centroid_id], internal_id)

        if internal_id >= len(self._internal_id_to_doc_id):
            self._internal_id_to_doc_id.extend([None] * (internal_id + 1 - len(self._internal_id_to_doc_id)))
        self._internal_id_to_doc_id[internal_id] = doc_id

        return True

    def query(self, query_embeddings: np.ndarray, top_k: int, num_to_rerank: int):
        centroid_ids = self.getNearestCentroids(query_embeddings, self._nprobe_query)
        return self.query_with_centroids(query_embeddings, centroid_ids, top_k, num_to_rerank)

    def query_with_centroids(self, embeddings: np.ndarray, centroid_ids: np.ndarray, top_k: int, num_to_rerank: int):
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
        top_k_internal_ids = remove_duplicates(top_k_internal_ids)
        reranked = self.rankDocuments(embeddings, top_k_internal_ids)

        result_size = min(len(reranked), top_k)
        result = [self._internal_id_to_doc_id[reranked[i]] for i in range(result_size)]

        return result

    def rankDocuments(self, query_embeddings: np.ndarray, internal_ids_to_rerank: np.ndarray):
        document_scores = self.getScores(query_embeddings, internal_ids_to_rerank)
        sorted_indices = argsort_descending(document_scores)
        return internal_ids_to_rerank[sorted_indices]

    def getScores(self, query_embeddings: np.ndarray, internal_ids_to_rerank: np.ndarray):
        return self._document_array.get_document_scores(query_embeddings, internal_ids_to_rerank)

    def getNearestCentroids(self, batch: np.ndarray, nprobe: int):
        num_vectors = batch.shape[0]
        eigen_result = np.dot(batch, self._centroids)
        nearest_centroids = np.zeros(num_vectors * nprobe, dtype=np.uint32)

        def process_row(i):
          probe_results = argmax(eigen_result[i], nprobe)
          for p in range(nprobe):
            nearest_centroids[i * nprobe + p] = probe_results[p]

        with ThreadPoolExecutor() as executor:
          executor.map(process_row, range(num_vectors))

        nearest_centroids = np.unique(nearest_centroids)
        return np.array(nearest_centroids)

    def frequencyCountCentroidBuckets(self, centroid_ids, num_to_rerank):
      # Initialize the count buffer
      count_buffer = np.zeros(self._largest_internal_id + 1, dtype=np.int32)

      def process_centroid_id(centroid_id):
        np.add.at(count_buffer, self._centroid_id_to_internal_id[centroid_id], 1)

      # Parallel counting of internal IDs
      with ThreadPoolExecutor() as executor:
        executor.map(process_centroid_id, centroid_ids)

      # Find the indices of the top_k counts
      heap = []

      for centroid_id in centroid_ids:
        for internal_id in self._centroid_id_to_internal_id[centroid_id]:
            if count_buffer[internal_id] < 0:
                continue
            count = count_buffer[internal_id]
            count_buffer[internal_id] = -1
            if len(heap) < num_to_rerank or count > heap[0][0]:
                heapq.heappush(heap, (count, internal_id))
            if len(heap) > num_to_rerank:
                heapq.heappop(heap)

      result = []
      while heap:
        result.append(heapq.heappop(heap)[1])

      result.reverse()
      return np.array(result)

    def serialize_to_file(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def deserialize_from_file(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)