import numpy as np
from typing import TypeVar, List, Tuple
import random
from concurrent.futures import ThreadPoolExecutor
import pickle
import heapq
from heapq import heappush, heappop
import torch
from tqdm import tqdm
# from torch.nn.functional import cosine_similarity

LABEL_T = TypeVar('LABEL_T', np.uint8, np.uint16, np.uint32)


def dot_scores(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1)) #TODO: this keeps allocating GPU memory
    # return torch.mm(a, b.transpose(0, 1)) #TODO: this keeps allocating GPU memory
# @profile

def compute_dependency_aware_sim_score0(curr_query_embedding, sub_corpus_embeddings, corpus_idx, grouped_sub_q_ids_ls, sub_q_ls_idx, device, bboxes_overlap_ls, query_itr, prob_agg = "prod", is_img_retrieval=False, dependency_topk=50, valid_patch_ids=None):
        if grouped_sub_q_ids_ls[query_itr] is not None:
            curr_grouped_sub_q_ids_ls = grouped_sub_q_ids_ls[query_itr][sub_q_ls_idx]
        else:
            curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
        # print("dependency topk::", dependency_topk)
        if prob_agg == "prod":
            curr_scores_ls= 1
        else:
            curr_scores_ls = 0
        # curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
        if is_img_retrieval:
            curr_sub_corpus_embeddings = sub_corpus_embeddings[0:-1]
        else:
            curr_sub_corpus_embeddings = sub_corpus_embeddings
        for curr_grouped_sub_q_ids in curr_grouped_sub_q_ids_ls:

            selected_embedding_idx = torch.arange(curr_sub_corpus_embeddings.shape[0])
            beam_search_topk=min(dependency_topk, curr_sub_corpus_embeddings.shape[0])
            if prob_agg == "prod":
                sub_curr_scores = torch.ones(1).to(device)
            else:
                sub_curr_scores = torch.zeros(1).to(device)
            selected_patch_ids_ls = None
            for sub_query_idx in range(len(curr_grouped_sub_q_ids)): #range(curr_query_embedding.shape[0]):
                # print(curr_grouped_sub_q_ids, sub_query_idx)
                if valid_patch_ids is not None:
                    selected_embedding_idx = torch.tensor(list(set(selected_embedding_idx.tolist()).intersection(valid_patch_ids)))

                curr_prod_mat = dot_scores(curr_query_embedding[curr_grouped_sub_q_ids[sub_query_idx]].to(device), curr_sub_corpus_embeddings[selected_embedding_idx].to(device)).view(-1,1)
                if prob_agg == "prod":
                    curr_prod_mat[curr_prod_mat < 0] = 0
                    prod_mat = curr_prod_mat*sub_curr_scores.view(1,-1)
                else:
                    prod_mat = curr_prod_mat+sub_curr_scores.view(1,-1)

                # beam_search_topk=max(20, int(torch.numel(prod_mat)*0.05) + 1)
                # print("beam_search_topk::", beam_search_topk)
                sub_curr_scores_ls, topk_ids = torch.topk(prod_mat.view(-1), k=min(beam_search_topk, torch.numel(prod_mat)), dim=-1)
                topk_emb_ids = topk_ids // prod_mat.shape[1]

                topk_emb_ids = selected_embedding_idx.to(device)[topk_emb_ids].tolist()
                # topk_emb_ids = list(set(topk_emb_ids.tolist()))
                if sub_query_idx == 0:
                    selected_patch_ids_ls = [[emb_id] for emb_id in topk_emb_ids]
                    # selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                else:
                    selected_seq_ids = topk_ids%prod_mat.shape[1]
                    curr_selected_patch_ids_ls = [selected_patch_ids_ls[selected_seq_ids[selected_seq_id_idx]]+ [topk_emb_ids[selected_seq_id_idx]] for selected_seq_id_idx in range(len(selected_seq_ids))]
                    selected_patch_ids_ls = curr_selected_patch_ids_ls
                    # curr_selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                    # selected_embedding_idx = torch.tensor(list(set(torch.cat([selected_embedding_idx, curr_selected_embedding_idx]).tolist())))
                existing_topk_emb_ids = set()
                for selected_patch_ids in selected_patch_ids_ls:
                    existing_topk_emb_ids.update(selected_patch_ids)
                # selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in existing_topk_emb_ids])
                selected_embedding_idx = set()
                for topk_id in existing_topk_emb_ids:
                    selected_embedding_idx.update(bboxes_overlap_ls[corpus_idx][topk_id])
                selected_embedding_idx = torch.tensor(list(selected_embedding_idx))
                sub_curr_scores = sub_curr_scores_ls
            if prob_agg == "prod":
                sub_curr_scores[sub_curr_scores <= 0] = 0
                curr_scores_ls *= torch.max(sub_curr_scores)
                assert torch.all(curr_scores_ls >= 0).item()
            else:
                curr_scores_ls += torch.max(sub_curr_scores)

        return curr_scores_ls


def compute_dependency_aware_sim_score(curr_query_embedding, sub_corpus_embeddings, corpus_idx, grouped_sub_q_ids_ls, sub_q_ls_idx, device, bboxes_overlap_ls, query_itr, prob_agg = "prod", is_img_retrieval=False, dependency_topk=50, valid_patch_ids=None):
        if grouped_sub_q_ids_ls[query_itr] is not None:
            curr_grouped_sub_q_ids_ls = grouped_sub_q_ids_ls[query_itr][sub_q_ls_idx]
        else:
            curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
        # print("dependency topk::", dependency_topk)
        if prob_agg == "prod":
            curr_scores_ls= 1
        else:
            curr_scores_ls = 0
        # curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
        if is_img_retrieval:
            curr_sub_corpus_embeddings = sub_corpus_embeddings[0:-1]
        else:
            curr_sub_corpus_embeddings = sub_corpus_embeddings
            
        full_scores =  dot_scores(curr_query_embedding.to(device), curr_sub_corpus_embeddings.to(device))
        for curr_grouped_sub_q_ids in curr_grouped_sub_q_ids_ls:
            
            # selected_embedding_idx = torch.arange(curr_sub_corpus_embeddings.shape[0])
            full_selected_embedding_idx = torch.zeros(curr_sub_corpus_embeddings.shape[0],device=device).bool()
            selected_embedding_idx = torch.arange(curr_sub_corpus_embeddings.shape[0])
            beam_search_topk=min(dependency_topk, curr_sub_corpus_embeddings.shape[0])
            if prob_agg == "prod":
                sub_curr_scores = torch.ones(1).to(device)
            else:
                sub_curr_scores = torch.zeros(1).to(device)
            selected_patch_ids_ls = None
            for sub_query_idx in range(len(curr_grouped_sub_q_ids)): #range(curr_query_embedding.shape[0]):
                # print(curr_grouped_sub_q_ids, sub_query_idx)
                if valid_patch_ids is not None:
                    selected_embedding_idx = torch.tensor(list(set(selected_embedding_idx.tolist()).intersection(valid_patch_ids)))
                
                # curr_prod_mat = dot_scores(curr_query_embedding[curr_grouped_sub_q_ids[sub_query_idx]].to(device), curr_sub_corpus_embeddings[selected_embedding_idx].to(device)).view(-1,1)
                curr_prod_mat =  full_scores[curr_grouped_sub_q_ids[sub_query_idx], selected_embedding_idx].view(-1,1)
                if prob_agg == "prod":
                    curr_prod_mat[curr_prod_mat < 0] = 0
                    prod_mat = curr_prod_mat*sub_curr_scores.view(1,-1)
                else:
                    prod_mat = curr_prod_mat+sub_curr_scores.view(1,-1)

                # beam_search_topk=max(20, int(torch.numel(prod_mat)*0.05) + 1)
                # print("beam_search_topk::", beam_search_topk)
                sub_curr_scores_ls, topk_ids = torch.topk(prod_mat.view(-1), k=min(beam_search_topk, torch.numel(prod_mat)), dim=-1)
                topk_emb_ids = topk_ids // prod_mat.shape[1]
                
                # topk_emb_ids = selected_embedding_idx.to(device)[topk_emb_ids].tolist()
                topk_emb_ids_tensor = selected_embedding_idx.to(device)[topk_emb_ids]
                # topk_emb_ids = list(set(topk_emb_ids.tolist()))
                if sub_query_idx == 0:
                    # selected_patch_ids_ls = [[emb_id] for emb_id in topk_emb_ids]
                    selected_patch_ids_ls_tensor = topk_emb_ids_tensor.view(-1,1)
                    # selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                else:
                    # selected_seq_ids = topk_ids%prod_mat.shape[1]
                    selected_seq_ids = torch.remainder(topk_ids, prod_mat.shape[1])
                    curr_selected_patch_ids_ls_tensor = torch.cat([selected_patch_ids_ls_tensor[selected_seq_ids], topk_emb_ids_tensor.view(-1,1)], dim=-1)
                    # curr_selected_patch_ids_ls = [selected_patch_ids_ls[selected_seq_ids[selected_seq_id_idx]]+ [topk_emb_ids[selected_seq_id_idx]] for selected_seq_id_idx in range(len(selected_seq_ids))]
                    # selected_patch_ids_ls = curr_selected_patch_ids_ls
                    selected_patch_ids_ls_tensor = curr_selected_patch_ids_ls_tensor
                    # curr_selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                    # selected_embedding_idx = torch.tensor(list(set(torch.cat([selected_embedding_idx, curr_selected_embedding_idx]).tolist())))
                # existing_topk_emb_ids_tensor = selected_patch_ids_ls_tensor.view(-1).unique().tolist()
                # existing_topk_emb_ids = set()
                # for selected_patch_ids in selected_patch_ids_ls:
                #     existing_topk_emb_ids.update(selected_patch_ids)
                existing_topk_emb_ids = set()
                for selected_patch_ids in selected_patch_ids_ls_tensor.tolist():
                    existing_topk_emb_ids.update(selected_patch_ids)
                # # selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in existing_topk_emb_ids])
                selected_embedding_idx = set()
                for topk_id in existing_topk_emb_ids:
                    selected_embedding_idx.update(bboxes_overlap_ls[corpus_idx][topk_id])
                # for topk_id in existing_topk_emb_ids_tensor:
                #     # selected_embedding_idx.update(bboxes_overlap_ls[corpus_idx][topk_id])
                #     full_selected_embedding_idx[bboxes_overlap_ls[corpus_idx][topk_id]] = True
                selected_embedding_idx = torch.tensor(list(selected_embedding_idx))
                # selected_embedding_idx = full_selected_embedding_idx.nonzero().view(-1)
                sub_curr_scores = sub_curr_scores_ls
            if prob_agg == "prod":
                sub_curr_scores[sub_curr_scores <= 0] = 0
                curr_scores_ls *= torch.max(sub_curr_scores)
                assert torch.all(curr_scores_ls >= 0).item()
            else:
                curr_scores_ls += torch.max(sub_curr_scores)
                
        return curr_scores_ls

class TinyTable:
    def __init__(self, num_tables: int, hash_range: int, num_elements: LABEL_T, hashes: torch.tensor, device="cpu"):
        self._device = device
        self._hash_range = hash_range
        self._num_elements = num_elements
        self._num_tables = num_tables
        self._table_start = self._num_tables * (self._hash_range + 1)
        self._index = torch.zeros((self._table_start + self._num_elements * self._num_tables,), dtype=torch.int32, device=device)

        for table in range(num_tables):
            # Generate inverted index from hashes to vec_ids
            temp_buckets = [[] for _ in range(hash_range)]

            for vec_id in range(num_elements):
                hash_value = hashes[vec_id * num_tables + table]
                temp_buckets[hash_value].append(vec_id)

            # Populate bucket start and end offsets
            table_offsets_start = table * (self._hash_range + 1)
            self._index[table_offsets_start + 1:table_offsets_start + self._hash_range + 1] = torch.from_numpy(np.cumsum([len(temp_buckets[i]) for i in range(hash_range)]))

            # Populate hashes into table itself
            current_offset = self._table_start + self._num_elements * table
            for bucket in range(hash_range):
                end_offset = current_offset + len(temp_buckets[bucket])
                self._index[current_offset:end_offset] = torch.tensor(temp_buckets[bucket])
                current_offset = end_offset

    # def query_by_count(self, hashes: torch.tensor, hash_offset: int, counts: torch.tensor, sub_patch_ids=None): #, copied_counts: torch.tensor):
    #     for table in range(self._num_tables):
    #         hash_value = hashes[hash_offset + table]
    #         start_offset = self._index[(self._hash_range + 1) * table + hash_value]
    #         end_offset = self._index[(self._hash_range + 1) * table + hash_value + 1]
    #         table_offset = self._table_start + table * self._num_elements
    #         # np.add.at(counts, self._index[table_offset + start_offset:table_offset + end_offset], 1)
    #         counts[self._index[table_offset + start_offset:table_offset + end_offset]] += 1
    #         # print(np.max(copied_counts - counts), np.min(copied_counts - counts))
    #         # print()
    #     return counts
    def query_by_count(self, hashes: torch.Tensor, hash_offset: int, counts: torch.Tensor):
        counts_copy = counts.clone()
        table_array = torch.arange(self._num_tables, device=self._device)
        hash_value = hashes[hash_offset + table_array]
        start_offset = self._index[(self._hash_range + 1) * table_array + hash_value]
        end_offset = self._index[(self._hash_range + 1) * table_array + hash_value + 1]
        table_offset = self._table_start + table_array * self._num_elements
        for idx in torch.nonzero(end_offset > start_offset).view(-1).tolist():
            counts_copy[self._index[table_offset[idx] + start_offset[idx]:table_offset[idx] + end_offset[idx]]] += 1
            
        # for table in range(self._num_tables):
        #     hash_value = hashes[hash_offset + table].item()
        #     start_offset = self._index[(self._hash_range + 1) * table + hash_value].item()
        #     end_offset = self._index[(self._hash_range + 1) * table + hash_value + 1].item()
        #     table_offset = self._table_start + table * self._num_elements
        #     counts[self._index[table_offset + start_offset:table_offset + end_offset]] += 1
            # counts.index_add_(0, self._index[table_offset + start_offset:table_offset + end_offset], torch.ones(end_offset - start_offset, dtype=counts.dtype, device=self._device))


    def num_tables(self) -> int:
        return self._num_tables

    def num_elements(self) -> LABEL_T:
        return self._num_elements

def remove_duplicates(v: torch.tensor) -> torch.tensor:
    return torch.unique(v)

def min_heap_pairs_to_descending(min_heap):
    result = []
    while min_heap:
        # heapq.heappop returns the smallest element
        result.append(heapq.heappop(min_heap)[1])

    result.reverse()
    # return np.array(result)
    return torch.tensor(result)

def argmax(input: torch.tensor, top_k: int) -> torch.tensor:
    # Identifies the indices of the largest top_k elements in an array.
    min_heap: List[Tuple[float, int]] = []
    for i in range(len(input)):
        if len(min_heap) < top_k:
            heappush(min_heap, (input[i], i))
        elif input[i] > min_heap[0][0]:
            heappop(min_heap)
            heappush(min_heap, (input[i], i))
    return min_heap_pairs_to_descending(min_heap)

def argsort_descending(to_argsort: torch.tensor) -> torch.tensor:
    # Perform argsort and then reverse the result to get descending order
    # return np.argsort(to_argsort)[::-1]
    # print(to_argsort)
    return torch.argsort(to_argsort, descending=True) #[::-1]

class SparseRandomProjection:
    def __init__(self, input_dim: int, srps_per_table: int, num_tables: int, device="cpu"):
        self._num_tables = num_tables
        self._srps_per_table = srps_per_table
        self._total_num_srps = srps_per_table * num_tables
        self._dim = input_dim
        self._sample_size = int(np.ceil(self._dim * 0.3))

        # if seed is not None:
        #     random.seed(seed)
        assert srps_per_table < 32

        a = torch.arange(self._dim)

        self._random_bits = torch.zeros(self._total_num_srps * self._sample_size, dtype=torch.int16, device=device)
        self._hash_indices = torch.zeros(self._total_num_srps * self._sample_size, dtype=torch.int32, device=device)

        for i in range(self._total_num_srps):
            # random.shuffle(a)  # Shuffle the array 'a'
            a = torch.randperm(self._dim)
            self._hash_indices[i * self._sample_size:(i + 1) * self._sample_size] = torch.sort(a[:self._sample_size])[0]
            self._random_bits[i * self._sample_size:(i + 1) * self._sample_size] = ((torch.randint(0, 2, (self._sample_size,)) * 2) - 1)
        del a
        self.powers_of_two = 2 ** torch.arange(self._srps_per_table, device=device)


    # def hash_single_dense(self, values: torch.tensor, dim: int, output: torch.tensor):
    #     assert values.shape[0] == dim

    #     for table in range(self._num_tables):
    #       table_sum = 0
    #       for srp in range(self._srps_per_table):
    #         # Corrected slices to include srp in the calculation
    #         start_index = table * self._srps_per_table * self._sample_size + srp * self._sample_size
    #         end_index = start_index + self._sample_size

    #         bit_indices = self._hash_indices[start_index:end_index]
    #         bits = self._random_bits[start_index:end_index]

    #         s = torch.sum(bits * values[bit_indices])
    #         to_add = (s > 0) << srp
    #         table_sum += to_add
    #       output[table] = table_sum
    def hash_single_dense(self, values: torch.Tensor, dim: int, output: torch.Tensor):
        assert values.size(0) == dim
        hash_indices = self._hash_indices.view(self._num_tables, self._srps_per_table, self._sample_size).cpu()
        random_bits = self._random_bits.view(self._num_tables, self._srps_per_table, self._sample_size)
        gathered_values = values[hash_indices]
        products = gathered_values * random_bits.to(values.device)
        sums = torch.sum(products, dim=2)
        binary_values = (sums > 0).int()
        table_sums = torch.sum(binary_values * self.powers_of_two.to(values.device), dim=1)
        output[:] = table_sums

    def num_tables(self) -> int:
        return self._num_tables

    def range(self) -> int:
        return 1 << self._srps_per_table

class MaxFlash:
    def __init__(self, num_tables: int, hash_range: int, num_elements: LABEL_T, hashes: torch.tensor, device="cpu"):
        self._hashtable = TinyTable(num_tables, hash_range, num_elements, hashes, device=device)

    def compute_score_single_query(self, query_hashes, vec_id, count_buffer, sub_patch_ids=None):
        count_buffer[:self._hashtable.num_elements()] = 0

        self._hashtable.query_by_count(query_hashes, vec_id * self._hashtable.num_tables(), count_buffer)
        if sub_patch_ids is not None:
            max_count = torch.max(count_buffer[sub_patch_ids])
        else:
            max_count = torch.max(count_buffer)
        return max_count

    def get_score(self, query_hashes: torch.tensor, num_elements: int,
                  count_buffer: torch.tensor, collision_count_to_sim: torch.tensor, prob_agg = "sum", is_img_retrieval=False):
        results = torch.zeros(num_elements, dtype=torch.int32)

        assert len(count_buffer) >= self._hashtable.num_elements()

        for vec_id in range(num_elements):
            if is_img_retrieval and num_elements == 1:
                sub_patch_ids = np.array([self._hashtable._num_elements-1])
            else:
                sub_patch_ids = None
            max_count = self.compute_score_single_query(query_hashes, vec_id, count_buffer, sub_patch_ids=sub_patch_ids)
            results[vec_id] = max_count
            
        full_scores = collision_count_to_sim[results]/num_elements
        
        if prob_agg == "sum":
            sum_sim = torch.sum(full_scores, dim=-1)
        else:
            sim_tensor = full_scores
            sim_tensor[sim_tensor < 0] = 0
            sum_sim = torch.prod(sim_tensor, dim=-1)
        return sum_sim
    
    def get_score_dependency(self, query_hashes: torch.tensor, num_elements: int,
                  count_buffer: torch.tensor, collision_count_to_sim: torch.tensor, query_itr, corpus_idx, sub_q_ls_idx, grouped_sub_q_ids_ls=None, bboxes_overlap_ls=None, dependency_topk=None, device=None, prob_agg=None, is_img_retrieval=False, **kwargs):
        # results = torch.zeros(num_elements, dtype=torch.int32)

        assert len(count_buffer) >= self._hashtable.num_elements()

        if prob_agg == "prod":
            curr_scores_ls= 1
        else:
            curr_scores_ls = 0
        
        if grouped_sub_q_ids_ls[query_itr] is not None:
            curr_grouped_sub_q_ids_ls = grouped_sub_q_ids_ls[query_itr][sub_q_ls_idx]
        else:
            curr_grouped_sub_q_ids_ls = [list(range(num_elements))]
        # if is_img_retrieval:
        #     curr_sub_corpus_embeddings = sub_corpus_embeddings[0:-1]
        # else:
        #     curr_sub_corpus_embeddings = 
        if is_img_retrieval:
            curr_num_elements = self._hashtable.num_elements() - 1
        else:
            curr_num_elements = self._hashtable.num_elements()
        
        for curr_grouped_sub_q_ids in curr_grouped_sub_q_ids_ls:
            selected_embedding_idx = torch.arange(curr_num_elements)
            beam_search_topk=min(dependency_topk, curr_num_elements)
            if prob_agg == "prod":
                sub_curr_scores = torch.ones(1).to(device)
            else:
                sub_curr_scores = torch.zeros(1).to(device)
            
            for sub_query_idx in range(len(curr_grouped_sub_q_ids)): 
                
                count_buffer[:self._hashtable.num_elements()] = 0

                self._hashtable.query_by_count(query_hashes, curr_grouped_sub_q_ids[sub_query_idx] * self._hashtable.num_tables(), count_buffer)
                # self._hashtable.query_by_count(query_hashes, sub_query_idx * self._hashtable.num_tables(), count_buffer)

                curr_prod_mat = collision_count_to_sim[count_buffer[selected_embedding_idx]]/num_elements

                # curr_prod_mat = curr_local_scores/num_elements #np.max(curr_local_scores)
                
                if prob_agg == "prod":
                    curr_prod_mat[curr_prod_mat < 0] = 0
                    prod_mat = curr_prod_mat.view(-1,1)*sub_curr_scores.view(1,-1)
                else:
                    prod_mat = curr_prod_mat.view(-1,1)+sub_curr_scores.view(1,-1)

                # beam_search_topk=max(20, int(torch.numel(prod_mat)*0.05) + 1)
                # print("beam_search_topk::", beam_search_topk)
                sub_curr_scores_ls, topk_ids = torch.topk(prod_mat.view(-1), k=min(beam_search_topk, torch.numel(prod_mat)), dim=-1)
                topk_emb_ids = topk_ids // prod_mat.shape[1]
                
                # topk_emb_ids = selected_embedding_idx.to(device)[topk_emb_ids].tolist()
                topk_emb_ids_tensor = selected_embedding_idx.to(device)[topk_emb_ids]
                # topk_emb_ids = list(set(topk_emb_ids.tolist()))
                if sub_query_idx == 0:
                    # selected_patch_ids_ls = [[emb_id] for emb_id in topk_emb_ids]
                    selected_patch_ids_ls_tensor = topk_emb_ids_tensor.view(-1,1)
                    # selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                else:
                    # selected_seq_ids = topk_ids%prod_mat.shape[1]
                    # curr_selected_patch_ids_ls = [selected_patch_ids_ls[selected_seq_ids[selected_seq_id_idx]]+ [topk_emb_ids[selected_seq_id_idx]] for selected_seq_id_idx in range(len(selected_seq_ids))]
                    # selected_patch_ids_ls = curr_selected_patch_ids_ls
                    selected_seq_ids = torch.remainder(topk_ids, prod_mat.shape[1])
                    # curr_selected_patch_ids_ls = [selected_patch_ids_ls[selected_seq_ids[selected_seq_id_idx]]+ [topk_emb_ids[selected_seq_id_idx]] for selected_seq_id_idx in range(len(selected_seq_ids))]
                    curr_selected_patch_ids_ls_tensor = torch.cat([selected_patch_ids_ls_tensor[selected_seq_ids], topk_emb_ids_tensor.view(-1,1)], dim=-1)
                    # print(torch.max(torch.abs(curr_selected_patch_ids_ls_tensor.cpu() - torch.tensor(curr_selected_patch_ids_ls))))
                    # selected_patch_ids_ls = curr_selected_patch_ids_ls
                    selected_patch_ids_ls_tensor = curr_selected_patch_ids_ls_tensor
                    # curr_selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                    # selected_embedding_idx = torch.tensor(list(set(torch.cat([selected_embedding_idx, curr_selected_embedding_idx]).tolist())))
                # existing_topk_emb_ids = set()
                # for selected_patch_ids in selected_patch_ids_ls:
                #     existing_topk_emb_ids.update(selected_patch_ids)
                existing_topk_emb_ids = set()
                for selected_patch_ids in selected_patch_ids_ls_tensor.tolist():
                    existing_topk_emb_ids.update(selected_patch_ids)
                # selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in existing_topk_emb_ids])
                selected_embedding_idx = set()
                for topk_id in existing_topk_emb_ids:
                    selected_embedding_idx.update(bboxes_overlap_ls[corpus_idx][topk_id])
                selected_embedding_idx = torch.tensor(list(selected_embedding_idx))
                
                sub_curr_scores = sub_curr_scores_ls
                
                
            
            if prob_agg == "prod":
                sub_curr_scores[sub_curr_scores <= 0] = 0
                curr_scores_ls *= torch.max(sub_curr_scores)
                assert torch.all(curr_scores_ls >= 0).item()
            else:
                curr_scores_ls += torch.max(sub_curr_scores)
                
        return curr_scores_ls
                

                # results[sub_query_idx] = max_count
            
        # for vec_id in range(num_elements):
        #     count_buffer[:self._hashtable.num_elements()] = 0

        #     self._hashtable.query_by_count(query_hashes, vec_id * self._hashtable.num_tables(), count_buffer)
        #     max_count = np.max(count_buffer)

        #     results[vec_id] = max_count
            
        # return collision_count_to_sim[results]

class MaxFlashArray:
    def __init__(self, function: SparseRandomProjection, hashes_per_table: int, max_doc_size: int, device="cpu"):
        self._device = device
        self._max_allowable_doc_size = min(max_doc_size, torch.iinfo(torch.int32).max)
        self._hash_function = function
        self._maxflash_array = []
        self._collision_count_to_sim = torch.zeros(self._hash_function.num_tables() + 1, dtype=torch.float32, device=device)

        # for collision_count in range(self._collision_count_to_sim.shape[0]):
        #     table_collision_probability = float(collision_count) / self._hash_function.num_tables()
        #     if table_collision_probability > 0:
        #         self._collision_count_to_sim[collision_count] = np.exp(np.log(table_collision_probability) / hashes_per_table)
        #     else:
        #         self._collision_count_to_sim[collision_count] = 0.0
        for collision_count in range(self._collision_count_to_sim.size(0)):
            table_collision_probability = float(collision_count) / self._hash_function.num_tables()
            if table_collision_probability > 0:
                self._collision_count_to_sim[collision_count] = torch.exp(torch.log(torch.tensor(table_collision_probability, device=device)) / hashes_per_table)
            else:
                self._collision_count_to_sim[collision_count] = 0.0
    # @profile
    def add_document(self, batch: torch.tensor, index_method="default") -> int:
        num_vectors = batch.shape[0]
        if index_method == "default":
            self._maxflash_array.append(None)
        else:
            num_elements = min(num_vectors, self._max_allowable_doc_size)
            hashes = self.hash(batch)
            self._maxflash_array.append(MaxFlash(self._hash_function.num_tables(), self._hash_function.range(), num_elements, hashes, device=self._device))
        return len(self._maxflash_array) - 1
    
    def compute_score_full(self, curr_query_embedding, sub_corpus_embeddings, algebra_method="two", prob_agg="prod", device="cuda", corpus_idx=None, grouped_sub_q_ids_ls=None, sub_q_ls_idx=None, bboxes_overlap_ls=None, query_itr=None, is_img_retrieval=False,dependency_topk=50, **kwargs):
        if curr_query_embedding.shape[0] == 1:
            if is_img_retrieval:
                curr_scores_ls = dot_scores(curr_query_embedding.to(device), sub_corpus_embeddings[-1].to(device))
            else:
                curr_scores_ls = torch.max(dot_scores(curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]    
            curr_scores = curr_scores_ls
            return curr_scores.item()
        # print("prob_agg::", prob_agg)
        # else:
            # curr_scores_ls = torch.max(self.cos_sim(curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
        # if self.algebra_method == one or self.algebra_method == three:
        #     curr_scores_ls = self.cos_sim(curr_query_embedding.to(device), sub_corpus_embeddings.to(device))#, dim=-1)
        if algebra_method == "two":
            if is_img_retrieval:
                curr_scores_ls = torch.max(dot_scores(curr_query_embedding.to(device), sub_corpus_embeddings[0:-1].to(device)), dim=-1)[0]
            else:
                curr_scores_ls = torch.max(dot_scores(curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
            
            # curr_scores_ls_max_id = torch.argmax(self.cos_sim(curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)
        else:
            # curr_query_embedding, sub_corpus_embeddings, corpus_idx, grouped_sub_q_ids_ls, sub_q_ls_idx, device, bboxes_overlap_ls, query_itr, prob_agg = "prod", is_img_retrieval=False, dependency_topk=50, valid_patch_ids=None
            curr_scores_ls = compute_dependency_aware_sim_score0(curr_query_embedding, sub_corpus_embeddings, corpus_idx, grouped_sub_q_ids_ls, sub_q_ls_idx, device, bboxes_overlap_ls, query_itr, is_img_retrieval=is_img_retrieval, prob_agg=prob_agg, dependency_topk=dependency_topk)
                # curr_scores_ls2 = torch.max(self.cos_sim(curr_query_embedding.to(device), sub_corpus_embeddings[0:-1].to(device)), dim=-1)[0]
                
        # else:    
        #     curr_scores_ls = self.cos_sim(curr_query_embedding.to(device), sub_corpus_embeddings.to(device))
        
        # whole_img_sim = self.cos_sim(curr_query_embedding.to(device), sub_corpus_embeddings[-1].to(device)).view(-1)
        # curr_scores = torch.prod(curr_scores_ls, dim=0)
        # for conj_id in range(len(curr_scores_ls)):
        #     curr_scores *= curr_scores_ls[conj_id]
        # full_curr_scores += curr_scores
        # if self.algebra_method == one:
        #     curr_scores = torch.max(torch.prod(curr_scores_ls, dim=0))
        #     full_curr_scores_ls.append(curr_scores.item())
        # elif self.algebra_method == three:
        #     curr_scores = torch.max(torch.sum(curr_scores_ls, dim=0))
        #     # curr_scores = torch.max(torch.max(curr_scores_ls, dim=0))
        #     full_curr_scores_ls.append(curr_scores.item())
        if algebra_method == "two":
            
            # if torch.sum(curr_scores_ls - whole_img_sim > 0.2) > 0:
            #     print()
            
            # curr_scores_ls[curr_scores_ls2 - whole_img_sim > 0.2] = whole_img_sim[curr_scores_ls2 - whole_img_sim > 0.2]
            # curr_scores_ls[whole_img_sim - curr_scores_ls2 > 0.2] = curr_scores_ls2[whole_img_sim - curr_scores_ls2 > 0.2]
            if prob_agg == "prod":
                curr_scores_ls[curr_scores_ls < 0] = 0
                curr_scores = torch.prod(curr_scores_ls)
            else:
                curr_scores = torch.sum(curr_scores_ls)
            # curr_scores = torch.sum(curr_scores_ls)
            # curr_scores = torch.sum(curr_scores_ls)
        else:
            curr_scores = curr_scores_ls
        return curr_scores.item()
        
    def get_document_scores(self, document_embs_ls:list, query: torch.tensor, documents_to_query: torch.tensor, method="two", query_idx=None, query_sub_idx =None, device="cuda", is_img_retrieval=False, prob_agg="sum",grouped_sub_q_ids_ls=None,bboxes_overlap_ls=None, index_method="default", **kwargs):
        query = query.to(self._device)
        
        num_vectors_in_query = query.shape[0]
        # print("query hashes::", hashes)
        def compute_score(i):
            flash_index = documents_to_query[i]
            
            if index_method == "default":
                score = self.compute_score_full(query, document_embs_ls[flash_index].to(self._device), method, prob_agg, device, corpus_idx=flash_index,grouped_sub_q_ids_ls=grouped_sub_q_ids_ls, sub_q_ls_idx=query_sub_idx, bboxes_overlap_ls=bboxes_overlap_ls, query_itr=query_idx, is_img_retrieval=is_img_retrieval, **kwargs)
            else:
                hashes = self.hash(query)
                if method == "two":
                    buffer = torch.zeros(self._max_allowable_doc_size, dtype=torch.int32, device=device)
                    score = self._maxflash_array[flash_index].get_score(hashes, num_vectors_in_query, buffer, self._collision_count_to_sim, prob_agg=prob_agg, is_img_retrieval=is_img_retrieval)
                else:
                    buffer = torch.zeros(self._max_allowable_doc_size, dtype=torch.int32, device=device)
                    if query.shape[0] == 1:
                        score = self._maxflash_array[flash_index].get_score(hashes, num_vectors_in_query, buffer, self._collision_count_to_sim, prob_agg=prob_agg, is_img_retrieval=is_img_retrieval)
                    else:
                        self._collision_count_to_sim = self._collision_count_to_sim.to(device)
                        score = self._maxflash_array[flash_index].get_score_dependency(hashes, num_vectors_in_query, buffer, self._collision_count_to_sim, query_itr=query_idx, corpus_idx=flash_index, sub_q_ls_idx=query_sub_idx,device=device, is_img_retrieval=is_img_retrieval, prob_agg=prob_agg,grouped_sub_q_ids_ls=grouped_sub_q_ids_ls,bboxes_overlap_ls=bboxes_overlap_ls, **kwargs)

            return score

        # with ThreadPoolExecutor() as executor:
        #     results = list(executor.map(compute_score, range(len(documents_to_query))))
        results=[]
        for i in range(len(documents_to_query)):
            results.append(compute_score(i))

        # return np.array(results)
        return torch.tensor(results)

    # def hash(self, batch: torch.tensor) -> torch.tensor:
    #     num_vectors, dim = batch.shape
    #     output = torch.zeros(num_vectors * self._hash_function.num_tables(), dtype=torch.int32)

    #     def compute_hash(i):
    #         data = batch[i]
    #         start_index = i * self._hash_function.num_tables()
    #         end_index = (i + 1) * self._hash_function.num_tables()
    #         self._hash_function.hash_single_dense(data, dim, output[start_index:end_index])

    #     # with ThreadPoolExecutor() as executor:
    #     #     list(executor.map(compute_hash, range(num_vectors)))
    #     results = []
    #     for i in range(num_vectors):
    #         curr_res = compute_hash(i)
    #         results.append(curr_res)
    #         # list(executor.map(compute_hash, range(num_vectors)))
    #     # print("doc hash::", output)
    #     return output
    def hash(self, batch: torch.Tensor) -> torch.Tensor:
        num_vectors, dim = batch.size()
        output = torch.zeros((num_vectors, self._hash_function.num_tables()), dtype=torch.int32, device=batch.device)

        for i in range(num_vectors):
            self._hash_function.hash_single_dense(batch[i], dim, output[i])

        return output.view(-1)

class DocRetrieval:
    def __init__(self, doc_size:int, hashes_per_table: int, num_tables: int, dense_input_dimension: int, centroids: torch.tensor, device="cpu"):
        self._dense_dim = dense_input_dimension
        self._nprobe_query = 2
        self._largest_internal_id = 0
        self._num_centroids = centroids.shape[0]
        self._centroid_id_to_internal_id = [torch.empty(0, dtype=torch.int32, device=device) for _ in range(self._num_centroids)]
        self._internal_id_to_doc_id: List[str] = []

        if dense_input_dimension == 0 or num_tables == 0 or hashes_per_table == 0:
            raise ValueError("The dense dimension, number of tables, and hashes per table must all be greater than 0.")
        if self._num_centroids == 0:
            raise ValueError("Must pass in at least one centroid, found 0.")
        if centroids.shape[1] != self._dense_dim:
            raise ValueError("The centroids array must have dimension equal to dense_dim.")
        self._device = device

        self._nprobe_query = min(len(centroids), self._nprobe_query)

        self._document_array = MaxFlashArray(SparseRandomProjection(dense_input_dimension, hashes_per_table, num_tables, self._device), hashes_per_table, doc_size,self._device)
        # self._centroids = np.transpose(centroids)
        # self._centroids = torch.t(centroids)
        self._centroids = centroids.T.to(self._device)
        # self.doc_embs_ls = []
        

    # @profile
    def add_doc(self, doc_embeddings: torch.tensor, doc_id: str, index_method="default") -> bool:
        # self.doc_embs_ls.append(doc_embeddings)
        centroid_ids = self.getNearestCentroids(doc_embeddings, 1)
        return self.add_doc_with_centroids(doc_embeddings, doc_id, centroid_ids, index_method=index_method)
    # @profile
    # def add_doc_with_centroids(self, doc_embeddings: torch.tensor, doc_id: str, doc_centroid_ids: torch.tensor) -> bool:
    #     internal_id = self._document_array.add_document(doc_embeddings)
    #     self._largest_internal_id = max(self._largest_internal_id, internal_id)

    #     for centroid_id in doc_centroid_ids:
    #         self._centroid_id_to_internal_id[centroid_id] = np.append(self._centroid_id_to_internal_id[centroid_id], internal_id)
    #         # self._centroid_id_to_internal_id[centroid_id] = torch.cat(self._centroid_id_to_internal_id[centroid_id], internal_id)

    #     if internal_id >= len(self._internal_id_to_doc_id):
    #         self._internal_id_to_doc_id.extend([None] * (internal_id + 1 - len(self._internal_id_to_doc_id)))
    #     self._internal_id_to_doc_id[internal_id] = doc_id

    #     return True
    # @profile
    def add_doc_with_centroids(self, doc_embeddings: torch.Tensor, doc_id: str, doc_centroid_ids: torch.Tensor, index_method="default") -> bool:
        internal_id = self._document_array.add_document(doc_embeddings, index_method=index_method)
        self._largest_internal_id = max(self._largest_internal_id, internal_id)

        for centroid_id in doc_centroid_ids:
            self._centroid_id_to_internal_id[centroid_id] = torch.cat((self._centroid_id_to_internal_id[centroid_id], torch.tensor([internal_id], dtype=torch.int32, device=self._device)))

        if internal_id >= len(self._internal_id_to_doc_id):
            self._internal_id_to_doc_id.extend([None] * (internal_id + 1 - len(self._internal_id_to_doc_id)))
        self._internal_id_to_doc_id[internal_id] = doc_id

        return True

    def query(self, document_embs_ls:list, query_embeddings: torch.tensor, top_k: int, num_to_rerank: int, prob_agg="prod", **kwargs):
        centroid_ids = self.getNearestCentroids(query_embeddings, self._nprobe_query)
        return self.query_with_centroids(document_embs_ls, query_embeddings, centroid_ids, top_k, num_to_rerank, prob_agg=prob_agg, **kwargs)

    def compute_scores_single_query(self, top_k_internal_ids, document_embs_ls:list, embeddings: torch.tensor, **kwargs):
        sum_sim = self.rankDocuments(document_embs_ls, embeddings, top_k_internal_ids, **kwargs)

        # result_size = min(len(reranked), top_k)
        # result = [self._internal_id_to_doc_id[reranked[i]] for i in range(result_size)]

        # return result
        
        
        # sorted_indices = argsort_descending(sum_sim).cpu()
        # print(sum_sim)
        
        return sum_sim #[sorted_indices]

    def query_multi_queries(self, document_embs_ls, query_embedding_ls, top_k: int, num_to_rerank: int, prob_agg="prod", dataset_name="", **kwargs):
        all_cos_scores = []
        query_ids = [str(idx+1) for idx in list(range(len(query_embedding_ls)))]
        corpus_ids = [str(idx+1) for idx in list(range(len(self._document_array._maxflash_array)))]
        all_results = {qid: {} for qid in query_ids}
        query_count = len(query_embedding_ls)
        
        all_cos_scores_tensor = torch.zeros(query_count, len(query_embedding_ls[0]), len(corpus_ids))
        expected_idx_ls = []
        for idx in tqdm(range(query_count)): #, desc="Querying":
            cos_scores_ls=[]
            sample_ids = self.query(document_embs_ls, torch.cat(query_embedding_ls[idx], dim=0), top_k, num_to_rerank, prob_agg=prob_agg, query_idx=idx, **kwargs)
            
            for sub_idx in range(len(query_embedding_ls[idx])):
                # top_k_internal_ids, document_embs_ls:list, embeddings: torch.tensor, method="two", prob_agg="prod", **kwargs
                cos_scores = self.compute_scores_single_query(sample_ids, document_embs_ls, query_embedding_ls[idx][sub_idx], prob_agg=prob_agg, query_idx=idx, query_sub_idx =sub_idx, **kwargs)
                # cos_scores_ls.append(cos_scores)
                # if sub_idx == 0:
                #     # print(idx, idx in sample_ids, sample_ids)
                #     if idx in sample_ids:
                #         expected_idx_ls.append(idx)
                all_cos_scores_tensor[idx, sub_idx, sample_ids] = cos_scores.cpu()
        #     cos_scores = torch.stack(cos_scores_ls)
        #     all_cos_scores.append(cos_scores)        
        # all_cos_scores_tensor = torch.stack(all_cos_scores)

        # 
        # all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
        if prob_agg == "prod":
            all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
            all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
        else:
            if dataset_name == "trec-covid":
                # all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
                all_cos_scores_tensor = torch.sum(all_cos_scores_tensor, dim=1)
            else:
                all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
                all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
        print(all_cos_scores_tensor.shape)
        #Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(all_cos_scores_tensor, min(top_k+1, len(all_cos_scores_tensor[0])), dim=1, largest=True)#, sorted=return_sorted)
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
        # for query_itr in range(len(query_embeddings)):
        for query_itr in range(query_count):
            query_id = query_ids[query_itr]                  
            for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                corpus_id = corpus_ids[sub_corpus_id]
                # if corpus_id != query_id:
                all_results[query_id][corpus_id] = score
        return all_results

    def query_with_centroids(self, document_embs_ls:list, embeddings: torch.tensor, centroid_ids: torch.tensor, top_k: int, num_to_rerank: int, method="two", prob_agg="prod", **kwargs):
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
        return top_k_internal_ids
        # return scores

    def rankDocuments(self, document_embs_ls: list, query_embeddings: torch.tensor, internal_ids_to_rerank: torch.tensor, method="two",prob_agg="sum", **kwargs):
        document_scores = self.getScores(document_embs_ls, query_embeddings, internal_ids_to_rerank, method=method,prob_agg=prob_agg, **kwargs)
        
        return document_scores

    def getScores(self, document_embs_ls: list, query_embeddings: torch.tensor, internal_ids_to_rerank: torch.tensor, method="two",prob_agg="sum", **kwargs):
        return self._document_array.get_document_scores(document_embs_ls, query_embeddings, internal_ids_to_rerank, method=method,prob_agg=prob_agg, **kwargs)

    # def getNearestCentroids(self, batch: torch.tensor, nprobe: int):
    #     num_vectors = batch.shape[0]
    #     eigen_result = torch.matmul(batch, self._centroids)
    #     nearest_centroids = torch.zeros(num_vectors * nprobe, dtype=torch.int32)

    #     def process_row(i):
    #       probe_results = argmax(eigen_result[i], nprobe)
    #       for p in range(nprobe):
    #         nearest_centroids[i * nprobe + p] = probe_results[p]

    #     with ThreadPoolExecutor() as executor:
    #       executor.map(process_row, range(num_vectors))

    #     nearest_centroids = torch.unique(nearest_centroids)
    #     # return np.array(nearest_centroids)
    #     return torch.tensor(nearest_centroids)
    def getNearestCentroids(self, batch: torch.Tensor, nprobe: int):
        # eigen_result = torch.matmul(batch.to(self._device), self._centroids.to(self._device))
        eigen_result = dot_scores(batch.to(self._device), self._centroids.T.to(self._device))
        nprobe = min(nprobe, self._centroids.shape[1])
        nearest_centroids = torch.topk(eigen_result, nprobe, dim=1).indices.view(-1)
        return torch.unique(nearest_centroids).cpu()

    # def frequencyCountCentroidBuckets(self, centroid_ids, num_to_rerank):
    #     # Initialize the count buffer
    # #   count_buffer = np.zeros(self._largest_internal_id + 1, dtype=np.int32)
    #     count_buffer = torch.zeros(self._largest_internal_id + 1, dtype=torch.int32, device=self._device)

    #     def process_centroid_id(count_buffer, centroid_id):
    #         # np.add.at(count_buffer, self._centroid_id_to_internal_id[centroid_id], 1)
    #         count_buffer[self._centroid_id_to_internal_id[centroid_id]] += 1
    #         return count_buffer

    #     # Parallel counting of internal IDs
    #     with ThreadPoolExecutor() as executor:
    #         executor.map(process_centroid_id, centroid_ids)
    #     for cid in centroid_ids.tolist():
    #         count_buffer = process_centroid_id(count_buffer, cid)
    #     # Find the indices of the top_k counts
    #     heap = []

    #     for centroid_id in centroid_ids:
    #         for internal_id in self._centroid_id_to_internal_id[centroid_id]:
    #             if count_buffer[internal_id] < 0:
    #                 continue
    #             count = count_buffer[internal_id]
    #             count_buffer[internal_id] = -1
    #             if len(heap) < num_to_rerank or count > heap[0][0]:
    #                 heapq.heappush(heap, (count, internal_id))
    #             if len(heap) > num_to_rerank:
    #                 heapq.heappop(heap)

    #     result = []
    #     while heap:
    #         result.append(heapq.heappop(heap)[1])

    #     result.reverse()
    # #   return np.array(result)
    #     return torch.tensor(result)
    
    def frequencyCountCentroidBuckets(self, centroid_ids: torch.Tensor, num_to_rerank: int):
        count_buffer = torch.zeros(self._largest_internal_id + 1, dtype=torch.int32, device=self._device)
        for centroid_id in centroid_ids:
            count_buffer.index_add_(0, self._centroid_id_to_internal_id[centroid_id], torch.ones_like(self._centroid_id_to_internal_id[centroid_id], dtype=torch.int32))
        top_counts, top_indices = torch.topk(count_buffer, num_to_rerank)
        return top_indices[top_counts > 0]

    def serialize_to_file(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def deserialize_from_file(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)