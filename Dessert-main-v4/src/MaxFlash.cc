#include "MaxFlash.h"
#include <cassert>
#include <stdexcept>

#include "Utils.h"
#include <cmath>
#include <algorithm>
#include <utility>
#include <set>
#include <numeric>
#include<iostream>

namespace thirdai::search {

template <typename LABEL_T>
MaxFlash<LABEL_T>::MaxFlash(uint32_t num_tables, uint32_t range,
                            LABEL_T num_elements,
                            const std::vector<uint32_t>& hashes)
    : _hashtable(num_tables, range, num_elements, hashes) {}

template <typename LABEL_T>
float MaxFlash<LABEL_T>::getScore(
    const std::vector<uint32_t>& query_hashes, uint32_t num_elements,
    std::vector<uint32_t>& count_buffer,
    const std::vector<float>& collision_count_to_sim,
    const std::string& prob_agg,
    bool is_img_retrieval) const {
  //std::cout << "Entered getScore" << std::endl;
  std::vector<uint32_t> results(num_elements);

  assert(count_buffer.size() >= _hashtable.numElements());

  for (uint64_t vec_id = 0; vec_id < num_elements; vec_id++) {
    std::fill(count_buffer.begin(),
              count_buffer.begin() + _hashtable.numElements(), 0);

    std::vector<LABEL_T> query_result;
    _hashtable.queryByCount(query_hashes, vec_id * _hashtable.numTables(),
                            count_buffer);
    uint32_t max_count = 0;
    if (is_img_retrieval && num_elements == 1) {
        max_count = count_buffer[_hashtable.numElements() - 1];
    } else {
        for (uint32_t i = 0; i < _hashtable.numElements(); i++) {
            if (count_buffer[i] > max_count) {
                max_count = count_buffer[i];
            }
        }
    }
    
    results.at(vec_id) = max_count;
  }
  
  //good
  std::vector<float> full_scores(num_elements);
  for (uint32_t i = 0; i < num_elements; i++) {
      full_scores[i] = collision_count_to_sim[results[i]] / num_elements;
  }
  float sum_sim = 0.0f;
  if (prob_agg == "sum") {
      for (uint32_t i = 0; i < full_scores.size(); i++) {
          sum_sim += full_scores[i];
      }
  } else {
      sum_sim = 1.0f;
      for (uint32_t i = 0; i < full_scores.size(); i++) {
          if (full_scores[i] < 0) {
              full_scores[i] = 0;
          }
          sum_sim *= full_scores[i];
      }
  }
  //std::cout << "Exited getScore" << std::endl;
  //good
  return sum_sim;
}

//good
template <typename LABEL_T>
float MaxFlash<LABEL_T>::getScoreDependency(
const std::vector<uint32_t>& query_hashes,
uint32_t num_elements,
std::vector<uint32_t>& count_buffer,
const std::vector<float>& collision_count_to_sim,
uint32_t sub_q_ls_idx, 
const std::vector<std::vector<std::vector<uint32_t>>>& partial_grouped_sub_q_ids_ls,
const std::vector<std::vector<uint32_t>>& partial_bboxes_overlap_ls,
uint32_t dependency_topk,
const std::string& prob_agg,
bool is_img_retrieval) const {
    //std::cout << "Entered getScoreDependency" << std::endl;
    //good
    assert(count_buffer.size() >= _hashtable.numElements());
    float curr_scores_ls = (prob_agg == "prod") ? 1.0f : 0.0f;
    //std::cout << "Line 1" << std::endl;
    //good
    std::vector<std::vector<uint32_t>> curr_grouped_sub_q_ids_ls;
    //if (!grouped_sub_q_ids_ls[query_itr].empty()) {
    if (!partial_grouped_sub_q_ids_ls.empty()) {
        curr_grouped_sub_q_ids_ls = partial_grouped_sub_q_ids_ls[sub_q_ls_idx];
    } else {
        std::vector<uint32_t> range_vector(num_elements);
        std::iota(range_vector.begin(), range_vector.end(), 0); // Populate with 0 to num_elements-1
        curr_grouped_sub_q_ids_ls.push_back(range_vector);
    }
    //std::cout << "Line 2" << std::endl;
    //good
    uint32_t curr_num_elements = is_img_retrieval ? _hashtable.numElements() - 1 : _hashtable.numElements();
    
    for (const auto& curr_grouped_sub_q_ids : curr_grouped_sub_q_ids_ls) {
        //good
        std::vector<uint32_t> selected_embedding_idx(curr_num_elements);
        std::iota(selected_embedding_idx.begin(), selected_embedding_idx.end(), 0);
        uint32_t beam_search_topk = std::min(dependency_topk, curr_num_elements);
        std::vector<float> sub_curr_scores((prob_agg == "prod") ? std::vector<float>(1, 1.0f) : std::vector<float>(1, 0.0f));
        //std::cout << "Line 3" << std::endl;
        std::vector<std::vector<uint32_t>> selected_patch_ids_ls;
        for (uint32_t sub_query_idx = 0; sub_query_idx < curr_grouped_sub_q_ids.size(); sub_query_idx++) {
            //good
            std::fill(count_buffer.begin(), count_buffer.begin() + _hashtable.numElements(), 0);
            _hashtable.queryByCount(query_hashes, curr_grouped_sub_q_ids[sub_query_idx] * _hashtable.numTables(), count_buffer);
            std::vector<float> curr_prod_mat(curr_num_elements);
            for (uint32_t idx = 0; idx < curr_num_elements; idx++) {
                curr_prod_mat[idx] = collision_count_to_sim[count_buffer[selected_embedding_idx[idx]]] / num_elements;
            }
            //std::cout << "Line 4" << std::endl;
            //good
            std::vector<std::vector<float>> prod_mat(curr_num_elements, std::vector<float>(sub_curr_scores.size()));
            if (prob_agg == "prod") {
                for (uint32_t i = 0; i < curr_prod_mat.size(); i++) {
                    if (curr_prod_mat[i] < 0.0f) {
                        curr_prod_mat[i] = 0.0f;
                    }
                }
                for (uint32_t i = 0; i < curr_prod_mat.size(); i++) {
                    for (uint32_t j = 0; j < sub_curr_scores.size(); j++) {
                        prod_mat[i][j] = curr_prod_mat[i] * sub_curr_scores[j];
                    }
                }
            } else {
                for (uint32_t i = 0; i < curr_prod_mat.size(); i++) {
                    for (uint32_t j = 0; j < sub_curr_scores.size(); j++) {
                        prod_mat[i][j] = curr_prod_mat[i] + sub_curr_scores[j];
                    }
                }
            }
            //std::cout << "Line 5" << std::endl;
            //good
            std::vector<float> flat_prod_mat;
            for (const std::vector<float> & row : prod_mat) {
                flat_prod_mat.insert(flat_prod_mat.end(), row.begin(), row.end());
            }
            // Find top k indices
            //good
            auto topk_results = argmax_topk(flat_prod_mat, beam_search_topk);
            std::vector<float> sub_curr_scores_ls = topk_results.second;
            std::vector<uint32_t> topk_ids = topk_results.first;
            //std::cout << "Line 6" << std::endl;
            // Calculate topk_emb_ids
            //good
            std::vector<uint32_t> topk_emb_ids(beam_search_topk);
            for (uint32_t i = 0; i < topk_ids.size(); i++) {
                topk_emb_ids[i] = selected_embedding_idx[topk_ids[i] / prod_mat[0].size()];
            }
            //good
            //std::cout << "Line 7" << std::endl;
            if (sub_query_idx == 0) {
                //std::cout << "Line 13" << std::endl;
                for (uint32_t i = 0; i < beam_search_topk; i++) {
                    std::vector<uint32_t> vec = {topk_emb_ids[i]};
                    selected_patch_ids_ls.push_back(vec);
                }
                //std::cout << "Line 14" << std::endl;
            } else {
                //std::cout << "Line 15" << std::endl;
                std::vector<uint32_t> selected_seq_ids(beam_search_topk);
                for (uint32_t i = 0; i < beam_search_topk; i++) {
                    selected_seq_ids[i] = topk_ids[i] % prod_mat[0].size();
                }
                //std::cout << "Line 16" << std::endl;
                // Create curr_selected_patch_ids_ls
                std::vector<std::vector<uint32_t>> curr_selected_patch_ids_ls;
                for (uint32_t selected_seq_id_idx = 0; selected_seq_id_idx < selected_seq_ids.size(); selected_seq_id_idx++) {
                    // Concatenate vectors
                    //std::cout << "Line 17" << std::endl;
                    std::vector<uint32_t> concatenated_vector = selected_patch_ids_ls[selected_seq_ids[selected_seq_id_idx]];
                    //std::cout << "Line 18" << std::endl;
                    concatenated_vector.push_back(topk_emb_ids[selected_seq_id_idx]);
                    //std::cout << "Line 19" << std::endl;
                    curr_selected_patch_ids_ls.push_back(concatenated_vector);
                    //std::cout << "Line 20" << std::endl;
                }
                //std::cout << "Line 21" << std::endl;
                selected_patch_ids_ls = curr_selected_patch_ids_ls;
            }
            //std::cout << "Line 8" << std::endl;
            //good
            std::set<uint32_t> existing_topk_emb_ids;
            for (const auto& selected_patch_ids : selected_patch_ids_ls) {
                existing_topk_emb_ids.insert(selected_patch_ids.begin(), selected_patch_ids.end());
            }
            //good
            std::set<uint32_t> updated_selected_embedding_idx;
            for (const auto& topk_id : existing_topk_emb_ids) {
                //updated_selected_embedding_idx.insert(bboxes_overlap_ls[corpus_idx][topk_id].begin(), bboxes_overlap_ls[corpus_idx][topk_id].end());
                updated_selected_embedding_idx.insert(partial_bboxes_overlap_ls[topk_id].begin(), partial_bboxes_overlap_ls[topk_id].end());
            }
            //std::cout << "Line 9" << std::endl;
            std::vector<uint32_t> updated_selected_embedding_idx_vec(updated_selected_embedding_idx.begin(), updated_selected_embedding_idx.end());
            selected_embedding_idx = updated_selected_embedding_idx_vec;
            //good
            sub_curr_scores = sub_curr_scores_ls;
            //std::cout << "Line 10" << std::endl;
        }
        //std::cout << "Line 11" << std::endl;
        //good
        if (prob_agg == "prod") {
            for (uint32_t i = 0; i < sub_curr_scores.size(); i++) {
                if (sub_curr_scores[i] < 0.0f) {
                    sub_curr_scores[i] = 0.0f;
                }
            }
            curr_scores_ls *= *std::max_element(sub_curr_scores.begin(), sub_curr_scores.end());
        } else {
            curr_scores_ls += *std::max_element(sub_curr_scores.begin(), sub_curr_scores.end());
        }
        //std::cout << "Line 12" << std::endl;
    }
    //good
    //std::cout << "Exited getScoreDependency" << std::endl;
    return curr_scores_ls;
}

template class MaxFlash<uint8_t>;
template class MaxFlash<uint16_t>;
template class MaxFlash<uint32_t>;

}  // namespace thirdai::search