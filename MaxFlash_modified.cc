#include "MaxFlash.h"
#include <cassert>
#include <stdexcept>

#include "Utils.h"
#include <cmath>
//#include <cfloat>  // Include this header for FLT_MAX
#include <algorithm>
#include <utility>
#include <set>
#include <numeric>
#include<iostream>

#include <Eigen/Dense>

//#include <torch/torch.h>

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
    // if (is_img_retrieval && num_elements == 1) {
    //     max_count = count_buffer[count_buffer.size()-1];
    // } else {
        for (uint32_t i = 0; i < _hashtable.numElements(); i++) {
            if (count_buffer[i] > max_count) {
                max_count = count_buffer[i];
            }
        // }
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

    float curr_scores_ls = (prob_agg == "prod") ? 1.0f : 0.0f;
    std::vector<std::vector<uint32_t>> curr_grouped_sub_q_ids_ls;

    if (!partial_grouped_sub_q_ids_ls.empty()) {
        curr_grouped_sub_q_ids_ls = partial_grouped_sub_q_ids_ls[sub_q_ls_idx];
    } else {
        curr_grouped_sub_q_ids_ls.push_back(std::vector<uint32_t>(num_elements));
        std::iota(curr_grouped_sub_q_ids_ls[0].begin(), curr_grouped_sub_q_ids_ls[0].end(), 0);
    }

    uint32_t curr_num_elements = is_img_retrieval ? _hashtable.numElements() - 1 : _hashtable.numElements();
    //std::vector<int> selected_embedding_idx(curr_num_elements);
    uint32_t beam_search_topk = std::min(dependency_topk, curr_num_elements);
    //Eigen::VectorXf curr_prod_mat(curr_num_elements);
    //std::vector<float> flat_prod_mat;
    
    #pragma omp parallel for reduction(reduce_op:curr_scores_ls)
    for (const auto& curr_grouped_sub_q_ids : curr_grouped_sub_q_ids_ls) {
        std::vector<int> selected_embedding_idx(curr_num_elements);
        Eigen::VectorXf curr_prod_mat(curr_num_elements);
        std::vector<float> flat_prod_mat;
        
        std::iota(selected_embedding_idx.begin(), selected_embedding_idx.end(), 0);
        Eigen::VectorXf sub_curr_scores = (prob_agg == "prod") ? Eigen::VectorXf::Ones(1) : Eigen::VectorXf::Zero(1);
        std::vector<std::vector<uint32_t>> selected_patch_ids_ls(beam_search_topk);
        
        for (uint32_t sub_query_idx = 0; sub_query_idx < curr_grouped_sub_q_ids.size(); ++sub_query_idx) {
            std::fill(count_buffer.begin(), count_buffer.begin() + _hashtable.numElements(), 0);
            Eigen::MatrixXf prod_mat(curr_num_elements, sub_curr_scores.size());
            
            _hashtable.queryByCount(query_hashes, curr_grouped_sub_q_ids[sub_query_idx] * _hashtable.numTables(), count_buffer);
            
            #pragma omp parallel for simd
            for (uint32_t idx = 0; idx < curr_num_elements; ++idx) {
                curr_prod_mat[idx] = collision_count_to_sim[count_buffer[selected_embedding_idx[idx]]];
            }
            
            curr_prod_mat /= num_elements;

            if (prob_agg == "prod") {
                prod_mat = curr_prod_mat * sub_curr_scores.transpose();
            } else {
                prod_mat = curr_prod_mat.replicate(1, sub_curr_scores.size()).matrix() + sub_curr_scores.transpose().matrix();
            }

            //std::vector<float> flat_prod_mat(prod_mat.data(), prod_mat.data() + prod_mat.size());
            // Flatten prod_mat in a parallelized and vectorized way
            flat_prod_mat.resize(prod_mat.size());
            #pragma omp parallel for simd
            for (Eigen::Index i = 0; i < prod_mat.size(); ++i) {
                flat_prod_mat[i] = prod_mat(i);
            }

            auto topk_results = argmax_topk(flat_prod_mat, beam_search_topk);
            Eigen::VectorXf sub_curr_scores_ls = Eigen::Map<Eigen::VectorXf>(topk_results.second.data(), topk_results.second.size());
            std::vector<int> topk_ids(topk_results.first.begin(), topk_results.first.end());

            // Vectorize the first loop
            Eigen::VectorXi eigen_topk_ids = Eigen::Map<Eigen::VectorXi>(topk_ids.data(), topk_ids.size());
            Eigen::VectorXi topk_emb_ids = eigen_topk_ids.array() / static_cast<int>(prod_mat.cols());
            
            #pragma omp parallel for simd
            for (Eigen::Index i = 0; i < topk_ids.size(); ++i) {
                topk_emb_ids[i] = selected_embedding_idx[topk_emb_ids[i]];
            }

            // Vectorized modulus operation
            Eigen::VectorXi selected_seq_ids = eigen_topk_ids.array() - (prod_mat.cols() * (eigen_topk_ids.array().cast<float>() / prod_mat.cols()).floor().cast<int>());

            if (sub_query_idx == 0) {
                #pragma omp parallel for
                for (uint32_t i = 0; i < beam_search_topk; ++i) {
                    selected_patch_ids_ls[i].push_back(topk_emb_ids[i]);
                }
            } else {
                std::vector<std::vector<uint32_t>> curr_selected_patch_ids_ls(selected_seq_ids.size());
                #pragma omp parallel for
                for (Eigen::Index selected_seq_id_idx = 0; selected_seq_id_idx < selected_seq_ids.size(); ++selected_seq_id_idx) {
                    curr_selected_patch_ids_ls[selected_seq_id_idx].push_back(selected_patch_ids_ls[selected_seq_ids[selected_seq_id_idx]].push_back(topk_emb_ids[selected_seq_id_idx]));
                }
                selected_patch_ids_ls = std::move(curr_selected_patch_ids_ls);
            }

            std::set<uint32_t> existing_topk_emb_ids;
            //hmm
            for (const auto& selected_patch_ids : selected_patch_ids_ls) {
                existing_topk_emb_ids.insert(selected_patch_ids.begin(), selected_patch_ids.end());
            }

            std::set<uint32_t> updated_selected_embedding_idx;
            //hmm
            for (const auto& topk_id : existing_topk_emb_ids) {
                updated_selected_embedding_idx.insert(partial_bboxes_overlap_ls[topk_id].begin(), partial_bboxes_overlap_ls[topk_id].end());
            }
            selected_embedding_idx.assign(updated_selected_embedding_idx.begin(), updated_selected_embedding_idx.end());
            sub_curr_scores = sub_curr_scores_ls;
        }

        if (prob_agg == "prod") {
            curr_scores_ls *= sub_curr_scores.maxCoeff();
        } else {
            curr_scores_ls += sub_curr_scores.maxCoeff();
        }
    }

    return curr_scores_ls;
}

template class MaxFlash<uint8_t>;
template class MaxFlash<uint16_t>;
template class MaxFlash<uint32_t>;

}  // namespace thirdai::search