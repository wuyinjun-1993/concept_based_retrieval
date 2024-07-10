#pragma once

#include <numeric>
#include <queue>
#include <vector>
#include "EigenDenseWrapper.h"

namespace thirdai::search {

template <class T>
void removeDuplicates(std::vector<T>& v) {
  std::sort(v.begin(), v.end());
  v.erase(unique(v.begin(), v.end()), v.end());
}

// Consumes a min heap of pairs and returns a vector sorted in descending order
// of the second items in the pairs, where descending order is defined by
// the min heap order (the current top element in the min heap is the smallest,
// etc.)
template <class T>
std::pair<std::vector<uint32_t>, std::vector<float>> minHeapPairsToDescending(
    std::priority_queue<std::pair<T, uint32_t>>& min_heap) {
  std::vector<uint32_t> result_indices;
  std::vector<float> result_values;
  while (!min_heap.empty()) {
    result_indices.push_back(min_heap.top().second);
    result_values.push_back(-min_heap.top().first);  // Negate back the value
    min_heap.pop();
  }

  std::reverse(result_indices.begin(), result_indices.end());
  std::reverse(result_values.begin(), result_values.end());
  return {result_indices, result_values};
}

// Identifies the indices of the largest k elements in an Eigen Float Vector
inline std::vector<uint32_t> argmax(const Eigen::VectorXf& input,
                                    uint32_t top_k) {
  // We negate values so that we can treat the stl priority queue, which is a
  // max-heap, as a min-heap.
  std::priority_queue<std::pair<float, uint32_t>> min_heap;
  for (uint32_t i = 0; i < input.size(); i++) {
    if (min_heap.size() < top_k) {
      min_heap.emplace(-input[i], i);
    } else if (-input[i] < min_heap.top().first) {
      min_heap.pop();
      min_heap.emplace(-input[i], i);
    }
  }

  return minHeapPairsToDescending(min_heap).first;
}

// Identifies the indices of the largest k elements in a std::vector<float>
inline std::pair<std::vector<uint32_t>, std::vector<float>> argmax_topk(
    const std::vector<float>& input, uint32_t top_k) {
  std::priority_queue<std::pair<float, uint32_t>> min_heap;
  for (uint32_t i = 0; i < input.size(); i++) {
    if (min_heap.size() < top_k) {
      min_heap.emplace(-input[i], i);  // Negate the value
    } else if (-input[i] < min_heap.top().first) {
      min_heap.pop();
      min_heap.emplace(-input[i], i);  // Negate the value
    }
  }

  return minHeapPairsToDescending(min_heap);
}

// Performs an argsort on the input vector. The sort is descending, e.g. the
// index of the largest element in to_argsort is the first element in the
// result. to_argsort should be a vector of size less than UINT32_MAX.
template <class T>
std::vector<uint32_t> argsort_descending(const std::vector<T> to_argsort) {
  // https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
  std::vector<uint32_t> indices(to_argsort.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&to_argsort](size_t i1, size_t i2) {
              return to_argsort[i1] > to_argsort[i2];
            });
  return indices;
}

}  // namespace thirdai::search