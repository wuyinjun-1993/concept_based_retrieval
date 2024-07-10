#include "../src/DocSearch.h"
#include <pybind11/stl.h>


// Pybind11 library
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <src/MaxFlashArray.h>

namespace thirdai::search::python {

PYBIND11_MODULE(dessert_py_dependency, m) {  // NOLINT

  // TODO(josh): Comment this class more
  py::class_<DocSearch>(
      m, "DocRetrieval",
      "The DocRetrieval module allows you to build, query, save, and load a "
      "semantic document search index.")
      .def(py::init<uint32_t, uint32_t,
                    uint32_t, uint32_t, const Centroids&>(), 
           py::arg("hashes_per_table"), py::arg("num_tables"), 
           py::arg("dense_input_dimension"), py::arg("nprobe_query"), py::arg("centroids"), 
           "Constructs a new DocRetrieval index. Centroids should be a "
           "two-dimensional array of floats, where each row is of length "
           "dense_input_dimension (the dimension of the document embeddings). "
           "hashes_per_table and num_tables are hyperparameters for the doc "
           "sketches. Roughly, increase num_tables to increase accuracy at the"
           "cost of speed and memory (you can try powers of 2; a good starting "
           "value is 32). Hashes_per_table should be around log_2 the average"
           "document size (by number of embeddings).")
      .def("add_doc", &DocSearch::addDocument, py::arg("doc_embeddings"),
           py::arg("doc_id"),
           "Adds a new document to the DocRetrieval index. If the doc_id "
           "already exists in the index, this will overwrite it. The "
           "doc_embeddings should be a two dimensional numpy array of the "
           "document's embeddings. Each row should be of length "
           "dense_input_dimension. Returns true if this"
           "was a new document and false otherwise.")
      .def("query", &DocSearch::query, py::arg("query_embeddings"),
          py::arg("top_k"), py::arg("num_to_rerank") = 8192, py::arg("prob_agg"),
          py::arg("use_frequency"), py::arg("is_img_retrieval"),
          "Finds the best top_k documents that are most likely to semantically "
          "answer the query. There is an additional optional parameter here "
          "called num_to_rerank that represents how many documents you want "
          "us to "
          "internally rerank. The default of 8192 is fine for most use cases.")
      .def("querywithdependency", &DocSearch::querywithdependency, py::arg("query_embeddings"), py::arg("top_k"), py::arg("num_to_rerank"), py::arg("prob_agg"), py::arg("use_frequency"), py::arg("is_img_retrieval"), py::arg("dependency_topk"), py::arg("query_idx"), py::arg("query_sub_idx"), py::arg("grouped_sub_q_ids_ls"), py::arg("bboxes_overlap_ls"),
          "Finds the best top_k documents that are most likely to semantically "
          "answer the query. There is an additional optional parameter here "
          "called num_to_rerank that represents how many documents you want "
          "us to "
          "internally rerank. The default of 8192 is fine for most use cases.")
      .def("serialize_to_file", &DocSearch::serialize_to_file,
           py::arg("output_path"),
           "Serialize the DocRetrieval index to a file.")
      .def_static("deserialize_from_file", &DocSearch::deserialize_from_file,
                  py::arg("input_path"),
                  "Deserialize the DocRetrieval index from a file.");
}

}  // namespace thirdai::search::python