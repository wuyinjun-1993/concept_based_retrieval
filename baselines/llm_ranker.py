import torch
import os,sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from baseline_utils import *
from llmrankers.setwise import SetwiseLlmRanker
from llmrankers.rankers import SearchResult

class LLM_ranker:
    def __init__(self, docs):
        docs = reformat_documents(docs)
        self.docs = [SearchResult(docid=i, text=docs[i], score=None) for i in range(len(docs))]
        self.ranker = SetwiseLlmRanker(model_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B',
                          tokenizer_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B',
                          device='cuda',
                          num_child=2,
                          scoring='generation',
                          method='heapsort',
                          k=2)
        
    def retrieval(self, queries):
        queries = reformat_queries(queries)
        all_scores = torch.zeros(len(queries), len(self.docs))
        with torch.no_grad():
            for q_idx in range(len(queries)):
                query = queries[q_idx]
                curr_res_ls = self.ranker.rerank(query, self.docs)
                for i in range(len(curr_res_ls)):
                    all_scores[q_idx][i] = curr_res_ls[i].score
        
        result = tensor_res_to_res_eval(all_cos_scores_tensor)
        return result
                
        
            
