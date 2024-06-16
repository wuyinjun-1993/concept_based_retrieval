import torch
import os,sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from baseline_utils import *
from llmrankers.setwise import SetwiseLlmRanker
from llmrankers.rankers import SearchResult
import time
class LLM_ranker:
    def __init__(self, docs):
        docs = reformat_documents(docs)
        self.docs = [SearchResult(docid=i, text=docs[i], score=None) for i in range(len(docs))]
        self.ranker = SetwiseLlmRanker(model_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B',
                          tokenizer_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B',
                          device='cuda',
                          num_child=1,
                          scoring='generation',
                          method='heapsort',
                          k=2)
        
    def retrieval(self, queries):
        queries = reformat_queries(queries)
        all_scores = torch.zeros(len(queries), len(self.docs))
        t1 = time.time()
        with torch.no_grad():
            for q_idx in range(len(queries)):
                query = queries[q_idx]
                curr_res_ls = self.ranker.rerank(query, self.docs)
                for i in range(len(curr_res_ls)):
                    all_scores[q_idx][i] = curr_res_ls[i].score
                    
        t2 = time.time()
        torch.save(all_scores, "llm_ranker_scores.pt")
        result = tensor_res_to_res_eval(all_scores)
        print("llm_ranker time::", t2 - t1)
        return result
                
        
            
