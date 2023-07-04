from .util import cos_sim, dot_score
import logging
import sys
import torch
from typing import Dict, List

logger = logging.getLogger(__name__)

#Parent class for any dense model
class DenseRetrievalExactSearch:
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True #TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.results = {}
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict, 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False, 
               query_negations: List=None, all_sub_corpus_embedding_ls=None,
               **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in query_ids]
        if query_negations is not None:
            query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
        query_embeddings=[]
        for idx in range(len(queries)):
            curr_query = queries[idx]
            if type(curr_query) is str:
                curr_query_embedding = self.model.encode_queries(
                    curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
            elif type(curr_query) is list:
                curr_query_embedding = []
                for k in range(len(curr_query)):
                    curr_conjunct = []
                    for j in range(len(curr_query[k])):
                        qe = self.model.encode_queries(
                            curr_query[k][j], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
                        curr_conjunct.append(qe)
                    curr_query_embedding.append(curr_conjunct)
            query_embeddings.append(curr_query_embedding)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)

        all_cos_scores = []
        if all_sub_corpus_embedding_ls is None:
            all_sub_corpus_embedding_ls = []
            for batch_num, corpus_start_idx in enumerate(itr):
                logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

                #Encode chunk of corpus    
                sub_corpus_embeddings = self.model.encode_corpus(
                    corpus[corpus_start_idx:corpus_end_idx],
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar, 
                    convert_to_tensor = self.convert_to_tensor
                    )
                
                all_sub_corpus_embedding_ls.append(sub_corpus_embeddings)
        
        for sub_corpus_embeddings in all_sub_corpus_embedding_ls:

            #Compute similarites using either cosine-similarity or dot product
            cos_scores = []
            for query_itr in range(len(query_embeddings)):
                curr_query_embedding = query_embeddings[query_itr]
                if type(curr_query_embedding) is list:
                    curr_scores = 0
                    for conj_id in range(len(curr_query_embedding)):
                        curr_cos_scores_ls = self.score_functions[score_function](torch.stack(curr_query_embedding[conj_id]), sub_corpus_embeddings)
                        if query_negations is not None and query_negations[query_itr] is not None:
                            curr_query_negations = torch.tensor(query_negations[query_itr])
                            curr_cos_scores_ls[curr_query_negations == 1] =  - curr_cos_scores_ls[curr_query_negations == 1]

                        curr_cos_scores_ls[torch.isnan(curr_cos_scores_ls)] = -1

                        curr_cos_scores = 1
                        for idx in range(len(curr_cos_scores_ls)):
                            curr_cos_scores *= curr_cos_scores_ls[idx]
                        curr_scores += curr_cos_scores
                else:
                    curr_cos_scores = self.score_functions[score_function](curr_query_embedding.unsqueeze(0), sub_corpus_embeddings)
                    curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
                    curr_scores = curr_cos_scores.squeeze(0)
                cos_scores.append(curr_scores)
            cos_scores = torch.stack(cos_scores)
            all_cos_scores.append(cos_scores)
        
        all_cos_scores_tensor = torch.cat(all_cos_scores, dim=-1)
        #Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(all_cos_scores_tensor, min(top_k+1, len(all_cos_scores_tensor[0])), dim=1, largest=True, sorted=return_sorted)
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
        for query_itr in range(len(query_embeddings)):
            query_id = query_ids[query_itr]                  
            for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                corpus_id = corpus_ids[sub_corpus_id]
                if corpus_id != query_id:
                    self.results[query_id][corpus_id] = score
        
        return self.results, all_sub_corpus_embedding_ls
