from .util import cos_sim, dot_score, load_single_embeddings
import logging
import sys
import torch
from typing import Dict, List
from tqdm import tqdm
logger = logging.getLogger(__name__)
import concurrent.futures
import multiprocessing
from joblib import Parallel, delayed
import threading
import os

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
               query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10, device = 'cuda',
               **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        if queries is not None:
            query_ids = list(queries.keys())
        else:
            if query_embeddings is not None:
                query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
            else:
                raise ValueError("Either queries or query_embeddings must be provided!")
        self.results = {qid: {} for qid in query_ids}
        if queries is not None:
            queries = [queries[qid] for qid in query_ids]
        if query_negations is not None:
            query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
        if query_embeddings is None:
            query_embeddings=[]
            for idx in range(len(queries)):
                curr_query = queries[idx]
                if type(curr_query) is str:
                    curr_query_embedding_ls = self.model.encode_queries(
                        curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
                elif type(curr_query) is list:
                    curr_query_embedding_ls = []
                    for k in range(len(curr_query)):
                        curr_conjunct = []
                        for j in range(len(curr_query[k])):
                            qe = self.model.encode_queries(
                                curr_query[k][j], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
                            curr_conjunct.append(qe)
                        curr_query_embedding_ls.append(curr_conjunct)
                query_embeddings.append(curr_query_embedding_ls)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        if corpus is not None:
            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
            corpus = [corpus[cid] for cid in corpus_ids]
        else:
            if all_sub_corpus_embedding_ls is not None:
                corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
            else:
                raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        if query_count < 0:
            query_count = len(query_embeddings)

        all_cos_scores = []
        if all_sub_corpus_embedding_ls is None:
            itr = range(0, len(corpus), self.corpus_chunk_size)
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
        
        # all_sub_corpus_embedding_ls = [item.to(device) for item in all_sub_corpus_embedding_ls]
        
        for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):

            #Compute similarites using either cosine-similarity or dot product
            cos_scores = []
            # for query_itr in range(len(query_embeddings)):
            for query_itr in range(query_count):
                curr_query_embedding_ls = query_embeddings[query_itr]
                if type(curr_query_embedding_ls) is list:
                    full_curr_scores_ls = []
                    # for conj_id in range(len(curr_query_embedding)):
                    #     curr_cos_scores_ls = self.score_functions[score_function](torch.stack(curr_query_embedding[conj_id]), sub_corpus_embeddings)
                    #     if query_negations is not None and query_negations[query_itr] is not None:
                    #         curr_query_negations = torch.tensor(query_negations[query_itr])
                    #         curr_cos_scores_ls[curr_query_negations == 1] =  - curr_cos_scores_ls[curr_query_negations == 1]

                    #     curr_cos_scores_ls[torch.isnan(curr_cos_scores_ls)] = -1

                    #     curr_cos_scores = 1
                    #     for idx in range(len(curr_cos_scores_ls)):
                    #         curr_cos_scores *= curr_cos_scores_ls[idx]
                    #     curr_scores += curr_cos_scores
                    for curr_query_embedding in curr_query_embedding_ls:
                        curr_scores = 1
                        if len(sub_corpus_embeddings.shape) == 2 and sub_corpus_embeddings.shape[0] > 1:
                            curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
                        else:    
                            curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device))
                        for conj_id in range(len(curr_scores_ls)):
                            curr_scores *= curr_scores_ls[conj_id]
                        # full_curr_scores += curr_scores
                        full_curr_scores_ls.append(curr_scores)
                    
                    curr_scores = torch.tensor(full_curr_scores_ls)
                    
                else:
                    # curr_cos_scores = self.score_functions[score_function](curr_query_embedding.unsqueeze(0), sub_corpus_embeddings)
                    curr_cos_scores = self.score_functions[score_function](curr_query_embedding_ls, sub_corpus_embeddings)
                    curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
                    curr_scores = curr_cos_scores.squeeze(0)
                    if len(curr_scores) > 1:
                        curr_scores = torch.max(curr_scores, dim=-1)[0]
                    curr_scores = curr_scores.view(-1)
                cos_scores.append(curr_scores)
            cos_scores = torch.stack(cos_scores)
            all_cos_scores.append(cos_scores)
        
        all_cos_scores_tensor = torch.stack(all_cos_scores, dim=-1)
        all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
        all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
        # all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
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
                self.results[query_id][corpus_id] = score
        
        return self.results, all_sub_corpus_embedding_ls
    
    def search_by_clusters(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict, 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False, 
               query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10, device = 'cuda', clustering_info=None, topk_embs = 100,
               **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        
        
        assert clustering_info is not None
        cluster_sub_X_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_sample_ids_ls = clustering_info
        
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        if queries is not None:
            query_ids = list(queries.keys())
        else:
            if query_embeddings is not None:
                query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
            else:
                raise ValueError("Either queries or query_embeddings must be provided!")
        self.results = {qid: {} for qid in query_ids}
        if queries is not None:
            queries = [queries[qid] for qid in query_ids]
        if query_negations is not None:
            query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
        if query_embeddings is None:
            query_embeddings=[]
            for idx in range(len(queries)):
                curr_query = queries[idx]
                if type(curr_query) is str:
                    curr_query_embedding_ls = self.model.encode_queries(
                        curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
                elif type(curr_query) is list:
                    curr_query_embedding_ls = []
                    for k in range(len(curr_query)):
                        curr_conjunct = []
                        for j in range(len(curr_query[k])):
                            qe = self.model.encode_queries(
                                curr_query[k][j], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
                            curr_conjunct.append(qe)
                        curr_query_embedding_ls.append(curr_conjunct)
                query_embeddings.append(curr_query_embedding_ls)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        if corpus is not None:
            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
            corpus = [corpus[cid] for cid in corpus_ids]
        else:
            if all_sub_corpus_embedding_ls is not None:
                corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
            else:
                raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        if query_count < 0:
            query_count = len(query_embeddings)

        all_cos_scores = []
        if all_sub_corpus_embedding_ls is None:
            itr = range(0, len(corpus), self.corpus_chunk_size)
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
        
        all_cos_scores_tensor = torch.zeros((len(all_sub_corpus_embedding_ls), len(query_embeddings[0]),  query_count), device=device)
        
        for query_itr in tqdm(range(query_count)):
            curr_query_embedding_ls = query_embeddings[query_itr]
            if type(curr_query_embedding_ls) is list:
                full_curr_scores_ls = []
                # for conj_id in range(len(curr_query_embedding)):
                #     curr_cos_scores_ls = self.score_functions[score_function](torch.stack(curr_query_embedding[conj_id]), sub_corpus_embeddings)
                #     if query_negations is not None and query_negations[query_itr] is not None:
                #         curr_query_negations = torch.tensor(query_negations[query_itr])
                #         curr_cos_scores_ls[curr_query_negations == 1] =  - curr_cos_scores_ls[curr_query_negations == 1]

                #     curr_cos_scores_ls[torch.isnan(curr_cos_scores_ls)] = -1

                #     curr_cos_scores = 1
                #     for idx in range(len(curr_cos_scores_ls)):
                #         curr_cos_scores *= curr_cos_scores_ls[idx]
                #     curr_scores += curr_cos_scores
                
                curr_scores = 1
                for sub_query_itr in range(len(curr_query_embedding_ls)):
                    curr_query_embedding = curr_query_embedding_ls[sub_query_itr]
                    
                    curr_scores_mat = self.score_functions[score_function](curr_query_embedding.to(device), cluster_centroid_tensor.to(device))
                    
                    # for sub_q_idx in range(curr_scores_mat.shape[0]):
                    #     curr_scores *= curr_scores_mat[sub_q_idx]
                    # 
                    
                    curr_scores = torch.prod(curr_scores_mat, dim=0)
                    
                    topk_cluster_ids = torch.argsort(curr_scores, descending=True)
                    
                    # cluster_sample_cum_count = torch.cumsum(cluster_sample_count_ls[topk_cluster_ids])
                    
                    covered_sample_id_set = set()
                    
                    for cluster_id in topk_cluster_ids:
                        curr_cluster_sample_ids = cluster_sample_ids_ls[cluster_id]
                    
                        curr_scores_mat_curr_cluster = self.score_functions[score_function](curr_query_embedding.to(device), cluster_sub_X_ls[cluster_id].to(device))
                        
                        curr_scores_mat_curr_cluster = torch.prod(curr_scores_mat_curr_cluster, dim=0)
                        
                        all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr] = (curr_scores_mat_curr_cluster > all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr])*curr_scores[cluster_id] \
                            + (curr_scores_mat_curr_cluster <= all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr])*all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr]
                        
                        covered_sample_id_set.update(curr_cluster_sample_ids.tolist())
                        if len(covered_sample_id_set) > topk_embs:
                            break
                    
                
                # curr_scores = torch.tensor(full_curr_scores_ls)
                
        #     else:
        #         # curr_cos_scores = self.score_functions[score_function](curr_query_embedding.unsqueeze(0), sub_corpus_embeddings)
        #         curr_cos_scores = self.score_functions[score_function](curr_query_embedding_ls, sub_corpus_embeddings)
        #         curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
        #         curr_scores = curr_cos_scores.squeeze(0)
        #         if len(curr_scores) > 1:
        #             curr_scores = torch.max(curr_scores, dim=-1)[0]
        #         curr_scores = curr_scores.view(-1)
        #     cos_scores.append(curr_scores)
        # cos_scores = torch.stack(cos_scores)
        
        
        # all_sub_corpus_embedding_ls = [item.to(device) for item in all_sub_corpus_embedding_ls]
        
        # for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):

        #     #Compute similarites using either cosine-similarity or dot product
        #     cos_scores = []
        #     # for query_itr in range(len(query_embeddings)):
        #     for query_itr in range(query_count):
        #         curr_query_embedding_ls = query_embeddings[query_itr]
        #         if type(curr_query_embedding_ls) is list:
        #             full_curr_scores_ls = []
        #             # for conj_id in range(len(curr_query_embedding)):
        #             #     curr_cos_scores_ls = self.score_functions[score_function](torch.stack(curr_query_embedding[conj_id]), sub_corpus_embeddings)
        #             #     if query_negations is not None and query_negations[query_itr] is not None:
        #             #         curr_query_negations = torch.tensor(query_negations[query_itr])
        #             #         curr_cos_scores_ls[curr_query_negations == 1] =  - curr_cos_scores_ls[curr_query_negations == 1]

        #             #     curr_cos_scores_ls[torch.isnan(curr_cos_scores_ls)] = -1

        #             #     curr_cos_scores = 1
        #             #     for idx in range(len(curr_cos_scores_ls)):
        #             #         curr_cos_scores *= curr_cos_scores_ls[idx]
        #             #     curr_scores += curr_cos_scores
        #             for curr_query_embedding in curr_query_embedding_ls:
        #                 curr_scores = 1
        #                 if len(sub_corpus_embeddings.shape) == 2 and sub_corpus_embeddings.shape[0] > 1:
        #                     curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
        #                 else:    
        #                     curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device))
        #                 for conj_id in range(len(curr_scores_ls)):
        #                     curr_scores *= curr_scores_ls[conj_id]
        #                 # full_curr_scores += curr_scores
        #                 full_curr_scores_ls.append(curr_scores)
                    
        #             curr_scores = torch.tensor(full_curr_scores_ls)
                    
        #         else:
        #             # curr_cos_scores = self.score_functions[score_function](curr_query_embedding.unsqueeze(0), sub_corpus_embeddings)
        #             curr_cos_scores = self.score_functions[score_function](curr_query_embedding_ls, sub_corpus_embeddings)
        #             curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
        #             curr_scores = curr_cos_scores.squeeze(0)
        #             if len(curr_scores) > 1:
        #                 curr_scores = torch.max(curr_scores, dim=-1)[0]
        #             curr_scores = curr_scores.view(-1)
        #         cos_scores.append(curr_scores)
        #     cos_scores = torch.stack(cos_scores)
        #     all_cos_scores.append(cos_scores)
        
        # all_cos_scores_tensor = torch.stack(all_cos_scores, dim=-1)
        all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
        all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
        # all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
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
                self.results[query_id][corpus_id] = score
        
        return self.results, all_sub_corpus_embedding_ls
    
    def search_in_disk(self, 
               corpus: Dict[str, Dict[str, str]], store_path: str,
               queries: Dict, 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False, 
               query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10, device = 'cuda',corpus_count=1,
               **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        if queries is not None:
            query_ids = list(queries.keys())
        else:
            if query_embeddings is not None:
                query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
            else:
                raise ValueError("Either queries or query_embeddings must be provided!")
        self.results = {qid: {} for qid in query_ids}
        if queries is not None:
            queries = [queries[qid] for qid in query_ids]
        if query_negations is not None:
            query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
        if query_embeddings is None:
            query_embeddings=[]
            for idx in range(len(queries)):
                curr_query = queries[idx]
                if type(curr_query) is str:
                    curr_query_embedding_ls = self.model.encode_queries(
                        curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
                elif type(curr_query) is list:
                    curr_query_embedding_ls = []
                    for k in range(len(curr_query)):
                        curr_conjunct = []
                        for j in range(len(curr_query[k])):
                            qe = self.model.encode_queries(
                                curr_query[k][j], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
                            curr_conjunct.append(qe)
                        curr_query_embedding_ls.append(curr_conjunct)
                query_embeddings.append(curr_query_embedding_ls)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        

        corpus_ids = [str(idx + 1) for idx in range(corpus_count)]

        # if corpus is not None:
        #     corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        #     corpus = [corpus[cid] for cid in corpus_ids]
        # else:
        #     if all_sub_corpus_embedding_ls is not None:
        #         corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
        #     else:
        #         raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        if query_count < 0:
            query_count = len(query_embeddings)

        all_cos_scores = []
        # if all_sub_corpus_embedding_ls is None:
        #     itr = range(0, len(corpus), self.corpus_chunk_size)
        #     all_sub_corpus_embedding_ls = []
        #     for batch_num, corpus_start_idx in enumerate(itr):
        #         logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
        #         corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

        #         #Encode chunk of corpus    
        #         sub_corpus_embeddings = self.model.encode_corpus(
        #             corpus[corpus_start_idx:corpus_end_idx],
        #             batch_size=self.batch_size,
        #             show_progress_bar=self.show_progress_bar, 
        #             convert_to_tensor = self.convert_to_tensor
        #             )
                
        #         all_sub_corpus_embedding_ls.append(sub_corpus_embeddings)
        
        # all_sub_corpus_embedding_ls = [item.to(device) for item in all_sub_corpus_embedding_ls]
        
        # for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):
        for corpus_id in tqdm(corpus_ids):
            # sub_folder = os.path.join(store_path, corpus_id)
            sub_corpus_embeddings = load_single_embeddings(store_path, int(corpus_id)-1)

            #Compute similarites using either cosine-similarity or dot product
            cos_scores = []
            # for query_itr in range(len(query_embeddings)):
            for query_itr in range(query_count):
                curr_query_embedding_ls = query_embeddings[query_itr]
                if type(curr_query_embedding_ls) is list:
                    full_curr_scores_ls = []
                    # for conj_id in range(len(curr_query_embedding)):
                    #     curr_cos_scores_ls = self.score_functions[score_function](torch.stack(curr_query_embedding[conj_id]), sub_corpus_embeddings)
                    #     if query_negations is not None and query_negations[query_itr] is not None:
                    #         curr_query_negations = torch.tensor(query_negations[query_itr])
                    #         curr_cos_scores_ls[curr_query_negations == 1] =  - curr_cos_scores_ls[curr_query_negations == 1]

                    #     curr_cos_scores_ls[torch.isnan(curr_cos_scores_ls)] = -1

                    #     curr_cos_scores = 1
                    #     for idx in range(len(curr_cos_scores_ls)):
                    #         curr_cos_scores *= curr_cos_scores_ls[idx]
                    #     curr_scores += curr_cos_scores
                    for curr_query_embedding in curr_query_embedding_ls:
                        curr_scores = 1
                        if len(sub_corpus_embeddings.shape) == 2 and sub_corpus_embeddings.shape[0] > 1:
                            curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
                        else:    
                            curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device))
                        for conj_id in range(len(curr_scores_ls)):
                            curr_scores *= curr_scores_ls[conj_id]
                        # full_curr_scores += curr_scores
                        full_curr_scores_ls.append(curr_scores)
                    
                    curr_scores = torch.tensor(full_curr_scores_ls)
                    
                else:
                    # curr_cos_scores = self.score_functions[score_function](curr_query_embedding.unsqueeze(0), sub_corpus_embeddings)
                    curr_cos_scores = self.score_functions[score_function](curr_query_embedding_ls, sub_corpus_embeddings)
                    curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
                    curr_scores = curr_cos_scores.squeeze(0)
                    if len(curr_scores) > 1:
                        curr_scores = torch.max(curr_scores, dim=-1)[0]
                    curr_scores = curr_scores.view(-1)
                cos_scores.append(curr_scores)
            cos_scores = torch.stack(cos_scores)
            all_cos_scores.append(cos_scores)
        
        all_cos_scores_tensor = torch.stack(all_cos_scores, dim=-1)
        all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
        all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
        # all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
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
                self.results[query_id][corpus_id] = score
        
        return self.results, all_sub_corpus_embedding_ls
    
    def search_parallel(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict, 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False, 
               query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10, batch_size=4, device='cuda',
               **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        if queries is not None:
            query_ids = list(queries.keys())
        else:
            if query_embeddings is not None:
                query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
            else:
                raise ValueError("Either queries or query_embeddings must be provided!")
        self.results = {qid: {} for qid in query_ids}
        if queries is not None:
            queries = [queries[qid] for qid in query_ids]
        if query_negations is not None:
            query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
        if query_embeddings is None:
            query_embeddings=[]
            for idx in range(len(queries)):
                curr_query = queries[idx]
                if type(curr_query) is str:
                    curr_query_embedding_ls = self.model.encode_queries(
                        curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
                elif type(curr_query) is list:
                    curr_query_embedding_ls = []
                    for k in range(len(curr_query)):
                        curr_conjunct = []
                        for j in range(len(curr_query[k])):
                            qe = self.model.encode_queries(
                                curr_query[k][j], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
                            curr_conjunct.append(qe)
                        curr_query_embedding_ls.append(curr_conjunct)
                query_embeddings.append(curr_query_embedding_ls)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        if corpus is not None:
            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
            corpus = [corpus[cid] for cid in corpus_ids]
        else:
            if all_sub_corpus_embedding_ls is not None:
                corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
            else:
                raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        if query_count < 0:
            query_count = len(query_embeddings)

        all_cos_scores = []
        if all_sub_corpus_embedding_ls is None:
            itr = range(0, len(corpus), self.corpus_chunk_size)
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
        
        # query_embeddings = [[torch.cat(item).to(device) for item in items] for items in query_embeddings]
        
        # for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):

            #Compute similarites using either cosine-similarity or dot product
        def compute_cos_scores(sub_corpus_embeddings):
            cos_scores = []
            # for query_itr in range(len(query_embeddings)):
            for query_itr in range(query_count):
                curr_query_embedding_ls = query_embeddings[query_itr]
                if type(curr_query_embedding_ls) is list:
                    full_curr_scores_ls = []
                    # for conj_id in range(len(curr_query_embedding)):
                    #     curr_cos_scores_ls = self.score_functions[score_function](torch.stack(curr_query_embedding[conj_id]), sub_corpus_embeddings)
                    #     if query_negations is not None and query_negations[query_itr] is not None:
                    #         curr_query_negations = torch.tensor(query_negations[query_itr])
                    #         curr_cos_scores_ls[curr_query_negations == 1] =  - curr_cos_scores_ls[curr_query_negations == 1]

                    #     curr_cos_scores_ls[torch.isnan(curr_cos_scores_ls)] = -1

                    #     curr_cos_scores = 1
                    #     for idx in range(len(curr_cos_scores_ls)):
                    #         curr_cos_scores *= curr_cos_scores_ls[idx]
                    #     curr_scores += curr_cos_scores
                    for curr_query_embedding in curr_query_embedding_ls:
                        curr_scores = 1
                        if len(sub_corpus_embeddings.shape) == 2 and sub_corpus_embeddings.shape[0] > 1:
                            curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
                        else:    
                            curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device))
                        for conj_id in range(len(curr_scores_ls)):
                            curr_scores *= curr_scores_ls[conj_id]
                        # full_curr_scores += curr_scores
                        full_curr_scores_ls.append(curr_scores)
                    
                    curr_scores = torch.tensor(full_curr_scores_ls)
                    
                else:
                    # curr_cos_scores = self.score_functions[score_function](curr_query_embedding.unsqueeze(0), sub_corpus_embeddings)
                    curr_cos_scores = self.score_functions[score_function](curr_query_embedding_ls, sub_corpus_embeddings)
                    curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
                    curr_scores = curr_cos_scores.squeeze(0)
                    if len(curr_scores) > 1:
                        curr_scores = torch.max(curr_scores, dim=-1)[0]
                    curr_scores = curr_scores.view(-1)
                cos_scores.append(curr_scores.cpu())
            cos_scores = torch.stack(cos_scores)
            return cos_scores
        
        class CountThread(threading.Thread):
            def __init__(self, param):
                threading.Thread.__init__(self)
                self.param = param
                self.result = None

            def run(self):
                self.result = compute_cos_scores(self.param)
        
        # Create a ThreadPoolExecutor with a maximum of 4 worker threads
        # with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        
        # all_cos_scores = Parallel(n_jobs=16)(delayed(compute_cos_scores)(item) for item in all_sub_corpus_embedding_ls)
        
        all_cos_scores = []
        # all_sub_corpus_embedding_ls = [item.to(device) for item in all_sub_corpus_embedding_ls]
        for idx in range(0, len(all_sub_corpus_embedding_ls), batch_size):
            all_sub_corpus_embedding_tuple_ls = all_sub_corpus_embedding_ls[idx:idx+batch_size]
            threads = []
            for sub_idx in range(len(all_sub_corpus_embedding_tuple_ls)):    
                thread = CountThread(all_sub_corpus_embedding_tuple_ls[sub_idx])
                # thread = threading.Thread(target=compute_cos_scores, args=(all_sub_corpus_embedding_tuple_ls[sub_idx],))
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
                all_cos_scores.append(thread.result)
        # with multiprocessing.Pool() as pool:
        #     # Submit the tasks to the executor
        #     # The map function will apply the process_item function to each item in parallel
        #     all_cos_scores = list(pool.map(compute_cos_scores, all_sub_corpus_embedding_ls))
        # all_cos_scores.append(cos_scores)
        
        all_cos_scores_tensor = torch.stack(all_cos_scores, dim=-1)
        all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
        all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
        # all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
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
                self.results[query_id][corpus_id] = score
        
        return self.results, all_sub_corpus_embedding_ls

    def search2(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict, 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False, 
               query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10,
               **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        if queries is not None:
            query_ids = list(queries.keys())
        else:
            if query_embeddings is not None:
                query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
            else:
                raise ValueError("Either queries or query_embeddings must be provided!")
        self.results = {qid: {} for qid in query_ids}
        if queries is not None:
            queries = [queries[qid] for qid in query_ids]
        if query_negations is not None:
            query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
        if query_embeddings is None:
            query_embeddings=[]
            for idx in range(len(queries)):
                curr_query = queries[idx]
                if type(curr_query) is str:
                    curr_query_embedding_ls = self.model.encode_queries(
                        curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
                elif type(curr_query) is list:
                    curr_query_embedding_ls = []
                    for k in range(len(curr_query)):
                        curr_conjunct = []
                        for j in range(len(curr_query[k])):
                            qe = self.model.encode_queries(
                                curr_query[k][j], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
                            curr_conjunct.append(qe)
                        curr_query_embedding_ls.append(curr_conjunct)
                query_embeddings.append(curr_query_embedding_ls)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        if corpus is not None:
            corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
            corpus = [corpus[cid] for cid in corpus_ids]
        else:
            if all_sub_corpus_embedding_ls is not None:
                corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
            else:
                raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        if query_count < 0:
            query_count = len(query_embeddings)

        all_cos_scores = []
        if all_sub_corpus_embedding_ls is None:
            itr = range(0, len(corpus), self.corpus_chunk_size)
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
        
        for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):

            #Compute similarites using either cosine-similarity or dot product
            cos_scores = []
            # for query_itr in range(len(query_embeddings)):
            for query_itr in range(query_count):
                curr_query_embedding_ls = query_embeddings[query_itr]
                if type(curr_query_embedding_ls) is list:
                    full_curr_scores_ls = []
                    # for conj_id in range(len(curr_query_embedding)):
                    #     curr_cos_scores_ls = self.score_functions[score_function](torch.stack(curr_query_embedding[conj_id]), sub_corpus_embeddings)
                    #     if query_negations is not None and query_negations[query_itr] is not None:
                    #         curr_query_negations = torch.tensor(query_negations[query_itr])
                    #         curr_cos_scores_ls[curr_query_negations == 1] =  - curr_cos_scores_ls[curr_query_negations == 1]

                    #     curr_cos_scores_ls[torch.isnan(curr_cos_scores_ls)] = -1

                    #     curr_cos_scores = 1
                    #     for idx in range(len(curr_cos_scores_ls)):
                    #         curr_cos_scores *= curr_cos_scores_ls[idx]
                    #     curr_scores += curr_cos_scores
                    for curr_query_embedding in curr_query_embedding_ls:
                        curr_scores = 1
                        if len(sub_corpus_embeddings.shape) == 2 and sub_corpus_embeddings.shape[0] > 1:
                            curr_scores_ls = torch.max(self.score_functions[score_function](torch.cat(curr_query_embedding), sub_corpus_embeddings), dim=-1)[0]
                        else:    
                            curr_scores_ls = self.score_functions[score_function](torch.cat(curr_query_embedding), sub_corpus_embeddings)
                        for conj_id in range(len(curr_scores_ls)):
                            curr_scores *= curr_scores_ls[conj_id]
                        # full_curr_scores += curr_scores
                        full_curr_scores_ls.append(curr_scores)
                    
                    curr_scores = torch.tensor(full_curr_scores_ls)
                    
                else:
                    # curr_cos_scores = self.score_functions[score_function](curr_query_embedding.unsqueeze(0), sub_corpus_embeddings)
                    curr_cos_scores = self.score_functions[score_function](curr_query_embedding_ls, sub_corpus_embeddings)
                    curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
                    curr_scores = curr_cos_scores.squeeze(0)
                    if len(curr_scores) > 1:
                        curr_scores = torch.max(curr_scores, dim=-1)[0]
                    curr_scores = curr_scores.view(-1)
                cos_scores.append(curr_scores)
            cos_scores = torch.stack(cos_scores)
            all_cos_scores.append(cos_scores)
        
        all_cos_scores_tensor = torch.stack(all_cos_scores, dim=-1)
        all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
        # all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
        all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
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
                self.results[query_id][corpus_id] = score
        
        return self.results, all_sub_corpus_embedding_ls


