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
import cv2
#Parent class for any dense model

one="one"
two="two"
three="three"
four="four"

def draw_bbox_on_single_image(img_file_name, bbox_ls):
    img = cv2.imread(img_file_name)
    for bbox in bbox_ls:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imwrite("img_with_bbox.jpg", img)


class DenseRetrievalExactSearch:
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, algebra_method="one", **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True #TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.algebra_method = algebra_method
        self.results = {}

    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict, 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False, 
               query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10, device = 'cuda', bboxes_ls=None, img_file_name_ls=None, bboxes_overlap_ls=None,grouped_sub_q_ids_ls=None,
               **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        # if queries is not None:
        #     query_ids = list(queries.keys())
        # else:
        assert query_embeddings is  not None
        if query_embeddings is not None:
            query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
        else:
            raise ValueError("Either queries or query_embeddings must be provided!")
        self.results = {qid: {} for qid in query_ids}
        # if queries is not None:
        #     queries = [queries[qid] for qid in query_ids]
        # if query_negations is not None:
        #     query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
        # if query_embeddings is None:
        #     query_embeddings=[]
        #     for idx in range(len(queries)):
        #         curr_query = queries[idx]
        #         if type(curr_query) is str:
        #             curr_query_embedding_ls = self.model.encode_queries(
        #                 curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
        #         elif type(curr_query) is list:
        #             curr_query_embedding_ls = []
        #             for k in range(len(curr_query)):
        #                 curr_conjunct = []
        #                 for j in range(len(curr_query[k])):
        #                     qe = self.model.encode_queries(
        #                         [curr_query[k][j]], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
        #                     curr_conjunct.append(qe)
        #                 curr_query_embedding_ls.append(torch.cat(curr_conjunct))
        #         query_embeddings.append(curr_query_embedding_ls)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        # if corpus is not None:
        #     corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        #     corpus = [corpus[cid] for cid in corpus_ids]
        # else:
        assert all_sub_corpus_embedding_ls is not None
        if all_sub_corpus_embedding_ls is not None:
            corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
        else:
            raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        if query_count < 0:
            query_count = len(query_embeddings)

        assert all_sub_corpus_embedding_ls is not None
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
                
        #         all_sub_corpus_embedding_ls.extend(sub_corpus_embeddings)
                
            # torch.save(all_sub_corpus_embedding_ls, 'output/all_sub_corpus_embedding_ls')
        
        # all_sub_corpus_embedding_ls = [item.to(device) for item in all_sub_corpus_embedding_ls]
        corpus_idx = 0
        for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):
            if corpus_idx == 40:
                print()
    
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
                    for sub_q_ls_idx in range(len(curr_query_embedding_ls)):
                        curr_query_embedding = curr_query_embedding_ls[sub_q_ls_idx]
                        curr_scores = 1
                        
                        if len(sub_corpus_embeddings.shape) == 2 and sub_corpus_embeddings.shape[0] > 1:
                            if curr_query_embedding.shape[0] == 1:
                                curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings[-1].to(device))
                            else:
                                # curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
                                if self.algebra_method == one or self.algebra_method == three:
                                    curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device))#, dim=-1)
                                elif self.algebra_method == two:
                                    curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
                                    
                                    curr_scores_ls_max_id = torch.argmax(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)
                                else:
                                    if grouped_sub_q_ids_ls[query_itr] is not None:
                                        curr_grouped_sub_q_ids_ls = grouped_sub_q_ids_ls[query_itr][sub_q_ls_idx]
                                    else:
                                        curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
                                    
                                    curr_scores_ls= 1
                                    # curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
                                    
                                    for curr_grouped_sub_q_ids in curr_grouped_sub_q_ids_ls:
                                    
                                        selected_embedding_idx = torch.arange(sub_corpus_embeddings.shape[0])
                                        beam_search_topk=min(20, sub_corpus_embeddings.shape[0])
                                        sub_curr_scores = torch.ones(1).to(device)
                                        for sub_query_idx in range(len(curr_grouped_sub_q_ids)): #range(curr_query_embedding.shape[0]):
                                            # print(curr_grouped_sub_q_ids, sub_query_idx)
                                            prod_mat = self.score_functions[score_function](curr_query_embedding[curr_grouped_sub_q_ids[sub_query_idx]].to(device), sub_corpus_embeddings[selected_embedding_idx].to(device)).view(-1,1)*sub_curr_scores.view(1,-1)
                                            sub_curr_scores_ls, topk_ids = torch.topk(prod_mat.view(-1), k=beam_search_topk, dim=-1)
                                            topk_emb_ids = topk_ids // prod_mat.shape[1]
                                            topk_emb_ids = selected_embedding_idx.to(device)[topk_emb_ids]
                                            topk_emb_ids = list(set(topk_emb_ids.tolist()))
                                            if sub_query_idx == 0:
                                                selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                                            else:
                                                curr_selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                                                selected_embedding_idx = torch.tensor(list(set(torch.cat([selected_embedding_idx, curr_selected_embedding_idx]).tolist())))
                                            sub_curr_scores = sub_curr_scores_ls
                                        curr_scores_ls *= torch.max(sub_curr_scores)
                                    # curr_scores_ls2 = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings[0:-1].to(device)), dim=-1)[0]
                                
                        else:    
                            curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device))
                        
                        # whole_img_sim = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings[-1].to(device)).view(-1)
                        # curr_scores = torch.prod(curr_scores_ls, dim=0)
                        # for conj_id in range(len(curr_scores_ls)):
                        #     curr_scores *= curr_scores_ls[conj_id]
                        # full_curr_scores += curr_scores
                        if self.algebra_method == one:
                            curr_scores = torch.max(torch.prod(curr_scores_ls, dim=0))
                            full_curr_scores_ls.append(curr_scores.item())
                        elif self.algebra_method == three:
                            curr_scores = torch.max(torch.sum(curr_scores_ls, dim=0))
                            # curr_scores = torch.max(torch.max(curr_scores_ls, dim=0))
                            full_curr_scores_ls.append(curr_scores.item())
                        elif self.algebra_method == two:
                            
                            # if torch.sum(curr_scores_ls - whole_img_sim > 0.2) > 0:
                            #     print()
                            
                            # curr_scores_ls[curr_scores_ls2 - whole_img_sim > 0.2] = whole_img_sim[curr_scores_ls2 - whole_img_sim > 0.2]
                            # curr_scores_ls[whole_img_sim - curr_scores_ls2 > 0.2] = curr_scores_ls2[whole_img_sim - curr_scores_ls2 > 0.2]
                            
                            curr_scores = torch.prod(curr_scores_ls)
                            # curr_scores = torch.sum(curr_scores_ls)
                            # curr_scores = torch.sum(curr_scores_ls)
                            full_curr_scores_ls.append(curr_scores.item())
                        else:
                            curr_scores = curr_scores_ls
                            full_curr_scores_ls.append(curr_scores.item())
                    
                    curr_scores = torch.tensor(full_curr_scores_ls)
                    
                else:
                    # curr_cos_scores = self.score_functions[score_function](curr_query_embedding.unsqueeze(0), sub_corpus_embeddings)
                    curr_cos_scores = self.score_functions[score_function](curr_query_embedding_ls.to(device), sub_corpus_embeddings.to(device))
                    curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
                    curr_scores = curr_cos_scores.squeeze(0)
                    if len(curr_scores) > 1:
                        curr_scores = torch.max(curr_scores, dim=-1)[0]
                    curr_scores = curr_scores.view(-1)
                cos_scores.append(curr_scores)
            cos_scores = torch.stack(cos_scores)
            all_cos_scores.append(cos_scores)
            
            corpus_idx += 1
        
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

    # @profile
    def search_by_clusters(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict, 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False, 
               query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10, device = 'cuda', clustering_info=None, topk_embs = 500,
               **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        
        
        assert clustering_info is not None
        cluster_sub_X_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_cat_patch_ids_ls = clustering_info
        
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        # logger.info("Encoding Queries...")
        # if queries is not None:
        #     query_ids = list(queries.keys())
        # else:
        
        assert query_embeddings is not None
        if query_embeddings is not None:
            query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
        else:
            raise ValueError("Either queries or query_embeddings must be provided!")
        self.results = {qid: {} for qid in query_ids}
        # if queries is not None:
        #     queries = [queries[qid] for qid in query_ids]
        # if query_negations is not None:
        #     query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
        
        # if query_embeddings is None:
        #     query_embeddings=[]
        #     for idx in range(len(queries)):
        #         curr_query = queries[idx]
        #         if type(curr_query) is str:
        #             curr_query_embedding_ls = self.model.encode_queries(
        #                 curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
        #         elif type(curr_query) is list:
        #             curr_query_embedding_ls = []
        #             for k in range(len(curr_query)):
        #                 curr_conjunct = []
        #                 for j in range(len(curr_query[k])):
        #                     qe = self.model.encode_queries(
        #                         curr_query[k][j], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
        #                     curr_conjunct.append(qe)
        #                 curr_query_embedding_ls.append(curr_conjunct)
        #         query_embeddings.append(curr_query_embedding_ls)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        # if corpus is not None:
        #     corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        #     corpus = [corpus[cid] for cid in corpus_ids]
        # else:
        assert all_sub_corpus_embedding_ls is not None
        if all_sub_corpus_embedding_ls is not None:
            corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
        else:
            raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        if query_count < 0:
            query_count = len(query_embeddings)

        # all_cos_scores = []
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
        
        all_cos_scores_tensor = torch.zeros((len(all_sub_corpus_embedding_ls), len(query_embeddings[0]),  query_count), device=device)
        
        topk_embs = min(topk_embs, len(all_sub_corpus_embedding_ls))
        
        for query_itr in tqdm(range(query_count)):
            curr_query_embedding_ls = query_embeddings[query_itr]
            if type(curr_query_embedding_ls) is list:                                
                for sub_query_itr in range(len(curr_query_embedding_ls)):
                    curr_query_embedding = curr_query_embedding_ls[sub_query_itr]
                    
                    curr_scores_mat = self.score_functions[score_function](curr_query_embedding.to(device), cluster_centroid_tensor.to(device))
                    
                    # for sub_q_idx in range(curr_scores_mat.shape[0]):
                    #     curr_scores *= curr_scores_mat[sub_q_idx]
                    # 
                    if self.algebra_method == one or self.algebra_method == three:
                        if self.algebra_method == one:
                            curr_scores = torch.prod(curr_scores_mat, dim=0)
                        else:
                            curr_scores = torch.sum(curr_scores_mat, dim=0)
                        
                        
                        topk_cluster_ids = torch.argsort(curr_scores, descending=True)
                        
                        # cluster_sample_cum_count = torch.cumsum(cluster_sample_count_ls[topk_cluster_ids])
                        
                        # covered_sample_id_set = set()
                        
                        # # if self.algebra_method == two:
                        
                        # for cluster_id in topk_cluster_ids:
                        #     curr_cluster_sample_ids = cluster_unique_sample_ids_ls[cluster_id]
                            
                        #     # if query_itr == 0 and 0 in curr_cluster_sample_ids:
                        #     #     print()
                        
                        #     curr_scores_mat_curr_cluster = self.score_functions[score_function](curr_query_embedding.to(device), cluster_sub_X_ls[cluster_id].to(device))
                            
                        #     curr_scores_mat_curr_cluster = torch.prod(curr_scores_mat_curr_cluster, dim=0)
                            
                        #     all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr] = (curr_scores_mat_curr_cluster > all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr])*curr_scores_mat_curr_cluster \
                        #         + (curr_scores_mat_curr_cluster <= all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr])*all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr]
                            
                        #     covered_sample_id_set.update(curr_cluster_sample_ids.tolist())
                            
                        #     # curr_cluster_sample_ids = torch.tensor(list(set(curr_cluster_sample_ids.tolist()).difference(covered_sample_id_set)))
                        #     # if len(curr_cluster_sample_ids) > 0:
                        #     #     all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr] = curr_scores[cluster_id]
                                
                        #     #     covered_sample_id_set.update(curr_cluster_sample_ids.tolist())
                        #     if len(covered_sample_id_set) >= topk_embs:
                        #         break
                        
                        sample_to_cat_patch_idx_mappings = dict()
                        for cluster_id in topk_cluster_ids:
                            curr_cat_patch_ids_mappings = cluster_sub_X_cat_patch_ids_ls[cluster_id]
                            for sample_id in curr_cat_patch_ids_mappings:
                                if sample_id not in sample_to_cat_patch_idx_mappings:
                                    sample_to_cat_patch_idx_mappings[sample_id] = []
                                sample_to_cat_patch_idx_mappings[sample_id].extend(curr_cat_patch_ids_mappings[sample_id])
                            
                            if len(sample_to_cat_patch_idx_mappings) >= topk_embs:
                                break
                                    
                        for sample_id in sample_to_cat_patch_idx_mappings:
                                # sample_id = int(sample_id)
                            # curr_sample_sub_x = []
                            # for cluster_idx in merged_sample_to_cluster_idx_mappings[sample_id]:
                            #     curr_sample_sub_x.append(cluster_sub_X_ls[cluster_idx][sample_id])
                            
                            patch_ids = torch.tensor(list(sample_to_cat_patch_idx_mappings[sample_id]))
                            curr_sample_sub_X_tensor = all_sub_corpus_embedding_ls[sample_id][patch_ids].to(device)
                            
                            # curr_sample_sub_X_tensor2 = torch.cat(curr_sample_sub_x).to(device)
                            cos_scores = torch.max(torch.prod(self.score_functions[score_function](curr_sample_sub_X_tensor, curr_query_embedding.to(device)), dim=-1))
                            all_cos_scores_tensor[sample_id, sub_query_itr, query_itr] = cos_scores
                            
                        
                        
                    else:   
                        sorted_scores, sorted_indices = torch.sort(curr_scores_mat, dim=-1, descending=True)
                        # local_sim_array = torch.ones(len(all_sub_corpus_embedding_ls), device=device)
                        # local_visited_times_tensor = torch.zeros([len(all_sub_corpus_embedding_ls), sorted_scores.shape[0]], device=device)
                        # for cluster_id in range(sorted_indices.shape[1]):
                        #     curr_scores = sorted_scores[:,cluster_id]
                        #     for sub_q_idx in range(len(curr_scores)):
                        #         curr_cluster_idx = sorted_indices[sub_q_idx,cluster_id]
                        #         curr_cluster_sample_ids = cluster_unique_sample_ids_ls[curr_cluster_idx].to(device)
                        #         # if 0 in curr_cluster_sample_ids and query_itr == 0 and sub_query_itr == 0:
                        #         #     print()
                        #         selected_rid = (local_visited_times_tensor[curr_cluster_sample_ids,sub_q_idx] == 0)
                        #         curr_cluster_sample_ids = curr_cluster_sample_ids[selected_rid]
                        #         curr_sub_X  = cluster_sub_X_ls[curr_cluster_idx].to(device)[selected_rid]
                        #         curr_sim_score=self.score_functions[score_function](curr_query_embedding[sub_q_idx].to(device), curr_sub_X)
                                
                                
                        #         local_sim_array[curr_cluster_sample_ids] *= curr_sim_score.view(-1) #curr_scores[sub_q_idx]
                        #         local_visited_times_tensor[curr_cluster_sample_ids, sub_q_idx] = 1
                        #     if torch.sum(torch.sum(local_visited_times_tensor, dim=-1) >= len(curr_scores)) >= topk_embs:
                        #         break
                        sample_to_cat_patch_idx_ls = [dict()]*curr_query_embedding.shape[0]
                        # sample_to_cluster_idx_ls = [dict()]*curr_query_embedding.shape[0]
                        for cluster_id in range(sorted_indices.shape[1]):
                            common_sample_ids = set()
                            for sub_q_idx in range(curr_query_embedding.shape[0]):
                                curr_cluster_idx = sorted_indices[sub_q_idx,cluster_id].item()
                                # curr_cluster_sample_ids = cluster_sample_ids_ls[curr_cluster_idx]
                                # curr_sample_idx_sub_X_mappings  = cluster_sub_X_ls[curr_cluster_idx]
                                curr_cat_patch_ids_mappings = cluster_sub_X_cat_patch_ids_ls[curr_cluster_idx]
                                for sample_id in curr_cat_patch_ids_mappings:
                                    if sample_id not in sample_to_cat_patch_idx_ls[sub_q_idx]:
                                        # sample_to_cluster_idx_ls[sub_q_idx][sample_id] = []
                                        sample_to_cat_patch_idx_ls[sub_q_idx][sample_id] = []# sample_to_sub_X_mappings_ls[sub_q_idx][sample_id].append(curr_sample_idx_sub_X_mappings[sample_id])
                                    # sample_to_cluster_idx_ls[sub_q_idx][sample_id].append(curr_cluster_idx)
                                    sample_to_cat_patch_idx_ls[sub_q_idx][sample_id].extend(curr_cat_patch_ids_mappings[sample_id])
                                if sub_q_idx == 0:
                                    common_sample_ids = set(sample_to_cat_patch_idx_ls[sub_q_idx].keys())
                                else:
                                    common_sample_ids = common_sample_ids.intersection(set(sample_to_cat_patch_idx_ls[sub_q_idx].keys()))
                            if len(common_sample_ids) >= topk_embs:
                                break
                            
                        # merged_sample_to_cluster_idx_mappings = dict()
                        merged_sample_to_cat_patch_idx_mappings = dict()
                        for sample_id in common_sample_ids:
                            for sub_q_idx in range(len(sample_to_cat_patch_idx_ls)):
                                if sub_q_idx == 0:
                                    # merged_sample_to_cluster_idx_mappings[sample_id] = set(sample_to_cluster_idx_ls[sub_q_idx][sample_id])
                                    merged_sample_to_cat_patch_idx_mappings[sample_id] = set(sample_to_cat_patch_idx_ls[sub_q_idx][sample_id])
                                else:
                                    # merged_sample_to_cluster_idx_mappings[sample_id] = merged_sample_to_cluster_idx_mappings[sample_id].union(set(sample_to_cluster_idx_ls[sub_q_idx][sample_id]))
                                    merged_sample_to_cat_patch_idx_mappings[sample_id] = merged_sample_to_cat_patch_idx_mappings[sample_id].union(set(sample_to_cat_patch_idx_ls[sub_q_idx][sample_id]))
                        
                        for sample_id in common_sample_ids:
                            # sample_id = int(sample_id)
                            # curr_sample_sub_x = []
                            # for cluster_idx in merged_sample_to_cluster_idx_mappings[sample_id]:
                            #     curr_sample_sub_x.append(cluster_sub_X_ls[cluster_idx][sample_id])
                            
                            patch_ids = torch.tensor(list(merged_sample_to_cat_patch_idx_mappings[sample_id]))
                            curr_sample_sub_X_tensor = all_sub_corpus_embedding_ls[sample_id][patch_ids].to(device)
                            
                            # curr_sample_sub_X_tensor2 = torch.cat(curr_sample_sub_x).to(device)
                            cos_scores = torch.prod(torch.max(self.score_functions[score_function](curr_sample_sub_X_tensor, curr_query_embedding.to(device)), dim=0)[0])
                            all_cos_scores_tensor[sample_id, sub_query_itr, query_itr] = cos_scores
                            
                            # for sample_id in common_sample_ids:
                            #     # for sub_q_idx in range(sample_to_sub_X_mappings_ls):
                            #         cos_scores = torch.prod(torch.max(self.score_functions(torch.cat(sample_to_cluster_idx_ls[0][sample_id]), curr_query_embedding.to(device)), dim=0))
                            #         all_cos_scores_tensor[sample_id, sub_q_idx, query_itr] = cos_scores
                                
                        # all_cos_scores_tensor[torch.sum(local_visited_times_tensor, dim=-1) >= len(curr_scores), sub_query_itr, query_itr] = local_sim_array

        all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=0, keepdim=True)
        all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
        # all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
        #Get top-k values
        all_cos_scores_tensor = all_cos_scores_tensor.t()
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


