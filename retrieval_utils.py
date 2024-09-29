import logging
import pathlib, os
import os, sys

# import openai
from openai import OpenAI
import json
from tqdm import tqdm
import torch

import re
from vector_dataset import Partitioned_vector_dataset, collate_fn
import time
import numpy as np
from scipy.stats import percentileofscore
from collections import defaultdict

# openai.api_key = os.getenv("OPENAI_API_KEY")
import pandas as pd
from scipy.stats import percentileofscore
from collections import defaultdict

import json

def store_json_results(results, output_name):
    with open(output_name, "w") as f:
        json.dump(results, f, indent=4)

def obtain_key_words(query):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    
    response = client.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "Say this is a test",
    #         }
    #     ],
    #     model="gpt-3.5",
    # )
    
    # response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Can you show me the keywords of the following sentence? Please return the list of keywords separated with commas. \"" + query + "\"",
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    kw_ls = response.choices[0].text.split(",")
    res_kw_ls = []
    for kw in kw_ls:
        res_kw_ls.append(kw.strip())
    
    return res_kw_ls

def intersect_res(results):
    intersect_keys = set(results[list(results.keys())[0]].keys())
    for idx in range(len(list(results.keys()))-1):
        intersect_keys = intersect_keys.intersection(set(results[list(results.keys())[idx + 1]].keys()))
        
    intersect_res = {}
    for key in intersect_keys:
        score = 1
        for idx in range(len(list(results.keys()))):
            score = score*results[list(results.keys())[idx]][key]/100
        intersect_res[key] = score*100
        
    return intersect_res

def dump_decomposed_queries(dq_file_name, dataset_name, decomposed_queries):
    # output_file_name = os.path.join(out_dir, dataset_name + "_dq.json")
    with open(dq_file_name, "w") as f:
        json.dump(decomposed_queries, f, indent=4)

def evaluate_for_query_batches(retriever, qrels, results):
    # all_results = dict()    
    # for key in tqdm(results):
    #     ndcg, _map, recall, precision = retriever.evaluate({key: qrels[key]}, {key: results[key]}, retriever.k_values)
    #     all_results[key] = dict()
    #     all_results[key].update(ndcg)
    #     all_results[key].update(_map)
    #     all_results[key].update(recall)
    #     all_results[key].update(precision)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    # return all_results

def retrieve_with_decomposition(retriever, corpus, queries, qrels, out_dir, dataset_name, all_sub_corpus_embedding_ls=None):
    print("results with decomposition::")
    decomposed_queries = dict()
    
    dq_file_name = os.path.join(out_dir, dataset_name + "_dq.json")
    if not os.path.exists(dq_file_name):
        print("start decompose queries")
        for key in queries:
            curr_query = queries[key]
            decomposed_q = obtain_key_words(curr_query)
            decomposed_queries[key] = decomposed_q
        dump_decomposed_queries(dq_file_name, dataset_name, decomposed_queries)
        print("end decompose queries")
    else:
        with open(dq_file_name, "r") as f:
            decomposed_queries = json.load(f)

    results, _ = retriever.retrieve(corpus, decomposed_queries, query_count = len(queries), all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    print("start evaluating performance for single query with decomposition")
    # return evaluate_for_query_batches(retriever, qrels, results), decomposed_queries


def retrieve_with_dessert(all_sub_corpus_embedding_ls, query_embeddings, doc_retrieval, prob_agg, method,dataset_name, **kwargs):
    top_k=min(kwargs['clustering_topk'],len(all_sub_corpus_embedding_ls))
    #print(top_k)
    #print(type(top_k))
    num_to_rerank=min(kwargs['clustering_topk'],len(all_sub_corpus_embedding_ls))
    #print(num_to_rerank)
    #print(type(num_to_rerank))
    is_img_retrieval=kwargs['is_img_retrieval']
    #print(is_img_retrieval)
    #print(type(is_img_retrieval))
    dependency_topk=kwargs['dependency_topk']
    #print(dependency_topk)
    #print(type(dependency_topk))
    grouped_sub_q_ids_ls=kwargs['grouped_sub_q_ids_ls']
    #print(grouped_sub_q_ids_ls)
    #print(type(grouped_sub_q_ids_ls))
    bboxes_overlap_ls=kwargs['bboxes_overlap_ls']
    #print(bboxes_overlap_ls)
    #print(type(bboxes_overlap_ls))
    
    all_cos_scores = []
    query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
    corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
    all_results = {qid: {} for qid in query_ids}
    query_count = len(query_embeddings)
    
    all_cos_scores_tensor = torch.zeros(query_count, len(query_embeddings[0]), len(corpus_ids))
    all_cos_scores_tensor[:] = 1e-6
    expected_idx_ls = []
    for idx in tqdm(range(query_count)): #, desc="Querying":
        cos_scores_ls=[]
        for sub_idx in range(len(query_embeddings[idx])):
            #print('ENTERED')
            #print(grouped_sub_q_ids_ls)
            embeddings_numpy = (query_embeddings[idx][sub_idx]).detach().cpu().numpy().astype(np.float32)
            #print(embeddings_numpy.shape)
            #print(type(embeddings_numpy))
            #results = doc_retrieval.query(embeddings_numpy, top_k, num_to_rerank, prob_agg, True)
            if (method == "two"):
                #5th input is use_frequency, just hard set to True
                results = doc_retrieval.query(embeddings_numpy, top_k, num_to_rerank, prob_agg, True, is_img_retrieval)
            else:
                #5th input is use_frequency, just hard set to True
                results = doc_retrieval.querywithdependency(embeddings_numpy, top_k, num_to_rerank, prob_agg, True, is_img_retrieval, dependency_topk, idx, sub_idx, grouped_sub_q_ids_ls, bboxes_overlap_ls)
            
            #time.sleep(5)
            cos_scores_tensor = torch.tensor(results[0]).cpu()
            sample_ids_tensor = torch.tensor(results[1]).cpu()

            #print(f"Cos Scores Tensor: {cos_scores_tensor}, type: {type(cos_scores_tensor)}, shape: {cos_scores_tensor.shape}")
            #print(f"Sample IDs Tensor: {sample_ids_tensor}, type: {type(sample_ids_tensor)}, shape: {sample_ids_tensor.shape}")

            all_cos_scores_tensor[idx, sub_idx, sample_ids_tensor] = cos_scores_tensor
            #print('done with all cos scores')

    all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
    if prob_agg == "prod":
        all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
    else:
        if dataset_name == "trec-covid":
            all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
        else:
            all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
    print(all_cos_scores_tensor.shape)
    #Get top-k values
    cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(all_cos_scores_tensor, min(top_k+1, len(all_cos_scores_tensor[0])), dim=1, largest=True)
    cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
    cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
    
    for query_itr in range(query_count):
        query_id = query_ids[query_itr]                  
        for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
            corpus_id = corpus_ids[sub_corpus_id]
            all_results[query_id][corpus_id] = score
    results = all_results
    
    # print("It works!")
    # print(results)
    return results


def retrieve_by_embeddings0(retriever, all_sub_corpus_embedding_ls, query_embeddings, qrels, query_count = 10, parallel=False, clustering_topk=500, batch_size=16,in_disk=False,doc_retrieval=None,use_clustering=False,prob_agg="prod",method="two",_nprobe_query=2, index_method="default",dataset_name="", **kwargs):
    print("results with decomposition::")
    # if parallel:
    #     all_sub_corpus_embedding_dataset= Partitioned_vector_dataset(all_sub_corpus_embedding_ls)
    #     all_sub_corpus_embedding_loader = torch.utils.data.DataLoader(all_sub_corpus_embedding_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    #     results,_ = retriever.retrieve(None, None, query_embeddings=query_embeddings, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_loader, query_count=query_count, parallel=parallel)
    # else:
    if type(all_sub_corpus_embedding_ls) is list:
        all_sub_corpus_embedding_ls = [torch.nn.functional.normalize(all_sub_corpus_embedding, p=2, dim=-1) for all_sub_corpus_embedding in all_sub_corpus_embedding_ls]
    else:
        all_sub_corpus_embedding_ls = torch.nn.functional.normalize(all_sub_corpus_embedding_ls, p=2, dim=-1)
    
    
    t1 = time.time()
    if type(query_embeddings[0]) is list:
        query_embeddings = [[torch.nn.functional.normalize(query_embedding, p=2, dim=-1) for query_embedding in local_query_embedding] for local_query_embedding in query_embeddings]
    else:
        query_embeddings = [torch.nn.functional.normalize(query_embedding, p=2, dim=-1) for query_embedding in query_embeddings]
    
    kwargs["dataset_name"] = dataset_name
    if not use_clustering:
        
        # results,_ = retriever.retrieve(None, None, query_embeddings=query_embeddings, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls, query_count=query_count, parallel=parallel, in_disk=in_disk)
        results,_ = retriever.retrieve(None, None, query_embeddings=query_embeddings, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls, query_count=query_count, parallel=parallel, in_disk=in_disk, **kwargs)
    else:
        if not index_method == "dessert":
            kwargs['index_method'] = index_method
            doc_retrieval._nprobe_query = _nprobe_query #max(2, int(clustering_topk/20))
            results = doc_retrieval.query_multi_queries(all_sub_corpus_embedding_ls, query_embeddings, top_k=min(clustering_topk,len(all_sub_corpus_embedding_ls)), num_to_rerank=min(clustering_topk,len(all_sub_corpus_embedding_ls)), prob_agg=prob_agg,method=method, **kwargs)
        else:
            kwargs['clustering_topk'] = clustering_topk
            results = retrieve_with_dessert(all_sub_corpus_embedding_ls, query_embeddings, doc_retrieval, prob_agg, method, **kwargs)
        # results = {str(idx+1): results[idx] for idx in range(len(results))}
    t2 = time.time()
    
    print(f"Time taken: {t2-t1:.2f}s")
    print("These are the queries:")
    print(queries)
    #Option 1: count of words
    if perc_method == "one":
        print("Option 1: count of words")
        query_lengths = [len(query.split()) for query in queries]
    #Option 2: length of sentence
    if perc_method == "two":
        print("Option 2: length of sentence")
        query_lengths = [len(query) for query in queries]
    #Option 3: number of subqueries
    #As an example, subquery is ['a car parked on a street', 'a street next to a tree', 'a street with a parking meter', 'a street with a lamp post', 'bikes near the tree.'], ['a car parked on a street next to a tree. the street has a parking meter and a lamp post. there are bikes near the tree.']]
    elif perc_method == "three":
        print("Option 3: number of subqueries")
        query_lengths = [len(subquery[0]) for subquery in full_sub_queries_ls]
    print("This is query_lengths:")
    print(query_lengths)
    
    #Calculate the percentile rank for each query length
    percentile_ranks = [min(percentileofscore(query_lengths, length), 100) for length in query_lengths]
    
    # Assign each query to a bucket
    query_buckets = []
    for rank in percentile_ranks:
        #print(rank)
        if rank >= 99:
            query_buckets.append(9.9) # 99-100
        elif rank >= 95:
            query_buckets.append(9.5) # 95-99
        elif rank >= 90:
            query_buckets.append(9) # 90-95
        else:
            query_buckets.append(int(rank // 10))  # 0-10, 10-20, ..., 80-90

    print("These are the percentiles:")
    print(query_buckets)

    # Grouping queries by their percentile bucket
    bucket_groups = defaultdict(list)
    for i, bucket in enumerate(query_buckets):
        if bucket < 9:
            bucket_groups[bucket].append(i)
        if bucket == 9: #90-100th percentile
            bucket_groups[9].append(i)
        if bucket == 9.5: #95-100th percentile
            bucket_groups[9].append(i)
            bucket_groups[9.5].append(i)
        if bucket == 9.9: #99-100th percenntile
            bucket_groups[9].append(i)
            bucket_groups[9.5].append(i)
            bucket_groups[9.9].append(i)
    print(bucket_groups)
    
    # Evaluate recall for each bucket
    bucket_recall = {}
    sorted_buckets = sorted(bucket_groups.keys())

    for bucket in sorted_buckets:
        indices = bucket_groups[bucket]
        filtered_queries = [queries[i] for i in indices]
        filtered_qrels = {str(i + 1): qrels[str(i + 1)] for i in indices}
        filtered_results = {str(i + 1): results[str(i + 1)] for i in indices}
        
        if bucket == 9:
            range_name = "90-100"
        elif bucket == 9.5:
            range_name = "95-100"
        elif bucket == 9.9:
            range_name = "99-100"
        else:
            range_name = f"{bucket * 10}-{bucket * 10 + 10}"

        print(f"Percentile range {range_name}:")
        print(f"{len(indices)} queries in this percentile range")
        
        ndcg, _map, recall, precision = retriever.evaluate(filtered_qrels, filtered_results, retriever.k_values, ignore_identical_ids=False)
        bucket_recall[range_name] = recall

    print("Overall recall by buckets:")
    for range_name, recall in bucket_recall.items():
        print(f"{range_name}: {recall}")
    
    print("Overall scores:")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values, ignore_identical_ids=False)
    
    return results

# retriever, all_sub_corpus_embedding_ls, query_embeddings, qrels, query_count = 10, parallel=False, clustering_topk=500, batch_size=16,in_disk=False,doc_retrieval=None,use_clustering=False,prob_agg="prod",method="two",_nprobe_query=2, index_method="default",dataset_name="", **kwargs
def retrieve_by_embeddings(perc_method, full_sub_queries_ls, queries, retriever, all_sub_corpus_embedding_ls, query_embeddings, qrels, query_count = 10, parallel=False, clustering_topk=500, batch_size=16,in_disk=False,doc_retrieval=None,use_clustering=False,prob_agg="prod",method="two",_nprobe_query=2, index_method="default",dataset_name="",avg_ratio=0.1, **kwargs):
    print("results with decomposition::")
    # if parallel:
    #     all_sub_corpus_embedding_dataset= Partitioned_vector_dataset(all_sub_corpus_embedding_ls)
    #     all_sub_corpus_embedding_loader = torch.utils.data.DataLoader(all_sub_corpus_embedding_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    #     results,_ = retriever.retrieve(None, None, query_embeddings=query_embeddings, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_loader, query_count=query_count, parallel=parallel)
    # else:
    if type(all_sub_corpus_embedding_ls) is list:
        all_sub_corpus_embedding_ls = [torch.nn.functional.normalize(all_sub_corpus_embedding, p=2, dim=-1) for all_sub_corpus_embedding in all_sub_corpus_embedding_ls]
    else:
        all_sub_corpus_embedding_ls = torch.nn.functional.normalize(all_sub_corpus_embedding_ls, p=2, dim=-1)
    
    
    t1 = time.time()
    if not method == "five":
        if type(query_embeddings[0]) is list:
            query_embeddings = [[torch.nn.functional.normalize(query_embedding, p=2, dim=-1) for query_embedding in local_query_embedding] for local_query_embedding in query_embeddings]
        else:
            query_embeddings = [torch.nn.functional.normalize(query_embedding, p=2, dim=-1) for query_embedding in query_embeddings]
    
    kwargs["dataset_name"] = dataset_name
    if not use_clustering:
        
        # results,_ = retriever.retrieve(None, None, query_embeddings=query_embeddings, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls, query_count=query_count, parallel=parallel, in_disk=in_disk)
        results,_ = retriever.retrieve(None, None, query_embeddings=query_embeddings, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls, query_count=query_count, parallel=parallel, in_disk=in_disk, **kwargs)
    else:
        if not index_method == "dessert":
            kwargs['index_method'] = index_method
            doc_retrieval._nprobe_query = _nprobe_query #max(2, int(clustering_topk/20))
            # results = doc_retrieval.query_multi_queries(all_sub_corpus_embedding_ls, query_embeddings, top_k=min(clustering_topk,len(all_sub_corpus_embedding_ls)), num_to_rerank=min(clustering_topk,len(all_sub_corpus_embedding_ls)), prob_agg=prob_agg,method=method, **kwargs)
            results = doc_retrieval.query_multi_queries(all_sub_corpus_embedding_ls, query_embeddings, top_k=min(clustering_topk,len(all_sub_corpus_embedding_ls)), num_to_rerank=min(clustering_topk,len(all_sub_corpus_embedding_ls)), prob_agg=prob_agg,method=method, avg_ratio=avg_ratio, **kwargs)
        else:
            kwargs['clustering_topk'] = clustering_topk
            results = retrieve_with_dessert(all_sub_corpus_embedding_ls, query_embeddings, doc_retrieval, prob_agg, method, **kwargs)
        # results = {str(idx+1): results[idx] for idx in range(len(results))}
    t2 = time.time()
    
    print(f"Time taken: {t2-t1:.2f}s")
    print("These are the queries:")
    print(queries)
    print("These are the sub-queries:")
    print(full_sub_queries_ls)
    #Option 1: count of words
    if perc_method == "one":
        print("Option 1: count of words")
        query_lengths = [len(query.split()) for query in queries]
    #Option 2: length of sentence
    if perc_method == "two":
        print("Option 2: length of sentence")
        query_lengths = [len(query) for query in queries]
    #Option 3: number of subqueries
    #As an example, subquery is ['a car parked on a street', 'a street next to a tree', 'a street with a parking meter', 'a street with a lamp post', 'bikes near the tree.'], ['a car parked on a street next to a tree. the street has a parking meter and a lamp post. there are bikes near the tree.']]
    elif perc_method == "three":
        print("Option 3: number of subqueries")
        query_lengths = [len(subquery[0]) for subquery in full_sub_queries_ls]
    print("This is query_lengths:")
    print(query_lengths)
    
    #Calculate the percentile rank for each query length
    percentile_ranks = [min(percentileofscore(query_lengths, length), 100) for length in query_lengths]
    
    # Assign each query to a bucket
    query_buckets = []
    for rank in percentile_ranks:
        #print(rank)
        if rank >= 99:
            query_buckets.append(9.9) # 99-100
        elif rank >= 95:
            query_buckets.append(9.5) # 95-99
        elif rank >= 90:
            query_buckets.append(9) # 90-95
        else:
            query_buckets.append(int(rank // 10))  # 0-10, 10-20, ..., 80-90

    print("These are the percentiles:")
    print(query_buckets)

    # Grouping queries by their percentile bucket
    bucket_groups = defaultdict(list)
    for i, bucket in enumerate(query_buckets):
        if bucket < 9:
            bucket_groups[bucket].append(i)
        if bucket == 9: #90-100th percentile
            bucket_groups[9].append(i)
        if bucket == 9.5: #95-100th percentile
            bucket_groups[9].append(i)
            bucket_groups[9.5].append(i)
        if bucket == 9.9: #99-100th percenntile
            bucket_groups[9].append(i)
            bucket_groups[9.5].append(i)
            bucket_groups[9.9].append(i)
    print(bucket_groups)
    
    # Evaluate recall for each bucket
    bucket_recall = {}
    sorted_buckets = sorted(bucket_groups.keys())

    for bucket in sorted_buckets:
        indices = bucket_groups[bucket]
        filtered_queries = [queries[i] for i in indices]
        filtered_qrels = {str(i + 1): qrels[str(i + 1)] for i in indices}
        filtered_results = {str(i + 1): results[str(i + 1)] for i in indices}
        
        if bucket == 9:
            range_name = "90-100"
        elif bucket == 9.5:
            range_name = "95-100"
        elif bucket == 9.9:
            range_name = "99-100"
        else:
            range_name = f"{bucket * 10}-{bucket * 10 + 10}"

        print(f"Percentile range {range_name}:")
        print(f"{len(indices)} queries in this percentile range")
        
        ndcg, _map, recall, precision = retriever.evaluate(filtered_qrels, filtered_results, retriever.k_values, ignore_identical_ids=False)
        bucket_recall[range_name] = recall

    print("Overall recall by buckets:")
    for range_name, recall in bucket_recall.items():
        print(f"{range_name}: {recall}")
    
    print("Overall scores:")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values, ignore_identical_ids=False)
    
    if len(results) > 1:
    
        for key in tqdm(results):
            ndcg, _map, recall, precision = retriever.evaluate({key: qrels[key]}, {key:results[key]}, retriever.k_values, ignore_identical_ids=False, need_logging=False)
            store_json_results(ndcg, os.path.join("output/", f"{dataset_name}_{method}_{key}_ndcg.json"))
            store_json_results(_map, os.path.join("output/", f"{dataset_name}_{method}_{key}_map.json"))
            store_json_results(recall, os.path.join("output/", f"{dataset_name}_{method}_{key}_recall.json"))
            store_json_results(precision, os.path.join("output/", f"{dataset_name}_{method}_{key}_precision.json"))
    
    return results


def decompose_queries_by_keyword(dataset_name, queries, out_dir="out/"):
    decomposed_queries = list()
    os.makedirs(out_dir, exist_ok=True)
    dq_file_name = os.path.join(out_dir, dataset_name + "_dq.json")
    if not os.path.exists(dq_file_name):
        print("start decompose queries")
        for key in tqdm(range(len(queries))):
            curr_query = queries[key]
            decomposed_q = obtain_key_words(curr_query)
            decomposed_queries.append(decomposed_q)
        dump_decomposed_queries(dq_file_name, dataset_name, decomposed_queries)
        print("end decompose queries")
    else:
        with open(dq_file_name, "r") as f:
            decomposed_queries = json.load(f)

    return decomposed_queries

def decompose_single_query(curr_query, reg_pattern = "[,.]"):
    decomposed_q = re.split(reg_pattern, curr_query)
    decomposed_q = [dq.strip() for dq in decomposed_q if len(dq.strip()) > 0]
    return decomposed_q

def decompose_single_query_ls(curr_query_ls):
    curr_query_ls = decompose_single_query(curr_query_ls, reg_pattern="#")
    all_decomposed_q_ls = []
    for query in curr_query_ls:
        sub_query_decomposed_ls = decompose_single_query(query, reg_pattern="\|")
        all_decomposed_q_ls.append(sub_query_decomposed_ls)
    # decomposed_q = re.split(reg_pattern, curr_query)
    # decomposed_q = [dq.strip() for dq in decomposed_q if len(dq.strip()) > 0]
    return all_decomposed_q_ls

def decompose_single_query_parition_groups(all_decomposed_q_ls, curr_group_ls, group_pattern="#", sub_group_pattern="\|"):
    if pd.isnull(curr_group_ls): # len(curr_group_ls) <= 0:
        return None
        # all_grouped_ids_ls = []
        # for idx in range(len(all_decomposed_q_ls)):
        #     decomposed_q_ls = all_decomposed_q_ls[idx]
        #     grouped_ids = [list(range(len(decomposed_q_ls)))]
        #     all_grouped_ids_ls.append(all_grouped_ids_ls)
        # return all_grouped_ids_ls
    
    curr_group_ls = decompose_single_query(curr_group_ls, reg_pattern=group_pattern)
    
    assert len(curr_group_ls) == len(all_decomposed_q_ls)
    
    all_grouped_ids_ls = []
    
    for curr_group_str in curr_group_ls:
        sub_group_str_decomposed_ls = decompose_single_query(curr_group_str, reg_pattern=sub_group_pattern)
        
        curr_grouped_ids_ls = []
        for sub_group_str_decomposed in sub_group_str_decomposed_ls:
            grouped_ids = decompose_single_query(sub_group_str_decomposed)
            grouped_ids = [int(gid) for gid in grouped_ids]
            curr_grouped_ids_ls.append(grouped_ids)
            
        all_grouped_ids_ls.append(curr_grouped_ids_ls)
        
    return all_grouped_ids_ls

def decompose_queries_by_clauses(queries):
    decomposed_queries = list()
    # reg_pattern = ",|."
    reg_pattern = "[,.]"
    
    for key in tqdm(range(len(queries))):
        curr_query = queries[key]
        # decomposed_q = curr_query.split(reg_pattern)
        decomposed_q = decompose_single_query(curr_query, reg_pattern)
        decomposed_queries.append(decomposed_q)
    return decomposed_queries
