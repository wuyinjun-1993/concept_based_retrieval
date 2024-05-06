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
# openai.api_key = os.getenv("OPENAI_API_KEY")

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

def retrieve_with_decomposition(retriever, corpus, queries, qrels, out_dir, dataset_name):
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

    results = retriever.retrieve(corpus, decomposed_queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    print("start evaluating performance for single query with decomposition")
    return evaluate_for_query_batches(retriever, qrels, results), decomposed_queries


def retrieve_by_embeddings(retriever, corpus, all_sub_corpus_embedding_ls, query_embeddings, qrels, query_count = 10, parallel=False, batch_size=16,in_disk=False, **kwargs):
    print("results with decomposition::")
    # if parallel:
    #     all_sub_corpus_embedding_dataset= Partitioned_vector_dataset(all_sub_corpus_embedding_ls)
    #     all_sub_corpus_embedding_loader = torch.utils.data.DataLoader(all_sub_corpus_embedding_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    #     results,_ = retriever.retrieve(None, None, query_embeddings=query_embeddings, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_loader, query_count=query_count, parallel=parallel)
    # else:
    results,_ = retriever.retrieve(corpus, None, query_embeddings=query_embeddings, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls, query_count=query_count, parallel=parallel, in_disk=in_disk, **kwargs)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values, ignore_identical_ids=False)
    # print("start evaluating performance for single query with decomposition")
    # return evaluate_for_query_batches(retriever, qrels, results)


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
