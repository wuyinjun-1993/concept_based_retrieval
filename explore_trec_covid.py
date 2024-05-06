from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import CLIPModel, AutoProcessor

from retrieval_utils import *
import logging
import pathlib, os
import os, sys

import openai
import json
from tqdm import tqdm
import torch
from text_utils import subset_corpus

# openai.api_key = os.getenv("OPENAI_API_KEY")

# def obtain_key_words(query):
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt="Can you show me the keywords of the following sentence? Please return the list of keywords separated with commas. \"" + query + "\"",
#         temperature=1,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#         )
#     kw_ls = response["choices"][0]["text"].split(",")
#     res_kw_ls = []
#     for kw in kw_ls:
#         res_kw_ls.append(kw.strip())
    
#     return res_kw_ls

# def intersect_res(results):
#     intersect_keys = set(results[list(results.keys())[0]].keys())
#     for idx in range(len(list(results.keys()))-1):
#         intersect_keys = intersect_keys.intersection(set(results[list(results.keys())[idx + 1]].keys()))
        
#     intersect_res = {}
#     for key in intersect_keys:
#         score = 1
#         for idx in range(len(list(results.keys()))):
#             score = score*results[list(results.keys())[idx]][key]/100
#         intersect_res[key] = score*100
        
#     return intersect_res


def compare_and_not_query(retriever, corpus, queries, qrels):
        
    results = retriever.retrieve(corpus, {"8": queries["8"]})

    print("length of the results::", len(results))

    print("results without decomposition::")

    ndcg, _map, recall, precision = retriever.evaluate({"8": qrels["8"]}, results, retriever.k_values)

    new_results = retriever.retrieve(corpus, {"8": ["lack of testing availability", "underreporting of true incidence of Covid-19"]})

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    

    new_merged_resuts = intersect_res(new_results)

    print("length of the merged results::", len(new_merged_resuts))

    print("results with decomposition::")

    new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({"8": qrels["8"]}, {"8": new_merged_resuts}, retriever.k_values)
    
    new_results2 = retriever.retrieve(corpus, {"8": ["testing availability", "underreporting of true incidence of Covid-19"]}, query_negations={"8":[True,False]})

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    

    new_merged_resuts2 = intersect_res(new_results2)

    print("length of the merged results::", len(new_merged_resuts2))

    print("results with decomposition::")

    new_ndcg2, new_map2, new_recall2, new_precision2 = retriever.evaluate({"8": qrels["8"]}, {"8": new_merged_resuts2}, retriever.k_values)
    
     
def compare_query_weather_example(retriever, corpus, queries, qrels):
    # retriever.top_k = 2500
    
    results = retriever.retrieve(corpus, {"2": queries["2"], "3": queries["2"]})

    print("length of the results::", len(results))

    print("results without decomposition::")

    ndcg, _map, recall, precision = retriever.evaluate({"2": qrels["2"], "3":qrels["2"]}, results, retriever.k_values)

    # retriever.top_k = 1000

    new_results = retriever.retrieve(corpus, {"2": ["change of weather", "coronaviruses"]})

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    

    new_merged_resuts = intersect_res(new_results)

    print("length of the merged results::", len(new_merged_resuts))

    print("results with decomposition::")

    new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({"2": qrels["2"], "3":qrels["2"]}, {"2": new_merged_resuts, "3":new_merged_resuts}, retriever.k_values)
    
    
    new_results2 = retriever.retrieve(corpus, {"2": ["weather", "coronaviruses"]})    
    
    new_merged_resuts2 = intersect_res(new_results2)

    print("length of the merged results::", len(new_merged_resuts2))

    print("results with decomposition 2::")

    new_ndcg2, new_map2, new_recall2, new_precision2 = retriever.evaluate({"2": qrels["2"], "3":qrels["2"]}, {"2": new_merged_resuts2, "3":new_merged_resuts2}, retriever.k_values)
    
def compare_query_social_distance(retriever, corpus, queries, qrels):
    # retriever.top_k = 2500
    
    results = retriever.retrieve(corpus, {"10": queries["10"]})

    print(queries["10"])

    print("length of the results::", len(results))

    print("results without decomposition::")

    ndcg, _map, recall, precision = retriever.evaluate({"10": qrels["10"]}, results, retriever.k_values)

    # retriever.top_k = 1000

    new_results = retriever.retrieve(corpus, {"10": ["social distancing", "impact", "slowing", "spread", "COVID-19"]})

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    

    new_merged_resuts = intersect_res(new_results)

    print("length of the merged results::", len(new_merged_resuts))

    print("results with decomposition::")

    new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({"10": qrels["10"]}, {"10": new_merged_resuts}, retriever.k_values)


def compare_query_bad_example(retriever, corpus, queries, qrels, all_sub_corpus_embedding_ls):
    # retriever.top_k = 2500
    idx = str(50)

    print("query::", queries[idx])

    sub_q_ls = ["what is known", "mRNA vaccine", "SARS-CoV-2 virus"]

    results,_ = retriever.retrieve(corpus, {idx: sub_q_ls}, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls)

    print(queries[idx])

    print("length of the results::", len(results))

    print("results with decomposition::")

    new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({idx: qrels[idx]}, results, retriever.k_values)

    # ndcg, _map, recall, precision = retriever.evaluate({idx: qrels[idx]}, results, retriever.k_values)

    # retriever.top_k = 1000

    # new_results = retriever.retrieve(corpus, {idx: sub_q_ls})

    # #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    

    # new_merged_resuts = intersect_res(new_results)

    # print("length of the merged results::", len(new_merged_resuts))

    # print("results with decomposition::")

    # new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({idx: qrels[idx]}, {idx: new_merged_resuts}, retriever.k_values)

    # sub_q_ls = ["Coronavirus", "live", "outside the body", "how long"]

    # new_results,_ = retriever.retrieve(corpus, {idx: sub_q_ls}, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls)

    # #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    

    # # new_merged_resuts = intersect_res(new_results)

    # print("length of the merged results::", len(new_results))

    # print("results with decomposition::")

    # new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({idx: qrels[idx]}, new_results, retriever.k_values)

    print()

def compare_drug_example(retriever, corpus, queries, qrels):
    # retriever.top_k = 2500
    id_str= "5"
    
    results = retriever.retrieve(corpus, {id_str: queries[id_str]})
    
    print("query::", queries[id_str])

    print("length of the results::", len(results))

    print("results without decomposition::")

    ndcg, _map, recall, precision = retriever.evaluate({id_str: qrels[id_str]}, results, retriever.k_values)

    # retriever.top_k = 1000

    new_results = retriever.retrieve(corpus, {id_str: ["drug", "SARS-CoV", "animal studies"]})

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    

    new_merged_resuts = intersect_res(new_results)

    print("length of the merged results::", len(new_merged_resuts))

    print("results with decomposition::")

    new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({id_str: qrels[id_str]}, {id_str: new_merged_resuts}, retriever.k_values)
    
    # new_results = retriever.retrieve(corpus, {"20": ["patients", "Angiotensin-converting enzyme inhibitors", "increased risk", "COVID-19"]})

    # #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    

    # new_merged_resuts = intersect_res(new_results)

    # print("length of the merged results 2::", len(new_merged_resuts))

    # print("results with decomposition 2::")

    # new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({"20": qrels["20"]}, {"20": new_merged_resuts}, retriever.k_values)

def compare_ace_example(retriever, corpus, queries, qrels):
    # retriever.top_k = 2500
    
    results = retriever.retrieve(corpus, {"20": queries["20"]})
    
    print("query::", queries["20"])

    print("length of the results::", len(results))

    print("results without decomposition::")

    ndcg, _map, recall, precision = retriever.evaluate({"20": qrels["20"]}, results, retriever.k_values)

    # retriever.top_k = 1000

    new_results = retriever.retrieve(corpus, {"20": ["Angiotensin-converting enzyme inhibitors", "increased risk", "COVID-19"]})

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    

    new_merged_resuts = intersect_res(new_results)

    print("length of the merged results::", len(new_merged_resuts))

    print("results with decomposition::")

    new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({"20": qrels["20"]}, {"20": new_merged_resuts}, retriever.k_values)
    
    new_results = retriever.retrieve(corpus, {"20": ["patients", "Angiotensin-converting enzyme inhibitors", "increased risk", "COVID-19"]})

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    

    new_merged_resuts = intersect_res(new_results)

    print("length of the merged results 2::", len(new_merged_resuts))

    print("results with decomposition 2::")

    new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({"20": qrels["20"]}, {"20": new_merged_resuts}, retriever.k_values)
    # new_results2 = retriever.retrieve(corpus, {"2": ["weather", "coronaviruses"]})    
    
    # new_merged_resuts2 = intersect_res(new_results2)

    # print("length of the merged results::", len(new_merged_resuts2))

    # print("results with decomposition 2::")

    # new_ndcg2, new_map2, new_recall2, new_precision2 = retriever.evaluate({"2": qrels["2"], "3":qrels["2"]}, {"2": new_merged_resuts2, "3":new_merged_resuts2}, retriever.k_values)

def evaluate_for_query_batches(qrels, results):
    all_results = dict()    
    for key in tqdm(results):
        ndcg, _map, recall, precision = retriever.evaluate({key: qrels[key]}, {key: results[key]}, retriever.k_values)
        all_results[key] = dict()
        all_results[key].update(ndcg)
        all_results[key].update(_map)
        all_results[key].update(recall)
        all_results[key].update(precision)
    return all_results

def retrieve_without_decomposition(retriever, corpus, queries, qrels):
    print("results without decomposition::")
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    print("start evaluating performance for single query with no decomposition")
    return evaluate_for_query_batches(qrels, results)





def compute_decompositions_for_all_queries(retriever, corpus, qrels, output_dir, all_sub_corpus_embedding_ls):
    with open(os.path.join(output_dir, "decomposition.json"), "r") as f:
        decomposed_queries = json.load(f)

    results,_ = retriever.retrieve(corpus, decomposed_queries, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls)
    
    new_ndcg, new_map, new_recall, new_precision = retriever.evaluate(qrels, results, retriever.k_values)

    print()

def output_res_to_files(output_dir, single_query_res_without_decomposition, single_query_res_with_decomposition, queries, decomposed_queries):
    final_res = dict()
    for key in single_query_res_without_decomposition:
        final_res[key] = dict()
        final_res[key]["query"] = queries[key]
        final_res[key]["decomposed_query"] = decomposed_queries[key]
        final_res[key]["performance_no_dp"] = single_query_res_without_decomposition[key]
        final_res[key]["performance_dp"] = single_query_res_with_decomposition[key]

    output_file_name = os.path.join(output_dir, "final_res.json")
    with open(output_file_name, "w") as f:
        json.dump(final_res, f, indent=4)
        

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
# dataset = "scifact"
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
dataset = "trec-covid"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "data")
out_dir = "/data6/wuyinjun/beir/data/"
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
# corpus, qrels = subset_corpus(corpus, qrels, 500)
#### Load the SBERT model and retrieve using cosine-similarity
# model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224').to(device)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
# processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
raw_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
# processor =  lambda images: raw_processor(images=images, return_tensors="pt", padding=False, do_resize=False, do_center_crop=False)["pixel_values"]
processor =  lambda images: raw_processor(images=images, return_tensors="pt")["pixel_values"]
text_processor =  lambda text: raw_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
img_processor =  lambda images: raw_processor(images=images, return_tensors="pt")["pixel_values"]
model = model.eval()




# model = DRES(models.clip_model(text_processor, model, device), batch_size=16)
model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)
retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "cos_sim" for cosine similarity

all_sub_corpus_embedding_ls=None

# cached_embedding_path = os.path.join(out_dir, "cached_embeddings")
cached_embedding_path = os.path.join("output/all_sub_corpus_embedding_ls")

# if os.path.exists(cached_embedding_path):
#     all_sub_corpus_embedding_ls = torch.load(cached_embedding_path)

query_embeddings = model.model.encode_queries(queries, convert_to_tensor=True)

results, all_sub_corpus_embedding_ls = retriever.retrieve(corpus, queries, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls, query_embeddings=query_embeddings, query_count=-1)

# if not os.path.exists(cached_embedding_path):
#     torch.save(all_sub_corpus_embedding_ls, cached_embedding_path)
    
# compute_decompositions_for_all_queries(retriever, corpus, qrels, out_dir, all_sub_corpus_embedding_ls)
# single_q_res_no_decomposition = retrieve_without_decomposition(retriever, corpus, queries, qrels)

# single_q_res_with_decomposition, decomposed_queries = retrieve_with_decomposition(retriever, corpus, queries, qrels, out_dir, dataset)

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

# output_res_to_files(out_dir, single_q_res_no_decomposition, single_q_res_with_decomposition, queries, decomposed_queries)

# compare_drug_example(retriever, corpus, queries, qrels)
# compare_ace_example(retriever, corpus, queries, qrels)
# compare_query_social_distance(retriever, corpus, queries, qrels)
# compare_query_bad_example(retriever, corpus, queries, qrels, all_sub_corpus_embedding_ls)

# compare_query_weather_example(retriever, corpus, queries, qrels)
# compare_and_not_query(retriever, corpus, queries, qrels)

# 

# new_results = retriever.retrieve(corpus, {"51": "change of weather", "52":"coronaviruses"})

# #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
# print("results without decomposition::")

# ndcg, _map, recall, precision = retriever.evaluate({"2": qrels["2"], "3":qrels["2"]}, results, retriever.k_values)

# new_merged_resuts = intersect_res(new_results)

# print("results with decomposition::")

# new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({"2": qrels["2"], "3":qrels["2"]}, {"2": new_merged_resuts, "3":new_merged_resuts}, retriever.k_values)


print()