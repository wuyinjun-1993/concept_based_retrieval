from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


import logging
import pathlib, os
import os, sys

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

#### Load the SBERT model and retrieve using cosine-similarity
model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)
retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "cos_sim" for cosine similarity


# compare_query_social_distance(retriever, corpus, queries, qrels)

compare_ace_example(retriever, corpus, queries, qrels)
# compare_query_weather_example(retriever, corpus, queries, qrels)
# compare_and_not_query(retriever, corpus, queries, qrels)

# results = retriever.retrieve(corpus, {"2": queries["2"], "3": queries["2"]})

# new_results = retriever.retrieve(corpus, {"51": "change of weather", "52":"coronaviruses"})

# #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
# print("results without decomposition::")

# ndcg, _map, recall, precision = retriever.evaluate({"2": qrels["2"], "3":qrels["2"]}, results, retriever.k_values)

# new_merged_resuts = intersect_res(new_results)

# print("results with decomposition::")

# new_ndcg, new_map, new_recall, new_precision = retriever.evaluate({"2": qrels["2"], "3":qrels["2"]}, {"2": new_merged_resuts, "3":new_merged_resuts}, retriever.k_values)


print()