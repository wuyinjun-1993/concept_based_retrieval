import torch

def reformat_documents(corpus, sep: str = " "):
    if type(corpus) is dict:
        sentences = [(corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
    else:
        if type(corpus[0]) is dict:
            sentences = [(doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        else:
            sentences = corpus
    return sentences

def reformat_queries(queries, sep: str = " "):
    if type(queries) is dict:
        queries = [queries[str(i+1)] for i in range(len(queries))]
    return queries

def tensor_res_to_res_eval(all_cos_scores_tensor, top_k=1000):
    cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(all_cos_scores_tensor, min(top_k+1, len(all_cos_scores_tensor[0])), dim=1, largest=True)#, sorted=return_sorted)
    cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
    cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
    query_count = all_cos_scores_tensor.shape[0]
    corpus_ids = [str(idx+1) for idx in list(range(all_cos_scores_tensor.shape[0]))]
    query_ids = [str(idx+1) for idx in list(range(query_count))]
    results = {qid: {} for qid in query_ids}
    # for query_itr in range(len(query_embeddings)):
    for query_itr in range(query_count):
        query_id = query_ids[query_itr]                  
        for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
            corpus_id = corpus_ids[sub_corpus_id]
            # if corpus_id != query_id:
            results[query_id][corpus_id] = score
    
    return results