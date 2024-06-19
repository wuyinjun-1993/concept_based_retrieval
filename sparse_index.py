from nltk import word_tokenize
from nltk.corpus import stopwords
import torch
import numpy as np
import string
import json,os

stopwords = set(stopwords.words('english') + list(string.punctuation))


def construct_sparse_index(jsonl_data, vocab_dict, batch_ids, next_token_logits, corpus, tokenizer, sep=" "):
    if type(corpus) is dict:
        sentences = [(corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
    else:
        if type(corpus) is list:
            if type(corpus[0]) is dict:
                sentences = [(doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
            else:
                sentences = corpus
    for docid, logits, text in zip(batch_ids, next_token_logits, sentences):
        vector = dict()
        words = [i for i in word_tokenize(text.lower()) if i not in stopwords]
        token_ids = set()
        for word in words:
            token_ids.update(tokenizer.encode(word, add_special_tokens=False))

        # top tokens in the text
        token_ids_in_text = torch.tensor(list(token_ids))
        if len(token_ids_in_text) == 0:
            top_k_values, top_k_indices = logits.topk(10, dim=-1)
            values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
            tokens = [vocab_dict[i.item()] for i in top_k_indices.cpu().detach().float().numpy()]
        else:
            top_k = min(len(token_ids_in_text), 128)
            top_k_values, top_k_indices = logits[token_ids_in_text].topk(top_k, dim=-1)
            values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
            tokens = [vocab_dict[i.item()] for i in token_ids_in_text[top_k_indices.cpu().detach().float().numpy()]]
            
        for token, v in zip(tokens, values):
            vector[token] = int(v)
        jsonl_data.append(
                dict(
                    id=docid,
                    content="",
                    vector=vector,
                )
            )


def store_sparse_index(sample_hash, jsonl_data, encoding_query = True):
    output_folder = f"output/sparse_index_{sample_hash}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if encoding_query:
        output_file = os.path.join(output_folder, "query.tsv")
    else:
        output_file = os.path.join(output_folder, "corpus_0.jsonl")
    
    # os.makedirs(data_args.sparse_output_dir, exist_ok=True)
    # with open(os.path.join(data_args.sparse_output_dir, 'query.tsv' if data_args.encode_is_query else f'corpus_{data_args.dataset_shard_index}.jsonl'), 'w') as f:
    with open(output_file, "w") as f:
        for data in jsonl_data:
            if encoding_query:
                id = data['id']
                vector = data['vector']
                query = " ".join(
                            [" ".join([str(token)] * freq) for token, freq in vector.items()])
                if len(query.strip()) == 0:
                    continue
                f.write(f"{id}\t{query}\n")
            else:
                f.write(json.dumps(data) + "\n")
                
                
def run_search_with_sparse_index(sample_hash):
    output_folder = f"output/sparse_index_{sample_hash}"
    if os.path.exists(output_folder + "/rank.sparse.trec"):
        return
    
    index_cmd = "python -m pyserini.index.lucene --collection JsonVectorCollection --input " + output_folder \
    + " --index " + output_folder + "/index --generator DefaultLuceneDocumentGenerator --threads 16 --impact --pretokenized"
    
    print("build index command::", index_cmd)
    os.system(index_cmd)
    
    search_cmd = "python -m pyserini.search.lucene --index " + output_folder + "/index" \
        + " --topics " + output_folder + "/query.tsv --output " + output_folder + "/rank.sparse.trec" \
         + " --output-format trec --batch 32 --threads 16 --hits 1000  --impact --pretokenized --remove-query"
    
    print("search with index command::", search_cmd)
    os.system(search_cmd)
    
    
def read_trec_run(sample_hash, query_count, doc_count):
    output_folder = f"output/sparse_index_{sample_hash}"
    file = os.path.join(output_folder, "rank.sparse.trec")
    # run = {}
    sim_scores = torch.zeros(query_count, doc_count) + 1e-6
    line_ls = []
    with open(file, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            sim_scores[int(qid)][int(docid)] = float(score)
            line_ls.append(line)
            # if qid not in run:
            #     run[qid] = {}
            # run[qid][docid] = float(score)
    return sim_scores