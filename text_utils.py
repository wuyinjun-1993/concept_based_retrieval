import torch
import os
import utils
from tqdm import tqdm

import re
import json
from retrieval_utils import decompose_single_query_parition_groups
from sparse_index import construct_sparse_index, store_sparse_index
sparse_prefix='<s><|system|>\\nYou are an AI assistant that can understand human language.<|end|>\\n<|user|>\\nQuery: "'
sparse_suffix='". Use one most important word to represent the query in retrieval task. Make sure your word is in lowercase.<|end|>\\n<|assistant|>\\nThe word is: "'


class ConceptLearner_text:
    # def __init__(self, samples: list[PIL.Image], input_to_latent, input_processor, device: str = 'cpu'):
    def __init__(self, corpus: list, model, dataset_name, device: str = 'cpu'):
        self.corpus = corpus
        self.device = torch.device(device)
        self.batch_size = 128
        # self.input_to_latent = input_to_latent
        # self.image_size = 224 if type(samples[0]) == PIL.Image.Image else None
        # self.input_processor = input_processor
        self.dataset_name = dataset_name
        self.model = model
        self.split_corpus_ls =[]

    def get_corpus_embeddings(self):
        return self.model.encode_corpus(self.corpus)

    
    def split_corpus(self):
        for corpus in self.corpus:
            corpus_content = corpus["text"]
            corpus_content_split = re.split(r"[.|\?|\!]", corpus_content)
            corpus_content_split = [x.strip() for x in corpus_content_split if len(x) > 0]
            corpus_content_split = [corpus["title"]] + corpus_content_split
            self.split_corpus_ls.append(corpus_content_split)
    
    
    def split_and_encoding_single_corpus(self, corpus_idx, patch_count=4):
        sentence_ls = []
        # parser = re.compile(r"[.|\?|\!]")
        # corpus_content = corpus["text"]
        # corpus_content_split = re.split(r"[.|\?|\!]", corpus_content)
        # corpus_content_split = [x.strip() for x in corpus_content_split if len(x) > 0]
        corpus_content_split = self.split_corpus_ls[corpus_idx] # [corpus["title"]] + corpus_content_split
        
        bbox_ls = []
        
        for idx in range(len(corpus_content_split)):
            end_idx = min(idx+patch_count, len(corpus_content_split))
            sentence_ls.append(" ".join(corpus_content_split[idx:end_idx]))
            bbox_ls.append([idx, end_idx])
            if idx+patch_count >= len(corpus_content_split):
                break
        
        curr_corpus_embedding = self.model.encode_str_ls(sentence_ls, convert_to_tensor=True, show_progress_bar=False)
        
        return curr_corpus_embedding.cpu(), bbox_ls
        
    
    
    def get_patches(self, model_name, samples_hash, method="slic", patch_count=32):
        
        if model_name == "default":
            cached_file_name=f"output/saved_patches_{method}_{patch_count}_{samples_hash}.pkl"
        elif model_name == "llm":
            cached_file_name=f"output/saved_patches_{method}_llm_{patch_count}_{samples_hash}.pkl"
        else:
            raise Exception("Invalid model name")
        
        if os.path.exists(cached_file_name):
            print("Loading cached patches")
            print(samples_hash)
            patch_activations, full_bbox_ls = utils.load(cached_file_name)
            # if image_embs is None and compute_img_emb:
            #     image_embs = self.get_corpus_embeddings()
            # if type(patch_activations) is list:
            #     patch_activations = [key.cpu() for key in patch_activations]
            # elif type(patch_activations) is dict:
            #     patch_activations = [patch_activations[key].cpu() for key in range(len(patch_activations))]
            #     utils.save(patch_activations, cached_file_name)
            return patch_activations, full_bbox_ls
        
        # if compute_img_emb:
        #     image_embs = self.get_corpus_embeddings()
        
        if len(self.split_corpus_ls) == 0:
            self.split_corpus()
        
        patch_activations = []
        full_bbox_ls = []
        
        for key in tqdm(range(len(self.corpus))):
            patch_activation, bbox_ls = self.split_and_encoding_single_corpus(key, patch_count=patch_count)
            # patch_activations[key] = torch.cat(patch_activation)
            patch_activations.append(patch_activation)
            full_bbox_ls.append(bbox_ls)
        
        utils.save((patch_activations, full_bbox_ls), cached_file_name)
        
        return patch_activations, full_bbox_ls

def generate_patch_ids_ls(patch_embs_ls):
    patch_ids_ls = []
    nested_patch_ids_ls = []
    transformed_patch_embs_ls = []
    for patch_embs in patch_embs_ls:
        curr_patch_ids_ls = []
        for idx in range(len(patch_embs)):
            curr_patch_ids_ls.append(torch.ones(patch_embs[idx].shape[0]).long()*idx)
        patch_ids_ls.append(torch.cat(curr_patch_ids_ls, dim=0))
        nested_patch_ids_ls.append(curr_patch_ids_ls)
        transformed_patch_embs_ls.append(torch.cat(patch_embs, dim=0))
    return nested_patch_ids_ls, patch_ids_ls, transformed_patch_embs_ls

def construct_dense_or_sparse_encodings_queries(queries, text_model,add_sparse_index):
    jsonl_data = []
    vocab_dict = text_model.q_model.tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    tokenizer = text_model.q_model.tokenizer
    text_emb_dense = text_model.encode_queries(queries, convert_to_tensor=True)
    if add_sparse_index:
        text_emb_sparse = text_model.encode_queries(queries, convert_to_tensor=True, is_sparse=True)
        if type(queries) is dict:
            queries = [queries[str(i+1)] for i in range(len(queries))]
        construct_sparse_index(jsonl_data, vocab_dict, list(range(len(queries))), text_emb_sparse, queries, tokenizer)
    return text_emb_dense, jsonl_data


def construct_dense_or_sparse_encodings(args, corpus, text_model, samples_hash, is_sparse=False):
    if args.model_name == "default":
        corpus_embedding_file_name = f"output/saved_corpus_embeddings_{samples_hash}"
    elif args.model_name == "llm":
        corpus_embedding_file_name = f"output/saved_corpus_embeddings_llm_{samples_hash}"
    elif args.model_name == "phi":
        corpus_embedding_file_name = f"output/saved_corpus_embeddings_phi_{samples_hash}"
    else:
        raise Exception("Invalid model name")
    if is_sparse:
        corpus_embedding_file_name += "_sparse"
    corpus_embedding_file_name += ".pkl"
    jsonl_data = []
    vocab_dict = text_model.q_model.tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    tokenizer = text_model.q_model.tokenizer
    if os.path.exists(corpus_embedding_file_name):
        img_emb = utils.load(corpus_embedding_file_name)
        if not is_sparse:
            img_emb = img_emb.cpu()
    else:
        local_bz = 1024
        bz = 1024
        if is_sparse or args.model_name == "llm":
            bz = 32
        for idx in tqdm(range(0, len(corpus), local_bz)):
            end_id = min(idx+local_bz, len(corpus))
            curr_corpus = corpus[idx:end_id]
            img_emb = text_model.encode_corpus(curr_corpus,convert_to_tensor=True, batch_size=bz, show_progress_bar=False, is_sparse=is_sparse)
            if is_sparse:
                construct_sparse_index(jsonl_data, vocab_dict, list(range(idx, end_id)), img_emb, curr_corpus, tokenizer)
            else:
                if idx == 0:
                    img_emb_ls = img_emb.cpu()
                else:
                    img_emb_ls = torch.cat([img_emb_ls, img_emb.cpu()], dim=0)
        # img_emb = text_model.encode_corpus(corpus,convert_to_tensor=True, show_progress_bar=False)    
        if not is_sparse:
            img_emb = img_emb_ls
        else:
            img_emb = jsonl_data
        utils.save(img_emb, corpus_embedding_file_name)
    return img_emb





def convert_samples_to_concepts_txt(args, text_model, corpus, device, patch_count_ls = [32], is_sparse=False):
    cl = ConceptLearner_text(corpus, text_model, args.dataset_name, device=device)
    
    sentences = text_model.convert_corpus_to_ls(corpus)
    
    if args.total_count > 0:
        sentences = sorted(sentences)
        samples_hash = utils.hashfn(sentences)
    else:
        samples_hash = f"{args.dataset_name}_full"
    print("sample hash::", samples_hash)
    
    # if args.model_name == "default":
    #     corpus_embedding_file_name = f"output/saved_corpus_embeddings_{samples_hash}.pkl"
    # elif args.model_name == "llm":
    #     corpus_embedding_file_name = f"output/saved_corpus_embeddings_llm_{samples_hash}.pkl"
    # elif args.model_name == "phi":
    #     corpus_embedding_file_name = f"output/saved_corpus_embeddings_phi_{samples_hash}.pkl"
    # else:
    #     raise Exception("Invalid model name")
    
    # if os.path.exists(corpus_embedding_file_name):
    #     img_emb = utils.load(corpus_embedding_file_name)
    #     img_emb = img_emb.cpu()
    # else:
    #     local_bz = 20480
    #     for idx in tqdm(range(0, len(corpus), local_bz)):
    #         end_id = min(idx+local_bz, len(corpus))
    #         curr_corpus = corpus[idx:end_id]
    #         img_emb = text_model.encode_corpus(curr_corpus,convert_to_tensor=True, batch_size=2048, show_progress_bar=False)
    #         if idx == 0:
    #             img_emb_ls = img_emb.cpu()
    #         else:
    #             img_emb_ls = torch.cat([img_emb_ls, img_emb.cpu()], dim=0)
    #     # img_emb = text_model.encode_corpus(corpus,convert_to_tensor=True, show_progress_bar=False)    
    #     img_emb = img_emb_ls
    #     utils.save(img_emb, corpus_embedding_file_name)
        
    img_emb = construct_dense_or_sparse_encodings(args, corpus, text_model, samples_hash)
    img_sparse_emb = None
    if args.add_sparse_index:
        img_sparse_emb = construct_dense_or_sparse_encodings(args, corpus, text_model, samples_hash, is_sparse=True)
        store_sparse_index(samples_hash, img_sparse_emb, encoding_query = False)
    if args.img_concept:
        patch_activation_ls=[]
        full_bbox_ls = []
        for idx in range(len(patch_count_ls)):
            patch_count = patch_count_ls[idx]
            patch_activations, bbox_ls = cl.get_patches(args.model_name, samples_hash, method="slic", patch_count=patch_count)
            # cos_sim_ls = []
            # for sub_idx in range(len(patch_activations)):
            #     cos_sim = torch.nn.functional.cosine_similarity(img_emb[sub_idx].view(1,-1), patch_activations[sub_idx].view(1,-1)).item()
            #     cos_sim_ls.append(cos_sim)
            # print()
            patch_activation_ls.append(patch_activations)
            full_bbox_ls.append(bbox_ls)
        
        return samples_hash, (img_emb,img_sparse_emb), patch_activation_ls, full_bbox_ls
    else:
        return samples_hash, (img_emb,img_sparse_emb), None, None

def convert_corpus_to_concepts_txt(corpus, qrels):
    key_ls = list(corpus.keys())
    
    key_str_idx_mappings = {key: idx for idx, key in enumerate(key_ls)}
    
    new_qrels = dict()
    
    new_corpus = [None]*len(corpus)
    
    for key in tqdm(corpus):
        new_key = key_str_idx_mappings[key]
        new_corpus[new_key] = corpus[key]
    # corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
    # new_corpus = [corpus[cid] for cid in corpus_ids]
        
    for key in tqdm(qrels):
        new_qrels[key] = dict()
        for sub_key in qrels[key]:
            if sub_key in key_ls:
                new_sub_key = key_str_idx_mappings[sub_key]
                new_qrels[key][str(new_sub_key+1)] = qrels[key][sub_key]
                
    return new_corpus, new_qrels
    
    


def subset_corpus(corpus, qrels, count):
    if count < 0:
        return corpus, qrels
    key_ls = list(corpus.keys())[:count]
    
    sub_corpus = {key: corpus[key] for key in key_ls}
    
    sub_qrels = {key: {sub_key: qrels[key][sub_key] for sub_key in qrels[key] if sub_key in key_ls} for key in tqdm(qrels)}
    
    return sub_corpus, sub_qrels

def subset_corpus2(corpus, qrels, count, idx_to_rid):
    if count < 0:
        return corpus, qrels
    key_ls = set()
    for rid in list(idx_to_rid.values()):
        key_ls.update(list(qrels[rid].keys()))
    
    key_ls = list(key_ls)
    
    if len(key_ls) < count:
        remaining_keys = list(set(corpus.keys()).difference(set(key_ls)))
        remaining_keys = sorted(remaining_keys)
        key_ls = key_ls + remaining_keys[:count - len(key_ls)]
        
    
    sub_corpus = {key: corpus[key] for key in key_ls}
    
    sub_qrels = {key: {sub_key: qrels[key][sub_key] for sub_key in qrels[key] if sub_key in key_ls} for key in tqdm(qrels)}
    
    return sub_corpus, sub_qrels
    
    
def reformat_patch_embeddings_txt(patch_emb_ls, img_emb):
    # img_per_patch_tensor = torch.tensor(img_per_patch_ls[0])
    max_img_id = len(patch_emb_ls[0])
    patch_emb_curr_img_ls = []
    cosin_sim_ls = []
    for idx in tqdm(range(max_img_id)):
        sub_patch_emb_curr_img_ls = []
        for sub_idx in range(len(patch_emb_ls)):
            patch_emb = patch_emb_ls[sub_idx]
            # img_per_batch = img_per_patch_ls[sub_idx]
            # img_per_patch_tensor = torch.tensor(img_per_batch)
            # for sub_sub_idx in range(patch_emb_ls[sub_idx]):
            patch_emb_curr_img = patch_emb[idx]
            sub_patch_emb_curr_img_ls.append(patch_emb_curr_img)
        sub_patch_emb_curr_img = torch.cat(sub_patch_emb_curr_img_ls, dim=0)
        curr_img_emb = img_emb[idx]
        if len(curr_img_emb.shape) == 1:
            curr_img_emb = curr_img_emb.unsqueeze(0)
        
        # patch_emb_curr_img = torch.cat([curr_img_emb, sub_patch_emb_curr_img], dim=0)
        patch_emb_curr_img = torch.cat([curr_img_emb, sub_patch_emb_curr_img], dim=0)
        
        # cosin_sim = torch.nn.functional.cosine_similarity(sub_patch_emb_curr_img.view(1,-1), curr_img_emb.view(1,-1))
        # cosin_sim_ls.append(cosin_sim)
        patch_emb_curr_img_ls.append(patch_emb_curr_img)
    return patch_emb_curr_img_ls


def read_queries_with_sub_queries_file(filename, subset_img_id=None):
    with open(filename, 'r') as json_file:
        json_list = list(json_file)

    queries = dict()
    
    sub_queries_ls= dict()
    # idx = 1
    rid = 1
    rid_ls = []
    idx_to_rid = dict()
    group_q_ls = dict()
    for json_str in json_list:
        result = json.loads(json_str)
        if "sub_text" in result:
            idx = result["_id"]
            queries[str(rid)] = result["text"]
            sub_queries_ls[str(rid)] = result["sub_text"]
            grouped_sub_q_ids_ls = None
            if "group" in result:
                group_q_ls_str = result["group"]
                grouped_sub_q_ids_ls = decompose_single_query_parition_groups(result["sub_text"], group_q_ls_str)
            group_q_ls[str(rid)] = grouped_sub_q_ids_ls
            idx_to_rid[str(rid)] = str(idx)
        # idx += 1
            rid_ls.append(str(rid))
            rid += 1
    
    if subset_img_id is None:
        return queries, sub_queries_ls, idx_to_rid
    else:
        key = rid_ls[subset_img_id]
        return {"1": queries[key]}, {"1":sub_queries_ls[key]}, {"1":idx_to_rid[key]}

def check_empty_mappings(curr_gt):
    value_ls = set(list(curr_gt.values()))
    if len(value_ls) == 1 and list(value_ls)[0] == 0:
        return True
    return False

def encode_sub_queries_ls(sub_queries_ls, text_model):
    all_sub_queries_emb_ls = []
    # for key in sub_queries_ls:
    for key in range(len(sub_queries_ls)):
        sub_queries_emb_ls=[]
        key_str = str(key + 1)
        for sub_key in range(len(sub_queries_ls[key_str])):
            sub_queries = sub_queries_ls[key_str][sub_key]
            sub_queries_emb = text_model.encode_str_ls(sub_queries, convert_to_tensor=True)
            sub_queries_emb_ls.append(sub_queries_emb)
        all_sub_queries_emb_ls.append(sub_queries_emb_ls)
    
    return all_sub_queries_emb_ls
