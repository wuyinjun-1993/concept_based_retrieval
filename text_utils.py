import torch
import os
import utils
from tqdm import tqdm

import re

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

    def get_corpus_embeddings(self):
        return self.model.encode_corpus(self.corpus)

    
    def split_and_encoding_single_corpus(self, corpus, patch_count=4):
        sentence_ls = [corpus["title"]]
        # parser = re.compile(r"[.|\?|\!]")
        corpus_content = corpus["text"]
        corpus_content_split = re.split(r"[.|\?|\!]", corpus_content)
        corpus_content_split = [x.strip() for x in corpus_content_split if len(x) > 0]
        
        for idx in range(len(corpus_content_split)):
            sentence_ls.append(".".join(corpus_content_split[idx:idx+patch_count]))
            if idx+patch_count >= len(corpus_content_split):
                break
        
        curr_corpus_embedding = self.model.encode_str_ls(sentence_ls, convert_to_tensor=True)
        
        return curr_corpus_embedding
        
    
    
    def get_patches(self, samples_hash, method="slic", patch_count=32):
        
        cached_file_name=f"output/saved_patches_{method}_{patch_count}_{samples_hash}.pkl"
        
        if os.path.exists(cached_file_name):
            print("Loading cached patches")
            print(samples_hash)
            patch_activations = utils.load(cached_file_name)
            # if image_embs is None and compute_img_emb:
            #     image_embs = self.get_corpus_embeddings()
            return patch_activations
        
        # if compute_img_emb:
        #     image_embs = self.get_corpus_embeddings()
        
        patch_activations = {}
        
        for key in range(len(self.corpus)):
            patch_activation = self.split_and_encoding_single_corpus(self.corpus[key], patch_count=patch_count)
            # patch_activations[key] = torch.cat(patch_activation)
            patch_activations[key] = patch_activation
        
        utils.save(patch_activations, cached_file_name)
        
        return patch_activations

def convert_samples_to_concepts_txt(args, text_model, corpus, device, patch_count_ls = [32]):
    cl = ConceptLearner_text(corpus, text_model, args.dataset_name, device=device)
    
    sentences = text_model.convert_corpus_to_ls(corpus)
    
    if args.total_count > 0:
        samples_hash = utils.hashfn(sentences)
    else:
        samples_hash = f"{args.dataset_name}_full"
    
    corpus_embedding_file_name = f"output/saved_corpus_embeddings_{samples_hash}.pkl"
    
    if os.path.exists(corpus_embedding_file_name):
        img_emb = utils.load(corpus_embedding_file_name)
    else:
        img_emb = text_model.encode_corpus(corpus,convert_to_tensor=True)    
        utils.save(img_emb, corpus_embedding_file_name)
    
    if args.img_concept:
        patch_activation_ls=[]
        for idx in tqdm(range(len(patch_count_ls))):
            patch_count = patch_count_ls[idx]
            patch_activations = cl.get_patches(samples_hash, method="slic", patch_count=patch_count)
            patch_activation_ls.append(patch_activations)
        
        return img_emb, patch_activation_ls
    else:
        return img_emb, None

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
    
    
def reformat_patch_embeddings_txt(patch_emb_ls, img_emb):
    # img_per_patch_tensor = torch.tensor(img_per_patch_ls[0])
    max_img_id = len(patch_emb_ls[0])
    patch_emb_curr_img_ls = []
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
        
        patch_emb_curr_img = torch.cat([curr_img_emb, sub_patch_emb_curr_img], dim=0)
        patch_emb_curr_img_ls.append(patch_emb_curr_img)
    return patch_emb_curr_img_ls

    