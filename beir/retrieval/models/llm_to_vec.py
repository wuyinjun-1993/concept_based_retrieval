from llm2vec import LLM2Vec
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from torch import Tensor
import torch.multiprocessing as mp
from typing import List, Dict, Union, Tuple
import numpy as np
import logging
from datasets import Dataset
from tqdm import tqdm


logger = logging.getLogger(__name__)


class LlmtoVec:
    def __init__(self, base_model_path: Union[str, Tuple] = None, peft_model_path: Union[str, Tuple] = None, pooling_mode="mean", q_max_length = 256, d_max_length=4096, sep: str = " ", **kwargs):
        self.sep = sep
        device = torch.device("cuda")
        if isinstance(base_model_path, str) and isinstance(peft_model_path, str):
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                config=config,
                torch_dtype=torch.float32,
                device_map="cpu",
            ).to(device)
            model = PeftModel.from_pretrained(
                model,
                base_model_path
            ).to(device)
            model = model.merge_and_unload()
            model = PeftModel.from_pretrained(
                model, peft_model_path
            ).to(device)
            self.q_model = LLM2Vec(model, tokenizer, pooling_mode, d_max_length)
            self.doc_model = self.q_model
        elif isinstance(base_model_path, tuple) and isinstance(peft_model_path, tuple):
            print("Using seperate models.")
            q_tokenizer = AutoTokenizer.from_pretrained(base_model_path[0])
            q_config = AutoConfig.from_pretrained(base_model_path[0], trust_remote_code=True)
            q_model = AutoModel.from_pretrained(
                base_model_path[0],
                trust_remote_code=True,
                config=q_config,
                torch_dtype=torch.float32,
                # device_map="cpu",
            ).to(device)
            q_model = PeftModel.from_pretrained(
                q_model,
                base_model_path[0]
            ).to(device)
            q_model = q_model.merge_and_unload()
            q_model = PeftModel.from_pretrained(
                q_model, peft_model_path[0]
            ).to(device)
            self.q_model = LLM2Vec(q_model, q_tokenizer, pooling_mode, q_max_length)
            
            # d_tokenizer = AutoTokenizer.from_pretrained(base_model_path[1])
            # d_config = AutoConfig.from_pretrained(base_model_path[1], trust_remote_code=True)
            # d_model = AutoModel.from_pretrained(
            #     base_model_path[1],
            #     trust_remote_code=True,
            #     config=d_config,
            #     torch_dtype=torch.float32,
            #     device_map="cpu",
            # ).to(device)
            # d_model = PeftModel.from_pretrained(
            #     d_model,
            #     base_model_path[1]
            # ).to(device)
            # d_model = d_model.merge_and_unload()
            # d_model = PeftModel.from_pretrained(
            #     d_model, peft_model_path[1]
            # ).to(device)
            self.doc_model = self.q_model #LLM2Vec(d_model, d_tokenizer, pooling_mode, d_max_length)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if type(queries) is dict:
            queries = [queries[str(i+1)] for i in range(len(queries))]
        text_feature_ls = self.q_model.encode(queries, batch_size=batch_size)
        #output_ls = [vec.unsqueeze(0) for vec in text_feature_ls.unbind(dim=0)]
        print("YO queries")
        text_feature_ls = torch.nn.functional.normalize(text_feature_ls, p=2, dim=1)
        print(text_feature_ls.shape)
        return text_feature_ls
    
    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        text_feature_ls = self.doc_model.encode(sentences, batch_size=batch_size)
        #output_ls = [vec.unsqueeze(0) for vec in text_feature_ls.unbind(dim=0)]
        text_feature_ls = torch.nn.functional.normalize(text_feature_ls, p=2, dim=1)
        print("YO docs")
        print(text_feature_ls.shape)
        return text_feature_ls

    def convert_corpus_to_ls(self, corpus):
        if type(corpus) is dict:
                # sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
            sentences = [corpus[i]["title"].strip() + self.sep + corpus[i]["text"].strip() if "title" in corpus[i] else corpus[i]["text"].strip() for i in corpus]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        
        return sentences

    ## Encoding corpus in parallel
    def encode_corpus_parallel(self, corpus: Union[List[Dict[str, str]], Dataset], pool: Dict[str, str], batch_size: int = 8, chunk_id: int = None, **kwargs):
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        
        if chunk_id is not None and chunk_id >= len(pool['processes']):
            output_queue = pool['output']
            output_queue.get()

        input_queue = pool['input']
        input_queue.put([chunk_id, batch_size, sentences])
        
    def encode_str_ls_query(self, str_ls, batch_size=8, **kwargs):
        with torch.no_grad():
            text_feature_ls = self.q_model.encode(str_ls, batch_size=batch_size)
            text_feature_ls = torch.nn.functional.normalize(text_feature_ls, p=2, dim=1)
        #output_ls = [vec.unsqueeze(0) for vec in text_feature_ls.unbind(dim=0)]
        return text_feature_ls
        
    def encode_str_ls_doc(self, str_ls, batch_size=8, **kwargs):
        with torch.no_grad():
            text_feature_ls = self.doc_model.encode(str_ls, batch_size=batch_size)
            text_feature_ls = torch.nn.functional.normalize(text_feature_ls, p=2, dim=1)
        #output_ls = [vec.unsqueeze(0) for vec in text_feature_ls.unbind(dim=0)]
        print("YO mane")
        print(text_feature_ls.shape)
        return text_feature_ls

    def encode_str_ls(self, str_ls, batch_size=8, is_sparse=False, **kwargs):
        # text_feature_ls = []
        # if is_sparse:
        #     str_ls = [self.prefix + s + self.suffix for s in str_ls]
        with torch.no_grad():
            text_feature_ls = self.doc_model.encode(str_ls, batch_size=batch_size, **kwargs)
            # for sentence in tqdm(str_ls):
            #     inputs = self.processor(sentence)
            #     inputs = {key: val.to(self.device) for key, val in inputs.items()}
            #     text_features = self.model.get_text_features(**inputs)
            #     text_feature_ls.append(text_features.cpu())
        return text_feature_ls