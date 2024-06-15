from sentence_transformers import SentenceTransformer
from torch import Tensor
import torch.multiprocessing as mp
from typing import List, Dict, Union, Tuple
import numpy as np
import logging
from datasets import Dataset
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification



logger = logging.getLogger(__name__)

def obtain_model_embeddings(qry, outputs, device):
    # logits = outputs.logits
    sequence_lengths = qry['attention_mask'].sum(dim=1) - 1
    batch_ids = torch.arange(len(qry['input_ids']), device=device)

    # next token logits
    

    # next token hidden states
    next_token_reps = outputs.hidden_states[-1][batch_ids, sequence_lengths]

    return next_token_reps

def obtain_model_tokens(embeddings, vocab_model):
    prediction_logits = vocab_model.vocab_transform(embeddings)
    prediction_logits = vocab_model.activation(prediction_logits)
    prediction_logits = vocab_model.vocab_layer_norm(prediction_logits)
    prediction_logits = vocab_model.vocab_projector(prediction_logits)
    # next_token_logits = logits[batch_ids, sequence_lengths]
    prediction_logits = torch.log(1 + torch.relu(prediction_logits))
    return prediction_logits


class ms_phi:
    def __init__(self, sep: str = " ", prefix="", suffix="", device = "cuda", **kwargs):
        self.sep = sep
        self.device = device
        
        # self.model = AutoModelForCausalLM.from_pretrained(
        #             "microsoft/Phi-3-mini-4k-instruct", 
        #             device_map="cuda", 
        #             torch_dtype="auto", 
        #             trust_remote_code=True, 
        #         ).to(device)
        # self.model = self.q_model
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        
        model_name = "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.vocab_model = DistilBertForMaskedLM.from_pretrained(model_name).to(device)
        self.vocab_model.eval()
        self.prefix = prefix
        self.suffix = suffix
        # if isinstance(model_path, str):
        #     self.q_model = SentenceTransformer(model_path)
        #     self.doc_model = self.q_model
        
        # elif isinstance(model_path, tuple):
        #     self.q_model = SentenceTransformer(model_path[0])
        #     self.doc_model = SentenceTransformer(model_path[1])
    
    def tokenize_str_ls(self, str_ls, is_sparse=False, max_length=512):
        collated_texts = self.tokenizer(str_ls, 
                     return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False,
                    # max_length=max_length,
                    padding=False,
                    truncation=True)
        
        if is_sparse:  
            collated_texts['input_ids'] = [self.tokenizer.encode(self.prefix, add_special_tokens=False) + input_ids +
                                       self.tokenizer.encode(self.suffix, add_special_tokens=False)
                                       for input_ids in collated_texts['input_ids']]
        
        features = self.tokenizer.pad(
            collated_texts,
            padding=True,
            pad_to_multiple_of=16,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return features
    
    def encode(self, str_ls, batch_size, is_sparse=False, **kwargs):
        token_logits_ls = []
        token_reps_ls = []
        with torch.no_grad():
            for start_idx in range(0, len(str_ls), batch_size):
                end_idx = min(start_idx + batch_size, len(str_ls))
                sub_str_ls = str_ls[start_idx:end_idx]
                features  =self.tokenize_str_ls(sub_str_ls, is_sparse)
                features["input_ids"] = features["input_ids"].to(self.device)
                features["attention_mask"] = features["attention_mask"].to(self.device)
                outputs = self.model(input_ids=features["input_ids"], attention_mask=features["attention_mask"], output_hidden_states=True,
                            return_dict=True)
                
                token_reps = obtain_model_embeddings(features, outputs, self.device)
                token_logits = obtain_model_tokens(token_reps, self.vocab_model)
                token_logits_ls.append(token_logits.cpu())
                token_reps_ls.append(token_reps.cpu())
        token_logits = torch.cat(token_logits_ls, dim=0)
        token_reps = torch.cat(token_reps_ls, dim=0)
        return token_logits, token_reps
    
    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for process_id, device_name in enumerate(target_devices):
            p = ctx.Process(target=SentenceTransformer._encode_multi_process_worker, args=(process_id, device_name, self.model, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    def stop_multi_process_pool(self, pool: Dict[str, object]):
        output_queue = pool['output']
        [output_queue.get() for _ in range(len(pool['processes']))]
        return self.model.stop_multi_process_pool(pool)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if type(queries) is dict:
            queries = [queries[str(i+1)] for i in range(len(queries))]
        return self.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 1, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.encode(sentences, batch_size=batch_size, **kwargs)

    def convert_corpus_to_ls(self, corpus):
        if type(corpus) is dict:
                # sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
            sentences = [corpus[i]["title"].strip() + self.sep + corpus[i]["text"].strip() if "title" in corpus[i] else corpus[i]["text"].strip() for i in corpus]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        
        return sentences

    ## Encoding corpus in parallel
    def encode_corpus_parallel(self, corpus: Union[List[Dict[str, str]], Dataset], pool: Dict[str, str], batch_size: int = 2, chunk_id: int = None, **kwargs):
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        
        if chunk_id is not None and chunk_id >= len(pool['processes']):
            output_queue = pool['output']
            output_queue.get()

        input_queue = pool['input']
        input_queue.put([chunk_id, batch_size, sentences])
        
    def encode_str_ls(self, str_ls, batch_size=2, **kwargs):
        # text_feature_ls = []
        with torch.no_grad():
            text_feature_ls = self.encode(str_ls, batch_size=batch_size, **kwargs)
            # for sentence in tqdm(str_ls):
            #     inputs = self.processor(sentence)
            #     inputs = {key: val.to(self.device) for key, val in inputs.items()}
            #     text_features = self.model.get_text_features(**inputs)
            #     text_feature_ls.append(text_features.cpu())
        return text_feature_ls
