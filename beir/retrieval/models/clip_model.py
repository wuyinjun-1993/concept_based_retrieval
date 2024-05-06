from sentence_transformers import SentenceTransformer
from torch import Tensor
import torch.multiprocessing as mp
from typing import List, Dict, Union, Tuple
import numpy as np
import logging
from datasets import Dataset
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)


class clip_model:
    def __init__(self, processor, model, device, sep: str = " "):
        self.processor = processor
        self.model = model
        # self.doc_model = model       
        self.device = device
        self.sep = sep
    
    def encode_str_ls(self, str_ls):
        text_feature_ls = []
        with torch.no_grad():
            for sentence in tqdm(str_ls):
                inputs = self.processor(sentence)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                text_features = self.model.get_text_features(**inputs)
                text_feature_ls.append(text_features.cpu())
        return torch.cat(text_feature_ls)
        # if isinstance(model_path, str):
        #     self.q_model = SentenceTransformer(model_path)
        #     self.doc_model = self.q_model
        
        # elif isinstance(model_path, tuple):
        #     self.q_model = SentenceTransformer(model_path[0])
        #     self.doc_model = SentenceTransformer(model_path[1])
    def convert_corpus_to_ls(self, corpus):
        if type(corpus) is dict:
                # sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
            sentences = [corpus[i]["title"].strip() + self.sep + corpus[i]["text"].strip() if "title" in corpus[i] else corpus[i]["text"].strip() for i in corpus]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        
        return sentences

    
    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for process_id, device_name in enumerate(target_devices):
            p = ctx.Process(target=SentenceTransformer._encode_multi_process_worker, args=(process_id, device_name, self.doc_model, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    def stop_multi_process_pool(self, pool: Dict[str, object]):
        output_queue = pool['output']
        [output_queue.get() for _ in range(len(pool['processes']))]
        return self.doc_model.stop_multi_process_pool(pool)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        # return self.q_model.encode(queries, batch_size=batch_size, **kwargs)
        return self.encode_str_ls(queries)
    
    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        # if type(corpus) is dict:
        #     # sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        #     sentences = [corpus[i]["title"].strip() + self.sep + corpus[i]["text"].strip() if "title" in corpus[i] else corpus[i]["text"].strip() for i in corpus]
        # else:
        #     sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        
        sentences = self.convert_corpus_to_ls(corpus)
        # return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)
        return self.encode_str_ls(sentences)
    
    def encode_corpus_by_partitions(self, corpus: Union[List[Dict[str, str]], Dataset], batch_size: int = 8, partition_size: int = 100000, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        return 

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
