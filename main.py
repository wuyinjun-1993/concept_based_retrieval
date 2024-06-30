import cv2
from transformers import CLIPModel, AutoProcessor
import torch
from image_utils import *
import argparse
from sklearn.metrics import top_k_accuracy_score
from beir.retrieval.evaluation import EvaluateRetrieval
from retrieval_utils import *
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.dense.exact_search import one, two, three, four
from beir.retrieval import models
from beir import LoggingHandler
from datasets import load_dataset
from datasets.download.download_config import DownloadConfig
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import time
# from beir.retrieval.models.clip_model import clip_model
from clustering import *
from text_utils import *
import copy
import os, shutil
from bbox_utils import *
from utils import *
from sparse_index import *
from baselines.llm_ranker import *
from baselines.bm25 import *
from derive_sub_query_dependencies import group_dependent_segments_seq_all
import random
from dessert_minheap_torch import *


image_retrieval_datasets = ["flickr", "AToMiC", "crepe", "crepe_full", "mscoco", "mscoco_40k"]
text_retrieval_datasets = ["trec-covid", "nq", "climate-fever", "hotpotqa", "msmarco"]




    
def embed_queries(filename_ls, filename_cap_mappings, processor, model, device):
    text_emb_ls = []
    with torch.no_grad():
        # for filename, caption in tqdm(filename_cap_mappings.items()):
        for file_name in filename_ls:
            caption = filename_cap_mappings[file_name]
            inputs = processor(caption)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            text_features = model.get_text_features(**inputs)
            # text_features = outputs.last_hidden_state[:, 0, :]
            text_emb_ls.append(text_features.cpu())
    return text_emb_ls

def embed_queries_with_input_queries(query_ls, processor, model, device):
    text_emb_ls = []
    with torch.no_grad():
        # for filename, caption in tqdm(filename_cap_mappings.items()):
        for caption in query_ls:
            # caption = filename_cap_mappings[file_name]
            inputs = processor(caption)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            text_features = model.get_text_features(**inputs)
            # text_features = outputs.last_hidden_state[:, 0, :]
            text_emb_ls.append(text_features.cpu())
    return text_emb_ls

def embed_queries_ls(full_sub_queries_ls, processor, model, device):
    text_emb_ls = []
    with torch.no_grad():
        # for filename, caption in tqdm(filename_cap_mappings.items()):
        # for file_name in filename_ls:
        for sub_queries_ls in tqdm(full_sub_queries_ls):
            sub_text_emb_ls = []
            for sub_queries in sub_queries_ls:
                sub_text_feature_ls = []
                for subquery in sub_queries:
                    # caption = filename_cap_mappings[file_name]
                    inputs = processor(subquery)
                    inputs = {key: val.to(device) for key, val in inputs.items()}
                    text_features = model.get_text_features(**inputs)
                    sub_text_feature_ls.append(text_features.cpu())
                # text_features = outputs.last_hidden_state[:, 0, :]
                sub_text_emb_ls.append(sub_text_feature_ls)
            text_emb_ls.append(sub_text_emb_ls)
    return text_emb_ls

        
def retrieve_by_full_query(img_emb, text_emb_ls):
    text_emb_tensor = torch.cat(text_emb_ls).cpu()
    scores = (img_emb @ text_emb_tensor.T).squeeze()/ (img_emb.norm(dim=-1) * text_emb_tensor.norm(dim=-1))
    true_rank = torch.tensor([i for i in range(len(text_emb_ls))])
    top_k_acc = top_k_accuracy_score(true_rank, scores, k=1)
    
    print(f"Top-k accuracy: {top_k_acc:.2f}")
    
    print()
    
    # for idx in range(len(text_emb_ls)):
    #     text_emb = text_emb_ls[idx]
    #     scores = (img_emb @ text_emb.T).squeeze()/ (img_emb.norm(dim=-1) * text_emb.norm(dim=-1))
        
        # print(scores)
        # print(scores.shape)
        # print(scores.argmax())
        # print()
    
def set_rand_seed(seed_value):
    # Set seed for Python's built-in random module
    random.seed(seed_value)
    
    # Set seed for NumPy
    np.random.seed(seed_value)
    
    # Set seed for PyTorch
    torch.manual_seed(seed_value)
    
    # Set seed for CUDA (if using a GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # If using multi-GPU.
    
    # Ensure deterministic operations for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='CUB concept learning')
    parser.add_argument('--data_path', type=str, default="/data6/wuyinjun/", help='config file')
    parser.add_argument('--dataset_name', type=str, default="crepe", help='config file')
    parser.add_argument('--model_name', type=str, default="default", help='config file')
    parser.add_argument('--query_count', type=int, default=-1, help='config file')
    parser.add_argument('--random_seed', type=int, default=0, help='config file')
    parser.add_argument('--query_concept', action="store_true", help='config file')
    parser.add_argument('--img_concept', action="store_true", help='config file')
    parser.add_argument('--total_count', type=int, default=-1, help='config file')
    parser.add_argument("--parallel", action="store_true", help="config file")
    parser.add_argument("--save_mask_bbox", action="store_true", help="config file")
    parser.add_argument("--search_by_cluster", action="store_true", help="config file")
    parser.add_argument('--algebra_method', type=str, default=one, help='config file')
    # closeness_threshold
    parser.add_argument('--closeness_threshold', type=float, default=0.1, help='config file')
    parser.add_argument('--subset_img_id', type=int, default=None, help='config file')
    parser.add_argument('--prob_agg', type=str, default="prod", choices=["prod", "sum"], help='config file')
    parser.add_argument('--segmentation_method', type=str, default="default", choices=["default", "scene_graph"], help='config file')
    parser.add_argument('--dependency_topk', type=int, default=20, help='config file')
    parser.add_argument('--clustering_topk', type=int, default=500, help='config file')
    parser.add_argument("--add_sparse_index", action="store_true", help="config file")
    
    parser.add_argument('--retrieval_method', type=str, default="ours", help='config file')
    parser.add_argument('--hashes_per_table', type=int, default=5, help='config file')
    # num_tables
    parser.add_argument('--num_tables', type=int, default=100, help='config file')
    
    
    args = parser.parse_args()
    return args

import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")


def construct_qrels(dataset_name, queries, cached_img_idx, img_idx_ls, query_count):
    qrels = {}
    # if query_count < 0:
    #     query_count = 
    
    for idx in range(len(queries)):
        curr_img_idx = img_idx_ls[idx]
        cached_idx = cached_img_idx.index(curr_img_idx)
        qrels[str(idx+1)] = {str(cached_idx+1): 2}
    q_idx_ls = list(range(len(queries)))
    if query_count > 0:
        if dataset_name == "crepe":
            subset_q_idx_ls = [q_idx_ls[idx] for idx in range(query_count)] 
        else:
            subset_q_idx_ls = random.sample(q_idx_ls, query_count)
        
        subset_q_idx_ls = sorted(subset_q_idx_ls)
        
        subset_qrels = {str(key_id + 1): qrels[str(subset_q_idx_ls[key_id] + 1)] for key_id in range(len(subset_q_idx_ls))}
    
        qrels = subset_qrels
        
        queries = [queries[idx] for idx in subset_q_idx_ls]
    else:
        subset_q_idx_ls = q_idx_ls #list(qrels.keys())
        
    return qrels, queries, subset_q_idx_ls

if __name__ == "__main__":       
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

    args = parse_args()
    args.is_img_retrieval = args.dataset_name in image_retrieval_datasets
    set_rand_seed(args.random_seed)

    # args.query_concept = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
    # model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224').to(device)
    if args.is_img_retrieval:
        if args.model_name == "default":
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
            # processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            raw_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            # processor =  lambda images: raw_processor(images=images, return_tensors="pt", padding=False, do_resize=False, do_center_crop=False)["pixel_values"]
            processor =  lambda images: raw_processor(images=images, return_tensors="pt")["pixel_values"]
            text_processor =  lambda text: raw_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            img_processor =  lambda images: raw_processor(images=images, return_tensors="pt")["pixel_values"]
            model = model.eval()
        elif args.model_name == "blip":
            from lavis.models import load_model_and_preprocess
            model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
            text_processor = lambda text: txt_processors["eval"](text)
            processor =  vis_processors# lambda images: vis_processors(images=images, return_tensors="pt")["pixel_values"]
        
        model = model.eval()
    else:
        # if args.dataset_name not in image_retrieval_datasets:
        # if not args.is_img_retrieval:
        # text_model = models.clip_model(text_processor, model, device)
        if args.model_name == "default":
            print("start loading distill-bert model")
            if not os.path.exists("output/msmarco-distilbert-base-tas-b.pkl"):
                text_model = models.SentenceBERT("msmarco-distilbert-base-tas-b", prefix = sparse_prefix, suffix=sparse_suffix)
                utils.save(text_model, "output/msmarco-distilbert-base-tas-b.pkl")
            else:
                text_model = utils.load("output/msmarco-distilbert-base-tas-b.pkl")
        # elif args.model_name == "phi":
        #     text_model = models.ms_phi(prefix=sparse_prefix, suffix=sparse_suffix)
        elif args.model_name == "llm":
            # text_model = models.LlmtoVec(("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp","McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"),
            #     ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised","McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"))
            text_model = models.LlmtoVec(("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"),
                ("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised", "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised"))
        else:
            raise ValueError("Invalid model name")
        # text_model = AutoModelForCausalLM.from_pretrained(
        #             "microsoft/Phi-3-mini-4k-instruct", 
        #             device_map="cuda", 
        #             torch_dtype="auto", 
        #             trust_remote_code=True, 
        #         )
        text_retrieval_model = DRES(batch_size=16)
        # retriever = EvaluateRetrieval(text_model, score_function="cos_sim") # or "cos_sim" for cosine similarity

        # text_processor = AutoProcessor.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")
        # model = models.SentenceBERT("msmarco-distilbert-base-tas-b")
        # model = model.eval()

    
    full_data_path = os.path.join(args.data_path, args.dataset_name)
    if args.dataset_name.startswith("crepe"):
        full_data_path = os.path.join(args.data_path, "crepe")
    
    query_path = os.path.dirname(os.path.realpath(__file__))
    
    if not os.path.exists(full_data_path):
        os.makedirs(full_data_path)
    
    
    # origin_corpus = None
    bboxes_ls = None
    grouped_sub_q_ids_ls = None
    bboxes_overlap_ls = None
    img_file_name_ls = None
    
    # , bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls
    
    if args.dataset_name == "flickr":
        queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls = load_flickr_dataset_full(full_data_path, full_data_path, subset_img_id=args.subset_img_id)
        
        img_idx_ls, img_file_name_ls = load_other_flickr_images(full_data_path, query_path, img_idx_ls, img_file_name_ls, total_count = args.total_count)
        
        # filename_ls, raw_img_ls, img_ls = read_images_from_folder(os.path.join(full_data_path, "flickr30k-images/"), total_count = args.total_count)

        # filename_cap_mappings = read_image_captions(os.path.join(full_data_path, "results_20130124.token"))    
        # args.algebra_method=one
    elif args.dataset_name == "mscoco":
        # queries, img_file_name_ls, sub_queries_ls, img_idx_ls = load_sharegpt4v_datasets(full_data_path, full_data_path)
        queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls=load_mscoco_120k_datasets_from_cached_files(full_data_path, full_data_path)
        img_idx_ls, img_file_name_ls = load_other_sharegpt4v_mscoco_images(full_data_path, img_idx_ls, img_file_name_ls, total_count = args.total_count)
    
    elif args.dataset_name == "mscoco_40k":
        queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls= load_mscoco_datasets_from_cached_files(full_data_path, full_data_path)
        
        # queries, img_file_name_ls, sub_queries_ls, img_idx_ls = load_sharegpt4v_datasets(full_data_path, full_data_path)
        # img_idx_ls, img_file_name_ls = load_other_sharegpt4v_mscoco_images(full_data_path, img_idx_ls, img_file_name_ls, total_count = args.total_count)
    
    elif args.dataset_name == "AToMiC":
        load_atom_datasets(full_data_path)
    
    elif args.dataset_name == "crepe":
        queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls = load_crepe_datasets(full_data_path, query_path, subset_img_id=args.subset_img_id)
        # queries, raw_img_ls, sub_queries_ls, img_idx_ls = load_crepe_datasets_full(full_data_path, query_path)
        img_idx_ls, img_file_name_ls = load_other_crepe_images(full_data_path, query_path, img_idx_ls, img_file_name_ls, total_count = args.total_count)
        # args.algebra_method=two
        
    elif args.dataset_name == "crepe_full":
        # queries, raw_img_ls, sub_queries_ls, img_idx_ls = load_crepe_datasets(full_data_path, query_path)
        queries, img_file_name_ls, sub_queries_ls, img_idx_ls = load_crepe_datasets_full(full_data_path, query_path)
        img_idx_ls, img_file_name_ls = load_other_crepe_images(full_data_path, query_path, img_idx_ls, img_file_name_ls, total_count = args.total_count)
        # args.algebra_method=two
        
    elif args.dataset_name == "trec-covid":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset_name)
        data_path = util.download_and_unzip(url, full_data_path)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")       
        full_query_key_ls = [str(idx + 1) for idx in range(len(queries))]
        try:
            queries= read_queries_from_file(os.path.join(full_data_path, "queries.jsonl")) #, subset_img_id=args.subset_img_id)
        except:
            pass
        query_key_ls = list(queries.keys())
        # query_key_ls = random.sample(full_query_key_ls, 100)
        query_key_ls = sorted(query_key_ls)
        sub_queries_ls, idx_to_rid = decompose_queries_into_sub_queries(queries, data_path, query_key_ls=query_key_ls)
        print(sub_queries_ls)
        
        subset_file_name = f"output/{args.dataset_name}_subset_{args.total_count}.txt"
        if False: #os.path.exists(subset_file_name):
            corpus, qrels = utils.load(subset_file_name)
        else:        
            corpus, qrels = subset_corpus(corpus, qrels, args.total_count)
            utils.save((corpus, qrels), subset_file_name)
        
        qrels = {key: qrels[idx_to_rid[key]] for key in sub_queries_ls if not check_empty_mappings(qrels[idx_to_rid[key]])}
        
        if len(qrels) == 0:
            print("no valid queries, exit!")
            exit(1)
        
        origin_corpus = None #copy.copy(corpus)
        corpus, qrels = convert_corpus_to_concepts_txt(corpus, qrels)
        query_key_idx_ls = [full_query_key_ls.index(key) for key in query_key_ls]
        grouped_sub_q_ids_ls = group_dependent_segments_seq_all(queries, sub_queries_ls, full_data_path, query_key_idx_ls, query_key_ls=query_key_ls) # [None for _ in range(len(queries))]
        # args.algebra_method=three
        queries = [queries[key] for key in query_key_ls]
        # filename_ls, raw_img_ls, img_ls = read_images_from_folder(os.path.join(full_data_path, "crepe/"))
        # filename_cap_mappings = read_image_captions(os.path.join(full_data_path, "crepe/crepe_captions.txt"))
    elif args.dataset_name == "hotpotqa":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset_name)
        data_path = util.download_and_unzip(url, full_data_path)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")       
        filter_queries_with_gt(full_data_path, args, queries)
        queries = {key:queries[key] for key in qrels}
        full_query_key_ls = list(queries.keys())
        if args.query_count > 0:
            query_key_ls = random.sample(full_query_key_ls, args.query_count)
            query_key_ls = sorted(query_key_ls)
        else:
            query_key_ls = full_query_key_ls
        print("all query key list::", query_key_ls)
        query_hash = utils.hashfn(query_key_ls)
        print("query hash::",query_hash)
        # queries, sub_queries_ls, idx_to_rid = read_queries_with_sub_queries_file(os.path.join(full_data_path, "queries_with_sub0.jsonl"), subset_img_id=args.subset_img_id)
        sub_queries_ls, idx_to_rid = decompose_queries_into_sub_queries(queries, data_path,query_hash=query_hash, query_key_ls=query_key_ls)
        
        print(sub_queries_ls)
        
        subset_file_name = f"output/{args.dataset_name}_subset_{args.total_count}.txt"
        if False: #os.path.exists(subset_file_name):
            corpus, qrels = utils.load(subset_file_name)
        else:        
            corpus, qrels = subset_corpus2(corpus, qrels, args.total_count, idx_to_rid)
            utils.save((corpus, qrels), subset_file_name)
        
        new_qrels = {}
        for key in sub_queries_ls:
            if not check_empty_mappings(qrels[idx_to_rid[key]]) and idx_to_rid[key] in qrels:
                new_qrels[key] = qrels[idx_to_rid[key]]
        
        qrels = new_qrels #{key: qrels[idx_to_rid[key]] for key in sub_queries_ls if not check_empty_mappings(qrels[idx_to_rid[key]]) and idx_to_rid[key] in qrels}
        queries = {key: queries[idx_to_rid[key]] for key in sub_queries_ls if not check_empty_mappings(qrels[key]) and key in qrels}
        if len(qrels) == 0:
            print("no valid queries, exit!")
            exit(1)
        
        origin_corpus = None #copy.copy(corpus)
        corpus, qrels = convert_corpus_to_concepts_txt(corpus, qrels)
        # grouped_sub_q_ids_ls = [None for _ in range(len(queries))]
        grouped_sub_q_ids_ls = group_dependent_segments_seq_all(queries, sub_queries_ls, full_data_path, query_hash=query_hash) # [None for _ in range(len(queries))]
    
    if args.is_img_retrieval:
        # if not args.query_concept:    
        #     patch_count_ls = [4, 8, 16, 32, 64, 128]
        # else:
            # patch_count_ls = [4, 8, 16, 32, 64, 128]
            if args.dataset_name.startswith("crepe"):
                patch_count_ls = [4,16,64]
                # patch_count_ls = [32]
            elif args.dataset_name.startswith("mscoco"):
                # patch_count_ls = [4, 8, 16, 64, 128]
                patch_count_ls = [4, 8, 16, 64]
                # patch_count_ls = [4, 16, 64]
            else:
                patch_count_ls = [4, 8, 16, 64]
                
            if args.segmentation_method == "scene_graph":
                patch_count_ls = [1, 4, 8, 16, 32]
    else:
        # patch_count_ls = [8, 24, 32]
        patch_count_ls = [1, 16, 8, 4, 32]
        # patch_count_ls = [1]
        # patch_count_ls = [32]
    
    if args.is_img_retrieval:
        samples_hash = obtain_sample_hash(img_idx_ls, img_file_name_ls)
        # cached_img_idx_ls, image_embs, patch_activations, masks, bboxes, img_for_patch
        # if args.save_mask_bbox:
        cached_img_ls, img_emb, patch_emb_ls, _, bboxes_ls, img_per_patch_ls = convert_samples_to_concepts_img(args, samples_hash, model, img_file_name_ls, img_idx_ls, processor, device, patch_count_ls=patch_count_ls,save_mask_bbox=args.save_mask_bbox)
        # else:
        #     cached_img_ls, img_emb, patch_emb_ls, img_per_patch_ls = convert_samples_to_concepts_img(args, samples_hash, model, img_file_name_ls, img_idx_ls, processor, device, patch_count_ls=patch_count_ls,save_mask_bbox=args.save_mask_bbox)
            
        

    elif args.dataset_name in text_retrieval_datasets:
        samples_hash,(img_emb, img_sparse_index), patch_emb_ls, bboxes_ls = convert_samples_to_concepts_txt(args, text_model, corpus, device, patch_count_ls=patch_count_ls)
        
        # img_emb = text_model.encode_corpus(corpus)
        # if args.img_concept:
        #     _,img_per_patch_ls, patch_emb_ls = generate_patch_ids_ls(patch_emb_ls)
    else:
        print("Invalid dataset name, exit!")
        exit(1)
    print("sample hash::", samples_hash)
    patch_emb_by_img_ls = patch_emb_ls
    if args.img_concept:
        # if args.is_img_retrieval:
        patch_emb_by_img_ls, bboxes_ls = reformat_patch_embeddings(patch_emb_ls, None, img_emb, bbox_ls=bboxes_ls)
        # else:
        #     patch_emb_by_img_ls, bboxes_ls = reformat_patch_embeddings(patch_emb_ls, img_per_patch_ls, img_emb, bbox_ls=bboxes_ls)
        
        # if args.is_img_retrieval:

        
        
    sample_patch_ids_to_cluster_id_mappings = None
    if args.search_by_cluster:
        if args.img_concept:
            
            # patch_clustering_info_cached_file = get_clustering_res_file_name(args, patch_count_ls)
            
            patch_clustering_info_cached_file = get_dessert_clustering_res_file_name(samples_hash, patch_count_ls)
            
            if not os.path.exists(patch_clustering_info_cached_file):
            
                centroids =sampling_and_clustering(patch_emb_by_img_ls, clustering_count_ratio=args.closeness_threshold)
                # centroids = torch.zeros([1, patch_emb_by_img_ls[-1].shape[-1]])
                # hashes_per_table: int, num_tables
                max_patch_count = max([len(patch_emb_by_img_ls[idx]) for idx in range(len(patch_emb_by_img_ls))])
                retrieval_method = DocRetrieval(max_patch_count, args.hashes_per_table, args.num_tables, patch_emb_by_img_ls[-1].shape[-1], centroids)

                for idx in tqdm(range(len(patch_emb_by_img_ls)), desc="add doc"):
                    retrieval_method.add_doc(patch_emb_by_img_ls[idx], idx)
                
                # utils.save(retrieval_method, "output/retrieval_method.pkl")
                utils.save(retrieval_method, patch_clustering_info_cached_file)
            else:
                retrieval_method = utils.load(patch_clustering_info_cached_file)
            # cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls
            # cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls,cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, cluster_sub_X_cat_patch_ids_ls, sample_patch_ids_to_cluster_id_mappings = clustering_img_patch_embeddings(patch_emb_by_img_ls, args.dataset_name + "_" + str(args.total_count), patch_emb_ls, closeness_threshold=args.closeness_threshold)
            
            
            
            # if False: #os.path.exists(patch_clustering_info_cached_file):
            #     cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls,cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, cluster_sub_X_cat_patch_ids_ls = utils.load(patch_clustering_info_cached_file)
            # else: 
            #     utils.save((cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls,cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, cluster_sub_X_cat_patch_ids_ls, sample_patch_ids_to_cluster_id_mappings), patch_clustering_info_cached_file)
        else:
            cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_sample_ids_ls = clustering_img_embeddings(img_emb)
    
    if args.img_concept:
        bboxes_overlap_ls, clustering_nbs_mappings = init_bbox_nbs(args, patch_count_ls, samples_hash, bboxes_ls, patch_emb_by_img_ls, sample_patch_ids_to_cluster_id_mappings)
    
        # else:
        #     patch_emb_by_img_ls = reformat_patch_embeddings_txt(patch_emb_ls, img_emb)
    sparse_sim_scores = None
    if args.is_img_retrieval:
        # if args.dataset_name == "flickr":
        #     qrels = construct_qrels(filename_ls, query_count=args.query_count)
        # else:
            qrels, queries, subset_q_idx = construct_qrels(args.dataset_name, queries, cached_img_ls, img_idx_ls, query_count=args.query_count)
            if args.query_count > 0:
                sub_queries_ls = [sub_queries_ls[idx] for idx in subset_q_idx]
                grouped_sub_q_ids_ls = [grouped_sub_q_ids_ls[idx] for idx in subset_q_idx]
            print("sub_q_index::", subset_q_idx)
            print("qrels::", qrels)
    
    
    
    if args.is_img_retrieval:
        if args.query_concept:
            # if not args.dataset_name.startswith("crepe"):
            #     queries = [filename_cap_mappings[file] for file in filename_ls]
            #     sub_queries_ls = decompose_queries_by_keyword(args.dataset_name, queries)
            #     full_sub_queries_ls = [sub_queries_ls[idx] + [[queries[idx]]] for idx in range(len(sub_queries_ls))]
            # else:
                # sub_queries_ls = decompose_queries_by_clauses(queries)
            # full_sub_queries_ls = sub_queries_ls
            full_sub_queries_ls = [sub_queries_ls[idx] + [[queries[idx]]] for idx in range(len(sub_queries_ls))]
                # full_sub_queries_ls = [[sub_queries_ls[idx]] for idx in range(len(sub_queries_ls))]
            text_emb_ls = embed_queries_ls(full_sub_queries_ls, text_processor, model, device)
            # text_emb_ls = embed_queries(filename_ls, filename_cap_mappings, text_processor, model, device)
        else:
            # if args.dataset_name == "flickr":
            #     text_emb_ls = embed_queries(filename_ls, filename_cap_mappings, text_processor, model, device)
            # else:
                text_emb_ls = embed_queries_with_input_queries(queries, text_processor, model, device)
    else:
        if not args.query_concept:
            # text_emb_ls = text_model.encode_queries(queries, convert_to_tensor=True)
            text_emb_ls, query_sparse_index = construct_dense_or_sparse_encodings_queries(queries, text_model, args.add_sparse_index)
                    
        else:
            _, query_sparse_index = construct_dense_or_sparse_encodings_queries(queries, text_model, args.add_sparse_index)
            full_sub_queries_ls = sub_queries_ls
            # full_sub_queries_ls = [sub_queries_ls[idx] + [[reformated_queries[idx]]] for idx in range(len(sub_queries_ls))]
            text_emb_ls = encode_sub_queries_ls(full_sub_queries_ls, text_model)
            text_emb_dense = text_model.encode_queries(queries, convert_to_tensor=True)
            text_emb_ls = [text_emb_ls[idx] + [text_emb_dense[idx].unsqueeze(0)] for idx in range(len(text_emb_ls))]
            
        
        if args.add_sparse_index:
            store_sparse_index(samples_hash, query_sparse_index, encoding_query = True)
            # text_emb_ls = text_retrieval_model.model.encode_queries(queries, convert_to_tensor=True)
    
            run_search_with_sparse_index(samples_hash)
            sparse_sim_scores = read_trec_run(samples_hash, len(queries), len(corpus))
    
    # retrieve_by_full_query(img_emb, text_emb_ls)
    
    # if args.is_img_retrieval:
    retrieval_model = DRES(batch_size=16, algebra_method=args.algebra_method, is_img_retrieval=args.is_img_retrieval, prob_agg=args.prob_agg, dependency_topk=args.dependency_topk)
    # else:
    #     retrieval_model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16, algebra_method=one)
    retriever = EvaluateRetrieval(retrieval_model, score_function="cos_sim") # or "cos_sim" for cosine similarity
    
    if args.query_concept:
        if args.is_img_retrieval:
            text_emb_ls = [[torch.cat(item) for item in items] for items in text_emb_ls]
    
    # if args.query_concept:
    if args.retrieval_method == "ours":
        if not args.img_concept:
            if not args.search_by_cluster:
                results=retrieve_by_embeddings(retriever, img_emb, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=None, grouped_sub_q_ids_ls=None, clustering_topk=args.clustering_topk, sparse_sim_scores=sparse_sim_scores)
            else:
                # results=retrieve_by_embeddings(retriever, img_emb, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, use_clustering=args.search_by_cluster, clustering_info=(cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_cat_patch_ids_ls, clustering_nbs_mappings), bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls, clustering_topk=args.clustering_topk, sparse_sim_scores=sparse_sim_scores)
                results=retrieve_by_embeddings(retriever, img_emb, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, use_clustering=args.search_by_cluster, bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls,doc_retrieval=retrieval_method)
        else:
            
            if not args.search_by_cluster:
                results=retrieve_by_embeddings(retriever, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls, clustering_topk=args.clustering_topk, sparse_sim_scores=sparse_sim_scores)
            else:
                # results=retrieve_by_embeddings(retriever, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, use_clustering=args.search_by_cluster, clustering_info=(cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_cat_patch_ids_ls, clustering_nbs_mappings), bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls, clustering_topk=args.clustering_topk, sparse_sim_scores=sparse_sim_scores)
                # grouped_sub_q_ids_ls, bboxes_overlap_ls, dependency_topk, device, prob_agg, is_img_retrieval
                results=retrieve_by_embeddings(retriever, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, use_clustering=args.search_by_cluster, bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, clustering_topk=args.clustering_topk, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls,doc_retrieval=retrieval_method, prob_agg=args.prob_agg, dependency_topk=args.dependency_topk, device=device, is_img_retrieval=args.is_img_retrieval, method=args.algebra_method)
    # elif args.retrieval_method == "llm_ranker":
    #     ranker = LLM_ranker(corpus)
    #     results = ranker.retrieval(queries)
    #     ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values, ignore_identical_ids=False)
    elif args.retrieval_method == "bm25":
        ranker = BuildIndex(samples_hash, corpus)
        t1 = time.time()
        results=ranker.retrieval(queries)
        t2 = time.time()
        print("retrieval time::", t2 - t1)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values, ignore_identical_ids=False)
    else:
        raise ValueError("Invalid retrieval method")
    
    final_res_file_name = utils.get_final_res_file_name(args, patch_count_ls)
    print("The results are stored at ", final_res_file_name)
    utils.save(results, final_res_file_name)
    
    # else:
    #     retrieve_by_embeddings(retriever, text_emb_ls, img_emb, qrels)
    
    # print(results_without_decomposition)