from transformers import CLIPModel, AutoProcessor
import torch
from image_utils import *
import argparse
from sklearn.metrics import top_k_accuracy_score
from beir.retrieval.evaluation import EvaluateRetrieval
from retrieval_utils import *
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.dense.exact_search import one, two
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

image_retrieval_datasets = ["flickr", "AToMiC", "crepe", "crepe_full"]
text_retrieval_datasets = ["trec-covid"]




    
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
    



def parse_args():
    parser = argparse.ArgumentParser(description='CUB concept learning')
    parser.add_argument('--data_path', type=str, default="/data6/wuyinjun/", help='config file')
    parser.add_argument('--dataset_name', type=str, default="crepe", help='config file')
    parser.add_argument('--query_count', type=int, default=-1, help='config file')
    parser.add_argument('--query_concept', action="store_true", help='config file')
    parser.add_argument('--img_concept', action="store_true", help='config file')
    parser.add_argument('--total_count', type=int, default=500, help='config file')
    parser.add_argument("--parallel", action="store_true", help="config file")
    parser.add_argument("--search_by_cluster", action="store_true", help="config file")
    parser.add_argument('--algebra_method', type=str, default=one, help='config file')
    # closeness_threshold
    parser.add_argument('--closeness_threshold', type=float, default=0.1, help='config file')
    
    
    args = parser.parse_args()
    return args

def construct_qrels(filename_ls, query_count):
    qrels = {}
    if query_count < 0:
        query_count = len(filename_ls)
    
    for idx in range(query_count):
        qrels[str(idx+1)] = {str(idx+1): 2}
    
    return qrels

if __name__ == "__main__":       
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

    args = parse_args()
    args.is_img_retrieval = args.dataset_name in image_retrieval_datasets


    # args.query_concept = False
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
    # if args.dataset_name not in image_retrieval_datasets:
    if not args.is_img_retrieval:
        # text_model = models.clip_model(text_processor, model, device)
        text_model = models.SentenceBERT("msmarco-distilbert-base-tas-b")
        text_retrieval_model = DRES(text_model, batch_size=16, algebra_method=one)
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
    
    if args.dataset_name == "flickr":
        filename_ls, raw_img_ls, img_ls = read_images_from_folder(os.path.join(full_data_path, "flickr30k-images/"))

        filename_cap_mappings = read_image_captions(os.path.join(full_data_path, "results_20130124.token"))    
    elif args.dataset_name == "AToMiC":
        load_atom_datasets(full_data_path)
    
    elif args.dataset_name == "crepe":
        queries, raw_img_ls, sub_queries_ls, img_idx_ls = load_crepe_datasets(full_data_path, query_path)
        # queries, raw_img_ls, sub_queries_ls, img_idx_ls = load_crepe_datasets_full(full_data_path, query_path)
        img_idx_ls, raw_img_ls = load_other_crepe_images(full_data_path, query_path, img_idx_ls, raw_img_ls, total_count = args.total_count)
        args.algebra_method=two
        
    elif args.dataset_name == "crepe_full":
        # queries, raw_img_ls, sub_queries_ls, img_idx_ls = load_crepe_datasets(full_data_path, query_path)
        queries, raw_img_ls, sub_queries_ls, img_idx_ls = load_crepe_datasets_full(full_data_path, query_path)
        img_idx_ls, raw_img_ls = load_other_crepe_images(full_data_path, query_path, img_idx_ls, raw_img_ls, total_count = args.total_count)
        args.algebra_method=two
        
    elif args.dataset_name == "trec-covid":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset_name)
        data_path = util.download_and_unzip(url, full_data_path)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")       
        
        queries, sub_queries_ls = read_queries_with_sub_queries_file(os.path.join(full_data_path, "queries_with_subs.jsonl"))
        
        subset_file_name = f"output/{args.dataset_name}_subset_{args.total_count}.txt"
        if False: #os.path.exists(subset_file_name):
            corpus, qrels = utils.load(subset_file_name)
        else:        
            corpus, qrels = subset_corpus(corpus, qrels, args.total_count)
            utils.save((corpus, qrels), subset_file_name)
        
        qrels = {key: qrels[key] for key in sub_queries_ls}
        origin_corpus = None #copy.copy(corpus)
        corpus, qrels = convert_corpus_to_concepts_txt(corpus, qrels)
        args.algebra_method=one
        # filename_ls, raw_img_ls, img_ls = read_images_from_folder(os.path.join(full_data_path, "crepe/"))
        # filename_cap_mappings = read_image_captions(os.path.join(full_data_path, "crepe/crepe_captions.txt"))
    
    if args.is_img_retrieval:
        if not args.query_concept:    
            patch_count_ls = [4, 8]
        else:
            patch_count_ls = [4, 8, 16, 32, 64, 128]
    else:
        # patch_count_ls = [8, 16, 24, 32]
        patch_count_ls = [1, 16, 8, 4, 2]
    
    if args.is_img_retrieval:
        img_emb, patch_emb_ls, masks_ls, bboxes_ls, img_per_patch_ls = convert_samples_to_concepts_img(args, model, raw_img_ls, processor, device, patch_count_ls=patch_count_ls)
        

    elif args.dataset_name in text_retrieval_datasets:
        img_emb, patch_emb_ls = convert_samples_to_concepts_txt(args, text_model, corpus, device, patch_count_ls=patch_count_ls)
        # img_emb = text_model.encode_corpus(corpus)
        if args.img_concept:
            bboxes_ls,img_per_patch_ls, patch_emb_ls = generate_patch_ids_ls(patch_emb_ls)
    
    patch_emb_by_img_ls = patch_emb_ls
    if args.img_concept:
        # if args.is_img_retrieval:
        patch_emb_by_img_ls = reformat_patch_embeddings(patch_emb_ls, img_per_patch_ls, img_emb)
    
    if args.search_by_cluster:
        if args.img_concept:
            # cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls
            cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls,cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, cluster_sub_X_cat_patch_ids_ls = clustering_img_patch_embeddings(patch_emb_by_img_ls, args.dataset_name + "_" + str(args.total_count), patch_emb_ls, bboxes_ls, img_per_patch_ls, closeness_threshold=args.closeness_threshold)
            
            patch_clustering_info_cached_file = get_clustering_res_file_name(args, patch_count_ls)
            
            if False: #os.path.exists(patch_clustering_info_cached_file):
                cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls,cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, cluster_sub_X_cat_patch_ids_ls = utils.load(patch_clustering_info_cached_file)
            else: 
                utils.save((cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls,cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, cluster_sub_X_cat_patch_ids_ls), patch_clustering_info_cached_file)
        else:
            cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_sample_ids_ls = clustering_img_embeddings(img_emb)
    
    
    
        # else:
        #     patch_emb_by_img_ls = reformat_patch_embeddings_txt(patch_emb_ls, img_emb)
    
    if args.is_img_retrieval:
        if args.query_concept:
            if not args.dataset_name.startswith("crepe"):
                queries = [filename_cap_mappings[file] for file in filename_ls]
                sub_queries_ls = decompose_queries_by_keyword(args.dataset_name, queries)
                full_sub_queries_ls = [sub_queries_ls[idx] + [[queries[idx]]] for idx in range(len(sub_queries_ls))]
            else:
                # sub_queries_ls = decompose_queries_by_clauses(queries)
                full_sub_queries_ls = [sub_queries_ls[idx] + [[queries[idx]]] for idx in range(len(sub_queries_ls))]
                # full_sub_queries_ls = [[sub_queries_ls[idx]] for idx in range(len(sub_queries_ls))]
            text_emb_ls = embed_queries_ls(full_sub_queries_ls, text_processor, model, device)
            # text_emb_ls = embed_queries(filename_ls, filename_cap_mappings, text_processor, model, device)
        else:
            if args.dataset_name == "flickr":
                text_emb_ls = embed_queries(filename_ls, filename_cap_mappings, text_processor, model, device)
            else:
                text_emb_ls = embed_queries_with_input_queries(queries, text_processor, model, device)
    else:
        if not args.query_concept:
            text_emb_ls = text_retrieval_model.model.encode_queries(queries, convert_to_tensor=True)
        
        else:
            text_emb_ls = encode_sub_queries_ls(sub_queries_ls, text_retrieval_model.model)
            # text_emb_ls = text_retrieval_model.model.encode_queries(queries, convert_to_tensor=True)
    
    # retrieve_by_full_query(img_emb, text_emb_ls)
    if args.is_img_retrieval:
        if args.dataset_name == "flickr":
            qrels = construct_qrels(filename_ls, query_count=args.query_count)
        else:
            qrels = construct_qrels(queries, query_count=args.query_count)
    
    # if args.is_img_retrieval:
    retrieval_model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16, algebra_method=args.algebra_method)
    # else:
    #     retrieval_model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16, algebra_method=one)
    retriever = EvaluateRetrieval(retrieval_model, score_function="cos_sim") # or "cos_sim" for cosine similarity
    
    if args.query_concept:
        if args.is_img_retrieval:
            text_emb_ls = [[torch.cat(item) for item in items] for items in text_emb_ls]
    
    # if args.query_concept:
    if not args.img_concept:
        if not args.search_by_cluster:
            results=retrieve_by_embeddings(retriever, img_emb, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel)
        else:
            results=retrieve_by_embeddings(retriever, img_emb, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, use_clustering=args.search_by_cluster, clustering_info=(cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_cat_patch_ids_ls))
    else:
        
        if not args.search_by_cluster:
            results=retrieve_by_embeddings(retriever, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel)
        else:
            results=retrieve_by_embeddings(retriever, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, use_clustering=args.search_by_cluster, clustering_info=(cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_cat_patch_ids_ls))
    
    final_res_file_name = utils.get_final_res_file_name(args, patch_count_ls)
    print("The results are stored at ", final_res_file_name)
    utils.save(results, final_res_file_name)
    
    # else:
    #     retrieve_by_embeddings(retriever, text_emb_ls, img_emb, qrels)
    
    # print(results_without_decomposition)