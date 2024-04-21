from transformers import CLIPModel, AutoProcessor
import torch
from image_utils import *
import argparse
from sklearn.metrics import top_k_accuracy_score
from beir.retrieval.evaluation import EvaluateRetrieval
from retrieval_utils import *
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
from beir import LoggingHandler
from datasets import load_dataset
from datasets.download.download_config import DownloadConfig
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import time
# from beir.retrieval.models.clip_model import clip_model

image_retrieval_datasets = ["flickr", "AToMiC", "crepe"]


def convert_samples_to_concepts(args, model, images, processor, device, patch_count_ls = [32]):
    # samples: list[PIL.Image], labels, input_to_latent, input_processor, dataset_name, device: str = 'cpu'
    # cl = ConceptLearner(images, labels, vit_forward, processor, img_processor, args.dataset_name, device)
    cl = ConceptLearner(images, model, vit_forward, processor, args.dataset_name, device)
    # bbox x1,y1,x2,y2
    # image_embs, patch_activations, masks, bboxes, img_for_patch
    # n_patches, images=None, method="slic", not_normalize=False
    patch_emb_ls = []
    masks_ls = []
    img_per_batch_ls = []
    bboxes_ls = []
    for idx in range(len(patch_count_ls)):
        patch_count = patch_count_ls[idx]
        if idx == 0:
            curr_img_emb, patch_emb, masks, bboxes, img_per_patch = cl.get_patches(patch_count, images=images, method="slic", compute_img_emb=True)
        else:
            curr_img_emb, patch_emb, masks, bboxes, img_per_patch = cl.get_patches(patch_count, images=images, method="slic", compute_img_emb=False)
        if curr_img_emb is not None:
            img_emb = curr_img_emb
        patch_emb_ls.append(patch_emb)
        masks_ls.append(masks)
        img_per_batch_ls.append(img_per_patch)
        bboxes_ls.append(bboxes)
    return img_emb, patch_emb_ls, masks_ls, bboxes_ls, img_per_batch_ls

def reformat_patch_embeddings(patch_emb_ls, img_per_patch_ls, img_emb):
    img_per_patch_tensor = torch.tensor(img_per_patch_ls[0])
    max_img_id = torch.max(img_per_patch_tensor).item()
    patch_emb_curr_img_ls = []
    for idx in tqdm(range(max_img_id)):
        sub_patch_emb_curr_img_ls = []
        for sub_idx in range(len(patch_emb_ls)):
            patch_emb = patch_emb_ls[sub_idx]
            img_per_batch = img_per_patch_ls[sub_idx]
            img_per_patch_tensor = torch.tensor(img_per_batch)
            patch_emb_curr_img = patch_emb[img_per_patch_tensor == idx]
            sub_patch_emb_curr_img_ls.append(patch_emb_curr_img)
        sub_patch_emb_curr_img = torch.cat(sub_patch_emb_curr_img_ls, dim=0)
        patch_emb_curr_img = torch.cat([img_emb[idx].unsqueeze(0), sub_patch_emb_curr_img], dim=0)
        patch_emb_curr_img_ls.append(patch_emb_curr_img)
    return patch_emb_curr_img_ls
    
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
    if args.dataset_name not in image_retrieval_datasets:
        text_model = DRES(models.clip_model(text_processor, model, device), batch_size=16)
        # retriever = EvaluateRetrieval(text_model, score_function="cos_sim") # or "cos_sim" for cosine similarity

        # text_processor = AutoProcessor.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")
        # model = models.SentenceBERT("msmarco-distilbert-base-tas-b")
        # model = model.eval()
    
    
    full_data_path = os.path.join(args.data_path, args.dataset_name)
    
    query_path = os.path.dirname(os.path.realpath(__file__))
    
    if not os.path.exists(full_data_path):
        os.makedirs(full_data_path)
    
    
    if args.dataset_name == "flickr":
        filename_ls, raw_img_ls, img_ls = read_images_from_folder(os.path.join(full_data_path, "flickr30k-images/"))

        filename_cap_mappings = read_image_captions(os.path.join(full_data_path, "results_20130124.token"))    
    elif args.dataset_name == "AToMiC":
        load_atom_datasets(full_data_path)
    
    elif args.dataset_name == "crepe":
        queries, raw_img_ls, sub_queries_ls, img_idx_ls = load_crepe_datasets(full_data_path, query_path)
        img_idx_ls, raw_img_ls = load_other_crepe_images(full_data_path, query_path, img_idx_ls, raw_img_ls, total_count = args.total_count)
        
    elif args.dataset_name == "trec-covid":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset_name)
        data_path = util.download_and_unzip(url, full_data_path)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        # filename_ls, raw_img_ls, img_ls = read_images_from_folder(os.path.join(full_data_path, "crepe/"))
        # filename_cap_mappings = read_image_captions(os.path.join(full_data_path, "crepe/crepe_captions.txt"))
    
    if not args.query_concept:    
        patch_count_ls = [4, 8]
    else:
        patch_count_ls = [32, 64, 128]
    
    if args.dataset_name in image_retrieval_datasets:
        img_emb, patch_emb_ls, masks_ls, bboxes_ls, img_per_patch_ls = convert_samples_to_concepts(args, model, raw_img_ls, processor, device, patch_count_ls=patch_count_ls)
    else:
        img_emb = text_model.encode_corpus(corpus)

    
    if args.img_concept:
        patch_emb_by_img_ls = reformat_patch_embeddings(patch_emb_ls, img_per_patch_ls, img_emb)
    
    if args.query_concept:
        if not args.dataset_name == "crepe":
            queries = [filename_cap_mappings[file] for file in filename_ls]
            sub_queries_ls = decompose_queries_by_keyword(args.dataset_name, queries)
            full_sub_queries_ls = [[sub_queries_ls[idx], [queries[idx]]] for idx in range(len(sub_queries_ls))]
        else:
            # sub_queries_ls = decompose_queries_by_clauses(queries)
            full_sub_queries_ls = [[sub_queries_ls[idx], [queries[idx]]] for idx in range(len(sub_queries_ls))]
            # full_sub_queries_ls = [[sub_queries_ls[idx]] for idx in range(len(sub_queries_ls))]
        text_emb_ls = embed_queries_ls(full_sub_queries_ls, text_processor, model, device)
        # text_emb_ls = embed_queries(filename_ls, filename_cap_mappings, text_processor, model, device)
    else:
        if not args.dataset_name == "crepe":
            text_emb_ls = embed_queries(filename_ls, filename_cap_mappings, text_processor, model, device)
        else:
            text_emb_ls = embed_queries_with_input_queries(queries, text_processor, model, device)
    
    # retrieve_by_full_query(img_emb, text_emb_ls)
    if args.dataset_name == "flickr":
        qrels = construct_qrels(filename_ls, query_count=args.query_count)
    else:
        qrels = construct_qrels(queries, query_count=args.query_count)
    retrieval_model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)
    retriever = EvaluateRetrieval(retrieval_model, score_function="cos_sim") # or "cos_sim" for cosine similarity
    
    # if args.query_concept:
    t1 = time.time()
    if not args.img_concept:
        retrieve_by_embeddings(retriever, img_emb, text_emb_ls, qrels, query_count=args.query_count)
    else:
        retrieve_by_embeddings(retriever, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count)
    
    t2 = time.time()
    
    print(f"Time taken: {t2-t1:.2f}s")    
    
    # else:
    #     retrieve_by_embeddings(retriever, text_emb_ls, img_emb, qrels)
    
    # print(results_without_decomposition)