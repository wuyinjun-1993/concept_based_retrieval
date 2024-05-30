
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
from clustering import *
from main import *

image_retrieval_datasets = ["flickr", "AToMiC", "crepe"]

def draw_image_with_bbox(image, bbox_ls, filename="with_bbox"):
    img = image.copy()
    for box in bbox_ls:
        color = list(np.random.random(size=3) * 256)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    
    
    cv2.imwrite(filename + ".jpg", img)
    return img


def select_largest_cluster_and_visualize_bbox_ls(cluster_sample_count_ls, cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, bboxes_ls, raw_img_ls):
    
    cluster_sample_count_tensor = torch.tensor(cluster_sample_count_ls)
    max_cluster_id = torch.argmax(cluster_sample_count_tensor).item()
    
    curr_sample_ids = cluster_sample_ids_ls[max_cluster_id]
    curr_patch_ids = cluster_sub_X_patch_ids_ls[max_cluster_id]
    curr_granularity_ids = cluster_sub_X_granularity_ids_ls[max_cluster_id]
    
    
    sample_id_bbox_ls_mappings = {}
    sample_id_image_mappings = {}
    for sample_id, patch_id, granularity_id in zip(curr_sample_ids, curr_patch_ids, curr_granularity_ids):
        if sample_id in sample_id_bbox_ls_mappings and len(sample_id_bbox_ls_mappings[sample_id]) >3:
            continue
        
        curr_img = raw_img_ls[sample_id]
        curr_bbox = bboxes_ls[int(granularity_id)][sample_id][patch_id]
        if sample_id not in sample_id_bbox_ls_mappings:
            sample_id_bbox_ls_mappings[sample_id] = []
        sample_id_bbox_ls_mappings[sample_id].append(curr_bbox)
        sample_id_image_mappings[sample_id] = curr_img
        
        if len(sample_id_bbox_ls_mappings) > 10:
            break
        
    for sample_id, bbox_ls in sample_id_bbox_ls_mappings.items():
        draw_image_with_bbox(np.array(sample_id_image_mappings[sample_id]), bbox_ls, filename=f"sample_{sample_id}")
        
        
    
    

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

        filename_cap_mappings = read_flickr_image_captions(os.path.join(full_data_path, "results_20130124.token"))    
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
        patch_count_ls = [4, 8, 16, 32, 64, 128]
    
    if args.dataset_name in image_retrieval_datasets:
        img_emb, patch_emb_ls, masks_ls, bboxes_ls, img_per_patch_ls = convert_samples_to_concepts_img(args, model, raw_img_ls, processor, device, patch_count_ls=patch_count_ls)
        if args.search_by_cluster:
            cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_sample_unique_ids_ls, cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls = clustering_img_patch_embeddings(patch_emb_ls, bboxes_ls, img_per_patch_ls)
            # f"output/saved_patches_{method}_{n_patches}_{samples_hash}
            patch_clustering_info_cached_file = f"output/saved_cluster_info"
            utils.save((cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_sample_unique_ids_ls, cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls), patch_clustering_info_cached_file)
    else:
        img_emb = text_model.encode_corpus(corpus)
    
    select_largest_cluster_and_visualize_bbox_ls(cluster_sample_count_ls, cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, bboxes_ls, raw_img_ls)
    print()