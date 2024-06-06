from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
from tqdm import tqdm
import os
import numpy as np
from PIL import Image

from skimage.measure import label, regionprops
from skimage.segmentation import slic
from dataclasses import dataclass
import PIL
import utils
import torchvision.transforms as transforms
from datasets import load_dataset
import pandas as pd
from retrieval_utils import decompose_single_query, decompose_single_query_ls, decompose_single_query_parition_groups
from scipy import ndimage
import cv2
from storage import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import gc
import pickle

@dataclass
class Patch:
    image: PIL.Image
    bbox: tuple
    patch: PIL.Image

def vit_forward(imgs, model, masks=None):
    # inputs = processor(imgs, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # Select the CLS token embedding from the last hidden layer
        # return model(pixel_values=imgs).last_hidden_state[:, 0, :]
        return model.get_image_features(pixel_values=imgs, output_hidden_states=True)

def filter_atom_images_by_langs(dataset, target_count = 10000):
    count = 0
    
    selected_dataset_ls = []
    
    for idx in range(len(dataset)):
        if len(dataset[idx]['language']) == 1 and 'en' in dataset[idx]['language']:
            selected_dataset_ls.append(dataset[idx])
            count += 1
        if count >= target_count:
            break        
            
    return selected_dataset_ls

def determine_n_patches(partition_strategy, depth):
    if partition_strategy == "one":
        if depth == 0:
            n_patches = 32
        # elif depth == 1:
        #     n_patches = 4
        else:
            n_patches = 2
    elif partition_strategy == "two":
        if depth == 0:
            n_patches = 32
        elif depth == 1:
            n_patches = 4
        else:
            n_patches = 2

    elif partition_strategy == "three":
        if depth == 0:
            n_patches = 32
        else:
            n_patches = 3
            
    elif partition_strategy == "four":
        if depth == 0:
            n_patches = 16
        elif depth == 1:
            n_patches = 4
        else:
            n_patches = 2

    return n_patches

def load_atom_datasets(data_path):
    image_ls = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train')
    
    text_ls = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2", split='train')
    
    # selected_dataset = filter_atom_images_by_langs(dataset) 

def load_flickr_dataset(data_path, query_path):
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all4.csv")
    
    # img_caption_file_name= os.path.join(query_path, "sub_queries.csv")
    img_caption_file_name= os.path.join(query_path, "sub_queries2.csv")


    img_folder = os.path.join(data_path, "flickr30k-images/")
    # img_folder2 = os.path.join(data_path, "VG_100K_2/")

    caption_pd = pd.read_csv(img_caption_file_name)
    
    # img_ls = []
    img_idx_ls = []
    caption_ls = []
    sub_caption_ls = []
    img_file_name_ls = []
    for idx in tqdm(range(len(caption_pd))):
        image_idx = caption_pd.iloc[idx]['image_id']
        if image_idx in img_idx_ls:
            continue
        
        full_img_file_name = os.path.join(img_folder, str(image_idx))
        # if not os.path.exists(full_img_file_name):
        #     full_img_file_name = os.path.join(img_folder2, str(image_idx) + ".jpg")
        if not os.path.exists(full_img_file_name):
            continue
        img_file_name_ls.append(full_img_file_name)
        
        # img = Image.open(full_img_file_name)
        # img = img.convert('RGB')
        caption = caption_pd.iloc[idx]['caption']
        # sub_caption_str = caption_pd.iloc[idx]['caption_triples']
        sub_caption_str = caption_pd.iloc[idx]['caption_triples_ls']
        
        # sub_captions = decompose_single_query(sub_caption_str)
        sub_captions = decompose_single_query_ls(sub_caption_str)
        print(sub_captions)
        # img_ls.append(img)
        img_idx_ls.append(image_idx)
        caption_ls.append(caption)
        sub_caption_ls.append(sub_captions)
    return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls

def load_sharegpt4v_datasets(data_path, query_path):
    # img_caption_file_name = query_path
    # with open(img_caption_file_name, 'rb') as f:
    # caption_pd = pickle.load(f)
    caption_pd = pd.read_csv(os.path.join(query_path, "sharegpt_query_20.csv"))
    img_file_name_ls = []
    img_idx_ls = []
    caption_ls = []
    sub_caption_ls = []
    for idx in range(len(caption_pd)):
        image_idx = caption_pd.iloc[idx]['id']
        if image_idx in img_idx_ls:
            continue

        img_path = caption_pd.iloc[idx]['image']
        #image_dir is the directory in which the root folder for the main image files are located, in this case in train2017 folder
        # image_dir = '/content/unzipped_images/train2017/train2017/'
        img_final_path = os.path.join(data_path, "images/train2017/", img_path)
        caption = caption_pd.iloc[idx]['caption_sharegpt4v']
        sub_caption_str = caption_pd.iloc[idx]['caption_triples_ls']
        sub_captions = decompose_single_query_ls(sub_caption_str)
        img_file_name_ls.append(img_final_path)
        img_idx_ls.append(image_idx)
        caption_ls.append(caption)
        sub_caption_ls.append(sub_captions)
    return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls


def load_other_sharegpt4v_mscoco_images(dataset_path, img_idx_ls, img_file_name_ls, total_count):
    #query_path = '/content/drive/MyDrive/'
    #img_caption_file_name = os.path.join(query_path, "sharegpt4v_mscoco_image_paths.pkl")
    # img_caption_file_name = dataset_path
    img_caption_file_name = os.path.join(dataset_path, "mscoco_120k.pkl")
    with open(img_caption_file_name, 'rb') as f:
      caption_pd = pickle.load(f)
    if total_count > 0 and len(img_file_name_ls) >= total_count:
        return img_idx_ls, img_file_name_ls       
    for idx in tqdm(range(len(caption_pd))):
        image_idx = caption_pd.iloc[idx]['id']
        if image_idx in img_idx_ls:
            continue
        
        img_path = caption_pd.iloc[idx]['image']
        #image_dir is the directory in which the root folder for the main image files are located, in this case in train2017 folder
        # image_dir = '/content/unzipped_images/train2017/train2017/'
        image_dir = os.path.join(dataset_path, "images/train2017/")
        img_final_path = os.path.join(image_dir, img_path)
        img_file_name_ls.append(img_final_path)
        img_idx_ls.append(image_idx)
        if total_count > 0 and len(img_file_name_ls) >= total_count:
            break
    
    return img_idx_ls, img_file_name_ls


def load_crepe_datasets(data_path, query_path, subset_img_id=None):
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all4.csv")
    
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all6.csv")
    img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all6.csv")
    
    with open("prod_hard_negatives/selected_img_id_ls", "rb") as f:
        selected_img_id_ls = pickle.load(f)

    img_folder = os.path.join(data_path, "VG_100K/")
    img_folder2 = os.path.join(data_path, "VG_100K_2/")

    caption_pd = pd.read_csv(img_caption_file_name)
    
    # img_ls = []
    img_idx_ls = []
    caption_ls = []
    sub_caption_ls = []
    img_file_name_ls = []
    all_grouped_sub_q_ids_ls = []
    for idx in range(len(caption_pd)):
        image_idx = caption_pd.iloc[idx]['image_id']
        if image_idx in img_idx_ls:
            continue
        if image_idx not in selected_img_id_ls:
            continue
        
        full_img_file_name = os.path.join(img_folder, str(image_idx) + ".jpg")
        if not os.path.exists(full_img_file_name):
            full_img_file_name = os.path.join(img_folder2, str(image_idx) + ".jpg")
            
        
        # img = Image.open(full_img_file_name)
        # img = img.convert('RGB')
        caption = caption_pd.iloc[idx]['caption']
        # sub_caption_str = caption_pd.iloc[idx]['caption_triples']
        sub_caption_str = caption_pd.iloc[idx]['caption_triples_ls']
        
        # sub_captions = decompose_single_query(sub_caption_str)
        sub_captions = decompose_single_query_ls(sub_caption_str)
        
        query_paritions_str = caption_pd.iloc[idx]['groups']
        grouped_sub_q_ids_ls = decompose_single_query_parition_groups(sub_captions, query_paritions_str)
        print(sub_captions)
        # img_ls.append(img)
        img_idx_ls.append(image_idx)
        caption_ls.append(caption)
        sub_caption_ls.append(sub_captions)
        img_file_name_ls.append(full_img_file_name)
        all_grouped_sub_q_ids_ls.append(grouped_sub_q_ids_ls)
    # return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls
    if subset_img_id is None:
        return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls, all_grouped_sub_q_ids_ls
    else:
        print(sub_caption_ls[subset_img_id])
        return [caption_ls[subset_img_id]], [img_file_name_ls[subset_img_id]], [sub_caption_ls[subset_img_id]], [img_idx_ls[subset_img_id]], [all_grouped_sub_q_ids_ls[subset_img_id]]

def replace_comma_with_vertical_line(caption_pd, file_name):
    for idx in range(len(caption_pd)):
        caption_pd.iloc[idx]['caption_triples'] = caption_pd.iloc[idx]['caption_triples'].replace(",","|")
        
    caption_pd.to_csv(file_name, index=False)


def load_crepe_datasets_full(data_path, query_path):
    img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all.csv")

    split_caption_file_name=os.path.join(query_path, "prod_hard_negatives/split2.csv")
    
    img_folder = os.path.join(data_path, "VG_100K/")
    img_folder2 = os.path.join(data_path, "VG_100K_2/")

    caption_pd = pd.read_csv(img_caption_file_name)
    
    split_file_pd = pd.read_csv(split_caption_file_name)
    
    # replace_comma_with_vertical_line(split_file_pd, os.path.join(query_path, "prod_hard_negatives/split2.csv"))
    
    # img_ls = []
    img_idx_ls = []
    caption_ls = []
    sub_caption_ls = []
    img_file_name_ls=[]
    split_file_pd['caption'] = split_file_pd['caption'].apply(lambda x: x.strip())
    for idx in range(len(caption_pd)):
        image_idx = caption_pd.iloc[idx]['image_id']
        if image_idx in img_idx_ls:
            continue
        
        full_img_file_name = os.path.join(img_folder, str(image_idx) + ".jpg")
        if not os.path.exists(full_img_file_name):
            full_img_file_name = os.path.join(img_folder2, str(image_idx) + ".jpg")
            
        
        # img = Image.open(full_img_file_name)
        # img = img.convert('RGB')
        caption = caption_pd.iloc[idx]['caption'].strip()
        img_file_name_ls.append(full_img_file_name)
        
        if caption in split_file_pd['caption'].values and image_idx not in img_idx_ls:
            sub_caption_str=split_file_pd[split_file_pd['caption'] == caption]["caption_triples"].values[0]
            # sub_caption_str=sub_caption_str.replace(",","|")
            # sub_caption_str = caption_pd.iloc[idx]['caption_triples']
        
            sub_captions = decompose_single_query_ls(sub_caption_str)
            # img_ls.append(img)
            img_idx_ls.append(image_idx)
            caption_ls.append(caption)
            sub_caption_ls.append(sub_captions)
    return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls



def load_other_crepe_images(data_path, query_path, img_idx_ls, img_file_name_ls, total_count=500):
    
    img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all.csv")
    img_folder = os.path.join(data_path, "VG_100K/")
    img_folder2 = os.path.join(data_path, "VG_100K_2/")

    with open("prod_hard_negatives/selected_img_id_ls", "rb") as f:
        selected_img_id_ls = pickle.load(f)

    caption_pd = pd.read_csv(img_caption_file_name)
    if total_count > 0 and len(img_file_name_ls) >= total_count:
        return img_idx_ls, img_file_name_ls          
    for idx in range(len(caption_pd)):
        image_idx = caption_pd.iloc[idx]['image_id']
        if image_idx in img_idx_ls:
            continue
        
        if not image_idx in selected_img_id_ls:
            continue
        
        full_img_file_name = os.path.join(img_folder, str(image_idx) + ".jpg")
        if not os.path.exists(full_img_file_name):
            full_img_file_name = os.path.join(img_folder2, str(image_idx) + ".jpg")
            
        
        # img = Image.open(full_img_file_name)
        # img = img.convert('RGB')
        # caption = caption_pd.iloc[idx]['caption']
        # sub_caption_str = caption_pd.iloc[idx]['caption_triples']
        
        # sub_captions = decompose_single_query(sub_caption_str)
        # img_ls.append(img)
        img_file_name_ls.append(full_img_file_name)
        img_idx_ls.append(image_idx)
        if total_count > 0 and len(img_file_name_ls) >= total_count:
            break
    
    return img_idx_ls, img_file_name_ls
        
    
def load_other_flickr_images(data_path, query_path, img_idx_ls, img_file_name_ls, total_count=500):
    
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all.csv")
    
    filename_cap_mappings = read_flickr_image_captions(os.path.join(data_path, "results_20130124.token"))    

    # img_folder = os.path.join(data_path, "VG_100K/")
    img_folder = os.path.join(data_path, "flickr30k-images/")
    # img_folder2 = os.path.join(data_path, "VG_100K_2/")

    # caption_pd = pd.read_csv(img_caption_file_name)
    if total_count > 0 and len(img_file_name_ls) >= total_count:
        return img_idx_ls, img_file_name_ls          
    # for idx in range(len(caption_pd)):
    for image_idx in tqdm(filename_cap_mappings):
        # image_idx = caption_pd.iloc[idx]['image_id']
        if image_idx in img_idx_ls:
            continue
        
        full_img_file_name = os.path.join(img_folder, str(image_idx))
        if not os.path.exists(full_img_file_name):
            continue
        #     full_img_file_name = os.path.join(img_folder2, str(image_idx) + ".jpg")
            
        
        # img = Image.open(full_img_file_name)
        # img = img.convert('RGB')
        # caption = caption_pd.iloc[idx]['caption']
        # sub_caption_str = caption_pd.iloc[idx]['caption_triples']
        
        # sub_captions = decompose_single_query(sub_caption_str)
        img_file_name_ls.append(full_img_file_name)
        img_idx_ls.append(image_idx)
        if total_count > 0 and len(img_file_name_ls) >= total_count:
            break
            
    return img_idx_ls, img_file_name_ls    

def obtain_sample_hash(img_idx_ls, img_ls):
    sorted_idx = sorted(range(len(img_idx_ls)), key=lambda k: img_idx_ls[k])
    
    sorted_img_idx_ls = [img_idx_ls[i] for i in sorted_idx]
    
    sorted_img_ls = [img_ls[i] for i in sorted_idx]

    hash_val = utils.hashfn(sorted_img_ls)
    
    return hash_val

def read_images_from_folder(folder_path, total_count=100):
    transform = transforms.Compose([
                # transforms.CenterCrop(resol),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])
    filename_ls=[]
    raw_img_ls = []
    img_ls = []
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            raw_img = Image.open(os.path.join(folder_path, filename)).convert('RGB')
            img = transform(raw_img)
            raw_img_ls.append(raw_img)
            img_ls.append(img)
            filename_ls.append(filename)
            count += 1
            if total_count > 0 and count >= total_count:
                break
    return filename_ls, raw_img_ls, img_ls

def read_flickr_image_captions(caption_file):
    filename_caption_mappings = dict()
    with open(caption_file, "r") as f:
        for line in f:
            filename, caption = line.split("\t")
            if filename.endswith("#0"):
                filename = filename[:-2]
                filename_caption_mappings[filename] = caption.strip()
    return filename_caption_mappings

def get_slic_segments_for_single_image(image, n_segments=32):
    segments = slic(np.array(image), n_segments=n_segments, compactness=10, sigma=1, start_label=1)
    return segments

def get_slic_segments(images, n_segments=32):
    all_labels = []
    for image in tqdm(images):
        segments = get_slic_segments_for_single_image(image, n_segments=n_segments)
        all_labels.append(segments)
    return all_labels



def plot_bbox(image, bbox):
    numpy_image = np.array(image)
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.imwrite("output.jpg", image)
    

def get_slic_segments_for_sub_images(images_ls, n_segments=32):
    all_labels = []
    for images in tqdm(images_ls):
        sub_labels = []
        for image in images:
            segments = slic(np.array(image), n_segments=n_segments, compactness=40, sigma=10, start_label=1)
            # segments = slic(np.array(image), n_segments=n_segments, compactness=40, sigma=1, start_label=1)
            sub_labels.append(segments)
        all_labels.append(sub_labels)
    return all_labels

def merge_bboxes_ls(all_bboxes, prev_bboxes, bboxes):
    if len(all_bboxes) == 0:
        for id1 in range(len(bboxes)):
            curr_all_bboxes = []
            for id2 in range(len(bboxes[id1])):
                curr_all_bboxes.extend(bboxes[id1][id2])
                
        
            all_bboxes.append(curr_all_bboxes)
        return all_bboxes, flatten_sub_image_ls(bboxes)
    
    all_curr_transformed_bboxes_ls = []
    for id1 in range(len(bboxes)):
        transformed_bboxes_ls = []
        for id2 in range(len(bboxes[id1])):
            base_bboxes = np.array(prev_bboxes[id1][id2])
            transformed_bboxes = np.array(bboxes[id1][id2])
            if len(transformed_bboxes) > 0:
                transformed_bboxes[:, 0] += base_bboxes[0]
                transformed_bboxes[:, 2] += base_bboxes[0]
                transformed_bboxes[:, 1] += base_bboxes[1]
                transformed_bboxes[:, 3] += base_bboxes[1]
            all_bboxes[id1].extend(transformed_bboxes.tolist())
            transformed_bboxes_ls.extend(transformed_bboxes.tolist())
        all_curr_transformed_bboxes_ls.append(transformed_bboxes_ls)
    return all_bboxes, all_curr_transformed_bboxes_ls

def get_patches_from_bboxes2(bboxes_for_img, images):
    patches_for_imgs = []
    for i, bboxes in enumerate(bboxes_for_img):
        patches = []
        for bbox in bboxes:
            patches.append(Patch(images[i], bbox, images[i].crop(bbox)))
        patches_for_imgs.append(patches)
    return patches_for_imgs

def transform_image_ls_to_sub_image_ls(images):
    sub_images_ls = []
    for image in images:
        sub_images_ls.append([image])
    return sub_images_ls

""" Mask to bounding boxes """
def masks_to_bboxes(masks):
    all_bboxes = []

    widths = []
    heights = []
    for img_mask in tqdm(masks):
        bboxes, curr_widths, curr_heights = masks_to_bboxes_single_img(img_mask)
        all_bboxes.append(bboxes)
        widths.extend(curr_widths)
        heights.extend(curr_heights)
        
    return all_bboxes

def masks_to_bboxes_single_img(img_mask):
    bboxes = []
    widths = []
    heights = []
    props = regionprops(img_mask)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])
        widths.append(x2 - x1)
        heights.append(y2 - y1)
    return bboxes, widths, heights

def extend_bbox(x1, y1, x2, y2, labels, extend_size=0):
    x1 = max(0, x1 - extend_size)
    y1 = max(0, y1 - extend_size)
    x2 = min(labels.shape[1], x2 + extend_size)
    y2 = min(labels.shape[0], y2 + extend_size)
    return x1, y1, x2, y2
# 5
def derive_and_extend_bbox(curr_mask_ids, labels, extend_size=0):
    x1 = np.min(curr_mask_ids[1])
    x2 = np.max(curr_mask_ids[1])
    y1 = np.min(curr_mask_ids[0])
    y2 = np.max(curr_mask_ids[0])
    x1, y1, x2, y2 = extend_bbox(x1, y1, x2, y2, labels, extend_size=extend_size)
    return x1, y1, x2, y2


def split_uncovered_boolean_image_mask(uncovered_img, bboxes, widths, heights, extend_size=0):
    labels, num_components = ndimage.label(uncovered_img)
    if num_components > 1:
        for i in range(1, num_components + 1):
            curr_mask_ids = np.nonzero(labels == i)
            x1, y1, x2, y2 = derive_and_extend_bbox(curr_mask_ids, labels, extend_size=extend_size)
            if x1 <= 0 and y1 <=0 and x2 >= labels.shape[1] and y2 >= labels.shape[0]:
                continue
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            bboxes.append([x1, y1, x2, y2])
            widths.append(x2 - x1)
            heights.append(y2 - y1)
    else:
        curr_mask_ids = np.nonzero(np.array(uncovered_img))
        x1, y1, x2, y2 = derive_and_extend_bbox(curr_mask_ids, labels, extend_size=extend_size)
        if not (x1 <= 0 and y1 <=0 and x2 >= labels.shape[1] and y2 >= labels.shape[0]):
            if x2 - x1 > 0 and y2 - y1 > 0:
                bboxes.append([x1, y1, x2, y2])
                widths.append(x2 - x1)
                heights.append(y2 - y1)
        
        
        
    

def masks_to_bboxes_for_subimages(masks_ls, extend_size=5):
    all_bboxes = []

    widths = []
    heights = []
    for masks in tqdm(masks_ls):
        bboxes_ls = []
        for img_mask in masks:
            bboxes = []
            # if len(all_bboxes) == 14 and len(bboxes_ls) > 10:
            #     print()
            # if len(all_bboxes) == 14 and len(bboxes_ls) == 18:
            #     print()
            if len(np.unique(img_mask)) > 1:
                covered_mask = np.zeros_like(img_mask)
                props = regionprops(img_mask)
                for prop in props:
                    x1 = prop.bbox[1]
                    y1 = prop.bbox[0]

                    x2 = prop.bbox[3]
                    y2 = prop.bbox[2]
                    # x1 = max(0, x1 - 10)
                    # y1 = max(0, y1 - 10)
                    # x2 = min(img_mask.shape[1], x2 + 10)
                    # y2 = min(img_mask.shape[0], y2 + 10)
                    x1, y1, x2, y2 = extend_bbox(x1, y1, x2, y2, img_mask, extend_size=extend_size)
                    
                    if x1 <= 0 and y1 <=0 and x2 >= img_mask.shape[1] and y2 >= img_mask.shape[0]:
                        continue
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        continue
                    bboxes.append([x1, y1, x2, y2])
                    widths.append(x2 - x1)
                    heights.append(y2 - y1)
                    covered_mask[y1:y2, x1:x2] = 1
                uncovered_img = Image.fromarray((1-covered_mask).astype(np.uint8) * 255)
                if np.sum((1-covered_mask)) > 0:
                    split_uncovered_boolean_image_mask(uncovered_img, bboxes, widths, heights, extend_size=extend_size)
            bboxes_ls.append(bboxes)
        all_bboxes.append(bboxes_ls)
        
    return all_bboxes

def flatten_sub_image_ls(all_sub_images_ls):
    all_sub_images_ls_new = []
    for sub_images_ls in all_sub_images_ls:
        new_sub_images = []
        for sub_images in sub_images_ls:
            new_sub_images.extend(sub_images)
        all_sub_images_ls_new.append(new_sub_images)
    return all_sub_images_ls_new


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [_to_device(xi, device) for xi in x]

def get_sub_image_by_bbox(image, bbox):
    sub_image = image.crop(bbox)
    return sub_image

def get_sub_image_by_bbox_for_images(images_ls, bboxes_ls):
    sub_images_ls_ls =[]
    for idx in tqdm(range(len(images_ls))):
        images = images_ls[idx]
        curr_bboxes_ls = bboxes_ls[idx]
        sub_images_ls =[]
        for sub_idx in range(len(images)):
            image = images[sub_idx]
            bboxes = curr_bboxes_ls[sub_idx]
            sub_images = []
            for bbox in bboxes:
                # sub_image = image.crop(bbox)
                sub_image = get_sub_image_by_bbox(image, bbox)
                sub_images.append(sub_image)
            sub_images_ls.append(sub_images)
        sub_images_ls_ls.append(sub_images_ls)
    return sub_images_ls_ls

def is_bbox_ls_full_empty(all_bboxes):
    emptiness_count = 0
    all_bbox_count = 0
    for bboxes in all_bboxes:
        for sub_bboxes in bboxes:
            if len(sub_bboxes) == 0:
                emptiness_count += 1
            all_bbox_count += 1
    return emptiness_count >= all_bbox_count


def merge_sub_images_ls_to_all_images_ls(sub_images_ls_ls, all_images_ls_ls):
    for idx in range(len(sub_images_ls_ls)):
        if sub_images_ls_ls[idx] is None:
            continue        
        all_images_ls_ls[idx] = torch.cat([all_images_ls_ls[idx], sub_images_ls_ls[idx]])
    return all_images_ls_ls

def embed_patches(forward_func, patches, model, input_processor, processor=None, device='cuda', resize=None, max_batch_size=100):
    patches = input_processor(patches)
    if processor is not None:
        patches = processor(patches)
    x = _to_device(patches, device)

    if resize:
        x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)
        
    if x.shape[0] < max_batch_size:
        return forward_func(x, model).cpu()
    else:
        embedded_batch_ls = []        
        for start_idx in range(0, x.shape[0], max_batch_size):
            
            end_idx = min(start_idx + max_batch_size, x.shape[0])
            embedded_batch = forward_func(x[start_idx: end_idx], model)
            embedded_batch_ls.append(embedded_batch)
        
        return torch.cat(embedded_batch_ls).cpu()

def get_patches_from_bboxes(patch_emb_ls, img_per_batch_ls, masks_ls, bboxes_ls,patch_count_for_compute_ls, patch_count_ls, forward_func,model, image_file_name_ls, input_processor, image_size=(224, 224), processor=None, resize=None, device="cpu",save_mask_bbox=False):
    # if sub_bboxes == None:
    #     sub_bboxes = [[[bbox] for bbox in img_bboxes] for img_bboxes in all_bboxes]

    # all_patches = []
    # img_labels = []
    # masks = []
    # bboxes_ls=[]
    # for i, (image_file_name) in tqdm(enumerate(zip(image_file_name_ls))):
    for i, image_file_name in tqdm(enumerate(image_file_name_ls)):
        
        image = Image.open(image_file_name).convert('RGB')
        
        for patch_count_idx in range(len(patch_count_ls)):
            if patch_count_for_compute_ls[patch_count_idx] == False:
                continue
            if i == 0:
                patch_emb_ls[patch_count_idx] = []
                img_per_batch_ls[patch_count_idx] = []
                masks_ls[patch_count_idx] = []
                bboxes_ls[patch_count_idx] = []
            
            n_patches = patch_count_ls[patch_count_idx]
            img_mask = get_slic_segments_for_single_image(image, n_segments=n_patches)
            bboxes, curr_widths, curr_heights = masks_to_bboxes_single_img(img_mask)
            visible = [[bbox] for bbox in bboxes]
            patches = []
            for bbox, viz in zip(bboxes, visible):
                # curr_patch = PIL.Image.new('RGB', image.size) #image.copy().filter(ImageFilter.GaussianBlur(radius=10))
                # for viz_box in viz:
                #     # add the visible patches from the original image to curr_patch
                #     curr_patch.paste(image.copy().crop(viz_box), box=viz_box)
                # # curr_patch = PIL.ImageOps.pad(curr_patch.crop(bbox), image_size)
                # curr_patch = curr_patch.crop(bbox)
                curr_patch = get_sub_image_by_bbox(image, bbox)
                img_per_batch_ls[patch_count_idx].append(i)
                patches.append(curr_patch)
            
            patch_embs = embed_patches(forward_func, patches, model, input_processor, processor, device=device, resize=resize)
            patch_emb_ls[patch_count_idx].append(patch_embs)
            bboxes_ls[patch_count_idx].append(bboxes)
            if save_mask_bbox:
                masks_ls[patch_count_idx].append(img_mask)
            # else:
            #     del img_mask, bboxes
            #     gc.collect()
    for patch_count_idx in range(len(patch_count_ls)):
        if patch_count_for_compute_ls[patch_count_idx] == False:
            continue
        # patch_emb_ls[patch_count_idx] = torch.cat(patch_emb_ls[patch_count_idx])
        # patches = input_processor(patches)
        # if processor is not None:
        #     patches = processor(patches)
        # x = _to_device(patches, device)

        # if resize:
        #     x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)
        # all_patches.append(forward_func(x, model).cpu())
    # if save_mask_bbox:
    #     return patch_emb_ls, img_per_batch_ls, masks_ls, bboxes_ls
    # else:
    #     return patch_emb_ls, img_per_batch_ls

def get_patches_from_bboxes0(patch_emb_ls, img_per_batch_ls, masks_ls, bboxes_ls,patch_count_for_compute_ls, patch_count_ls, forward_func,model, image_file_name_ls, input_processor, sub_bboxes=None, image_size=(224, 224), processor=None, resize=None, device="cpu", save_mask_bbox=False):
    # if sub_bboxes == None:
    #     sub_bboxes = [[[bbox] for bbox in img_bboxes] for img_bboxes in all_bboxes]

    # all_patches = []
    # img_labels = []
    
    # masks = []
    # bboxes_ls=[]
    for i, (image_file_name) in tqdm(enumerate(zip(image_file_name_ls))):
        patches = []
        image = Image.open(image_file_name).convert('RGB')
        for patch_count_idx in range(len(patch_count_ls)):
            if patch_count_for_compute_ls[patch_count_idx] == False:
                continue
            if i == 0:
                patch_emb_ls[patch_count_idx] = []
                img_per_batch_ls[patch_count_idx] = []
                masks_ls[patch_count_idx] = []
                bboxes_ls[patch_count_idx] = []
            n_patches = patch_count_ls[patch_count_idx]
            img_mask = get_slic_segments_for_single_image(image, n_segments=n_patches)
            bboxes, curr_widths, curr_heights = masks_to_bboxes_single_img(img_mask)
            visible = [[bbox] for bbox in bboxes]
            # for bbox, viz in zip(bboxes, visible):
            for j in range(len(bboxes)):
                bbox = bboxes[j]
                viz = visible[j]
                curr_patch = PIL.Image.new('RGB', image.size) #image.copy().filter(ImageFilter.GaussianBlur(radius=10))
                for viz_box in viz:
                    # add the visible patches from the original image to curr_patch
                    masked_image = np.copy(image)
                    masked_image[img_mask != (j+1)] = 255
                    masked_image = Image.fromarray(masked_image)
                    curr_patch.paste(masked_image.copy().crop(viz_box), box=viz_box)
                curr_patch = PIL.ImageOps.pad(curr_patch.crop(bbox), image_size)
                img_per_batch_ls[patch_count_idx].append(i)
                patches.append(curr_patch)
            patches = input_processor(patches)
            if processor is not None:
                patches = processor(patches)
            x = _to_device(patches, device)

            if resize:
                x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)
            patch_emb_ls[patch_count_idx].append(forward_func(x, model).cpu())
            bboxes_ls[patch_count_idx].append(bboxes)
            if save_mask_bbox:
                masks_ls[patch_count_idx].append(img_mask)
                
            else:
                del img_mask, bboxes
                gc.collect()
        # patches.append(model(x).cpu())            
            
        #     # patches.append(input_processor(curr_patch))
            
        # all_patches += patches
    for patch_count_idx in range(len(patch_count_ls)):
        if patch_count_for_compute_ls[patch_count_idx] == False:
            continue
        # patch_emb_ls[patch_count_idx] = torch.cat(patch_emb_ls[patch_count_idx])
    
    if save_mask_bbox:
        return patch_emb_ls, img_per_batch_ls, masks_ls, bboxes_ls
    else:
        return patch_emb_ls, img_per_batch_ls

def get_image_embeddings(img_file_name_ls, input_processor, forward_func,model,device='cuda', not_normalize=False):
    results = []
    for _, file_name in tqdm(enumerate(img_file_name_ls)):
        image = Image.open(file_name).convert('RGB')
        patches = input_processor([image])
        x = _to_device(patches, device)
        results.append(forward_func(x, model).cpu())
    results = torch.cat(results)
    # if not not_normalize:
    #     print("normalize image embeddings::")
    #     results = normalize(results)
    return results

def embed_patches_ls(forward_func, patches_ls, model, input_processor, processor = None, device='cuda', resize=None):
    patch_embs_ls = []
    for patches in tqdm(patches_ls):
        if len(patches) > 0:
            patch_embs = embed_patches(forward_func, patches, model, input_processor, processor=processor, device=device, resize=resize)
        else:
            patch_embs = None
        patch_embs_ls.append(patch_embs)
    return patch_embs_ls

def embed_patches_two_level_ls(forward_func, full_patches_ls, model, input_processor, processor = None, device='cuda', resize=None):
    
    full_patch_embs_ls = []    
    for patches_ls in tqdm(full_patches_ls):
        patch_embs_ls = []    
        for patches in patches_ls:
            if len(patches) > 0:
                patch_embs = embed_patches(forward_func, patches, model, input_processor, processor=processor, device=device, resize=resize)
            else:
                patch_embs = None
            patch_embs_ls.append(patch_embs)
        full_patch_embs_ls.append(patch_embs_ls)
    return full_patch_embs_ls


def concat_patch_embs(patch_embs_ls):
    concat_patch_embs_ls = []
    concat_patch_emb_idx_ls = []
    for idx in range(len(patch_embs_ls)):
        # concat_patch_embs_ls.extend(patch_embs_ls[idx])
        concat_patch_emb_idx_ls.extend([idx] * len(patch_embs_ls[idx]))
    return patch_embs_ls, concat_patch_emb_idx_ls
    
    

class ConceptLearner:
    # def __init__(self, samples: list[PIL.Image], input_to_latent, input_processor, device: str = 'cpu'):
    def __init__(self, img_file_name_ls: list, model, input_to_latent, input_processor, dataset_name, device: str = 'cpu'):
        self.samples = img_file_name_ls
        self.device = torch.device(device)
        self.batch_size = 128
        self.input_to_latent = input_to_latent
        self.image_size = 224 # if type(samples[0]) == PIL.Image.Image else None
        self.input_processor = input_processor
        self.dataset_name = dataset_name
        self.model = model

    def patches(self, images=None, patch_method="slic"):
        if images is None:
            images = self.samples

        samples_hash = utils.hashfn(images)
        if os.path.exists(f"output/saved_patches_{patch_method}_{samples_hash}.pkl"):
            print("Loading cached patches")
            print(samples_hash)
            patches = utils.load(f"output/saved_patches_{patch_method}_{samples_hash}.pkl")
            return patches

        
        masks = get_slic_segments(images, n_segments=8 * 8)
        bboxes_for_imgs = masks_to_bboxes(masks)
        patches_for_imgs = get_patches_from_bboxes2(bboxes_for_imgs, images)
            
        utils.save(patches_for_imgs, f"output/saved_patches_{patch_method}_{samples_hash}.pkl")
        return patches_for_imgs
    
    def store_patch_encodings_separately(self, img_file_name_ls, patch_activations, img_for_patch):
        img_for_patch_tensor = torch.tensor(img_for_patch)
        patch_activation_ls = []
        for idx in tqdm(range(len(img_file_name_ls))):
            # img_idx = img_for_patch[idx]
            patch_activation = patch_activations[img_for_patch_tensor == idx]
            patch_activation_ls.append(patch_activation)
        return patch_activation_ls
        

    def get_patches(self, patch_count_ls, samples_hash, img_idx_ls=None, img_file_name_ls=None, method="slic", not_normalize=False, use_mask=False, compute_img_emb=True, save_mask_bbox=False):
        """Get patches from images using different segmentation methods."""
        if img_file_name_ls is None:
            img_file_name_ls = self.samples

        patch_count_for_compute_ls = [True]*len(patch_count_ls)
        patch_emb_ls = [None]*len(patch_count_ls)
        masks_ls = [None]*len(patch_count_ls)
        img_per_batch_ls = [None]*len(patch_count_ls)
        bboxes_ls = [None]*len(patch_count_ls)
        cached_img_idx_ls = None
        if not os.path.exists(f"output/"):
            os.mkdir(f"output/")
        for idx in range(len(patch_count_ls)):
            n_patches = patch_count_ls[idx]
            cached_file_name = f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}{'_use_mask' if use_mask else ''}.pkl"
            if os.path.exists(cached_file_name):
                print("Loading cached patches")
                print(samples_hash)
                cached_data = utils.load(cached_file_name)
                
                # if len(cached_data) == 6:
                bboxes = None
                if save_mask_bbox:
                    patch_activations, masks, bboxes, img_for_patch = cached_data
                else:
                    if len(cached_data) == 3:
                        patch_activations, bboxes, img_for_patch = cached_data
                    else:
                        patch_activations, img_for_patch = cached_data
                    
                # else:
                #     image_embs, patch_activations, masks, bboxes, img_for_patch = cached_data
                    # utils.save((img_idx_ls, image_embs, patch_activations, masks, bboxes, img_for_patch), f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}{'_use_mask' if use_mask else ''}.pkl")
                
                # if image_embs is None and compute_img_emb:
                #     image_embs = get_image_embeddings(img_file_name_ls, self.input_processor, self.input_to_latent, self.model, not_normalize=not_normalize)  
                
                try:
                    patch_activations = self.store_patch_encodings_separately(img_file_name_ls, patch_activations, img_for_patch)
                except:
                    print("no need to separate")
                patch_emb_ls[idx] = patch_activations
                img_per_batch_ls[idx] = img_for_patch
                bboxes_ls[idx] = bboxes
                if save_mask_bbox:
                    masks_ls[idx] = masks
                    
                    
                patch_count_for_compute_ls[idx] = False
                
                # if save_mask_bbox:
                #     utils.save((patch_activations, masks, bboxes, img_for_patch), cached_file_name)
                #     # return cached_img_idx_ls, image_embs, patch_activations, masks, bboxes, img_for_patch
                # else:
                #     utils.save((patch_activations, bboxes, img_for_patch), cached_file_name)
                
                # if save_mask_bbox:  
                #     return img_idx_ls, image_embs, patch_activations, masks, bboxes, img_for_patch
                # else:
                #     return img_idx_ls, image_embs, patch_activations, img_for_patch
        cached_img_file_name = f"output/saved_img_embs_{method}_{samples_hash}.pkl"
        if os.path.exists(cached_img_file_name):
            image_embs, cached_img_idx_ls = utils.load(cached_img_file_name)
        else:
            image_embs = get_image_embeddings(img_file_name_ls, self.input_processor, self.input_to_latent, self.model, not_normalize=not_normalize)
            utils.save((image_embs, img_idx_ls), cached_img_file_name)
            cached_img_idx_ls = img_idx_ls
        
        # if compute_img_emb:
        #     image_embs = get_image_embeddings(img_file_name_ls, self.input_processor, self.input_to_latent, self.model, not_normalize=not_normalize)
        # else:
        #     image_embs = None
        patch_activations = None
        masks = None
        bboxes = None
        img_for_patch = None
        # if method == "slic":
        # masks = get_slic_segments(img_file_name_ls, n_segments=n_patches)
        # bboxes = masks_to_bboxes(masks)
        # get_patches_from_bboxes(model, images, all_bboxes, input_processor, sub_bboxes=None, image_size=(224, 224), processor=None, resize=None, device="cpu"):
        if True in patch_count_for_compute_ls:
            if not use_mask:
                get_patches_from_bboxes(patch_emb_ls, img_per_batch_ls, masks_ls, bboxes_ls, patch_count_for_compute_ls, patch_count_ls, self.input_to_latent, self.model, img_file_name_ls, self.input_processor, device=self.device, resize=self.image_size, save_mask_bbox=save_mask_bbox)
            else:
                get_patches_from_bboxes0(patch_emb_ls, img_per_batch_ls, masks_ls, bboxes_ls, patch_count_for_compute_ls, patch_count_ls, self.input_to_latent, self.model, img_file_name_ls, bboxes, self.input_processor, device=self.device, resize=self.image_size, save_mask_bbox=save_mask_bbox)
        
        # if save_mask_bbox:
        #     patch_activations, img_for_patch, masks, bboxes = res
        # else:
        #     patch_activations, img_for_patch = res
        
        #     # patches = self.input_processor(patches)
        # elif method == "sam":
        #     masks = get_sam_segments(images)
        #     bboxes = masks_to_bboxes(masks)
        #     # Merge close boxes to create relation patches
        #     if merge:
        #         bboxes = [merge_boxes(boxes, 8, 8) for boxes in bboxes]
        #     patches, img_for_patch = get_patches_from_bboxes(images, bboxes, self.input_processor)
        #     patches = self.input_processor(patches)
        # elif method == "window":
        #     patch_size = int(self.image_size // n_patches)
        #     strides = int(patch_size)
        #     samples = self.input_processor(images)
        #     patches = torch.nn.functional.unfold(samples, kernel_size=patch_size, stride=strides)
        #     patches = patches.transpose(1, 2).contiguous().view(-1, 3, patch_size, patch_size)
        #     # TODO: add the bbox definition
        #     bboxes = None
        #     img_for_patch = None
        # elif callable(method):
        #     patches = method(images, n_patches)
        #     patches = self.input_processor(patches)
        #     # TODO: add the bbox definition
        #     bboxes = None
        #     img_for_patch = None
        # else:
        #     raise ValueError("method must be either 'slic' or 'sam' or 'window'.")

        # print(len(patches))
        # model, dataset: Dataset, batch_size=128, resize=None, processor=None, device='cuda'
        # patch_activations = utils._batch_inference(self.input_to_latent, patches, self.batch_size, self.image_size,
        #     device=self.device)

        # cache the result

        for patch_count_idx in range(len(patch_count_ls)):
            if patch_count_for_compute_ls[patch_count_idx] == False:
                continue        
            n_patches = patch_count_ls[patch_count_idx]
            cached_file_name = f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}{'_use_mask' if use_mask else ''}.pkl"
            if save_mask_bbox:
                patch_activations, img_for_patch, masks, bboxes = patch_emb_ls[patch_count_idx], img_per_batch_ls[patch_count_idx], masks_ls[patch_count_idx], bboxes_ls[patch_count_idx]
                utils.save((patch_activations, masks, bboxes, img_for_patch), cached_file_name)
                # return cached_img_idx_ls, image_embs, patch_activations, masks, bboxes, img_for_patch
            else:
                patch_activations, img_for_patch, bboxes = patch_emb_ls[patch_count_idx], img_per_batch_ls[patch_count_idx], bboxes_ls[patch_count_idx]
                utils.save((patch_activations, bboxes, img_for_patch), cached_file_name)
                # return img_idx_ls, image_embs, patch_activations, img_for_patch
        
        # if save_mask_bbox:
        return cached_img_idx_ls, image_embs, patch_emb_ls, masks_ls, bboxes_ls, img_per_batch_ls
        # else:
        #     return cached_img_idx_ls, image_embs, patch_activations, img_for_patch
        
    
    def get_patches_by_hierarchies(self, images=None, method="slic", not_normalize=False, use_mask=False, compute_img_emb=True, depth_lim=5, partition_strategy="one", extend_size=0, recompute_img_emb=False):
        """Get patches from images using different segmentation methods."""
        if images is None:
            images = self.samples

        samples_hash = utils.hashfn(images)
        
        cached_file_name=f"output/saved_patches_hierarchy_{method}_{partition_strategy}_{depth_lim}_{extend_size}_{samples_hash}{'_not_normalize' if not_normalize else ''}{'_use_mask' if use_mask else ''}.pkl"
        if os.path.exists(cached_file_name) and not recompute_img_emb:
            print("Loading cached patches")
            print(samples_hash)
            image_embs, patch_activations, masks, bboxes, img_for_patch = utils.load(cached_file_name)
            if image_embs is None and compute_img_emb:
                image_embs = get_image_embeddings(images, self.input_processor, self.input_to_latent, self.model, not_normalize=not_normalize)    
            return image_embs, patch_activations, masks, bboxes, img_for_patch
        
        if compute_img_emb:
            image_embs = get_image_embeddings(images, self.input_processor, self.input_to_latent, self.model, not_normalize=not_normalize)
        else:
            image_embs = None
        patch_activations = None
        masks = None
        bboxes = None
        img_for_patch = None
        # if method == "slic":
        
        all_sub_images_ls = transform_image_ls_to_sub_image_ls(images)
        # embed_patches_ls(forward_func, patches_ls, model, input_processor, processor, device='cuda', resize=None)
        all_sub_embs_ls = embed_patches_ls(self.input_to_latent, all_sub_images_ls, self.model, self.input_processor, device=self.device, resize=self.image_size)
        all_bboxes_ls = []
        curr_sub_images_ls = transform_image_ls_to_sub_image_ls(images)
        depth=0
        prev_bboxes_ls = None
        while True:
            if depth > depth_lim:
                break
            # if depth == 0:
            #     n_patches = 32
            # elif depth == 1:
            #     n_patches = 4
            # else:
            #     n_patches = 2
            n_patches = determine_n_patches(partition_strategy, depth)
            masks = get_slic_segments_for_sub_images(curr_sub_images_ls, n_segments=n_patches)
            # bboxes = masks_to_bboxes(masks)
            bboxes_ls = masks_to_bboxes_for_subimages(masks, extend_size=extend_size)
            all_bboxes_ls, prev_bboxes_ls = merge_bboxes_ls(all_bboxes_ls, prev_bboxes_ls, bboxes_ls)
            curr_sub_images_ls = get_sub_image_by_bbox_for_images(curr_sub_images_ls, bboxes_ls)
            curr_sub_images_ls = flatten_sub_image_ls(curr_sub_images_ls)
            # prev_bboxes_ls = flatten_sub_image_ls(bboxes_ls)
            if is_bbox_ls_full_empty(bboxes_ls):
                break
            
            curr_sub_embs_ls = embed_patches_ls(self.input_to_latent, curr_sub_images_ls, self.model, self.input_processor, device=self.device, resize=self.image_size)
            all_sub_embs_ls = merge_sub_images_ls_to_all_images_ls(curr_sub_embs_ls, all_sub_embs_ls)
            depth += 1
        
        patch_activations, img_for_patch = concat_patch_embs(all_sub_embs_ls)
            # get_patches_from_bboxes(model, images, all_bboxes, input_processor, sub_bboxes=None, image_size=(224, 224), processor=None, resize=None, device="cpu"):
        # if not use_mask:
        #     patch_activations, img_for_patch = get_patches_from_bboxes(self.input_to_latent, self.model, images, all_bboxes, self.input_processor, device=self.device, resize=self.image_size)
        # else:
        #     patch_activations, img_for_patch = get_patches_from_bboxes0(self.input_to_latent, self.model, images, masks, bboxes, self.input_processor, device=self.device, resize=self.image_size)
                
            
        
        if not os.path.exists(f"output/"):
            os.mkdir(f"output/")
        utils.save((image_embs, patch_activations, masks, all_bboxes_ls, img_for_patch), cached_file_name)
        return image_embs, patch_activations, masks, all_bboxes_ls, img_for_patch
    
    def get_patches_by_hierarchies_by_trees(self, images=None, method="slic", not_normalize=False, use_mask=False, compute_img_emb=True, depth_lim=5, partition_strategy="one", extend_size=0):
        """Get patches from images using different segmentation methods."""
        if images is None:
            images = self.samples

        samples_hash = utils.hashfn(images)
        cached_file_name=f"output/saved_patches_hierarchy_by_trees_{method}_{partition_strategy}_{depth_lim}_{extend_size}_{samples_hash}{'_not_normalize' if not_normalize else ''}{'_use_mask' if use_mask else ''}.pkl"
        if os.path.exists(cached_file_name):
            print("Loading cached patches")
            print(samples_hash)
            # image_embs, patch_activations, masks, bboxes, img_for_patch = utils.load(cached_file_name)
            root_node_ls = utils.load(cached_file_name)
            # if image_embs is None and compute_img_emb:
            #     image_embs = get_image_embeddings(images, self.input_processor, self.input_to_latent, self.model, not_normalize=not_normalize)    
            # return image_embs, patch_activations, masks, bboxes, img_for_patch
            return root_node_ls
        if compute_img_emb:
            image_embs = get_image_embeddings(images, self.input_processor, self.input_to_latent, self.model, not_normalize=not_normalize)
        else:
            image_embs = None
        patch_activations = None
        masks = None
        bboxes = None
        img_for_patch = None
        # if method == "slic":
        
        all_sub_images_ls = transform_image_ls_to_sub_image_ls(images)
        # embed_patches_ls(forward_func, patches_ls, model, input_processor, processor, device='cuda', resize=None)
        all_sub_embs_ls = embed_patches_ls(self.input_to_latent, all_sub_images_ls, self.model, self.input_processor, device=self.device, resize=self.image_size)
        all_bboxes_ls = []
        curr_sub_images_ls = transform_image_ls_to_sub_image_ls(images)
                
        # root_node_ls = create_nodes_same_layer(curr_sub_images_ls, None, all_sub_embs_ls)
        root_node_ls = init_root_nodes(curr_sub_images_ls, None, all_sub_embs_ls)
        parent_node_ls = root_node_ls
        depth=0
        prev_bboxes_ls = None
        while True:
            if depth > depth_lim:
                break
            n_patches = determine_n_patches(partition_strategy, depth)
            masks = get_slic_segments_for_sub_images(curr_sub_images_ls, n_segments=n_patches)
            # bboxes = masks_to_bboxes(masks)
            bboxes_ls = masks_to_bboxes_for_subimages(masks, extend_size=extend_size)
            all_bboxes_ls, prev_bboxes_ls = merge_bboxes_ls(all_bboxes_ls, prev_bboxes_ls, bboxes_ls)
            curr_sub_images_ls = get_sub_image_by_bbox_for_images(curr_sub_images_ls, bboxes_ls)
            # curr_sub_images_ls = flatten_sub_image_ls(curr_sub_images_ls)
            # prev_bboxes_ls = flatten_sub_image_ls(bboxes_ls)
            if is_bbox_ls_full_empty(bboxes_ls):
                break
            
            curr_sub_embs_ls = embed_patches_two_level_ls(self.input_to_latent, curr_sub_images_ls, self.model, self.input_processor, device=self.device, resize=self.image_size)
            # all_sub_embs_ls = merge_sub_images_ls_to_all_images_ls(curr_sub_embs_ls, all_sub_embs_ls)
            curr_node_ls = create_non_root_nodes_same_layer(curr_sub_images_ls, bboxes_ls, curr_sub_embs_ls, parent_node_ls)
            curr_sub_images_ls = flatten_sub_image_ls(curr_sub_images_ls)
            parent_node_ls = curr_node_ls #flatten_sub_image_ls(curr_node_ls)
            depth += 1
        
        # patch_activations, img_for_patch = concat_patch_embs(all_sub_embs_ls)
            # get_patches_from_bboxes(model, images, all_bboxes, input_processor, sub_bboxes=None, image_size=(224, 224), processor=None, resize=None, device="cpu"):
        # if not use_mask:
        #     patch_activations, img_for_patch = get_patches_from_bboxes(self.input_to_latent, self.model, images, all_bboxes, self.input_processor, device=self.device, resize=self.image_size)
        # else:
        #     patch_activations, img_for_patch = get_patches_from_bboxes0(self.input_to_latent, self.model, images, masks, bboxes, self.input_processor, device=self.device, resize=self.image_size)
                
            
        
        if not os.path.exists(f"output/"):
            os.mkdir(f"output/")
        # utils.save((image_embs, patch_activations, masks, all_bboxes_ls, img_for_patch), cached_file_name)
        utils.save(root_node_ls, cached_file_name)
        # return image_embs, patch_activations, masks, all_bboxes_ls, img_for_patch
        return root_node_ls


def load_image_as_numpy(image_path):
    # Open the image using PIL
    pil_image = Image.open(image_path)
    rgb_image = pil_image.convert('RGB')
    # Convert PIL image to a NumPy array
    image_array = np.array(rgb_image)
    
    return image_array

def segment_single_image(image, mask_generator):
    masks = mask_generator.generate(image)
    return masks


def segment_all_images(data_folder, img_name_ls, split="train", sam_model_type="vit_l", sam_model_folder="/data6/wuyinjun/sam/sam_vit_l_0b3195.pth", device = torch.device("cuda")):
    image_mappings = dict()
    segment_mappings = dict()
    sam = sam_model_registry[sam_model_type](checkpoint=sam_model_folder)
    sam = sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    for img_name in tqdm(img_name_ls):
        image = load_image_as_numpy(os.path.join(data_folder, os.path.join("images/" + split, img_name)))
        segment_mask_ls = segment_single_image(image, mask_generator)
        image_mappings[img_name] = image
        segment_mappings[img_name] = segment_mask_ls
    
    return image_mappings, segment_mappings


def convert_samples_to_concepts_img(args, samples_hash, model, img_file_name_ls, img_idx_ls, processor, device, patch_count_ls = [32], save_mask_bbox=False):
    # samples: list[PIL.Image], labels, input_to_latent, input_processor, dataset_name, device: str = 'cpu'
    # cl = ConceptLearner(images, labels, vit_forward, processor, img_processor, args.dataset_name, device)
    cl = ConceptLearner(img_file_name_ls, model, vit_forward, processor, args.dataset_name, device)
    # bbox x1,y1,x2,y2
    # image_embs, patch_activations, masks, bboxes, img_for_patch
    # n_patches, images=None, method="slic", not_normalize=False

    
    res = cl.get_patches(patch_count_ls, samples_hash, img_idx_ls=img_idx_ls, img_file_name_ls=img_file_name_ls, method="slic", compute_img_emb=True, save_mask_bbox=save_mask_bbox)
    
    return res
    
    # for idx in range(len(patch_count_ls)):
    #     patch_count = patch_count_ls[idx]
    #     if idx == 0:
    #         res = cl.get_patches(patch_count, samples_hash, img_idx_ls=img_idx_ls, img_file_name_ls=img_file_name_ls, method="slic", compute_img_emb=True, save_mask_bbox=save_mask_bbox)
    #     else:
    #         res = cl.get_patches(patch_count, samples_hash, img_idx_ls=img_idx_ls, img_file_name_ls=img_file_name_ls, method="slic", compute_img_emb=False, save_mask_bbox=save_mask_bbox)
        
    #     if save_mask_bbox:
    #         curr_img_ls, curr_img_emb, patch_emb, masks, bboxes, img_per_patch = res
    #     else:
    #         curr_img_ls, curr_img_emb, patch_emb, img_per_patch = res
    #     if curr_img_emb is not None:
    #         img_emb = curr_img_emb
    #         img_ls = curr_img_ls
    #     patch_emb_ls.append(patch_emb)
        
    #     img_per_batch_ls.append(img_per_patch)
    #     if save_mask_bbox:
    #         bboxes_ls.append(bboxes)
    #         masks_ls.append(masks)
    # if save_mask_bbox:
    #     return img_ls, img_emb, patch_emb_ls, None, None, img_per_batch_ls
    # else:
    #     return img_ls, img_emb, patch_emb_ls, img_per_batch_ls        


def reformat_patch_embeddings(patch_emb_ls, img_per_patch_ls, img_emb, bbox_ls=None):
    img_per_patch_tensor = torch.tensor(img_per_patch_ls[0])
    max_img_id = int(torch.max(img_per_patch_tensor).item())
    patch_emb_curr_img_ls = []
    transformed_bbox_ls = []
    for idx in tqdm(range(max_img_id + 1)):
        sub_patch_emb_curr_img_ls = []
        sub_transformed_bbox_ls = []
        for sub_idx in range(len(patch_emb_ls)):
            patch_emb = patch_emb_ls[sub_idx]
            # img_per_batch = img_per_patch_ls[sub_idx]
            # img_per_patch_tensor = torch.tensor(img_per_batch)
            # curr_selected_ids = torch.nonzero(img_per_patch_tensor == idx).view(-1)
            patch_emb_curr_img = patch_emb[idx] #[curr_selected_ids]
            
            # curr_selected_ids = torch.nonzero(img_per_patch_tensor == idx).view(-1)
            # patch_emb_curr_img = patch_emb[curr_selected_ids]
            sub_patch_emb_curr_img_ls.append(patch_emb_curr_img)
            if bbox_ls is not None:
                sub_transformed_bbox_ls.extend(bbox_ls[sub_idx][idx])
        sub_patch_emb_curr_img_ls.append(img_emb[idx].unsqueeze(0))
        patch_emb_curr_img = torch.cat(sub_patch_emb_curr_img_ls, dim=0)
        # patch_emb_curr_img = torch.cat([img_emb[idx].unsqueeze(0), sub_patch_emb_curr_img], dim=0)
        patch_emb_curr_img_ls.append(patch_emb_curr_img)
        if bbox_ls is not None:
            transformed_bbox_ls.append(sub_transformed_bbox_ls)
    
    return patch_emb_curr_img_ls, transformed_bbox_ls


def is_bbox_overlapped(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    intersection_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    
    # Check if the intersection area is positive
    size1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    size2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    min_size = min(size1, size2)
    return intersection_area > 0.5 * min_size


def determine_overlapped_bboxes(bboxes_ls):
    
    bbox_nb_ls = []
    
    for b_idx in range(len(bboxes_ls)):
        bboxes = bboxes_ls[b_idx]
        # curr_nb_ls = [[] for _ in range(len(bboxes) + 1)]
        curr_nb_ls = [[] for _ in range(len(bboxes))]
        for idx in range(len(bboxes)):
            bbox = bboxes[idx]
            
            for sub_idx in range(len(bboxes)):
                if idx != sub_idx:
                    sub_bbox = bboxes[sub_idx]
                    if is_bbox_overlapped(bbox, sub_bbox):
                        curr_nb_ls[idx].append(sub_idx)
                else:
                    curr_nb_ls[idx].append(sub_idx)
            # curr_nb_ls[idx].append(len(bboxes))
        
        # curr_nb_ls[len(bboxes)].extend(list(range(len(bboxes) + 1)))
        bbox_nb_ls.append(curr_nb_ls)
    
    return bbox_nb_ls


# def determine_overlapped_bboxes(bboxes_ls):
    
#     bbox_nb_ls = []
    
#     for b_idx in range(len(bboxes_ls)):
#         bboxes = bboxes_ls[b_idx]
#         curr_nb_ls = [[] for _ in range(len(bboxes) + 1)]
#         # curr_nb_ls = [[] for _ in range(len(bboxes))]
#         for idx in range(len(bboxes)):
#             bbox = bboxes[idx]
            
#             for sub_idx in range(len(bboxes)):
#                 if idx != sub_idx:
#                     sub_bbox = bboxes[sub_idx]
#                     if is_bbox_overlapped(bbox, sub_bbox):
#                         curr_nb_ls[idx].append(sub_idx)
#                 else:
#                     curr_nb_ls[idx].append(sub_idx)
#             curr_nb_ls[idx].append(len(bboxes))
        
#         curr_nb_ls[len(bboxes)].extend(list(range(len(bboxes) + 1)))
#         bbox_nb_ls.append(curr_nb_ls)
    
#     return bbox_nb_ls