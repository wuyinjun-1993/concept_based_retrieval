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
from retrieval_utils import decompose_single_query, decompose_single_query_ls
from scipy import ndimage
import cv2
from storage import *


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
    
    img_caption_file_name= os.path.join(query_path, "sub_queries2.csv")

    img_folder = os.path.join(data_path, "flickr30k-images/")
    # img_folder2 = os.path.join(data_path, "VG_100K_2/")

    caption_pd = pd.read_csv(img_caption_file_name)
    
    img_ls = []
    img_idx_ls = []
    caption_ls = []
    sub_caption_ls = []
    for idx in tqdm(range(len(caption_pd))):
        image_idx = caption_pd.iloc[idx]['image_id']
        if image_idx in img_idx_ls:
            continue
        
        full_img_file_name = os.path.join(img_folder, str(image_idx))
        # if not os.path.exists(full_img_file_name):
        #     full_img_file_name = os.path.join(img_folder2, str(image_idx) + ".jpg")
            
        
        img = Image.open(full_img_file_name)
        img = img.convert('RGB')
        caption = caption_pd.iloc[idx]['caption']
        # sub_caption_str = caption_pd.iloc[idx]['caption_triples']
        sub_caption_str = caption_pd.iloc[idx]['caption_triples_ls']
        
        # sub_captions = decompose_single_query(sub_caption_str)
        sub_captions = decompose_single_query_ls(sub_caption_str)
        print(sub_captions)
        img_ls.append(img)
        img_idx_ls.append(image_idx)
        caption_ls.append(caption)
        sub_caption_ls.append(sub_captions)
    return caption_ls, img_ls, sub_caption_ls, img_idx_ls


def load_crepe_datasets(data_path, query_path):
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all4.csv")
    
    img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all5.csv")

    img_folder = os.path.join(data_path, "VG_100K/")
    img_folder2 = os.path.join(data_path, "VG_100K_2/")

    caption_pd = pd.read_csv(img_caption_file_name)
    
    img_ls = []
    img_idx_ls = []
    caption_ls = []
    sub_caption_ls = []
    for idx in range(len(caption_pd)):
        image_idx = caption_pd.iloc[idx]['image_id']
        if image_idx in img_idx_ls:
            continue
        
        full_img_file_name = os.path.join(img_folder, str(image_idx) + ".jpg")
        if not os.path.exists(full_img_file_name):
            full_img_file_name = os.path.join(img_folder2, str(image_idx) + ".jpg")
            
        
        img = Image.open(full_img_file_name)
        img = img.convert('RGB')
        caption = caption_pd.iloc[idx]['caption']
        # sub_caption_str = caption_pd.iloc[idx]['caption_triples']
        sub_caption_str = caption_pd.iloc[idx]['caption_triples_ls']
        
        # sub_captions = decompose_single_query(sub_caption_str)
        sub_captions = decompose_single_query_ls(sub_caption_str)
        img_ls.append(img)
        img_idx_ls.append(image_idx)
        caption_ls.append(caption)
        sub_caption_ls.append(sub_captions)
    return caption_ls, img_ls, sub_caption_ls, img_idx_ls

def load_crepe_datasets_full(data_path, query_path):
    img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all.csv")

    split_caption_file_name=os.path.join(query_path, "prod_hard_negatives/split.csv")
    
    img_folder = os.path.join(data_path, "VG_100K/")
    img_folder2 = os.path.join(data_path, "VG_100K_2/")

    caption_pd = pd.read_csv(img_caption_file_name)
    split_file_pd = pd.read_csv(split_caption_file_name)
    
    img_ls = []
    img_idx_ls = []
    caption_ls = []
    sub_caption_ls = []
    split_file_pd['caption'] = split_file_pd['caption'].apply(lambda x: x.strip())
    for idx in range(len(caption_pd)):
        image_idx = caption_pd.iloc[idx]['image_id']
        if image_idx in img_idx_ls:
            continue
        
        full_img_file_name = os.path.join(img_folder, str(image_idx) + ".jpg")
        if not os.path.exists(full_img_file_name):
            full_img_file_name = os.path.join(img_folder2, str(image_idx) + ".jpg")
            
        
        img = Image.open(full_img_file_name)
        img = img.convert('RGB')
        caption = caption_pd.iloc[idx]['caption'].strip()
        
        if caption in split_file_pd['caption'].values and image_idx not in img_idx_ls:
            sub_caption_str=split_file_pd[split_file_pd['caption'] == caption]["caption_triples"].values[0]
        
            # sub_caption_str = caption_pd.iloc[idx]['caption_triples']
        
            sub_captions = decompose_single_query_ls(sub_caption_str)
            img_ls.append(img)
            img_idx_ls.append(image_idx)
            caption_ls.append(caption)
            sub_caption_ls.append(sub_captions)
    return caption_ls, img_ls, sub_caption_ls, img_idx_ls



def load_other_crepe_images(data_path, query_path, img_idx_ls, img_ls, total_count=500):
    
    img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all.csv")
    img_folder = os.path.join(data_path, "VG_100K/")
    img_folder2 = os.path.join(data_path, "VG_100K_2/")

    caption_pd = pd.read_csv(img_caption_file_name)
    if total_count > 0 and len(img_ls) >= total_count:
        return img_idx_ls, img_ls          
    for idx in range(len(caption_pd)):
        image_idx = caption_pd.iloc[idx]['image_id']
        if image_idx in img_idx_ls:
            continue
        
        full_img_file_name = os.path.join(img_folder, str(image_idx) + ".jpg")
        if not os.path.exists(full_img_file_name):
            full_img_file_name = os.path.join(img_folder2, str(image_idx) + ".jpg")
            
        
        img = Image.open(full_img_file_name)
        img = img.convert('RGB')
        # caption = caption_pd.iloc[idx]['caption']
        # sub_caption_str = caption_pd.iloc[idx]['caption_triples']
        
        # sub_captions = decompose_single_query(sub_caption_str)
        img_ls.append(img)
        img_idx_ls.append(image_idx)
        if total_count > 0 and len(img_ls) >= total_count:
            break
    
    return img_idx_ls, img_ls    
        
    
def load_other_flickr_images(data_path, query_path, img_idx_ls, img_ls, total_count=500):
    
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all.csv")
    
    filename_cap_mappings = read_flickr_image_captions(os.path.join(data_path, "results_20130124.token"))    

    # img_folder = os.path.join(data_path, "VG_100K/")
    img_folder = os.path.join(data_path, "flickr30k-images/")
    # img_folder2 = os.path.join(data_path, "VG_100K_2/")

    # caption_pd = pd.read_csv(img_caption_file_name)
    if total_count > 0 and len(img_ls) >= total_count:
        return img_idx_ls, img_ls          
    # for idx in range(len(caption_pd)):
    for image_idx in tqdm(filename_cap_mappings):
        # image_idx = caption_pd.iloc[idx]['image_id']
        if image_idx in img_idx_ls:
            continue
        
        full_img_file_name = os.path.join(img_folder, str(image_idx))
        # if not os.path.exists(full_img_file_name):
        #     full_img_file_name = os.path.join(img_folder2, str(image_idx) + ".jpg")
            
        
        img = Image.open(full_img_file_name)
        img = img.convert('RGB')
        # caption = caption_pd.iloc[idx]['caption']
        # sub_caption_str = caption_pd.iloc[idx]['caption_triples']
        
        # sub_captions = decompose_single_query(sub_caption_str)
        img_ls.append(img)
        img_idx_ls.append(image_idx)
        if total_count > 0 and len(img_ls) >= total_count:
            break
            
    return img_idx_ls, img_ls    

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

def get_slic_segments(images, n_segments=32):
    all_labels = []
    for image in tqdm(images):
        segments = slic(np.array(image), n_segments=n_segments, compactness=10, sigma=1, start_label=1)
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
        bboxes = []
        props = regionprops(img_mask)
        for prop in props:
            x1 = prop.bbox[1]
            y1 = prop.bbox[0]

            x2 = prop.bbox[3]
            y2 = prop.bbox[2]

            bboxes.append([x1, y1, x2, y2])
            widths.append(x2 - x1)
            heights.append(y2 - y1)
        all_bboxes.append(bboxes)
        
    return all_bboxes


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

def get_patches_from_bboxes(forward_func,model, images, all_bboxes, input_processor, sub_bboxes=None, image_size=(224, 224), processor=None, resize=None, device="cpu"):
    if sub_bboxes == None:
        sub_bboxes = [[[bbox] for bbox in img_bboxes] for img_bboxes in all_bboxes]

    all_patches = []
    img_labels = []
    for i, (bboxes, visible, image) in tqdm(enumerate(zip(all_bboxes, sub_bboxes, images))):
        patches = []
        for bbox, viz in zip(bboxes, visible):
            # curr_patch = PIL.Image.new('RGB', image.size) #image.copy().filter(ImageFilter.GaussianBlur(radius=10))
            # for viz_box in viz:
            #     # add the visible patches from the original image to curr_patch
            #     curr_patch.paste(image.copy().crop(viz_box), box=viz_box)
            # # curr_patch = PIL.ImageOps.pad(curr_patch.crop(bbox), image_size)
            # curr_patch = curr_patch.crop(bbox)
            curr_patch = get_sub_image_by_bbox(image, bbox)
            img_labels.append(i)
            patches.append(curr_patch)
            
        patch_embs = embed_patches(forward_func, patches, model, input_processor, processor, device=device, resize=resize)
        all_patches.append(patch_embs)
        # patches = input_processor(patches)
        # if processor is not None:
        #     patches = processor(patches)
        # x = _to_device(patches, device)

        # if resize:
        #     x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)
        # all_patches.append(forward_func(x, model).cpu())
        
    return torch.cat(all_patches), img_labels

def get_patches_from_bboxes0(forward_func,model, images, masks, all_bboxes, input_processor, sub_bboxes=None, image_size=(224, 224), processor=None, resize=None, device="cpu"):
    if sub_bboxes == None:
        sub_bboxes = [[[bbox] for bbox in img_bboxes] for img_bboxes in all_bboxes]

    all_patches = []
    img_labels = []
    for i, (bboxes, visible, mask, image) in tqdm(enumerate(zip(all_bboxes, sub_bboxes, masks, images))):
        patches = []
        # for bbox, viz in zip(bboxes, visible):
        for j in range(len(bboxes)):
            bbox = bboxes[j]
            viz = visible[j]
            curr_patch = PIL.Image.new('RGB', image.size) #image.copy().filter(ImageFilter.GaussianBlur(radius=10))
            for viz_box in viz:
                # add the visible patches from the original image to curr_patch
                masked_image = np.copy(image)
                masked_image[mask != (j+1)] = 255
                masked_image = Image.fromarray(masked_image)
                curr_patch.paste(masked_image.copy().crop(viz_box), box=viz_box)
            curr_patch = PIL.ImageOps.pad(curr_patch.crop(bbox), image_size)
            img_labels.append(i)
            patches.append(curr_patch)
        patches = input_processor(patches)
        if processor is not None:
            patches = processor(patches)
        x = _to_device(patches, device)

        if resize:
            x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)
        all_patches.append(forward_func(x, model).cpu())
        # patches.append(model(x).cpu())            
            
        #     # patches.append(input_processor(curr_patch))
            
        # all_patches += patches
    return torch.cat(all_patches), img_labels

def get_image_embeddings(images, input_processor, forward_func,model,device='cuda', not_normalize=False):
    results = []
    for _, image in tqdm(enumerate(images)):
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
    def __init__(self, samples: list, model, input_to_latent, input_processor, dataset_name, device: str = 'cpu'):
        self.samples = samples
        self.device = torch.device(device)
        self.batch_size = 128
        self.input_to_latent = input_to_latent
        self.image_size = 224 if type(samples[0]) == PIL.Image.Image else None
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

    def get_patches(self, n_patches, samples_hash, img_idx_ls=None, images=None, method="slic", not_normalize=False, use_mask=False, compute_img_emb=True):
        """Get patches from images using different segmentation methods."""
        if images is None:
            images = self.samples

        if os.path.exists(f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}{'_use_mask' if use_mask else ''}.pkl"):
            print("Loading cached patches")
            print(samples_hash)
            cached_data = utils.load(f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}{'_use_mask' if use_mask else ''}.pkl")
            
            # if len(cached_data) == 6:
            img_idx_ls, image_embs, patch_activations, masks, bboxes, img_for_patch = cached_data
            # else:
            #     image_embs, patch_activations, masks, bboxes, img_for_patch = cached_data
                # utils.save((img_idx_ls, image_embs, patch_activations, masks, bboxes, img_for_patch), f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}{'_use_mask' if use_mask else ''}.pkl")
            
            if image_embs is None and compute_img_emb:
                image_embs = get_image_embeddings(images, self.input_processor, self.input_to_latent, self.model, not_normalize=not_normalize)    
            return img_idx_ls, image_embs, patch_activations, masks, bboxes, img_for_patch
        if compute_img_emb:
            image_embs = get_image_embeddings(images, self.input_processor, self.input_to_latent, self.model, not_normalize=not_normalize)
        else:
            image_embs = None
        patch_activations = None
        masks = None
        bboxes = None
        img_for_patch = None
        # if method == "slic":
        masks = get_slic_segments(images, n_segments=n_patches)
        bboxes = masks_to_bboxes(masks)
        # get_patches_from_bboxes(model, images, all_bboxes, input_processor, sub_bboxes=None, image_size=(224, 224), processor=None, resize=None, device="cpu"):
        if not use_mask:
            patch_activations, img_for_patch = get_patches_from_bboxes(self.input_to_latent, self.model, images, bboxes, self.input_processor, device=self.device, resize=self.image_size)
        else:
            patch_activations, img_for_patch = get_patches_from_bboxes0(self.input_to_latent, self.model, images, masks, bboxes, self.input_processor, device=self.device, resize=self.image_size)
        
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
        if not os.path.exists(f"output/"):
            os.mkdir(f"output/")
        utils.save((img_idx_ls, image_embs, patch_activations, masks, bboxes, img_for_patch), f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}{'_use_mask' if use_mask else ''}.pkl")
        return img_idx_ls, image_embs, patch_activations, masks, bboxes, img_for_patch
    
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


def convert_samples_to_concepts_img(args, samples_hash, model, images, img_idx_ls, processor, device, patch_count_ls = [32]):
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
            curr_img_ls, curr_img_emb, patch_emb, masks, bboxes, img_per_patch = cl.get_patches(patch_count, samples_hash, img_idx_ls=img_idx_ls, images=images, method="slic", compute_img_emb=True)
        else:
            curr_img_ls, curr_img_emb, patch_emb, masks, bboxes, img_per_patch = cl.get_patches(patch_count, samples_hash, img_idx_ls=img_idx_ls, images=images, method="slic", compute_img_emb=False)
        if curr_img_emb is not None:
            img_emb = curr_img_emb
            img_ls = curr_img_ls
        patch_emb_ls.append(patch_emb)
        masks_ls.append(masks)
        img_per_batch_ls.append(img_per_patch)
        bboxes_ls.append(bboxes)
    return img_ls, img_emb, patch_emb_ls, masks_ls, bboxes_ls, img_per_batch_ls


def reformat_patch_embeddings(patch_emb_ls, img_per_patch_ls, img_emb):
    img_per_patch_tensor = torch.tensor(img_per_patch_ls[0])
    max_img_id = int(torch.max(img_per_patch_tensor).item())
    patch_emb_curr_img_ls = []
    for idx in tqdm(range(max_img_id + 1)):
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