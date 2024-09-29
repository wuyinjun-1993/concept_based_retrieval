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
from LLM4split.prompt_utils import obtain_response_from_openai, prompt_check_correctness, update_decomposed_queries
import math
import networkx as nx
from beir.retrieval import models
from transformers import BlipProcessor, BlipForConditionalGeneration

import json

import requests
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer, BitsAndBytesConfig
from parse_queries_to_trees import *
from agg_query_processing import Tree, TreeNode
# from llava.model import LlavaLlamaForCausalLM
# import torch
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.utils import disable_torch_init
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from transformers import TextStreamer

# def init_llava_model():
#     model_path = "liuhaotian/llava-v1.6-mistral-7b"
#     kwargs = {"device_map": "auto"}
#     kwargs['load_in_4bit'] = True
#     kwargs['quantization_config'] = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type='nf4'
#     )
#     model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

#     vision_tower = model.get_vision_tower()
#     if not vision_tower.is_loaded:
#         vision_tower.load_model()
#     vision_tower.to(device='cuda')
#     image_processor = vision_tower.image_processor
#     return image_processor, tokenizer, model

# def caption_image_llama(image_file, prompt, image_processor, tokenizer, model):
#     if image_file.startswith('http') or image_file.startswith('https'):
#         response = requests.get(image_file)
#         image = Image.open(BytesIO(response.content)).convert('RGB')
#     else:
#         image = Image.open(image_file).convert('RGB')
#     disable_torch_init()
#     conv_mode = "llava_v0"
#     conv = conv_templates[conv_mode].copy()
#     roles = conv.roles
#     image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
#     inp = f"{roles[0]}: {prompt}"
#     inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
#     conv.append_message(conv.roles[0], inp)
#     conv.append_message(conv.roles[1], None)
#     raw_prompt = conv.get_prompt()
#     input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
#     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
#     with torch.inference_mode():
#       output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
#                                   max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
#     outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
#     conv.messages[-1][-1] = outputs
#     output = outputs.rsplit('</s>', 1)[0]
#     return image, output

@dataclass
class Patch:
    image: PIL.Image
    bbox: tuple
    patch: PIL.Image

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
          (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def generate_caption_and_bboxes(img, scene_graph_model):
    CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

    REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
                
    # Some transformation functions
    transform = transforms.Compose([
      transforms.Resize(800),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    im = transform(img).unsqueeze(0)
    with torch.no_grad():
        captions = []
        outputs = scene_graph_model.forward(im)
        # keep only predictions with >0.3 confidence
        probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
        keep = torch.logical_and(probas.max(-1).values > 0.05, torch.logical_and(probas_sub.max(-1).values > 0.05,
                                                                        probas_obj.max(-1).values > 0.05))
        # convert boxes from [0; 1] to image scales
        #print(img.size)
        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], img.size)
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], img.size)
        topk = 100 # display up to 10 images
        keep_queries = torch.nonzero(keep, as_tuple=True)[0]
        indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
        keep_queries = keep_queries[indices]
        # get the feature map shape
        #print(img.size)
        im_w, im_h = img.size
        result = ''
        combined_bboxes = []

        if len(indices) >= 1:
          for idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
              val = str(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()])
              if val not in captions:
                # Add the caption to the list
                captions.append(val)
                combined_bboxes.append([[round(sxmin.item(),2), round(symin.item(),2), round(sxmax.item(),2), round(symax.item(),2)],
                 [round(oxmin.item(),2), round(oymin.item(),2), round(oxmax.item(),2), round(oymax.item(),2)]])

          result = '"' + '"|"'.join(captions) + '"'
          #print(result)
        else:
          result = ''
          #print(result)
        #print(result)
        #print(combined_bboxes)
        return (result, combined_bboxes)


def vit_forward(imgs, model, masks=None):
    # inputs = processor(imgs, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # Select the CLS token embedding from the last hidden layer
        # return model(pixel_values=imgs).last_hidden_state[:, 0, :]
        return model.get_image_features(pixel_values=imgs, output_hidden_states=True)
def blip_vit_forward(image, model):
    with torch.no_grad():
        return model.extract_features({"image": image}, mode="image").image_embeds_proj[:,0,:]


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

def load_flickr_dataset(data_path, query_path, subset_img_id=None, redecompose=False):
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
    all_grouped_sub_q_ids_ls = []
    
    if 'caption_triples_ls' not in caption_pd.columns:
        caption_pd['caption_triples_ls'] = np.nan
    if "groups" not in caption_pd.columns:
        caption_pd['groups'] = np.nan 

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
        
        
        if pd.isnull(caption_pd.iloc[idx]['caption_triples_ls']) or redecompose:
            sub_caption_str=obtain_response_from_openai(dataset_name="flickr", query=caption)
            caption_pd.at[idx, "caption_triples_ls"] = sub_caption_str
        else:
            sub_caption_str = caption_pd.iloc[idx]['caption_triples_ls']
        
        sub_captions = decompose_single_query_ls(sub_caption_str)
        
        query_paritions_str = caption_pd.iloc[idx]['groups']
        grouped_sub_q_ids_ls = decompose_single_query_parition_groups(sub_captions, query_paritions_str)
        print(sub_captions)
        # img_ls.append(img)
        img_idx_ls.append(image_idx)
        caption_ls.append(caption)
        sub_caption_ls.append(sub_captions)
        all_grouped_sub_q_ids_ls.append(grouped_sub_q_ids_ls)
        
    if subset_img_id is None:
        return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls, all_grouped_sub_q_ids_ls
    else:
        print(sub_caption_ls[subset_img_id])
        return [caption_ls[subset_img_id]], [img_file_name_ls[subset_img_id]], [sub_caption_ls[subset_img_id]], [img_idx_ls[subset_img_id]], [all_grouped_sub_q_ids_ls[subset_img_id]]

    # return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls, all_grouped_sub_q_ids_ls

def load_flickr_dataset_full(data_path, query_path, subset_img_id=None, redecompose=False, total_count = 1000, algebra_method="five"):
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all4.csv")
    
    # img_caption_file_name= os.path.join(query_path, "sub_queries.csv")
    
    selected_query_file = os.path.join(data_path, "flickr_500_choose.csv")   
    query_df = pd.read_csv(selected_query_file)
    df = pd.read_csv(os.path.join(data_path, 'flickr_annotations_30k.csv'))
    
    # selected_filename = "1000092795.jpg"
    # selected_filename = "10002456.jpg"
    # selected_filename = "125272627.jpg"
    # selected_filename = "1000366164.jpg"
    
    selected_img_idx = 30 #df[df['filename'] == selected_filename].index[0]
    
    # full_query = "Two men in Germany jumping over a rail at the same time without shirts."
    # full_query = "Several women are gather around a table in a corner surrounded by bookshelves."
    # selected_img_idx = 11
    
    
    image_path_11 = os.path.join(data_path, "flickr30k-images/") + query_df.iloc[selected_img_idx]['filename']
    
    full_query = query_df.iloc[selected_img_idx]['choose']
    origin_file_idx = df[df['filename'] == query_df.iloc[selected_img_idx]['filename']].index[0]
    print("full origin query::", full_query)
    print("selected image::", query_df.iloc[selected_img_idx]['filename'])
    print("origin file idx::", origin_file_idx)
    if algebra_method == "five":
    
        root = TreeNode(0, full_query, 'root', 1)
        child1 = TreeNode(1, 'a table in a corner', 'plain', 1)
        child2 = TreeNode(2, 'a corner surrounded by bookshelves', 'plain', 1)
        child3 = TreeNode(3, 'One woman is gathered around a table', 'count', 2)
        child4 = TreeNode(4, 'One woman is gathered around a table', 'plain', 1)
        tree_11 = Tree(root)
        tree_11.add_child(root, child1)
        tree_11.add_child(root, child2)
        tree_11.add_child(root, child3)
        tree_11.add_child(child3, child4)
    
    
    
        # root = TreeNode(0, 'Several men in hard hats are operating a giant pulley system.', 'root', 1)
        # child1 = TreeNode(1, 'One man in hard hats are operating a giant pulley system.', 'count', 2)
        # child2 = TreeNode(2, 'One man in hard hats', 'plain', 1)
        # child3 = TreeNode(3, 'a man is operating a giant pulley system', 'plain', 1)
        # # child4 = TreeNode(4, 'One young guy hanging out in the yard', 'plain', 1)
        # tree_11 = Tree(root)
        # tree_11.add_child(root, child1)
        # tree_11.add_child(child1, child2)
        # tree_11.add_child(child1, child3)
        # tree_11.add_child(child1, child4)
        
        
        
        
        
        # root = TreeNode(0, 'Two young guys with shaggy hair look at their hands while hanging out in the yard.', 'root', 1)
        # child1 = TreeNode(1, 'One young guys with shaggy hair look at their hands while hanging out in the yard.', 'count', 2)
        # child2 = TreeNode(2, 'One young guys with shaggy hair', 'plain', 1)
        # child3 = TreeNode(3, 'One young guy looks at his hands', 'plain', 1)
        # child4 = TreeNode(4, 'One young guy hanging out in the yard', 'plain', 1)
        # tree_11 = Tree(root)
        # tree_11.add_child(root, child1)
        # tree_11.add_child(child1, child2)
        # tree_11.add_child(child1, child3)
        # tree_11.add_child(child1, child4)
        
        # tree_11.add_child(child1, child5)
        # tree_11.add_child(child1, child6)
        
        # root = TreeNode(0, 'Three young men and a young woman wearing sneakers are leaping in midair at the top of a flight of concrete stairs.', 'root', 1)
        # child1 = TreeNode(1, 'Three young men wearing sneakers are leaping in midair at the top of a flight of concrete stairs.', 'count', 3)
        # child2 = TreeNode(2, 'A young woman wearing sneakers.', 'plain', 1)
        # child3 = TreeNode(3, 'A young woman leaping in midair at the top of a flight of concrete stairs.', 'plain', 1)
        # # child4 = TreeNode(4, 'A flight of concrete stairs.', 'plain', 1)
        # child5 = TreeNode(5, 'A young man wearing sneakers.', 'plain', 1)
        # child6 = TreeNode(6, 'A young man leaping in midair at the top of a flight of concrete stairs.', 'plain', 1)
        # tree_11 = Tree(root)
        # tree_11.add_child(root, child1)
        # tree_11.add_child(root, child3)
        # tree_11.add_child(root, child2)
        # # tree_11.add_child(root, child4)
        # tree_11.add_child(child1, child5)
        # tree_11.add_child(child1, child6)
        
        
        
        # Display the tree
        tree_11.display(tree_11.root)
        sub_caption_ls = [[tree_11]]
        all_grouped_sub_q_ids_ls = [None]
    else:
        # sub_caption_ls = [['A young woman wearing sneakers.', 'A young woman leaping in midair at the top of a flight of concrete stairs.', 'Three young men wearing sneakers.', 'Three young men leaping in midair at the top of a flight of concrete stairs.']]
        # sub_caption_ls = [['Two young guys with shaggy hair', 'Two young guy looks at his hands', 'Two young guy hanging out in the yard']]
        sub_caption_ls = [["Several women are gather around a table","a table in a corner","a corner surrounded by bookshelves"]]
        # all_grouped_sub_q_ids_ls = [[[0,1], [2,3]]]
        all_grouped_sub_q_ids_ls = [None]
    
    # caption_ls = ["Two young guys with shaggy hair look at their hands while hanging out in the yard."]
    caption_ls = [full_query]
    
    img_file_name_ls = [image_path_11]
    
    # img_idx_ls = [df.iloc[11]['filename']]
    img_idx_ls = [origin_file_idx]
    
    return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls, all_grouped_sub_q_ids_ls
    
    
def load_flickr_dataset_full0(data_path, query_path, subset_img_id=None, redecompose=False, total_count = 1000, algebra_method="five"):
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all4.csv")
    
    # img_caption_file_name= os.path.join(query_path, "sub_queries.csv")
    selected_query_file_name = "flickr_500_choose.csv"
    decomposed_res_file_name = "flickr_500_result.txt"
    # simple_decomposed_res_file_name = "flickr_500_simple_decompose_result.txt"
    simple_decomposed_res_file_name = "flickr_simple_decompose_result_240928.json"
    simple_decomposed_dependency_file_name = "flickr_simple_decompose_dependency_240928.json"
    # selected_query_file_name = "flickr_adjust.csv"
    # decomposed_res_file_name = "flickr_adjust_result.txt"
    # simple_decomposed_res_file_name = "flickr_adjust_simple_decompose_result.txt"
    
    
    selected_query_file = os.path.join(data_path, selected_query_file_name)   
    query_df = pd.read_csv(selected_query_file)
    df = pd.read_csv(os.path.join(data_path, 'flickr_annotations_30k.csv'))
    
    # selected_filename = "1000092795.jpg"
    # selected_filename = "10002456.jpg"
    # selected_filename = "125272627.jpg"
    # selected_filename = "1000366164.jpg"
    sub_caption_ls = []
    all_grouped_sub_q_ids_ls = [None]
    query_count=100
    if subset_img_id is not None:
    
        selected_img_idx = subset_img_id #df[df['filename'] == selected_filename].index[0]
        
        # full_query = "Two men in Germany jumping over a rail at the same time without shirts."
        # full_query = "Several women are gather around a table in a corner surrounded by bookshelves."
        # selected_img_idx = 11
        
        
        image_path_11 = os.path.join(data_path, "flickr30k-images/") + query_df.iloc[selected_img_idx]['filename']
        
        full_query = query_df.iloc[selected_img_idx]['choose']
        origin_file_idx = df[df['filename'] == query_df.iloc[selected_img_idx]['filename']].index[0]
        print("full origin query::", full_query)
        print("selected image::", query_df.iloc[selected_img_idx]['filename'])
        print("origin file idx::", origin_file_idx)
        caption_ls = [full_query]
    
        img_file_name_ls = [image_path_11]
        
        # img_idx_ls = [df.iloc[11]['filename']]
        img_idx_ls = [origin_file_idx]
        
        if algebra_method == "five":
        
            decomposed_res_file = os.path.join(data_path, decomposed_res_file_name)
            sub_caption_ls = []
            line_idx = 0
            
            root = TreeNode(0, full_query, 'root', 1)
            # child1 = TreeNode(1, 'One young guys with shaggy hair look at their hands while hanging out in the yard.', 'count', 2)
            child1 = TreeNode(1, "One woman is standing", 'count', 2)
            child2 = TreeNode(2, 'a bus with buildings behind it', 'plain', 1)
            child3 = TreeNode(3, 'One woman is standing', 'plain', 1)
            tree = Tree(root)
            tree.add_child(root, child1)
            tree.add_child(root, child2)
            tree.add_child(child1, child3)
            
            root = TreeNode(0, full_query, 'root', 1)
            # child1 = TreeNode(1, 'One young guys with shaggy hair look at their hands while hanging out in the yard.', 'count', 2)
            child1 = TreeNode(1, "Several women are standing", 'plain', 1)
            child2 = TreeNode(2, 'a bus with buildings behind it', 'plain', 1)
            # child3 = TreeNode(3, 'One woman is standing', 'plain', 1)
            tree2 = Tree(root)
            tree2.add_child(root, child1)
            tree2.add_child(root, child2)
            
            # root = TreeNode(0, 'Two young guys with shaggy hair look at their hands while hanging out in the yard.', 'root', 1)
            # child1 = TreeNode(1, 'One young guys with shaggy hair look at their hands while hanging out in the yard.', 'count', 2)
            # child2 = TreeNode(2, 'Two young guys with shaggy hair', 'plain', 1)
            # child3 = TreeNode(3, 'Two young guy looks at his hands', 'plain', 1)
            # child4 = TreeNode(4, 'Two young guy hanging out in the yard', 'plain', 1)
            # tree2 = Tree(root)
            # tree2.add_child(root, child1)
            # tree2.add_child(child1, child2)
            # tree2.add_child(child1, child3)
            # tree2.add_child(child1, child4)
            # tree_11.add_child(child1, child4)
            
            # with open(decomposed_res_file, 'r') as f:
            #     for line in f:
                    

            #         # parsed_nodes = parse_nodes(line)
            #         # root = build_tree(parsed_nodes, 0)
            #         # tree = Tree(root)

            #         # tree_11.display(tree_11.root)
                    
            #         if line_idx == subset_img_id:
            #             tree = construct_tree_from_string(line)
            sub_caption_ls.append([tree, tree2])
            #             break
            #         line_idx += 1
            all_grouped_sub_q_ids_ls = None
        else:
            # sub_caption_ls = [['A young woman wearing sneakers.', 'A young woman leaping in midair at the top of a flight of concrete stairs.', 'Three young men wearing sneakers.', 'Three young men leaping in midair at the top of a flight of concrete stairs.']]
            sub_caption_ls = [[['One young guy with shaggy hair', 'One young guy looks at his hands', 'One young guy is hanging out in the yard', 'Two young guys with shaggy hair', 'Two young guys looks at his hands', 'Two young guys is hanging out in the yard']]]
            # sub_caption_ls = [["Two women, both wearing glasses","Two women are playing clarinets","an elderly woman is playing a stringed instrument"]]
            # sub_caption_ls = [[[full_query]]]
            # all_grouped_sub_q_ids_ls = [[[0,1], [2,3]]]
            all_grouped_sub_q_ids_ls = None
    else:
        img_file_name_ls = []
        img_idx_ls = []
        caption_ls = []
        for selected_img_idx in range(query_count):
            image_path_11 = os.path.join(data_path, "flickr30k-images/") + query_df.iloc[selected_img_idx]['filename']
            origin_file_idx = df[df['filename'] == query_df.iloc[selected_img_idx]['filename']].index[0]
            full_query = query_df.iloc[selected_img_idx]['choose']
            img_idx_ls.append(origin_file_idx)
            img_file_name_ls.append(image_path_11)
            caption_ls.append(full_query)
        all_grouped_sub_q_ids_ls = None
        
        if algebra_method == "five":
        
            decomposed_res_file = os.path.join(data_path, decomposed_res_file_name)
            sub_caption_ls = []
            with open(decomposed_res_file, 'r') as f:
                for line in f:
                    tree = construct_tree_from_string(line)

                    # parsed_nodes = parse_nodes(line)
                    # root = build_tree(parsed_nodes, 0)
                    # tree = Tree(root)

                    # tree_11.display(tree_11.root)
                    sub_caption_ls.append([tree])
                    # sub_caption_ls.append([])
                    if len(sub_caption_ls) == query_count:
                        break
            
        # else:
        decomposed_res_file = os.path.join(data_path, simple_decomposed_res_file_name)
        origin_sub_caption_ls = []
        with open(decomposed_res_file, 'r') as f:
            json_obj = json.load(f)
            for line_idx in range(query_count):
                decomposed_subqueries = json_obj[str(line_idx+1)]
                # sub_caption_ls.append(decomposed_subqueries)
                
                
                if algebra_method == "five":
                    # sub_caption_ls[len(origin_sub_caption_ls)][0].decomposed_subqueries = decomposed_subqueries
                    root = TreeNode(1, caption_ls[len(origin_sub_caption_ls)], 'root', 1)
                    tree = Tree(root)
                    for subquery_idx in range(len(decomposed_subqueries[0])):
                        subquery=decomposed_subqueries[0][subquery_idx]
                        node = TreeNode(subquery_idx, subquery, 'plain', 1)
                        tree.add_child(root, node)
                    sub_caption_ls[len(origin_sub_caption_ls)].append(tree)
                origin_sub_caption_ls.append(decomposed_subqueries)
                if len(origin_sub_caption_ls) == query_count:
                    break
                
        decomposed_dependency_res_file = os.path.join(data_path, simple_decomposed_dependency_file_name)
        
        all_grouped_sub_q_ids_ls = []
        
        with open(decomposed_dependency_res_file, 'r') as f:
            json_obj = json.load(f)
            for line_idx in range(query_count):
                decomposed_subqueries_dependency = json_obj[line_idx]
                
                all_sub_query_ids = []
                for sub_query_idx in range(len(decomposed_subqueries_dependency[0])):
                    all_sub_query_ids.extend(decomposed_subqueries_dependency[0][sub_query_idx])
                
                assert len(all_sub_query_ids) == len(origin_sub_caption_ls[line_idx][0])
                
                all_grouped_sub_q_ids_ls.append(decomposed_subqueries_dependency)
                if len(all_grouped_sub_q_ids_ls) == query_count:
                    break
            # for line in f:
                # decomposed_subqueries = [line.strip().split("|")]
                # if algebra_method == "five":
                #     # sub_caption_ls[len(origin_sub_caption_ls)][0].decomposed_subqueries = decomposed_subqueries
                #     root = TreeNode(1, caption_ls[len(origin_sub_caption_ls)], 'root', 1)
                #     tree = Tree(root)
                #     for subquery_idx in range(len(decomposed_subqueries[0])):
                #         subquery=decomposed_subqueries[0][subquery_idx]
                #         node = TreeNode(subquery_idx, subquery, 'plain', 1)
                #         tree.add_child(root, node)
                #     sub_caption_ls[len(origin_sub_caption_ls)].append(tree)
                # origin_sub_caption_ls.append(decomposed_subqueries)
                # if len(origin_sub_caption_ls) == query_count:
                #     break
            
        if not algebra_method == "five":
            sub_caption_ls = origin_sub_caption_ls
            # sub_caption_ls = [['A young woman wearing sneakers.', 'A young woman leaping in midair at the top of a flight of concrete stairs.', 'Three young men wearing sneakers.', 'Three young men leaping in midair at the top of a flight of concrete stairs.']]
            # sub_caption_ls = [[['One young guy with shaggy hair', 'One young guy looks at his hands', 'One young guy is hanging out in the yard']]]
            # sub_caption_ls = [["Two women, both wearing glasses","Two women are playing clarinets","an elderly woman is playing a stringed instrument"]]
            # sub_caption_ls = [[[full_query]]]#[["Two young guys with shaggy hair| guys look at their hands| guys hanging out in the yard."]]
            # all_grouped_sub_q_ids_ls = [[[0,1], [2,3]]]
            # all_grouped_sub_q_ids_ls = None

        # root = TreeNode(0, full_query, 'root', 1)
        # Two women, both wearing glasses, are playing clarinets and an elderly woman is playing a stringed instrument.
        # child1 = TreeNode(1, 'Two women, both wearing glasses, are playing clarinets', 'count', 2)
        # child2 = TreeNode(2, 'One women wearing glasses', 'plain', 1)
        # child3 = TreeNode(3, 'One women is playing clarinets', 'plain', 1)
        # child4 = TreeNode(4, 'an elderly woman is playing a stringed instrument', 'plain', 1)
        
        # # tree_11 = Tree(root)
        # # tree_11.add_child(root, child1)
        # # tree_11.add_child(root, child4)
        # # tree_11.add_child(child1, child2)
        # # tree_11.add_child(child1, child3)
        
        # # Several men in hard hats are operating a giant pulley system.
        # child1 = TreeNode(1, 'One man in hard hats are operating a giant pulley system', 'count', 2)
        # child2 = TreeNode(2, 'One man in hard hats', 'plain', 1)
        # child3 = TreeNode(3, 'One man is operating a giant pulley system', 'plain', 1)
        # # child4 = TreeNode(4, 'one men standing near a stove', 'plain', 1)
        
        # tree_11 = Tree(root)
        # tree_11.add_child(root, child1)
        # tree_11.add_child(child1, child2)
        # tree_11.add_child(child1, child3)
        # tree_11.add_child(child3, child4)
        # tree_11.add_child(child1, child3)
    
    
    
    #     # root = TreeNode(0, 'Several men in hard hats are operating a giant pulley system.', 'root', 1)
    #     # child1 = TreeNode(1, 'One man in hard hats are operating a giant pulley system.', 'count', 2)
    #     # child2 = TreeNode(2, 'One man in hard hats', 'plain', 1)
    #     # child3 = TreeNode(3, 'a man is operating a giant pulley system', 'plain', 1)
    #     # # child4 = TreeNode(4, 'One young guy hanging out in the yard', 'plain', 1)
    #     # tree_11 = Tree(root)
    #     # tree_11.add_child(root, child1)
    #     # tree_11.add_child(child1, child2)
    #     # tree_11.add_child(child1, child3)
    #     # tree_11.add_child(child1, child4)
        
        
        
        
        
    #     # root = TreeNode(0, 'Two young guys with shaggy hair look at their hands while hanging out in the yard.', 'root', 1)
    #     # child1 = TreeNode(1, 'One young guys with shaggy hair look at their hands while hanging out in the yard.', 'count', 2)
    #     # child2 = TreeNode(2, 'One young guys with shaggy hair', 'plain', 1)
    #     # child3 = TreeNode(3, 'One young guy looks at his hands', 'plain', 1)
    #     # child4 = TreeNode(4, 'One young guy hanging out in the yard', 'plain', 1)
    #     # tree_11 = Tree(root)
    #     # tree_11.add_child(root, child1)
    #     # tree_11.add_child(child1, child2)
    #     # tree_11.add_child(child1, child3)
    #     # tree_11.add_child(child1, child4)
        
    #     # tree_11.add_child(child1, child5)
    #     # tree_11.add_child(child1, child6)
        
    #     # root = TreeNode(0, 'Three young men and a young woman wearing sneakers are leaping in midair at the top of a flight of concrete stairs.', 'root', 1)
    #     # child1 = TreeNode(1, 'Three young men wearing sneakers are leaping in midair at the top of a flight of concrete stairs.', 'count', 3)
    #     # child2 = TreeNode(2, 'A young woman wearing sneakers.', 'plain', 1)
    #     # child3 = TreeNode(3, 'A young woman leaping in midair at the top of a flight of concrete stairs.', 'plain', 1)
    #     # # child4 = TreeNode(4, 'A flight of concrete stairs.', 'plain', 1)
    #     # child5 = TreeNode(5, 'A young man wearing sneakers.', 'plain', 1)
    #     # child6 = TreeNode(6, 'A young man leaping in midair at the top of a flight of concrete stairs.', 'plain', 1)
    #     # tree_11 = Tree(root)
    #     # tree_11.add_child(root, child1)
    #     # tree_11.add_child(root, child3)
    #     # tree_11.add_child(root, child2)
    #     # # tree_11.add_child(root, child4)
    #     # tree_11.add_child(child1, child5)
    #     # tree_11.add_child(child1, child6)
        
        
        
    #     # Display the tree
        # tree_11.display(tree_11.root)
        # sub_caption_ls = [[tree_11]]
        # all_grouped_sub_q_ids_ls = [None]    
    # caption_ls = ["Two young guys with shaggy hair look at their hands while hanging out in the yard."]
    
    
    return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls, all_grouped_sub_q_ids_ls
    
    
    
    # if total_count < 0:
    #     img_caption_file_name= os.path.join(query_path, "full_queries.csv")
    # else:
    #     img_caption_file_name= os.path.join(query_path, "full_queries_" + str(total_count) + ".csv")
    # is_full_query_file = True
    # if not os.path.exists(img_caption_file_name):
    #     is_full_query_file = False
    #     img_caption_file_name= os.path.join(query_path, "results_20130124.token")
    #     caption_pd = pd.read_csv(img_caption_file_name, sep="\t")
    # else:
    #     caption_pd = pd.read_csv(img_caption_file_name)



    # img_folder = os.path.join(data_path, "flickr30k-images/")
    # # img_folder2 = os.path.join(data_path, "VG_100K_2/")

    # if not is_full_query_file:
    #     header_names = ["image_id", "caption"]
    # else:
    #     # if len(caption_pd.columns) == 3:
    #     header_names = ["image_id", "caption", "caption_triples_ls", "groups"]
            
    # caption_pd.columns = header_names[0:len(caption_pd.columns)]
    
    
    # # img_ls = []
    # img_idx_ls = []
    # caption_ls = []
    # sub_caption_ls = []
    # img_file_name_ls = []
    # all_grouped_sub_q_ids_ls = []
    
    # if 'caption_triples_ls' not in caption_pd.columns:
    #     caption_pd['caption_triples_ls'] = np.nan
    # if "groups" not in caption_pd.columns:
    #     caption_pd['groups'] = np.nan 

    # for idx in tqdm(range(len(caption_pd))):
    #     image_idx = caption_pd.iloc[idx]['image_id']
    #     image_idx = image_idx.split("#")[0]
    #     if image_idx in img_idx_ls:
    #         continue
        
    #     full_img_file_name = os.path.join(img_folder, str(image_idx))
    #     # if not os.path.exists(full_img_file_name):
    #     #     full_img_file_name = os.path.join(img_folder2, str(image_idx) + ".jpg")
    #     if not os.path.exists(full_img_file_name):
    #         continue
    #     img_file_name_ls.append(full_img_file_name)
        
    #     # img = Image.open(full_img_file_name)
    #     # img = img.convert('RGB')
    #     caption = caption_pd.iloc[idx]['caption']
    #     # sub_caption_str = caption_pd.iloc[idx]['caption_triples']
    #     sub_caption_str = caption_pd.iloc[idx]['caption_triples_ls']
    #     # sub_captions = decompose_single_query(sub_caption_str)
        
        
    #     if pd.isnull(caption_pd.iloc[idx]['caption_triples_ls']) or redecompose:
    #         sub_caption_str=obtain_response_from_openai(dataset_name="flickr", query=caption)
    #         sub_caption_str_ls = sub_caption_str.split("|")
    #         segmented_sub_caption_str_ls = []
    #         for sub_caption_str in sub_caption_str_ls:
    #             segmented_sub_caption_str_ls.append(obtain_response_from_openai(dataset_name="flickr_two", query=sub_caption_str))
            
    #         sub_caption_str = "|".join(segmented_sub_caption_str_ls)
    #         caption_pd.at[idx, "caption_triples_ls"] = sub_caption_str
    #     else:
    #         sub_caption_str = caption_pd.iloc[idx]['caption_triples_ls']
        
    #     sub_captions = decompose_single_query_ls(sub_caption_str)
        
    #     query_paritions_str = caption_pd.iloc[idx]['groups']
    #     grouped_sub_q_ids_ls = decompose_single_query_parition_groups(sub_captions, query_paritions_str)
    #     print(sub_captions)
    #     # img_ls.append(img)
    #     img_idx_ls.append(image_idx)
    #     caption_ls.append(caption)
    #     sub_caption_ls.append(sub_captions)
    #     all_grouped_sub_q_ids_ls.append(grouped_sub_q_ids_ls)
        
    #     if total_count > 0 and len(img_file_name_ls) >= total_count:
    #         break

    # caption_pd.to_csv(img_caption_file_name, index=False)
    # if subset_img_id is None:
    #     return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls, all_grouped_sub_q_ids_ls
    # else:
    #     print(sub_caption_ls[subset_img_id])
    #     return [caption_ls[subset_img_id]], [img_file_name_ls[subset_img_id]], [sub_caption_ls[subset_img_id]], [img_idx_ls[subset_img_id]], [all_grouped_sub_q_ids_ls[subset_img_id]]

    # return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls, all_grouped_sub_q_ids_ls


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


def init_blip_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
    return model, processor

def load_mscoco_text_datasets(data_path, query_path, img_idx_ls, cached_caption_file_name="mscoco40k_blip_captioning.pkl", data_file_name="mscoco_40kdecomposed_dependencies_wholeexp.pkl"):
    # img_caption_file_name = query_path
    # with open(img_caption_file_name, 'rb') as f:
    # caption_pd = pickle.load(f)
    
    cached_blip_caption_file_name = os.path.join(query_path, cached_caption_file_name)
    
    if os.path.exists(cached_blip_caption_file_name):
        img_idx_caption_text_mappings = utils.load(cached_blip_caption_file_name)
        caption_text_ls=[]
        for img_id in tqdm(img_idx_ls, desc="Loading captions"):
        
            caption_text_ls.append(img_idx_caption_text_mappings[img_id])
        return caption_text_ls
    else:
        caption_pd = utils.load(os.path.join(data_path, data_file_name))
            
        caption_text_ls = []
        img_idx_caption_text_mappings = dict()
        model, processor = init_blip_captioning_model()
        for img_id in tqdm(img_idx_ls, desc="Loading captions"):
            
            img_file_name = os.path.join(data_path, "images/train2017/", caption_pd[caption_pd['id'] == img_id]["image"].values[0])
            raw_image = Image.open(img_file_name).convert('RGB')

            # conditional image captioning
            text = "a photography of"
            inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

            out = model.generate(**inputs)
            curr_text = processor.decode(out[0], skip_special_tokens=True)
            
            # curr_text = list(caption_pd[caption_pd['id'] == img_id]["caption_sharegpt4v"])[0]
            caption_text_ls.append(curr_text)
            img_idx_caption_text_mappings[img_id] = curr_text
            
            
        utils.save(img_idx_caption_text_mappings, cached_blip_caption_file_name) 
        
        return caption_text_ls

def load_mscoco_datasets_from_cached_files(data_path, query_path):
    # img_caption_file_name = query_path
    # with open(img_caption_file_name, 'rb') as f:
    # caption_pd = pickle.load(f)
    
    caption_pd = utils.load(os.path.join(query_path, "mscoco_40kdecomposed_dependencies_wholeexp.pkl"))
    
    # caption_pd = pd.read_csv(os.path.join(query_path, "sharegpt_query_20.csv"))
    img_file_name_ls = []
    img_idx_ls = []
    caption_ls = []
    sub_caption_ls = []
    all_grouped_sub_q_ids_ls = []
    # if 'caption_triples_ls' not in caption_pd.columns:
    #     caption_pd['caption_triples_ls'] = np.nan
    # if "groups" not in caption_pd.columns:
    #    caption_pd['groups'] = np.nan 
    for idx in tqdm(range(len(caption_pd))):
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
        
        
        query_paritions_str = caption_pd.iloc[idx]['groups']
        grouped_sub_q_ids_ls = decompose_single_query_parition_groups(sub_captions, query_paritions_str)
        
        
        img_file_name_ls.append(img_final_path)
        img_idx_ls.append(image_idx)
        caption_ls.append(caption)
        sub_caption_ls.append(sub_captions)
        all_grouped_sub_q_ids_ls.append(grouped_sub_q_ids_ls)
    return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls, all_grouped_sub_q_ids_ls


def load_mscoco_120k_datasets_from_cached_files(data_path, query_path):
    # img_caption_file_name = query_path
    # with open(img_caption_file_name, 'rb') as f:
    # caption_pd = pickle.load(f)
    
    # caption_pd = utils.load(os.path.join(query_path, "mscoco_120kdecomposed_wholeexp.pkl"))
    caption_pd = utils.load(os.path.join(query_path, "sharegpt4v_random_1000_queries.pkl"))
    
    # caption_pd = pd.read_csv(os.path.join(query_path, "sharegpt_query_20.csv"))
    img_file_name_ls = []
    img_idx_ls = []
    caption_ls = []
    sub_caption_ls = []
    all_grouped_sub_q_ids_ls = []
    # if 'caption_triples_ls' not in caption_pd.columns:
    #     caption_pd['caption_triples_ls'] = np.nan
    # if "groups" not in caption_pd.columns:
    #    caption_pd['groups'] = np.nan 
    for idx in tqdm(range(len(caption_pd))):
        image_idx = caption_pd.iloc[idx]['id']
        if image_idx in img_idx_ls:
            continue

        img_path = caption_pd.iloc[idx]['image']
        #image_dir is the directory in which the root folder for the main image files are located, in this case in train2017 folder
        # image_dir = '/content/unzipped_images/train2017/train2017/'
        img_final_path = os.path.join(data_path, "images/train2017/", img_path)
        caption = caption_pd.iloc[idx]['caption_sharegpt4v']
        try:
            sub_caption_str = caption_pd.iloc[idx]['sentence_decompositions']
            sub_captions = decompose_single_query_ls(sub_caption_str)
            
            
            query_paritions_str = caption_pd.iloc[idx]['sentence_level_groups']
            grouped_sub_q_ids_ls = decompose_single_query_parition_groups(sub_captions, query_paritions_str)
            
            
            img_file_name_ls.append(img_final_path)
            img_idx_ls.append(image_idx)
            caption_ls.append(caption)
            sub_caption_ls.append(sub_captions)
            all_grouped_sub_q_ids_ls.append(None)#(grouped_sub_q_ids_ls)
        except:
            pass
    return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls, all_grouped_sub_q_ids_ls




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


def load_crepe_datasets(data_path, query_path, subset_img_id=None, redecompose=False, is_test=False):
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all4.csv")
    
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all6_2.csv")
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all.csv")
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all_output.csv")
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all_output3.csv")
    if is_test:
        img_caption_file_name = os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all_output_with_dependency_example.csv")
    else:
        img_caption_file_name = os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all_output_with_dependency3.csv")

    with open(os.path.join(query_path, "prod_hard_negatives/selected_img_id_ls"), "rb") as f:
        selected_img_id_ls = pickle.load(f)
    # selected_img_id_ls.append(0)
    img_folder = os.path.join(data_path, "VG_100K/")
    img_folder2 = os.path.join(data_path, "VG_100K_2/")

    caption_pd = pd.read_csv(img_caption_file_name)
    
    # img_ls = []
    img_idx_ls = []
    caption_ls = []
    sub_caption_ls = []
    img_file_name_ls = []
    all_grouped_sub_q_ids_ls = []
    if 'caption_triples_ls' not in caption_pd.columns:
        caption_pd['caption_triples_ls'] = np.nan
    if "groups" not in caption_pd.columns:
       caption_pd['groups'] = np.nan 

    for idx in tqdm(range(len(caption_pd))):
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
        # sub_caption_str = caption_pd.iloc[idx]['caption_triples_ls']
        if pd.isnull(caption_pd.iloc[idx]['caption_triples_ls']) or redecompose:
            sub_caption_str=obtain_response_from_openai(query=caption)
            sub_caption_str = sub_caption_str.strip()
            if sub_caption_str[-1] == "|":
                sub_caption_str = sub_caption_str[:-1]
            if sub_caption_str[-2] == "|":
                sub_caption_str = sub_caption_str[:-2]
            
            if not prompt_check_correctness(caption, sub_caption_str):
                sub_caption_str = update_decomposed_queries(caption, sub_caption_str)
            
            # sub_caption_str = prompt_check_correctness(caption, sub_caption_str)
            caption_pd.at[idx, "caption_triples_ls"] = sub_caption_str
        else:
            sub_caption_str = caption_pd.iloc[idx]['caption_triples_ls']
            
            sub_caption_str = sub_caption_str.strip()
            if sub_caption_str[-1] == "|":
                sub_caption_str = sub_caption_str[:-1]
            if sub_caption_str[-2] == "|":
                sub_caption_str = sub_caption_str[:-2]
            
            # if not prompt_check_correctness(caption, sub_caption_str):
            #     sub_caption_str = update_decomposed_queries(caption, sub_caption_str)
            # sub_caption_str = sub_caption_str.strip()
            # if sub_caption_str[-1] == "|":
            #     sub_caption_str = sub_caption_str[:-1]
            # if sub_caption_str[-2] == "|":
            #     sub_caption_str = sub_caption_str[:-2]
            caption_pd.at[idx, "caption_triples_ls"] = sub_caption_str

            
            
        
        # sub_captions = decompose_single_query(sub_caption_str)
        sub_captions = decompose_single_query_ls(sub_caption_str)
        
        query_paritions_str = caption_pd.iloc[idx]['groups']
        grouped_sub_q_ids_ls = decompose_single_query_parition_groups(sub_captions, query_paritions_str)
        # print(sub_captions)
        # img_ls.append(img)
        img_idx_ls.append(image_idx)
        caption_ls.append(caption)
        sub_caption_ls.append(sub_captions)
        img_file_name_ls.append(full_img_file_name)
        all_grouped_sub_q_ids_ls.append(grouped_sub_q_ids_ls)
        
    # img_caption_file_name_output= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all_output3.csv")
    # caption_pd.to_csv(img_caption_file_name_output, index=False)
    # return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls
    if subset_img_id is None:
        return caption_ls, img_file_name_ls, sub_caption_ls, img_idx_ls, all_grouped_sub_q_ids_ls
    else:
        # print(sub_caption_ls[subset_img_id])
        return [caption_ls[subset_img_id]], [img_file_name_ls[subset_img_id]], [sub_caption_ls[subset_img_id]], [img_idx_ls[subset_img_id]], [all_grouped_sub_q_ids_ls[subset_img_id]]


def load_crepe_text_datasets(data_path, query_path, img_idx_ls):
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all4.csv")
    
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all6_2.csv")
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all.csv")
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all_output.csv")
    # img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all_output3.csv")
    img_caption_file_name = os.path.join(data_path, "df_crepe_500_llava_captions.pkl")

    with open(img_caption_file_name, "rb") as f:
        caption_pd = pickle.load(f)
        
    caption_text_ls = []
    for img_id in img_idx_ls:
        curr_text = list(caption_pd[caption_pd['image_id'] == img_id]["llava_caption_text"])[0]
        caption_text_ls.append(curr_text)
        
    return caption_text_ls
    
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

    with open(os.path.join(query_path, "prod_hard_negatives/selected_img_id_ls"), "rb") as f:
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
    
    # filename_cap_mappings = read_flickr_image_captions(os.path.join(data_path, "results_20130124.token"))    

    df = pd.read_csv(os.path.join(data_path, 'flickr_annotations_30k.csv'))
    
    # img_folder = os.path.join(data_path, "VG_100K/")
    # img_folder = os.path.join(data_path, "flickr30k-images/")
    # img_folder2 = os.path.join(data_path, "VG_100K_2/")

    # caption_pd = pd.read_csv(img_caption_file_name)
    if total_count > 0 and len(img_file_name_ls) >= total_count:
        return img_idx_ls, img_file_name_ls          
    # for idx in range(len(caption_pd)):
    # for image_idx in tqdm(filename_cap_mappings):
    for image_idx in tqdm(range(len(df))):
        # image_idx = caption_pd.iloc[idx]['image_id']
        if image_idx in img_idx_ls:
            continue
        
        # full_img_file_name = os.path.join(img_folder, str(image_idx))
        full_img_file_name = os.path.join(data_path, "flickr30k-images/") + df.iloc[image_idx]['filename']
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

def process_input_and_bfs_search(img, input_str, input_boxes, depths=[1]):
    def calculate_center(xmin, ymin, xmax, ymax):
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        return x_center, y_center

    def calculate_distance(center1, center2):
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def find_existing_key(center, nodes_with_boxes, obj_name):
        for key, stored_center in nodes_with_boxes.items():
            if key.startswith(obj_name) and calculate_distance(center, stored_center) < 15:
                return key
        return None

    def convert_edge_list(input_str, input_boxes):
        node_patches = []
        node_bboxes = []
        # Split the input string into individual relationships
        relationships = input_str.replace('"','').split("|")

        # Convert the relationships into a list of tuples
        edge_list = []
        node_counter = {}  # To track occurrences of each node name
        nodes_with_boxes = {}  # To track bounding box centers for unique identifiers
        nodes_with_full_boxes = {}  # To track the full bounding box for each node

        for i in range(0, len(relationships)):
            parts = relationships[i].split()
            if len(parts) == 3:  # Format: object1 relationship object2
                obj1, rel, obj2 = parts[0], parts[1], parts[2]
            elif len(parts) == 4:  # Format: object1 relationship preposition object2
                obj1, rel, obj2 = parts[0], f"{parts[1]} {parts[2]}", parts[3]
            elif len(parts) == 5:  # Format: object1 relationship preposition1 preposition2 object2
                obj1, rel, obj2 = parts[0], f"{parts[1]} {parts[2]} {parts[3]}", parts[4]

            # Create unique identifiers for objects based on their bounding boxes
            sub_box = input_boxes[i][0]
            obj_box = input_boxes[i][1]
            sub_center = calculate_center(*sub_box)
            obj_center = calculate_center(*obj_box)

            if obj1 not in node_counter:
                node_counter[obj1] = 0
            if obj2 not in node_counter:
                node_counter[obj2] = 0

            sub_key = find_existing_key(sub_center, nodes_with_boxes, obj1)
            if sub_key is None:
                sub_key = f"{obj1}_{node_counter[obj1]}"
                node_counter[obj1] += 1
                nodes_with_boxes[sub_key] = sub_center
                nodes_with_full_boxes[sub_key] = sub_box

            obj_key = find_existing_key(obj_center, nodes_with_boxes, obj2)
            if obj_key is None:
                obj_key = f"{obj2}_{node_counter[obj2]}"
                node_counter[obj2] += 1
                nodes_with_boxes[obj_key] = obj_center
                nodes_with_full_boxes[obj_key] = obj_box

            edge_list.append((sub_key, rel, obj_key))
        for key, val in nodes_with_full_boxes.items():
            xmin, ymin, xmax, ymax = val
            patch = img.crop((xmin, ymin, xmax, ymax))
            node_patches.append(patch)
            node_bboxes.append(val)
        return edge_list, nodes_with_full_boxes, node_patches, node_bboxes

    edge_list, nodes_with_full_boxes, node_patches, node_bboxes = convert_edge_list(input_str, input_boxes)
    # Convert the edge list to a list of strings
    edge_list_strings = [f"{obj1} and {obj2}" for obj1, rel, obj2 in edge_list]
    # Build the directed graph
    G = nx.DiGraph()
    for obj1, rel, obj2 in edge_list:
        if (obj1 != obj2):
            G.add_edge(obj1, obj2, relationship=rel)
        #G.add_edge(obj1, obj2, relationship=rel)

    def bfs_collectors(start_node, depth):
        visited = set()
        queue = [(start_node, 0)]
        neighbors = [start_node]
        boxes = []
        depth_reached = False

        while queue:
            current_node, current_depth = queue.pop(0)
            if current_node not in visited:
                visited.add(current_node)
                if current_depth == depth:
                  depth_reached = True
                if current_depth > 0:
                    neighbors.append(current_node)
                    boxes.append(nodes_with_full_boxes[current_node])
                if current_depth < depth:
                    for neighbor in G.successors(current_node):
                        queue.append((neighbor, current_depth + 1))
                    for neighbor in G.predecessors(current_node):
                        queue.append((neighbor, current_depth + 1))

        return neighbors, boxes, depth_reached

    results = [[] for _ in range(len(depths))]
    overall_boxes = [[] for _ in range(len(depths))]
    for i in range(0,len(depths)):
        #print("Depth: " + str(depths[i]))
        for node in G.nodes:
            neighbors, boxes, depth_reached = bfs_collectors(node, depths[i])
            if neighbors and depth_reached:
                #print(neighbors)
                intermed_str = f"{' and '.join(neighbors)}"
                #print(intermed_str)
                if intermed_str not in edge_list_strings:
                    min_x = min(box[0] for box in boxes)
                    min_y = min(box[1] for box in boxes)
                    max_x = max(box[2] for box in boxes)
                    max_y = max(box[3] for box in boxes)
                    overall_box = [min_x, min_y, max_x, max_y]
                    results[i].append(intermed_str)
                    overall_boxes[i].append(overall_box)

    if not results:
        return "", [], node_patches, node_bboxes

    final_str_list = [[] for _ in range(len(depths))]
    for i in range(len(depths)):
      final_str = '"' + '"|"'.join(results[i]) + '"'
      final_str_list[i] = final_str

    return final_str_list, overall_boxes, node_patches, node_bboxes


def crop_image(img, img_fgb, img_bfs):
  fine_grained_patches = []
  # Iterate through the bounding boxes and extract patches
  for box in img_fgb:
    xmin, ymin, xmax, ymax = box
    # Crop the patch from the image
    patch = img.crop((xmin, ymin, xmax, ymax))
    fine_grained_patches.append(patch)
  bfs_patches = [[] for _ in range(len(img_bfs))]
  for i in range(len(img_bfs)):
    for box in img_bfs[i]:
      xmin, ymin, xmax, ymax = box
      patch = img.crop((xmin, ymin, xmax, ymax))
      bfs_patches[i].append(patch)
  return fine_grained_patches, bfs_patches

def combine_boxes(relationships_list):
  final_list = []
  for boxes_list in relationships_list:
    #print(boxes_list)
    sxmin, symin, sxmax, symax = boxes_list[0]
    oxmin, oymin, oxmax, oymax = boxes_list[1]
    # Calculate the overall bounding box
    overall_xmin = round(min(sxmin, oxmin),2)
    overall_ymin = round(min(symin, oymin),2)
    overall_xmax = round(max(sxmax, oxmax),2)
    overall_ymax = round(max(symax, oymax),2)
    final_list.append([overall_xmin, overall_ymin, overall_xmax, overall_ymax])
  return final_list

def get_patches_from_bboxes_scenegraph(patch_emb_ls, img_per_batch_ls, masks_ls, bboxes_ls,patch_count_for_compute_ls, patch_count_ls, forward_func,model, image_file_name_ls, input_processor, image_size=(224, 224), processor=None, resize=None, device="cpu",save_mask_bbox=False):
    # scene_graph_model = models.reltr_model(model_checkpoint='/content/drive/MyDrive/concept_based_retrieval-main-sg-official/beir/retrieval/models/RelTR/checkpoint0149.pth')
    scene_graph_model = models.reltr_model(model_checkpoint='sgg_model/checkpoint0149.pth')
    for i, image_file_name in tqdm(enumerate(image_file_name_ls)):
        
        image = Image.open(image_file_name).convert('RGB')
        depths = [1,2,3]
        #generate basic scene graph for image
        result = generate_caption_and_bboxes(image, scene_graph_model)
        relationship_triples = result[0]
        obj_bboxes = result[1]
        if (relationship_triples == ""):
            node_patches = [image]
            node_bboxes = [[0, 0, image.size[0]-1, image.size[1]-1]]
            relationship_triples_patches = [image]
            relationship_triples_bboxes = [[0, 0, image.size[0]-1, image.size[1]-1]]
            bfs_patches = [[] for _ in range(len(depths))]
            bfs_bboxes = [[] for _ in range(len(depths))]
            for depth in range(len(depths)):
                bfs_patches[depth] = [image]
                bfs_bboxes[depth] = [[0, 0, image.size[0]-1, image.size[1]-1]]
        else:
            relationship_triples_bboxes = combine_boxes(obj_bboxes)
            bfs_str, bfs_bboxes, node_patches, node_bboxes = process_input_and_bfs_search(image, relationship_triples, obj_bboxes, depths)
            #print(bfs_str)
            for depth in range(len(depths)):
                if (bfs_str[depth] == '""'):
                    if i == 0:
                        bfs_str[depth] = relationship_triples
                        bfs_bboxes[depth] = relationship_triples_bboxes
                    else:
                        #print("Inside")
                        bfs_str[depth] = bfs_str[depth-1]
                        bfs_bboxes[depth] = bfs_bboxes[depth-1]
            relationship_triples_patches, bfs_patches = crop_image(image, relationship_triples_bboxes, bfs_bboxes)
        #end, key outputs being (node_patches, node_bboxes), (relationship_triples_patches, relationship_triples_bboxes), (bfs_patches, bfs_bboxes)
        
        for patch_count_idx in range(len(patch_count_ls)):
            if patch_count_for_compute_ls[patch_count_idx] == False:
                continue
            patches = []
            if i == 0:
                #print("ENTERED")
                patch_emb_ls[patch_count_idx] = []
                img_per_batch_ls[patch_count_idx] = []
                masks_ls[patch_count_idx] = []
                bboxes_ls[patch_count_idx] = []
            
            n_patches = patch_count_ls[patch_count_idx]
            if (n_patches == 1):
                patches=node_patches
                bboxes_ls[patch_count_idx].append(node_bboxes)
                #print(node_bboxes)
                for x in range(0, len(node_bboxes)):
                    img_per_batch_ls[patch_count_idx].append(i)
            elif (n_patches == 4):
                patches=relationship_triples_patches
                bboxes_ls[patch_count_idx].append(relationship_triples_bboxes)
                #print(relationship_triples_bboxes)
                for x in range(0, len(relationship_triples_bboxes)):
                    img_per_batch_ls[patch_count_idx].append(i)
            elif (n_patches == 8):
                patches=bfs_patches[0]
                bboxes_ls[patch_count_idx].append(bfs_bboxes[0])
                #print(bfs_bboxes)
                for x in range(0, len(bfs_bboxes[0])):
                    img_per_batch_ls[patch_count_idx].append(i)
            elif (n_patches == 16):
                patches=bfs_patches[1]
                bboxes_ls[patch_count_idx].append(bfs_bboxes[1])
                #print(bfs_bboxes)
                for x in range(0, len(bfs_bboxes[1])):
                    img_per_batch_ls[patch_count_idx].append(i)
            elif (n_patches == 32):
                patches=bfs_patches[2]
                bboxes_ls[patch_count_idx].append(bfs_bboxes[2])
                #print(bfs_bboxes)
                for x in range(0, len(bfs_bboxes[2])):
                    img_per_batch_ls[patch_count_idx].append(i)
            #print(patches)    
            patch_embs = embed_patches(forward_func, patches, model, input_processor, processor, device=device, resize=resize)
            patch_emb_ls[patch_count_idx].append(patch_embs)
    return patch_emb_ls, img_per_batch_ls


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
        

    def get_patches(self, segmentation_method, model_name, patch_count_ls, samples_hash, img_idx_ls=None, img_file_name_ls=None, method="slic", not_normalize=False, use_mask=False, compute_img_emb=True, save_mask_bbox=False):
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
        
        if self.img_concept:
            for idx in range(len(patch_count_ls)):
                n_patches = patch_count_ls[idx]
                cached_file_name = utils.obtain_cached_file_name(segmentation_method, model_name, method, n_patches, samples_hash, not_normalize=not_normalize, use_mask=use_mask)
                # cached_file_name = f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}{'_use_mask' if use_mask else ''}.pkl"
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
        if model_name == "default":
            cached_img_file_name = f"output/saved_img_embs_{method}_{samples_hash}.pkl"
        else:
            cached_img_file_name = f"output/saved_img_embs_{method}_{model_name}_{samples_hash}.pkl"
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
        if self.img_concept:
            if True in patch_count_for_compute_ls:
                if segmentation_method == "scene_graph":
                    get_patches_from_bboxes_scenegraph(patch_emb_ls, img_per_batch_ls, masks_ls, bboxes_ls, patch_count_for_compute_ls, patch_count_ls, self.input_to_latent, self.model, img_file_name_ls, self.input_processor, device=self.device, resize=None, save_mask_bbox=False)
                else:
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
                # cached_file_name = f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}{'_use_mask' if use_mask else ''}.pkl"
                cached_file_name = utils.obtain_cached_file_name(segmentation_method, model_name, method, n_patches, samples_hash, not_normalize=not_normalize, use_mask=use_mask)
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
    if args.model_name == "default":
        cl = ConceptLearner(img_file_name_ls, model, vit_forward, processor, args.dataset_name, device)
    elif args.model_name == "blip":
        cl = ConceptLearner(img_file_name_ls, model, blip_vit_forward, processor, args.dataset_name, device)
    else:
        raise ValueError("Invalid model name")
    # bbox x1,y1,x2,y2
    # image_embs, patch_activations, masks, bboxes, img_for_patch
    # n_patches, images=None, method="slic", not_normalize=False

    cl.img_concept = args.img_concept
    res = cl.get_patches(args.segmentation_method, args.model_name, patch_count_ls, samples_hash, img_idx_ls=img_idx_ls, img_file_name_ls=img_file_name_ls, method="slic", compute_img_emb=True, save_mask_bbox=save_mask_bbox)
    
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
    # img_per_patch_tensor = torch.tensor(img_per_patch_ls[0])
    # max_img_id = int(torch.max(img_per_patch_tensor).item())
    max_img_id = len(img_emb)
    patch_emb_curr_img_ls = []
    transformed_bbox_ls = []
    for idx in tqdm(range(max_img_id)):
        sub_patch_emb_curr_img_ls = []
        sub_transformed_bbox_ls = []
        for sub_idx in range(len(patch_emb_ls)):
            # if idx == 1 and sub_idx > 0:
            #     continue
            patch_emb = patch_emb_ls[sub_idx]
            if img_per_patch_ls is not None:
                img_per_batch = img_per_patch_ls[sub_idx]
                img_per_patch_tensor = torch.tensor(img_per_batch)
                curr_selected_ids = torch.nonzero(img_per_patch_tensor == idx).view(-1)
                patch_emb_curr_img = patch_emb[curr_selected_ids]
            else:
                patch_emb_curr_img = patch_emb[idx] #[curr_selected_ids]
            
            # curr_selected_ids = torch.nonzero(img_per_patch_tensor == idx).view(-1)
            # patch_emb_curr_img = patch_emb[curr_selected_ids]
            sub_patch_emb_curr_img_ls.append(patch_emb_curr_img)
            if bbox_ls is not None and bbox_ls[sub_idx] is not None:
                sub_transformed_bbox_ls.extend(bbox_ls[sub_idx][idx])
        sub_patch_emb_curr_img_ls.append(img_emb[idx].unsqueeze(0))
        patch_emb_curr_img = torch.cat(sub_patch_emb_curr_img_ls, dim=0)
        # patch_emb_curr_img = torch.cat([img_emb[idx].unsqueeze(0), sub_patch_emb_curr_img], dim=0)
        patch_emb_curr_img_ls.append(patch_emb_curr_img)
        if bbox_ls is not None:
            transformed_bbox_ls.append(sub_transformed_bbox_ls)
    
    return patch_emb_curr_img_ls, transformed_bbox_ls


       

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