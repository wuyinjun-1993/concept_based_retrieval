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
from retrieval_utils import decompose_single_query

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


def load_atom_datasets(data_path):
    image_ls = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train')
    
    text_ls = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2", split='train')
    
    # selected_dataset = filter_atom_images_by_langs(dataset)
    
def load_crepe_datasets(data_path, query_path):
    img_caption_file_name= os.path.join(query_path, "prod_hard_negatives/prod_vg_hard_negs_swap_all4.csv")

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
        sub_caption_str = caption_pd.iloc[idx]['caption_triples']
        
        sub_captions = decompose_single_query(sub_caption_str)
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
        if len(img_ls) >= total_count:
            break
    
    return img_idx_ls, img_ls    
        
    

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

def read_image_captions(caption_file):
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


def get_patches_from_bboxes2(bboxes_for_img, images):
    patches_for_imgs = []
    for i, bboxes in enumerate(bboxes_for_img):
        patches = []
        for bbox in bboxes:
            patches.append(Patch(images[i], bbox, images[i].crop(bbox)))
        patches_for_imgs.append(patches)
    return patches_for_imgs


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

def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [_to_device(xi, device) for xi in x]

def get_patches_from_bboxes(forward_func,model, images, all_bboxes, input_processor, sub_bboxes=None, image_size=(224, 224), processor=None, resize=None, device="cpu"):
    if sub_bboxes == None:
        sub_bboxes = [[[bbox] for bbox in img_bboxes] for img_bboxes in all_bboxes]

    all_patches = []
    img_labels = []
    for i, (bboxes, visible, image) in tqdm(enumerate(zip(all_bboxes, sub_bboxes, images))):
        patches = []
        for bbox, viz in zip(bboxes, visible):
            curr_patch = PIL.Image.new('RGB', image.size) #image.copy().filter(ImageFilter.GaussianBlur(radius=10))
            for viz_box in viz:
                # add the visible patches from the original image to curr_patch
                curr_patch.paste(image.copy().crop(viz_box), box=viz_box)
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

    def get_patches(self, n_patches, images=None, method="slic", not_normalize=False, compute_img_emb=True):
        """Get patches from images using different segmentation methods."""
        if images is None:
            images = self.samples

        samples_hash = utils.hashfn(images)
        if os.path.exists(f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}.pkl"):
            print("Loading cached patches")
            print(samples_hash)
            image_embs, patch_activations, masks, bboxes, img_for_patch = utils.load(f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}.pkl")
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
        masks = get_slic_segments(images, n_segments=n_patches)
        bboxes = masks_to_bboxes(masks)
        # get_patches_from_bboxes(model, images, all_bboxes, input_processor, sub_bboxes=None, image_size=(224, 224), processor=None, resize=None, device="cpu"):
        patch_activations, img_for_patch = get_patches_from_bboxes(self.input_to_latent, self.model, images, bboxes, self.input_processor, device=self.device, resize=self.image_size)
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
        utils.save((image_embs, patch_activations, masks, bboxes, img_for_patch), f"output/saved_patches_{method}_{n_patches}_{samples_hash}{'_not_normalize' if not_normalize else ''}.pkl")
        return image_embs, patch_activations, masks, bboxes, img_for_patch


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