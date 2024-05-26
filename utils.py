import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm
import hashlib
import PIL

def get_final_res_file_name(args, patch_count_ls):
    patch_count_ls = sorted(patch_count_ls)
    patch_count_str = "_".join([str(patch_count) for patch_count in patch_count_ls])
    file_prefix=f"output/saved_patches_{args.dataset_name}_{patch_count_str}"
    if args.total_count > 0:
        file_prefix=file_prefix + "_subset_" + str(args.total_count)
    #--query_concept","--img_concept"],// "--search_by_cluster
    if args.query_concept: 
        file_prefix =  file_prefix + "_query"
    if args.img_concept:
        file_prefix =  file_prefix + "_img"
        
    if args.search_by_cluster:
        file_prefix =  file_prefix + "_cluster"
        
    patch_clustering_info_cached_file = f"{file_prefix}.pkl"
        
    return patch_clustering_info_cached_file

def hashfn(x: list):
    if type(x[0]) == PIL.Image.Image:
        samples_hash = hashlib.sha1(np.stack([img.resize((32, 32)) for img in tqdm(x)])).hexdigest()
    else:
        if type(x[0]) is not str:
            samples_hash = hashlib.sha1(np.array(x)).hexdigest()
        else:
            samples_hash = hashlib.sha256(str(x).encode()).hexdigest()
    return samples_hash

def load(filename):
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
    except:
        raise Exception('File not found')
    return obj

def save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)