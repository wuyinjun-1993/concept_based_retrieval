import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm
import hashlib
import PIL
import json
import os

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
        
def filter_queries_with_gt(full_data_path, args, queries):
    output_file_path = os.path.join(full_data_path, args.dataset_name, "queries_with_gt.jsonl")
    if os.path.exists(output_file_path):
        return 


    full_query_set_file = os.path.join(full_data_path, args.dataset_name, "queries.jsonl")
        
    with open(full_query_set_file, 'r') as json_file:
        json_list = list(json_file)
    
    result_ls = []
    for json_str in json_list:
        result = json.loads(json_str)
        if result["_id"] in queries:
            result_ls.append(result)
            
    
    # Open the file in write mode
    with open(output_file_path, 'w') as jsonl_file:
        # Iterate over each JSON object in the list
        for json_obj in result_ls:
            # Convert the JSON object to a JSON string
            json_str = json.dumps(json_obj)
            # Write the JSON string to the file with a newline character
            jsonl_file.write(json_str + '\n')