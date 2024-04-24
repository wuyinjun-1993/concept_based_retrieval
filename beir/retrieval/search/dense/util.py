import torch
import numpy as np
import csv
import os
from collections import deque
from tqdm import tqdm

def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1)) #TODO: this keeps allocating GPU memory

def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

def normalize(a: np.ndarray) -> np.ndarray:
    return a/np.linalg.norm(a, ord=2, axis=1, keepdims=True)

def save_dict_to_tsv(_dict, output_path, keys=[]):
    
    with open(output_path, 'w') as fIn:
        writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        if keys: writer.writerow(keys)
        for key, value in _dict.items():
            writer.writerow([key, value])

def load_tsv_to_dict(input_path, header=True):
    
    mappings = {}
    reader = csv.reader(open(input_path, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    if header: next(reader)
    for row in reader: 
        mappings[row[0]] = int(row[1])
    
    return mappings


def store_single_node_tree(root_node, path, root_idx):
    result = []
    queue = deque()
    queue.append(root_node)

    while queue:
        node = queue.popleft()
        # result.append(node.value)
        
        node.store_node(path, root_idx)
        for child in node.children:
            queue.append(child)

    return result
    

def obtain_patch_embedding_file_name(path, root_idx):
    root_node_folder = os.path.join(path, str(root_idx))
    if os.path.exists(root_node_folder) is False:
        os.makedirs(root_node_folder)    
    filename = os.path.join(root_node_folder, "full_data.pt")
    return filename

def store_all_node_trees(path, root_nodes_ls):
    for root_idx in tqdm(range(len(root_nodes_ls))):
        root_node = root_nodes_ls[root_idx][0]
        # store_single_node_tree(root_nodes_ls[root_idx][0], path, root_idx)
        subtree_filename_ls = []
        for child_idx in range(len(root_node.children)):
            subtree_filename = root_node.children[child_idx].store_subtree(path, root_idx)
            subtree_filename_ls.append(subtree_filename)
        del root_node.children
        root_node.children = subtree_filename_ls

def store_single_embeddings(path, patch_embs, root_idx):
    filename = obtain_patch_embedding_file_name(path, root_idx)
    torch.save(patch_embs, filename)
    
def load_single_embeddings(path, root_idx):
    filename = obtain_patch_embedding_file_name(path, root_idx)
    patch_embs = torch.load(filename)
    return patch_embs

def store_all_embeddings(path, patch_emb_ls):
    for root_idx in range(len(patch_emb_ls)):
        store_single_embeddings(path, patch_emb_ls[root_idx], root_idx)