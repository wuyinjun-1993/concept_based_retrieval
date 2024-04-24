from tqdm import tqdm

import torch.nn.functional as F
import torch

class image_storage_node:
    def __init__(self, image, bbox, embedding, parent=None):
        self.image = image
        self.bbox = bbox
        self.parent = parent
        self.embedding = embedding
        self.children = []
        self.children_embeddings = None

    # def set_embedding(self, embedding):
    #     self.embedding = embedding
    def add_child(self, child):
        self.children.append(child)
    
    def add_children_embeddings(self, children_embeddings):
        self.children_embeddings = children_embeddings
    
    def compare_query_embedding_and_children_embeddings(self, query_embedding, device):
        if self.children_embeddings is None:
            return None, None
        self.children_embeddings = self.children_embeddings.to(device)
        q_child_sims = F.cosine_similarity(query_embedding, self.children_embeddings, dim=-1)
        max_q_child_sim = torch.max(q_child_sims)
        max_q_child_sim_idx = torch.argmax(q_child_sims)
        next_child = self.children[max_q_child_sim_idx]
        return max_q_child_sim.item(), next_child
        

    def get(self):
        # get image from disk
        pass
    
    
class image_storage_tree:
    def __init__(self, root):
        self.root = root
        self.current_node = root
    
    
        
def create_single_node(image, bbox, embedding, parent=None):
    return image_storage_node(image, bbox, embedding, parent)

def create_non_root_nodes_same_layer(images_ls, bboxes_ls, children_embs_ls, parent_nodes_ls=None):
    
    all_image_node_ls = []
    for idx in tqdm(range(len(images_ls))):
        if parent_nodes_ls is None:
            parent = None
        else:
            parent = parent_nodes_ls[idx]

        image_node_ls = []
        for sub_idx in range(len(images_ls[idx])):
            for sub_sub_idx in range(len(images_ls[idx][sub_idx])):
                if bboxes_ls is not None:
                    image_node = create_single_node(images_ls[idx][sub_idx][sub_sub_idx], bboxes_ls[idx][sub_idx][sub_sub_idx], children_embs_ls[idx][sub_idx], parent)
                else:
                    image_node = create_single_node(images_ls[idx][sub_idx][sub_sub_idx], None, children_embs_ls[idx][sub_idx][sub_sub_idx], parent)
                image_node_ls.append(image_node)
                if parent is not None:
                    parent[sub_idx].add_child(image_node)
                    parent[sub_idx].add_children_embeddings(children_embs_ls[idx][sub_idx])
                # parent.children.append(image_node)
        
        all_image_node_ls.append(image_node_ls)
    
    return all_image_node_ls



def init_root_nodes(images_ls, bboxes_ls, children_embs_ls):
    
    all_image_node_ls = []
    for idx in tqdm(range(len(images_ls))):
        image_node_ls = []
        for sub_idx in range(len(images_ls[idx])):
            # for sub_sub_idx in range(len(images_ls[idx][sub_idx])):
            if bboxes_ls is not None:
                image_node = create_single_node(images_ls[idx][sub_idx], bboxes_ls[idx][sub_idx], children_embs_ls[idx][sub_idx], None)
            else:
                image_node = create_single_node(images_ls[idx][sub_idx], None, children_embs_ls[idx][sub_idx], None)
            image_node_ls.append(image_node)        
        all_image_node_ls.append(image_node_ls)
    
    return all_image_node_ls


# compute the lower bound and upper bound of a1*a2 given a1*b and a2*b
def compute_upper_lower_bound(a1_times_b, a2_times_b, a1_norm, a2_norm, b_norm):
    lb = (a1_times_b*a2_times_b/(b_norm**2))*2 - a1_norm*a2_norm
    ub = (a1_times_b*a2_times_b/(b_norm**2))*2 + a1_norm*a2_norm
    return lb, ub
    

