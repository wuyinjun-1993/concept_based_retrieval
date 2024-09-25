
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import os
import random
import faiss

def sampling_sample_ids(X_ls, typical_doclen):
    num_passages = len(X_ls)

    # typical_doclen = 1  # let's keep sampling independent of the actual doc_maxlen
    sampled_pids = 16 * np.sqrt(typical_doclen * num_passages)
    # sampled_pids = int(2 ** np.floor(np.log2(1 + sampled_pids)))
    sampled_pids = min(1 + int(sampled_pids), num_passages)

    sampled_pids = random.sample(range(num_passages), sampled_pids)
    
    return sampled_pids


def sampling_patch_ids0(X_ls, patch_ids):
    selected_patch_X = []
    for pid in tqdm(patch_ids, desc="Sampling patch ids"):
        curr_X = X_ls[pid]
        # curr_X = curr_X/ torch.norm(curr_X, dim=1, keepdim=True)
        # kmeans = KMeans(n_clusters=int(len(curr_X)*0.2), random_state=0).fit(curr_X)
        # # kmeans_labels = kmeans.labels_
        # centroids = torch.from_numpy(kmeans.cluster_centers_).float()
        # centroids = centroids/ torch.norm(centroids, dim=1, keepdim=True)
        # closest_sample_ids = torch.matmul(curr_X, centroids.T).argmax(dim=0)
        selected_patch_X.append(curr_X)
    return torch.cat(selected_patch_X)

def sampling_patch_ids(X_ls, patch_ids, clustering_number=0.1):
    selected_patch_X = []
    for pid in tqdm(patch_ids, desc="Sampling patch ids"):
        curr_X = X_ls[pid]
        curr_X = curr_X/ torch.norm(curr_X, dim=1, keepdim=True)
        # kmeans = KMeans(n_clusters=int(len(curr_X)*0.2), random_state=0).fit(curr_X)
        # kmeans_labels = kmeans.labels_
        # centroids = torch.from_numpy(kmeans.cluster_centers_).float()
        
        clustering = DBSCAN(eps=0.01, min_samples=1, metric="cosine").fit(curr_X.cpu().numpy())
        centroid_ls = [torch.mean(curr_X[clustering.labels_ == label], dim=0) for label in range(len(np.unique(clustering.labels_)))]
        centroids = torch.stack(centroid_ls)
        
        
        centroids = centroids/ torch.norm(centroids, dim=1, keepdim=True)
        closest_sample_ids = torch.matmul(curr_X, centroids.T).argmax(dim=0)
        selected_patch_X.append(curr_X[closest_sample_ids])
    return torch.cat(selected_patch_X)

def compute_faiss_kmeans(dim, num_partitions, kmeans_niters, sample_embeddings):
    sample_embeddings = sample_embeddings/ torch.norm(sample_embeddings, dim=-1, keepdim=True)
    use_gpu = torch.cuda.is_available()
    print("is gpu available::", use_gpu)
    sample = sample_embeddings.float().numpy()
    try:
        kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=use_gpu, verbose=True, seed=123)
        # kmeans = faiss.Kmeans(dim, num_partitions, gpu=use_gpu, verbose=True, seed=123)
    
        kmeans.train(sample)
    except:
        print("start using cpus for faiss")
        kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=False, verbose=True, seed=123)
        kmeans.train(sample)
    centroids = torch.from_numpy(kmeans.centroids)
    centroids = centroids/torch.norm(centroids, dim=-1, keepdim=True)
    # if use_gpu:
    #   centroids = centroids.half()    
    # else:
    #   centroids = centroids.float()
    return centroids

def sampling_and_clustering(X_ls, dataset_name, clustering_number=0.1, typical_doclen=1):
    if dataset_name == "crepe":
        sampled_patch_X = torch.cat(X_ls, dim=0)
    else:
        sampled_pids = sampling_sample_ids(X_ls, typical_doclen=typical_doclen)
        # sampled_patch_X = sampling_patch_ids0(X_ls, sampled_pids)
        sampled_patch_X = sampling_patch_ids(X_ls, sampled_pids, clustering_number=clustering_number)
    cluster_count = int(clustering_number*len(sampled_patch_X)) # max(int(len(sampled_patch_X)*clustering_count_ratio), 10)
    print("sampled data count::", len(sampled_patch_X))
    print("cluster count::", cluster_count)
    num_partitions = cluster_count
    centroids = compute_faiss_kmeans(sampled_patch_X.shape[-1], num_partitions, 100, sampled_patch_X)
    # centroid_ls, _ = online_clustering(sampled_patch_X, closeness_threshold=clustering_number)
    # # # # kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(sampled_patch_X)
    # centroids = torch.stack(centroid_ls) #torch.from_numpy(kmeans.cluster_centers_).float()
    
    # clustering = DBSCAN(eps=clustering_number, min_samples=1, metric="cosine").fit(sampled_patch_X.cpu().numpy())
    # centroid_ls = [torch.mean(sampled_patch_X[clustering.labels_ == label], dim=0) for label in range(len(np.unique(clustering.labels_)))]
    # centroids = torch.stack(centroid_ls)
    print("cluster count::", len(centroids))
    
    # clustering_labels = torch.nn.functional.cosine_similarity(X.unsqueeze(1), centroids.unsqueeze(0)).argmax(dim=1)
    
    
    return centroids


def online_clustering(X, closeness_threshold=0.1):
    print("closeness threshold::", closeness_threshold)
    centroid_ls = []
    labels = torch.ones(X.shape[0])*(-1)
    for idx in tqdm(range(X.shape[0]), desc="Online clustering"):
        if len(centroid_ls) == 0:
            centroid_ls.append(X[idx])
            labels[idx] = 0
            all_centroids = torch.stack(centroid_ls)
        else:
            
            similarities = F.cosine_similarity(all_centroids, X[idx].view(1,-1))
            max_sim_idx = torch.argmax(similarities).item()
            if similarities[max_sim_idx] > 1 - closeness_threshold:
                
                centroid_sample_count = torch.sum(labels == max_sim_idx).item()
                centroid_ls[max_sim_idx] = (centroid_ls[max_sim_idx]*centroid_sample_count +  X[idx])/(centroid_sample_count+1)
                all_centroids[max_sim_idx] = centroid_ls[max_sim_idx]
                labels[idx] = max_sim_idx
            else:
                centroid_ls.append(X[idx])
                labels[idx] = len(centroid_ls)-1
                all_centroids = torch.cat([all_centroids, X[idx].view(1,-1)])
    centroid_ls = [centroid.cpu() for centroid in centroid_ls]
    return centroid_ls, labels

def verify_clustering(X, labels):
    for label in tqdm(np.unique(labels), desc="Verifying clusters"):
        sub_X = X[labels == label]
        
        sub_X_norm = torch.norm(sub_X, dim=-1)
        
        cos_sim = torch.mm(sub_X, sub_X.t())/torch.mm(sub_X_norm.unsqueeze(1), sub_X_norm.unsqueeze(0))
                
        max_cos_sim = torch.min(cos_sim)
        
        print(f"Max cosine similarity for cluster {label}: {max_cos_sim.item()}")


def kmeans_cosine(X, k, max_iters=1000):
    """
    Perform k-means clustering on the given data using cosine similarity as the distance metric.

    Parameters:
    - X: torch.Tensor, shape (N, D), input data points
    - k: int, number of clusters
    - max_iters: int, maximum number of iterations

    Returns:
    - centroids: torch.Tensor, shape (k, D), final centroids of clusters
    - cluster_assignments: torch.Tensor, shape (N,), cluster assignments for each data point
    """
    # Initialize centroids randomly
    
    centroids = X[torch.randperm(X.size(0))[:k]]
    
    X_unsqueezed = X.unsqueeze(1)
    
    with torch.no_grad():
        for _ in tqdm(range(max_iters)):
            # Compute cosine similarity between each data point and centroids
            # similarities = F.cosine_similarity(X_unsqueezed, centroids.unsqueeze(0), dim=-1)
            similarities = compute_similarities_by_batches(X, k, centroids, device = 'cuda', batch_size = 100)
            
            # Assign each data point to the closest centroid
            cluster_assignments = torch.argmax(similarities, dim=1)
            
            max_similarity = torch.max(similarities, dim=1)[0].cpu()
            
            # Update centroids
            new_centroids = torch.stack([X[cluster_assignments == i].mean(0) for i in range(k)])
            
            # Check for convergence
            # if torch.all(torch.eq(new_centroids, centroids)):
            if torch.abs(new_centroids - centroids).max() < 0.005:
                break
            
            del centroids
            
            centroids = new_centroids
            
            del new_centroids
    
    del X_unsqueezed
    
    return centroids.cpu(), cluster_assignments.cpu(), max_similarity


def compute_similarities_by_batches(X, k, centroids, device = 'cuda', batch_size = 100):
    similarities = torch.zeros(X.size(0), k, device=device)
    for i in range(0, X.size(0), batch_size):
        batch_X = X[i:i+batch_size]
        batch_X = batch_X.cuda()
        batch_similarities = F.cosine_similarity(batch_X.unsqueeze(1), centroids.unsqueeze(0), dim=-1)
        similarities[i:i+batch_size] = batch_similarities
        del batch_X
    return similarities


def select_patch_embeddings_closest_to_centroids(mean_sub_X, unique_sub_sample_ids, sub_X, sub_sample_ids, sub_sample_patch_ids, sub_sample_granularity_ids, sub_cat_patch_ids):
    most_similar_sample_ls = []
    most_similar_sample_mappings = dict()
    most_similar_patch_ids_ls = []
    most_similar_granularity_ids_ls = []
    most_similar_sample_ids_ls = []
    most_similar_cat_patch_ids_ls = []
    most_similar_cat_patch_ids_mappings = dict()
    for unique_sub_sample_id in unique_sub_sample_ids:
        unique_sub_sample_id = int(unique_sub_sample_id.item())
        curr_sub_X = sub_X[sub_sample_ids == unique_sub_sample_id]
        if sub_sample_patch_ids is not None:
            curr_sub_sample_patch_ids = sub_sample_patch_ids[sub_sample_ids == unique_sub_sample_id]
            most_similar_patch_ids_ls.extend(curr_sub_sample_patch_ids.tolist())
        else:
            curr_sub_sample_patch_ids =  None
        if sub_sample_granularity_ids is not None:
            curr_sub_sample_granularity_ids = sub_sample_granularity_ids[sub_sample_ids == unique_sub_sample_id]
            most_similar_granularity_ids_ls.extend(curr_sub_sample_granularity_ids.tolist())
        else:
            curr_sub_sample_granularity_ids = None
        
        # curr_sub_X_centroid_sims = F.cosine_similarity(curr_sub_X, mean_sub_X.view(1,-1))
        # most_similar_patch_id_curr_sample = torch.argmax(curr_sub_X_centroid_sims, dim=0)
        # most_similar_sample_ls.append(curr_sub_X[most_similar_patch_id_curr_sample])
        
        # most_similar_sample_ls.append(torch.mean(curr_sub_X, dim=0))
        most_similar_sample_ls.append(curr_sub_X)
        most_similar_sample_mappings[unique_sub_sample_id] = curr_sub_X
        
        
        most_similar_sample_ids_ls.extend([unique_sub_sample_id]*len(curr_sub_X))
        most_similar_cat_patch_ids_ls.extend(sub_cat_patch_ids[sub_sample_ids == unique_sub_sample_id].tolist())
        most_similar_cat_patch_ids_mappings[unique_sub_sample_id] = sub_cat_patch_ids[sub_sample_ids == unique_sub_sample_id].tolist()
    # return torch.stack(most_similar_sample_ls), most_similar_patch_ids_ls, most_similar_granularity_ids_ls, most_similar_sample_ids_ls, most_similar_cat_patch_ids_ls
    return most_similar_sample_mappings, most_similar_patch_ids_ls, most_similar_granularity_ids_ls, most_similar_sample_ids_ls, most_similar_cat_patch_ids_mappings
        

def construct_sample_patch_ids_ls(all_bboxes_ls):
    sample_sub_ids_ls = []
    sample_granularity_ids_ls = []
    for idx in range(len(all_bboxes_ls)):
        bboxes_ls = all_bboxes_ls[idx]
        curr_sample_sub_ids_ls = []
        for sub_idx in range(len(bboxes_ls)):
            bboxes = bboxes_ls[sub_idx]
            curr_sample_sub_ids_ls.extend(list(range(len(bboxes))))
        
        
        sample_sub_ids_ls.append(torch.tensor(curr_sample_sub_ids_ls))
        sample_granularity_ids_ls.append(torch.ones(len(curr_sample_sub_ids_ls)).long()*idx)
        
    return sample_sub_ids_ls, sample_granularity_ids_ls

def get_patch_count_str(patch_count_ls):
    # patch_count_ls = sorted(patch_count_ls)
    patch_count_str = "_".join([str(patch_count) for patch_count in patch_count_ls])
    return patch_count_str

def get_clustering_res_file_name(args, hashes, patch_count_ls):
    patch_count_ls = sorted(patch_count_ls)
    patch_count_str = get_patch_count_str(patch_count_ls)
    extra_suffix=""
    if args.model_name == "llm":
        extra_suffix += "_llm"
        
    if args.use_raptor:
        extra_suffix += "_raptor"
    
    
    if args.clustering_doc_count_factor == 1:
        centroid_ls_file_name=f"output/centroid_ls_{hashes}_{patch_count_str}_{args.clustering_number}{extra_suffix}.pt"
    else:
        centroid_ls_file_name=f"output/centroid_ls_{hashes}_{patch_count_str}_{args.clustering_number}{extra_suffix}_doclen_{args.clustering_doc_count_factor}.pt"
    
    # patch_clustering_info_cached_file =  f"output/saved_patches_{args.dataset_name}_{patch_count_str}.pkl"
    return centroid_ls_file_name


def get_dessert_clustering_res_file_name(args, hashes, patch_count_ls,clustering_number=1000, index_method="default", typical_doclen=1,num_tables=100, hashes_per_table=5):
    patch_count_str = get_patch_count_str(patch_count_ls)
    
    extra_suffix=""
    if args.use_raptor:
        extra_suffix += "_raptor"
    if args.model_name=="llm":
        extra_suffix += "_llm"
    
    if index_method == "default":
        if typical_doclen == 1:
            patch_clustering_info_cached_file =  f"output/dessert_clustering_res_{hashes}_{patch_count_str}_{index_method}_{clustering_number}{extra_suffix}.pkl"
        else:
            patch_clustering_info_cached_file =  f"output/dessert_clustering_res_{hashes}_{patch_count_str}_{index_method}_{clustering_number}_doclen_{typical_doclen}{extra_suffix}.pkl"
    else:
        patch_clustering_info_cached_file =  f"output/dessert_clustering_res_{hashes}_{patch_count_str}_{index_method}_{clustering_number}_doclen_{typical_doclen}{extra_suffix}_num_table_{num_tables}_hashes_per_table_{hashes_per_table}.pkl"
    return patch_clustering_info_cached_file

# 0.12 for trec covid 10000
# 0.2
def clustering_img_patch_embeddings(X_by_img_ls, dataset_name, X_ls, closeness_threshold = 0.1):
    """
    Determine the optimal number of clusters using the elbow method.

    Parameters:
    - X: torch.Tensor, shape (N, D), input data points
    - max_k: int, maximum number of clusters to consider

    Returns:
    - optimal_k: int, optimal number of clusters
    """
    inertias = []
    X = torch.cat(X_by_img_ls, dim=0)
    # img_per_patch_tensor = torch.cat([torch.tensor(img_per_patch).view(-1) for img_per_patch in img_per_patch_ls])
    img_per_patch_tensor = torch.cat([torch.ones(len(X_by_img_ls[idx]))*idx for idx in range(len(X_by_img_ls))])
    # sample_cat_patch_id_ls = torch.cat([torch.arange(len(curr_img_per_patch)) for curr_img_per_patch in img_per_patch_ls])
    sample_cat_patch_id_ls = torch.cat([torch.arange(len(sub_X)) for sub_X in X_by_img_ls])
    # sample_patch_ids_ls, sample_granularity_ids_ls = construct_sample_patch_ids_ls(all_bboxes_ls)
    # sample_patch_ids_tensor = torch.cat(sample_patch_ids_ls)
    # sample_granularity_ids_tensor = torch.cat(sample_granularity_ids_ls)
    # X = X_ls[0]
    # X = X/ torch.norm(X, dim=1, keepdim=True)
    
    # clustering = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete", distance_threshold=0.1).fit(X.cpu().numpy())
    
    # clustering = Birch(threshold=0.3, n_clusters=None).fit(X.cpu().numpy())
    # clustering = DBSCAN(eps=0.1, min_samples=2, metric="cosine").fit(X.cpu().numpy())
    # clustering_labels = clustering.labels_
    # threshold
    
    centroid_ls_file_name=f"output/{dataset_name}_centroid_ls_{closeness_threshold}.pt"
    clustering_labels_file_name=f"output/{dataset_name}_clustering_labels_{closeness_threshold}.pt"
    if os.path.exists(centroid_ls_file_name) and os.path.exists(clustering_labels_file_name):
        centroid_ls = torch.load(centroid_ls_file_name)
        clustering_labels = torch.load(clustering_labels_file_name)
    else:
        centroid_ls, clustering_labels = online_clustering(X, closeness_threshold=closeness_threshold)
        torch.save(centroid_ls, f"output/{dataset_name}_centroid_ls_{closeness_threshold}.pt")
        torch.save(clustering_labels, f"output/{dataset_name}_clustering_labels_{closeness_threshold}.pt")
    print(f"Number of clusters: {len(centroid_ls)}")
    # verify_clustering(X, clustering_labels)

    cosine_sim_min_ls = []
    cosine_sim_mat_ls = []
    min_cosine_sim_min = 1
    min_cosine_sim_min_idx = 0
    
    min_cosine_sim_min2 = 1
    min_cosine_sim_min_idx2 = 0
    cluster_centroid_ls = []
    cluster_sample_count_ls = []
    cluster_sample_ids_ls = []
    cluster_unique_sample_ids_ls = []
    cluster_sub_X_tensor_ls = []
    cluster_sub_X_patch_ids_ls=[]
    cluster_sub_X_cat_patch_ids_ls = []
    cluster_sub_X_granularity_ids_ls = []
    sample_patch_ids_to_cluster_id_mappings = dict()
    for label in tqdm(np.unique(clustering_labels)):
        sub_X = X[clustering_labels == label]      
        sub_cat_patch_ids = sample_cat_patch_id_ls[clustering_labels == label]
        sub_sample_ids = img_per_patch_tensor[clustering_labels == label]
        # sub_sample_patch_ids = sample_patch_ids_tensor[clustering_labels == label]
        # sub_sample_granularity_ids = sample_granularity_ids_tensor[clustering_labels == label]
        sub_sample_patch_ids, sub_sample_granularity_ids = None, None
        
        mean_sub_X = sub_X.mean(0).unsqueeze(0)
        cosine_sim_mat = torch.mm(sub_X, sub_X.t())
        cosine_sim_mat2 = torch.mm(mean_sub_X, sub_X.t())/(torch.norm(mean_sub_X, dim=-1).unsqueeze(1)*torch.norm(sub_X, dim=-1).unsqueeze(0))

        
        cosine_sim_min = torch.min(cosine_sim_mat).item()
        cosine_sim_min2 = torch.min(cosine_sim_mat2).item()
        
        
        if cosine_sim_min < min_cosine_sim_min:
            min_cosine_sim_min = cosine_sim_min
            min_cosine_sim_min_idx = label
        
        if cosine_sim_min2 < min_cosine_sim_min2:
            min_cosine_sim_min2 = cosine_sim_min2
            min_cosine_sim_min_idx2 = label
        
        cosine_sim_min_ls.append(cosine_sim_min)
        cosine_sim_mat_ls.append(cosine_sim_mat)
        
        cluster_centroid_ls.append(mean_sub_X)
        
        unique_sub_sample_ids = sub_sample_ids.unique()
        
        most_similar_sub_X_tensor, most_similar_patch_ids_ls, most_similar_granularity_ids_ls, most_similar_sample_ids_ls, most_similar_cat_patch_ids_ls = select_patch_embeddings_closest_to_centroids(mean_sub_X, unique_sub_sample_ids, sub_X, sub_sample_ids, sub_sample_patch_ids, sub_sample_granularity_ids, sub_cat_patch_ids)
        
        
        cluster_sample_count_ls.append(len(sub_sample_ids.unique()))
        cluster_unique_sample_ids_ls.append(unique_sub_sample_ids)
        cluster_sample_ids_ls.append(most_similar_sample_ids_ls)
        cluster_sub_X_tensor_ls.append(most_similar_sub_X_tensor)
        cluster_sub_X_patch_ids_ls.append(most_similar_patch_ids_ls)
        cluster_sub_X_granularity_ids_ls.append(most_similar_granularity_ids_ls)
        cluster_sub_X_cat_patch_ids_ls.append(most_similar_cat_patch_ids_ls)
        
        # construct sample patch idx to cluster label mappings
        for sample_idx in most_similar_cat_patch_ids_ls:
            if not sample_idx in sample_patch_ids_to_cluster_id_mappings:
                sample_patch_ids_to_cluster_id_mappings[sample_idx] = dict()
            for patch_idx in most_similar_cat_patch_ids_ls[sample_idx]:
                sample_patch_ids_to_cluster_id_mappings[sample_idx][patch_idx] = int(label)
        
    cluster_centroid_tensor = torch.cat(cluster_centroid_ls, dim=0)
    
    return cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, cluster_sub_X_cat_patch_ids_ls, sample_patch_ids_to_cluster_id_mappings


def clustering_img_embeddings(X_embeddings):
    """
    Determine the optimal number of clusters using the elbow method.

    Parameters:
    - X: torch.Tensor, shape (N, D), input data points
    - max_k: int, maximum number of clusters to consider

    Returns:
    - optimal_k: int, optimal number of clusters
    """
    # inertias = []
    # X = torch.cat(X_ls, dim=0)
    # img_per_patch_tensor = torch.cat([torch.tensor(img_per_patch).view(-1) for img_per_patch in img_per_patch_ls])
    # sample_patch_ids_ls, sample_granularity_ids_ls = construct_sample_patch_ids_ls(all_bboxes_ls)
    # sample_patch_ids_tensor = torch.cat(sample_patch_ids_ls)
    # sample_granularity_ids_tensor = torch.cat(sample_granularity_ids_ls)
    # # X = X_ls[0]
    # X = X/ torch.norm(X, dim=1, keepdim=True)
    
    # clustering = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete", distance_threshold=0.1).fit(X.cpu().numpy())
    
    clustering = Birch(threshold=0.5, n_clusters=None).fit(X_embeddings.cpu().numpy())

    cosine_sim_min_ls = []
    cosine_sim_mat_ls = []
    min_cosine_sim_min = 1
    min_cosine_sim_min_idx = 0
    
    min_cosine_sim_min2 = 1
    min_cosine_sim_min_idx2 = 0
    cluster_centroid_ls = []
    cluster_sample_count_ls = []
    cluster_sample_ids_ls = []
    cluster_unique_sample_ids_ls = []
    cluster_sub_X_tensor_ls = []
    cluster_sub_X_patch_ids_ls=[]
    cluster_sub_X_granularity_ids_ls = []
    for label in tqdm(np.unique(clustering.labels_)):
        sub_X = X_embeddings[clustering.labels_ == label]      
        sub_sample_ids = np.nonzero(clustering.labels_ == label)[0]
        # sub_sample_ids = img_per_patch_tensor[clustering.labels_ == label]
        # sub_sample_patch_ids = sample_patch_ids_tensor[clustering.labels_ == label]
        # sub_sample_granularity_ids = sample_granularity_ids_tensor[clustering.labels_ == label]
        
        mean_sub_X = sub_X.mean(0).unsqueeze(0)
        cosine_sim_mat = torch.mm(sub_X, sub_X.t())
        cosine_sim_mat2 = torch.mm(mean_sub_X, sub_X.t())/(torch.norm(mean_sub_X, dim=-1).unsqueeze(1)*torch.norm(sub_X, dim=-1).unsqueeze(0))

        
        cosine_sim_min = torch.min(cosine_sim_mat).item()
        cosine_sim_min2 = torch.min(cosine_sim_mat2).item()
        
        
        if cosine_sim_min < min_cosine_sim_min:
            min_cosine_sim_min = cosine_sim_min
            min_cosine_sim_min_idx = label
        
        if cosine_sim_min2 < min_cosine_sim_min2:
            min_cosine_sim_min2 = cosine_sim_min2
            min_cosine_sim_min_idx2 = label
        
        cosine_sim_min_ls.append(cosine_sim_min)
        cosine_sim_mat_ls.append(cosine_sim_mat)
        
        cluster_centroid_ls.append(mean_sub_X)
        
        # unique_sub_sample_ids = np.unique(sub_sample_ids)
        
        # most_similar_sub_X_tensor, most_similar_patch_ids_ls, most_similar_granularity_ids_ls, most_similar_sample_ids_ls = select_patch_embeddings_closest_to_centroids(unique_sub_sample_ids, sub_X, sub_sample_ids, sub_sample_patch_ids, sub_sample_granularity_ids)
        
        
        cluster_sample_count_ls.append(len(sub_sample_ids))
        # cluster_unique_sample_ids_ls.append(unique_sub_sample_ids)
        cluster_sample_ids_ls.append(sub_sample_ids.tolist())
        cluster_sub_X_tensor_ls.append(sub_X)
        # cluster_sub_X_patch_ids_ls.append(most_similar_patch_ids_ls)
        # cluster_sub_X_granularity_ids_ls.append(most_similar_granularity_ids_ls)
        
    cluster_centroid_tensor = torch.cat(cluster_centroid_ls, dim=0)
    
    return cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_sample_ids_ls

    
    # for k in range(1000, max_k + 1, 100):
    #     centroids, cluster_assignments, max_similarity = kmeans_cosine(X, k)
    #     min_max_similarity = torch.arccos(torch.min(max_similarity))/np.pi*180
    #     print("min max similarity: ", min_max_similarity.item())
    #     if min_max_similarity < max_deg:
    #         break
    #     torch.cuda.empty_cache()
    #     # inertia = 0
    #     # for i in range(k):
    #     #     inertia += F.cosine_similarity(X[cluster_assignments == i], centroids[i].unsqueeze(0)).sum()
    #     # inertias.append(inertia.item())
    
    # # inertias = torch.tensor(inertias)
    # # inertias_diff = inertias[:-1] - inertias[1:]
    
    # # optimal_k = torch.argmin(inertias_diff) + 1
    # # return optimal_k
    # return k, centroids, cluster_assignments, max_similarity


