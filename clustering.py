
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch

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

def clustering_determine_k(X_ls, max_k=2000, max_deg = 20):
    """
    Determine the optimal number of clusters using the elbow method.

    Parameters:
    - X: torch.Tensor, shape (N, D), input data points
    - max_k: int, maximum number of clusters to consider

    Returns:
    - optimal_k: int, optimal number of clusters
    """
    inertias = []
    X = torch.cat(X_ls, dim=0)
    X = X_ls[0]
    X = X/ torch.norm(X, dim=1, keepdim=True)
    
    # clustering = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete", distance_threshold=0.1).fit(X.cpu().numpy())
    
    clustering = Birch(threshold=0.3, n_clusters=None).fit(X.cpu().numpy())

    cosine_sim_min_ls = []
    cosine_sim_mat_ls = []
    min_cosine_sim_min = 1
    min_cosine_sim_min_idx = 0
    
    min_cosine_sim_min2 = 1
    min_cosine_sim_min_idx2 = 0
    for label in np.unique(clustering.labels_):
        sub_X = X[clustering.labels_ == label]
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
    
    for k in range(1000, max_k + 1, 100):
        centroids, cluster_assignments, max_similarity = kmeans_cosine(X, k)
        min_max_similarity = torch.arccos(torch.min(max_similarity))/np.pi*180
        print("min max similarity: ", min_max_similarity.item())
        if min_max_similarity < max_deg:
            break
        torch.cuda.empty_cache()
        # inertia = 0
        # for i in range(k):
        #     inertia += F.cosine_similarity(X[cluster_assignments == i], centroids[i].unsqueeze(0)).sum()
        # inertias.append(inertia.item())
    
    # inertias = torch.tensor(inertias)
    # inertias_diff = inertias[:-1] - inertias[1:]
    
    # optimal_k = torch.argmin(inertias_diff) + 1
    # return optimal_k
    return k, centroids, cluster_assignments, max_similarity


