import utils
from tqdm import tqdm
import os

def calculate_bbox_size(bbox1):
    left1, upper1, right1, lower1 = bbox1
    area_bbox1 = (right1 - left1) * (lower1 - upper1)
    return area_bbox1

def calculate_iou(bbox1, bbox2):
    #print(bbox1)
    #print(bbox2)
    left1, upper1, right1, lower1 = bbox1
    left2, upper2, right2, lower2 = bbox2

    # Determine the coordinates of the intersection rectangle
    inter_left = max(left1, left2)
    inter_upper = max(upper1, upper2)
    inter_right = min(right1, right2)
    inter_lower = min(lower1, lower2)

    # Compute the area of intersection
    inter_width = max(0, inter_right - inter_left)
    inter_height = max(0, inter_lower - inter_upper)
    intersection_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    area_bbox1 = (right1 - left1) * (lower1 - upper1)
    area_bbox2 = (right2 - left2) * (lower2 - upper2)

    # Determine which patch is smaller
    smaller_patch_area = min(area_bbox1, area_bbox2)

    # Compute the containment as the intersection area over the smaller patch area
    containment = intersection_area / smaller_patch_area if smaller_patch_area > 0 else 0
    return containment

def generate_containment_list(intersect_ratio, all_valid_bboxes_list):
  all_containment_list = []
  for box_list in all_valid_bboxes_list:
    containment_list = [[] for _ in range(len(box_list))]

    # Check for overlap (IoU > 50%) between valid bounding boxes
    for i, bbox1 in enumerate(box_list):
      for j, bbox2 in enumerate(box_list):
          if i != j:
              iou = calculate_iou(bbox1, bbox2)
              if iou > intersect_ratio:
                    # if calculate_bbox_size(bbox1) > 5*calculate_bbox_size(bbox2):
                    #     containment_list[i].append(j)
                    # elif calculate_bbox_size(bbox1) > calculate_bbox_size(bbox2):
                    #     containment_list[i].append(j)
                    if not calculate_bbox_size(bbox2) > 2*calculate_bbox_size(bbox1):
                        containment_list[i].append(j)
    all_containment_list.append(containment_list)
  return all_containment_list

def generate_all_containment_list(all_bboxes_ls, intersection_threshold=0.9):
    
    all_containment_ls = []
    print("intersect_ratio: ", intersection_threshold)
    for bboxes_ls in tqdm(all_bboxes_ls):
        all_valid_bboxes_list_11 = []
        all_valid_bboxes_list_11.append(bboxes_ls)
        containment_list = generate_containment_list(intersection_threshold, all_valid_bboxes_list_11)[0]
        all_containment_ls.append(containment_list)
    return all_containment_ls

def is_bbox_overlapped(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    intersection_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    
    # Check if the intersection area is positive
    size1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    size2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    min_size = min(size1, size2)
    # others: 0.1
    return intersection_area > 0.1 * min_size
    # return intersection_area > 0.5 * min_size


def is_bbox_overlapped_text(bbox1, bbox2):
    x1_1, x2_1 = bbox1
    x1_2, x2_2 = bbox2
    
    # Calculate intersection area
    intersection_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    
    # Check if the intersection area is positive
    size1 = (x2_1 - x1_1)# * (y2_1 - y1_1)
    size2 = (x2_2 - x1_2)# * (y2_2 - y1_2)
    min_size = min(size1, size2)
    thres = 0.1 * min_size
    # print(intersection_area, thres)
    return intersection_area > 0.2 * min_size

def add_clustering_nbs_info(clustering_nbs_mappings, sample_idx, cluster_idx1, cluster_idx2):
    if cluster_idx1 not in clustering_nbs_mappings:
        clustering_nbs_mappings[(sample_idx, cluster_idx1)] = set()
    if cluster_idx2 not in clustering_nbs_mappings:
        clustering_nbs_mappings[(sample_idx, cluster_idx2)] = set()
    
    clustering_nbs_mappings[(sample_idx, cluster_idx1)].add(cluster_idx2)
    clustering_nbs_mappings[(sample_idx, cluster_idx1)].add(cluster_idx1)
    clustering_nbs_mappings[(sample_idx, cluster_idx2)].add(cluster_idx1)
    clustering_nbs_mappings[(sample_idx, cluster_idx2)].add(cluster_idx2)

def determine_overlapped_bboxes(bboxes_ls, is_img_retrieval=False, sample_patch_ids_to_cluster_id_mappings=None):
    
    bbox_nb_ls = []
    
    # if sample_patch_ids_to_cluster_id_mappings is not None:
    clustering_nbs_mappings = dict()
    
    for b_idx in tqdm(range(len(bboxes_ls))):
        bboxes = bboxes_ls[b_idx]
        
        # curr_nb_ls = [[] for _ in range(len(bboxes) + 1)]
        if is_img_retrieval:
            curr_nb_ls = [[] for _ in range(len(bboxes))]
        else:
            curr_nb_ls = [[] for _ in range(len(bboxes) + 1)]
        for idx in range(len(bboxes)):
            bbox = bboxes[idx]
            
            if sample_patch_ids_to_cluster_id_mappings is not None:
                cluster_idx = sample_patch_ids_to_cluster_id_mappings[b_idx][idx]
            
            for sub_idx in range(len(bboxes)):
                if sample_patch_ids_to_cluster_id_mappings is not None:
                    sub_cluster_idx = sample_patch_ids_to_cluster_id_mappings[b_idx][sub_idx]
                
                
                if idx != sub_idx:
                    sub_bbox = bboxes[sub_idx]
                    if is_img_retrieval:
                        if is_bbox_overlapped(bbox, sub_bbox):
                            curr_nb_ls[idx].append(sub_idx)
                            if sample_patch_ids_to_cluster_id_mappings is not None:
                                add_clustering_nbs_info(clustering_nbs_mappings, b_idx, cluster_idx, sub_cluster_idx)
                    else:
                        if is_bbox_overlapped_text(bbox, sub_bbox):
                            curr_nb_ls[idx].append(sub_idx)
                            if sample_patch_ids_to_cluster_id_mappings is not None:
                                add_clustering_nbs_info(clustering_nbs_mappings, b_idx, cluster_idx, sub_cluster_idx)
                else:
                    curr_nb_ls[idx].append(sub_idx)
            if not is_img_retrieval:
                curr_nb_ls[idx].append(len(bboxes))
        if not is_img_retrieval:
            curr_nb_ls[len(bboxes)].extend(list(range(len(bboxes) + 1)))
        bbox_nb_ls.append(curr_nb_ls)
    
    return bbox_nb_ls, clustering_nbs_mappings


def add_full_bbox_to_bbox_nb_ls(bbox_nb_ls, bboxes_ls, patch_img_embes_ls):
    for idx in range(len(bbox_nb_ls)):
        patch_img_embes = patch_img_embes_ls[idx]
        bboxes = bboxes_ls[idx]
        bbox_nbs = bbox_nb_ls[idx]
        if len(bbox_nbs) < len(patch_img_embes):
            
            for sub_idx in range(len(bbox_nbs)):
                bbox_nbs[sub_idx].append(len(bboxes))
            
            bbox_nbs.append(list(range(len(bboxes) + 1)))
            
def remove_full_bbox_from_bbox_nb_ls(bbox_nb_ls, bboxes_ls, patch_img_embes_ls):
    for idx in tqdm(range(len(bbox_nb_ls)), desc="remove full bbox"):
        patch_img_embes = patch_img_embes_ls[idx]
        bboxes = bboxes_ls[idx]
        bbox_nbs = bbox_nb_ls[idx]
        if len(bbox_nbs) == len(patch_img_embes):
            del bbox_nbs[-1]
            for sub_idx in range(len(bbox_nbs)):
                del bbox_nbs[sub_idx][-1]

def init_bbox_nbs(args, patch_count_ls, samples_hash, bboxes_ls, patch_emb_by_img_ls, sample_patch_ids_to_cluster_id_mappings=None):
    # patch_count_ls = sorted(patch_count_ls)
    patch_count_str = "_".join([str(item) for item in patch_count_ls])
    
    extra_suffix=""
            
    if args.use_raptor:
        extra_suffix += "_raptor"
    
    bboxes_overlap_file_name = "output/bboxes_overlap_" + samples_hash + "_" + patch_count_str + extra_suffix + ".pkl"   
    
    if os.path.exists(bboxes_overlap_file_name):
        print("load bbox neighbor information from file: ", bboxes_overlap_file_name)
        bboxes_overlap_ls, clustering_nbs_mappings = utils.load(bboxes_overlap_file_name)
    else:
        print("start generating bbox neighbor information from file: ")
        bboxes_overlap_ls, clustering_nbs_mappings = determine_overlapped_bboxes(bboxes_ls, is_img_retrieval=args.is_img_retrieval, sample_patch_ids_to_cluster_id_mappings=sample_patch_ids_to_cluster_id_mappings)
        utils.save((bboxes_overlap_ls, clustering_nbs_mappings), bboxes_overlap_file_name)
    if not args.is_img_retrieval:
        if args.dataset_name == "webis-touche2020":
            add_full_bbox_to_bbox_nb_ls(bboxes_overlap_ls, bboxes_ls, patch_emb_by_img_ls)
        else:
            remove_full_bbox_from_bbox_nb_ls(bboxes_overlap_ls, bboxes_ls, patch_emb_by_img_ls)
    return bboxes_overlap_ls, clustering_nbs_mappings
