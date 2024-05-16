
# Prepare data

To run the experiments on crepe dataset, you need to download visual genome data from [here](https://drive.google.com/drive/folders/11dMtJByk7zmbQjV47PXVwfmakN3Gr5Ic) and extract both "VG_100K.zip" and "VG_100K_2.zip". Suppose this dataset is stored in /path/to/data/. Also for other datasets like the text retrieval dataset trec-covid dataset, we also assume that the dataset is stored in /path/to/data/ folder.


The query files are stored in "prod_hard_negatives/". In this demo, we only used a subset of queries, which are stored in "prod_hard_negatives/prod_vg_hard_negs_swap_all4.csv"

# Example command on how to run the code

## Run the code without decomposing images or queries:

```
python explore_image_retrieval.py --dataset_name crepe --data_path /path/to/data/ --query_count -1 --total_count -1 
```

in this command, "--total_count" represents the number of documents used for retrieval tasks, -1 means that we used all documents while a positive number means that we only used a subset of the entire document set. For the purpose of quick demonstration, "total_count" could be 500 or 1000.


## Run the code by decomposing images and queries:
```
python explore_image_retrieval.py --dataset_name crepe --data_path /path/to/data/ --query_count -1 --total_count -1  --img_concept --query_concept
```

in this command, "--img_concept" represents partitioning images or documents while "--query_concept" represents partitioning queries.


## Run the code by decomposing images and queries while using the clustering-based indexes at the same time

```
python explore_image_retrieval.py --dataset_name crepe --data_path /path/to/data/ --query_count -1 --total_count -1  --img_concept --query_concept --search_by_cluster
```

in this command, "--search_by_cluster" means that we construct the clustering-based indexes for speed-ups




# Incorporating other datasets

## Run the code on another image retrieval dataset

Here we assume that the dataset is adapted by an image captioning dataset. Then this indicates that each image will have only one caption as one query. If there are multiple captions for one image, we just use the first one. To adapt the other image captioning dataset for the image retrieval task, we can simply replace this function "load_crepe_datasets" in the main.py file with another function that can return four variables "queries", "raw_img_ls", "sub_queries_ls", "img_idx_ls", in which "queries" is a list of queries (i.e., the image captions), "raw_img_ls" represents a list of raw images (in pillow image format), "sub_queries_ls" represents a list of sub-query lists which correspond to each of the query while "img_idx_ls" represents the list of image ids.


Note that the retrieval performance is evaluated based on the ground-truth mappings between the queries and the documents/images, which specify the similarity between each query and each image (2 for very similar while 0 means not similar at all). Since such mappings don't exist for image captioning dataset, we therefore defined one function called "construct_qrels" to create such mappings in which each pair of the image and the caption has the similarity score 2. If such ground-truth mappings are given for a image retrieval dataset, we can comment out this function.


## Run the code on another text retrieval dataset

We can start from the datasets listed in this  [git repo](https://github.com/beir-cellar/beir/tree/main). Note that the splited queries should be put in the folder /path/to/data/${dataset_name}. For example, for trec-covid dataset, we need to move the file "queries_with_subs.jsonl" to /path/to/data/${dataset_name} before the experiments.





