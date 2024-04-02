
# Prepare data

You need to download visual genome data from [here](https://drive.google.com/drive/folders/11dMtJByk7zmbQjV47PXVwfmakN3Gr5Ic) and extract both "VG_100K.zip" and "VG_100K_2.zip". Suppose this dataset is stored in /path/to/data/


The query files are stored in "prod_hard_negatives/". In this demo, we only used a subset of queries, which are stored in "prod_hard_negatives/prod_vg_hard_negs_swap_all4.csv"

# How to run the code

## Run the code without decomposing images or queries:

```
python explore_image_retrieval.py --dataset_name crepe --data_path /path/to/data/
```

The retrieval performance is (larger numbers are better):

NDCG@1: 0.2273
NDCG@3: 0.2846
MAP@1: 0.2273
MAP@3: 0.2727
Recall@1: 0.2273
Recall@3: 0.3182
P@1: 0.2273
P@3: 0.1061


## Run the code by decomposing images:
```
python explore_image_retrieval.py --dataset_name crepe --data_path /path/to/data/ --img_concept
```

The retrieval performance is ():
NDCG@1: 0.2273
NDCG@3: 0.3074
MAP@1: 0.2273
MAP@3: 0.2879
Recall@1: 0.2273
Recall@3: 0.3636
P@1: 0.2273
P@3: 0.1212
