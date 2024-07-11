#!/bin/bash

#[4]+  Done                    CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name trec-covid --data_path /data2/wuyinjun/ --query_count -1 --total_count -1 --query_concept --img_concept --algebra_method four --search_by_cluster --clustering_topk 5000 --clustering_number 0.02 --clustering_doc_count_factor 20000 --nprobe_query 50 --prob_agg sum > /data2/wuyinjun/trec-covid/output_full_clustering_method_four.txt 2>&1



function handle_sigint {
    echo
    echo "Caught SIGINT (Ctrl+C)! Cleaning up..."
    # Perform any necessary cleanup here
    exit 1
}

# Set the trap to call handle_sigint on SIGINT
trap handle_sigint SIGINT


data_path="/data2/wuyinjun/"

dataset_name="trec-covid"

method=$1

export CUDA_VISIBLE_DEVICES=$2

nprobe_query=50

cd ../


patch_count_size=(1 2 3 4 5)

for val in ${patch_count_size[@]}
do
	command="python main.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count -1 --total_count -1 --query_concept --img_concept --algebra_method ${method}  --search_by_cluster --clustering_topk 150  --clustering_number 0.02  --clustering_doc_count_factor 20000 --nprobe_query ${nprobe_query} --prob_agg sum --subset_patch_count ${val}"
        echo "$command"
	$command > ${data_path}/${dataset_name}/output_full_query_full_data_clustering_method_${method}_patch_count_${val}.txt 2>&1 

done
