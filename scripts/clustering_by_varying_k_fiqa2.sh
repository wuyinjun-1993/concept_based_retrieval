#!/bin/bash




function handle_sigint {
    echo
    echo "Caught SIGINT (Ctrl+C)! Cleaning up..."
    # Perform any necessary cleanup here
    exit 1
}

# Set the trap to call handle_sigint on SIGINT
trap handle_sigint SIGINT


data_path="/data2/wuyinjun/"

dataset_name="fiqa"

method=$1

export CUDA_VISIBLE_DEVICES=$2

nprobe_query=20

cd ../


# --data_path /data2/wuyinjun/ --query_count -1 --total_count -1 --query_concept --img_concept --algebra_method two  --search_by_cluster --clustering_topk 2000  --clustering_number 0.01  --clustering_doc_count_factor 100 --nprobe_query 20 

topk_vals=(500 1000 1500 2000 5000 10000)

for val in ${topk_vals[@]}
do
	command="python main.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count -1 --total_count -1 --query_concept --img_concept --algebra_method ${method}  --search_by_cluster --clustering_topk $val  --clustering_number 0.01  --clustering_doc_count_factor 20000 --nprobe_query ${nprobe_query}"
        echo "$command"
	$command > ${data_path}/${dataset_name}/output_full_query_full_data_clustering_method_${method}_2_topk_${val}.txt 2>&1 

done
