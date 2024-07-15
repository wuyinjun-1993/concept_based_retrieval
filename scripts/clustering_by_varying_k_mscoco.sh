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

dataset_name="mscoco_40k"

method=$1

export CUDA_VISIBLE_DEVICES=$2

nprobe_query=20

cd ../

topk_vals=(50 80 100 150 200 250)

for val in ${topk_vals[@]}
do
	command="python main.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count 100 --total_count -1 --query_concept --img_concept --algebra_method ${method}  --search_by_cluster --clustering_topk $val  --clustering_number 0.02  --clustering_doc_count_factor 10000 --nprobe_query ${nprobe_query} --dependency_topk 30"
        echo "$command"
	$command > ${data_path}/${dataset_name}/output_full_query_full_data_clustering_method_${method}_topk_${val}.txt 2>&1 

done
