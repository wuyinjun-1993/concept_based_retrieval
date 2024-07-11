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

random_seed=$3

nprobe_query=100

cd ../

patch_count_size=(1 2 3 4)

for val in ${patch_count_size[@]}
do
	command="python main.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count 100 --total_count -1 --query_concept --img_concept --algebra_method ${method}  --search_by_cluster --clustering_topk 2000  --clustering_number 0.02  --clustering_doc_count_factor 10000 --nprobe_query ${nprobe_query} --dependency_topk 30 --random_seed ${random_seed} --subset_patch_count ${val}"
        echo "$command"
	$command > ${data_path}/${dataset_name}/output_full_query_full_data_clustering_method_${method}_patch_count_${val}_random_${random_seed}.txt 2>&1 

done
