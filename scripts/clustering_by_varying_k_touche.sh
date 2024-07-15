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

dataset_name="webis-touche2020"

method=two

export CUDA_VISIBLE_DEVICES=1

nprobe_query=20

cd ../

topk_vals=(100 200 300 400 500 800 1000)

# 1902  CUDA_VISIBLE_DEVICES=1  python main.py --dataset_name webis-touche2020 --data_path /data2/wuyinjun/ --query_count -1 --total_count -1 --query_concept --img_concept --algebra_method four --search_by_cluster --clustering_topk 500 --clustering_number 0.001 --clustering_doc_count_factor 200 --nprobe_query 20

for val in ${topk_vals[@]}
do
	command="python main.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count -1 --total_count -1 --query_concept --img_concept --algebra_method ${method}  --search_by_cluster --clustering_topk $val  --clustering_number 0.001  --clustering_doc_count_factor 200 --nprobe_query ${nprobe_query} --prob_agg sum"
        echo "$command"
	$command > ${data_path}/${dataset_name}/output_full_query_full_data_clustering_method_${method}_topk_${val}.txt 2>&1 

done

method=four

for val in ${topk_vals[@]}
do
        command="python main.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count -1 --total_count -1 --query_concept --img_concept --algebra_method ${method}  --search_by_cluster --clustering_topk $val  --clustering_number 0.001  --clustering_doc_count_factor 200 --nprobe_query ${nprobe_query} --prob_agg sum"
        echo "$command"
        $command > ${data_path}/${dataset_name}/output_full_query_full_data_clustering_method_${method}_topk_${val}.txt 2>&1

done

