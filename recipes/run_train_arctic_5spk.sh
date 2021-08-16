#!/bin/sh

# Copyright 2021 Hirokazu Kameoka
# 
# Usage:
# ./run_train_arctic_5spk.sh [-g gpu] [-s stage] [-e exp_name]
# Options:
#     -g: GPU device# 
#     -s: Stage to start (0 or 1)
#     -e: Experiment name (e.g., "exp1")

# Default values
db_dir="/misc/raid58/kameoka.hirokazu/db/arctic_5spk/wav/training"
dataset_name="arctic_5spk"
gpu=0
start_stage=0
exp_name="conv_exp1"

while getopts "g:s:e:" opt; do
       case $opt in
              g ) gpu=$OPTARG;;
              s ) start_stage=$OPTARG;;
              e ) exp_name=$OPTARG;;
       esac
done

feat_dir="./dump/${dataset_name}/feat/train"
dconf_path="./dump/${dataset_name}/data_config.json"
stat_path="./dump/${dataset_name}/stat.pkl"
normfeat_dir="./dump/${dataset_name}/norm_feat/train"
model_dir="./model/${dataset_name}"
log_dir="./logs/${dataset_name}"

# Stage 0: Feature extraction
if [[ ${start_stage} -le 0 ]]; then
       python extract_features.py --src ${db_dir} --dst ${feat_dir} --conf ${dconf_path}
       python compute_statistics.py --src ${feat_dir} --stat ${stat_path}
       python normalize_features.py --src ${feat_dir} --dst ${normfeat_dir} --stat ${stat_path}
fi 

# Stage 1: Model training
if [[ ${start_stage} -le 1 ]]; then
       python train.py -g ${gpu} \
              --data_rootdir ${normfeat_dir} \
              --model_rootdir ${model_dir} \
              --log_dir ${log_dir} \
              --experiment_name ${exp_name} \
              ${cond}
fi