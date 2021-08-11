#!/bin/sh

# Usage:
# ./run_train_whisper_flen64ms_fshift8ms.sh 1 0 experiment1
# If you want to start from the beggining, set the first argument to 0.
# If you want to skip the first stage, set it to 1.

db_dir="/misc/raid58/kameoka.hirokazu/db/ATR503Seki/train"
dataset_name="whisper"
exp_name=${3}
#exp_name="experiment1"
cond="--resume 0 --epochs 1000 --snapshot 100"

feat_dir="./dump/${dataset_name}/feat/train"
dconf_path="./dump/${dataset_name}/data_config.json"
stat_path="./dump/${dataset_name}/stat.pkl"
normfeat_dir="./dump/${dataset_name}/norm_feat/train"
model_dir="./model/${dataset_name}"
log_dir="./logs/${dataset_name}"

if [[ ${1} -le 0 ]]; then
       python extract_features.py --src ${db_dir} --dst ${feat_dir} --conf ${dconf_path}
       python compute_statistics.py --src ${feat_dir} --stat ${stat_path}
       python normalize_features.py --src ${feat_dir} --dst ${normfeat_dir} --stat ${stat_path}
fi

if [[ ${1} -le 1 ]]; then
       python main.py -g ${2} \
              --data_rootdir ${normfeat_dir} \
              --model_rootdir ${model_dir} \
              --log_dir ${log_dir} \
              --experiment_name ${exp_name} \
              ${cond}
fi