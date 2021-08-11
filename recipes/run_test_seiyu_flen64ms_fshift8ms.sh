#!/bin/sh

# Example:
# ./run_test_seiyu_flen64ms_fshift8ms.sh 0 experiment1
# ./run_test_seiyu_flen64ms_fshift8ms.sh 0

db_dir="/misc/raid58/kameoka.hirokazu/db/seiyu/wav/test"
dataset_name="seiyu"
exp_name=${2}
#exp_name="experiment1"

dconf_path="./dump/${dataset_name}/data_config.json"
stat_path="./dump/${dataset_name}/stat.pkl"
out_dir="./out/${dataset_name}"
model_dir="./model/${dataset_name}"
vocoder_dir="pwg/egs/seiyu_flen64ms_fshift8ms/voc1"

python convert.py -g ${1} \
	--input ${db_dir} \
	--dataconf ${dconf_path} \
	--stat ${stat_path} \
	--out ${out_dir} \
	--model_rootdir ${model_dir} \
	--experiment_name ${exp_name} \
	--voc_dir ${vocoder_dir}