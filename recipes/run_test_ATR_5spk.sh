#!/bin/sh

# Copyright 2021 Hirokazu Kameoka
# 
# Usage:
# ./run_test_ATR_5spk.sh [-g gpu] [-e exp_name] [-f] [-d]
# Options:
#     -g: GPU device#  
#     -e: Experiment name (e.g., "conv_exp1")
#     -f: Forward attention mode if specified
#     -d: Diagonal attention mode if specified

db_dir="/misc/raid58/kameoka.hirokazu/db/ATR_Bset_5spk/wav/test"
dataset_name="ATR_5spk"
refine_type="raw"

while getopts "g:e:fd" opt; do
       case $opt in
              g ) gpu=$OPTARG;;
              e ) exp_name=$OPTARG;;
			  f ) refine_type="forward";;
			  d ) refine_type="diagonal";;
       esac
done

echo "Experiment name: ${exp_name}, Attention mode: ${refine_type}"

dconf_path="./dump/${dataset_name}/data_config.json"
stat_path="./dump/${dataset_name}/stat.pkl"
out_dir="./out/${dataset_name}"
model_dir="./model/${dataset_name}"
vocoder_dir="pwg/egs/ATR_all_flen64ms_fshift8ms/voc1"

python convert.py -g ${gpu} \
	--input ${db_dir} \
	--dataconf ${dconf_path} \
	--stat ${stat_path} \
	--out ${out_dir} \
	--model_rootdir ${model_dir} \
	--experiment_name ${exp_name} \
	--voc_dir ${vocoder_dir} \
	--refine_type ${refine_type}