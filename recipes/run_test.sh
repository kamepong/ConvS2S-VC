#!/bin/bash

# Copyright 2021 Hirokazu Kameoka
# 
# Usage:
# ./run_test.sh [-g gpu] [-e exp_name] [-c checkpoint] [-a attention_mode] [-v vocoder_type]
# Options:
#     -g: GPU device# (-1 indicates CPU)
#     -e: Experiment name (e.g., "conv_exp1")
#     -c: Model checkpoint to load (0 indicates the newest model)
#     -a: Attention mode ("raw" (default), "forward", or "diagonal")
#     -v: Vocoder type ("parallel_wavegan.v1" (default) or "hifigan.v1")

db_dir="/path/to/dataset/test"
dataset_name="mydataset"
attention_mode="raw"
checkpoint=0
vocoder_type="parallel_wavegan.v1"

while getopts "g:e:c:a:v:" opt; do
	case $opt in
		g ) gpu=$OPTARG;;
		e ) exp_name=$OPTARG;;
		c ) checkpoint=$OPTARG;;
		a ) attention_mode=$OPTARG;;
		v ) vocoder_type=$OPTARG;;
	esac
done

# If the -a option is abbreviated...
case ${attention_mode} in
	"r" ) attention_mode="raw";;
	"f" ) attention_mode="forward";;
	"d" ) attention_mode="diagonal";;
esac

# If the -v option is abbreviated...
case ${vocoder_type} in
	"pwg" ) vocoder_type="parallel_wavegan.v1";;
	"hfg" ) vocoder_type="hifigan.v1";;
esac

echo "Experiment name: ${exp_name}, Attention mode: ${attention_mode}, Vocoder: ${vocoder_type}"

dconf_path="./dump/${dataset_name}/data_config.json"
stat_path="./dump/${dataset_name}/stat.pkl"
out_dir="./out/${dataset_name}"
model_dir="./model/${dataset_name}"
vocoder_dir="pwg/egs/arctic_4spk_flen64ms_fshift8ms/voc1"

python convert.py -g ${gpu} \
	--input ${db_dir} \
	--dataconf ${dconf_path} \
	--stat ${stat_path} \
	--out ${out_dir} \
	--model_rootdir ${model_dir} \
	--experiment_name ${exp_name} \
	--vocoder ${vocoder_type} \
	--voc_dir ${vocoder_dir} \
	--attention_mode ${attention_mode} \
	--checkpoint ${checkpoint}