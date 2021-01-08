#!/bin/bash 
#$ -cwd 
#$ -S /bin/bash 
#$ -M azpoliak@cs.jhu.edu 
#$ -m eas 
#$ -l 'gpu=1,hostname=b1[123456789]*,mem_free=2.1g,ram_free=2.1g' 
#$ -e /export/c02/apoliak/recast-experiments/infersent/emnlp-camera-ready/factuality/infersent__nopretaining/out.err 
#$ -o /export/c02/apoliak/recast-experiments/infersent/emnlp-camera-ready/factuality/infersent__nopretaining/out.log 

source /home/apoliak/.bashrc 
source activate mtie 

#VARS 
NLI_PATH=/export/c02/apoliak/recast-NLI-data/camera-ready-data/experiments/factuality/ 
OUTPUT_PATH=/export/c02/apoliak/recast-experiments/infersent/emnlp-camera-ready/factuality/infersent__nopretaining/ 

device=`free-gpu` 
echo $device 

time CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$device python /home/apoliak/InferSent-DNC/train_nli.py \
--nlipath $NLI_PATH \
--outputdir ${OUTPUT_PATH} \
--gpu_id $device 