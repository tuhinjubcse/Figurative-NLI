#!/bin/bash 
#$ -cwd 
#$ -S /bin/bash 
#$ -M azpoliak@cs.jhu.edu 
#$ -m eas 
#$ -l 'gpu=1,hostname=b1[123456789]*|c*|b20,mem_free=2.1g,ram_free=2.1g' 
#$ -e /export/c02/apoliak/recast-experiments/infersent/emnlp-camera-ready/verbnet/infersent_combined_fixed/out.err 
#$ -o /export/c02/apoliak/recast-experiments/infersent/emnlp-camera-ready/verbnet/infersent_combined_fixed/out.log 

source /home/apoliak/.bashrc 
source activate mtie 

#VARS 
NLI_PATH=/export/c02/apoliak/recast-NLI-data/camera-ready-data/experiments/verbnet/ 
OUTPUT_PATH=/export/c02/apoliak/recast-experiments/infersent/emnlp-camera-ready/verbnet/infersent_combined_fixed/ 
PRETRAINED_PATH=/export/c02/apoliak/recast-experiments/infersent/emnlp-camera-ready/combined/infersent__nopretaining/ 

device=`free-gpu` 
echo $device 

time CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$device python /home/apoliak/InferSent-DNC/eval.py \
--nlipath $NLI_PATH \
--outputdir ${OUTPUT_PATH} \
--model ${PRETRAINED_PATH}model.pickle \
--gpu_id $device 