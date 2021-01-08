#!/bin/bash 
#$ -cwd 
#$ -S /bin/bash 
#$ -M azpoliak@cs.jhu.edu 
#$ -m eas 
#$ -l 'gpu=1,hostname=c*,mem_free=2.1g,ram_free=2.1g' 
#$ -e /export/c02/apoliak/recast-experiments/infersent/emnlp-camera-ready/ner/infersent_mnli_fixed/out.err 
#$ -o /export/c02/apoliak/recast-experiments/infersent/emnlp-camera-ready/ner/infersent_mnli_fixed/out.log 

source /home/apoliak/.bashrc 
source activate mtie 

#VARS 
NLI_PATH=/export/c02/apoliak/recast-NLI-data/camera-ready-data/experiments/ner/ 
OUTPUT_PATH=/export/c02/apoliak/recast-experiments/infersent/emnlp-camera-ready/ner/infersent_mnli_fixed/ 
PRETRAINED_PATH=/export/a13/ahaldar/InferSent/output-mnli-3way/ 

device=`free-gpu` 
echo $device 

time CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$device python /home/apoliak/InferSent-DNC/eval.py \
--nlipath $NLI_PATH \
--outputdir ${OUTPUT_PATH} \
--model ${PRETRAINED_PATH}model.pickle \
--gpu_id $device 