#!/bin/bash 
#$ -cwd 
#$ -S /bin/bash 
#$ -M azpoliak@cs.jhu.edu 
#$ -m eas 
#$ -l 'gpu=1,hostname=b1[123456789]*|c*|b20,mem_free=2.1g,ram_free=2.1g' 
#$ -o /export/c02/apoliak/recast-experiments/hypoths/emnlp-camera-ready/verbcorner/hypoths_combined_fixed/out.log 
#$ -e /export/c02/apoliak/recast-experiments/hypoths/emnlp-camera-ready/verbcorner/hypoths_combined_fixed/out.err 

source /home/apoliak/.bashrc 
source activate mtie 

NLI_PATH=/export/c02/apoliak/recast-NLI-data/camera-ready-data/experiments/verbcorner/ 
OUTPUT_PATH=/export/c02/apoliak/recast-experiments/hypoths/emnlp-camera-ready/verbcorner/hypoths_combined_fixed/ 
PRETRAINED_PATH=/export/c02/apoliak/recast-experiments/hypoths/emnlp-camera-ready/combined/hypoths__nopretaining/ 

device=`free-gpu` 
echo $device 

time CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$device python /home/apoliak/nli-hypothes-only/src/eval.py \
--embdfile /export/b02/apoliak/embeddings/glove/glove.840B.300d.txt \
--outputdir ${OUTPUT_PATH} \
--pool_type max \
--val_lbls_file ${NLI_PATH}labels.dev \
--val_src_file ${NLI_PATH}s2.dev.hyps \
--train_lbls_file ${NLI_PATH}labels.train \
--train_src_file ${NLI_PATH}s2.train.hyps \
--test_lbls_file ${NLI_PATH}labels.test \
--test_src_file ${NLI_PATH}s2.test.hyps \
--gpu_id $device \
--model ${PRETRAINED_PATH}model.pickle 