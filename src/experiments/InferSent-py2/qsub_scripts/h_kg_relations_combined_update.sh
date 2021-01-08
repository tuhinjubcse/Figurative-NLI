#!/bin/bash 
#$ -cwd 
#$ -S /bin/bash 
#$ -M azpoliak@cs.jhu.edu 
#$ -m eas 
#$ -l 'gpu=1,hostname=b1[123456789]*|c*|b20,mem_free=2.1g,ram_free=2.1g' 
#$ -o /export/c02/apoliak/recast-experiments/hypoths/emnlp-camera-ready/kg_relations/hypoths_combined_update/out.log 
#$ -e /export/c02/apoliak/recast-experiments/hypoths/emnlp-camera-ready/kg_relations/hypoths_combined_update/out.err 

source /home/apoliak/.bashrc 
source activate mtie 

NLI_PATH=/export/c02/apoliak/recast-NLI-data/camera-ready-data/experiments/kg_relations/ 
OUTPUT_PATH=/export/c02/apoliak/recast-experiments/hypoths/emnlp-camera-ready/kg_relations/hypoths_combined_update/ 
PRETRAINED_PATH=/export/c02/apoliak/recast-experiments/hypoths/emnlp-camera-ready/combined/hypoths__nopretaining/ 

device=`free-gpu` 
echo $device 

time CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$device python /home/apoliak/nli-hypothes-only/src/train.py \
--embdfile /export/b02/apoliak/embeddings/glove/glove.840B.300d.txt \
--outputdir ${OUTPUT_PATH} \
--pool_type max \
--val_lbls_file ${NLI_PATH}labels.dev \
--val_src_file ${NLI_PATH}s2.dev.hyps \
--train_lbls_file ${NLI_PATH}labels.train \
--train_src_file ${NLI_PATH}s2.train.hyps \
--test_lbls_file ${NLI_PATH}labels.test \
--test_src_file ${NLI_PATH}s2.test.hyps \
--gpu_id $device --pre_trained_model ${PRETRAINED_PATH}model.pickle 