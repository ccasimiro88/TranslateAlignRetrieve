#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

TRAIN_FILE=$1
SQUAD_V2=$2
TRAIN_FROM_MODEL=$3

TRAINING_DIR=${SCRIPT_DIR}/data/training/m-bert_$(basename ${TRAIN_FILE})
mkdir -p ${TRAINING_DIR}

# If no starting model is provided,
# start the training from the Multilingual BERT
if [[ -z ${TRAIN_FROM_MODEL} ]]; then
  TRAIN_FROM_MODEL=bert-base-multilingual-cased
fi
TRANSFORMERS_DIR=${SCRIPT_DIR}/tools/transformers

if [[ ! -z ${SQUAD_V2} ]]; then
  python ${TRANSFORMERS_DIR}/examples/run_squad.py \
         --model_type bert \
         --model_name_or_path ${TRAIN_FROM_MODEL} \
         --do_train \
         --train_file ${TRAIN_FILE}\
         --save_steps 10000 \
         --predict_file "" \
         --version_2_with_negative \
         --overwrite_output_dir \
         --overwrite_cache \
         --output_dir ${TRAINING_DIR}
else
  python ${TRANSFORMERS_DIR}/examples/run_squad.py \
         --model_type bert \
         --model_name_or_path ${TRAIN_FROM_MODEL} \
         --do_train \
         --train_file ${TRAIN_FILE}\
         --save_steps 10000 \
         --predict_file "" \
         --overwrite_output_dir \
         --overwrite_cache \
         --output_dir ${TRAINING_DIR}
fi
