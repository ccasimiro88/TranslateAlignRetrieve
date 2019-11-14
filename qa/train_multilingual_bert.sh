#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

TRAIN_FILE=$1
SQUAD_V2=$2

TRANSFORMERS_DIR=${SCRIPT_DIR}/tools/transformers
TRAINING_DIR=${SCRIPT_DIR}/data/training/$(basename ${TRAIN_FILE})
mkdir -p ${TRAINING_DIR}

if [[ ! -z ${SQUAD_V2} ]]; then
  python ${TRANSFORMERS_DIR}/examples/run_squad.py \
         --model_type bert \
         --model_name_or_path bert-base-multilingual-cased \
         --do_train \
         --do_lower_case \
         --train_file ${TRAIN_FILE}\
         --save_steps 30000 \
         --predict_file "" \
         --logging_steps 10000 \
         --version_2_with_negative \
         --overwrite_output_dir \
         --output_dir ${TRAINING_DIR}
else
  python ${TRANSFORMERS_DIR}/examples/run_squad.py \
         --model_type bert \
         --model_name_or_path bert-base-multilingual-cased \
         --do_train \
         --do_lower_case \
         --train_file ${TRAIN_FILE}\
         --save_steps 30000 \
         --predict_file "" \
         --logging_steps 10000 \
         --overwrite_output_dir \
         --output_dir ${TRAINING_DIR}
fi
