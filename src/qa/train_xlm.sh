#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

TRAIN_FILE=$1
EXP_NAME=$2
SQUAD_V2=$3


TRAIN_FROM_MODEL=xlm-mlm-tlm-xnli15-1024
TRANSFORMERS_DIR=${SCRIPT_DIR}/tools/transformers
TRAINING_DIR=${SCRIPT_DIR}/data/training/$(basename ${EXP_NAME})
mkdir -p ${TRAINING_DIR}

if [[ ! -z ${SQUAD_V2} ]]; then
  python ${TRANSFORMERS_DIR}/examples/run_squad.py \
         --model_type xlm \
         --model_name_or_path ${TRAIN_FROM_MODEL} \
         --do_train \
         --do_lower_case \
         --train_file ${TRAIN_FILE}\
         --save_steps 10000 \
         --predict_file "" \
         --version_2_with_negative \
         --overwrite_output_dir \
         --overwrite_cache \
         --output_dir ${TRAINING_DIR}
else
  python ${TRANSFORMERS_DIR}/examples/run_squad.py \
         --model_type xlm \
         --model_name_or_path ${TRAIN_FROM_MODEL} \
         --do_train \
         --do_lower_case \
         --train_file ${TRAIN_FILE}\
         --save_steps 10000 \
         --predict_file "" \
         --overwrite_output_dir \
         --overwrite_cache \
         --output_dir ${TRAINING_DIR}
fi
