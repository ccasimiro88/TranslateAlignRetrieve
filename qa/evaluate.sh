#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/../env
source ${ENV_DIR}/bin/activate

TEST_FILE=$1

TRANSFORMERS_DIR=${SCRIPT_DIR}/../tools/transformers
EVALUATE_DIR=${SCRIPT_DIR}/data/evaluate/$(basename ${TRAIN_FILE})
mkdir -p ${EVALUATE_DIR}

python ${TRANSFORMERS_DIR}/examples/run_squad.py \
       --model_type bert \
       --model_name_or_path bert-base-multilingual-cased \
       --do_eval \
       --do_lower_case \
       --predict_file ${TEST_FILE} \
       --save_steps 10000 \
       --logging_steps 10000 \
       --version_2_with_negative \
       --output_dir ${EVALUATE_DIR}
