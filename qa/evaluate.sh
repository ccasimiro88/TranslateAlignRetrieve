# This script evaluate a QA model on the MLQA dataset.
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

TEST_FILE=$1
MODEL_DIR=$2
ANSWER_LANG=$3

TRANSFORMERS_DIR=${SCRIPT_DIR}/tools/transformers
MLQA_DIR=${SCRIPT_DIR}/tools/MLQA
EVALUATE_DIR=${SCRIPT_DIR}/data/evaluate

# Generate predictions
# Add the train file path and pass it to the script to avoid errors
TRAIN_FILE=${SCRIPT_DIR}/corpora/squad_es/$(basename ${MODEL_DIR})
python ${TRANSFORMERS_DIR}/examples/run_squad.py \
         --model_type bert \
         --model_name_or_path ${MODEL_DIR} \
         --train_file ${TRAIN_FILE} \
         --do_eval \
         --do_lower_case \
         --predict_file ${TEST_FILE} \
         --n_best_size 5 \
         --output_dir ${MODEL_DIR}

# Evaluate the predictions with the MLQA original evaluation script
PREDICTION_FILE=${MODEL_DIR}/predictions_.json
python ${MLQA_DIR}/mlqa_evaluation_v1.py \
       ${TEST_FILE} \
       ${PREDICTION_FILE} \
       ${ANSWER_LANG} \
       > ${EVALUATE_DIR}/eval_$(basename ${TEST_FILE})