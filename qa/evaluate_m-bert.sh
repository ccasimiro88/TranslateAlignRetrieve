#!/bin/bash
# This script evaluate a QA model on the MLQA dataset.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

MODEL_DIR=$1
CONTEXT_LANG=$2
QUESTION_LANG=$3
TEST_SET=$4
EVALUATE_DIR=$5
mkdir -p ${EVALUATE_DIR}


TRANSFORMERS_DIR=${SCRIPT_DIR}/tools/transformers
MLQA_DIR=${SCRIPT_DIR}/tools/MLQA
SQUAD_DIR=${SCRIPT_DIR}/tools/squad

if [[ ${TEST_SET} == "mlqa" ]]; then
    # Select the test file from the MLQA corpus
    TEST_FILE=${SCRIPT_DIR}/corpora/MLQA_V1/test/test-context-${CONTEXT_LANG}-question-${QUESTION_LANG}.json
elif [[ ${TEST_SET} == "xquad" ]]; then
    # Select the test file from the XSQUAD datasets
    TEST_FILE=${SCRIPT_DIR}/corpora/XQUAD/xquad.${CONTEXT_LANG}.json
else
    TEST_FILE=${TEST_SET}
fi

# Add the train file path and pass it to the script to avoid errors
TRAIN_FILE=${SCRIPT_DIR}/corpora/squad_es/$(basename ${MODEL_DIR})
python ${TRANSFORMERS_DIR}/examples/run_squad.py \
         --model_type bert \
         --model_name_or_path ${MODEL_DIR} \
         --train_file ${TRAIN_FILE} \
         --do_eval \
         --predict_file ${TEST_FILE} \
         --overwrite_cache \
         --n_best_size 5 \
         --output_dir ${MODEL_DIR}

PREDICTION_FILE=${MODEL_DIR}/predictions_.json
EVALUATION_FILE=${EVALUATE_DIR}/$(basename ${MODEL_DIR})_eval

if [[ ${TEST_SET} == "mlqa" ]]; then
   # Evaluate with the MLQA original evaluation script
   python ${MLQA_DIR}/mlqa_evaluation_v1.py \
          ${TEST_FILE} \
          ${PREDICTION_FILE} \
          ${CONTEXT_LANG} \
          >> ${EVALUATION_FILE}

elif [[ ${TEST_SET} == "xquad" ]]; then
   # Evaluate with the official SQUAD v2.0 script
   python ${SQUAD_DIR}/eval_squad_v2.0.py \
          ${TEST_FILE} \
          ${PREDICTION_FILE} \
          >> ${EVALUATION_FILE}
else
   # Evaluate with the official SQUAD v2.0 script
   python ${SQUAD_DIR}/eval_squad_v2.0.py \
          ${TEST_FILE} \
          ${PREDICTION_FILE} \
          >> ${EVALUATION_FILE}
fi