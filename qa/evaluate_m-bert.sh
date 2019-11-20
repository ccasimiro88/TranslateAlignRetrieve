# This script evaluate a QA model on the MLQA dataset.
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

MODEL_DIR=$1
CONTEXT_LANG=$2
QUESTION_LANG=$3
TEST_SET=$4

TRANSFORMERS_DIR=${SCRIPT_DIR}/tools/transformers
MLQA_DIR=${SCRIPT_DIR}/tools/MLQA

if [[ ${TEST_SET} == "mlqa" ]]; then
    # Select the test file from the MLQA corpus
    EVALUATE_DIR=${SCRIPT_DIR}/data/evaluate/mlqa
    mkdir -p ${EVALUATE_DIR}
    TEST_FILE=${SCRIPT_DIR}/corpora/MLQA_V1/test/test-context-${CONTEXT_LANG}-question-${QUESTION_LANG}.json
elif [[ ${TEST_SET} == "xquad" ]]; then
    # Select the test file from the XSQUAD datasets
    EVALUATE_DIR=${SCRIPT_DIR}/data/evaluate/xsquad
    mkdir -p ${EVALUATE_DIR}
    TEST_FILE=${SCRIPT_DIR}/corpora/XQUAD/xquad.${CONTEXT_LANG}.json
else
    EVALUATE_DIR=${SCRIPT_DIR}
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
         --n_best_size 5 \
         --output_dir ${MODEL_DIR}

# Evaluate the predictions with the MLQA original evaluation script
PREDICTION_FILE=${MODEL_DIR}/predictions_.json
EVALUATION_FILE=${EVALUATE_DIR}/$(basename ${MODEL_DIR})_eval
python ${MLQA_DIR}/mlqa_evaluation_v1.py \
       ${TEST_FILE} \
       ${PREDICTION_FILE} \
       ${CONTEXT_LANG} \
       >> ${EVALUATION_FILE}