#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/../../env
source ${ENV_DIR}/bin/activate

export LC_ALL=en_US.UTF-8
INPUT_SRC=$1
OUTPUT_FILE=$2

#Preprocess functions
PREPROCESS_DIR=${SCRIPT_DIR}/data/en2es/preprocess
JOINT_BPE=${PREPROCESS_DIR}/joint_bpe
VOCAB_EN=${PREPROCESS_DIR}/vocab.en
TRUECASE_EN=${PREPROCESS_DIR}/truecase-model.en

TOOLS_DIR=${SCRIPT_DIR}/../../tools
MOSES_DIR=${TOOLS_DIR}/mosesdecoder
ONMT_DIR=${TOOLS_DIR}/OpenNMT-py


preprocess_src() {
  INPUT_FILE=$1
  LANG=$2

  cat ${INPUT_FILE} |
  perl ${MOSES_DIR}/scripts/tokenizer/normalize-punctuation.perl -l $LANG |
  perl ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -l ${LANG} -no-escape |
  perl ${MOSES_DIR}/scripts/recaser/truecase.perl --model ${TRUECASE_EN} |
  subword-nmt apply-bpe -c ${JOINT_BPE} --vocabulary ${VOCAB_EN} --vocabulary-threshold 50
}

postprocess_pred() {
  INPUT_FILE=$1
  LANG=$2
  cat ${INPUT_FILE} |
  sed -r 's/(@@ )|(@@ ?$)//g' ${INPUT_FILE} |
  perl ${MOSES_DIR}/scripts/recaser/detruecase.perl |
  perl ${MOSES_DIR}/scripts/tokenizer/detokenizer.perl -l $LANG
}

TEST_SRC_BPE=$(mktemp)
preprocess_src ${INPUT_SRC} en > ${TEST_SRC_BPE}

#Translate Transformer
echo "Translate..."
# Select the model
MODEL_CHECKPOINT=${SCRIPT_DIR}/../nmt/data/en2es/train/shared/en2es_average_model.pt
echo "Using average model across checkpoints: ${MODEL_CHECKPOINT}"

PREDS_BPE=$(mktemp)
python ${ONMT_DIR}/translate.py \
       -model ${MODEL_CHECKPOINT} \
       -src ${TEST_SRC_BPE} \
       -output ${PREDS_BPE} \
  	   -verbose -replace_unk \
       -gpu 0

#Postprocess predictions
postprocess_pred $PREDS_BPE es > $OUTPUT_FILE

rm $PREDS_BPE $TEST_SRC_BPE