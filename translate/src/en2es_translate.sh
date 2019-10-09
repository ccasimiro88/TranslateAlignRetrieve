#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=$SCRIPT_DIR/../env
source $ENV_DIR/bin/activate

export LC_ALL=en_US.UTF-8
INPUT_SRC=$1
OUTPUT_FILE=$2
BATCH_SIZE=$3

#Preprocess functions
JOINT_BPE=$SCRIPT_DIR/../data/nmt/en2es/joint_bpe
VOCAB_EN=$SCRIPT_DIR/../data/nmt/en2es/vocab.en
TRUECASE_EN=$SCRIPT_DIR/../data/nmt/en2es/truecase-model.en
MOSES_DIR=$SCRIPT_DIR/../tools/mosesdecoder

preprocess_src() {
  INPUT_FILE=$1
  LANG=$2

  cat $INPUT_FILE |
  perl $MOSES_DIR/scripts/tokenizer/normalize-punctuation.perl -l $LANG |
  perl $MOSES_DIR/scripts/tokenizer/tokenizer.perl -l $LANG -no-escape |
  perl $MOSES_DIR/scripts/recaser/truecase.perl --model $TRUECASE_EN |
  subword-nmt apply-bpe -c $JOINT_BPE --vocabulary $VOCAB_EN --vocabulary-threshold 50
}

postprocess_pred() {
  INPUT_FILE=$1
  LANG=$2
  cat $INPUT_FILE |
  sed -r 's/(@@ )|(@@ ?$)//g' $INPUT_FILE |
  perl $MOSES_DIR/scripts/recaser/detruecase.perl |
  perl $MOSES_DIR/scripts/tokenizer/detokenizer.perl -l $LANG
}

TEST_SRC_BPE=$(mktemp)
preprocess_src $INPUT_SRC en > $TEST_SRC_BPE

#Translate Transformer
echo "Translate..."
# Select the model
echo "Using best model checkpoint:"
MODEL_CHECKPOINT=$SCRIPT_DIR/../data/nmt/en2es/en2es_model.pt

PREDS_BPE=$(mktemp)
ONMT_DIR=$SCRIPT_DIR/../tools/OpenNMT-py
python $ONMT_DIR/translate.py \
       -model $MODEL_CHECKPOINT \
       -src $TEST_SRC_BPE \
       -output $PREDS_BPE \
  	   -verbose -replace_unk \
       -gpu 0 \
       -batch_size $BATCH_SIZE

#Postprocess predictions
postprocess_pred $PREDS_BPE es > $OUTPUT_FILE

rm $PREDS_BPE $TEST_SRC_BPE