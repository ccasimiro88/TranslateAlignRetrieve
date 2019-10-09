#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=/home/casimiro/projects/hutoma/nmt_en-es_general/env
source $ENV_DIR/bin/activate

export LC_ALL=en_US.UTF-8
INPUT_SRC=$1
LANG_SRC=$2
LANG_TGT=$3
MODEL_CHECKPOINT=$4

TRANSLATE_DIR=$SCRIPT_DIR/data/en2es/translate
mkdir -p $TRANSLATE_DIR

#Preprocess functions
PREPROCESS_DIR=$SCRIPT_DIR/data/en2es/preprocess
MOSES_DIR=/home/casimiro/projects/hutoma/nmt_en-es_general/tools/mosesdecoder
SUBWORDNMT_DIR=/home/casimiro/projects/hutoma/nmt_en-es_general/tools/subword-nmt

preprocess_src() {
  INPUT_FILE=$1
  LANG=$2

  cat $INPUT_FILE |
  perl $MOSES_DIR/scripts/tokenizer/normalize-punctuation.perl -l $LANG |
  perl $MOSES_DIR/scripts/tokenizer/tokenizer.perl -l $LANG -no-escape |
  perl $MOSES_DIR/scripts/recaser/truecase.perl --model $PREPROCESS_DIR/truecase-model.en |
  python $SUBWORDNMT_DIR/subword_nmt/apply_bpe.py -c $PREPROCESS_DIR/joint_bpe --vocabulary $PREPROCESS_DIR/vocab.en --vocabulary-threshold 50
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
preprocess_src $INPUT_SRC $LANG_SRC > $TEST_SRC_BPE

#Translate Transformer
echo "Translate..."
# Select the model
if [[ -z "$MODEL_CHECKPOINT" ]]; then
    # If model not provided use best model checkpoint
    echo "Using best model checkpoint:"
    if [[ "$LANG_SRC" == "en" ]]
      then
        MODEL_CHECKPOINT=$SCRIPT_DIR/data/en2es/train/en2es_transformer_shared_vocab_embs_best_in_40-60k.pt
        echo $MODEL_CHECKPOINT
    elif [[ "$LANG_SRC" == "es" ]]
      then
        MODEL_CHECKPOINT=$SCRIPT_DIR/data/es2en/train/es2en_transformer_shared_vocab_embs_best_60-80k.pt
        echo $MODEL_CHECKPOINT
    fi
else
  echo "Using model: $MODEL_CHECKPOINT"
fi

PREDS_BPE=$(mktemp)
ONMT_DIR=/home/casimiro/projects/hutoma/nmt_en-es_general/tools/OpenNMT-py
python $ONMT_DIR/translate.py \
       -model $MODEL_CHECKPOINT \
       -src $TEST_SRC_BPE \
       -output $PREDS_BPE \
  	   -verbose -replace_unk \
       -gpu 0 \

#Postprocess predictions
postprocess_pred $PREDS_BPE $LANG_TGT > \
                 $TRANSLATE_DIR/$(basename $INPUT_SRC).$LANG_TGT

rm $PREDS_BPE $TEST_SRC_BPE
