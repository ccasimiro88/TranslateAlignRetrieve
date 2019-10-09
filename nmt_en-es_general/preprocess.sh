    #!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=$SCRIPT_DIR/env/bin
source $ENV_DIR/activate

DATASETS_DIR=$1
LANG_SRC=$2
LANG_TGT=$3

#Create preprocess data dir
SRC_TO_TGT=$LANG_SRC'2'$LANG_TGT
PREPROCESS_DIR=$SCRIPT_DIR/data/$SRC_TO_TGT/preprocess
mkdir -p $PREPROCESS_DIR

#Preprocess vars and functions
TOOLS_DIR=$SCRIPT_DIR/tools
MOSES_DIR=$TOOLS_DIR/mosesdecoder

##Tokenizing (also convert to utf-8 to avoid break the tokenizer)
#echo "Tokenize..."
#for data in train valid test; do
#  for lang in $LANG_SRC $LANG_TGT; do
#     cat $DATASETS_DIR/$data.$lang | \
#       iconv -t utf8 | \
#       perl $MOSES_DIR/scripts/tokenizer/normalize-punctuation.perl -l $lang | \
#       perl $MOSES_DIR/scripts/tokenizer/tokenizer.perl -l $lang -no-escape > \
#       $PREPROCESS_DIR/$data.tok.$lang
#  done
#done

##Clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
#echo "Clean the training data..."
#perl $MOSES_DIR/scripts/training/clean-corpus-n.perl \
#     $PREPROCESS_DIR/train.tok $LANG_SRC $LANG_TGT \
#     $PREPROCESS_DIR/train.tok.clean 1 80
#
##Truecasing training and apply
#echo "Train truecase model..."
##Learn truecase model for source and target
#for lang in $LANG_SRC $LANG_TGT; do
#  perl $MOSES_DIR/scripts/recaser/train-truecaser.perl -corpus $PREPROCESS_DIR/train.tok.clean.$lang \
#                                                       -model $PREPROCESS_DIR/truecase-model.$lang
#done

echo "Apply truecase model..."
# Train
for lang in $LANG_SRC $LANG_TGT; do
    cat $PREPROCESS_DIR/train.tok.clean.$lang | \
    perl $MOSES_DIR/scripts/recaser/truecase.perl --model $PREPROCESS_DIR/truecase-model.$lang > \
    $PREPROCESS_DIR/train.tc.$lang
done

# Valid and test
for data in valid test; do
  for lang in $LANG_SRC $LANG_TGT; do
    cat $PREPROCESS_DIR/$data.tok.$lang | \
    perl $MOSES_DIR/scripts/recaser/truecase.perl --model $PREPROCESS_DIR/truecase-model.$lang > \
    $PREPROCESS_DIR/$data.tc.$lang
  done
done


#BPE
#Learn bpe on both source and target training set
echo "Learn bpe on joint source and target training data..."
subword-nmt learn-joint-bpe-and-vocab --input $PREPROCESS_DIR/train.tc.$LANG_SRC $PREPROCESS_DIR/train.tc.$LANG_TGT \
	                                  -s 50000 -o $PREPROCESS_DIR/joint_bpe \
							          --write-vocabulary $PREPROCESS_DIR/vocab.$LANG_SRC $PREPROCESS_DIR/vocab.$LANG_TGT

#Apply bpe
echo "Apply bpe..."
for data in train valid; do
  for lang in $LANG_SRC $LANG_TGT; do
     subword-nmt apply-bpe -c $PREPROCESS_DIR/joint_bpe --vocabulary $PREPROCESS_DIR/vocab.$lang \
                           --vocabulary-threshold 50 \
                           < $PREPROCESS_DIR/$data.tc.$lang > $PREPROCESS_DIR/$data.bpe.$lang
  done
done
