#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_CHECKPOINT=$1

ENV_DIR=/home/casimiro/projects/hutoma/nmt_en-es_general/env
source $ENV_DIR/bin/activate

PREPROCESS_DIR=$SCRIPT_DIR/data/preprocess
mkdir -p $SCRIPT_DIR/data/train
TRAIN_DIR=$SCRIPT_DIR/data/train

mkdir -p $TRAIN_DIR/tensorboard/train_from_50k
TENSORBOARD_LOG_DIR=$TRAIN_DIR/tensorboard/train_from_50k

ONMT_DIR=/home/casimiro/projects/hutoma/nmt_en-es_general/tools/OpenNMT-py
if [ ! -d $ONMT_DIR ]
  then
    echo 'Download OpenNMT-py and install it...'
    git clone https://github.com/OpenNMT/OpenNMT-py.git $ONMT_DIR
    pip install $ONMT_DIR
    pip install -r $ONMT_DIR/requirements.txt
fi

LANG_SRC=en
LANG_TGT=es
TRAIN_SRC=$PREPROCESS_DIR/train.bpe.$LANG_SRC
TRAIN_TGT=$PREPROCESS_DIR/train.bpe.$LANG_TGT
VALID_SRC=$PREPROCESS_DIR/valid.bpe.$LANG_SRC
VALID_TGT=$PREPROCESS_DIR/valid.bpe.$LANG_TGT

echo "Prepare data for training..."
# Prepare parallel data with shared vocabulary
python $ONMT_DIR/preprocess.py -train_src $TRAIN_SRC -train_tgt $TRAIN_TGT \
	   -valid_src $VALID_SRC -valid_tgt $VALID_TGT \
	   -share_vocab \
	   -save_data $TRAIN_DIR/en2es_transformer_shared_vocab_embs \
	   -log_file $TRAIN_DIR/log_train

#Train Transformer with shared vocab and embeddings
echo "Train transformer with shared vocab and embeddings..."
python $ONMT_DIR/train.py -data $TRAIN_DIR/en2es_transformer_shared_vocab_embs \
       -save_model $TRAIN_DIR/en2es_transformer_shared_vocab_embs \
       -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
       -encoder_type transformer -decoder_type transformer -position_encoding \
       -train_steps 100000  -max_generator_batches 2 -dropout 0.1 \
       -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 4 \
       -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
       -max_grad_norm 0 -param_init 0  -param_init_glorot \
       -label_smoothing 0.1 -valid_steps 1000 --valid_batch_size 16 \
       -report_every 100 -save_checkpoint_steps 1000 \
       -world_size 2 -gpu_ranks 0 1 \
       --tensorboard  --tensorboard_log_dir $TENSORBOARD_LOG_DIR \
       --log_file $TRAIN_DIR/log_train --train_from $MODEL_CHECKPOINT


