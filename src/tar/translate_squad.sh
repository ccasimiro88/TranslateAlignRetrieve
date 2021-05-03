#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

source_file=$1
target_langs=$2

name="$(basename "$(dirname $source_file)")"

#for lang in ar de el es hi ro ru th tr vi zh; do
for lang in $target_langs; do
  python $SCRIPT_DIR/src/retrieve/translate_squad.py \
  --squad_file $source_file \
  --lang_target $lang --answers_from_alignment \
  --output_dir $SCRIPT_DIR/data/${name}-tar/$lang --batch_size 32 --no_cuda
done
