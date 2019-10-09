#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TENSORBOARD_DIR=$1

ENV_DIR=/home/casimiro/projects/hutoma/nmt_en-es_general/env
source $ENV_DIR/bin/activate

tensorboard --logdir $TENSORBOARD_DIR