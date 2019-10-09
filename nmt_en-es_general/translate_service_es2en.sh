#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ENV_DIR=/home/casimiro/projects/hutoma/nmt_en-es_general/env
source $ENV_DIR/bin/activate

LANG_SRC="es"
LANG_TGT="en"
export IP="10.8.0.22"
export PORT=5200
export URL_ROOT="/translator"
export CONFIG="./translation_service_es2en_conf.json"

# NOTE that these parameters are optional
# here, we explicitly set to default values
ONMT_DIR=/home/casimiro/projects/hutoma/nmt_en-es_general/tools/OpenNMT-py
python $ONMT_DIR/server.py --source_language $LANG_SRC --target_language $LANG_TGT \
                           --ip $IP --port $PORT --url_root $URL_ROOT --config $CONFIG
