#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

INPUT_SRC=$1

HOST="10.8.0.22"
PORT=5200
URL_ROOT="/translator"

# Send request to the translation service
# The sequence of quotes '" before and after a shell variable
# are necessary to read the variable (including whitespaces)
curl -i -X POST -H "Content-Type: application/json" \
     -d '[{"src": "'"$INPUT_SRC"'", "id": 200}]' \
     http://$HOST:$PORT$URL_ROOT/translate
