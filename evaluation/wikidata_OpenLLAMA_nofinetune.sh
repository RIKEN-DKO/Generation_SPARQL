#!/bin/bash

DEVICE_MAP="auto"
MAX_NEW_TOKENS=256
BATCH_SIZE=20
OUTPUT_DIR="assets/models/wikidata_OpenLlama_nofinetune"
CHUNK=1000
PROMPT_TYPE="plain"
DATA="data/kqapro_lcquad_test.json"

# Measure the time of execution using the `time` command
time python -m text2sparql.evaluate \
--device_map $DEVICE_MAP \
--max_new_tokens $MAX_NEW_TOKENS \
--batch_size $BATCH_SIZE \
--output_dir $OUTPUT_DIR \
--chunk $CHUNK \
--prompt_type $PROMPT_TYPE \
--dataset $DATA --model_id openlm-research/open_llama_7b_v2 
