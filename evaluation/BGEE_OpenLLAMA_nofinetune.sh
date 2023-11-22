#!/bin/bash
#"Test adding wikidata to the contrastive model"
DEVICE_MAP="auto"
MAX_NEW_TOKENS=1024
BATCH_SIZE=10
OUTPUT_DIR="assets/models/BGEE_OpenLlama_nofinetune"

PROMPT_TYPE="plainbgee"
CHUNK=1000

cd ..
# Measure the time of execution using the `time` command
time python -m text2sparql.evaluate --device_map $DEVICE_MAP \
--max_new_tokens $MAX_NEW_TOKENS \
--batch_size $BATCH_SIZE \
--output_dir $OUTPUT_DIR \
--prompt_type $PROMPT_TYPE \
--chunk $CHUNK --model_id openlm-research/open_llama_7b_v2 


