#!/bin/bash
#"Test adding wikidata to the contrastive model"
DEVICE_MAP="auto"
MAX_NEW_TOKENS=256
BATCH_SIZE=20
OUTPUT_DIR="assets/models/KQAPRO_LCQUAD_OpenLLAMA"
PROMPT_TYPE="wikidatael"
CHUNK=1000

cd ..
# Measure the time of execution using the `time` command
time python -m text2sparql.evaluate --device_map $DEVICE_MAP \
--max_new_tokens $MAX_NEW_TOKENS \
--batch_size $BATCH_SIZE \
--output_dir $OUTPUT_DIR \
--prompt_type $PROMPT_TYPE \
--chunk $CHUNK
