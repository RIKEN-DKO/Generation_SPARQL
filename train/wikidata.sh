#!/bin/bash

# cd ..
# Define variables for command arguments
OUTPUT_DIR1="assets/models/wikidata"
OUTPUT_DIR2="assets/models/wikidata_con"
TRAIN_DATA="data/kqapro_lcquad_train.json"
MAX_STEPS=13500

# Run the first command
python -m text2sparql.peft_finetune \
  --output_dir $OUTPUT_DIR1 \
  --train_data $TRAIN_DATA \
  --max_steps $MAX_STEPS

# Run the second command
python -m text2sparql.peft_finetune_contrastive \
  --output_dir $OUTPUT_DIR2 \
  --train_data $TRAIN_DATA \
  --max_steps $MAX_STEPS
