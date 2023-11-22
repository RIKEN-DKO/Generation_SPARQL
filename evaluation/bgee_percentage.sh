#!/bin/bash
DEVICE_MAP="auto"
MAX_NEW_TOKENS=1024
BATCH_SIZE=10
CHUNK=1000
PROMPT_TYPE="plainbgee"

# Fixed file for training
test_file="data/bgee_percent_meaningful_vars_comments/meaningful_vars_comments_augmented_test.json"

# Array of data percentages
data_percentages=(20 40 60 80)

# Loop through each data percentage
for percentage in "${data_percentages[@]}"; do
    # Construct the output directory name
    output_dir="assets/models/BGEE_partition_${percentage}/checkpoint-2000"

    # Construct and execute the command to train the model with the specified data percentage
    command="python -m text2sparql.evaluate --device_map $DEVICE_MAP --max_new_tokens $MAX_NEW_TOKENS --batch_size $BATCH_SIZE --output_dir $output_dir --chunk $CHUNK --prompt_type $PROMPT_TYPE --dataset $test_file"

    echo "Executing: $command"
    eval $command
done
