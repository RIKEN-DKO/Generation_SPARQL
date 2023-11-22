#!/bin/bash
DEVICE_MAP="auto"
MAX_NEW_TOKENS=256
BATCH_SIZE=20
CHUNK=1000
PROMPT_TYPE="plainbgee"

# Directory containing the testing data files
data_dir="data/bgee_augment"
# cd ..
# Loop through each testing data file in the specified directory
for test_file in $data_dir/*_test.json; do
    # Extract the base name of the testing data file (excluding the directory and file extension)
    base_name=$(basename "$test_file" _test.json)
    
    # Construct the output directory name
    output_dir="assets/models/wikidatacon_BGEE_$base_name/checkpoint-2000"
    

    command="python -m text2sparql.evaluate --device_map $DEVICE_MAP --max_new_tokens $MAX_NEW_TOKENS --batch_size $BATCH_SIZE --output_dir $output_dir --chunk $CHUNK --prompt_type $PROMPT_TYPE --dataset $test_file"

    echo "Executing: $command"
    eval $command
done
