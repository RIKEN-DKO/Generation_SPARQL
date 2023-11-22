#!/bin/bash

# Directory containing the training data files
data_dir="data/bgee_augment"
model_dir="assets/models/wikidata/checkpoint-13500"
# cd ..
# Loop through each training data file in the specified directory
for train_file in $data_dir/*_train.json; do
    # Extract the base name of the training data file (excluding the directory and file extension)
    base_name=$(basename "$train_file" _train.json)
    
    # Construct the output directory name
    output_dir="assets/models/wikidata_BGEE_$base_name"
    
    # Construct and execute the command to train the model
    command="python -m text2sparql.peft_finetune_continue --output_dir $output_dir --train_data $train_file --max_steps 2000 --model_id $model_dir"
    echo "Executing: $command"
    eval $command
done
