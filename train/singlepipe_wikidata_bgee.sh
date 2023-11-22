#!/bin/bash

# Directory containing the training data files
data_dir="data/bgee_augment"

# cd ..
# Loop through each training data file in the specified directory
for train_file in $data_dir/*_train.json; do
    # Extract the base name of the training data file (excluding the directory and file extension)
    base_name=$(basename "$train_file" _train.json)
    
    # Construct the output directory name
    output_dir="assets/models/wikidata_BGEE_$base_name"
    
    # Construct and execute the command to train the model
    command="python -m text2sparql.peft_finetune_wiki_bgee --output_dir $output_dir --bgee_data $train_file"
    echo "Executing: $command"
    eval $command
done
