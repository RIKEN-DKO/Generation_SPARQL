#!/bin/bash

# Fixed file for training
train_file="data/bgee_original_with_coments/original_with_comments_train.json"

# Array of data percentages
data_percentages=(25 50 75 100)

# Loop through each data percentage
for percentage in "${data_percentages[@]}"; do
    # Construct the output directory name
    output_dir="assets/models/BGEE_${percentage}_percent"

    # Construct and execute the command to train the model with the specified data percentage
    command="python -m text2sparql.peft_finetune --output_dir $output_dir --train_data $train_file --data_percentage $percentage --max_steps 2000"
    echo "Executing: $command"
    eval $command
done
