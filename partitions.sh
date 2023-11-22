#!/bin/bash

echo "Starting batch execution..."
# List of scripts
declare -a scripts=("train/bgee_percentage_subsets.sh" "evaluation/bgee_percentage.sh")

# Execute each script
for script in "${scripts[@]}"
do
  echo "Starting $script..."
  SECONDS=0 # Reset bash timer

  # Execute the script (considering you have cd to the proper directory already)
  ./$script

  # Calculate minutes and seconds
  duration=$SECONDS
  mins=$((duration / 60))
  secs=$((duration % 60))

  echo "$script finished. Execution time: $mins minute(s) and $secs second(s)."
done

echo "Batch execution completed."
