import json
from random import shuffle

# Load the data from the file
def load_data_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to save a partition to a file
def save_partition_to_file(partition, file_name):
    with open(file_name, 'w') as file:
        json.dump(partition, file, indent=4)

# Function to create overlapping partitions
def create_partitions(data, percentages):
    partitions = {}
    total_length = len(data)

    # Shuffle the data for randomness
    shuffle(data)

    for percentage in percentages:
        size = int(total_length * (percentage / 100))
        partitions[percentage] = data[:size]

    return partitions

# Load the data
file_path = 'data/bgee_percent_meaningful_vars_comments/meaningful_vars_comments_augmented_train.json'  # Replace with your file path
data = load_data_from_file(file_path)

# Define the desired percentages
percentages = [20, 40, 60, 80]

# Create the partitions
data_partitions = create_partitions(data, percentages)

# Save each partition to a separate JSON file
for percentage in percentages:
    file_name = f'partition_{percentage}.json'
    save_partition_to_file(data_partitions[percentage], file_name)
