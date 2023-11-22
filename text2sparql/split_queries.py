import json
import os
import random
import argparse


def collect_ids(input_dir):
    all_ids = set()
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            input_file = os.path.join(input_dir, file_name)
            with open(input_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    all_ids.add(item['id'])
    return list(all_ids)


def split_data(input_file, train_ids, test_ids, train_file, test_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    train_data = [item for item in data if item['id'] in train_ids]
    test_data = [item for item in data if item['id'] in test_ids]

    with open(train_file, 'w', encoding='utf-8') as file:
        json.dump(train_data, file, indent=4)
    with open(test_file, 'w', encoding='utf-8') as file:
        json.dump(test_data, file, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description='Split JSON data into training and testing sets.')
    parser.add_argument(
        'input_dir', help='The directory containing the input JSON files.')
    parser.add_argument(
        'output_dir', help='The directory to save the output train and test JSON files.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_ids = collect_ids(args.input_dir)
    random.shuffle(all_ids)
    split_index = int(0.8 * len(all_ids))
    train_ids, test_ids = set(
        all_ids[:split_index]), set(all_ids[split_index:])

    for file_name in os.listdir(args.input_dir):
        if file_name.endswith('.json'):
            input_file = os.path.join(args.input_dir, file_name)
            base_name, _ = os.path.splitext(file_name)
            train_file = os.path.join(
                args.output_dir, f'{base_name}_train.json')
            test_file = os.path.join(args.output_dir, f'{base_name}_test.json')
            split_data(input_file, train_ids, test_ids, train_file, test_file)


if __name__ == '__main__':
    main()
