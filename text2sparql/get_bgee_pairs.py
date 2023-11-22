import argparse
import os
import json


def get_file_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        question = lines[0].strip('#').strip()
        sparql = ''.join(lines[1:]).strip()
        return question, sparql


def main():
    parser = argparse.ArgumentParser(
        description='Create a dataset from a bunch of files.')
    parser.add_argument('input_dir', help='The directory of files.')
    parser.add_argument('output_file', help='The output JSON file.')
    args = parser.parse_args()

    data = []
    for file_name in os.listdir(args.input_dir):
        if file_name.endswith('.rq'):
            file_id = file_name.split('_')[1:]
            file_id = '_'.join(file_id).replace('.rq', '')
            file_path = os.path.join(args.input_dir, file_name)
            question, sparql = get_file_data(file_path)
            data.append({
                "id": file_id,
                "question": question,
                "sparql": sparql
            })

    with open(args.output_file, 'w', encoding='utf-8') as out_file:
        json.dump(data, out_file, indent=4)


if __name__ == "__main__":
    main()
