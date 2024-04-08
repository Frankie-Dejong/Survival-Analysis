import os
import json
import argparse
import shutil
from tqdm import tqdm


def refactor_TCGA_files(json_name : str, input_path : str, output_path : str):
    
    try:
        with open(json_name, 'r') as fp:
            json_data = json.load(fp)
    except:
        print("Error: JSON file doesn't exist")

    os.makedirs(output_path, exist_ok=True)
    print("Refactoring files...")

    for i, data in enumerate(tqdm(json_data)):
        file_id = data['file_id']
        file_name = data['file_name']
        shutil.copy(os.path.join(input_path, file_id, file_name), output_path)
        os.rename(os.path.join(output_path, file_name), os.path.join(output_path, '{:0>4d}'.format(i) + '.svs'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_name', type=str, default=None)
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    refactor_TCGA_files(args.json_name, args.input_path, args.output_path)
    