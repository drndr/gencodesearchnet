import random
import argparse
import json


def select_random_lines(input_file, output_file, num_lines=100):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    random_objects = random.sample(data, min(num_lines, len(data)))

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for obj in random_objects:
            json.dump(obj, outfile, ensure_ascii=False)
            outfile.write('\n')
            

def create_cbt_jsonl_file(input_file, output_file):
    data_with_fields = []
    data = []

    # Load data from the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error: JSON decoding failed - {e}")
                continue
            data.append(item)

    # Create a list of indices to sample from

    for idx, item in enumerate(data):
        
        # Create True Pair
        input = item["doc"] + " [SEP] " + item["code"]
        target = 1
        target_options = [0, 1]
        
        data_with_fields.append({
            "input": input,
            "target": target,
            "target_options": target_options
        })
        
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in data_with_fields:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

def main():

    
    num_lines_to_select = 100
    
    input_file = 'test_webquery.json'
    output_file = 'test_sample_raw.jsonl'
    cbt_format_file = "test_sample_cbt.jsonl"
            
    select_random_lines(input_file, output_file, num_lines_to_select)
    create_cbt_jsonl_file(output_file, cbt_format_file)

if __name__ == '__main__':
    main()