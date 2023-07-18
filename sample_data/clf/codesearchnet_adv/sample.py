import random
import argparse
import json


def select_random_lines_train(input_file, output_file, num_lines=10):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    random_lines = random.sample(lines, min(num_lines, len(lines)))

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(random_lines)

def process_jsonl_file_train(input_file, output_file):
    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error: JSON decoding failed - {e}")
                continue

            processed_item = {
                "code": ' '.join(item.get("code_tokens", "")),
                "docstring": ' '.join(item.get("docstring_tokens", "")),
                "idx": item.get("idx", None)
            }
            processed_data.append(processed_item)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in processed_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

def create_cbt_jsonl_file_train(input_file, output_file):
    data_with_fields = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error: JSON decoding failed - {e}")
                continue

            input_text = item["docstring"] + " [SEP] " + item["code"]
            target = 1
            target_options = [0, 1]

            data_with_fields.append({
                "input": input_text,
                "target": target,
                "target_options": target_options
            })

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in data_with_fields:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')
            
def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('split', default='train', type=str, help="Valid choices: train, test")
    
    args = parser.parse_args()
    
    num_lines_to_select = 10
    
    if args.split =="train":
        input_file = 'train.jsonl'
        raw_output_file = 'train_sample_raw.jsonl'
        processed_output_file = 'train_sample_processed.jsonl'
        cbt_format_file = "train_sample_cbt.jsonl"
        
    
        select_random_lines_train(input_file, raw_output_file, num_lines_to_select)
        process_jsonl_file_train(raw_output_file, processed_output_file)
        create_cbt_jsonl_file_train(processed_output_file, cbt_format_file)
        
        
    elif args.split =="test":
        input_file = 'test.jsonl'
        raw_output_file = 'test_sample_raw.jsonl'
        processed_output_file = 'train_sample_processed.jsonl'
        
    else:
        pass

if __name__ == '__main__':
    main()