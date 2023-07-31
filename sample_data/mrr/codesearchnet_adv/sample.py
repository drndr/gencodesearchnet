import random
import argparse
import json


def select_random_lines(input_file, output_file, num_lines=100):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    random_lines = random.sample(lines, min(num_lines, len(lines)))

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(random_lines)

def process_jsonl_file(input_file, output_file):
    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error: JSON decoding failed - {e}")
                continue
            
            if "train" in input_file:
                processed_item = {
                    "code": ' '.join(item.get("code_tokens", "")),
                    "docstring": ' '.join(item.get("docstring_tokens", "")),
                    "idx": item.get("idx", None)
                }
                
            else:
                processed_item = {
                    "code": ' '.join(item.get("function_tokens", "")),
                    "docstring": ' '.join(item.get("docstring_tokens", "")),
                    "idx": item.get("idx", None)
                }
            processed_data.append(processed_item)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in processed_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

def create_cbt_jsonl_file(input_file, output_file, split):
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
    sample_indices = list(range(len(data)))

    for idx, item in enumerate(data):
        
        # Create True Pair
        input = item["docstring"] + " [SEP] " + item["code"]
        target = 1
        target_options = [0, 1]
        
        data_with_fields.append({
            "input": input,
            "target": target,
            "target_options": target_options
        })
        if split=="train":
            # Create False Pair
            sample_indices.remove(idx)  # To avoid sampling from the same line
            sampled_idx = random.choice(sample_indices) # Randomly sample code from another line
            input = item["docstring"] + " [SEP] " + data[sampled_idx]["code"]
            sample_indices.append(idx)  # Add back the removed index for future sampling

            target = 0
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

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('split', default='train', type=str, help="Valid choices: train, test")
    
    args = parser.parse_args()
    
    num_lines_to_select = 100
    
    if args.split =="train":
        input_file = 'train.jsonl'
        raw_output_file = 'train_sample_raw.jsonl'
        processed_output_file = 'train_sample_processed.jsonl'
        cbt_format_file = "train_sample_cbt.jsonl"
        
    
        select_random_lines(input_file, raw_output_file, num_lines_to_select)
        process_jsonl_file(raw_output_file, processed_output_file)
        create_cbt_jsonl_file(processed_output_file, cbt_format_file,"train")
        
        
    elif args.split =="test":
        input_file = 'test.jsonl'
        raw_output_file = 'test_sample_raw.jsonl'
        processed_output_file = 'test_sample_processed.jsonl'
        cbt_format_file = "test_sample_cbt.jsonl"
        
        select_random_lines(input_file, raw_output_file, num_lines_to_select)
        process_jsonl_file(raw_output_file, processed_output_file)
        create_cbt_jsonl_file(processed_output_file, cbt_format_file,"test")
        
    else:
        print("Invalid split name, use either train or test as argument")

if __name__ == '__main__':
    main()