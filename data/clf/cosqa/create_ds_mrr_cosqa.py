import random
import argparse
import json
            

def create_cbt_jsonl_file(input_file, output_file):
    data_with_fields = []
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

        for line in infile:
            data.append(item)
    
    # Create a list of indices to sample from
    sample_indices = list(range(len(data)))
    for idx, item in enumerate(data):
        
        if item["label"]==0:
            # Create True Pair
            input = item["docstring_tokens"] + " [CODESPLIT] " + item["code_tokens"]
            target = 1
            target_options = ["no_match", "match"]
            
            data_with_fields.append({
                "input": input,
                "target": target,
                "target_options": target_options
            })
            # Create False Pair
            sample_indices.remove(idx)  # To avoid sampling from the same line
            sampled_idx = random.choice(sample_indices) # Randomly sample code from another line
            input = item["docstring"] + " [CODESPLIT] " + data[sampled_idx]["code"]
            sample_indices.append(idx)  # Add back the removed index for future sampling

            target = 0
            target_options = ["no_match", "match"]

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
    
    random.seed(42)
    input_file = 'cosqa-train.json'
    cbt_format_file = "test_cosqa_clf.jsonl"
            
    create_cbt_jsonl_file(input_file, cbt_format_file)

if __name__ == '__main__':
    main()