from datasets import load_dataset
import random
import json
import ast


def sample_create_cbt_jsonl_file(dataset, output_file, num_samples=10):
    sampled_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    sampled_items = [dataset[idx] for idx in sampled_indices]
    data_with_fields = []
    sampled_indices = list(range(len(sampled_indices)))
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for idx,item in enumerate(sampled_items):
        
            # Create True Pair
            input = item["func_documentation_string"] + " [SEP] " + item["func_code_string"]
            target = 1
            target_options = [0, 1]
            
            data_with_fields.append({
                "input": input,
                "target": target,
                "target_options": target_options
            })
            
            # Create False Pair
            sampled_indices.remove(idx)  # To avoid sampling from the same line
            sampled_idx = random.choice(sampled_indices) # Randomly sample code from another line
            input = item["func_documentation_string"] + " [SEP] " + sampled_items[sampled_idx]["func_code_string"]
            sampled_indices.append(idx)  # Add back the removed index for future sampling

            target = 0
            target_options = [0, 1]

            data_with_fields.append({
                "input": input,
                "target": target,
                "target_options": target_options
        })
        for item in data_with_fields:
                json.dump(item, outfile, ensure_ascii=False)
                outfile.write('\n')

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in data_with_fields:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

def main():

    
    num_lines_to_select = 10
    
    input_file = load_dataset("code_search_net", "ruby")["test"]
    output_file = 'test_sample_cbt.json'
    cbt_format_file = "test_sample_cbt.jsonl"
            
    sample_create_cbt_jsonl_file(input_file, output_file,10)
    
if __name__ == '__main__':
    main()