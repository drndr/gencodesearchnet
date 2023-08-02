from datasets import load_dataset
import random
import json
import ast


def sample_create_cbt_jsonl_file(dataset, output_file):
    data_with_fields = []
    indices = list(range(len(dataset)))
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for idx,item in enumerate(dataset):
        
            # Create True Pair
            input = item["func_documentation_string"] + " [SEP] " + item["func_code_string"]
            target = 1
            target_options = ["no_match", "match"]
            
            data_with_fields.append({
                "input": input,
                "target": target,
                "target_options": target_options
            })
            
            # Create False Pair
            indices.remove(idx)  # To avoid sampling from the same line
            sampled_idx = random.choice(indices) # Randomly sample code from another line
            input = item["func_documentation_string"] + " [SEP] " + dataset[sampled_idx]["func_code_string"]
            indices.append(idx)  # Add back the removed index for future sampling

            target = 0
            target_options = ["no_match", "match"]

            data_with_fields.append({
                "input": input,
                "target": target,
                "target_options": target_options
        })
        for item in data_with_fields:
                json.dump(item, outfile, ensure_ascii=False)
                outfile.write('\n')

def main():

    
    input_file = load_dataset("code_search_net", "go")["test"]
    output_file = 'go_clf.jsonl'
            
    sample_create_cbt_jsonl_file(input_file, output_file)
    
if __name__ == '__main__':
    main()