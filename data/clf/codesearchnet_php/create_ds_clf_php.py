from datasets import load_dataset
import random
import json
import ast


def create_cbt_jsonl_file(dataset, output_file):
    data_with_fields = []
    indices = list(range(len(dataset)))
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for idx,item in enumerate(dataset):
        
             # Create True Pair
            input = " ".join(item["func_documentation_tokens"]) + " [CODESPLIT] " + " ".join(item["func_code_tokens"])
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
            input = " ".join(item["func_documentation_tokens"]) + " [CODESPLIT] " + " ".join(dataset[sampled_idx]["func_code_string"])
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

    random.seed(42)
    input_file = load_dataset("code_search_net", "php")["test"]
    output_file = 'test_php_clf.jsonl'
            
    create_cbt_jsonl_file(input_file, output_file)
    
if __name__ == '__main__':
    main()