import json

def count_samples_in_jsonl(file_path):
    count = 0
    doc_length = 0
    code_length = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_dict = json.loads(line)
            count += 1
            split_parts = json_dict["input"].split("[CODESPLIT]")
            doc_length += len(split_parts[0])
            code_length += len(split_parts[1])
            
            
    return count, doc_length/count, code_length/count

# Example usage
jsonl_file_path = './codesearchnet_adv/test_adv_mrr.jsonl'
sample_count, avg_doc_length, avg_code_length = count_samples_in_jsonl(jsonl_file_path)
print("Number of samples: ", sample_count, " AVG doc lenght: ", avg_doc_length, " AVG code_length: ", avg_code_length)