from datasets import load_dataset
import random
import json
import ast
import csv


def csv_to_jsonl(input_file, output_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        comments = [row[2] for row in reader]
        csvfile.seek(0)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                # Positive Sampling
                target = 1
                target_options = ["no_match", "match"]
                item = {
                    "input": row[1] + " [CODESPLIT] " + row[2],
                    "target": target,
                    "target_options": target_options
                }
                data.append(item)
                
                # Negative Sampling
                comments_for_sampling = [comment for comment in comments if comment != row[2]]
                sampled_comment = random.choice(comments_for_sampling) if comments_for_sampling else row[2]
                
                target = 0
                target_options = ["no_match", "match"]
                item = {
                    "input": row[1] + " [CODESPLIT] " + sampled_comment,
                    "target": target,
                    "target_options": target_options
                }
                
                data.append(item)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')
            
def main():

    random.seed(42)
    input_file = 'R-code-comment-after-filtering.csv'
    output_file = 'test_statcodesearch_clf.jsonl'
            
    csv_to_jsonl(input_file, output_file)
    
if __name__ == '__main__':
    main()