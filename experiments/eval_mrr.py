import argparse
import json
import logging
import random
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models import CodeT5pEmbForSequenceClassification
from utils import _convert_examples_to_features

DS_FOLDER = '../data'

# [folder, file]
FILES = [
    ['statcodesearch', 'test_statcodesearch'],
    ['codesearchnet_adv', 'test_adv'],
    ['codesearchnet_go', 'test_go'],
    ['codesearchnet_java', 'test_java'],
    ['codesearchnet_javascript', 'test_javascript'],
    ['codesearchnet_php', 'test_php'],
    ['codesearchnet_ruby', 'test_ruby'],
    ['cosqa', 'test_cosqa']
]


def load_data(tokenizer, file):
    # create dataset
    comments = []
    codes = []
    labels = []
    skipped = 0

    is_sep_token_set = tokenizer.sep_token is not None
    is_cls_token_set = tokenizer.cls_token is not None
    is_pad_token_set = tokenizer.pad_token is not None
    is_eos_token_set = tokenizer.eos_token is not None

    with open(file, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                item = json.loads(line.strip())
                input = item['input']
                # split at [CODESPLIT] token
                input = input.split('[CODESPLIT]')
                if len(input) != 2:
                    # skip cases with more than one [SEP] token
                    logging.warning(f"Input contains more than one [CODESPLIT] token: {input}")
                    skipped += 1
                    continue
                # skip every sample that contains special tokens
                if is_sep_token_set and (
                        tokenizer.sep_token in input[0] or tokenizer.sep_token in input[1]):
                    logging.warning(f"Input contains special tokens: {input}")
                    skipped += 1
                    continue
                if is_cls_token_set and (
                        tokenizer.cls_token in input[0] or tokenizer.cls_token in input[1]):
                    logging.warning(f"Input contains special tokens: {input}")
                    skipped += 1
                    continue
                if is_pad_token_set and (
                        tokenizer.pad_token in input[0] or tokenizer.pad_token in input[1]):
                    logging.warning(f"Input contains special tokens: {input}")
                    skipped += 1
                    continue
                if is_eos_token_set and (
                        tokenizer.eos_token in input[0] or tokenizer.eos_token in input[1]):
                    logging.warning(f"Input contains special tokens: {input}")
                    skipped += 1
                    continue
                comments.append(input[0])
                codes.append(input[1])
                labels.append(item['target'])
            except json.JSONDecodeError as e:
                print(f"Error: JSON decoding failed - {e}")
                continue
    logging.info(f"Skipped {skipped} samples due to special tokens")

    return comments, codes




def mrr(model, tokenizer, file, args):
    random.seed(42)

    # load data
    comments, codes = load_data(tokenizer, file)

    # create mrr chunks with (default 99) distractors

    chunks = []
    for i, sample in enumerate(zip(comments, codes)):
        comment, code = sample
        codes_without_sample = codes[:i] + codes[i + 1:]
        # select 99 random codes
        distractors = random.sample(codes_without_sample, args.distractors)
        # create samples
        codes = [code] + distractors
        comments = [comment] * len(codes)
        labels = [1] + [0] * len(distractors)
        # convert to features
        features = _convert_examples_to_features(comments, codes, labels,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=args.seq_len,
                                                 cls_token=tokenizer.cls_token,
                                                 sep_token=tokenizer.sep_token,
                                                 cls_token_segment_id=tokenizer.cls_token_id,
                                                 pad_token_segment_id=tokenizer.pad_token_id,
                                                 eos_token=tokenizer.eos_token)

        chunks.append(features)

    # make predictions for all chunks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    model.to(device)
    model.eval()

    ranks = []
    for chunk in tqdm(chunks):
        # calc correct sample (always the first one)
        correct = chunk[0]
        input_ids = correct['input_ids'].unsqueeze(0).to(device)
        attention_mask = correct['attention_mask'].unsqueeze(0).to(device)
        labels = correct['labels'].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            correct_score = logits[0][0].item()

        # calc scores for the rest of the samples
        scores = []
        # add correct score to scores
        scores.append(correct_score)
        # create batches of size args.batch_size
        batch_size = args.batch_size
        for i in range(1, len(chunk), batch_size):
            batch = chunk[i:i + batch_size]
            input_ids = torch.stack([sample['input_ids'] for sample in batch]).to(device)
            attention_mask = torch.stack([sample['attention_mask'] for sample in batch]).to(device)
            labels = torch.stack([sample['labels'] for sample in batch]).to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                scores.extend(logits[:, 1].cpu().numpy().tolist())

        rank = np.sum(np.array(scores) >= correct_score)
        ranks.append(rank)

    mean_mrr = np.mean(1.0 / np.array(ranks))

    return mean_mrr


def main():
    """Main function."""
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/RoBERTa')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=512, help='maximum sequence length')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument('--distractors', type=int, default=99, help='number of distractors per true pair')
    parser.add_argument('--use_T5', action='store_true', help='use T5 model')

    args = parser.parse_args()

    # logging
    logging.basicConfig(level=args.log_level)

    # load model from file
    logging.info('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.use_T5:
        model = CodeT5pEmbForSequenceClassification.from_pretrained(args.model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model)

    # load data
    results = {}
    for meta_data in FILES:
        logging.info(f'Evaluating on {meta_data}...')
        metrics = mrr(model, tokenizer, f'{DS_FOLDER}/mrr/{meta_data[0]}/{meta_data[1]}_mrr.jsonl', args)
        results[meta_data[0]] = metrics
        logging.info(f'Test results for {meta_data}: {metrics}')

    logging.info(f'Test results: {results}')


if __name__ == '__main__':
    main()
