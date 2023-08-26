import argparse
import json
import logging
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

from utils import load_data

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

def mrr(model, tokenizer, comments, codes, args):
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

        chunks.append(list(zip(comments, codes)))

    # make predictions for all chunks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    model.to(device)
    model.eval()

    ranks = []
    for chunk in tqdm(chunks):
        # calc correct sample (always the first one)
        correct = chunk[0]
        # tokenize
        correct_comment = correct[0]
        correct_code = correct[1]
        correct_comment = tokenizer(correct_comment, max_length=args.seq_len, pad_to_max_length=True,
                                    return_tensors='pt')
        correct_code = tokenizer(correct_code, max_length=args.seq_len, pad_to_max_length=True, return_tensors='pt')
        # move to device
        correct_comment = {k: v.to(device) for k, v in correct_comment.items()}
        correct_code = {k: v.to(device) for k, v in correct_code.items()}
        # get embeddings
        with torch.no_grad():
            correct_comment_emb = model(**correct_comment)
            correct_code_emb = model(**correct_code)
            # calc cosine similarity
            correct_sim = torch.cosine_similarity(correct_comment_emb, correct_code_emb)
            # get correct score
            correct_score = correct_sim[0].item()

        # calc scores for the rest of the samples
        scores = []
        # add correct score to scores
        scores.append(correct_score)
        # create batches of size args.batch_size
        batch_size = args.batch_size
        for i in range(1, len(chunk), batch_size):
            batch = chunk[i:i + batch_size]
            # tokenize
            comments = [sample[0] for sample in batch]
            codes = [sample[1] for sample in batch]
            comments = tokenizer.batch_encode_plus(comments, max_length=args.seq_len, pad_to_max_length=True,
                                                    return_tensors='pt')
            codes = tokenizer.batch_encode_plus(codes, max_length=args.seq_len, pad_to_max_length=True,
                                                return_tensors='pt')
            # move to device
            comments = {k: v.to(device) for k, v in comments.items()}
            codes = {k: v.to(device) for k, v in codes.items()}
            # get embeddings
            with torch.no_grad():
                comments_emb = model(**comments)
                codes_emb = model(**codes)
                # calc cosine similarity
                sim = torch.cosine_similarity(comments_emb, codes_emb)
                # get scores
                scores.extend(sim.tolist())

        rank = np.sum(np.array(scores) >= correct_score)
        ranks.append(rank)

    mean_mrr = np.mean(1.0 / np.array(ranks))

    return mean_mrr


def main():
    """Main function."""
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=512, help='maximum sequence length')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument('--distractors', type=int, default=99, help='number of distractors per true pair')

    args = parser.parse_args()

    # set seed
    random.seed(42)

    # logging
    logging.basicConfig(level=args.log_level)

    checkpoint = "Salesforce/codet5p-110m-embedding"

    # load model from file
    logging.info('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

    # load data
    results = {}
    for meta_data in FILES:
        logging.info(f'Evaluating on {meta_data}...')
        comments, codes = load_data(tokenizer, f'{DS_FOLDER}/mrr/{meta_data[0]}/{meta_data[1]}_mrr.jsonl')
        metrics = mrr(model, tokenizer, comments, codes, args)
        results[meta_data[0]] = metrics
        logging.info(f'Test results for {meta_data}: {metrics}')

    logging.info(f'Test results: {results}')


if __name__ == '__main__':
    main()
