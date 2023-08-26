import argparse
import json
import logging
import random

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

    return comments, codes, labels


def predict(model, dataloader, tokenizer, args):
    """Predict on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []
    labels = []
    logging.info('Evaluating...')
    for batch in tqdm(dataloader):
        comments = batch[0]
        codes = batch[1]
        labels += batch[2].tolist()
        # encode
        comments = tokenizer.batch_encode_plus(comments, max_length=args.seq_len, padding=True, truncation=True,
                                               return_tensors='pt')
        codes = tokenizer.batch_encode_plus(codes, max_length=args.seq_len, padding=True, truncation=True,
                                            return_tensors='pt')
        # move to device
        comments = {k: v.to(device) for k, v in comments.items()}
        codes = {k: v.to(device) for k, v in codes.items()}
        with torch.no_grad():
            # get embeddings
            comments_emb = model(**comments)
            codes_emb = model(**codes)
            # calc pairwise cosine similarity
            sim = torch.cosine_similarity(comments_emb, codes_emb)
            # get predictions 1 if sim > 0.5 else 0
            pred = torch.where(sim > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            predictions.extend(pred.tolist())

    metrics = {}
    # calc metrics

    # calc accuracy
    accuracy = accuracy_score(labels, predictions)
    metrics['accuracy'] = accuracy

    # calc precision
    precision = precision_score(labels, predictions)
    metrics['precision'] = precision

    # calc recall
    recall = recall_score(labels, predictions)
    metrics['recall'] = recall

    # calc f1
    f1 = f1_score(labels, predictions)
    metrics['f1'] = f1

    return metrics


class CodeSearchDataset(torch.utils.data.Dataset):
    def __init__(self, comments, codes, labels):
        assert len(comments) == len(codes)
        assert len(comments) == len(labels)
        self.comments = comments
        self.codes = codes
        self.labels = labels

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return self.comments[idx], self.codes[idx], self.labels[idx]


def main():
    """Main function."""
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=512, help='maximum sequence length')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

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
        comments, code, labels = load_data(tokenizer, f'{DS_FOLDER}/clf/{meta_data[0]}/{meta_data[1]}_clf.jsonl')
        # create dataset
        dataset = CodeSearchDataset(comments, code, labels)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        metrics = predict(model, dataloader, tokenizer, args)
        results[meta_data[0]] = metrics
        logging.info(f'Test results for {meta_data}: {metrics}')

    logging.info(f'Test results: {results}')


if __name__ == '__main__':
    main()
