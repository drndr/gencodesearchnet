import argparse
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models import CodeT5pEmbForSequenceClassification
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


def predict(model, dataloader, args):
    """Predict on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []
    labels = []
    logging.info('Evaluating...')
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            predictions.extend(outputs.logits.argmax(-1).cpu().numpy().tolist())
            labels.extend(batch['labels'].cpu().numpy().tolist())

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


def main():
    """Main function."""
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=512, help='maximum sequence length')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
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
        dataloader = load_data(tokenizer, args.batch_size, args.seq_len,
                               f'{DS_FOLDER}/clf/{meta_data[0]}/{meta_data[1]}_clf.jsonl')
        metrics = predict(model, dataloader, args)
        results[meta_data[0]] = metrics
        logging.info(f'Test results for {meta_data}: {metrics}')

    logging.info(f'Test results: {results}')


if __name__ == '__main__':
    main()
