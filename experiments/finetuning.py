import argparse
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import get_scheduler, AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, T5ForSequenceClassification
from torch.optim import AdamW

from models import CodeT5pEmbForSequenceClassification, CodeT5pEmbForSequenceClassificationConfig

from utils import load_data

MODELS = {
    'BERT': 'bert-base-uncased',
    'RoBERTa': 'roberta-base',
    'CodeBERT': 'microsoft/codebert-base',
    'T5': 't5-base',
    'T5-code': 'Salesforce/codet5-base',
    'T5-code-p': 'Salesforce/codet5p-220m',
    'T5-code-p-emb': 'Salesforce/codet5p-110m-embedding',
}

T5_MODELS = ['T5', 'T5-code', 'T5-code-p', 'T5-code-p-bi']

TRAIN_FILE = '../data/clf/codesearchnet_adv/train_adv_clf.jsonl'





def train(model: PreTrainedModel, dataloader: DataLoader, args: argparse.Namespace):
    """
    Fine-tune the model.
    :param model: the pretrained model to be fine-tuned
    :param dataloader: an iterable data loader
    :param args: training arguments (and also some other arguments)
    :return: the fine-tuned model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    num_training_steps = args.epochs * len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    for epoch in range(args.epochs):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


def main():
    """Main function."""
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='genbench_ds/data/clf/codesearchnet_adv')
    parser.add_argument('--model', choices=MODELS.keys(), default='RoBERTa')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--seq_len', type=int, default=512, help='maximum sequence length')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    args = parser.parse_args()

    # logging
    logging.basicConfig(level=args.log_level)

    # load model
    logging.info('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])

    if args.model in T5_MODELS:
        model = T5ForSequenceClassification.from_pretrained(MODELS[args.model])
    elif args.model == 'T5-code-p-emb':
        model = CodeT5pEmbForSequenceClassification(
            CodeT5pEmbForSequenceClassificationConfig(
                encoder_name = MODELS[args.model],
                num_labels = 2,
            )
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODELS[args.model])

    # load data
    logging.info('Loading data...')
    dataloader = load_data(tokenizer, args.batch_size, args.seq_len, TRAIN_FILE)

    # train
    logging.info('Training...')
    train(model, dataloader, args)

    # save model
    logging.info('Saving model...')
    model.save_pretrained(f'{args.output_dir}/{args.model}')
    # also soave tokenizer
    tokenizer.save_pretrained(f'{args.output_dir}/{args.model}')


if __name__ == '__main__':
    main()



