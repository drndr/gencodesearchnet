import json
import logging

import torch
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return len(self.features)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _convert_examples_to_features(comments, codes, labels, max_seq_length,
                                  tokenizer,
                                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                  eos_token='</s>',
                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                                  cls_token_segment_id=1, pad_token_segment_id=0,
                                  mask_padding_with_zero=True):
    features = []
    for ex_index, (comment, code, label) in enumerate(zip(comments, codes, labels)):
        # As was done in CodeBERT
        tokens_comment = tokenizer.tokenize(comment)[:50]
        tokens_code = tokenizer.tokenize(code)

        # update max_seq_length to account for [CLS], [SEP], [SEP] tokens (-3)
        n_special_tokens = 3
        if cls_token is None:
            n_special_tokens -= 1
        s_max_seq_length = max_seq_length - n_special_tokens
        _truncate_seq_pair(tokens_comment, tokens_code, s_max_seq_length)

        # change sep for eos if no sep_token
        if sep_token is None:
            sep_token = eos_token

        # [SEP] inbetween and at the end
        tokens = tokens_comment + [sep_token] + tokens_code + [sep_token]
        # CLS at the beginning
        if cls_token is not None:
            tokens = [cls_token] + tokens

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 1 for tokens, 0 for padding
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # padding with 0 up to max_seq_length
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        # check
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        # convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        features.append({
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": label
        })
    return features


def load_data(tokenizer, batch_size, seq_len, train_file):
    # create dataset
    comments = []
    codes = []
    labels = []
    skipped = 0

    is_sep_token_set = tokenizer.sep_token is not None
    is_cls_token_set = tokenizer.cls_token is not None
    is_pad_token_set = tokenizer.pad_token is not None
    is_eos_token_set = tokenizer.eos_token is not None

    with open(train_file, 'r', encoding='utf-8') as infile:
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
    # tokenize
    features = _convert_examples_to_features(
        comments,
        codes,
        labels,
        max_seq_length=seq_len,
        tokenizer=tokenizer,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=tokenizer.cls_token_id,
        pad_token_segment_id=tokenizer.pad_token_id,
        eos_token=tokenizer.eos_token,
    )

    # Convert to Dataset
    features = Dataset(features)

    return DataLoader(features, batch_size=batch_size, shuffle=True)
