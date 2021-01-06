import json
from spacy.lang.en import English
from functools import partial
import linecache
import torch
import random
from io import StringIO
import pandas as pd


# function for preprocessing sentences for bert training
def preprocess_bert(sentences, tokenizer, config, return_attention_masks=True):
    """
    perform the following steps:
    1. add [CLS] token to the start of every sentence
    2. add [SEP] token to the end of every sentence
    3. add [PAD] token to pad/truncate sentences into a pre-defined fixed length
    4. use the `tokenizer` to tokenize the sentences
    5. use the `tokenizer` to convert the sentences into indices
    6. convert indexed sentences to pytorch tensors

    remember to move the returned tensor to gpu (if necessary)

    :param sentences: a list of sentences that need to be preprocessed
    :param tokenizer: bert tokenizer to be used for the preprocessing
    :return: preprocessed list of sentences
    :return: preprocessed list of sentences, attention masks if `attention_masks` is set to `True`
    """

    # embedding_config = json.load(open("utils/preprocessing_config/embedding_config.json", "r"))
    max_length = config["dataset"]["max_sequence_length"]  # embedding_config["max_length"]
    attention_masks = []

    for i, sentence in enumerate(sentences):
        # modify in-place!
        sentences[i] = "[CLS] " + sentence
        sentences[i] += " [SEP]"

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_sentence) for tokenized_sentence in tokenized_sentences]

    for tokens in indexed_tokens:
        tokens_length = len(tokens)
        if return_attention_masks:
            # when an array is multiplied by a negative number, the result is just `[]`
            attention_masks.append([1] * min(tokens_length, max_length) + [0] * (max_length - tokens_length))
        if tokens_length < max_length:
            tokens.extend([0] * (max_length - tokens_length))
        elif tokens_length > max_length:
            # reminder: use `del` to modify list in-place
            del tokens[max_length:]

    indexed_tokens = torch.tensor(indexed_tokens)

    return indexed_tokens.to("cuda"), torch.tensor(attention_masks).to("cuda") \
        if return_attention_masks else indexed_tokens.to("cuda")


# functions for loading lines from the amazon dataset
def process_amazon_lines(lines):
    X, y = [], []

    for line in lines:
        try:
            split_idx = line.index(" ")
            label = 0 if line[:split_idx] == "__label__1" else 1
            text = line[split_idx + 1:][:-1]  # ignore "\n"

            X.append(text)
            y.append(label)
        except ValueError:
            pass

    return X, y


def load_amazon_lines_random(file_dir, batch_size):
    amazon_config = json.load(open("datasets/amazon/amazon_config.json", "r"))
    max_length = amazon_config["train_total_length"] if "train.ft.txt" in file_dir \
        else amazon_config["test_total_length"]
    lines_idxs = [random.randrange(0, max_length) for _ in range(batch_size)]
    lines = [linecache.getline(file_dir, i + 1) for i in lines_idxs]
    X, y = process_amazon_lines(lines)
    return X, y


def load_amazon_lines(file_dir, start_idx, end_idx):
    # linecache does not use 0 indexing!
    lines = [linecache.getline(file_dir, i + 1) for i in range(start_idx, end_idx)]
    X, y = process_amazon_lines(lines)
    return X, y


# functions for loading lines from the meld dataset
def process_meld_lines(lines, mode, meld_config):
    X, y = [], []

    for line in lines:
        try:
            # not a great implementation, but whatever!
            line = list(pd.read_csv(StringIO(line)))
            text = line[1]

            if mode == "categorical":
                # maps emotion to class label (7 classes total)
                label = meld_config["emotions_mapping"][line[3]]
            elif mode == "binary":
                # maps sentiment to label (0, 0.5, or 1)
                # "." to address the problem of pandas adding a ".1"
                sentiment = line[4].split(".")[0]
                if sentiment == "positive":
                    label = 1
                elif sentiment == "negative":
                    label = 0
                else:
                    # there must be a better way of handling this
                    continue
            else:
                raise ValueError

            X.append(text)
            y.append(label)
        except ValueError:
            pass

    return X, y


def load_meld_lines_random(file_dir, batch_size, num_labels=7):
    meld_config = json.load(open("datasets/MELD/meld_config.json", "r"))
    split = file_dir.split("/")[-1].split("_")[0]  # train/dev/test
    max_length = meld_config[split + "_total_length"]

    lines_idxs = [random.randrange(0, max_length) for _ in range(batch_size)]
    # the files are in ".csv" format, so the first line is ignored (therefore j + 1)
    lines = [linecache.getline(file_dir, i + 2) for i in lines_idxs]

    X, y = process_meld_lines(lines, "categorical" if num_labels > 2 else "binary", meld_config)

    return X, y


def load_meld_lines(file_dir, start_idx, end_idx, mode="categorical"):
    meld_config = json.load(open("datasets/MELD/meld_config.json", "r"))
    # skips the first line (header)
    lines = [linecache.getline(file_dir, i + 2) for i in range(start_idx, end_idx)]
    X, y = process_meld_lines(lines, mode, meld_config)
    return X, y
