from utils.preprocessing import load_amazon_lines_random, load_meld_lines_random, \
    preprocess_bert, load_amazon_lines, load_meld_lines
from transformers import BertForSequenceClassification, AdamW, \
    get_linear_schedule_with_warmup, BertTokenizer
import torch
from torch.nn import Parameter
import os
import json


def load_configuration(config, datasets_config, train=True):
    """
    Utility function to use before training.
    :param config: the training configuration file
    :param datasets_config: the datasets configuration file
    :param train: whether loading configuration for train or test
    :return: configuration as dictionary
    """

    # using a dictionary to facilitate retrieving variables
    combined_config = {}

    # add configuration for dataset
    dataset_name = config["dataset"]["name"]
    try:
        combined_config["dataset"] = {}
        combined_config["dataset"].update(config["dataset"])
        combined_config["dataset"].update(datasets_config[dataset_name])
        combined_config["dataset"]["name"] = dataset_name
    except KeyError:
        raise ValueError("The dataset \"" + dataset_name + "\" is not available")

    # add configuration for model
    combined_config["model"] = {}
    combined_config["model"].update(config["model"])
    combined_config["model"]["num_labels"] = config["dataset"]["num_classes"]

    # add configuration for visuals
    combined_config["visuals"] = {}
    combined_config["visuals"].update(config["visuals"])

    if train:
        combined_config.update(config["train"])
    else:
        combined_config["batch_size"] = config["batch_size"]
        combined_config["use_dataset"] = config["use_dataset"]

    # calculate number of batches in each =part of the dataset
    combined_config["batches_per_epoch"] = \
        combined_config["dataset"]["train_total_length"] // combined_config["batch_size"]
    # bad implementation (for benchmarking), fix later
    combined_config["batches_per_train"] = combined_config["batches_per_epoch"]
    combined_config["batches_per_validation"] = \
        combined_config["dataset"]["validation_total_length"] // combined_config["batch_size"]
    combined_config["batches_per_test"] = \
        combined_config["dataset"]["test_total_length"] // combined_config["batch_size"]

    # add configuration specific to training; print out configuration if necessary
    if train:
        combined_config["total_steps"] = combined_config["epochs"] * combined_config["dataset"]["train_total_length"]

        combined_config["batches_per_save"] = config["saving"]["batches_per_save"] \
            if config["saving"]["batches_per_save"] else combined_config["batches_per_epoch"]
        combined_config["save_folder"] = config["saving"]["save_folder"]

        if combined_config["verbose"] > 0:
            for key, value in combined_config.items():
                print(key + ":", value)

    return combined_config


def load_model(config, train=True):
    if config["model"]["type"] not in ["bert"]:
        raise ValueError("The model\"" + config["model"] + "\" is not available")

    if config["model"]["type"] == "bert":
        # `use_pretrained` --> use model trained and saved locally on the machine
        if not train or config["model"]["use_pretrained"]:
            model = BertForSequenceClassification.from_pretrained(
                config["model"]["pretrained_folder"]
            )
        else:
            # use pre-trained model from hugging face
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=config["model"]["num_labels"],
                output_attentions=False,
                output_hidden_states=False
            )
        model.cuda()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        if train:
            optimizer = AdamW(
                model.parameters(),
                lr=config["learning_rate"],
                eps=1e-8
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=config["total_steps"]
            )

            if config["model"]["use_pretrained"]:
                optimizer.load_state_dict(torch.load(
                    os.path.join(config["model"]["pretrained_folder"], "optimizer.pt"
                )))

            return model, optimizer, scheduler, tokenizer
        return model, tokenizer


def load_batch_bert(tokenizer, config, batch_idx, mode="train"):
    if mode not in ["train", "validation", "test"]:
        raise ValueError

    dataset_name = config["dataset"]["name"]
    if dataset_name == "amazon":
        sentences, y_train = load_amazon_lines(
            config["dataset"][mode + "_dir"],
            batch_idx * config["batch_size"],
            (batch_idx + 1) * config["batch_size"]
        )
    elif dataset_name == "meld":
        sentences, y_train = load_meld_lines(
            config["dataset"][mode + "_dir"],
            batch_idx * config["batch_size"],
            (batch_idx + 1) * config["batch_size"],
            num_labels=config["model"]["num_labels"],
            meld_config=config["dataset"]
        )
    else:
        raise ValueError("The dataset \"" + dataset_name + "\" is not available")

    X_train, X_mask = preprocess_bert(sentences, tokenizer, config)
    y_train = torch.tensor(y_train).reshape(-1, 1)
    y_train = y_train.to("cuda")

    return X_train, X_mask, y_train
