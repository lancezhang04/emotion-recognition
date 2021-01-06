from utils.preprocessing import load_amazon_lines_random, load_meld_lines_random, \
    preprocess_bert, load_amazon_lines, load_meld_lines
from transformers import BertForSequenceClassification, AdamW, \
    get_linear_schedule_with_warmup, BertTokenizer
import torch
from torch.nn import Parameter
import os
import json


def load_configuration(train_config, datasets_config):
    """
    Utility function to use before training.
    :param train_config: the training configuration file
    :param datasets_config: the datasets configuration file
    :return: configuration as dictionary
    """

    # using a dictionary to facilitate retrieving variable name
    config = {}

    dataset_name = train_config["dataset"]["name"]
    try:
        config["dataset"] = {}
        config["dataset"].update(train_config["dataset"])
        config["dataset"].update(datasets_config[dataset_name])
        config["dataset"]["name"] = dataset_name
    except KeyError:
        raise ValueError("The dataset \"" + dataset_name + "\" is not available")

    config.update(train_config["train"])

    config["visuals"] = {}
    config["visuals"].update(train_config["visuals"])

    config["model"] = {}
    config["model"].update(train_config["model"])
    config["model"]["num_labels"] = train_config["dataset"]["num_classes"]

    config["total_steps"] = config["epochs"] * config["dataset"]["train_total_length"]
    config["batches_per_epoch"] = config["dataset"]["train_total_length"] // config["batch_size"]
    config["batches_per_validation"] = config["dataset"]["validation_total_length"] // config["batch_size"]

    config["batches_per_save"] = train_config["saving"]["batches_per_save"] \
        if train_config["saving"]["batches_per_save"] else config["batches_per_epoch"]
    config["save_folder"] = train_config["saving"]["save_folder"]

    if config["verbose"] > 0:
        for key, value in config.items():
            print(key + ":", value)

    return config


def load_model(config, train=True):
    if config["model"]["type"] not in ["bert"]:
        raise ValueError("The model\"" + config["model"] + "\" is not available")

    if config["model"]["type"] == "bert":
        # `use_pretrained` --> use model trained and saved locally on the machine
        if config["model"]["use_pretrained"]:
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
            mode="categorical" if config["model"]["num_labels"] > 2 else "binary"
        )
    else:
        raise ValueError("The dataset \"" + dataset_name + "\" is not available")

    X_train, X_mask = preprocess_bert(sentences, tokenizer, config)
    y_train = torch.tensor(y_train).reshape(-1, 1)
    y_train = y_train.to("cuda")

    return X_train, X_mask, y_train
