from utils.train import load_model, load_configuration, load_batch_bert
from utils.misc import format_description
from utils.metrics import accuracy

from tqdm import trange
import torch
import json
import os


config = load_configuration(
    json.load(open("config/train_config.json")),
    json.load(open("config/datasets_config.json"))
)
model, optimizer, scheduler, tokenizer = load_model(
    config, train=True
)

# loading training log
if config["model"]["use_pretrained"]:
    history = json.load(open(os.path.join(config["model"]["pretrained_folder"], "train.log")))
else:
    # losses and metrics to save in log file
    history = {"train_loss": [], "train_acc": [], "validation_loss": [], "validation_acc": [], "total_batches": 0}


def save():
    folder_dir = os.path.join(config["save_folder"], str(history["total_batches"]))
    if config["model"]["type"] == "bert":
        model.save_pretrained(folder_dir)
        torch.save(optimizer.state_dict(), os.path.join(folder_dir, "optimizer.pt"))
        json.dump(history, open(os.path.join(folder_dir, "train.log"), "w"))


for e in range(config["epochs"]):
    # doesn't work for some reason --> creates an empty progress bar that no one likes
    if config["verbose"] > 0:
        print("\nepoch %i/%i" % (e + 1, config["epochs"]))

    model.train()
    epoch_history = {"epoch_loss": 0, "epoch_acc": 0}
    history["train_loss"].append([])
    history["train_acc"].append([])
    train_progress_bar = trange(config["batches_per_epoch"], desc="", ncols=config["visuals"]["ncols"])
    for b in train_progress_bar:
        model.zero_grad()

        if config["model"]["type"] == "bert":
            X_train, X_mask, y_train = load_batch_bert(tokenizer, config, b, mode="train")
            outputs = model(
                X_train,
                token_type_ids=None,
                attention_mask=X_mask,
                labels=y_train
            )
            # batch_loss is kept as a tensor for backpropagation
            batch_loss = outputs[0]
            batch_acc = accuracy(outputs.logits, y_train)
        else:
            raise ValueError("The model \"" + config["model"]["type"] + "\" is not available")

        epoch_history["epoch_loss"] += batch_loss.item()
        history["train_loss"][-1].append(batch_loss.item())
        epoch_history["epoch_acc"] += batch_acc
        history["train_acc"][-1].append(batch_acc)
        history["total_batches"] += 1

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_progress_bar.set_description(format_description(epoch_history, b + 1))
        train_progress_bar.refresh()

        if history["total_batches"] % config["batches_per_save"] == 0:
            if config["verbose"] > 0:
                print("saving...")
            save()

    model.eval()
    model.zero_grad()
    validation_history = {"validation_loss": 0, "validation_acc": 0}
    history["validation_loss"].append([])
    history["validation_acc"].append([])
    validation_progress_bar = trange(config["batches_per_validation"], desc="", ncols=config["visuals"]["ncols"])
    for b in validation_progress_bar:
        if config["model"]["type"] == "bert":
            with torch.no_grad():
                X_val, X_mask, y_val = load_batch_bert(tokenizer, config, b, mode="validation")
                outputs = model(
                    X_val,
                    token_type_ids=None,
                    attention_mask=X_mask,
                    labels=y_val
                )
                batch_loss = outputs[0].item()
                batch_acc = accuracy(outputs.logits, y_val)
        else:
            raise ValueError("The model \"" + config["model"]["type"] + "\" is not available")

        validation_history["validation_loss"] += batch_loss
        history["validation_loss"][-1].append(batch_loss)
        validation_history["validation_acc"] += batch_acc
        history["validation_acc"][-1].append(batch_acc)

        validation_progress_bar.set_description(format_description(validation_history, b + 1))
        validation_progress_bar.refresh()

print("execution complete")
