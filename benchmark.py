from utils.train import load_model, load_configuration, load_batch_bert
from utils.preprocessing import preprocess_bert
from utils.misc import format_description
from utils.metrics import accuracy

from tqdm import trange
import numpy as np
import torch
import json


config = load_configuration(
    json.load(open("config/benchmark_config.json")),
    json.load(open("config/datasets_config.json")),
    train=False
)
model, tokenizer = load_model(
    config, train=False
)
history = {"loss": 0, "acc": 0}


model.eval()
# if `use_dataset` == False, the for loop should be infinite...
validation_progress_bar = trange(config["batches_per_" + config["dataset"]["mode"]],
                                 desc="", ncols=config["visuals"]["ncols"]) if config["use_dataset"] else range(100)
for b in validation_progress_bar:
    if config["model"]["type"] == "bert":
        with torch.no_grad():
            if config["use_dataset"]:
                X_test, X_mask, y_test = load_batch_bert(tokenizer, config, b, mode=config["dataset"]["mode"])
                outputs = model(
                    X_test,
                    token_type_ids=None,
                    attention_mask=X_mask,
                    labels=y_test
                )
                batch_loss = outputs[0].item()
                batch_acc = accuracy(outputs.logits, y_test)
            else:
                user_input = input("enter sentences: ")
                if not user_input:
                    print("no inputs received")
                    pass
                else:
                    X_test, X_mask = preprocess_bert([user_input], tokenizer, config, return_attention_masks=True)
                    outputs = model(X_test, token_type_ids=None, attention_mask=X_mask)

                    preds = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
                    preds_ordered = np.argsort(preds)[::-1]

                    print(dict([(config["dataset"]["idx_to_text"][str(i)], preds[i]) for i in preds_ordered]),
                          end="\n\n")
    else:
        raise ValueError("The model \"" + config["model"]["type"] + "\" is not available")

    if config["use_dataset"]:
        history["loss"] += batch_loss
        history["acc"] += batch_acc
        validation_progress_bar.set_description(format_description(history, b + 1))
        validation_progress_bar.refresh()

print("execution complete")
