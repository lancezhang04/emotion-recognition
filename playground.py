from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import torch
from utils.train import load_configuration, load_model
import json


config = load_configuration(
     json.load(open("config/train_config.json")),
     json.load(open("config/datasets_config.json")),
     train=True
)
config["model"]["type"] = "mobilebert"

model = load_model(config)


# tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
# model = MobileBertForSequenceClassification.from_pretrained(
#     "google/mobilebert-uncased",
#     num_labels=2,
#     output_attentions=False,
#     output_hidden_states=False
# )
#
# # "pt" --> pytorch
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# labels = torch.tensor([[1, 0]]).unsqueeze(0)
#
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits


print("test complete")
