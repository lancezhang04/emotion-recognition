import torch
from utils.preprocessing import numericalize, preprocess_bert
from transformers import BertTokenizer
import numpy as np


def test_bert_binary(model, sentences):
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    X, X_mask = preprocess_bert(sentences, tokenizer)
    X = X.to("cuda")
    X_mask = X_mask.to("cuda")

    with torch.no_grad():
        outputs = model(
            X,
            token_type_ids=None,
            attention_mask=X_mask
        )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    pred = np.argmax(logits, axis=1)
    return pred, logits


def test_sentences_binary(model, word_to_index, device, sentences):
    model.eval()

    sentences = numericalize(sentences, word_to_index)
    sentences = torch.LongTensor(sentences).to(device)

    preds = torch.sigmoid(model(sentences))
    preds = torch.flatten(preds).tolist()

    return [1 if pred >= 0.5 else 0 for pred in preds]
