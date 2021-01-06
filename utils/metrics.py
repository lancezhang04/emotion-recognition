import numpy as np


def accuracy(preds, labels):
    if type(preds) != np.ndarray:
        preds = preds.detach().cpu().numpy()
    if type(labels) != np.ndarray:
        labels = labels.detach().cpu().numpy()

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(preds_flat == labels_flat) / len(labels_flat)


def top_k_accuracies(preds, labels, k):
    if type(preds) != np.ndarray:
        preds = preds.detach().cpu().numpy()
    if type(labels) != np.ndarray:
        labels = labels.detach().cpu().numpy()

    preds = np.argsort(preds, axis=1)[:, ::-1]
    labels = labels.flatten()
    accuracies = []
    for i in range(k):
        accuracies.append(np.sum(preds[:, i] == labels) / len(labels))

    return [sum(accuracies[:i + 1]) for i in range(k)]
