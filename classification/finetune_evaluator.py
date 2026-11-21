"""Scores and predictions on one participant.

Adapted from https://github.com/wjq-learning/CBraMod (MIT licence).
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_metrics_for_multiclass(self, model):
        """Accuracy, balanced accuracy, kappa, macro-F1 and confusion matrix."""
        model.eval()

        truths = []
        preds = []
        for x, y, _ in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]

            truths += y.tolist()
            preds += pred_y.cpu().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        acc = accuracy_score(truths, preds)
        bacc = balanced_accuracy_score(truths, preds)
        kappa = cohen_kappa_score(truths, preds)
        f1 = f1_score(truths, preds, labels=np.unique(truths), average='macro', zero_division=0)
        cm = confusion_matrix(truths, preds, labels=list(range(self.params.num_of_classes)))
        return acc, bacc, kappa, f1, cm

    def predict(self, model):
        """Per-epoch predictions, sorted by epoch_id."""
        model.eval()
        epoch_ids, y_true, y_pred, y_prob = [], [], [], []
        with torch.no_grad():
            for x, y, batch_ids in tqdm(self.data_loader, mininterval=1):
                logits = model(x.cuda())
                epoch_ids += batch_ids
                y_true += y.tolist()
                y_pred += torch.argmax(logits, dim=-1).cpu().tolist()
                y_prob.append(torch.softmax(logits, dim=-1).cpu().numpy())

        epoch_ids = np.array(epoch_ids)
        order = np.argsort(epoch_ids)
        subject_ids = np.array([int(e.split('_')[0][len('sub-'):]) for e in epoch_ids], dtype=np.int64)
        return {
            'epoch_id': epoch_ids[order],
            'y_true': np.array(y_true, dtype=np.int64)[order],
            'y_pred': np.array(y_pred, dtype=np.int64)[order],
            'y_prob': np.concatenate(y_prob).astype(np.float32)[order],
            'subject_id': subject_ids[order],
        }
