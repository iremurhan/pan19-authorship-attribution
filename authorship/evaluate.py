"""PAN-19 attribution scoring: macro-F1 over the known candidate authors.

Adapted from the official PAN-19 evaluator (pan-webis-de/pan-code,
MIT licence, Copyright (c) 2020 PAN Shared Task Series). As in the
official script, <UNK> is a valid prediction but is excluded from the
macro average, and missing or unknown predictions count as <UNK>.
"""
import json
import os
import warnings

from sklearn.metrics import f1_score, precision_score, recall_score

UNK = "<UNK>"


def score(gold, pred):
    """gold, pred: {unknown-text: author}. Returns (macro-F1, precision, recall)."""
    ids = sorted(gold)
    authors = sorted(set(gold.values()) - {UNK})
    y_true = [gold[i] for i in ids]
    y_pred = [pred.get(i, UNK) if pred.get(i, UNK) in authors else UNK for i in ids]
    labels = sorted(set(y_true) - {UNK})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = dict(labels=labels, average="macro", zero_division=0)
        return (f1_score(y_true, y_pred, **kw), precision_score(y_true, y_pred, **kw),
                recall_score(y_true, y_pred, **kw))


def evaluate_problem(path, problem, answers_dir):
    with open(os.path.join(path, problem, "ground-truth.json"), encoding="utf-8") as f:
        gold = {g["unknown-text"]: g["true-author"] for g in json.load(f)["ground_truth"]}
    with open(os.path.join(answers_dir, f"answers-{problem}.json"), encoding="utf-8") as f:
        pred = {}
        for a in json.load(f):
            pred.setdefault(a["unknown-text"], a["predicted-author"])
    return score(gold, pred)
