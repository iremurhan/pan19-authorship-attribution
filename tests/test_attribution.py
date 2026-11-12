"""Tests on a tiny synthetic PAN-style collection (no real data, no NLTK models)."""
import json
import os

import numpy as np

from authorship.attribute import UNK, attribute_collection, extract_vocabulary, reject
from authorship.evaluate import evaluate_problem, score
from authorship.preprocess import preprocess

STYLES = {
    "candidate00001": "the quick zebra zigzags zealously; zany zoo zone. ",
    "candidate00002": "moonlit harbour, quiet oars; hollow moors, cool shores. ",
    "candidate00003": "xylophones vex; jinxed lynx fix boxes, exactly sixty. ",
}


def _make_collection(root):
    problem = os.path.join(root, "problem00001")
    os.makedirs(os.path.join(problem, "unknown"))
    rng = np.random.default_rng(0)
    for author, style in STYLES.items():
        os.makedirs(os.path.join(problem, author))
        for k in range(6):
            words = style.split()
            text = " ".join(rng.permutation(words).tolist() * 8)
            with open(os.path.join(problem, author, f"known{k:05d}.txt"), "w") as f:
                f.write(text)
    truth = []
    for k, (author, style) in enumerate(STYLES.items()):
        name = f"unknown{k:05d}.txt"
        with open(os.path.join(problem, "unknown", name), "w") as f:
            f.write(style * 6)
        truth.append({"unknown-text": name, "true-author": author})
    with open(os.path.join(problem, "ground-truth.json"), "w") as f:
        json.dump({"ground_truth": truth}, f)
    with open(os.path.join(problem, "problem-info.json"), "w") as f:
        json.dump({"unknown-folder": "unknown",
                   "candidate-authors": [{"author-name": a} for a in STYLES]}, f)
    with open(os.path.join(root, "collection-info.json"), "w") as f:
        json.dump([{"problem-name": "problem00001", "language": "en"}], f)


def test_end_to_end_on_separable_authors(tmp_path):
    data, out = tmp_path / "data", tmp_path / "answers"
    _make_collection(data)
    attribute_collection(str(data), str(out), model="svm", n=3, ft=2, pt=0.0)
    answers = json.load(open(out / "answers-problem00001.json"))
    assert {a["unknown-text"] for a in answers} == {f"unknown{k:05d}.txt" for k in range(3)}
    f1, _, _ = evaluate_problem(str(data), "problem00001", str(out))
    assert f1 == 1.0


def test_reject_option():
    proba = np.array([[0.50, 0.45, 0.05], [0.90, 0.05, 0.05]])
    assert reject(["a", "a"], proba, pt=0.1) == [UNK, "a"]


def test_vocabulary_frequency_threshold():
    assert extract_vocabulary(["aaaa", "ab"], n=2, ft=3) == ["aa"]


def test_score_ignores_unk_in_macro_average():
    gold = {"u1": "a", "u2": "b", "u3": UNK}
    assert score(gold, {"u1": "a", "u2": "b", "u3": UNK})[0] == 1.0
    assert score(gold, {"u1": "a"})[0] == 0.5          # missing u2 counts as <UNK>


def test_preprocess_order_removes_stopwords_before_stemming():
    assert preprocess("He was running, quickly!", {"punctuation", "stopwords", "stem"}) == "run quickli"
