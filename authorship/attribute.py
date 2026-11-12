"""Character n-gram authorship attribution with a reject option.

The representation and reject rule follow the PAN-19 SVM baseline
(E. Stamatatos, MIT licence; see original/pan19_baseline_svm.py):
character n-grams that occur at least `ft` times in the training texts,
length-normalised counts, MaxAbs scaling, and a calibrated classifier. A
text is left unattributed (<UNK>) when the two highest class
probabilities differ by less than `pt`.
"""
import os
from collections import Counter

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

from authorship.io import load_collection, load_problem, read_texts, write_answers
from authorship.preprocess import augment_pos_sentiment, preprocess

UNK = "<UNK>"


def make_model(name, seed=42):
    models = {
        "svm": lambda: OneVsRestClassifier(SVC(C=1)),
        "bagging_svm": lambda: BaggingClassifier(SVC(C=1), n_estimators=10, random_state=seed),
        "adaboost_svm": lambda: AdaBoostClassifier(SVC(C=1, kernel="linear"), n_estimators=50,
                                                   random_state=seed),
        "knn": lambda: KNeighborsClassifier(n_neighbors=5),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=seed),
    }
    return CalibratedClassifierCV(models[name]())


MODELS = ["svm", "bagging_svm", "adaboost_svm", "knn", "random_forest"]


def extract_vocabulary(texts, n, ft):
    """Character n-grams occurring at least `ft` times across `texts`."""
    counts = Counter()
    for text in texts:
        counts.update(text[i:i + n] for i in range(len(text) - n + 1))
    return sorted(g for g, c in counts.items() if c >= ft)


def vectorize(texts, vectorizer):
    """Character n-gram counts divided by text length."""
    X = vectorizer.transform(texts).astype(float)
    lengths = np.array([max(len(t), 1) for t in texts], dtype=float)
    return X.multiply(1.0 / lengths[:, None]).tocsr()


def reject(predictions, proba, pt):
    out = np.array(predictions, dtype=object)
    top2 = np.sort(proba, axis=1)[:, -2:]
    out[(top2[:, 1] - top2[:, 0]) < pt] = UNK
    return out.tolist()


def prepare(text, steps, pos_sentiment):
    text = preprocess(text, steps)
    return augment_pos_sentiment(text) if pos_sentiment else text


def attribute_problem(path, problem, model="svm", n=3, ft=5, pt=0.1, steps=(),
                      pos_sentiment=False, seed=42):
    """Return (unknown file names, predicted authors) for one problem."""
    candidates, unk_folder = load_problem(path, problem)
    train_texts, train_labels = [], []
    for author in candidates:
        for _, text in read_texts(os.path.join(path, problem, author)):
            train_texts.append(prepare(text, steps, pos_sentiment))
            train_labels.append(author)
    unknown = read_texts(os.path.join(path, problem, unk_folder))
    test_texts = [prepare(text, steps, pos_sentiment) for _, text in unknown]

    vocabulary = extract_vocabulary(train_texts, n, ft)
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(n, n), lowercase=False,
                                 vocabulary=vocabulary)
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(vectorize(train_texts, vectorizer))
    X_test = scaler.transform(vectorize(test_texts, vectorizer))

    clf = make_model(model, seed)
    clf.fit(X_train, train_labels)
    predictions = reject(clf.predict(X_test), clf.predict_proba(X_test), pt)
    return [name for name, _ in unknown], predictions


def attribute_collection(path, outdir, problems=None, **kwargs):
    """Run every (or the selected) problem and write answers-<problem>.json files."""
    for problem, language in load_collection(path):
        if problems and problem not in problems:
            continue
        names, predictions = attribute_problem(path, problem, **kwargs)
        n_unk = sum(p == UNK for p in predictions)
        print(f"{problem} ({language}): {len(names)} unknown texts, {n_unk} left unattributed")
        write_answers(outdir, problem, names, predictions)
