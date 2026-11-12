# Cross-Domain Authorship Attribution (PAN-19)

Given a handful of known texts from each of nine candidate authors, decide who wrote each unknown text, or answer `<UNK>` if none of them did. The known and unknown texts come from different fanfiction "domains" (fandoms), which makes topic cues unreliable.

Term project for CMPE 493 (Introduction to Information Retrieval), Boğaziçi University, Spring 2023, done as a team project. The code was refactored into a small package in 2026; the 2023 script is kept in [`original/`](original/) and the exploration notebooks in [`notebooks/`](notebooks/).

## Data

Not included. Request the [PAN-19 cross-domain authorship attribution corpus](https://doi.org/10.5281/zenodo.3530313) from Zenodo and point `--data` at the extracted folder (the one containing `collection-info.json`). The project uses the five English problems: `problem00002` and `problem00004` as the development set, and `problem00001`, `problem00003` and `problem00005` as the evaluation set.

## Method

Built on the official PAN-19 SVM baseline:

1. Character *n*-grams that occur at least `ft` times in the known texts form the vocabulary.
2. Counts are divided by text length and MaxAbs-scaled.
3. A probability-calibrated classifier is trained on the known texts: one-vs-rest SVM (baseline), bagged SVMs, AdaBoost over linear SVMs, *k*-NN or a random forest.
4. **Reject option:** if the two most likely authors differ by less than `pt` in probability, the text is labelled `<UNK>`.

Optional extensions explored in the project:

- **Preprocessing:** punctuation removal, stop-word removal, Porter stemming, case folding.
- **POS + sentiment augmentation:** the text's part-of-speech tag sequence (NLTK) and a polarity token (TextBlob) are appended before taking character *n*-grams, so syntactic patterns contribute features.

Scoring follows the PAN evaluator: macro-F1 over the candidate authors, where `<UNK>` is a valid answer but not averaged as a class.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng

python main.py run --data DATA_DIR --out answers/svm --model svm -n 3 --pt 0.1
python main.py run --data DATA_DIR --out answers/ada --model adaboost_svm -n 4 --pt 0.025 --pos-sentiment
python main.py evaluate --data DATA_DIR --answers answers/ada

pytest -q    # synthetic-data tests; no corpus needed
```

## Results

Mean macro-F1, re-run with this code on 2026-11-12 (scikit-learn 1.9):

| Model | n | pt | Features | Dev (P2, P4) | Eval (P1, P3, P5) | 2023 slides (dev / eval) |
|---|---|---|---|---|---|---|
| SVM baseline | 3 | 0.1 | character n-grams | 0.348 | 0.509 | 0.348 / 0.509 |
| SVM baseline | 4 | 0.1 | character n-grams | 0.408 | 0.540 | 0.401 / 0.536 |
| Random forest | 4 | 0.01 | character n-grams | 0.300 | 0.457 | 0.362 / 0.48 |
| AdaBoost (linear SVM) | 4 | 0.025 | + POS, sentiment | **0.436** | **0.605** | 0.436 / 0.605 |

The baseline and the best model (AdaBoost with POS and sentiment features) reproduce the 2023 numbers exactly. Small differences elsewhere come from library versions and model randomness. The 2023 slides also contain two different random-forest evaluation scores (0.48 in the results table, 0.605 on a summary slide); the table value is the one listed here. Hyper-parameters were chosen on the development problems, but with only two of them the evaluation numbers are noisy.

## What changed from the 2023 version

- Five near-identical ~100-line functions (one per classifier) were merged into one pipeline with a model registry and a CLI.
- Stop words are now removed *before* stemming; the old order let stemmed forms (e.g. "was" → "wa") slip past the stop-word list.
- Files are read in sorted order, so answers are deterministic across operating systems.
- Unit tests on a synthetic collection.

## Licence

MIT. The baseline representation and the scoring logic are adapted from [pan-webis-de/pan-code](https://github.com/pan-webis-de/pan-code) (MIT, © 2020 PAN Shared Task Series); see [`LICENSE`](LICENSE).
