"""Cross-domain authorship attribution on PAN-19 (English problems 1-5).

    python main.py run --data DATA_DIR --out answers/ [--model svm] [-n 3] [--ft 5] [--pt 0.1]
                       [--preprocess punctuation stopwords stem casefold] [--pos-sentiment]
    python main.py evaluate --data DATA_DIR --answers answers/

Problems 2 and 4 are the development set and 1, 3, 5 the evaluation set,
as in the course project.
"""
import argparse

from authorship.attribute import MODELS, attribute_collection
from authorship.evaluate import evaluate_problem
from authorship.preprocess import STEPS

DEV = ["problem00002", "problem00004"]
EVAL = ["problem00001", "problem00003", "problem00005"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="attribute the unknown texts")
    run.add_argument("--data", required=True, help="PAN-19 collection folder")
    run.add_argument("--out", required=True, help="folder for answers-*.json")
    run.add_argument("--model", choices=MODELS, default="svm")
    run.add_argument("-n", type=int, default=3, help="character n-gram length")
    run.add_argument("--ft", type=int, default=5, help="minimum n-gram frequency")
    run.add_argument("--pt", type=float, default=0.1, help="reject-option probability gap")
    run.add_argument("--preprocess", nargs="*", default=[], choices=list(STEPS))
    run.add_argument("--pos-sentiment", action="store_true",
                     help="append POS tags and a sentiment token before taking n-grams")
    run.add_argument("--problems", nargs="*", default=DEV + EVAL)

    ev = sub.add_parser("evaluate", help="macro-F1 per problem and per split")
    ev.add_argument("--data", required=True)
    ev.add_argument("--answers", required=True)

    args = parser.parse_args()
    if args.command == "run":
        attribute_collection(args.data, args.out, problems=set(args.problems), model=args.model,
                             n=args.n, ft=args.ft, pt=args.pt, steps=set(args.preprocess),
                             pos_sentiment=args.pos_sentiment)
        return

    split_scores = {"dev": [], "eval": []}
    for problem in DEV + EVAL:
        f1, precision, recall = evaluate_problem(args.data, problem, args.answers)
        split = "dev" if problem in DEV else "eval"
        split_scores[split].append(f1)
        print(f"{problem} [{split}]  macro-F1 {f1:.3f}  P {precision:.3f}  R {recall:.3f}")
    for split, scores in split_scores.items():
        print(f"mean macro-F1 ({split}): {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    main()
