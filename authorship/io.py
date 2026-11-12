"""Reading PAN-19 attribution collections and writing answers in PAN format."""
import glob
import json
import os


def load_collection(path):
    """Return [(problem_name, language), ...] from collection-info.json."""
    with open(os.path.join(path, "collection-info.json"), encoding="utf-8") as f:
        return [(p["problem-name"], p["language"]) for p in json.load(f)]


def load_problem(path, problem):
    """Return (candidate author names, unknown-folder name) for one problem."""
    with open(os.path.join(path, problem, "problem-info.json"), encoding="utf-8") as f:
        info = json.load(f)
    return [c["author-name"] for c in info["candidate-authors"]], info["unknown-folder"]


def read_texts(folder):
    """Return [(file_name, text), ...] for every .txt file, in sorted order."""
    out = []
    for file in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(file, encoding="utf-8") as f:
            out.append((os.path.basename(file), f.read()))
    return out


def write_answers(outdir, problem, names, predictions):
    os.makedirs(outdir, exist_ok=True)
    data = [{"unknown-text": n, "predicted-author": p} for n, p in zip(names, predictions)]
    path = os.path.join(outdir, f"answers-{problem}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return path
