import argparse
import csv
from pathlib import Path


METRICS = ("semantic", "quality", "total")
METHODS = ("tome", "importance", "quadtree")


def read_scores(path: Path):
    scores = {}
    if not path.exists():
        return scores
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratio = row["ratio"]
            method = row["method"]
            scores[(ratio, method)] = row
    return scores


def fmt(scores, ratio, method, metric):
    row = scores.get((ratio, method))
    if not row:
        return "--"
    value = row.get(metric, "")
    if value == "":
        return "--"
    return f"{float(value):.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path("outputs/vbench_table_anim/scores.csv"))
    parser.add_argument("--ratios", nargs="+", default=["0.40", "0.60", "0.70", "0.75"])
    args = parser.parse_args()

    scores = read_scores(args.scores)
    header = ["r"]
    for metric in METRICS:
        for method in METHODS:
            header.append(f"{metric}_{method}")

    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for ratio in args.ratios:
        cells = [ratio]
        for metric in METRICS:
            for method in METHODS:
                cells.append(fmt(scores, ratio, method, metric))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
