import argparse
import csv
import os


FIELDS = [
    "accept_rate",
    "mean_pairwise_dist",
    "unique_scaffolds",
    "mae_sel",
    "rmse_sel",
    "r2_sel",
    "spearman_sel",
    "mae_audit",
    "rmse_audit",
    "r2_audit",
    "spearman_audit",
]


def summarize_one(path: str) -> None:
    if not os.path.exists(path):
        print(f"[missing] {path}")
        return
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = [row for row in r if row.get("phase") == "round_summary"]
    if not rows:
        print(f"[no round_summary] {path}")
        return
    last = rows[-1]
    round_num = last.get("round", "?")
    # Count selected entries for this round
    with open(path, "r", newline="") as f:
        r2 = csv.DictReader(f)
        n_selected = sum(1 for row in r2 if row.get("phase") == "round" and row.get("round") == str(round_num))
    # Print compact summary
    print(f"\n{path}")
    print(f"  round: {round_num}  selected: {n_selected}")
    for k in FIELDS:
        v = last.get(k, "")
        print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser(description="Summarize last round_summary metrics from CSV logs")
    ap.add_argument("csv", nargs="+", help="Path(s) to CSV file(s)")
    args = ap.parse_args()
    for p in args.csv:
        summarize_one(p)


if __name__ == "__main__":
    main()

