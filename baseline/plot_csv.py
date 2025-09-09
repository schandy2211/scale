import argparse
import csv
import os
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import matplotlib

# Use non-interactive backend so this works on headless/CLI
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _to_float(x: str) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def load_csv(path: str) -> Tuple[List[dict], List[dict]]:
    rounds: List[dict] = []
    finals: List[dict] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            phase = row.get("phase", "")
            row["round"] = int(row.get("round", 0) or 0)
            row["rank"] = int(row.get("rank", 0) or 0)
            row["qed"] = _to_float(row.get("qed", ""))
            row["sa"] = _to_float(row.get("sa", ""))
            row["strain"] = _to_float(row.get("strain", ""))
            row["score"] = _to_float(row.get("score", ""))
            if phase == "round":
                rounds.append(row)
            elif phase == "final":
                finals.append(row)
    return rounds, finals


def plot_learning_curves(round_rows: List[dict], out_path: str) -> None:
    by_round: Dict[int, List[dict]] = defaultdict(list)
    for r in round_rows:
        by_round[r["round"]].append(r)
    xs = sorted(by_round.keys())
    if not xs:
        return
    max_qed = [max((row["qed"] or 0.0) for row in by_round[i]) for i in xs]
    mean_qed = [
        sum((row["qed"] or 0.0) for row in by_round[i]) / max(1, len(by_round[i])) for i in xs
    ]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, max_qed, marker="o", label="Best QED (selected)")
    plt.plot(xs, mean_qed, marker="s", label="Mean QED (selected)")
    plt.xlabel("Round")
    plt.ylabel("QED")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_qed_sa_scatter(round_rows: List[dict], out_path: str) -> None:
    xs = []
    ys = []
    cs = []
    for r in round_rows:
        q = r["qed"]
        s = r["sa"]
        if q is None or s is None:
            continue
        xs.append(q)
        ys.append(s)
        cs.append(r["round"])
    if not xs:
        return
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(xs, ys, c=cs, cmap="viridis", s=12, alpha=0.7)
    cbar = plt.colorbar(sc)
    cbar.set_label("Round")
    plt.xlabel("QED")
    plt.ylabel("SA score (lower better)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_top_scaffolds(final_rows: List[dict], out_path: str, top_n: int = 20) -> None:
    if not final_rows:
        return
    scaffs = [row.get("scaffold", "") or "" for row in final_rows]
    freq = Counter(scaffs)
    items = freq.most_common(top_n)
    labels = [k if k else "" for k, _ in items]
    counts = [v for _, v in items]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), [l[:20] + ("…" if len(l) > 20 else "") for l in labels], rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Top Murcko scaffolds (final top list)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot learning curves and QED vs SA from CSV log")
    ap.add_argument("--csv", required=True, help="Path to CSV log produced by baseline_opt")
    ap.add_argument("--out_prefix", required=False, default="runs/plot", help="Prefix for output PNG files")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    rounds, finals = load_csv(args.csv)

    plot_learning_curves(rounds, args.out_prefix + "_curves.png")
    plot_qed_sa_scatter(rounds, args.out_prefix + "_qed_vs_sa.png")
    plot_top_scaffolds(finals, args.out_prefix + "_scaffolds.png")

    print("Saved plots:")
    print(args.out_prefix + "_curves.png")
    print(args.out_prefix + "_qed_vs_sa.png")
    print(args.out_prefix + "_scaffolds.png")


if __name__ == "__main__":
    main()

