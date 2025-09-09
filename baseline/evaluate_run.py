import argparse
import csv
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def load_csv(path: str) -> Tuple[List[str], List[dict]]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        header = r.fieldnames or []
    return header, rows


def last_round(rows: List[dict]) -> int:
    rnums = [int(r.get("round", 0) or 0) for r in rows if r.get("phase") == "round"]
    if not rnums:
        raise RuntimeError("No phase=round rows found")
    return max(rnums)


def pick_true_key(rows: List[dict]) -> str:
    # Prefer eff_score if any numeric values are present; else qed
    for r in rows:
        if r.get("phase") == "round" and _is_float(r.get("eff_score", "")):
            return "eff_score"
    return "qed"


def precision_at_k(pairs: List[Tuple[float, float]], k: int, hit_thresh: float) -> float:
    # pairs = [(pred, true), ...]
    if not pairs:
        return 0.0
    pairs = sorted(pairs, key=lambda t: t[0], reverse=True)
    top = pairs[: max(1, min(k, len(pairs)))]
    hits = sum(1 for _, t in top if t >= hit_thresh)
    return hits / max(1, len(top))


def plot_precision_curve(pairs: List[Tuple[float, float]], out_path: str, hit_thresh: float) -> None:
    if not pairs:
        return
    pairs = sorted(pairs, key=lambda t: t[0], reverse=True)
    ks = list(range(5, max(6, len(pairs) + 1), max(5, len(pairs) // 15 or 5)))
    precs = [precision_at_k(pairs, k, hit_thresh) for k in ks]
    base = sum(1 for _, t in pairs if t >= hit_thresh) / max(1, len(pairs))
    plt.figure(figsize=(6, 4))
    plt.plot(ks, precs, marker="o", label="precision@k")
    plt.axhline(base, color="red", linestyle="--", label=f"baseline={base:.2f}")
    plt.xlabel("k")
    plt.ylabel(f"precision@k (true ≥ {hit_thresh:.2f})")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_score_hist(y_true: List[float], out_path: str, hit_thresh: float) -> None:
    if not y_true:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(y_true, bins=20, color="#4c78a8", alpha=0.8)
    plt.axvline(hit_thresh, color="red", linestyle="--", label=f"threshold={hit_thresh:.2f}")
    plt.xlabel("True score")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate(csv_path: str, out_dir: str, round_num: int = 0, hit_thresh: float = 0.9) -> Dict:
    header, rows = load_csv(csv_path)
    if round_num <= 0:
        round_num = last_round(rows)

    # Selected set for the round
    sel = [r for r in rows if r.get("phase") == "round" and int(r.get("round", 0) or 0) == round_num]
    if not sel:
        raise RuntimeError(f"No selected rows for round {round_num}")

    true_key = pick_true_key(sel)
    # Build (pred, true) pairs; skip rows missing predictions
    pairs = []
    y_true = []
    y_pred = []
    for r in sel:
        yt = r.get(true_key, "")
        yp = r.get("y_pred_mean", "")
        if not (_is_float(yt) and _is_float(yp)):
            continue
        yt = float(yt)
        yp = float(yp)
        pairs.append((yp, yt))
        y_true.append(yt)
        y_pred.append(yp)

    # Precision curve
    os.makedirs(out_dir, exist_ok=True)
    plot_precision_curve(pairs, os.path.join(out_dir, "precision_curve.png"), hit_thresh)
    plot_score_hist(y_true, os.path.join(out_dir, "true_score_hist.png"), hit_thresh)

    # Basic counts
    n = len(pairs)
    hits = sum(1 for _, t in pairs if t >= hit_thresh)
    base_rate = hits / max(1, n)
    prec_at_20 = precision_at_k(pairs, 20, hit_thresh)
    prec_at_50 = precision_at_k(pairs, 50, hit_thresh)

    # Pull round_summary metrics if present
    summ = [r for r in rows if r.get("phase") == "round_summary" and int(r.get("round", 0) or 0) == round_num]
    summary = summ[-1] if summ else {}

    out = {
        "csv": os.path.abspath(csv_path),
        "round": round_num,
        "true_key": true_key,
        "selected_count": n,
        "hit_thresh": hit_thresh,
        "hits": hits,
        "baseline_hit_rate": base_rate,
        "precision_at_20": prec_at_20,
        "precision_at_50": prec_at_50,
        "accept_rate": summary.get("accept_rate"),
        "mean_pairwise_dist": summary.get("mean_pairwise_dist"),
        "unique_scaffolds": summary.get("unique_scaffolds"),
        "mae_sel": summary.get("mae_sel"),
        "rmse_sel": summary.get("rmse_sel"),
        "r2_sel": summary.get("r2_sel"),
        "spearman_sel": summary.get("spearman_sel"),
        "mae_audit": summary.get("mae_audit"),
        "rmse_audit": summary.get("rmse_audit"),
        "r2_audit": summary.get("r2_audit"),
        "spearman_audit": summary.get("spearman_audit"),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate a run CSV: compute hit stats, precision@k, and plots")
    ap.add_argument("--csv", required=True, help="Path to CSV produced by baseline_opt/demo/compare_modes")
    ap.add_argument("--out", required=True, help="Directory to save metrics and plots")
    ap.add_argument("--round", type=int, default=0, help="Round to evaluate (0 = last round)")
    ap.add_argument("--hit", type=float, default=0.9, help="Hit threshold for true score (eff_score or qed)")
    args = ap.parse_args()

    res = evaluate(args.csv, args.out, args.round, args.hit)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

