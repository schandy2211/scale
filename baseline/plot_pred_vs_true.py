import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_rows(csv_path: str):
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        header = r.fieldnames or []
    return header, rows


def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except Exception:
        return False


def pick_true_value(row):
    # Prefer composite eff_score if present and numeric; else fall back to qed
    val = row.get("eff_score", "")
    try:
        return float(val)
    except Exception:
        pass
    try:
        return float(row.get("qed", ""))
    except Exception:
        return None


def plot(csv_path: str, round_num: int, out_path: str):
    header, all_rows = load_rows(csv_path)
    if "y_pred_mean" not in header:
        raise RuntimeError("CSV has no y_pred_mean column. Rerun optimizer after prediction logging was added.")
    # Auto-pick last round if round_num <= 0
    if round_num <= 0:
        rounds = [int(r.get("round", 0) or 0) for r in all_rows if r.get("phase") == "round"]
        if not rounds:
            raise RuntimeError("No 'phase=round' rows found in CSV.")
        round_num = max(rounds)
    rows = [r for r in all_rows if r.get("phase") == "round" and int(r.get("round", 0) or 0) == round_num]
    if not rows:
        raise RuntimeError(f"No rows for round {round_num} in {csv_path}")

    # Determine which true label we are plotting for the axis label
    # Prefer eff_score if any row has a numeric value; otherwise use qed
    use_eff = any(
        (r.get("phase") == "round" and int(r.get("round", 0) or 0) == round_num and _is_float(r.get("eff_score", "")))
        for r in all_rows
    )
    x_label = f"True ({'eff_score' if use_eff else 'qed'})"

    y_true = []
    y_pred = []
    for row in rows:
        yt = pick_true_value(row)
        try:
            yp = float(row.get("y_pred_mean", ""))
        except Exception:
            yp = None
        if yt is None or yp is None:
            continue
        y_true.append(yt)
        y_pred.append(yp)

    if not y_true:
        raise RuntimeError("No valid y_true/y_pred rows found")

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    plt.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel("Predicted (mean)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Write sidecar metadata JSON for traceability
    try:
        import json

        meta = {
            "source_csv": os.path.abspath(csv_path),
            "round": round_num,
            "true_label": "eff_score" if use_eff else "qed",
            "n_points": len(y_true),
            "x_min": float(min(y_true)),
            "x_max": float(max(y_true)),
            "y_min": float(min(y_pred)),
            "y_max": float(max(y_pred)),
        }
        with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Scatter plot of true vs predicted from CSV")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plot(args.csv, args.round, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
