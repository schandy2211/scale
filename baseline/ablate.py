import argparse
import json
import os
from datetime import datetime
from typing import Dict

# Ensure project root on path
import os as _os, sys as _sys
_ROOT = _os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from baseline.baseline_opt import OptConfig, run_optimization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def run_variant(name: str, seeds, base_cfg: OptConfig, overrides: Dict, out_dir: str) -> Dict:
    cfg = OptConfig(**base_cfg.__dict__)  # shallow copy
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.log_csv = os.path.join(out_dir, f"{name}.csv")
    ensure_dir(out_dir)
    out = run_optimization(seed_smiles=seeds, config=cfg)
    with open(os.path.join(out_dir, f"{name}_history.json"), "w") as f:
        json.dump(out.get("history", {}), f, indent=2)
    with open(os.path.join(out_dir, f"{name}_top.json"), "w") as f:
        json.dump(out.get("top", []), f, indent=2)
    return out


def overlay_curves(histories: Dict[str, Dict], out_path: str) -> None:
    plt.figure(figsize=(7, 4))
    for label, hist in histories.items():
        xs = list(range(1, len(hist.get("best", [])) + 1))
        if not xs:
            continue
        plt.plot(xs, hist["best"], marker="o", label=f"{label} best")
        plt.plot(xs, hist["avg"], marker="s", linestyle="--", label=f"{label} avg")
    plt.xlabel("Round")
    plt.ylabel("QED")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def overlay_acceptance(csv_dir: str, variant_names, out_path: str) -> None:
    import csv as _csv
    import collections as _co
    plt.figure(figsize=(7, 4))
    for name in variant_names:
        path = os.path.join(csv_dir, f"{name}.csv")
        by_round = _co.defaultdict(list)
        try:
            with open(path, "r", newline="") as f:
                r = _csv.DictReader(f)
                for row in r:
                    if row.get("phase") == "round_summary":
                        rnd = int(row.get("round", 0) or 0)
                        acc = row.get("accept_rate")
                        if acc:
                            try:
                                by_round[rnd].append(float(acc))
                            except Exception:
                                pass
        except Exception:
            continue
        xs = sorted(by_round.keys())
        if not xs:
            continue
        ys = [sum(by_round[i]) / max(1, len(by_round[i])) for i in xs]
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("Round")
    plt.ylabel("Acceptance rate (selected/prescreened)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Run ablations and plot overlay curves")
    ap.add_argument("--out_dir", default="runs/ablate", help="Directory for outputs")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seeds", default="c1ccccc1,c1ccncc1,COc1ccccc1,CC(=O)N,CCN")
    args = ap.parse_args()

    ts_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    ensure_dir(ts_dir)

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    base_cfg = OptConfig(
        objective="qed",
        rounds=max(1, args.rounds),
        init_train_size=96,
        candidates_per_round=400,
        topk_per_round=40,
        k_exploration=1.2,
        lambda_strain=0.1,
        mmff_prescreen_factor=4.0,
        diversity_penalty=0.25,
        sa_soft_beta=0.1,
        scaffold_penalty_alpha=0.3,
        scaffold_cap_per_round=3,
        train_on_composite=True,
        novelty_penalty_alpha=0.2,
        use_controller=True,
    )

    variants = {
        "baseline": {},
        "no_physics": {"lambda_strain": 0.0, "mmff_prescreen_factor": 1.0},
        "no_uncertainty": {"k_exploration": 0.0, "ensemble_size": 1},
        "no_diversity": {"diversity_penalty": 0.0, "scaffold_penalty_alpha": 0.0, "scaffold_cap_per_round": None, "novelty_penalty_alpha": 0.0},
        "no_agent": {"use_controller": False},
    }

    histories = {}
    aubc_norm = {}
    for name, overrides in variants.items():
        print(f"Running variant: {name}")
        out = run_variant(name, seeds, base_cfg, overrides, ts_dir)
        histories[name] = out.get("history", {})
        # Compute normalized AUBC for best curve
        best_curve = histories[name].get("best", [])
        if best_curve:
            aubc = sum(best_curve)
            aubc_norm[name] = aubc / len(best_curve)
        else:
            aubc_norm[name] = 0.0
    overlay_curves(histories, os.path.join(ts_dir, "compare_curves.png"))
    overlay_acceptance(ts_dir, variants.keys(), os.path.join(ts_dir, "compare_accept.png"))

    # Summarize SA/strain stats from CSVs
    summary = {}
    for name in variants.keys():
        csv_path = os.path.join(ts_dir, f"{name}.csv")
        stats = {"sa_median": None, "strain_median": None}
        try:
            import csv as _csv
            sa_vals = []
            st_vals = []
            with open(csv_path, "r", newline="") as f:
                r = _csv.DictReader(f)
                for row in r:
                    if row.get("phase") != "round":
                        continue
                    sa = row.get("sa")
                    st = row.get("strain")
                    if sa:
                        try:
                            sa_vals.append(float(sa))
                        except Exception:
                            pass
                    if st:
                        try:
                            st_vals.append(float(st))
                        except Exception:
                            pass
            import statistics as _stats
            if sa_vals:
                stats["sa_median"] = float(_stats.median(sa_vals))
            if st_vals:
                stats["strain_median"] = float(_stats.median(st_vals))
        except Exception:
            pass
        summary[name] = {
            "aubc_norm": aubc_norm.get(name, 0.0),
            **stats,
        }

    with open(os.path.join(ts_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Ablations complete. Outputs in {ts_dir}")


if __name__ == "__main__":
    main()
