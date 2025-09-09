import argparse
import json
import os
import time
from datetime import datetime

# Ensure project root on sys.path
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from baseline.baseline_opt import OptConfig, run_optimization
from rdkit import Chem
from rdkit.Chem import Draw


def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def save_top_grid(top_list, out_path: str, mols_per_row: int = 6, max_mols: int = 36) -> None:
    smis = [s for s, _ in (top_list or [])][:max_mols]
    mols = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
    if not mols:
        return
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(250, 200), legends=["" for _ in mols])
    img.save(out_path)


def run_mode(name: str, seeds, cfg: OptConfig, out_dir: str) -> dict:
    ensure_dir(out_dir)
    # Attach CSV path
    csv_path = os.path.join(out_dir, f"{name}.csv")
    cfg.log_csv = csv_path
    # Save config
    with open(os.path.join(out_dir, f"{name}_config.json"), "w") as f:
        d = cfg.__dict__.copy()
        d["seeds"] = seeds
        json.dump(d, f, indent=2)
    t0 = time.time()
    out = run_optimization(seed_smiles=seeds, config=cfg)
    out["elapsed_sec"] = time.time() - t0
    # Save outputs
    with open(os.path.join(out_dir, f"{name}_history.json"), "w") as f:
        json.dump(out.get("history", {}), f, indent=2)
    with open(os.path.join(out_dir, f"{name}_top.json"), "w") as f:
        json.dump(out.get("top", []), f, indent=2)
    # Plots
    try:
        from baseline.plot_csv import main as plot_main
        argv_old = _sys.argv
        _sys.argv = [
            "plot_csv.py",
            "--csv",
            csv_path,
            "--out_prefix",
            os.path.join(out_dir, name),
        ]
        try:
            plot_main()
        finally:
            _sys.argv = argv_old
    except Exception:
        pass
    # Top grid
    try:
        save_top_grid(out.get("top", []), os.path.join(out_dir, f"{name}_top_grid.png"))
    except Exception:
        pass
    return out


def write_report(base_dir: str, diverse_dir: str, preserve_dir: str) -> None:
    # Build a simple HTML report
    html = [
        "<html><head><meta charset='utf-8'><title>SCALE — Compare Modes</title></head><body>",
        "<h1>SCALE — Scaffold-Diverse vs Scaffold-Preserving</h1>",
        f"<p>Output directory: {base_dir}</p>",
        "<h2>Diverse Mode</h2>",
        f"<img src='{os.path.relpath(os.path.join(diverse_dir, 'diverse_curves.png'), base_dir)}' width='480'>",
        f"<img src='{os.path.relpath(os.path.join(diverse_dir, 'diverse_qed_vs_sa.png'), base_dir)}' width='480'>",
        f"<img src='{os.path.relpath(os.path.join(diverse_dir, 'diverse_scaffolds.png'), base_dir)}' width='480'>",
        f"<img src='{os.path.relpath(os.path.join(diverse_dir, 'diverse_top_grid.png'), base_dir)}' width='480'>",
        "<h2>Preserving Mode</h2>",
        f"<img src='{os.path.relpath(os.path.join(preserve_dir, 'preserve_curves.png'), base_dir)}' width='480'>",
        f"<img src='{os.path.relpath(os.path.join(preserve_dir, 'preserve_qed_vs_sa.png'), base_dir)}' width='480'>",
        f"<img src='{os.path.relpath(os.path.join(preserve_dir, 'preserve_scaffolds.png'), base_dir)}' width='480'>",
        f"<img src='{os.path.relpath(os.path.join(preserve_dir, 'preserve_top_grid.png'), base_dir)}' width='480'>",
        "</body></html>",
    ]
    with open(os.path.join(base_dir, "index.html"), "w") as f:
        f.write("\n".join(html))


def main():
    ap = argparse.ArgumentParser(description="Run scaffold-diverse and scaffold-preserving modes side-by-side")
    ap.add_argument("--out_dir", default="runs/compare_modes", help="Base directory for outputs")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--diverse_seeds", default="c1ccncc1,COc1ccccc1,CC(=O)N,CCN")
    ap.add_argument("--preserve_seed", default="COc1ccccc1")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.out_dir, ts)
    diverse_dir = os.path.join(base_dir, "diverse")
    preserve_dir = os.path.join(base_dir, "preserve")
    ensure_dir(diverse_dir)
    ensure_dir(preserve_dir)

    diverse_seeds = [s.strip() for s in args.diverse_seeds.split(",") if s.strip()]
    preserve_seeds = [args.preserve_seed.strip()]

    # Diverse config
    diverse_cfg = OptConfig(
        objective="qed",
        rounds=args.rounds,
        init_train_size=128,
        candidates_per_round=1200,
        topk_per_round=80,
        k_exploration=1.2,
        lambda_strain=0.15,
        sa_soft_beta=0.15,
        mmff_prescreen_factor=4.0,
        diversity_penalty=0.30,
        scaffold_penalty_alpha=0.40,
        scaffold_cap_per_round=2,
        novelty_penalty_alpha=0.20,
        use_controller=True,
        audit_k=50,
        train_on_composite=True,
        use_physchem=True,
    )

    # Preserving config
    preserve_cfg = OptConfig(
        objective="qed",
        rounds=args.rounds,
        init_train_size=64,
        candidates_per_round=800,
        topk_per_round=80,
        k_exploration=1.0,
        lambda_strain=0.10,
        sa_soft_beta=0.10,
        mmff_prescreen_factor=4.0,
        diversity_penalty=0.0,
        scaffold_penalty_alpha=0.0,
        scaffold_cap_per_round=None,
        novelty_penalty_alpha=0.0,
        use_controller=True,
        audit_k=50,
        train_on_composite=True,
        preserve_seed_scaffold=True,
        use_physchem=True,
    )

    out_div = run_mode("diverse", diverse_seeds, diverse_cfg, diverse_dir)
    out_pre = run_mode("preserve", preserve_seeds, preserve_cfg, preserve_dir)

    write_report(base_dir, diverse_dir, preserve_dir)
    print(f"Report written to {os.path.join(base_dir, 'index.html')}")


if __name__ == "__main__":
    main()
