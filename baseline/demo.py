import argparse
import json
import os
import time
from datetime import datetime

# Ensure project root is on sys.path when running as a script
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


def main():
    ap = argparse.ArgumentParser(description="One-command demo runner with agentic controller")
    ap.add_argument("--preset", default="qed_sa", choices=["qed_sa", "penlogp"], help="Which preset to run")
    ap.add_argument("--out_dir", default="runs", help="Base directory for outputs")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seeds", default="", help="Comma-separated seed SMILES to override preset")
    ap.add_argument("--seed_file", default="", help="Path to a file with one SMILES per line (overrides preset)")
    ap.add_argument("--physchem", action="store_true", help="Enable physchem descriptors in surrogate features")
    args = ap.parse_args()

    print("SCALE — Scaffold-Conscious Agent for Learning & Exploration")
    print("Agentic, guardrailed, physics-aware molecular optimization\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"demo_{args.preset}_{ts}")
    ensure_dir(run_dir)

    # Presets
    if args.preset == "qed_sa":
        seeds = ["c1ccccc1", "c1ccncc1", "COc1ccccc1", "CC(=O)N", "CCN"]
        cfg = OptConfig(
            objective="qed",
            rounds=args.rounds,
            init_train_size=128,
            candidates_per_round=1200,
            topk_per_round=80,
            k_exploration=1.2,
            lambda_strain=0.1,
            mmff_prescreen_factor=4.0,
            diversity_penalty=0.25,
            sa_soft_beta=0.1,
            scaffold_penalty_alpha=0.3,
            log_csv=os.path.join(run_dir, "selections.csv"),
            use_controller=True,
            use_physchem=args.physchem,
            train_on_composite=True,
        )
    else:  # penlogp
        seeds = ["CC", "CCC", "CCO", "c1ccccc1"]
        cfg = OptConfig(
            objective="pen_logp",
            rounds=args.rounds,
            init_train_size=128,
            candidates_per_round=1200,
            topk_per_round=80,
            k_exploration=1.0,
            lambda_strain=0.2,
            mmff_prescreen_factor=4.0,
            diversity_penalty=0.2,
            sa_soft_beta=0.0,
            scaffold_penalty_alpha=0.2,
            log_csv=os.path.join(run_dir, "selections.csv"),
            use_controller=True,
            use_physchem=args.physchem,
            train_on_composite=False,
        )

    # Override seeds from CLI if provided
    cli_seeds = []
    if args.seeds:
        cli_seeds.extend([s.strip() for s in args.seeds.split(",") if s.strip()])
    if args.seed_file:
        try:
            with open(args.seed_file, "r") as sf:
                for line in sf:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    cli_seeds.append(line)
        except Exception:
            pass
    if cli_seeds:
        seeds = cli_seeds

    # Save config (+ seeds)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        cfg_dict = cfg.__dict__.copy()
        cfg_dict["seeds"] = seeds
        json.dump(cfg_dict, f, indent=2)

    t0 = time.time()
    out = run_optimization(seed_smiles=seeds, config=cfg)
    dt = time.time() - t0

    # Save history + decisions + top
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(out.get("history", {}), f, indent=2)
    with open(os.path.join(run_dir, "decisions.json"), "w") as f:
        json.dump(out.get("decisions", []), f, indent=2)
    with open(os.path.join(run_dir, "top.json"), "w") as f:
        json.dump(out.get("top", []), f, indent=2)

    # Export top grid image
    top = out.get("top", [])
    if isinstance(top, list) and top:
        smis = [s for s, _ in top[:36]]
        mols = []
        for s in smis:
            try:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    mols.append(m)
            except Exception:
                continue
        if mols:
            try:
                img = Draw.MolsToGridImage(mols, molsPerRow=6, subImgSize=(250, 200), legends=["" for _ in mols])
                img_path = os.path.join(run_dir, "top_grid.png")
                img.save(img_path)
            except Exception:
                pass

    # Generate plots from CSV if available
    try:
        from baseline.plot_csv import main as plot_main  # type: ignore

        # Mimic CLI
        import sys

        argv_old = sys.argv
        sys.argv = [
            "plot_csv.py",
            "--csv",
            os.path.join(run_dir, "selections.csv"),
            "--out_prefix",
            os.path.join(run_dir, "report"),
        ]
        try:
            plot_main()
        finally:
            sys.argv = argv_old
    except Exception:
        pass

    print(f"Demo finished in {dt:.1f}s. Outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
