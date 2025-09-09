import argparse
import glob
import os
import subprocess
import sys


HERE = os.path.abspath(os.path.dirname(__file__))
PY = sys.executable or "python"


def run(cmd: list):
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, check=True)
    return res.returncode


def latest_dir(base: str) -> str:
    paths = sorted(glob.glob(os.path.join(base, "*")), key=os.path.getmtime, reverse=True)
    if not paths:
        raise RuntimeError(f"No subfolders found under {base}")
    return paths[0]


def main():
    ap = argparse.ArgumentParser(description="Make-like runner: compare modes, evaluate, plot, collect to docs")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--compare_out", default=os.path.join("runs", "compare_modes"))
    ap.add_argument("--docs_dst", default=os.path.join("docs", "figs", "latest"))
    ap.add_argument("--hit", type=float, default=0.9, help="Hit threshold for eff_score/qed")
    args = ap.parse_args()

    print("SCALE — Scaffold-Conscious Agent for Learning & Exploration")
    print("Compare modes → evaluate → collect for docs\n")

    # 1) Run compare_modes
    run([PY, os.path.join(HERE, "compare_modes.py"), "--rounds", str(args.rounds)])
    cmp_root = latest_dir(args.compare_out)
    diverse_csv = os.path.join(cmp_root, "diverse", "diverse.csv")
    preserve_csv = os.path.join(cmp_root, "preserve", "preserve.csv")

    # 2) Plot pred vs true (last round)
    run([PY, os.path.join(HERE, "plot_pred_vs_true.py"), "--csv", diverse_csv, "--round", "0", "--out", os.path.join(cmp_root, "diverse", "pred_vs_true.png")])
    run([PY, os.path.join(HERE, "plot_pred_vs_true.py"), "--csv", preserve_csv, "--round", "0", "--out", os.path.join(cmp_root, "preserve", "pred_vs_true.png")])

    # 3) Evaluate both (last round)
    run([PY, os.path.join(HERE, "evaluate_run.py"), "--csv", diverse_csv, "--out", os.path.join(cmp_root, "diverse", "eval"), "--round", "0", "--hit", str(args.hit)])
    run([PY, os.path.join(HERE, "evaluate_run.py"), "--csv", preserve_csv, "--out", os.path.join(cmp_root, "preserve", "eval"), "--round", "0", "--hit", str(args.hit)])

    # 4) Collect files into docs
    run([PY, os.path.join(HERE, "collect_for_docs.py"), "--src", cmp_root, "--dst", args.docs_dst])

    print("\nDone. Open this report and docs figs:")
    print(os.path.join(cmp_root, "index.html"))
    print(args.docs_dst)


if __name__ == "__main__":
    main()
