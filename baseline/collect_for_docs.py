import argparse
import os
import shutil


def maybe_copy(src: str, dst: str):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Collect key images from a compare_modes run into docs")
    ap.add_argument("--src", required=True, help="Source directory (e.g., runs/compare_modes/<timestamp>)")
    ap.add_argument("--dst", default="docs/figs/latest", help="Destination directory for slides")
    args = ap.parse_args()

    diverse_dir = os.path.join(args.src, "diverse")
    preserve_dir = os.path.join(args.src, "preserve")

    pairs = [
        (os.path.join(diverse_dir, "diverse_curves.png"), os.path.join(args.dst, "diverse_curves.png")),
        (os.path.join(diverse_dir, "diverse_scaffolds.png"), os.path.join(args.dst, "diverse_scaffolds.png")),
        (os.path.join(diverse_dir, "pred_vs_true.png"), os.path.join(args.dst, "diverse_pred_vs_true.png")),
        (os.path.join(diverse_dir, "diverse_top_grid.png"), os.path.join(args.dst, "diverse_top_grid.png")),
        (os.path.join(preserve_dir, "preserve_curves.png"), os.path.join(args.dst, "preserve_curves.png")),
        (os.path.join(preserve_dir, "preserve_scaffolds.png"), os.path.join(args.dst, "preserve_scaffolds.png")),
        (os.path.join(preserve_dir, "pred_vs_true.png"), os.path.join(args.dst, "preserve_pred_vs_true.png")),
        (os.path.join(preserve_dir, "preserve_top_grid.png"), os.path.join(args.dst, "preserve_top_grid.png")),
        (os.path.join(args.src, "index.html"), os.path.join(args.dst, "compare_modes_index.html")),
    ]

    copied = 0
    for src, dst in pairs:
        if maybe_copy(src, dst):
            copied += 1
            print(f"copied: {src} -> {dst}")
        else:
            print(f"missing: {src}")

    print(f"\nDone. Copied {copied} files to {args.dst}")


if __name__ == "__main__":
    main()

