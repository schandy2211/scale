## SCALE: Scaffold‑Conscious Agent for Learning & Exploration

SCALE is a fast, reproducible, and agentic molecular optimizer that designs small molecules under guardrails. It supports two application modes:

- Scaffold‑diverse (hit finding, scaffold hopping)
- Scaffold‑preserving (lead optimization around a known core)

The optimizer uses RDKit for chemistry, a Random‑Forest ensemble as a surrogate, lightweight physics (MMFF strain), and diversity/scaffold constraints. An agentic controller adapts edit operators and knobs per round. Evaluation scripts produce plots and summaries for slides/reports.

### Key Ideas
- Guardrails: RDKit sanitize, drug‑likeness limits, PAINS filter; optional SA gate.
- Physics‑aware: penalize MMFF94 strain/atom; prescreen + caching for speed.
- Agentic: per‑round decisions (operator, exploration/diversity/penalties), logged.
- Diversity: Tanimoto and scaffold‑usage penalties, scaffold cap; optional novelty vs seeds.
- Realism objective: optimize QED or composite eff = QED − β·SA − γ·max(strain,0).
- Validation: prediction metrics (MAE/RMSE/R²/Spearman), acceptance, novelty/diversity.

## Install
RDKit is easiest via Conda.

Option A — Conda (recommended):
```bash
conda env create -f environment.yml
conda activate llm-agent-chem
```

Option B — Pip (if you have rdkit wheels):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## One‑Command Demo (Agentic)
Runs QED optimization with physics, diversity, and agent decisions, then saves a report folder with plots and a molecule grid.

```bash
python baseline/demo.py --preset qed_sa --rounds 6 --physchem
```

Outputs under `runs/demo_qed_sa_YYYYMMDD_HHMMSS/`:
- `config.json`, `history.json`, `decisions.json`, `top.json`
- `selections.csv` (per‑round selections + metrics)
- Plots: `report_curves.png`, `report_qed_vs_sa.png`, `report_scaffolds.png`
- Grid: `top_grid.png`

## Compare Modes (Hit Finding vs Lead Optimization)
Run both scaffold‑diverse and scaffold‑preserving modes, evaluate, and collect figures for slides in one go:

```bash
python baseline/make_latest.py --rounds 6
# Open the report
open "$(ls -td runs/compare_modes/* | head -1)/index.html"
# Slide‑ready figures are copied to
open docs/figs/latest
```

## Power‑User CLI
Direct optimizer with JSON exports and physchem features:

```bash
python baseline/baseline_opt.py \
  --seed "c1ccncc1,COc1ccccc1,CC(=O)N,CCN" \
  --objective qed --rounds 6 --init 128 --cands 1200 --topk 80 \
  --k 1.2 --lam 0.15 --sa_beta 0.15 --pre_factor 4 \
  --div 0.30 --scaf_alpha 0.40 --scaffold_cap 2 \
  --agent --physchem \
  --csv runs/qed_run.csv \
  --history_json runs/qed_history.json \
  --decisions_json runs/qed_decisions.json \
  --top_json runs/qed_top.json
```

Scaffold‑preserving example (lead optimization around a single core):
```bash
python baseline/baseline_opt.py --seed "COc1ccccc1" --objective qed --rounds 6 \
  --init 64 --cands 800 --topk 80 --k 1.0 --lam 0.10 --sa_beta 0.10 \
  --pre_factor 4 --div 0.0 --scaf_alpha 0.0 --scaffold_cap -1 \
  --preserve_scaffold --agent --physchem --csv runs/preserve.csv
```

## Evaluation & Plots
Per‑run plots from CSV:
```bash
python baseline/plot_csv.py --csv runs/qed_run.csv --out_prefix runs/qed_run
```

Prediction sanity (True vs Predicted, last round):
```bash
python baseline/plot_pred_vs_true.py --csv runs/qed_run.csv --round 0 --out runs/qed_pred_vs_true.png
```

Summarize final round metrics (acceptance/diversity/prediction):
```bash
python baseline/summarize_csv.py runs/qed_run.csv
```

End‑to‑end evaluation with hit stats and precision@k (writes metrics.json + plots):
```bash
python baseline/evaluate_run.py --csv runs/qed_run.csv --out runs/eval --round 0 --hit 0.9
```

## Repository Map
- `baseline/baseline_opt.py` — Core optimizer (generator, filters, surrogate, physics, selection, agent integration, logging)
- `agent/controller.py` — Heuristic agent (decides op/knobs per round; LLM‑ready interface)
- `baseline/demo.py` — One‑command agentic demo
- `baseline/compare_modes.py` — Run scaffold‑diverse vs scaffold‑preserving, build report
- `baseline/make_latest.py` — Make‑like runner: compare, evaluate, plot, collect for docs
- `baseline/ablate.py` — Ablations (no physics/uncertainty/diversity/agent) with overlays
- `baseline/plot_csv.py` — Learning curves, QED–SA scatter, scaffold bars
- `baseline/plot_pred_vs_true.py` — True vs predicted scatter (per round)
- `baseline/evaluate_run.py` — Hit stats (+ precision@k) and histograms
- `baseline/summarize_csv.py` — Print last round summary metrics
- `baseline/collect_for_docs.py` — Copy key images into `docs/figs/latest`

## What “Good” Looks Like
- Hit finding (diverse):
  - High unique scaffolds/round, high mean pairwise distance
  - Best score rises quickly; acceptance stable
  - Precision@k (true ≥ threshold) well above baseline
- Lead optimization (preserving):
  - Best/avg improve over the seed core; fewer scaffolds per round
  - Pred vs true scatter tight at the top (reliable ranking of best analogs)

## Notes & Limits
- RDKit SA and PAINS are heuristics; they don’t guarantee synthesizability or safety.
- MMFF is a cheap physics signal, not docking/QM; use as a realism hint.
- No biological activity model in this repo; plug in docking/ML activity if desired.

## Roadmap
- Optional LLM controller (JSON menu) with strict schema + fallback
- Route‑based feasibility (retro planning) for stronger synth claims
- Multi‑objective presets (QED − β·SA − γ·strain) – already supported via flags

## Latest Run (Artifacts for Slides)
- Latest figures folder: `docs/figs/latest`
- Compare report (copied): `docs/figs/latest/compare_modes_index.html`

[![Latest Report](https://img.shields.io/badge/SCALE-latest_report-blue)](docs/figs/latest/compare_modes_index.html)


## Baseline Agentic Demo (New)
- One-command demo with agentic controller and physics-aware acquisition:
  - `python baseline/demo.py --preset qed_sa --rounds 6`
- Customize seeds:
  - Inline: `python baseline/demo.py --preset qed_sa --rounds 6 --seeds "c1ccccc1,CCN,CC(=O)N,COc1ccccc1"`
  - File: `python baseline/demo.py --preset qed_sa --rounds 6 --seed_file seeds.txt`
- Artifacts (under `runs/demo_qed_sa_YYYYMMDD_HHMMSS/`):
  - `config.json` — run configuration including seeds
  - `history.json` — best/avg/n_train per round
  - `decisions.json` — per-round agent choices (op, k, lam, div, SA beta, scaffold alpha)
  - `top.json` — final top SMILES with objective values
  - `selections.csv` — per-round selections (QED/SA/strain/scaffold/score)
  - `report_curves.png` — learning curves
  - `report_qed_vs_sa.png` — QED vs SA scatter
  - `report_scaffolds.png` — top scaffold counts
  - `top_grid.png` — grid image of final top molecules

### Power-User CLI
- Direct agentic run with JSON exports:
  - `python baseline/baseline_opt.py --seed "c1ccccc1,c1ccncc1,COc1ccccc1,CC(=O)N,CCN" --objective qed --rounds 6 --init 128 --cands 1200 --topk 80 --k 1.2 --lam 0.1 --pre_factor 4 --div 0.25 --sa_beta 0.1 --scaf_alpha 0.3 --agent --csv runs/qed_run.csv --history_json runs/qed_history.json --decisions_json runs/qed_decisions.json --top_json runs/qed_top.json`
- Plot from CSV:
  - `python baseline/plot_csv.py --csv runs/qed_run.csv --out_prefix runs/qed_run`

### What’s Under The Hood
- Generator: BRICS recombination with a fragment-attach fallback.
- Guardrails: sanitization, drug-likeness, PAINS, optional SA gate and soft penalty.
- Scoring: RF ensemble UCB + MMFF strain/atom (prescreen + cache).
- Diversity: Tanimoto penalty and scaffold-usage penalty.
- Agent: heuristic controller adapts operator and knobs per round; logs decisions.
