# Slide Outline: Guardrailed, Physics-Aware Agentic Optimizer

## 1. Title & Pitch
- Guardrailed, physics-aware molecule design with an agentic controller
- One-liner: adaptive, auditable, fast demo that reaches high QED with realism filters

## 2. Problem & Constraints
- Need safe, synthesizable proposals; avoid assay interferers and unstable motifs
- Fast iteration under small budgets; reproducible decisions

## 3. Approach (System Diagram)
- Controller (LLM-ready) → chooses operators/knobs from a safe menu
- RDKit executor → BRICS/attach + sanitization and filters (PAINS/SA)
- Surrogate + physics → RF-ensemble UCB + MMFF strain/atom
- Diversity → Tanimoto + scaffold-usage penalties

## 4. Guardrails
- Drug-likeness limits; PAINS filters; optional SA hard gate
- Soft SA penalty in acquisition

## 5. Physics-Aware Scoring
- MMFF94 strain/atom as a cheap 3D penalty
- Prescreen + caching to keep rounds fast

## 6. Agentic Controller
- Observes progress; selects op (BRICS/attach) and knobs (k, λ, div, SA β, scaffold α)
- Deterministic heuristic today; drop-in LLM controller tomorrow (JSON menu)

## 7. Results (Screenshots)
- Learning curves: `report_curves.png`
- QED vs SA scatter: `report_qed_vs_sa.png`
- Scaffolds: `report_scaffolds.png`
- Top grid: `top_grid.png`

## 8. Ablations (Why it matters)
- − physics → unstable/high-strain artifacts, slower improvement
- − uncertainty → mode collapse, worse sample-efficiency
- − diversity → near-duplicate scaffolds
- − agent → slower escape from local optima

## 9. Live Demo
- `python baseline/demo.py --preset qed_sa --rounds 6`
- Show decisions.json updating each round

## 10. Limitations & Next Steps
- No retrosynthesis; QED/pen-logP only; MMFF-lite physics
- Add LLM controller with rationale; add multi-objective (QED − β·SA − γ·strain)
- Optional: route-based feasibility (ASKCOS/Retro*) if time allows

## 11. Team Roles
- SWE: controller & run manager; reporting/plots
- Comp Chem: seeds/fragment pools; thresholds; QC top molecules

