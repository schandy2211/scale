from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Observation:
    round_index: int
    rounds_total: int
    train_size: int
    best: float
    avg: float
    last_best: float
    last_avg: float


@dataclass
class Action:
    op: str = "brics"  # "brics" | "attach"
    cands: Optional[int] = None
    topk: Optional[int] = None
    k: Optional[float] = None
    lam: Optional[float] = None
    div: Optional[float] = None
    sa_beta: Optional[float] = None
    scaf_alpha: Optional[float] = None


class HeuristicController:
    """Simple, deterministic controller that nudges knobs based on progress.

    Rules:
    - If best did not improve this round → increase exploration (k + 0.2), increase diversity (div + 0.05), reduce strain penalty (lam * 0.9).
    - If avg improved strongly (> 0.01) → keep knobs steady.
    - Alternate op between BRICS and attach if stagnation persists (every 2 rounds of no best improvement).
    """

    def __init__(self):
        self._no_improve_streak = 0

    def decide(self, obs: Observation) -> Action:
        improved = obs.best > obs.last_best + 1e-6
        if improved:
            self._no_improve_streak = 0
        else:
            self._no_improve_streak += 1

        act = Action(op="brics")
        if not improved:
            act.k = 1.2  # more exploration
            act.div = 0.25
            act.lam = 0.9  # interpreted as scale later
            # switch operator if stagnating for 2+ rounds
            if self._no_improve_streak >= 2:
                act.op = "attach"
        else:
            # gentle defaults
            act.k = 1.0
            act.div = 0.2
            act.lam = 1.0

        return act

