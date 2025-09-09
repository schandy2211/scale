from __future__ import annotations

from typing import Tuple
from rdkit import Chem


def _canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def propose_smiles(smiles: str) -> tuple[str, str]:
    """Takes a SMILES and returns a modified SMILES and a reason.

    Strategy (rule-based mock):
    1) Try adding a hydroxyl ("O") at the end: e.g., CC -> CCO
    2) Try increasing polarity by introducing an amine: first 'C' -> 'N'
    3) Try halogenation ("Cl") at the end
    Falls back to original if no valid modification is found.
    """

    orig_canonical = _canonicalize(smiles)
    if orig_canonical is None:
        # Invalid input; return a simple valid fallback and reason
        return (
            "CCO",
            "Input invalid; returning ethanol as a valid fallback with higher polarity",
        )

    candidates: list[tuple[str, str]] = [
        (smiles + "O", "Added -OH to increase polarity"),
        (smiles.replace("C", "N", 1), "Introduced amine to increase polarity"),
        (smiles + "Cl", "Added chlorine to modulate lipophilicity"),
    ]

    for cand, reason in candidates:
        can = _canonicalize(cand)
        if can is None:
            continue
        if can != orig_canonical:
            return can, reason

    # Fallback: no change possible
    return orig_canonical, "No safe modification found; returning original"


if __name__ == "__main__":
    s, r = propose_smiles("CC")
    print("Original SMILES: CC")
    print("Modified SMILES:", s)
    print("Reason:", r)

