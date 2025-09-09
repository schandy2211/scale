from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "qmugs_subset.csv"


def _mol_from_smiles(s: str):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m


def _fp_ecfp4(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


@lru_cache(maxsize=1)
def _load_index():
    df = pd.read_csv(DATA_PATH)
    if "smiles" not in df.columns:
        raise ValueError("Dataset must contain a 'smiles' column")
    # drop invalid
    rows = []
    fps = []
    for _, row in df.iterrows():
        smi = str(row["smiles"])  # type: ignore[index]
        mol = _mol_from_smiles(smi)
        if mol is None:
            continue
        fp = _fp_ecfp4(mol)
        rows.append(row)
        fps.append(fp)
    clean_df = pd.DataFrame(rows).reset_index(drop=True)
    return clean_df, fps


def top_k_neighbors(query_smiles: str, k: int = 5) -> list[tuple[str, float]]:
    """Return top-k similar SMILES from qmugs_subset.csv using Tanimoto.

    Returns a list of (smiles, similarity) sorted by decreasing similarity.
    """
    df, fps = _load_index()
    query_mol = _mol_from_smiles(query_smiles)
    if query_mol is None:
        return []
    qfp = _fp_ecfp4(query_mol)

    sims = [DataStructs.TanimotoSimilarity(qfp, fp) for fp in fps]
    # attach and sort
    tmp = list(zip(df["smiles"].tolist(), sims))
    # drop exact identical strings if present
    tmp = [t for t in tmp if t[0] != query_smiles]
    tmp.sort(key=lambda x: x[1], reverse=True)
    return tmp[: min(k, len(tmp))]


if __name__ == "__main__":
    print(top_k_neighbors("CCO", k=5))

