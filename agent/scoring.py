from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "qmugs_subset.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "results" / "ridge_model.joblib"
PROPERTY_COL = "property"


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "smiles" not in df.columns or PROPERTY_COL not in df.columns:
        raise ValueError("Dataset must have 'smiles' and 'property' columns")
    # If duplicates exist, take mean property per canonical SMILES
    df = df.copy()
    df["smiles"] = df["smiles"].astype(str)
    df = df.groupby("smiles", as_index=False)[PROPERTY_COL].mean()
    return df


def _mol_from_smiles(s: str):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m


# Prepare descriptors in a deterministic order
_DESC_FUNCS = sorted(Descriptors.descList, key=lambda x: x[0])
_DESC_NAMES = [name for name, _ in _DESC_FUNCS]


def _compute_descriptor_row(mol) -> List[float]:
    values: List[float] = []
    for _, func in _DESC_FUNCS:
        try:
            v = func(mol)
        except Exception:
            v = np.nan
        # Coerce to float when possible
        try:
            v = float(v)
        except Exception:
            v = np.nan
        values.append(v)
    return values


def _make_X_y(df: pd.DataFrame):
    rows = []
    y = []
    for _, row in df.iterrows():
        smi = row["smiles"]
        mol = _mol_from_smiles(smi)
        if mol is None:
            continue
        desc = _compute_descriptor_row(mol)
        if all(np.isfinite(desc)):
            rows.append(desc)
            y.append(float(row[PROPERTY_COL]))
    if not rows:
        raise RuntimeError("No valid molecules with finite descriptors found")
    X = pd.DataFrame(rows, columns=_DESC_NAMES)
    y = np.asarray(y, dtype=float)
    # drop zero-variance columns
    nunique = X.nunique(dropna=False)
    keep_cols = nunique[nunique > 1].index.tolist()
    X = X[keep_cols]
    return X, y, keep_cols


def score_knn(query: str, neighbors: list) -> float:
    """Score molecule based on weighted average of top neighbors' properties.

    neighbors: list of (smiles, similarity)
    If all weights are ~0 or properties missing, fall back to dataset mean.
    """
    df = _load_dataset()
    prop_map = dict(zip(df["smiles"], df[PROPERTY_COL]))

    weights = []
    props = []
    for smi, sim in neighbors:
        p = prop_map.get(smi, None)
        if p is None:
            continue
        w = float(sim) + 1e-6  # avoid zero weight
        weights.append(w)
        props.append(float(p))

    if not weights or np.isclose(sum(weights), 0.0):
        return float(df[PROPERTY_COL].mean())

    weights = np.asarray(weights)
    props = np.asarray(props)
    return float(np.average(props, weights=weights))


def train_ridge():
    """Train Ridge model on dataset using RDKit descriptors.

    Saves a joblib artifact containing (pipeline, feature_names) to results/.
    Returns the fitted pipeline.
    """
    df = _load_dataset()
    X, y, keep_cols = _make_X_y(df)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]
    )
    pipe.fit(X[keep_cols].values, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": keep_cols}, MODEL_PATH)
    return pipe


def _load_or_train_model():
    if MODEL_PATH.exists():
        obj = joblib.load(MODEL_PATH)
        return obj["pipeline"], obj["features"]
    pipe = train_ridge()
    obj = joblib.load(MODEL_PATH)
    return pipe, obj["features"]


def score_ridge(smiles: str) -> float:
    """Predict property using trained Ridge model.

    Trains on first use if no artifact exists.
    Returns dataset mean if the input is invalid or descriptors are non-finite.
    """
    pipe, feat = _load_or_train_model()
    mol = _mol_from_smiles(smiles)
    df = _load_dataset()
    fallback = float(df[PROPERTY_COL].mean())
    if mol is None:
        return fallback
    desc = _compute_descriptor_row(mol)
    X = pd.DataFrame([desc], columns=_DESC_NAMES)
    # If a feature used in training is missing or non-finite, fallback
    try:
        x_row = X[feat].values
        if not np.all(np.isfinite(x_row)):
            return fallback
    except KeyError:
        return fallback
    try:
        pred = float(pipe.predict(x_row)[0])
    except Exception:
        pred = fallback
    return pred


if __name__ == "__main__":
    # quick smoke test
    train_ridge()
    print("score_ridge('CCO'):", score_ridge("CCO"))

