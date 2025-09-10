import math
import random
import time
import os
import csv
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", message=".*GetValence.*", category=DeprecationWarning)

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, rdMolDescriptors, Descriptors, Lipinski, QED
from rdkit import DataStructs
from rdkit.Chem import FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold

# Ensure project root is on sys.path when running as a script
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from agent.controller import HeuristicController, Observation, Action
from agent.llm_controller import LLMController
try:
    # Preferred modern fingerprint API (avoids deprecation warnings)
    from rdkit.Chem import rdFingerprintGenerator as _rdFG  # type: ignore
except Exception:  # pragma: no cover
    _rdFG = None  # type: ignore


# ------------------------
# Basic chemistry helpers
# ------------------------

# ------------------------
# Simple metrics helpers
# ------------------------
def _safe_mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _r2_score(y_true: List[float], y_pred: List[float]) -> Optional[float]:
    if not y_true or len(y_true) != len(y_pred):
        return None
    mean_y = _safe_mean(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    if ss_tot <= 1e-12:
        return None
    return 1.0 - ss_res / ss_tot


def _mae(y_true: List[float], y_pred: List[float]) -> Optional[float]:
    if not y_true or len(y_true) != len(y_pred):
        return None
    return _safe_mean([abs(a - b) for a, b in zip(y_true, y_pred)])


def _rmse(y_true: List[float], y_pred: List[float]) -> Optional[float]:
    if not y_true or len(y_true) != len(y_pred):
        return None
    mse = _safe_mean([(a - b) ** 2 for a, b in zip(y_true, y_pred)])
    return math.sqrt(mse)


def _rank(values: List[float]) -> List[float]:
    pairs = list(enumerate(values))
    pairs.sort(key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n == 0 or n != len(y):
        return None
    mx = _safe_mean(x)
    my = _safe_mean(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def _spearman(y_true: List[float], y_pred: List[float]) -> Optional[float]:
    if not y_true or len(y_true) != len(y_pred):
        return None
    r1 = _rank(y_true)
    r2 = _rank(y_pred)
    return _pearson(r1, r2)

def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=True)


_MORGAN_GEN_CACHE: Dict[Tuple[int, int], object] = {}


def _get_morgan_generator(radius: int, fp_size: int):
    if _rdFG is None:
        return None
    key = (radius, fp_size)
    gen = _MORGAN_GEN_CACHE.get(key)
    if gen is None:
        try:
            gen = _rdFG.GetMorganGenerator(radius=radius, fpSize=fp_size)
            _MORGAN_GEN_CACHE[key] = gen
        except Exception:
            return None
    return gen


def morgan_fp_bits(mol: Chem.Mol, radius: int = 2, nbits: int = 2048) -> List[int]:
    gen = _get_morgan_generator(radius, nbits)
    if gen is not None:
        bv = gen.GetFingerprint(mol)
    else:
        # Fallback for older RDKit versions
        bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    # Convert to list of ints (0/1)
    return [int(x) for x in bv.ToBitString()]


# Optional SA score support (off by default)
_USE_SA: bool = False
_SA_MAX: float = 6.0
_SA_AVAILABLE: bool = False
try:
    # RDKit contrib SA scorer (may not be installed in all envs)
    from rdkit.Chem import rdMolDescriptors as _rdmd  # noqa: F401 (ensure RDKit is present)
    from rdkit.Chem import MolFromSmiles as _  # noqa: F401
    # Some distributions package sascorer under rdkit.Chem (e.g., via pip install sascorer)
    try:
        from rdkit.Chem import SA_Score as _SA  # type: ignore
        _SA_AVAILABLE = hasattr(_SA, "sascorer") and hasattr(_SA.sascorer, "calculateScore")
        _SA_CALC = _SA.sascorer.calculateScore if _SA_AVAILABLE else None
    except Exception:
        try:
            from rdkit.Chem import sascorer as _SA  # type: ignore
            _SA_AVAILABLE = hasattr(_SA, "calculateScore")
            _SA_CALC = _SA.calculateScore if _SA_AVAILABLE else None
        except Exception:
            _SA_AVAILABLE = False
            _SA_CALC = None
except Exception:
    _SA_AVAILABLE = False
    _SA_CALC = None


def set_sa_constraints(use_sa: bool, sa_max: float = 6.0) -> None:
    global _USE_SA, _SA_MAX
    _USE_SA = bool(use_sa) and _SA_AVAILABLE
    _SA_MAX = float(sa_max)


def sa_value(mol: Chem.Mol) -> Optional[float]:
    if _SA_AVAILABLE and _SA_CALC is not None:
        try:
            return float(_SA_CALC(mol))
        except Exception:
            return None
    return None


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    except Exception:
        return ""


# ------------------------
# Physchem feature helpers
# ------------------------

def physchem_features(mol: Chem.Mol) -> List[float]:
    """Robust set of simple RDKit physchem descriptors.
    Returns a fixed-length list of floats.
    """
    try:
        mw = float(Descriptors.MolWt(mol))
    except Exception:
        mw = 0.0
    try:
        logp = float(Descriptors.MolLogP(mol))
    except Exception:
        logp = 0.0
    try:
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    except Exception:
        tpsa = 0.0
    try:
        hba = float(Lipinski.NumHAcceptors(mol))
    except Exception:
        hba = 0.0
    try:
        hbd = float(Lipinski.NumHDonors(mol))
    except Exception:
        hbd = 0.0
    try:
        rotb = float(Lipinski.NumRotatableBonds(mol))
    except Exception:
        rotb = 0.0
    try:
        frac_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    except Exception:
        frac_csp3 = 0.0
    try:
        aro_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    except Exception:
        aro_rings = 0.0
    try:
        aliph_rings = float(rdMolDescriptors.CalcNumAliphaticRings(mol))
    except Exception:
        aliph_rings = 0.0
    try:
        hetero = float(rdMolDescriptors.CalcNumHeteroatoms(mol))
    except Exception:
        hetero = 0.0
    heavy = float(mol.GetNumHeavyAtoms() or 0)
    return [
        mw,
        logp,
        tpsa,
        hba,
        hbd,
        rotb,
        frac_csp3,
        aro_rings,
        aliph_rings,
        hetero,
        heavy,
    ]


# ------------------------
# Simple property objectives
# ------------------------

def objective_qed(mol: Chem.Mol) -> float:
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def _largest_ring_size(mol: Chem.Mol) -> int:
    try:
        sssr = Chem.GetSymmSSSR(mol)
        if not sssr:
            return 0
        return max(len(r) for r in sssr)
    except Exception:
        return 0


def objective_penalized_logp_simple(mol: Chem.Mol) -> float:
    """
    A lightweight variant of penalized logP that avoids SA dependency:
    score = logP - ring_penalty
    where ring_penalty = max(0, largest_ring_size - 6)
    Note: This is a simplification. Use classic penalized logP (logP - SA - cycle penalty)
    if SA scoring is available in your environment.
    """
    try:
        logp = Descriptors.MolLogP(mol)
    except Exception:
        logp = 0.0
    ring_penalty = max(0, _largest_ring_size(mol) - 6)
    return float(logp - ring_penalty)


# ------------------------
# Filters / constraints
# ------------------------

def passes_druglikeness(mol: Chem.Mol) -> bool:
    try:
        mw = Descriptors.MolWt(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotb = Lipinski.NumRotatableBonds(mol)
        if mw >= 500:
            return False
        if hbd > 5 or hba > 10:
            return False
        if tpsa > 140:
            return False
        if rotb > 10:
            return False
        return True
    except Exception:
        return False


def sanitize_and_filter(mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if not passes_druglikeness(mol):
        return None
    if not passes_pains(mol):
        return None
    if _USE_SA and _SA_AVAILABLE and _SA_CALC is not None:
        try:
            sa = float(_SA_CALC(mol))
            if sa > _SA_MAX:
                return None
        except Exception:
            pass
    return mol


# ------------------------
# Simple fragment-attach fallback generator
# ------------------------

_FALLBACK_FRAGS = [
    "[*:1]C",              # methyl
    "[*:1]CC",             # ethyl
    "[*:1]O",              # hydroxyl
    "[*:1]OC",             # methoxy
    "[*:1]N",              # amino
    "[*:1]F",              # fluoro
    "[*:1]Cl",             # chloro
    "[*:1]c1ccccc1",       # phenyl
    "[*:1]c1ccncc1",       # pyridine
    "[*:1]c1ncccc1",       # pyridazine-like
    "[*:1]c1cnccc1",       # pyrimidine-like
    "[*:1]c1cc(OC)ccc1",   # anisole
    "[*:1]c1ccc(F)cc1",    # fluoro-phenyl
    "[*:1]c1ccc(Cl)cc1",   # chloro-phenyl
    "[*:1]C(F)(F)F",       # trifluoromethyl
    "[*:1]C#N",            # cyano
    "[*:1]S(=O)(=O)N",     # sulfonamide (primary)
    "[*:1]N(C)C",          # dimethylamino
    "[*:1]C(=O)N",         # amide
]


def _attach_fragment_once(base: Chem.Mol, frag: Chem.Mol, atom_idx: int) -> Optional[Chem.Mol]:
    # Find dummy attachment in fragment
    dummies = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummies:
        return None
    d_idx = dummies[0]
    nbrs = list(frag.GetAtomWithIdx(d_idx).GetNeighbors())
    if len(nbrs) != 1:
        return None
    n_idx = nbrs[0].GetIdx()

    try:
        combo = Chem.CombineMols(base, frag)
        base_offset = 0
        frag_offset = base.GetNumAtoms()
        em = Chem.EditableMol(combo)
        # Add bond base_atom <-> frag neighbor atom (not the dummy)
        em.AddBond(atom_idx + base_offset, n_idx + frag_offset, order=Chem.rdchem.BondType.SINGLE)
        # Remove the dummy atom (index after combine)
        em.RemoveAtom(d_idx + frag_offset)
        new_mol = em.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol
    except Exception:
        return None


def simple_attach_candidates(
    seed_mols: Sequence[Chem.Mol],
    n_candidates: int = 500,
    random_seed: int = 0,
) -> List[Chem.Mol]:
    random.seed(random_seed)
    frags = [mol_from_smiles(s) for s in _FALLBACK_FRAGS]
    frags = [f for f in frags if f is not None]
    built: List[Chem.Mol] = []
    seen: set = set()
    for m in seed_mols:
        atoms = [
            a.GetIdx()
            for a in m.GetAtoms()
            if a.GetAtomicNum() in (6, 7) and a.GetImplicitValence() > 0
        ]
        if not atoms:
            continue
        for _ in range(max(10, n_candidates // max(1, len(seed_mols)))):
            aidx = random.choice(atoms)
            frag = random.choice(frags)
            cand = _attach_fragment_once(m, frag, aidx)
            cand = sanitize_and_filter(cand)
            if cand is None:
                continue
            smi = canonical_smiles(cand)
            if smi in seen:
                continue
            seen.add(smi)
            built.append(cand)
            if len(built) >= n_candidates:
                return built
    return built


# ------------------------
# BRICS candidate generator (with fallback)
# ------------------------

def brics_candidates(
    seed_mols: Sequence[Chem.Mol],
    n_candidates: int = 500,
    random_seed: int = 0,
) -> List[Chem.Mol]:
    random.seed(random_seed)
    frags = set()
    for m in seed_mols:
        try:
            # Use molecule fragments for better compatibility
            frags |= set(BRICS.BRICSDecompose(m, returnMols=True))
        except Exception:
            continue
    # If seeds are too simple, augment with a tiny fragment library
    if not frags:
        lib_smiles = [
            "c1ccccc1",  # benzene
            "CCO",       # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccncc1",  # pyridine
            "CCN",      # ethylamine
            "OCCO",     # ethylene glycol
            "c1cc(Cl)ccc1",  # chlorobenzene
        ]
        lib_mols = [mol_from_smiles(s) for s in lib_smiles]
        lib_mols = [m for m in lib_mols if m is not None]
        for m in lib_mols:
            try:
                frags |= set(BRICS.BRICSDecompose(m, returnMols=True))
            except Exception:
                continue
    if not frags:
        # Fall back entirely to simple attachment
        return simple_attach_candidates(seed_mols, n_candidates=n_candidates, random_seed=random_seed)
    # Build generator; iterate until we collect enough valid candidates
    built: List[Chem.Mol] = []
    try:
        gen = BRICS.BRICSBuild(frags)
    except Exception:
        # Some versions expect a list, not a set
        gen = BRICS.BRICSBuild(list(frags))
    seen: set = set()
    for cand in gen:
        # RDKit versions differ: BRICSBuild may yield Mol or SMILES
        if isinstance(cand, Chem.Mol):
            mol = cand
        else:
            try:
                smi_cand = str(cand)
            except Exception:
                continue
            mol = mol_from_smiles(smi_cand)
        mol = sanitize_and_filter(mol)
        if mol is None:
            continue
        smi = canonical_smiles(mol)
        if smi in seen:
            continue
        seen.add(smi)
        built.append(mol)
        if len(built) >= n_candidates:
            break
    # If BRICS produced too few, augment with simple attachments
    if len(built) < n_candidates // 4:
        extra = simple_attach_candidates(seed_mols, n_candidates=n_candidates - len(built), random_seed=random_seed + 7)
        built.extend(extra)
    return built


# ------------------------
# Lightweight physics: MMFF strain per atom
# ------------------------

def mmff_strain_per_atom(mol: Chem.Mol, max_attempts: int = 2) -> float:
    """Return MMFF94 energy per heavy atom. If failure, return a modest penalty."""
    try:
        molH = Chem.AddHs(mol)
    except Exception:
        return 1.0
    best_e = None
    for attempt in range(max_attempts):
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 17 + attempt
            if AllChem.EmbedMolecule(molH, params) != 0:
                continue
            if AllChem.MMFFOptimizeMolecule(molH) != 0:
                # Non-zero means it may not have fully converged; still try to get energy
                pass
            props = AllChem.MMFFGetMoleculeProperties(molH)
            if props is None:
                continue
            ff = AllChem.MMFFGetMoleculeForceField(molH, props)
            if ff is None:
                continue
            e = ff.CalcEnergy()
            if best_e is None or e < best_e:
                best_e = e
        except Exception:
            continue
    if best_e is None:
        return 1.0
    heavy = mol.GetNumHeavyAtoms() or 1
    return float(best_e / heavy)


# Cache for MMFF strain per SMILES to avoid recomputation across rounds
_strain_cache: Dict[str, float] = {}


def _get_strain_cached(mol: Chem.Mol) -> float:
    smi = canonical_smiles(mol)
    val = _strain_cache.get(smi)
    if val is not None:
        return val
    val = mmff_strain_per_atom(mol)
    _strain_cache[smi] = val
    return val


# ------------------------
# RF-ensemble surrogate with UCB acquisition
# ------------------------

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception as e:  # pragma: no cover
    RandomForestRegressor = None  # type: ignore


def featurize_mols(
    mols: Sequence[Chem.Mol], radius: int = 2, nbits: int = 2048, include_physchem: bool = False
) -> List[List[float]]:
    feats: List[List[float]] = []
    for m in mols:
        bits = morgan_fp_bits(m, radius, nbits)
        if include_physchem:
            pc = physchem_features(m)
            feats.append([float(b) for b in bits] + pc)
        else:
            feats.append([float(b) for b in bits])
    return feats


# Morgan bit vector for Tanimoto similarity (cached)
_bv_cache: Dict[Tuple[str, int, int], DataStructs.ExplicitBitVect] = {}


def morgan_bv(mol: Chem.Mol, radius: int = 2, nbits: int = 2048) -> DataStructs.ExplicitBitVect:
    smi = canonical_smiles(mol)
    key = (smi, radius, nbits)
    bv = _bv_cache.get(key)
    if bv is not None:
        return bv
    gen = _get_morgan_generator(radius, nbits)
    if gen is not None:
        bv = gen.GetFingerprint(mol)
    else:
        bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    _bv_cache[key] = bv
    return bv


def train_rf_ensemble(
    X: List[List[float]],
    y: List[float],
    n_models: int = 5,
    random_seed: int = 0,
) -> List[RandomForestRegressor]:
    if RandomForestRegressor is None:
        raise RuntimeError("scikit-learn is required for the RF ensemble")
    models: List[RandomForestRegressor] = []
    for i in range(n_models):
        rs = random_seed + i
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            random_state=rs,
            n_jobs=-1,
        )
        model.fit(X, y)
        models.append(model)
    return models


def ensemble_predict(
    models: Sequence[RandomForestRegressor], X: List[List[float]]
) -> Tuple[List[float], List[float]]:
    preds = [m.predict(X) for m in models]
    # transpose
    means: List[float] = []
    stds: List[float] = []
    for i in range(len(X)):
        vals = [p[i] for p in preds]
        mu = float(sum(vals) / len(vals))
        var = float(sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1))
        means.append(mu)
        stds.append(math.sqrt(max(0.0, var)))
    return means, stds


# ------------------------
# Optimization loop
# ------------------------

ObjectiveFn = Callable[[Chem.Mol], float]


@dataclass
class OptConfig:
    objective: str = "qed"  # "qed" | "pen_logp"
    rounds: int = 8
    init_train_size: int = 64
    candidates_per_round: int = 800
    topk_per_round: int = 80
    k_exploration: float = 0.7
    lambda_strain: float = 0.2
    radius: int = 2
    nbits: int = 2048
    ensemble_size: int = 5
    random_seed: int = 0
    mmff_prescreen_factor: float = 5.0  # compute MMFF on top factor*topk by UCB (no-strain)
    diversity_penalty: float = 0.2      # subtract alpha * max Tanimoto to already-picked in round
    sa_soft_beta: float = 0.0           # subtract beta * SA score (soft penalty)
    scaffold_penalty_alpha: float = 0.0 # subtract alpha * scaffold-usage (normalized)
    log_csv: Optional[str] = None       # write per-round selections to CSV if set
    use_controller: bool = False        # enable agentic controller
    use_llm_controller: bool = False    # use LLM-based controller instead of heuristic
    scaffold_cap_per_round: Optional[int] = None  # hard cap per Murcko scaffold among selected
    # Composite objective training (align model to realism):
    # if True, train RF on eff = QED - sa_soft_beta*SA - lambda_strain*strain
    train_on_composite: bool = False
    # Novelty vs seeds penalty during selection (subtract alpha * max_sim_to_seed)
    novelty_penalty_alpha: float = 0.0
    audit_k: int = 0
    preserve_seed_scaffold: bool = False
    use_physchem: bool = False


def get_objective(name: str) -> ObjectiveFn:
    name = name.lower()
    if name in ("qed",):
        return objective_qed
    elif name in ("pen_logp", "penalized_logp", "pen-logp"):
        return objective_penalized_logp_simple
    else:
        raise ValueError(f"Unknown objective: {name}")


def bootstrap_training_set(
    seed_mols: Sequence[Chem.Mol],
    obj_fn: ObjectiveFn,
    init_train_size: int,
    random_seed: int,
    radius: int,
    nbits: int,
    include_physchem: bool,
) -> Tuple[List[Chem.Mol], List[List[float]], List[float]]:
    # Use BRICS to expand initial pool and score true objective
    pool = list(seed_mols)
    if len(pool) < init_train_size:
        more = brics_candidates(seed_mols, n_candidates=init_train_size * 5, random_seed=random_seed)
        # Sample unique molecules to reach init size
        seen = {canonical_smiles(m) for m in pool}
        for m in more:
            smi = canonical_smiles(m)
            if smi in seen:
                continue
            pool.append(m)
            seen.add(smi)
            if len(pool) >= init_train_size:
                break
    # Score true objective
    y = [obj_fn(m) for m in pool]
    X = featurize_mols(pool, radius, nbits, include_physchem=include_physchem)
    return pool, X, y


def ucb_scores(
    mu: Sequence[float],
    sigma: Sequence[float],
    strain: Sequence[float],
    k_exploration: float,
    lambda_strain: float,
) -> List[float]:
    scores: List[float] = []
    for m, s, st in zip(mu, sigma, strain):
        scores.append(float(m + k_exploration * s - lambda_strain * st))
    return scores


def summarize_pool(tag: str, mols: Sequence[Chem.Mol], y: Sequence[float]) -> None:
    if not mols:
        print(f"[{tag}] empty pool")
        return
    best = max(y)
    avg = sum(y) / len(y)
    uniq = len({canonical_smiles(m) for m in mols})
    print(f"[{tag}] n={len(mols)} uniq={uniq} best={best:.3f} avg={avg:.3f}")


def run_optimization(
    seed_smiles: Sequence[str] = ("CC",),
    config: Optional[OptConfig] = None,
) -> Dict[str, object]:
    cfg = config or OptConfig()
    random.seed(cfg.random_seed)

    # Prepare objective
    obj_fn = get_objective(cfg.objective)

    # Seed molecules
    seed_mols = [mol_from_smiles(s) for s in seed_smiles]
    seed_mols = [m for m in seed_mols if m is not None]
    if not seed_mols:
        raise ValueError("No valid seed molecules provided")
    # Precompute seed Morgan bit vectors for novelty
    seed_bvs = [morgan_bv(m, cfg.radius, cfg.nbits) for m in seed_mols]
    seed_scaffolds = {murcko_scaffold_smiles(m) for m in seed_mols}

    # Bootstrap training set with true objective labels
    train_mols, X, y = bootstrap_training_set(
        seed_mols, obj_fn, cfg.init_train_size, cfg.random_seed, cfg.radius, cfg.nbits, cfg.use_physchem
    )
    if cfg.train_on_composite:
        # Relabel y to composite eff = QED - beta*SA - gamma*max(strain, 0)
        y = []
        for m in train_mols:
            q = objective_qed(m)
            sa = sa_value(m) or 0.0
            st = _get_strain_cached(m)
            stp = max(0.0, st)
            eff = q - cfg.sa_soft_beta * sa - cfg.lambda_strain * stp
            y.append(float(eff))
    summarize_pool("boot", train_mols, y)

    # Optimization rounds
    history = {
        "best": [],
        "avg": [],
        "n_train": [],
    }

    csv_writer = None
    csv_file = None
    if cfg.log_csv:
        try:
            # Ensure parent folder exists
            folder = os.path.dirname(cfg.log_csv)
            if folder:
                os.makedirs(folder, exist_ok=True)
            write_header = True
            if os.path.exists(cfg.log_csv) and os.path.getsize(cfg.log_csv) > 0:
                write_header = False
            csv_file = open(cfg.log_csv, "a", newline="")
            csv_writer = csv.writer(csv_file)
            if write_header:
                csv_writer.writerow([
                    "phase",
                    "round",
                    "rank",
                    "smiles",
                    "qed",
                    "sa",
                    "strain",
                    "scaffold",
                    "score",
                    "eff_score",
                    "accept_rate",
                    "novelty_to_seeds",
                    "mean_pairwise_dist",
                    "unique_scaffolds",
                    "y_pred_mean",
                    "y_pred_std",
                    "mae_sel",
                    "rmse_sel",
                    "r2_sel",
                    "spearman_sel",
                    "mae_audit",
                    "rmse_audit",
                    "r2_audit",
                    "spearman_audit",
                ])
        except Exception:
            csv_writer = None
            csv_file = None

    # Initialize controller (LLM or heuristic)
    if cfg.use_controller:
        if cfg.use_llm_controller:
            try:
                controller = LLMController()
            except Exception as e:
                print(f"Failed to initialize LLM controller: {e}. Falling back to heuristic controller.")
                controller = HeuristicController()
        else:
            controller = HeuristicController()
    else:
        controller = None
    decisions: List[Dict[str, object]] = []

    for r in range(cfg.rounds):
        t0 = time.time()
        # Train ensemble
        models = train_rf_ensemble(X, y, n_models=cfg.ensemble_size, random_seed=cfg.random_seed + r)

        # Controller decides operator and knobs (if enabled)
        if controller is not None:
            last_best = max(y) if y else 0.0
            last_avg = sum(y) / len(y) if y else 0.0
            obs = Observation(
                round_index=r + 1,
                rounds_total=cfg.rounds,
                train_size=len(y),
                best=last_best,
                avg=last_avg,
                last_best=last_best,
                last_avg=last_avg,
            )
            action: Action = controller.decide(obs)
            # Apply knob overrides with clamping to safe ranges
            def _clamp(v, lo, hi):
                try:
                    x = float(v)
                except Exception:
                    return lo
                return max(lo, min(hi, x))

            k = cfg.k_exploration if action.k is None else _clamp(action.k, 0.0, 2.0)
            div = cfg.diversity_penalty if action.div is None else _clamp(action.div, 0.0, 1.0)
            lam_scale = 1.0 if action.lam is None else _clamp(action.lam, 0.0, 2.0)
            topk = cfg.topk_per_round if action.topk is None else int(max(1, action.topk))
            cands = cfg.candidates_per_round if action.cands is None else int(max(topk, action.cands))
            sa_beta = cfg.sa_soft_beta if action.sa_beta is None else _clamp(action.sa_beta, 0.0, 1.0)
            scaf_alpha = (
                cfg.scaffold_penalty_alpha if action.scaf_alpha is None else _clamp(action.scaf_alpha, 0.0, 1.0)
            )
            # overwrite local vars
            local_topk = topk
            local_cands = cands
            local_k = k
            local_div = div
            local_lam = cfg.lambda_strain * lam_scale
            local_sa_beta = sa_beta
            local_scaf_alpha = scaf_alpha
            op = action.op or "brics"
        else:
            local_topk = cfg.topk_per_round
            local_cands = cfg.candidates_per_round
            local_k = cfg.k_exploration
            local_div = cfg.diversity_penalty
            local_lam = cfg.lambda_strain
            local_sa_beta = cfg.sa_soft_beta
            local_scaf_alpha = cfg.scaffold_penalty_alpha
            op = "brics"

        # Propose candidates from current pool
        if op == "attach":
            cand_mols = simple_attach_candidates(
                seed_mols=train_mols,
                n_candidates=local_cands,
                random_seed=cfg.random_seed + 123 + r,
            )
        else:
            cand_mols = brics_candidates(
                seed_mols=train_mols,
                n_candidates=local_cands,
                random_seed=cfg.random_seed + 123 + r,
            )
        # Optional strict scaffold preservation (keep only seed scaffolds)
        if cfg.preserve_seed_scaffold:
            cand_mols = [m for m in cand_mols if murcko_scaffold_smiles(m) in seed_scaffolds]
        if not cand_mols:
            print("No candidates generated; stopping early.")
            break

        # Featurize and predict
        Xc = featurize_mols(cand_mols, cfg.radius, cfg.nbits, include_physchem=cfg.use_physchem)
        mu, sigma = ensemble_predict(models, Xc)

        # Pre-screen by UCB without strain to limit MMFF calls
        pre_scores = [float(m + local_k * s) for m, s in zip(mu, sigma)]
        n_prescreen = int(max(local_topk, min(len(cand_mols), cfg.mmff_prescreen_factor * local_topk)))
        pre_idx = sorted(range(len(pre_scores)), key=lambda i: pre_scores[i], reverse=True)[: n_prescreen]

        # Physics: MMFF strain per atom with caching for prescreened set
        strains_prescreen = []
        for i in pre_idx:
            strains_prescreen.append(_get_strain_cached(cand_mols[i]))

        # Acquisition: UCB with strain penalty on prescreened set
        mu_pre = [mu[i] for i in pre_idx]
        sig_pre = [sigma[i] for i in pre_idx]
        # If training on composite, don't double-count strain here
        lam_for_acq = (0.0 if cfg.train_on_composite else local_lam)
        scores_pre = ucb_scores(mu_pre, sig_pre, strains_prescreen, local_k, lam_for_acq)

        # Optional soft SA penalty (skip if training on composite to avoid double-counting)
        if (not cfg.train_on_composite) and local_sa_beta > 0.0 and _SA_AVAILABLE and _SA_CALC is not None:
            sa_vals = []
            for i in pre_idx:
                v = sa_value(cand_mols[i])
                sa_vals.append(0.0 if v is None else float(v))
            scores_pre = [s - local_sa_beta * v for s, v in zip(scores_pre, sa_vals)]
        else:
            sa_vals = [0.0 for _ in pre_idx]

        # Scaffold usage penalty based on current training pool
        if local_scaf_alpha > 0.0:
            sc_counts: Dict[str, int] = {}
            for tm in train_mols:
                sc = murcko_scaffold_smiles(tm)
                sc_counts[sc] = sc_counts.get(sc, 0) + 1
            max_count = max(sc_counts.values()) if sc_counts else 0
            sc_pens = []
            for i in pre_idx:
                sc = murcko_scaffold_smiles(cand_mols[i])
                cnt = sc_counts.get(sc, 0)
                pen = local_scaf_alpha * (cnt / max(1, max_count))
                sc_pens.append(pen)
            scores_pre = [s - p for s, p in zip(scores_pre, sc_pens)]
        else:
            sc_pens = [0.0 for _ in pre_idx]

        # Novelty penalty vs seeds
        if cfg.novelty_penalty_alpha > 0.0 and seed_bvs:
            nov_pens = []
            for i in pre_idx:
                bv = morgan_bv(cand_mols[i], cfg.radius, cfg.nbits)
                max_sim = 0.0
                for sbv in seed_bvs:
                    sim = DataStructs.TanimotoSimilarity(bv, sbv)
                    if sim > max_sim:
                        max_sim = sim
                nov_pens.append(cfg.novelty_penalty_alpha * max_sim)
            scores_pre = [s - p for s, p in zip(scores_pre, nov_pens)]
        else:
            nov_pens = [0.0 for _ in pre_idx]

        # Diversity-aware greedy selection: penalize similarity to already-picked
        selected_local: List[int] = []
        selected_scores: List[float] = []
        bvs = [morgan_bv(cand_mols[i], cfg.radius, cfg.nbits) for i in pre_idx]
        # precompute scaffolds for candidates
        pre_scaffolds = [murcko_scaffold_smiles(cand_mols[i]) for i in pre_idx]
        scaf_counts_local: Dict[str, int] = {}
        target_k = min(local_topk, len(pre_idx))
        while len(selected_local) < target_k:
            best_j = None
            best_eff = None
            for j in range(len(pre_idx)):
                if j in selected_local:
                    continue
                # enforce hard scaffold cap if configured
                if cfg.scaffold_cap_per_round is not None and cfg.scaffold_cap_per_round > 0:
                    sc = pre_scaffolds[j]
                    cnt = scaf_counts_local.get(sc, 0)
                    if cnt >= cfg.scaffold_cap_per_round:
                        continue
                # penalty based on max similarity to already selected within this round
                if not selected_local or local_div <= 0:
                    penalty = 0.0
                else:
                    max_sim = 0.0
                    for jj in selected_local:
                        sim = DataStructs.TanimotoSimilarity(bvs[j], bvs[jj])
                        if sim > max_sim:
                            max_sim = sim
                    penalty = local_div * max_sim
                eff = scores_pre[j] - penalty
                if (best_eff is None) or (eff > best_eff):
                    best_eff = eff
                    best_j = j
            if best_j is None:
                break
            selected_local.append(best_j)
            selected_scores.append(float(best_eff) if best_eff is not None else 0.0)
            # update scaffold usage counts
            sc_sel = pre_scaffolds[best_j]
            scaf_counts_local[sc_sel] = scaf_counts_local.get(sc_sel, 0) + 1
        # Backfill to reach top-k by relaxing caps/penalties if underfilled
        if len(selected_local) < local_topk:
            remaining = [j for j in range(len(pre_idx)) if j not in selected_local]
            remaining.sort(key=lambda j: scores_pre[j], reverse=True)
            for j in remaining:
                selected_local.append(j)
                selected_scores.append(float(scores_pre[j]))
                if len(selected_local) >= local_topk:
                    break
        idx = [pre_idx[j] for j in selected_local]
        sel_mols = [cand_mols[i] for i in idx]

        # Evaluate labels for selected molecules, add to training set
        if cfg.train_on_composite:
            y_sel = []
            for m in sel_mols:
                q = objective_qed(m)
                sa = sa_value(m) or 0.0
                st = _get_strain_cached(m)
                stp = max(0.0, st)
                y_sel.append(float(q - cfg.sa_soft_beta * sa - cfg.lambda_strain * stp))
        else:
            y_sel = [obj_fn(m) for m in sel_mols]
        X_sel = [Xc[i] for i in idx]
        train_mols.extend(sel_mols)
        X.extend(X_sel)
        y.extend(y_sel)

        # Log
        best = max(y)
        avg = sum(y) / len(y)
        history["best"].append(best)
        history["avg"].append(avg)
        history["n_train"].append(len(y))
        dt = time.time() - t0
        accept_rate = (len(sel_mols) / max(1, len(pre_idx)))
        print(
            f"[round {r+1}/{cfg.rounds}] added={len(sel_mols)} train={len(y)} best={best:.3f} avg={avg:.3f} acc={accept_rate:.2f} time={dt:.1f}s"
        )
        if controller is not None:
            print(
                f"  agent: op={op} k={local_k:.2f} lam={local_lam:.2f} div={local_div:.2f} saB={local_sa_beta:.2f} scafA={local_scaf_alpha:.2f} cands={local_cands} topk={local_topk}"
            )

        if controller is not None:
            decisions.append({
                "round": r + 1,
                "op": op,
                "k": local_k,
                "lam": local_lam,
                "div": local_div,
                "sa_beta": local_sa_beta,
                "scaf_alpha": local_scaf_alpha,
                "cands": local_cands,
                "topk": local_topk,
            })

        if csv_writer is not None:
            for rank, (i_cand, eff_score) in enumerate(zip(idx, selected_scores), start=1):
                m = cand_mols[i_cand]
                smi = canonical_smiles(m)
                q = objective_qed(m)
                sa = sa_value(m)
                st = _get_strain_cached(m)
                scaf = murcko_scaffold_smiles(m)
                # effective scalarized score (measured): QED - sa_beta*SA - lam*max(strain,0)
                stp = max(0.0, st if st is not None else 0.0)
                eff_meas = q - (local_sa_beta * (sa if sa is not None else 0.0)) - (local_lam * stp)
                # novelty vs seeds
                if seed_bvs:
                    bv = morgan_bv(m, cfg.radius, cfg.nbits)
                    max_sim_seed = max((DataStructs.TanimotoSimilarity(bv, sbv) for sbv in seed_bvs), default=0.0)
                else:
                    max_sim_seed = 0.0
                yhat = mu[i_cand]
                ystd = sigma[i_cand]
                csv_writer.writerow(["round", r + 1, rank, smi, f"{q:.3f}", f"{sa:.3f}" if sa is not None else "", f"{st:.3f}", scaf, f"{eff_score:.4f}", f"{eff_meas:.4f}", f"{accept_rate:.3f}", f"{1.0 - max_sim_seed:.3f}", "", "", f"{yhat:.4f}", f"{ystd:.4f}"])  # novelty reported as distance = 1 - max_sim
            # round summary row
            # compute simple diversity metrics for selected set
            sel_bvs = [morgan_bv(cand_mols[i], cfg.radius, cfg.nbits) for i in idx]
            # mean pairwise tanimoto distance
            if len(sel_bvs) > 1:
                sims = []
                for a in range(len(sel_bvs)):
                    for b in range(a + 1, len(sel_bvs)):
                        sims.append(DataStructs.TanimotoSimilarity(sel_bvs[a], sel_bvs[b]))
                mean_sim = sum(sims) / len(sims)
                mean_dist = 1.0 - mean_sim
            else:
                mean_dist = 0.0
            unique_scaffs = len({murcko_scaffold_smiles(cand_mols[i]) for i in idx})
            # prediction metrics for selected set
            y_true_sel = y_sel
            y_pred_sel = [mu[i] for i in idx]
            mae_sel = _mae(y_true_sel, y_pred_sel)
            rmse_sel = _rmse(y_true_sel, y_pred_sel)
            r2_sel = _r2_score(y_true_sel, y_pred_sel)
            sp_sel = _spearman(y_true_sel, y_pred_sel)

            # audit (random prescreened not selected)
            mae_aud = rmse_aud = r2_aud = sp_aud = None
            if cfg.audit_k and len(pre_idx) > len(idx):
                pool = [i for i in pre_idx if i not in idx]
                random.seed(cfg.random_seed + r)
                take = min(cfg.audit_k, len(pool))
                aud_idx = random.sample(pool, take)
                y_true_aud = []
                for i in aud_idx:
                    m = cand_mols[i]
                    if cfg.train_on_composite:
                        q = objective_qed(m)
                        sa = sa_value(m) or 0.0
                        stp = max(0.0, _get_strain_cached(m))
                        y_true_aud.append(float(q - cfg.sa_soft_beta * sa - cfg.lambda_strain * stp))
                    else:
                        y_true_aud.append(obj_fn(m))
                y_pred_aud = [mu[i] for i in aud_idx]
                mae_aud = _mae(y_true_aud, y_pred_aud)
                rmse_aud = _rmse(y_true_aud, y_pred_aud)
                r2_aud = _r2_score(y_true_aud, y_pred_aud)
                sp_aud = _spearman(y_true_aud, y_pred_aud)

            csv_writer.writerow([
                "round_summary",
                r + 1,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                f"{accept_rate:.3f}",
                "",
                f"{mean_dist:.3f}",
                str(unique_scaffs),
                f"{mae_sel:.4f}" if mae_sel is not None else "",
                f"{rmse_sel:.4f}" if rmse_sel is not None else "",
                f"{r2_sel:.3f}" if r2_sel is not None else "",
                f"{sp_sel:.3f}" if sp_sel is not None else "",
                f"{mae_aud:.4f}" if mae_aud is not None else "",
                f"{rmse_aud:.4f}" if rmse_aud is not None else "",
                f"{r2_aud:.3f}" if r2_aud is not None else "",
                f"{sp_aud:.3f}" if sp_aud is not None else "",
            ])
            try:
                csv_file.flush()
            except Exception:
                pass

    # Finalize and return results
    summarize_pool("final", train_mols, y)
    # Return top molecules
    scored = list(zip(train_mols, y))
    scored.sort(key=lambda t: t[1], reverse=True)
    top = [(canonical_smiles(m), float(v)) for m, v in scored[:50]]
    if csv_writer is not None:
        for rank, (smi, v) in enumerate(top, start=1):
            m = mol_from_smiles(smi)
            if m is None:
                continue
            q = objective_qed(m)
            sa = sa_value(m)
            st = _get_strain_cached(m)
            scaf = murcko_scaffold_smiles(m)
            stp = max(0.0, st if st is not None else 0.0)
            eff_meas = q - (cfg.sa_soft_beta * (sa if sa is not None else 0.0)) - (cfg.lambda_strain * stp)
            csv_writer.writerow(["final", cfg.rounds, rank, smi, f"{q:.3f}", f"{sa:.3f}" if sa is not None else "", f"{st:.3f}", scaf, f"{v:.4f}", f"{eff_meas:.4f}", "", "", "", ""]) 
        try:
            csv_file.close()
        except Exception:
            pass
    return {
        "history": history,
        "top": top,
        "objective": cfg.objective,
        "decisions": decisions,
    }

# Simple fragment-attach fallback generator
# ------------------------

# PAINS filter catalog (built once)
_pains_catalog = None


def _get_pains_catalog():
    global _pains_catalog
    if _pains_catalog is not None:
        return _pains_catalog
    try:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
        _pains_catalog = FilterCatalog.FilterCatalog(params)
    except Exception:
        _pains_catalog = None
    return _pains_catalog


def passes_pains(mol: Chem.Mol) -> bool:
    cat = _get_pains_catalog()
    if cat is None:
        return True  # if PAINS not available, do not block
    try:
        return not cat.HasMatch(mol)
    except Exception:
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline BRICS + RF-UCB + MMFF optimizer")
    parser.add_argument("--seed", dest="seed", default="CC", help="Seed SMILES (comma-separated)")
    parser.add_argument("--objective", dest="objective", default="qed", choices=["qed", "pen_logp"], help="Objective")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--init", type=int, default=64, help="Initial training set size")
    parser.add_argument("--cands", type=int, default=600, help="Candidates per round")
    parser.add_argument("--topk", type=int, default=80, help="Top-k per round")
    parser.add_argument("--k", dest="kexp", type=float, default=0.7, help="UCB exploration weight")
    parser.add_argument("--lam", dest="lam", type=float, default=0.2, help="Strain penalty weight")
    parser.add_argument("--seed_int", type=int, default=0, help="Random seed")
    parser.add_argument("--pre_factor", type=float, default=5.0, help="MMFF prescreen multiplier over top-k")
    parser.add_argument("--div", dest="div", type=float, default=0.2, help="Diversity penalty alpha (Tanimoto)")
    parser.add_argument("--use_sa", action="store_true", help="Enable SA filtering (requires sascorer)")
    parser.add_argument("--sa_max", type=float, default=6.0, help="Max SA allowed if --use_sa")
    parser.add_argument("--sa_beta", type=float, default=0.0, help="Soft SA penalty beta in acquisition")
    parser.add_argument("--scaf_alpha", type=float, default=0.0, help="Scaffold usage penalty alpha")
    parser.add_argument("--csv", dest="csv", default="", help="CSV path to log per-round selections")
    parser.add_argument("--agent", action="store_true", help="Enable agentic controller (heuristic)")
    parser.add_argument("--llm", action="store_true", help="Use LLM-based controller (requires OPENAI_API_KEY)")
    parser.add_argument("--history_json", default="", help="Path to save history JSON (best/avg/n_train)")
    parser.add_argument("--decisions_json", default="", help="Path to save per-round decisions JSON")
    parser.add_argument("--top_json", default="", help="Path to save final top list JSON")
    parser.add_argument("--scaffold_cap", type=int, default=-1, help="Hard cap per Murcko scaffold among selected per round (-1 to disable)")
    parser.add_argument("--audit_k", type=int, default=0, help="Per round, audit k random prescreened (not selected) for prediction error metrics")
    parser.add_argument("--preserve_scaffold", action="store_true", help="Strictly preserve seed Murcko scaffold(s) when selecting candidates")
    parser.add_argument("--physchem", action="store_true", help="Concatenate RDKit physchem descriptors to ECFP features for the surrogate")
    args = parser.parse_args()

    # Optional SA constraints
    set_sa_constraints(args.use_sa, args.sa_max)

    cfg = OptConfig(
        objective=args.objective,
        rounds=args.rounds,
        init_train_size=args.init,
        candidates_per_round=args.cands,
        topk_per_round=args.topk,
        k_exploration=args.kexp,
        lambda_strain=args.lam,
        random_seed=args.seed_int,
        mmff_prescreen_factor=args.pre_factor,
        diversity_penalty=args.div,
        sa_soft_beta=args.sa_beta,
        scaffold_penalty_alpha=args.scaf_alpha,
        log_csv=(args.csv if args.csv else None),
        use_controller=args.agent or args.llm,
        use_llm_controller=args.llm,
        scaffold_cap_per_round=(None if args.scaffold_cap is None or args.scaffold_cap < 0 else args.scaffold_cap),
        audit_k=args.audit_k,
        preserve_seed_scaffold=args.preserve_scaffold,
        use_physchem=args.physchem,
    )
    seed_smiles = [s.strip() for s in args.seed.split(",") if s.strip()]
    out = run_optimization(seed_smiles=seed_smiles, config=cfg)
    print("Top molecules:")
    for smi, v in out["top"]:
        print(f"{v:.3f}\t{smi}")

    # Optional JSON exports
    def _ensure_parent(path: str) -> None:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    if args.history_json:
        try:
            _ensure_parent(args.history_json)
            with open(args.history_json, "w") as f:
                import json

                json.dump(out.get("history", {}), f, indent=2)
        except Exception:
            pass
    if args.decisions_json:
        try:
            _ensure_parent(args.decisions_json)
            with open(args.decisions_json, "w") as f:
                import json

                json.dump(out.get("decisions", []), f, indent=2)
        except Exception:
            pass
    if args.top_json:
        try:
            _ensure_parent(args.top_json)
            with open(args.top_json, "w") as f:
                import json

                json.dump(out.get("top", []), f, indent=2)
        except Exception:
            pass
