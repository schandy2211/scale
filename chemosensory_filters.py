"""
Chemosensory-specific molecular filters for odorant optimization.
Based on typical odorant properties and safety considerations.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
from typing import List, Optional, Tuple
import warnings

# Suppress RDKit warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")


def passes_odorant_filters(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """
    Check if molecule passes basic odorant feasibility filters.
    
    Based on typical odorant properties:
    - MW: 90-250 (some exceptions allowed)
    - logP: 1-4 (volatility window)
    - TPSA: < 40 (permeability/volatility)
    - HBD: ≤ 1 (hydrogen bond donors)
    - HBA: ≤ 6 (hydrogen bond acceptors)
    
    Returns:
        Tuple of (passes_all_filters, list_of_failed_filters)
    """
    if mol is None:
        return False, ["Invalid molecule"]
    
    failed_filters = []
    
    try:
        # Molecular weight filter (85-250 Da) - slightly relaxed for small esters
        mw = Descriptors.MolWt(mol)
        if mw < 85 or mw > 250:
            failed_filters.append(f"MW={mw:.1f} (want 85-250)")
        
        # LogP filter (0.5-4 for good volatility) - relaxed for small volatiles
        logp = Descriptors.MolLogP(mol)
        if logp < 0.5 or logp > 4.0:
            failed_filters.append(f"logP={logp:.2f} (want 0.5-4)")
        
        # Topological polar surface area (< 40 for volatility)
        tpsa = Descriptors.TPSA(mol)
        if tpsa > 40:
            failed_filters.append(f"TPSA={tpsa:.1f} (want <40)")
        
        # Hydrogen bond donors (≤ 1)
        hbd = Descriptors.NumHDonors(mol)
        if hbd > 1:
            failed_filters.append(f"HBD={hbd} (want ≤1)")
        
        # Hydrogen bond acceptors (≤ 6)
        hba = Descriptors.NumHAcceptors(mol)
        if hba > 6:
            failed_filters.append(f"HBA={hba} (want ≤6)")
        
        # Rotatable bonds (≤ 8 for reasonable flexibility)
        rotbonds = Descriptors.NumRotatableBonds(mol)
        if rotbonds > 8:
            failed_filters.append(f"RotBonds={rotbonds} (want ≤8)")
            
    except Exception as e:
        failed_filters.append(f"Calculation error: {e}")
    
    return len(failed_filters) == 0, failed_filters


def passes_safety_alerts(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """
    Check for basic safety/regulatory alerts relevant to odorants.
    
    Returns:
        Tuple of (passes_safety, list_of_alerts)
    """
    if mol is None:
        return False, ["Invalid molecule"]
    
    alerts = []
    smiles = Chem.MolToSmiles(mol)
    
    # Known problematic patterns for fragrances/flavors
    problematic_patterns = [
        # Nitro compounds (often sensitizers)
        ("[N+](=O)[O-]", "Nitro group (potential sensitizer)"),
        # Strong electrophiles
        ("C(=O)Cl", "Acid chloride (reactive)"),
        ("C(=O)F", "Acid fluoride (reactive)"),
        # Polycyclic aromatics (some restricted)
        ("c1ccc2c(c1)ccc3c2cccc3", "Anthracene core (check regulations)"),
        # Long-chain aldehydes (simplified - at least 8 carbons with aldehyde)
        ("CCCCCCCCC=O", "Long-chain aldehyde (potential sensitizer)"),
        # Known allergens (simplified patterns)
        ("C=CCc1ccc(O)cc1", "Eugenol-like (allergen list)"),
        ("C=CCc1ccc(OC)cc1", "Estragole-like (check regulations)"),
    ]
    
    for pattern, alert_msg in problematic_patterns:
        try:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is not None and mol.HasSubstructMatch(pattern_mol):
                alerts.append(alert_msg)
        except Exception:
            # Skip invalid patterns
            continue
    
    # Additional safety checks
    try:
        # Very high logP might indicate bioaccumulation
        logp = Descriptors.MolLogP(mol)
        if logp > 5.0:
            alerts.append(f"Very high logP={logp:.2f} (bioaccumulation risk)")
            
        # Very low or high MW outliers
        mw = Descriptors.MolWt(mol)
        if mw > 300:
            alerts.append(f"High MW={mw:.1f} (unusual for odorant)")
        
    except Exception:
        alerts.append("Safety calculation error")
    
    return len(alerts) == 0, alerts


def calculate_odorant_score(mol: Chem.Mol) -> float:
    """
    Calculate a simple odorant-likeness score.
    
    This is a placeholder that combines QED with odorant-specific penalties.
    In a real system, this would be replaced with a trained ML model
    on olfactory data (e.g., from DREAM challenge, Pyrfume).
    
    Returns:
        Score between 0-1 (higher = more odorant-like)
    """
    if mol is None:
        return 0.0
    
    try:
        # Start with QED as base drug-likeness
        base_score = QED.qed(mol)
        
        # Odorant-specific adjustments
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Penalty for being outside odorant MW range
        mw_penalty = 0.0
        if mw < 90 or mw > 250:
            mw_penalty = 0.3
        
        # Penalty for being outside odorant logP range
        logp_penalty = 0.0
        if logp < 1.0 or logp > 4.0:
            logp_penalty = 0.2
        
        # Penalty for high TPSA (poor volatility)
        tpsa_penalty = 0.0
        if tpsa > 40:
            tpsa_penalty = 0.2
        
        # Bonus for aromatic content (many odorants are aromatic)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        aromatic_bonus = min(0.1 * aromatic_rings, 0.2)
        
        # Final score
        odorant_score = base_score - mw_penalty - logp_penalty - tpsa_penalty + aromatic_bonus
        return max(0.0, min(1.0, odorant_score))
        
    except Exception:
        return 0.0


def filter_odorant_candidates(mols: List[Chem.Mol], verbose: bool = False) -> List[Chem.Mol]:
    """
    Filter a list of molecules for odorant feasibility.
    
    Args:
        mols: List of RDKit molecules
        verbose: Print filtering details
        
    Returns:
        List of molecules that pass odorant filters
    """
    passed_mols = []
    
    for mol in mols:
        if mol is None:
            continue
            
        # Check odorant properties
        passes_props, prop_failures = passes_odorant_filters(mol)
        passes_safety, safety_alerts = passes_safety_alerts(mol)
        
        if passes_props and passes_safety:
            passed_mols.append(mol)
        elif verbose:
            smiles = Chem.MolToSmiles(mol)
            print(f"❌ Filtered {smiles}: {prop_failures + safety_alerts}")
    
    if verbose:
        print(f"✅ Odorant filter: {len(passed_mols)}/{len(mols)} molecules passed")
    
    return passed_mols


# Example usage
if __name__ == "__main__":
    # Test with some known odorants
    test_smiles = [
        "COc1ccccc1",      # anisole (good odorant)
        "CCOC(=O)C",       # ethyl acetate (good odorant)
        "CCCCCCCCCCCCCC",  # tetradecane (too large, low volatility)
        "O",               # water (too small)
        "c1ccc2c(c1)ccc3c2cccc3",  # anthracene (potential regulatory issue)
    ]
    
    print("🧪 Testing odorant filters:")
    for smiles in test_smiles:
        mol = Chem.MolFromSmiles(smiles)
        passes_props, prop_failures = passes_odorant_filters(mol)
        passes_safety, safety_alerts = passes_safety_alerts(mol)
        score = calculate_odorant_score(mol)
        
        status = "✅" if (passes_props and passes_safety) else "❌"
        print(f"{status} {smiles}: score={score:.3f}")
        if prop_failures:
            print(f"   Property issues: {prop_failures}")
        if safety_alerts:
            print(f"   Safety alerts: {safety_alerts}")
