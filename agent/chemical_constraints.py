"""
Chemical validity and stability constraints for molecular optimization.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
import numpy as np


class ChemicalConstraints:
    """Chemical validity and stability constraints for molecular optimization."""
    
    def __init__(self):
        self.max_molecular_weight = 500.0  # Drug-like molecules
        self.max_heavy_atoms = 50
        self.max_rotatable_bonds = 10
        self.max_rings = 6
        self.max_aromatic_rings = 4
        
    def is_chemically_valid(self, smiles: str) -> Tuple[bool, str]:
        """
        Check if a SMILES string represents a chemically valid and stable molecule.
        
        Returns:
            (is_valid, reason)
        """
        # Basic SMILES validity
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES syntax"
        
        # Sanitization check
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return False, f"Molecular sanitization failed: {str(e)}"
        
        # Check for unstable functional groups
        stability_check, stability_reason = self._check_stability(mol)
        if not stability_check:
            return False, stability_reason
        
        # Check molecular properties
        property_check, property_reason = self._check_properties(mol)
        if not property_check:
            return False, property_reason
        
        return True, "Valid and stable molecule"
    
    def _check_stability(self, mol) -> Tuple[bool, str]:
        """Check for chemically unstable groups."""
        
        # Check for polyoxides (O-O-O chains) - extremely unstable
        if self._has_polyoxide_chain(mol):
            return False, "Contains unstable polyoxide chain (O-O-O)"
        
        # Check for other unstable groups
        unstable_patterns = [
            # Peroxides (O-O single bonds)
            ('[O][O]', 'Contains unstable peroxide group'),
            # Azides (N-N-N)
            ('[N]=[N]=[N]', 'Contains unstable azide group'),
            # Nitro groups on aliphatic carbons (can be explosive)
            ('[C][N+](=O)[O-]', 'Contains potentially explosive nitro group'),
        ]
        
        for pattern, reason in unstable_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return False, reason
        
        return True, "No unstable groups detected"
    
    def _has_polyoxide_chain(self, mol) -> bool:
        """Check for polyoxide chains (3+ consecutive O atoms)."""
        # Look for O-O-O pattern
        pattern = Chem.MolFromSmarts('[O][O][O]')
        return mol.HasSubstructMatch(pattern)
    
    def _check_properties(self, mol) -> Tuple[bool, str]:
        """Check molecular properties for drug-likeness."""
        
        mw = Descriptors.MolWt(mol)
        if mw > self.max_molecular_weight:
            return False, f"Molecular weight too high: {mw:.1f} > {self.max_molecular_weight}"
        
        heavy_atoms = mol.GetNumHeavyAtoms()
        if heavy_atoms > self.max_heavy_atoms:
            return False, f"Too many heavy atoms: {heavy_atoms} > {self.max_heavy_atoms}"
        
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if rotatable_bonds > self.max_rotatable_bonds:
            return False, f"Too many rotatable bonds: {rotatable_bonds} > {self.max_rotatable_bonds}"
        
        rings = rdMolDescriptors.CalcNumRings(mol)
        if rings > self.max_rings:
            return False, f"Too many rings: {rings} > {self.max_rings}"
        
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        if aromatic_rings > self.max_aromatic_rings:
            return False, f"Too many aromatic rings: {aromatic_rings} > {self.max_aromatic_rings}"
        
        return True, "Properties within acceptable ranges"
    
    def get_synthetic_accessibility_score(self, smiles: str) -> Tuple[float, str]:
        """
        Calculate synthetic accessibility score (0-1, higher is more accessible).
        Simplified version based on molecular complexity.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0, "Invalid molecule"
        
        # Factors that make synthesis harder
        complexity_penalties = []
        
        # Molecular weight penalty
        mw = Descriptors.MolWt(mol)
        mw_penalty = min(mw / 1000.0, 0.3)  # Max 30% penalty
        complexity_penalties.append(mw_penalty)
        
        # Ring penalty
        rings = rdMolDescriptors.CalcNumRings(mol)
        ring_penalty = min(rings * 0.05, 0.2)  # Max 20% penalty
        complexity_penalties.append(ring_penalty)
        
        # Stereocenter penalty
        stereocenters = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        stereo_penalty = min(stereocenters * 0.1, 0.3)  # Max 30% penalty
        complexity_penalties.append(stereo_penalty)
        
        # Functional group penalty
        fg_penalty = self._calculate_functional_group_penalty(mol)
        complexity_penalties.append(fg_penalty)
        
        # Calculate final score (1.0 = very accessible, 0.0 = very difficult)
        total_penalty = sum(complexity_penalties)
        score = max(0.0, 1.0 - total_penalty)
        
        return score, f"SA Score: {score:.2f} (penalties: {total_penalty:.2f})"
    
    def _calculate_functional_group_penalty(self, mol) -> float:
        """Calculate penalty based on complex functional groups."""
        penalty = 0.0
        
        # Penalize complex functional groups
        complex_groups = [
            ('[N+](=O)[O-]', 0.1),  # Nitro
            ('[S](=O)(=O)', 0.05),  # Sulfonyl
            ('[C](=O)[N]', 0.05),   # Amide
            ('[C](=O)[O]', 0.03),   # Ester
        ]
        
        for pattern, group_penalty in complex_groups:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                penalty += group_penalty
        
        return min(penalty, 0.2)  # Max 20% penalty


def filter_valid_molecules(smiles_list: List[str], constraints: ChemicalConstraints = None) -> List[Tuple[str, str]]:
    """
    Filter a list of SMILES to only include chemically valid molecules.
    
    Returns:
        List of (smiles, reason) tuples for valid molecules
    """
    if constraints is None:
        constraints = ChemicalConstraints()
    
    valid_molecules = []
    for smiles in smiles_list:
        is_valid, reason = constraints.is_chemically_valid(smiles)
        if is_valid:
            valid_molecules.append((smiles, reason))
    
    return valid_molecules


if __name__ == "__main__":
    # Test the constraints
    constraints = ChemicalConstraints()
    
    test_molecules = [
        "CCO",           # Ethanol - should be valid
        "CCOOOO",        # Polyoxide - should be invalid
        "CC(=O)O",       # Acetic acid - should be valid
        "C1=CC=CC=C1",   # Benzene - should be valid
        "CCOO",          # Ethyl hydroperoxide - might be valid but unstable
    ]
    
    print("Chemical Validity Testing:")
    print("=" * 50)
    
    for smiles in test_molecules:
        is_valid, reason = constraints.is_chemically_valid(smiles)
        sa_score, sa_reason = constraints.get_synthetic_accessibility_score(smiles)
        
        print(f"\n{smiles}:")
        print(f"  Valid: {is_valid}")
        print(f"  Reason: {reason}")
        print(f"  SA Score: {sa_score:.2f}")
        print(f"  SA Reason: {sa_reason}")
