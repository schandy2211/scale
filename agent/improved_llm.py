"""
Improved LLM agent with chemical reasoning and constraints.
"""

from __future__ import annotations

import random
from typing import List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from .chemical_constraints import ChemicalConstraints


class ImprovedLLMAgent:
    """Enhanced LLM agent with chemical reasoning and constraints."""
    
    def __init__(self, constraints: ChemicalConstraints = None):
        self.constraints = constraints or ChemicalConstraints()
        self.max_attempts = 10  # Maximum attempts to find valid modifications
        
    def propose_smiles(self, smiles: str) -> Tuple[str, str]:
        """
        Propose a chemically valid modification to a SMILES string.
        
        Returns:
            (modified_smiles, reason)
        """
        # Validate input
        is_valid, reason = self.constraints.is_chemically_valid(smiles)
        if not is_valid:
            return self._get_fallback_molecule(), f"Input invalid: {reason}"
        
        # Generate multiple modification strategies
        strategies = [
            self._add_functional_group,
            self._modify_existing_group,
            self._add_ring,
            self._substitute_atom,
            self._extend_chain,
        ]
        
        # Try each strategy until we find a valid modification
        for attempt in range(self.max_attempts):
            strategy = random.choice(strategies)
            candidates = strategy(smiles)
            
            # Filter for chemically valid molecules
            valid_candidates = []
            for candidate, reason in candidates:
                is_valid, validity_reason = self.constraints.is_chemically_valid(candidate)
                if is_valid:
                    sa_score, _ = self.constraints.get_synthetic_accessibility_score(candidate)
                    valid_candidates.append((candidate, reason, sa_score))
            
            if valid_candidates:
                # Choose the best candidate (highest SA score)
                best_candidate = max(valid_candidates, key=lambda x: x[2])
                return best_candidate[0], best_candidate[1]
        
        # Fallback if no valid modification found
        return smiles, "No valid chemical modification found"
    
    def _add_functional_group(self, smiles: str) -> List[Tuple[str, str]]:
        """Add common functional groups to the molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        candidates = []
        
        # Add hydroxyl group
        candidates.append((smiles + "O", "Added hydroxyl group (-OH)"))
        
        # Add carboxyl group
        candidates.append((smiles + "C(=O)O", "Added carboxyl group (-COOH)"))
        
        # Add amino group
        candidates.append((smiles + "N", "Added amino group (-NH2)"))
        
        # Add methyl group
        candidates.append((smiles + "C", "Added methyl group (-CH3)"))
        
        # Add fluorine (common in drug design)
        candidates.append((smiles + "F", "Added fluorine atom"))
        
        return candidates
    
    def _modify_existing_group(self, smiles: str) -> List[Tuple[str, str]]:
        """Modify existing functional groups."""
        candidates = []
        
        # Convert alcohol to aldehyde
        if "O" in smiles and not "C(=O)" in smiles:
            modified = smiles.replace("O", "C(=O)", 1)
            candidates.append((modified, "Converted alcohol to aldehyde"))
        
        # Convert single bond to double bond
        if "C-C" in smiles:
            modified = smiles.replace("C-C", "C=C", 1)
            candidates.append((modified, "Introduced double bond"))
        
        # Add methyl to existing group
        if "C" in smiles:
            modified = smiles.replace("C", "CC", 1)
            candidates.append((modified, "Added methyl substituent"))
        
        return candidates
    
    def _add_ring(self, smiles: str) -> List[Tuple[str, str]]:
        """Add aromatic or aliphatic rings."""
        candidates = []
        
        # Add benzene ring
        candidates.append((smiles + "c1ccccc1", "Added benzene ring"))
        
        # Add cyclohexane ring
        candidates.append((smiles + "C1CCCCC1", "Added cyclohexane ring"))
        
        # Add pyridine ring
        candidates.append((smiles + "c1ccncc1", "Added pyridine ring"))
        
        return candidates
    
    def _substitute_atom(self, smiles: str) -> List[Tuple[str, str]]:
        """Substitute atoms with similar ones."""
        candidates = []
        
        # Replace carbon with nitrogen
        if "C" in smiles:
            modified = smiles.replace("C", "N", 1)
            candidates.append((modified, "Substituted carbon with nitrogen"))
        
        # Replace hydrogen with fluorine
        if "H" in smiles:
            modified = smiles.replace("H", "F", 1)
            candidates.append((modified, "Substituted hydrogen with fluorine"))
        
        return candidates
    
    def _extend_chain(self, smiles: str) -> List[Tuple[str, str]]:
        """Extend carbon chains."""
        candidates = []
        
        # Add carbon to chain
        candidates.append((smiles + "C", "Extended carbon chain"))
        
        # Add two carbons
        candidates.append((smiles + "CC", "Extended carbon chain by two atoms"))
        
        return candidates
    
    def _get_fallback_molecule(self) -> str:
        """Return a simple, valid molecule as fallback."""
        fallbacks = ["CCO", "CC(=O)O", "CCN", "C1CCCCC1"]
        return random.choice(fallbacks)
    
    def propose_multiple(self, smiles: str, n_proposals: int = 5) -> List[Tuple[str, str, float]]:
        """
        Propose multiple valid modifications.
        
        Returns:
            List of (smiles, reason, sa_score) tuples
        """
        proposals = []
        seen = set()
        
        for _ in range(n_proposals * 3):  # Try more to account for duplicates
            if len(proposals) >= n_proposals:
                break
                
            modified, reason = self.propose_smiles(smiles)
            
            if modified not in seen and modified != smiles:
                sa_score, _ = self.constraints.get_synthetic_accessibility_score(modified)
                proposals.append((modified, reason, sa_score))
                seen.add(modified)
        
        # Sort by synthetic accessibility score (descending)
        proposals.sort(key=lambda x: x[2], reverse=True)
        return proposals[:n_proposals]


def propose_smiles(smiles: str) -> Tuple[str, str]:
    """
    Backward compatibility function.
    Uses the improved agent to propose a single modification.
    """
    agent = ImprovedLLMAgent()
    return agent.propose_smiles(smiles)


if __name__ == "__main__":
    # Test the improved agent
    agent = ImprovedLLMAgent()
    
    test_molecules = ["CC", "CCO", "CC(=O)O"]
    
    print("Improved LLM Agent Testing:")
    print("=" * 50)
    
    for smiles in test_molecules:
        print(f"\nOriginal: {smiles}")
        
        # Single proposal
        modified, reason = agent.propose_smiles(smiles)
        print(f"Modified: {modified}")
        print(f"Reason: {reason}")
        
        # Multiple proposals
        proposals = agent.propose_multiple(smiles, n_proposals=3)
        print("Multiple proposals:")
        for i, (prop, reason, sa_score) in enumerate(proposals, 1):
            print(f"  {i}. {prop} (SA: {sa_score:.2f}) - {reason}")

