"""
LLM-based agent for molecular modification using OpenAI API.
"""

import os
import json
import warnings
from typing import List, Tuple, Optional
from rdkit import Chem
import openai

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")

from .chemical_constraints import ChemicalConstraints


class LLMAgent:
    """LLM-based agent that uses GPT to propose molecular modifications."""
    
    def __init__(self, 
                 constraints: ChemicalConstraints = None,
                 api_key: Optional[str] = None,
                 model: str = "gpt-5"):
        """
        Initialize the LLM agent.
        
        Args:
            constraints: Chemical constraints validator
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var.
            model: OpenAI model to use (default: gpt-5 for cost efficiency)
        """
        self.constraints = constraints or ChemicalConstraints()
        self.model = model
        
        # Set up OpenAI client
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
    
    def propose_smiles(self, smiles: str) -> Tuple[str, str]:
        """
        Use LLM to propose a chemically valid modification to a SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            (modified_smiles, reason)
        """
        # Validate input
        is_valid, reason = self.constraints.is_chemically_valid(smiles)
        if not is_valid:
            return self._get_fallback_molecule(), f"Input invalid: {reason}"
        
        try:
            # Create prompt for molecular modification
            prompt = self._create_modification_prompt(smiles)
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medicinal chemist. You propose small, chemically reasonable modifications to molecules to improve their drug-like properties. Always respond with valid SMILES strings and brief explanations."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=150
            )
            
            response_text = response.choices[0].message.content.strip()
            return self._parse_modification_response(response_text, smiles)
            
        except Exception as e:
            print(f"LLM API error: {e}. Falling back to rule-based modification.")
            return self._fallback_modification(smiles)
    
    def _create_modification_prompt(self, smiles: str) -> str:
        """Create a prompt for molecular modification."""
        prompt = f"""
Given this molecule: {smiles}

Propose a small, chemically reasonable modification to improve its drug-like properties. Consider:
- Adding functional groups (OH, NH2, F, Cl, etc.)
- Modifying existing groups
- Adding small rings
- Substituting atoms

Respond with:
1. The modified SMILES string
2. A brief explanation of the change

Example format:
Modified SMILES: CCO
Explanation: Added hydroxyl group to increase polarity

Your response:
"""
        return prompt
    
    def _parse_modification_response(self, response_text: str, original_smiles: str) -> Tuple[str, str]:
        """Parse LLM response to extract modified SMILES and explanation."""
        try:
            # Try to extract SMILES and explanation
            lines = response_text.strip().split('\n')
            modified_smiles = None
            explanation = "LLM proposed modification"
            
            for line in lines:
                line = line.strip()
                if line.startswith("Modified SMILES:") or line.startswith("SMILES:"):
                    modified_smiles = line.split(":", 1)[1].strip()
                elif line.startswith("Explanation:"):
                    explanation = line.split(":", 1)[1].strip()
            
            # If no clear format, try to extract SMILES from the text
            if not modified_smiles:
                # Look for potential SMILES patterns
                words = response_text.split()
                for word in words:
                    if self._looks_like_smiles(word):
                        modified_smiles = word
                        break
            
            if not modified_smiles:
                raise ValueError("No SMILES found in response")
            
            # Validate the proposed SMILES
            mol = Chem.MolFromSmiles(modified_smiles)
            if mol is None:
                raise ValueError("Invalid SMILES proposed")
            
            # Check if it's chemically valid
            is_valid, validity_reason = self.constraints.is_chemically_valid(modified_smiles)
            if not is_valid:
                raise ValueError(f"Chemically invalid: {validity_reason}")
            
            return modified_smiles, explanation
            
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response_text}")
            return self._fallback_modification(original_smiles)
    
    def _looks_like_smiles(self, text: str) -> bool:
        """Simple heuristic to check if text looks like a SMILES string."""
        if not text or len(text) < 2:
            return False
        
        # SMILES typically contain letters, numbers, and some special characters
        smiles_chars = set("CNOSPFClBrI()[]=#+-\\/")
        if all(c in smiles_chars or c.isalnum() for c in text):
            # Try to create a molecule from it
            mol = Chem.MolFromSmiles(text)
            return mol is not None
        
        return False
    
    def _fallback_modification(self, smiles: str) -> Tuple[str, str]:
        """Fallback to simple rule-based modification if LLM fails."""
        # Simple rule-based modifications
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._get_fallback_molecule(), "Invalid input, using fallback"
        
        # Try adding a hydroxyl group
        modified = smiles + "O"
        mol_modified = Chem.MolFromSmiles(modified)
        if mol_modified is not None:
            is_valid, _ = self.constraints.is_chemically_valid(modified)
            if is_valid:
                return modified, "Added hydroxyl group (fallback)"
        
        # Try adding a methyl group
        modified = smiles + "C"
        mol_modified = Chem.MolFromSmiles(modified)
        if mol_modified is not None:
            is_valid, _ = self.constraints.is_chemically_valid(modified)
            if is_valid:
                return modified, "Added methyl group (fallback)"
        
        # Return original if no modification works
        return smiles, "No valid modification found (fallback)"
    
    def _get_fallback_molecule(self) -> str:
        """Return a simple, valid molecule as fallback."""
        fallbacks = ["CCO", "CC(=O)O", "CCN", "C1CCCCC1"]
        for fallback in fallbacks:
            is_valid, _ = self.constraints.is_chemically_valid(fallback)
            if is_valid:
                return fallback
        return "CCO"  # Default fallback
    
    def propose_multiple(self, smiles: str, n_proposals: int = 5) -> List[Tuple[str, str, float]]:
        """
        Propose multiple valid modifications using LLM.
        
        Args:
            smiles: Input SMILES string
            n_proposals: Number of proposals to generate
            
        Returns:
            List of (smiles, reason, sa_score) tuples
        """
        proposals = []
        seen = set()
        
        for _ in range(n_proposals * 2):  # Try more to account for duplicates
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


# Backward compatibility function
def propose_smiles(smiles: str) -> Tuple[str, str]:
    """
    Backward compatibility function.
    Uses the LLM agent to propose a single modification.
    """
    agent = LLMAgent()
    return agent.propose_smiles(smiles)
