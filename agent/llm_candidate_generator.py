"""
LLM-based molecular candidate generator that replaces heuristic BRICS/attach operations.
"""

import json
import random
import warnings
from typing import List, Optional, Tuple, Dict, Any
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors, QED
import openai

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")


class LLMCandidateGenerator:
    """LLM-based candidate generator that uses GPT to intelligently propose molecular modifications."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or openai.api_key)
        
        # Fallback fragment library for when LLM fails
        self.fallback_frags = [
            "[*:1]C",              # methyl
            "[*:1]CC",             # ethyl
            "[*:1]O",              # hydroxyl
            "[*:1]OC",             # methoxy
            "[*:1]N",              # amino
            "[*:1]F",              # fluoro
            "[*:1]Cl",             # chloro
            "[*:1]c1ccccc1",       # phenyl
            "[*:1]c1ccncc1",       # pyridine
            "[*:1]C(=O)N",         # amide
            "[*:1]C#N",            # cyano
            "[*:1]S(=O)(=O)N",     # sulfonamide
        ]
        
        # Cache for generated candidates to avoid duplicates
        self.generated_cache = set()
    
    def generate_candidates(
        self, 
        seed_mols: List[Chem.Mol], 
        n_candidates: int = 500,
        objective: str = "qed",
        current_best: float = 0.0,
        round_info: Optional[Dict[str, Any]] = None
    ) -> List[Chem.Mol]:
        """
        Generate molecular candidates using LLM-based reasoning.
        
        Args:
            seed_mols: List of seed molecules to modify
            n_candidates: Number of candidates to generate
            objective: Optimization objective ("qed", "pen_logp", etc.)
            current_best: Current best score for context
            round_info: Additional round information for context
            
        Returns:
            List of valid molecular candidates
        """
        candidates = []
        seen = set()
        
        # Try LLM-based generation first
        try:
            llm_candidates = self._generate_llm_candidates(
                seed_mols, n_candidates, objective, current_best, round_info
            )
            
            for mol in llm_candidates:
                if mol is not None:
                    smi = Chem.MolToSmiles(mol, canonical=True)
                    if smi not in seen:
                        seen.add(smi)
                        candidates.append(mol)
                        if len(candidates) >= n_candidates:
                            break
                            
        except Exception as e:
            print(f"LLM candidate generation failed: {e}. Falling back to heuristic methods.")
        
        # Fallback to BRICS if we don't have enough candidates
        if len(candidates) < n_candidates // 2:
            brics_candidates = self._generate_brics_candidates(seed_mols, n_candidates - len(candidates))
            for mol in brics_candidates:
                if mol is not None:
                    smi = Chem.MolToSmiles(mol, canonical=True)
                    if smi not in seen:
                        seen.add(smi)
                        candidates.append(mol)
                        if len(candidates) >= n_candidates:
                            break
        
        # Final fallback to simple attachment
        if len(candidates) < n_candidates // 4:
            attach_candidates = self._generate_attach_candidates(seed_mols, n_candidates - len(candidates))
            for mol in attach_candidates:
                if mol is not None:
                    smi = Chem.MolToSmiles(mol, canonical=True)
                    if smi not in seen:
                        seen.add(smi)
                        candidates.append(mol)
                        if len(candidates) >= n_candidates:
                            break
        
        return candidates[:n_candidates]
    
    def _generate_llm_candidates(
        self, 
        seed_mols: List[Chem.Mol], 
        n_candidates: int,
        objective: str,
        current_best: float,
        round_info: Optional[Dict[str, Any]]
    ) -> List[Chem.Mol]:
        """Generate candidates using LLM reasoning."""
        
        # Prepare context about current molecules
        seed_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in seed_mols[:5]]  # Limit to first 5
        seed_properties = []
        
        for mol in seed_mols[:3]:  # Analyze first 3 in detail
            try:
                qed = QED.qed(mol)
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                tpsa = Descriptors.TPSA(mol)
                
                seed_properties.append({
                    "smiles": Chem.MolToSmiles(mol, canonical=True),
                    "qed": round(qed, 3),
                    "mw": round(mw, 1),
                    "logp": round(logp, 2),
                    "hbd": hbd,
                    "hba": hba,
                    "tpsa": round(tpsa, 1)
                })
            except Exception:
                continue
        
        # Create prompt for LLM
        prompt = self._create_generation_prompt(
            seed_smiles, seed_properties, objective, current_best, round_info, n_candidates
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medicinal chemist and molecular optimization specialist. You generate chemically valid SMILES strings that improve drug-like properties. Always respond with valid JSON containing an array of SMILES strings and brief explanations."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            return self._parse_llm_response(response_text)
            
        except Exception as e:
            print(f"LLM API error in candidate generation: {e}")
            return []
    
    def _create_generation_prompt(
        self, 
        seed_smiles: List[str], 
        seed_properties: List[Dict], 
        objective: str, 
        current_best: float,
        round_info: Optional[Dict[str, Any]],
        n_candidates: int
    ) -> str:
        """Create a detailed prompt for candidate generation."""
        
        context = f"""Generate {n_candidates} novel molecular candidates for optimization.

OBJECTIVE: {objective.upper()}
CURRENT BEST SCORE: {current_best:.3f}

SEED MOLECULES:
{chr(10).join(seed_smiles)}

DETAILED PROPERTIES:
{json.dumps(seed_properties, indent=2)}

OPTIMIZATION CONTEXT:"""
        
        if round_info:
            context += f"""
- Round: {round_info.get('round', 'N/A')}
- Progress: {round_info.get('progress', 'N/A')}
- Strategy: {round_info.get('strategy', 'N/A')}"""
        
        context += f"""

GUIDELINES:
- Generate chemically valid SMILES strings
- Focus on improving {objective} while maintaining drug-likeness
- Consider Lipinski's Rule of Five (MW < 500, LogP < 5, HBD ≤ 5, HBA ≤ 10)
- Add/modify functional groups strategically
- Consider ring systems, heteroatoms, and polarity
- Avoid overly complex or unstable structures

RESPONSE FORMAT:
{{
  "candidates": [
    {{
      "smiles": "CCO",
      "reason": "Added hydroxyl group to increase polarity and drug-likeness"
    }},
    {{
      "smiles": "CC(=O)N",
      "reason": "Introduced amide group for better ADMET properties"
    }}
  ]
}}"""
        
        return context
    
    def _parse_llm_response(self, response_text: str) -> List[Chem.Mol]:
        """Parse LLM response and convert to molecules."""
        molecules = []
        
        try:
            # Try to extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                # Try to parse the entire response as JSON
                json_text = response_text
            
            data = json.loads(json_text)
            candidates = data.get("candidates", [])
            
            for candidate in candidates:
                smiles = candidate.get("smiles", "")
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        try:
                            Chem.SanitizeMol(mol)
                            molecules.append(mol)
                        except Exception:
                            continue
                            
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response_text[:200]}...")
        
        return molecules
    
    def _generate_brics_candidates(self, seed_mols: List[Chem.Mol], n_candidates: int) -> List[Chem.Mol]:
        """Fallback BRICS candidate generation."""
        try:
            frags = set()
            for m in seed_mols:
                try:
                    frags |= set(BRICS.BRICSDecompose(m, returnMols=True))
                except Exception:
                    continue
            
            if not frags:
                return []
            
            built = []
            seen = set()
            
            try:
                gen = BRICS.BRICSBuild(frags)
            except Exception:
                gen = BRICS.BRICSBuild(list(frags))
            
            for cand in gen:
                if isinstance(cand, Chem.Mol):
                    mol = cand
                else:
                    mol = Chem.MolFromSmiles(str(cand))
                
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                        smi = Chem.MolToSmiles(mol, canonical=True)
                        if smi not in seen:
                            seen.add(smi)
                            built.append(mol)
                            if len(built) >= n_candidates:
                                break
                    except Exception:
                        continue
            
            return built
            
        except Exception as e:
            print(f"BRICS generation failed: {e}")
            return []
    
    def _generate_attach_candidates(self, seed_mols: List[Chem.Mol], n_candidates: int) -> List[Chem.Mol]:
        """Fallback simple attachment generation."""
        built = []
        seen = set()
        
        frags = [Chem.MolFromSmiles(s) for s in self.fallback_frags]
        frags = [f for f in frags if f is not None]
        
        for m in seed_mols:
            atoms = [
                a.GetIdx() for a in m.GetAtoms()
                if a.GetAtomicNum() in (6, 7) and a.GetImplicitValence() > 0
            ]
            if not atoms:
                continue
                
            for _ in range(max(10, n_candidates // max(1, len(seed_mols)))):
                if len(built) >= n_candidates:
                    break
                    
                aidx = random.choice(atoms)
                frag = random.choice(frags)
                cand = self._attach_fragment_once(m, frag, aidx)
                
                if cand is not None:
                    try:
                        Chem.SanitizeMol(cand)
                        smi = Chem.MolToSmiles(cand, canonical=True)
                        if smi not in seen:
                            seen.add(smi)
                            built.append(cand)
                    except Exception:
                        continue
        
        return built
    
    def _attach_fragment_once(self, base: Chem.Mol, frag: Chem.Mol, atom_idx: int) -> Optional[Chem.Mol]:
        """Attach a fragment to a base molecule."""
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
            em.AddBond(atom_idx + base_offset, n_idx + frag_offset, order=Chem.rdchem.BondType.SINGLE)
            em.RemoveAtom(d_idx + frag_offset)
            new_mol = em.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol
        except Exception:
            return None


def generate_candidates_llm(
    seed_mols: List[Chem.Mol], 
    n_candidates: int = 500,
    objective: str = "qed",
    current_best: float = 0.0,
    round_info: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None
) -> List[Chem.Mol]:
    """
    Convenience function for LLM-based candidate generation.
    
    Args:
        seed_mols: List of seed molecules
        n_candidates: Number of candidates to generate
        objective: Optimization objective
        current_best: Current best score
        round_info: Round information
        api_key: OpenAI API key
        
    Returns:
        List of molecular candidates
    """
    generator = LLMCandidateGenerator(api_key=api_key)
    return generator.generate_candidates(
        seed_mols, n_candidates, objective, current_best, round_info
    )
