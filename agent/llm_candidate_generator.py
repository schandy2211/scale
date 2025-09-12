"""
LLM-based molecular candidate generator that replaces heuristic BRICS/attach operations.
"""

import json
import os
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
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1"):
        self.model = model
        
        # Set up OpenAI client - use same pattern as controller
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        print(f"🎯 LLM Candidate Generator initialized with model: {self.model}")
        
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

    def generate_candidates_with_reasoning(
        self, 
        seed_mols: List[Chem.Mol], 
        n_candidates: int,
        objective: str,
        current_best: float,
        round_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate candidates with LLM reasoning."""
        print(f"🧪 Generating {n_candidates} candidates with reasoning for {objective} objective...")
        
        # Try LLM generation first
        llm_candidates = self._generate_llm_candidates_with_reasoning(seed_mols, n_candidates, objective, current_best, round_info)
        
        if len(llm_candidates) >= n_candidates // 2:  # If we got at least half from LLM
            return llm_candidates[:n_candidates]
        
        # Fallback to BRICS with synthetic reasoning
        brics_candidates = self._generate_brics_candidates_with_reasoning(seed_mols, n_candidates - len(llm_candidates))
        
        return llm_candidates + brics_candidates[:n_candidates - len(llm_candidates)]
    
    def _generate_llm_candidates_with_reasoning(
        self, 
        seed_mols: List[Chem.Mol], 
        n_candidates: int,
        objective: str,
        current_best: float,
        round_info: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
        
        # Create prompt
        prompt = self._create_generation_prompt_with_reasoning(seed_smiles, seed_properties, objective, current_best, round_info)
        
        try:
            response = self.llm_client.generate_response(prompt)
            if response and response.strip():
                response_text = response.strip()
                print(f"🧪 LLM Candidate Generator Raw Response: '{response_text[:200]}...' (showing first 200 chars)")
            else:
                print("⚠️ LLM returned None or empty response")
            
            return self._parse_llm_response_with_reasoning(response_text)
            
        except Exception as e:
            print(f"❌ LLM API error in candidate generation: {e}")
            print(f"🔍 Exception type: {type(e).__name__}")
            return []
    
    def _generate_brics_candidates_with_reasoning(self, seed_mols: List[Chem.Mol], n_candidates: int) -> List[Dict[str, Any]]:
        """Generate BRICS candidates with synthetic reasoning."""
        candidates_with_reasoning = []
        
        try:
            frags = set()
            for m in seed_mols:
                try:
                    frags.update(BRICS.BRICSDecompose(m))
                except Exception:
                    continue
            
            if not frags:
                return candidates_with_reasoning
            
            seen = set()
            for _ in range(n_candidates * 3):  # Try more to account for failures
                if len(candidates_with_reasoning) >= n_candidates:
                    break
                    
                try:
                    # Randomly select fragments
                    frag1, frag2 = random.sample(list(frags), 2)
                    combo = Chem.CombineMols(frag1, frag2)
                    
                    # Find dummy atoms and connect them
                    dummies = [a.GetIdx() for a in combo.GetAtoms() if a.GetAtomicNum() == 0]
                    if len(dummies) >= 2:
                        em = Chem.EditableMol(combo)
                        em.AddBond(dummies[0], dummies[1], order=Chem.rdchem.BondType.SINGLE)
                        em.RemoveAtom(dummies[1])
                        em.RemoveAtom(dummies[0])
                        new_mol = em.GetMol()
                        Chem.SanitizeMol(new_mol)
                        
                        smi = Chem.MolToSmiles(new_mol, canonical=True)
                        if smi not in seen:
                            seen.add(smi)
                            candidates_with_reasoning.append({
                                'molecule': new_mol,
                                'smiles': smi,
                                'reason': f"BRICS fragment combination: {Chem.MolToSmiles(frag1)} + {Chem.MolToSmiles(frag2)}",
                                'descriptors': "fragment_combination",
                                'volatility': ""
                            })
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"BRICS generation failed: {e}")
        
        return candidates_with_reasoning
    
    def _create_generation_prompt_with_reasoning(self, seed_smiles: List[str], seed_properties: List[Dict], objective: str, current_best: float, round_info: Optional[Dict[str, Any]]) -> str:
        """Create prompt for LLM candidate generation with reasoning."""
        context = f"""You are an expert medicinal chemist and molecular designer. Generate {min(5, len(seed_smiles) * 2)} novel molecular candidates optimized for {objective.upper()} score.

CURRENT MOLECULES:
{chr(10).join([f"- {smiles}" for smiles in seed_smiles[:3]])}

CURRENT PROPERTIES:
{chr(10).join([f"- {prop['smiles']}: QED={prop['qed']}, MW={prop['mw']}, LogP={prop['logp']}, HBD={prop['hbd']}, HBA={prop['hba']}, TPSA={prop['tpsa']}" for prop in seed_properties])}

CURRENT BEST {objective.upper()} SCORE: {current_best:.3f}

OBJECTIVE: {objective.upper()}
- QED: Drug-likeness (0-1, higher is better)
- LogP: Lipophilicity (higher = more fat-soluble)
- Odor: Odorant properties for fragrance applications

GUIDELINES:
- Generate novel molecules that improve upon current candidates
- Use chemical reasoning for each modification
- Ensure molecules are chemically valid and synthetically accessible
- Focus on functional group modifications that enhance the target property

RESPONSE FORMAT:"""
        
        if objective == "odor":
            context += """
{
  "candidates": [
    {
      "smiles": "COc1ccccc1",
      "reason": "Anisole derivative with sweet floral character, ideal volatility",
      "descriptors": "floral, sweet",
      "volatility": "middle"
    },
    {
      "smiles": "CCOC(=O)C",
      "reason": "Ethyl acetate for fruity top note, high volatility",
      "descriptors": "fruity, sweet",
      "volatility": "top"
    }
  ]
}"""
        else:
            context += """
{
  "candidates": [
    {
      "smiles": "CCO",
      "reason": "Added hydroxyl group to increase polarity and drug-likeness"
    },
    {
      "smiles": "CC(=O)N",
      "reason": "Introduced amide group for better ADMET properties"
    }
  ]
}"""
        
        return context
    
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
            print(f"🔧 Sending request to {self.model} with prompt length: {len(prompt)}")
            print(f"📤 Prompt preview: '{prompt[:300]}...'")
            response = openai.chat.completions.create(
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
            
            response_text = response.choices[0].message.content
            print(f"🧬 LLM Candidate Generator Raw Response: {repr(response_text)}")
            print(f"📏 Response length: {len(response_text) if response_text else 0}")
            
            if response_text:
                response_text = response_text.strip()
                print(f"🧪 LLM Candidate Generator Cleaned Response: '{response_text[:200]}...' (showing first 200 chars)")
            else:
                print("⚠️ LLM returned None or empty response")
            
            return self._parse_llm_response(response_text)
            
        except Exception as e:
            print(f"❌ LLM API error in candidate generation: {e}")
            print(f"🔍 Exception type: {type(e).__name__}")
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
        
        # Cap the request to a reasonable size for LLM
        request_size = min(n_candidates, 20)  # Ask for max 20 candidates per LLM call
        context = f"""Generate {request_size} novel molecular candidates for optimization.

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

GUIDELINES:"""
        
        # Objective-specific guidelines
        if objective == "odor":
            context += """
- Generate odorant molecules for fragrance/flavor applications
- Target volatility window: MW 85-250 Da, LogP 0.5-4, TPSA < 40
- Consider olfactory descriptors: floral, sweet, fruity, citrus, woody, green
- Include functional groups relevant to odor: esters, ethers, alcohols, aldehydes
- Avoid known allergens: cinnamaldehyde-like, nitro groups, strong sensitizers
- Consider volatility for fragrance layering (top/middle/base notes)
- Examples: anisole (floral), ethyl acetate (fruity), terpineol (citrus)"""
        else:
            context += """
- Generate chemically valid SMILES strings
- Focus on improving {objective} while maintaining drug-likeness
- Consider Lipinski's Rule of Five (MW < 500, LogP < 5, HBD ≤ 5, HBA ≤ 10)
- Add/modify functional groups strategically
- Consider ring systems, heteroatoms, and polarity
- Avoid overly complex or unstable structures

RESPONSE FORMAT:"""
        
        if objective == "odor":
            context += """
{{
  "candidates": [
    {{
      "smiles": "COc1ccccc1",
      "reason": "Anisole derivative with sweet floral character, ideal volatility",
      "descriptors": "floral, sweet",
      "volatility": "middle"
    }},
    {{
      "smiles": "CCOC(=O)C",
      "reason": "Ethyl acetate for fruity top note, high volatility",
      "descriptors": "fruity, sweet",
      "volatility": "top"
    }}
  ]
}}"""
        else:
            context += """
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
        print(f"🔬 Parsing candidate generator response...")
        
        try:
            # Try to extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
                print(f"📄 Extracted JSON from code block")
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
                print(f"📄 Extracted JSON from braces")
            else:
                # Try to parse the entire response as JSON
                json_text = response_text
                print(f"📄 Using entire response as JSON")
            
            data = json.loads(json_text)
            print(f"✅ Successfully parsed candidate JSON with {len(data.get('candidates', []))} candidates")
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

    def _parse_llm_response_with_reasoning(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response and return molecules with their reasoning."""
        candidates_with_reasoning = []
        print(f"🔬 Parsing candidate generator response with reasoning...")
        
        try:
            # Try to extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
                print(f"📄 Extracted JSON from code block")
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
                print(f"📄 Extracted JSON from braces")
            else:
                # Try to parse the entire response as JSON
                json_text = response_text
                print(f"📄 Using entire response as JSON")
            
            data = json.loads(json_text)
            print(f"✅ Successfully parsed candidate JSON with {len(data.get('candidates', []))} candidates")
            candidates = data.get("candidates", [])
            
            for candidate in candidates:
                smiles = candidate.get("smiles", "")
                reason = candidate.get("reason", "No reasoning provided")
                descriptors = candidate.get("descriptors", "")
                volatility = candidate.get("volatility", "")
                
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        try:
                            Chem.SanitizeMol(mol)
                            candidates_with_reasoning.append({
                                'molecule': mol,
                                'smiles': smiles,
                                'reason': reason,
                                'descriptors': descriptors,
                                'volatility': volatility
                            })
                        except Exception:
                            continue
                            
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response_text[:200]}...")
        
        return candidates_with_reasoning
    
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
