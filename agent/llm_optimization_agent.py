"""
Comprehensive LLM-based optimization agent that replaces all heuristic components.
"""

import json
import random
import warnings
from typing import List, Optional, Tuple, Dict, Any, Sequence
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, Lipinski
import openai

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")


class LLMOptimizationAgent:
    """
    Comprehensive LLM-based optimization agent that handles:
    - Molecular modification strategies
    - Chemical reasoning and validation
    - Multi-objective optimization
    - Adaptive parameter adjustment
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1"):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or openai.api_key)
        self.optimization_history = []
        self.chemical_knowledge_base = self._initialize_chemical_knowledge()
    
    def _initialize_chemical_knowledge(self) -> Dict[str, Any]:
        """Initialize chemical knowledge base for context."""
        return {
            "functional_groups": {
                "polar": ["OH", "NH2", "COOH", "SO3H", "PO4H"],
                "lipophilic": ["CH3", "C2H5", "C6H5", "CF3", "Cl", "Br"],
                "hydrogen_bonding": ["OH", "NH2", "COOH", "CONH2", "SO2NH2"],
                "electron_withdrawing": ["NO2", "CN", "CF3", "SO2F", "COF"],
                "electron_donating": ["OH", "NH2", "OCH3", "CH3", "C2H5"]
            },
            "ring_systems": {
                "aromatic": ["benzene", "pyridine", "pyrimidine", "imidazole", "thiophene"],
                "aliphatic": ["cyclohexane", "cyclopentane", "piperidine", "pyrrolidine"]
            },
            "drug_likeness_rules": {
                "lipinski": {"mw": 500, "logp": 5, "hbd": 5, "hba": 10},
                "veber": {"tpsa": 140, "rotbonds": 10}
            }
        }
    
    def propose_molecular_modifications(
        self, 
        current_molecules: List[Chem.Mol],
        objective: str = "qed",
        current_scores: Optional[List[float]] = None,
        optimization_context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chem.Mol, str, Dict[str, float]]]:
        """
        Propose intelligent molecular modifications using LLM reasoning.
        
        Args:
            current_molecules: List of current molecules to modify
            objective: Optimization objective ("qed", "pen_logp", etc.)
            current_scores: Current scores for the molecules
            optimization_context: Additional context about optimization state
            
        Returns:
            List of (modified_molecule, reason, properties) tuples
        """
        modifications = []
        
        # Analyze current molecules
        molecule_analysis = self._analyze_molecules(current_molecules, current_scores)
        
        # Generate modification strategies
        strategies = self._generate_modification_strategies(
            molecule_analysis, objective, optimization_context
        )
        
        # Apply strategies to generate modifications
        for strategy in strategies:
            try:
                modifications.extend(
                    self._apply_modification_strategy(strategy, current_molecules)
                )
            except Exception as e:
                print(f"Strategy {strategy['name']} failed: {e}")
                continue
        
        # Validate and filter modifications
        valid_modifications = []
        for mol, reason, properties in modifications:
            if self._validate_modification(mol, properties):
                valid_modifications.append((mol, reason, properties))
        
        return valid_modifications[:10]  # Return top 10 modifications
    
    def _analyze_molecules(
        self, 
        molecules: List[Chem.Mol], 
        scores: Optional[List[float]]
    ) -> Dict[str, Any]:
        """Analyze current molecules to understand their properties and limitations."""
        analysis = {
            "molecules": [],
            "common_patterns": [],
            "limitations": [],
            "improvement_areas": []
        }
        
        for i, mol in enumerate(molecules[:5]):  # Analyze top 5 molecules
            try:
                properties = self._calculate_molecular_properties(mol)
                score = scores[i] if scores and i < len(scores) else 0.0
                
                analysis["molecules"].append({
                    "smiles": Chem.MolToSmiles(mol, canonical=True),
                    "properties": properties,
                    "score": score
                })
            except Exception:
                continue
        
        # Identify common patterns and limitations
        analysis["common_patterns"] = self._identify_common_patterns(analysis["molecules"])
        analysis["limitations"] = self._identify_limitations(analysis["molecules"])
        analysis["improvement_areas"] = self._identify_improvement_areas(analysis["molecules"])
        
        return analysis
    
    def _calculate_molecular_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate comprehensive molecular properties."""
        try:
            return {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "tpsa": Descriptors.TPSA(mol),
                "rotbonds": Descriptors.NumRotatableBonds(mol),
                "qed": QED.qed(mol),
                "lipinski_violations": sum([
                    1 if Descriptors.MolWt(mol) > 500 else 0,
                    1 if Descriptors.MolLogP(mol) > 5 else 0,
                    1 if Descriptors.NumHDonors(mol) > 5 else 0,
                    1 if Descriptors.NumHAcceptors(mol) > 10 else 0
                ])
            }
        except Exception:
            return {}
    
    def _identify_common_patterns(self, molecules: List[Dict]) -> List[str]:
        """Identify common structural patterns in the molecules."""
        patterns = []
        
        # Simple pattern detection
        for mol_data in molecules:
            smiles = mol_data["smiles"]
            if "c1ccccc1" in smiles:
                patterns.append("aromatic_rings")
            if "N" in smiles:
                patterns.append("nitrogen_containing")
            if "O" in smiles:
                patterns.append("oxygen_containing")
            if "S" in smiles:
                patterns.append("sulfur_containing")
        
        return list(set(patterns))
    
    def _identify_limitations(self, molecules: List[Dict]) -> List[str]:
        """Identify limitations in current molecules."""
        limitations = []
        
        for mol_data in molecules:
            props = mol_data.get("properties", {})
            
            if props.get("molecular_weight", 0) > 500:
                limitations.append("high_molecular_weight")
            if props.get("logp", 0) > 5:
                limitations.append("high_lipophilicity")
            if props.get("hbd", 0) > 5:
                limitations.append("too_many_hbd")
            if props.get("hba", 0) > 10:
                limitations.append("too_many_hba")
            if props.get("tpsa", 0) > 140:
                limitations.append("high_tpsa")
            if props.get("lipinski_violations", 0) > 0:
                limitations.append("lipinski_violations")
        
        return list(set(limitations))
    
    def _identify_improvement_areas(self, molecules: List[Dict]) -> List[str]:
        """Identify areas for improvement."""
        improvements = []
        
        avg_qed = sum(mol.get("properties", {}).get("qed", 0) for mol in molecules) / max(len(molecules), 1)
        
        if avg_qed < 0.5:
            improvements.append("increase_drug_likeness")
        if any(mol.get("properties", {}).get("molecular_weight", 0) < 200 for mol in molecules):
            improvements.append("increase_molecular_size")
        if any(mol.get("properties", {}).get("logp", 0) < 0 for mol in molecules):
            improvements.append("increase_lipophilicity")
        
        return improvements
    
    def _generate_modification_strategies(
        self, 
        analysis: Dict[str, Any], 
        objective: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate intelligent modification strategies using LLM."""
        
        prompt = self._create_strategy_prompt(analysis, objective, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medicinal chemist. Analyze the molecular optimization context and propose specific modification strategies. Always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1500
            )
            
            response_text = response.choices[0].message.content.strip()
            return self._parse_strategy_response(response_text)
            
        except Exception as e:
            print(f"LLM strategy generation failed: {e}")
            return self._get_fallback_strategies(analysis, objective)
    
    def _create_strategy_prompt(
        self, 
        analysis: Dict[str, Any], 
        objective: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Create a detailed prompt for strategy generation."""
        
        prompt = f"""Analyze these molecules and propose modification strategies for {objective.upper()} optimization.

CURRENT MOLECULES:
{json.dumps(analysis['molecules'], indent=2)}

COMMON PATTERNS: {', '.join(analysis['common_patterns'])}
LIMITATIONS: {', '.join(analysis['limitations'])}
IMPROVEMENT AREAS: {', '.join(analysis['improvement_areas'])}

OPTIMIZATION CONTEXT:"""
        
        if context:
            prompt += f"""
- Round: {context.get('round', 'N/A')}
- Progress: {context.get('progress', 'N/A')}
- Best Score: {context.get('best_score', 'N/A')}"""
        
        prompt += f"""

CHEMICAL KNOWLEDGE:
- Polar groups: {', '.join(self.chemical_knowledge_base['functional_groups']['polar'])}
- Lipophilic groups: {', '.join(self.chemical_knowledge_base['functional_groups']['lipophilic'])}
- H-bonding groups: {', '.join(self.chemical_knowledge_base['functional_groups']['hydrogen_bonding'])}

PROPOSE 3-5 SPECIFIC MODIFICATION STRATEGIES:

Return JSON:
{{
  "strategies": [
    {{
      "name": "increase_polarity",
      "description": "Add polar functional groups to improve solubility",
      "modifications": ["add_OH", "add_NH2", "add_COOH"],
      "target_properties": {{"tpsa": "increase", "logp": "decrease"}},
      "rationale": "Current molecules may be too lipophilic"
    }},
    {{
      "name": "optimize_size",
      "description": "Adjust molecular size for better drug-likeness",
      "modifications": ["add_ring", "extend_chain", "add_branch"],
      "target_properties": {{"mw": "optimize"}},
      "rationale": "Molecules may be too small for optimal activity"
    }}
  ]
}}"""
        
        return prompt
    
    def _parse_strategy_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM strategy response."""
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            
            data = json.loads(json_text)
            return data.get("strategies", [])
            
        except Exception as e:
            print(f"Failed to parse strategy response: {e}")
            return []
    
    def _get_fallback_strategies(self, analysis: Dict[str, Any], objective: str) -> List[Dict[str, Any]]:
        """Get fallback strategies when LLM fails."""
        strategies = []
        
        if "high_molecular_weight" in analysis["limitations"]:
            strategies.append({
                "name": "reduce_size",
                "description": "Reduce molecular size",
                "modifications": ["remove_ring", "shorten_chain"],
                "target_properties": {"mw": "decrease"}
            })
        
        if "increase_drug_likeness" in analysis["improvement_areas"]:
            strategies.append({
                "name": "improve_drug_likeness",
                "description": "Improve drug-like properties",
                "modifications": ["add_aromatic", "add_heteroatom"],
                "target_properties": {"qed": "increase"}
            })
        
        return strategies
    
    def _apply_modification_strategy(
        self, 
        strategy: Dict[str, Any], 
        molecules: List[Chem.Mol]
    ) -> List[Tuple[Chem.Mol, str, Dict[str, float]]]:
        """Apply a modification strategy to generate new molecules."""
        modifications = []
        
        for mol in molecules[:3]:  # Apply to top 3 molecules
            for modification_type in strategy.get("modifications", []):
                try:
                    modified_mol = self._apply_single_modification(mol, modification_type)
                    if modified_mol is not None:
                        properties = self._calculate_molecular_properties(modified_mol)
                        reason = f"{strategy['name']}: {modification_type}"
                        modifications.append((modified_mol, reason, properties))
                except Exception:
                    continue
        
        return modifications
    
    def _apply_single_modification(self, mol: Chem.Mol, modification_type: str) -> Optional[Chem.Mol]:
        """Apply a single modification to a molecule."""
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            
            if modification_type == "add_OH":
                return Chem.MolFromSmiles(smiles + "O")
            elif modification_type == "add_NH2":
                return Chem.MolFromSmiles(smiles + "N")
            elif modification_type == "add_COOH":
                return Chem.MolFromSmiles(smiles + "C(=O)O")
            elif modification_type == "add_aromatic":
                return Chem.MolFromSmiles(smiles + "c1ccccc1")
            elif modification_type == "add_ring":
                return Chem.MolFromSmiles(smiles + "C1CCCCC1")
            elif modification_type == "extend_chain":
                return Chem.MolFromSmiles(smiles + "C")
            else:
                return None
                
        except Exception:
            return None
    
    def _validate_modification(self, mol: Chem.Mol, properties: Dict[str, float]) -> bool:
        """Validate that a modification is chemically reasonable."""
        try:
            # Basic chemical validation
            Chem.SanitizeMol(mol)
            
            # Property-based validation
            if properties.get("molecular_weight", 0) > 1000:
                return False
            if properties.get("lipinski_violations", 0) > 2:
                return False
            
            return True
            
        except Exception:
            return False


def create_llm_optimization_agent(api_key: Optional[str] = None) -> LLMOptimizationAgent:
    """Convenience function to create an LLM optimization agent."""
    return LLMOptimizationAgent(api_key=api_key)
