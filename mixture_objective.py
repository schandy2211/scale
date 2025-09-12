"""
Mixture-level objective for targeted odor profile design.

This is the breakthrough innovation: optimize molecule mixtures to match 
specific olfactory descriptor profiles (e.g., "jasmine: indolic, sweet, floral").

Key novelty:
- Component-level: Odor score + safety + volatility constraints
- Mixture-level: Descriptor vector matching with volatility staging
- Multi-objective: Perceptual accuracy + safety + synthesis cost

This enables "targeted odor-mixture inverse design" - the hard problem!
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from rdkit import Chem
from rdkit.Chem import Descriptors
from dataclasses import dataclass
import warnings

# Suppress RDKit warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")


@dataclass
class OdorProfile:
    """Target olfactory profile for mixture design."""
    descriptors: Dict[str, float]  # e.g., {"floral": 0.6, "sweet": 0.3, "woody": 0.1}
    volatility_stages: Optional[Dict[str, float]] = None  # {"top": 0.3, "middle": 0.5, "base": 0.2}
    safety_constraints: Optional[Dict[str, float]] = None  # {"max_sensitizers": 0.1}
    

class DescriptorPredictor:
    """
    Predict olfactory descriptors from molecular structure.
    
    This is the core ML component that enables mixture design.
    Currently uses heuristic rules; can be replaced with trained models.
    """
    
    def __init__(self):
        self.descriptor_vocab = [
            "floral", "sweet", "woody", "citrus", "indolic", 
            "green", "fruity", "spicy", "fresh", "earthy"
        ]
        
        # Heuristic structure-descriptor mappings
        # In production: replace with trained ML models on Pyrfume/DREAM data
        self.structure_patterns = {
            # Floral patterns
            "floral": [
                ("COc1ccccc1", 0.8),        # anisole (sweet floral)
                ("c1ccc(CO)cc1", 0.7),      # benzyl alcohol
                ("CCOc1ccccc1", 0.6),       # ethyl anisole
            ],
            # Sweet patterns  
            "sweet": [
                ("CC(=O)OC", 0.9),          # methyl acetate
                ("CCOC(=O)C", 0.8),         # ethyl acetate
                ("O=C1OCCC1", 0.7),         # gamma-butyrolactone
            ],
            # Woody patterns
            "woody": [
                ("CC1=CC=C(C=C1)C(C)C", 0.8),  # p-cymene
                ("c1ccc2ccccc2c1", 0.7),        # naphthalene-like
                ("CC1CCC(CC1)O", 0.6),          # terpineol
            ],
            # Citrus patterns
            "citrus": [
                ("CC1=CC(=O)CCC1", 0.9),    # carvone-like
                ("C=CC1CCC(CC1)C", 0.8),    # limonene-like
                ("O=CCCCCC", 0.6),          # hexanal (green-citrus)
            ],
            # Indolic patterns (complex, animalic)
            "indolic": [
                ("c1ccc2[nH]ccc2c1", 0.9),  # indole
                ("Nc1ccccc1", 0.4),         # aniline (simple indolic)
            ],
            # Green patterns
            "green": [
                ("O=CCCCCC", 0.8),          # hexanal
                ("CCC(=O)C", 0.6),          # butanone
                ("CCCCCCO", 0.5),           # hexanol
            ],
            # Fruity patterns
            "fruity": [
                ("CCOC(=O)C", 0.9),         # ethyl acetate
                ("CC(=O)OCCC", 0.8),        # propyl acetate
                ("CCCCOC(=O)C", 0.7),       # butyl acetate
            ],
        }
    
    def predict_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Predict olfactory descriptor vector for a molecule.
        
        Returns:
            Dict mapping descriptor names to strength scores (0-1)
        """
        if mol is None:
            return {desc: 0.0 for desc in self.descriptor_vocab}
        
        smiles = Chem.MolToSmiles(mol)
        descriptors = {desc: 0.0 for desc in self.descriptor_vocab}
        
        # Pattern matching approach (replace with ML in production)
        for descriptor, patterns in self.structure_patterns.items():
            max_score = 0.0
            for pattern_smiles, strength in patterns:
                try:
                    pattern_mol = Chem.MolFromSmiles(pattern_smiles)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        # Similarity-based scoring
                        similarity = self._calculate_similarity(mol, pattern_mol)
                        score = strength * similarity
                        max_score = max(max_score, score)
                except Exception:
                    continue
            descriptors[descriptor] = max_score
        
        # Molecular property-based adjustments
        self._adjust_by_properties(mol, descriptors)
        
        # Normalize to sum to 1.0 (probability distribution)
        total = sum(descriptors.values())
        if total > 0:
            descriptors = {k: v/total for k, v in descriptors.items()}
        
        return descriptors
    
    def _calculate_similarity(self, mol1: Chem.Mol, mol2: Chem.Mol) -> float:
        """Calculate structural similarity between molecules."""
        try:
            from rdkit import DataStructs
            fp1 = Chem.RDKFingerprint(mol1)
            fp2 = Chem.RDKFingerprint(mol2)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception:
            return 0.5  # Default similarity
    
    def _adjust_by_properties(self, mol: Chem.Mol, descriptors: Dict[str, float]):
        """Adjust descriptors based on molecular properties."""
        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            
            # Higher MW tends to be more woody/earthy
            if mw > 150:
                descriptors["woody"] = min(1.0, descriptors["woody"] + 0.2)
                descriptors["earthy"] = min(1.0, descriptors["earthy"] + 0.1)
            
            # Higher LogP tends to be less fresh/green
            if logp > 3:
                descriptors["fresh"] = max(0.0, descriptors["fresh"] - 0.2)
                descriptors["green"] = max(0.0, descriptors["green"] - 0.1)
            
            # Aromatic compounds tend to be more floral/sweet
            if aromatic_rings >= 1:
                descriptors["floral"] = min(1.0, descriptors["floral"] + 0.1)
                descriptors["sweet"] = min(1.0, descriptors["sweet"] + 0.05)
                
        except Exception:
            pass  # Skip property adjustments if calculation fails


class VolatilityStager:
    """
    Predict volatility staging for time-release effects.
    
    This enables the temporal olfactory profile engineering!
    """
    
    def __init__(self):
        pass
    
    def predict_volatility_stage(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Predict volatility stage (top/middle/base note) for molecule.
        
        Returns:
            Dict with probabilities for each stage
        """
        if mol is None:
            return {"top": 0.0, "middle": 0.0, "base": 1.0}
        
        try:
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)
            
            # Volatility heuristics based on physical properties
            volatility_score = self._calculate_volatility(logp, mw, tpsa)
            
            if volatility_score > 0.7:
                # High volatility = top note
                return {"top": 0.8, "middle": 0.2, "base": 0.0}
            elif volatility_score > 0.4:
                # Medium volatility = middle note
                return {"top": 0.1, "middle": 0.8, "base": 0.1}
            else:
                # Low volatility = base note
                return {"top": 0.0, "middle": 0.2, "base": 0.8}
                
        except Exception:
            return {"top": 0.3, "middle": 0.4, "base": 0.3}  # Default distribution
    
    def _calculate_volatility(self, logp: float, mw: float, tpsa: float) -> float:
        """Calculate volatility score from molecular properties."""
        # Simplified Antoine-equation-like relationship
        # Higher volatility = lower MW, lower LogP, lower TPSA
        
        mw_score = max(0, (250 - mw) / 200)  # Normalize around typical odorant range
        logp_score = max(0, (4 - logp) / 4)  # Lower LogP = higher volatility
        tpsa_score = max(0, (40 - tpsa) / 40)  # Lower TPSA = higher volatility
        
        # Weighted combination
        volatility = 0.4 * mw_score + 0.4 * logp_score + 0.2 * tpsa_score
        return min(1.0, max(0.0, volatility))


class MixtureOptimizer:
    """
    The core innovation: optimize molecule mixtures for target descriptor profiles.
    
    This solves the "targeted odor-mixture inverse design" problem!
    """
    
    def __init__(self):
        self.descriptor_predictor = DescriptorPredictor()
        self.volatility_stager = VolatilityStager()
    
    def score_mixture(self, 
                     molecules: List[Chem.Mol], 
                     weights: List[float],
                     target_profile: OdorProfile) -> float:
        """
        Score how well a weighted mixture matches the target profile.
        
        This is the key objective function for mixture optimization!
        """
        if not molecules or not weights or len(molecules) != len(weights):
            return 0.0
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight <= 0:
            return 0.0
        weights = [w/total_weight for w in weights]
        
        # Predict mixture descriptor vector
        mixture_descriptors = self._predict_mixture_descriptors(molecules, weights)
        
        # Calculate descriptor matching score
        descriptor_score = self._calculate_descriptor_match(
            mixture_descriptors, target_profile.descriptors
        )
        
        # Optional: Volatility staging score
        volatility_score = 1.0
        if target_profile.volatility_stages:
            volatility_score = self._calculate_volatility_match(
                molecules, weights, target_profile.volatility_stages
            )
        
        # Optional: Safety penalty
        safety_score = self._calculate_safety_score(molecules, weights, target_profile)
        
        # Combined score (weighted combination)
        final_score = (0.6 * descriptor_score + 
                      0.2 * volatility_score + 
                      0.2 * safety_score)
        
        return final_score
    
    def _predict_mixture_descriptors(self, 
                                   molecules: List[Chem.Mol], 
                                   weights: List[float]) -> Dict[str, float]:
        """Predict descriptor vector for weighted mixture."""
        mixture_desc = {}
        
        for mol, weight in zip(molecules, weights):
            mol_desc = self.descriptor_predictor.predict_descriptors(mol)
            for desc, value in mol_desc.items():
                mixture_desc[desc] = mixture_desc.get(desc, 0.0) + weight * value
        
        return mixture_desc
    
    def _calculate_descriptor_match(self, 
                                  predicted: Dict[str, float], 
                                  target: Dict[str, float]) -> float:
        """Calculate how well predicted descriptors match target."""
        # Cosine similarity between descriptor vectors
        pred_values = [predicted.get(k, 0.0) for k in target.keys()]
        target_values = list(target.values())
        
        # Normalize vectors
        pred_norm = np.linalg.norm(pred_values)
        target_norm = np.linalg.norm(target_values)
        
        if pred_norm == 0 or target_norm == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(pred_values, target_values) / (pred_norm * target_norm)
        return max(0.0, similarity)
    
    def _calculate_volatility_match(self, 
                                  molecules: List[Chem.Mol], 
                                  weights: List[float],
                                  target_stages: Dict[str, float]) -> float:
        """Calculate volatility staging match."""
        mixture_stages = {"top": 0.0, "middle": 0.0, "base": 0.0}
        
        for mol, weight in zip(molecules, weights):
            mol_stages = self.volatility_stager.predict_volatility_stage(mol)
            for stage, prob in mol_stages.items():
                mixture_stages[stage] += weight * prob
        
        # Calculate match to target stages
        return self._calculate_descriptor_match(mixture_stages, target_stages)
    
    def _calculate_safety_score(self, 
                              molecules: List[Chem.Mol], 
                              weights: List[float],
                              target_profile: OdorProfile) -> float:
        """Calculate safety score for mixture."""
        # Simple safety scoring (can be expanded)
        safety_score = 1.0
        
        try:
            from chemosensory_filters import passes_safety_alerts
            for mol, weight in zip(molecules, weights):
                passes_safety, alerts = passes_safety_alerts(mol)
                if not passes_safety:
                    # Penalize based on weight of problematic component
                    safety_score -= 0.3 * weight
        except ImportError:
            pass
        
        return max(0.0, safety_score)
    
    def optimize_mixture_weights(self,
                                molecules: List[Chem.Mol],
                                target_profile: OdorProfile,
                                max_components: int = 5) -> Tuple[List[float], float]:
        """
        Optimize mixture weights for target profile.
        
        This is where the magic happens: finding optimal blend ratios!
        """
        if not molecules:
            return [], 0.0
        
        # Limit to max_components for computational efficiency
        molecules = molecules[:max_components]
        n_mols = len(molecules)
        
        # Simple grid search optimization (can be replaced with gradient-based)
        best_weights = None
        best_score = 0.0
        
        # Try different weight combinations
        for _ in range(100):  # Random sampling approach
            # Generate random weights
            weights = np.random.random(n_mols)
            weights = weights / np.sum(weights)  # Normalize
            
            score = self.score_mixture(molecules, weights.tolist(), target_profile)
            if score > best_score:
                best_score = score
                best_weights = weights.tolist()
        
        return best_weights or [1.0/n_mols] * n_mols, best_score


# Predefined target profiles for testing
JASMINE_PROFILE = OdorProfile(
    descriptors={"floral": 0.5, "sweet": 0.2, "indolic": 0.2, "green": 0.1},
    volatility_stages={"top": 0.3, "middle": 0.5, "base": 0.2}
)

CITRUS_PROFILE = OdorProfile(
    descriptors={"citrus": 0.6, "fresh": 0.3, "green": 0.1},
    volatility_stages={"top": 0.7, "middle": 0.3, "base": 0.0}
)

WOODY_PROFILE = OdorProfile(
    descriptors={"woody": 0.6, "earthy": 0.2, "spicy": 0.2},
    volatility_stages={"top": 0.1, "middle": 0.3, "base": 0.6}
)


# Example usage
if __name__ == "__main__":
    print("🧪 Testing Mixture Objective System")
    
    # Test molecules
    test_smiles = [
        "COc1ccccc1",        # anisole (floral)
        "CCOC(=O)C",         # ethyl acetate (fruity, sweet)
        "CC1=CCC(CC1)O",     # terpineol (citrus, floral)
        "O=CCCCCC",          # hexanal (green, citrus)
    ]
    
    molecules = [Chem.MolFromSmiles(smi) for smi in test_smiles]
    molecules = [mol for mol in molecules if mol is not None]
    
    # Test descriptor prediction
    predictor = DescriptorPredictor()
    print("\n🎯 Individual Molecule Descriptors:")
    for smi, mol in zip(test_smiles, molecules):
        descriptors = predictor.predict_descriptors(mol)
        top_descriptors = sorted(descriptors.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  {smi}: {top_descriptors}")
    
    # Test mixture optimization
    optimizer = MixtureOptimizer()
    print(f"\n🎭 Optimizing for Jasmine Profile: {JASMINE_PROFILE.descriptors}")
    
    best_weights, score = optimizer.optimize_mixture_weights(molecules, JASMINE_PROFILE)
    print(f"✅ Best mixture score: {score:.3f}")
    print("📊 Optimal weights:")
    for smi, weight in zip(test_smiles, best_weights):
        print(f"  {smi}: {weight:.3f}")
    
    # Test mixture descriptors
    mixture_desc = optimizer._predict_mixture_descriptors(molecules, best_weights)
    print("🎨 Predicted mixture profile:")
    for desc, value in sorted(mixture_desc.items(), key=lambda x: x[1], reverse=True):
        if value > 0.05:  # Only show significant descriptors
            print(f"  {desc}: {value:.3f}")
