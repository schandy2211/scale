"""
Simple odor prediction oracle for chemosensory optimization.

This implements a basic ML-based odorant score predictor that can be used
as an objective function for molecular optimization. In a production system,
this would be trained on datasets like DREAM Olfaction, Pyrfume, or GoodScents.
"""

import pickle
import os
import numpy as np
from typing import List, Optional, Dict, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")


class OdorOracle:
    """
    Machine learning-based odorant score predictor.
    
    This is a simplified version that combines molecular descriptors to predict
    odorant quality. In practice, this would be trained on real olfactory data.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the odor oracle.
        
        Args:
            model_path: Path to saved model. If None, uses a simple heuristic model.
        """
        self.model = None
        self.scaler = None
        self.feature_names = [
            'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'AromaticRings',
            'SaturatedRings', 'HeteroAtoms', 'NumAliphaticRings'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._create_heuristic_model()
    
    def _create_heuristic_model(self):
        """
        Create a simple heuristic model based on known odorant properties.
        This serves as a placeholder until real training data is available.
        """
        print("🧪 Creating heuristic odor model (replace with trained model for production)")
        
        # Simple heuristic: good odorants typically have:
        # - MW: 100-200 (sweet spot)
        # - LogP: 1-3 (volatile but not too lipophilic)
        # - Low TPSA (< 40)
        # - Some aromatic content (many odorants are aromatic)
        # - Not too flexible (rotatable bonds < 6)
        
        self.model = "heuristic"
        self.scaler = None
    
    def _calculate_features(self, mol: Chem.Mol) -> List[float]:
        """Calculate molecular descriptors for prediction."""
        if mol is None:
            return [0.0] * len(self.feature_names)
        
        try:
            features = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.NumAliphaticRings(mol),
            ]
            return features
        except Exception as e:
            print(f"    Feature calculation error: {e}")
            return [0.0] * len(self.feature_names)
    
    def _heuristic_score(self, features: List[float]) -> float:
        """
        Calculate heuristic odorant score based on known patterns.
        """
        mw, logp, tpsa, hbd, hba, rotbonds, aromatic, saturated, hetero, aliphatic = features
        
        score = 0.3  # Base score
        
        # MW preference (bell curve around 150)
        if 100 <= mw <= 200:
            mw_bonus = 0.3 * (1 - abs(mw - 150) / 50)
            score += mw_bonus
        elif 85 <= mw <= 250:
            score += 0.1
        else:
            score -= 0.3
        
        # LogP preference (1-3 is ideal)
        if 1 <= logp <= 3:
            score += 0.25
        elif 0.5 <= logp <= 4:
            score += 0.1
        else:
            score -= 0.2
        
        # TPSA preference (low for volatility)
        if tpsa <= 40:
            score += 0.2
        else:
            score -= 0.1 * (tpsa - 40) / 20
        
        # Aromatic content bonus (many odorants are aromatic)
        if aromatic >= 1:
            score += 0.15 * min(aromatic, 2)
        
        # Flexibility penalty (too flexible = less specific binding)
        if rotbonds <= 4:
            score += 0.1
        elif rotbonds > 8:
            score -= 0.2
        
        # Heteroatom bonus (O, N, S often important for odor)
        if 1 <= hetero <= 3:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def predict_odor_score(self, smiles: str) -> float:
        """
        Predict odorant quality score for a SMILES string.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            Odorant score between 0-1 (higher = better odorant potential)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        
        features = self._calculate_features(mol)
        
        if self.model == "heuristic":
            return self._heuristic_score(features)
        else:
            # For trained ML models
            try:
                features_scaled = self.scaler.transform([features]) if self.scaler else [features]
                score = self.model.predict(features_scaled)[0]
                return max(0.0, min(1.0, score))
            except Exception:
                return 0.0
    
    def predict_batch(self, smiles_list: List[str]) -> List[float]:
        """Predict scores for a batch of SMILES."""
        return [self.predict_odor_score(smi) for smi in smiles_list]
    
    def save_model(self, path: str):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Saved odor model to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            print(f"📂 Loaded odor model from {path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self._create_heuristic_model()


def create_sample_training_data():
    """
    Create sample training data for demonstration.
    In practice, this would come from DREAM Olfaction, Pyrfume, etc.
    """
    # Known odorants with rough quality scores (0-1)
    training_data = [
        # Good odorants (fruity/floral)
        ("COc1ccccc1", 0.8),          # anisole (sweet, floral)
        ("CCOC(=O)C", 0.7),           # ethyl acetate (fruity)
        ("CC(=O)OCC", 0.7),           # ethyl acetate (fruity)
        ("CC(=O)OCCC", 0.8),          # propyl acetate (fruity)
        ("O=C1OCCC1", 0.6),           # γ-butyrolactone (sweet)
        ("CC1=CCC(CC1)O", 0.9),       # terpineol (citrus, floral)
        ("O=CCCCCC", 0.6),            # hexanal (green, grassy)
        ("CCC(=O)C", 0.5),            # butanone (solvent-like)
        
        # Moderate odorants
        ("CCO", 0.3),                 # ethanol (alcoholic)
        ("CC(=O)C", 0.4),             # acetone (solvent)
        ("CCCCC", 0.2),               # pentane (gasoline-like)
        
        # Poor odorants (too large, wrong properties)
        ("CCCCCCCCCCCC", 0.1),        # dodecane (waxy, weak)
        ("O", 0.0),                   # water (no significant odor)
        ("CC(C)(C)C", 0.1),           # tert-butyl (chemical)
    ]
    
    return training_data


def train_odor_model(training_data: List[tuple], save_path: str = "models/odor_model.pkl"):
    """
    Train an odor prediction model on sample data.
    
    Args:
        training_data: List of (smiles, score) tuples
        save_path: Where to save the trained model
    """
    oracle = OdorOracle()
    
    # Extract features and targets
    X = []
    y = []
    
    for smiles, score in training_data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            features = oracle._calculate_features(mol)
            X.append(features)
            y.append(score)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"🎯 Training odor model on {len(X)} samples...")
    
    # Train scaler and model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
    model.fit(X_scaled, y)
    
    # Update oracle
    oracle.model = model
    oracle.scaler = scaler
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    oracle.save_model(save_path)
    
    # Evaluate on training data
    predictions = model.predict(X_scaled)
    r2 = np.corrcoef(y, predictions)[0, 1] ** 2
    print(f"📊 Training R² = {r2:.3f}")
    
    return oracle


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Odor Oracle")
    
    # Create and test heuristic model
    oracle = OdorOracle()
    
    test_molecules = [
        "COc1ccccc1",      # anisole (should score high)
        "CCOC(=O)C",       # ethyl acetate (should score high)
        "CCCCCCCCCCCC",    # dodecane (should score low)
        "O",               # water (should score very low)
        "CC1=CCC(CC1)O",   # terpineol (should score high)
    ]
    
    print("\n🎯 Heuristic Model Predictions:")
    for smiles in test_molecules:
        score = oracle.predict_odor_score(smiles)
        print(f"  {smiles}: {score:.3f}")
    
    # Train a simple ML model
    print("\n🎓 Training ML Model...")
    training_data = create_sample_training_data()
    ml_oracle = train_odor_model(training_data)
    
    print("\n🤖 ML Model Predictions:")
    for smiles in test_molecules:
        score = ml_oracle.predict_odor_score(smiles)
        print(f"  {smiles}: {score:.3f}")
