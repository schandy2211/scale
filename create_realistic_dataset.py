"""
Create a more realistic synthetic dataset for molecular optimization.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import random

def create_realistic_dataset(n_molecules=1000, output_file="data/realistic_subset.csv"):
    """
    Create a realistic synthetic dataset with diverse molecular structures.
    """
    print("Creating realistic molecular dataset...")
    
    # Define molecular templates with different complexity levels
    molecular_templates = [
        # Simple alkanes
        "CC", "CCC", "CCCC", "CCCCC", "CCCCCC",
        
        # Alcohols
        "CCO", "CCCO", "CCCCCO", "CC(C)O", "CC(C)(C)O",
        
        # Carboxylic acids
        "CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "CC(C)(=O)O",
        
        # Amines
        "CCN", "CCCN", "CCCCN", "CC(C)N", "CC(C)(C)N",
        
        # Aromatics
        "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "c1ccc(cc1)O",
        "c1ccc(cc1)C(=O)O", "c1ccc(cc1)N", "c1ccc(cc1)Cl",
        
        # Heterocycles
        "c1ccncc1", "c1ccoc1", "c1ccsc1", "c1cnccn1",
        
        # Ethers
        "CCOC", "CCOCC", "CCOCCC", "CC(C)OC(C)C",
        
        # Ketones
        "CC(=O)C", "CCC(=O)CC", "CC(C)(=O)C",
        
        # Esters
        "CCOC(=O)C", "CCOC(=O)CC", "CC(C)OC(=O)C",
        
        # Halides
        "CCCl", "CCCCl", "CCBr", "CCF", "CCI",
        
        # Complex molecules
        "CCc1ccccc1O", "CCc1ccccc1C(=O)O", "CCc1ccccc1N",
        "CC(C)c1ccccc1O", "CC(C)c1ccccc1C(=O)O",
        "c1ccc(cc1)c2ccccc2", "c1ccc(cc1)c2ccncc2",
        "CCc1ccccc1c2ccccc2", "CCc1ccccc1c2ccncc2",
    ]
    
    # Generate variations
    molecules_data = []
    
    for template in molecular_templates:
        # Add the base template
        molecules_data.append(template)
        
        # Generate variations
        for _ in range(n_molecules // len(molecular_templates)):
            # Random modifications
            modified = apply_random_modification(template)
            if modified and modified != template:
                molecules_data.append(modified)
    
    # Add more random molecules
    while len(molecules_data) < n_molecules:
        template = random.choice(molecular_templates)
        modified = apply_random_modification(template)
        if modified and modified not in molecules_data:
            molecules_data.append(modified)
    
    # Limit to requested number
    molecules_data = molecules_data[:n_molecules]
    
    print(f"Generated {len(molecules_data)} molecular structures")
    
    # Calculate properties for each molecule
    dataset = []
    
    for i, smiles in enumerate(molecules_data):
        if i % 100 == 0:
            print(f"Processing molecule {i+1}/{len(molecules_data)}")
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Calculate realistic properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotbonds = Descriptors.NumRotatableBonds(mol)
            rings = Descriptors.RingCount(mol)
            
            # Create a composite "drug-likeness" score
            # This simulates a real molecular property we want to optimize
            drug_likeness_score = calculate_drug_likeness_score(
                mw, logp, tpsa, hbd, hba, rotbonds, rings
            )
            
            dataset.append({
                'smiles': smiles,
                'property': drug_likeness_score,
                'molecular_weight': mw,
                'logp': logp,
                'tpsa': tpsa,
                'hbd': hbd,
                'hba': hba,
                'rotbonds': rotbonds,
                'rings': rings
            })
            
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(dataset)
    
    # Save to CSV
    output_path = output_file
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset created successfully!")
    print(f"Total molecules: {len(df)}")
    print(f"Property range: {df['property'].min():.4f} to {df['property'].max():.4f}")
    print(f"Molecular weight range: {df['molecular_weight'].min():.1f} to {df['molecular_weight'].max():.1f}")
    print(f"Saved to: {output_path}")
    
    # Show sample
    print(f"\nSample molecules:")
    print(df[['smiles', 'property', 'molecular_weight', 'logp']].head(10))
    
    return df

def apply_random_modification(smiles):
    """Apply random modifications to a SMILES string."""
    modifications = [
        lambda s: s + "C",  # Add carbon
        lambda s: s + "O",  # Add oxygen
        lambda s: s + "N",  # Add nitrogen
        lambda s: s + "Cl", # Add chlorine
        lambda s: s + "F",  # Add fluorine
        lambda s: s.replace("C", "CC", 1) if "C" in s else s,  # Extend chain
        lambda s: s + "c1ccccc1" if len(s) < 20 else s,  # Add benzene
        lambda s: s + "C(=O)O" if len(s) < 15 else s,  # Add carboxyl
    ]
    
    # Try a few random modifications
    for _ in range(3):
        mod = random.choice(modifications)
        try:
            modified = mod(smiles)
            # Validate the modification
            mol = Chem.MolFromSmiles(modified)
            if mol is not None:
                return modified
        except:
            continue
    
    return smiles

def calculate_drug_likeness_score(mw, logp, tpsa, hbd, hba, rotbonds, rings):
    """
    Calculate a composite drug-likeness score based on Lipinski's Rule of Five
    and other drug-like properties.
    """
    score = 0.0
    
    # Molecular weight (150-500 is ideal)
    if 150 <= mw <= 500:
        score += 0.3
    elif 100 <= mw <= 600:
        score += 0.2
    else:
        score += 0.1
    
    # LogP (-0.4 to 5.6 is ideal)
    if -0.4 <= logp <= 5.6:
        score += 0.3
    elif -1 <= logp <= 6:
        score += 0.2
    else:
        score += 0.1
    
    # TPSA (20-130 is ideal)
    if 20 <= tpsa <= 130:
        score += 0.2
    elif 10 <= tpsa <= 150:
        score += 0.15
    else:
        score += 0.1
    
    # Hydrogen bond donors (≤5 is ideal)
    if hbd <= 5:
        score += 0.1
    else:
        score += 0.05
    
    # Hydrogen bond acceptors (≤10 is ideal)
    if hba <= 10:
        score += 0.1
    else:
        score += 0.05
    
    # Rotatable bonds (≤10 is ideal)
    if rotbonds <= 10:
        score += 0.1
    else:
        score += 0.05
    
    # Rings (1-3 is ideal)
    if 1 <= rings <= 3:
        score += 0.1
    elif 0 <= rings <= 4:
        score += 0.05
    else:
        score += 0.02
    
    # Normalize to 0-1 range
    return min(1.0, max(0.0, score))

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create realistic dataset
    df = create_realistic_dataset(n_molecules=1000, output_file="data/realistic_subset.csv")

