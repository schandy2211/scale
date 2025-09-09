"""
Create a subset of QMugs dataset for molecular optimization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import openqdc
from rdkit import Chem
from rdkit.Chem import Descriptors
import random

def create_qmugs_subset(n_molecules=1000, output_file="data/qmugs_real_subset.csv"):
    """
    Create a subset of QMugs dataset for molecular optimization.
    
    Args:
        n_molecules: Number of molecules to include
        output_file: Output CSV file path
    """
    print("Loading QMugs dataset...")
    
    # Load QMugs dataset
    dataset = openqdc.datasets.QMugs()
    
    print(f"QMugs dataset loaded with {len(dataset)} molecules")
    
    # Get a random subset of molecules
    total_molecules = len(dataset)
    if n_molecules > total_molecules:
        n_molecules = total_molecules
        print(f"Requested {n_molecules} molecules, but only {total_molecules} available")
    
    # Randomly sample molecules
    random_indices = random.sample(range(total_molecules), n_molecules)
    
    print(f"Creating subset with {n_molecules} molecules...")
    
    # Extract data
    molecules_data = []
    
    for i, idx in enumerate(random_indices):
        if i % 100 == 0:
            print(f"Processing molecule {i+1}/{n_molecules}")
        
        try:
            # Get molecule data
            mol_data = dataset[idx]
            
            # Extract SMILES
            smiles = mol_data['smiles']
            
            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Get molecular properties from QMugs
            # QMugs has different structure, let's check what's available
            available_keys = list(mol_data.keys())
            
            # Choose a meaningful property to optimize
            # Let's use formation energy as it's commonly available
            if 'formation_energy' in mol_data:
                property_value = float(mol_data['formation_energy'])
            elif 'total_energy' in mol_data:
                property_value = float(mol_data['total_energy'])
            elif 'per_atom_formation_energy' in mol_data:
                property_value = float(mol_data['per_atom_formation_energy'])
            else:
                # Use molecular weight as fallback
                property_value = Descriptors.MolWt(mol)
            
            molecules_data.append({
                'smiles': smiles,
                'property': property_value,
                'molecular_weight': Descriptors.MolWt(mol),
                'n_atoms': mol.GetNumAtoms()
            })
            
        except Exception as e:
            print(f"Error processing molecule {idx}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(molecules_data)
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset created successfully!")
    print(f"Total molecules: {len(df)}")
    print(f"Property range: {df['property'].min():.4f} to {df['property'].max():.4f}")
    print(f"Molecular weight range: {df['molecular_weight'].min():.1f} to {df['molecular_weight'].max():.1f}")
    print(f"Saved to: {output_path}")
    
    # Show sample
    print(f"\nSample molecules:")
    print(df.head(10))
    
    return df

def analyze_qmugs_properties(dataset, n_sample=100):
    """Analyze available properties in QMugs dataset."""
    print("Analyzing QMugs properties...")
    
    # Sample a few molecules to see what properties are available
    sample_indices = random.sample(range(len(dataset)), min(n_sample, len(dataset)))
    
    all_props = set()
    for idx in sample_indices:
        try:
            mol_data = dataset[idx]
            all_props.update(mol_data.keys())
        except:
            continue
    
    print(f"Available properties in QMugs:")
    for prop in sorted(all_props):
        print(f"  - {prop}")
    
    return list(all_props)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # First, analyze what properties are available
    print("=== ANALYZING QMUGS PROPERTIES ===")
    dataset = openqdc.datasets.QMugs()
    properties = analyze_qmugs_properties(dataset, n_sample=50)
    
    print(f"\n=== CREATING QMUGS SUBSET ===")
    # Create subset
    df = create_qmugs_subset(n_molecules=1000, output_file="data/qmugs_real_subset.csv")
