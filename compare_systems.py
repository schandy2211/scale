"""
Compare original vs improved molecular optimization systems.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from agent.llm import propose_smiles as original_propose
from agent.improved_optimization import run_improved_optimization
from agent.chemical_constraints import ChemicalConstraints


def run_original_optimization(seed_smiles: str = "CC", n_iter: int = 15):
    """Run the original optimization system."""
    from agent.retrieval import top_k_neighbors
    from agent.scoring import score_knn, train_ridge, score_ridge
    import random
    from tqdm import tqdm
    
    # Train ridge model
    train_ridge()
    
    current = seed_smiles
    current_score = score_ridge(current)
    visited = set([current])
    history = []
    
    print(f"Original system - Starting with: {current} (score: {current_score:.4f})")
    
    for t in tqdm(range(n_iter), desc="Original optimization"):
        proposal, reason = original_propose(current)
        neigh = top_k_neighbors(proposal, k=5)
        knn_s = score_knn(proposal, neigh)
        ridge_s = score_ridge(proposal)
        
        accepted = False
        if ridge_s >= current_score + 1e-3:
            accepted = True
        elif random.random() < 0.1 and proposal not in visited:
            accepted = True
        
        if accepted:
            visited.add(proposal)
            current, current_score = proposal, ridge_s
            print(f"Iter {t}: Accepted {proposal} (score: {ridge_s:.4f}) - {reason}")
        
        history.append({
            'iter': t,
            'current_smiles': current,
            'proposal': proposal,
            'proposal_reason': reason,
            'score_ridge': float(ridge_s),
            'accepted': accepted,
            'current_score': float(current_score),
            'unique_molecules': len(visited)
        })
    
    return {
        'final_smiles': current,
        'final_score': current_score,
        'improvement': current_score - score_ridge(seed_smiles),
        'unique_molecules': len(visited),
        'history': pd.DataFrame(history)
    }


def compare_systems(seed_smiles: str = "CC", n_iter: int = 15):
    """Compare original vs improved systems."""
    print("=" * 60)
    print("MOLECULAR OPTIMIZATION SYSTEM COMPARISON")
    print("=" * 60)
    
    # Run original system
    print("\n1. RUNNING ORIGINAL SYSTEM")
    print("-" * 30)
    original_results = run_original_optimization(seed_smiles, n_iter)
    
    # Run improved system
    print("\n2. RUNNING IMPROVED SYSTEM")
    print("-" * 30)
    improved_results = run_improved_optimization(seed_smiles, n_iter, save_plot=False)
    
    # Compare results
    print("\n3. COMPARISON RESULTS")
    print("-" * 30)
    print(f"Original system:")
    print(f"  Final molecule: {original_results['final_smiles']}")
    print(f"  Final score: {original_results['final_score']:.4f}")
    print(f"  Improvement: {original_results['improvement']:.4f}")
    print(f"  Unique molecules: {original_results['unique_molecules']}")
    
    print(f"\nImproved system:")
    print(f"  Final molecule: {improved_results['final_smiles']}")
    print(f"  Final score: {improved_results['final_score']:.4f}")
    print(f"  Improvement: {improved_results['improvement']:.4f}")
    print(f"  Unique molecules: {improved_results['unique_molecules']}")
    
    # Check chemical validity
    constraints = ChemicalConstraints()
    orig_valid, orig_reason = constraints.is_chemically_valid(original_results['final_smiles'])
    imp_valid, imp_reason = constraints.is_chemically_valid(improved_results['final_smiles'])
    
    print(f"\nChemical validity:")
    print(f"  Original final molecule: Valid={orig_valid} - {orig_reason}")
    print(f"  Improved final molecule: Valid={imp_valid} - {imp_reason}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Score progression
    axes[0, 0].plot(original_results['history']['iter'], 
                   original_results['history']['current_score'], 
                   'o-', label='Original', linewidth=2, alpha=0.7)
    axes[0, 0].plot(improved_results['history']['iter'], 
                   improved_results['history']['current_score'], 
                   's-', label='Improved', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Property Score')
    axes[0, 0].set_title('Score Progression Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Unique molecules explored
    axes[0, 1].plot(original_results['history']['iter'], 
                   original_results['history']['unique_molecules'], 
                   'o-', label='Original', linewidth=2, alpha=0.7)
    axes[0, 1].plot(improved_results['history']['iter'], 
                   improved_results['history']['unique_molecules'], 
                   's-', label='Improved', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Unique Molecules')
    axes[0, 1].set_title('Chemical Space Explored')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Acceptance rates
    orig_acceptance = original_results['history']['accepted'].rolling(window=3).mean()
    imp_acceptance = improved_results['history']['accepted'].rolling(window=3).mean()
    
    axes[1, 0].plot(original_results['history']['iter'], orig_acceptance, 
                   'o-', label='Original', linewidth=2, alpha=0.7)
    axes[1, 0].plot(improved_results['history']['iter'], imp_acceptance, 
                   's-', label='Improved', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Acceptance Rate (3-iter avg)')
    axes[1, 0].set_title('Acceptance Rate Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Synthetic accessibility (improved only)
    if 'sa_score' in improved_results['history'].columns:
        axes[1, 1].plot(improved_results['history']['iter'], 
                       improved_results['history']['sa_score'], 
                       's-', color='green', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Synthetic Accessibility')
        axes[1, 1].set_title('Synthetic Accessibility (Improved)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'SA scores not available\nin original system', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Synthetic Accessibility')
    
    plt.tight_layout()
    
    # Save comparison plot
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('results/figures/system_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to results/figures/system_comparison.png")
    plt.show()
    
    return original_results, improved_results


if __name__ == "__main__":
    # Run comparison
    original_results, improved_results = compare_systems(seed_smiles="CC", n_iter=15)

