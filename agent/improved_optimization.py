"""
Improved molecular optimization with chemical constraints and better scoring.
"""

from __future__ import annotations

import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from .improved_llm import ImprovedLLMAgent
from .chemical_constraints import ChemicalConstraints
from .retrieval import top_k_neighbors
from .scoring import score_knn, train_ridge, score_ridge


class ImprovedOptimizer:
    """Enhanced molecular optimizer with chemical constraints."""
    
    def __init__(self, 
                 constraints: ChemicalConstraints = None,
                 llm_agent: ImprovedLLMAgent = None):
        self.constraints = constraints or ChemicalConstraints()
        self.llm_agent = llm_agent or ImprovedLLMAgent(self.constraints)
        self.history = []
        
    def optimize(self, 
                 seed_smiles: str,
                 n_iter: int = 20,
                 delta: float = 1e-3,
                 p_explore: float = 0.15,
                 sa_threshold: float = 0.3,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Run improved molecular optimization with chemical constraints.
        
        Args:
            seed_smiles: Starting molecule
            n_iter: Number of optimization iterations
            delta: Minimum improvement to accept
            p_explore: Exploration probability
            sa_threshold: Minimum synthetic accessibility score
            verbose: Print progress
            
        Returns:
            Dictionary with optimization results
        """
        # Validate seed molecule
        is_valid, reason = self.constraints.is_chemically_valid(seed_smiles)
        if not is_valid:
            raise ValueError(f"Invalid seed molecule: {reason}")
        
        # Initialize
        current = seed_smiles
        current_score = score_ridge(current)
        visited = set([current])
        self.history = []
        
        if verbose:
            print(f"Starting optimization with: {current} (score: {current_score:.4f})")
            print(f"Constraints: MW<{self.constraints.max_molecular_weight}, "
                  f"SA>{sa_threshold:.2f}")
        
        # Train ridge model
        train_ridge()
        
        for t in tqdm(range(n_iter), desc="Optimizing"):
            # Generate multiple proposals
            proposals = self.llm_agent.propose_multiple(current, n_proposals=5)
            
            # Filter by synthetic accessibility
            valid_proposals = [
                (smiles, reason, sa_score) for smiles, reason, sa_score in proposals
                if sa_score >= sa_threshold
            ]
            
            if not valid_proposals:
                if verbose:
                    print(f"Iter {t}: No synthetically accessible proposals found")
                continue
            
            # Choose best proposal
            best_proposal = max(valid_proposals, key=lambda x: x[2])  # Highest SA score
            proposal, reason, sa_score = best_proposal
            
            # Get neighbors and scores
            neighbors = top_k_neighbors(proposal, k=5)
            knn_score = score_knn(proposal, neighbors)
            ridge_score = score_ridge(proposal)
            
            # Decision logic
            accepted = False
            acceptance_reason = ""
            
            if ridge_score >= current_score + delta:
                accepted = True
                acceptance_reason = "Improvement in property score"
            elif random.random() < p_explore and proposal not in visited:
                accepted = True
                acceptance_reason = "Exploration (random)"
            elif sa_score > 0.8:  # High synthetic accessibility
                accepted = True
                acceptance_reason = "High synthetic accessibility"
            
            if accepted:
                visited.add(proposal)
                current, current_score = proposal, ridge_score
                if verbose:
                    print(f"Iter {t}: Accepted {proposal} (score: {ridge_score:.4f}, "
                          f"SA: {sa_score:.2f}) - {reason}")
            
            # Record history
            self.history.append({
                'iter': t,
                'current_smiles': current,
                'proposal': proposal,
                'proposal_reason': reason,
                'score_knn': float(knn_score),
                'score_ridge': float(ridge_score),
                'sa_score': float(sa_score),
                'accepted': accepted,
                'acceptance_reason': acceptance_reason,
                'current_score': float(current_score),
                'unique_molecules': len(visited)
            })
        
        # Compile results
        results = {
            'final_smiles': current,
            'final_score': current_score,
            'improvement': current_score - score_ridge(seed_smiles),
            'unique_molecules': len(visited),
            'history': pd.DataFrame(self.history),
            'constraints_used': {
                'max_mw': self.constraints.max_molecular_weight,
                'max_heavy_atoms': self.constraints.max_heavy_atoms,
                'sa_threshold': sa_threshold
            }
        }
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Final molecule: {current}")
            print(f"Final score: {current_score:.4f}")
            print(f"Improvement: {results['improvement']:.4f}")
            print(f"Unique molecules explored: {len(visited)}")
        
        return results
    
    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """Plot optimization results."""
        df = results['history']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Score progression
        axes[0, 0].plot(df['iter'], df['current_score'], 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Property Score')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Synthetic accessibility
        axes[0, 1].plot(df['iter'], df['sa_score'], 'o-', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Synthetic Accessibility')
        axes[0, 1].set_title('SA Score Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Acceptance rate
        acceptance_rate = df['accepted'].rolling(window=5).mean()
        axes[1, 0].plot(df['iter'], acceptance_rate, 'o-', color='red', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Acceptance Rate (5-iter avg)')
        axes[1, 0].set_title('Acceptance Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Unique molecules explored
        axes[1, 1].plot(df['iter'], df['unique_molecules'], 'o-', color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Unique Molecules')
        axes[1, 1].set_title('Chemical Space Explored')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization results."""
        df = results['history']
        
        analysis = {
            'total_iterations': len(df),
            'accepted_proposals': df['accepted'].sum(),
            'acceptance_rate': df['accepted'].mean(),
            'final_improvement': results['improvement'],
            'unique_molecules': results['unique_molecules'],
            'avg_sa_score': df['sa_score'].mean(),
            'max_sa_score': df['sa_score'].max(),
            'min_sa_score': df['sa_score'].min(),
        }
        
        # Acceptance reasons
        acceptance_reasons = df[df['accepted']]['acceptance_reason'].value_counts()
        analysis['acceptance_reasons'] = acceptance_reasons.to_dict()
        
        return analysis


def run_improved_optimization(seed_smiles: str = "CC", 
                            n_iter: int = 20,
                            save_plot: bool = True) -> Dict[str, Any]:
    """
    Run the improved optimization pipeline.
    
    Args:
        seed_smiles: Starting molecule
        n_iter: Number of iterations
        save_plot: Whether to save the results plot
        
    Returns:
        Optimization results
    """
    # Create optimizer
    constraints = ChemicalConstraints()
    agent = ImprovedLLMAgent(constraints)
    optimizer = ImprovedOptimizer(constraints, agent)
    
    # Run optimization
    results = optimizer.optimize(
        seed_smiles=seed_smiles,
        n_iter=n_iter,
        delta=1e-3,
        p_explore=0.15,
        sa_threshold=0.3,
        verbose=True
    )
    
    # Plot results
    if save_plot:
        Path('results/figures').mkdir(parents=True, exist_ok=True)
        optimizer.plot_results(results, 'results/figures/improved_optimization.png')
    
    # Analyze results
    analysis = optimizer.analyze_results(results)
    print(f"\nAnalysis: {analysis}")
    
    return results


if __name__ == "__main__":
    # Run improved optimization
    results = run_improved_optimization(seed_smiles="CC", n_iter=15)

