#!/usr/bin/env python3
"""
Test script for the comprehensive LLM-based molecular optimization system.
"""

import os
import sys
import time
import warnings
from pathlib import Path

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from baseline.demo import main as demo_main
from baseline.baseline_opt import run_optimization, OptConfig
from agent.llm_controller import LLMController
from agent.llm_candidate_generator import LLMCandidateGenerator
from agent.llm_optimization_agent import LLMOptimizationAgent


def test_llm_components():
    """Test individual LLM components."""
    print("🧪 Testing LLM Components")
    print("=" * 50)
    
    # Test LLM Controller
    print("\n1. Testing LLM Controller...")
    try:
        controller = LLMController()
        print("✅ LLM Controller initialized successfully")
        
        # Test with mock observation
        from agent.controller import Observation
        obs = Observation(
            round_index=1,
            rounds_total=6,
            train_size=100,
            best=0.5,
            avg=0.4,
            last_best=0.45,
            last_avg=0.35
        )
        
        action = controller.decide(obs)
        print(f"✅ LLM Controller decision: {action}")
        
    except Exception as e:
        print(f"❌ LLM Controller failed: {e}")
    
    # Test LLM Candidate Generator
    print("\n2. Testing LLM Candidate Generator...")
    try:
        generator = LLMCandidateGenerator()
        print("✅ LLM Candidate Generator initialized successfully")
        
        # Test with mock molecules
        from rdkit import Chem
        test_mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
        
        candidates = generator.generate_candidates(
            seed_mols=test_mols,
            n_candidates=5,
            objective="qed",
            current_best=0.5
        )
        
        print(f"✅ Generated {len(candidates)} candidates")
        for i, mol in enumerate(candidates[:3]):
            print(f"   {i+1}. {Chem.MolToSmiles(mol, canonical=True)}")
        
    except Exception as e:
        print(f"❌ LLM Candidate Generator failed: {e}")
    
    # Test LLM Optimization Agent
    print("\n3. Testing LLM Optimization Agent...")
    try:
        agent = LLMOptimizationAgent()
        print("✅ LLM Optimization Agent initialized successfully")
        
        # Test with mock molecules
        from rdkit import Chem
        test_mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
        
        modifications = agent.propose_molecular_modifications(
            current_molecules=test_mols,
            objective="qed",
            current_scores=[0.4, 0.6]
        )
        
        print(f"✅ Proposed {len(modifications)} modifications")
        for i, (mol, reason, props) in enumerate(modifications[:3]):
            print(f"   {i+1}. {Chem.MolToSmiles(mol, canonical=True)} - {reason}")
        
    except Exception as e:
        print(f"❌ LLM Optimization Agent failed: {e}")


def test_optimization_comparison():
    """Compare heuristic vs LLM-based optimization."""
    print("\n🔬 Optimization Comparison")
    print("=" * 50)
    
    seed_smiles = ["CCO", "c1ccccc1"]
    
    # Test configurations
    configs = [
        {
            "name": "Heuristic System",
            "config": OptConfig(
                objective="qed",
                rounds=2,
                init_train_size=32,
                candidates_per_round=100,
                topk_per_round=20,
                use_controller=False,
                use_llm_controller=False,
                use_llm_candidates=False
            )
        },
        {
            "name": "LLM Controller Only",
            "config": OptConfig(
                objective="qed",
                rounds=2,
                init_train_size=32,
                candidates_per_round=100,
                topk_per_round=20,
                use_controller=True,
                use_llm_controller=True,
                use_llm_candidates=False
            )
        },
        {
            "name": "LLM Candidates Only",
            "config": OptConfig(
                objective="qed",
                rounds=2,
                init_train_size=32,
                candidates_per_round=100,
                topk_per_round=20,
                use_controller=False,
                use_llm_controller=False,
                use_llm_candidates=True
            )
        },
        {
            "name": "Complete LLM System",
            "config": OptConfig(
                objective="qed",
                rounds=2,
                init_train_size=32,
                candidates_per_round=100,
                topk_per_round=20,
                use_controller=True,
                use_llm_controller=True,
                use_llm_candidates=True
            )
        }
    ]
    
    results = {}
    
    for config_info in configs:
        name = config_info["name"]
        config = config_info["config"]
        
        print(f"\n🧪 Testing {name}...")
        
        try:
            start_time = time.time()
            result = run_optimization(seed_smiles=seed_smiles, config=config)
            end_time = time.time()
            
            best_score = max([score for _, score in result.get("top", [])]) if result.get("top") else 0.0
            n_molecules = len(result.get("top", []))
            
            results[name] = {
                "best_score": best_score,
                "n_molecules": n_molecules,
                "time": end_time - start_time,
                "success": True
            }
            
            print(f"✅ {name}: Best score = {best_score:.3f}, Molecules = {n_molecules}, Time = {end_time - start_time:.1f}s")
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = {
                "best_score": 0.0,
                "n_molecules": 0,
                "time": 0.0,
                "success": False,
                "error": str(e)
            }
    
    # Summary
    print("\n📊 Results Summary")
    print("=" * 50)
    print(f"{'System':<25} {'Best Score':<12} {'Molecules':<10} {'Time (s)':<10} {'Status'}")
    print("-" * 70)
    
    for name, result in results.items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        print(f"{name:<25} {result['best_score']:<12.3f} {result['n_molecules']:<10} {result['time']:<10.1f} {status}")


def test_demo_commands():
    """Test demo commands with different LLM configurations."""
    print("\n🎮 Demo Command Testing")
    print("=" * 50)
    
    # Note: This would require modifying the demo to run programmatically
    # For now, just show the commands that would be run
    
    commands = [
        "python baseline/demo.py --preset qed_sa --rounds 2",
        "python baseline/demo.py --llm --preset qed_sa --rounds 2",
        "python baseline/demo.py --llm-candidates --preset qed_sa --rounds 2",
        "python baseline/demo.py --llm --llm-candidates --preset qed_sa --rounds 2"
    ]
    
    print("Commands to test manually:")
    for i, cmd in enumerate(commands, 1):
        print(f"{i}. {cmd}")


def main():
    """Main test function."""
    print("🚀 LLM-Based Molecular Optimization System Test")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. LLM components will fall back to heuristics.")
        print("   Set your API key with: export OPENAI_API_KEY=your_key_here")
        print()
    
    try:
        # Test individual components
        test_llm_components()
        
        # Test optimization comparison
        test_optimization_comparison()
        
        # Show demo commands
        test_demo_commands()
        
        print("\n🎉 Testing completed!")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run: python baseline/demo.py --llm --llm-candidates --preset qed_sa --rounds 4")
        print("3. Compare results with heuristic system")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
