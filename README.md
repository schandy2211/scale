# SCALE: Scaffold‑Conscious Agent for Learning & Exploration

**AI-Powered Molecular Optimization with LLM Intelligence and Chemosensory Focus**

SCALE is a fast, agentic molecular optimizer that designs small molecules under guardrails. It now features **LLM-powered controllers** and **chemosensory optimization** for flavor/fragrance applications.

## 🚀 Key Features

### **🤖 LLM-Powered Intelligence**
- **LLM Controller**: GPT models make intelligent optimization decisions
- **LLM Candidate Generator**: AI-powered molecular design with chemical reasoning
- **Flexible Models**: Support for GPT-4.1, GPT-5, and other models

### **🧪 Chemosensory Optimization**
- **Odor Preset**: Specialized for fragrance/flavor molecules
- **Volatility Filters**: MW, LogP, TPSA constraints for odorant properties
- **Odor Oracle**: ML-based odorant quality prediction
- **Safety Alerts**: Regulatory and allergen screening

### **⚙️ Core Capabilities**
- **Scaffold-diverse** (hit finding) vs **scaffold-preserving** (lead optimization)
- **Physics-aware**: MMFF strain penalties for realistic molecules
- **Guardrails**: Drug-likeness limits, PAINS filter, safety screening
- **Agentic**: Per-round adaptive decisions with full logging

## 📦 Installation

**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate llm-agent-chem
```

**Pip:**
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### **Basic Drug-like Optimization**
```bash
python baseline/demo.py --preset qed_sa --rounds 6
```

### **🧪 Chemosensory/Odor Optimization**
```bash
# Basic odor optimization
python baseline/demo.py --preset odor --rounds 5

# With LLM enhancement
python baseline/demo.py --preset odor --llm --llm-candidates --rounds 5
```

### **🤖 LLM-Powered Optimization**
```bash
# Use LLM controller and candidate generator
python baseline/demo.py --preset qed_sa --llm --llm-candidates --rounds 6

# Specify custom model
python baseline/demo.py --preset qed_sa --llm --model gpt-4o --rounds 6
```

## 🎯 Application Modes

### **1. Drug Discovery (QED Optimization)**
- **Seeds**: Drug-like scaffolds (benzene, pyridine, acetamide)
- **Objective**: Maximize QED score
- **Filters**: Lipinski's Rule of 5, PAINS screening

### **2. Chemosensory/Olfaction** 🧪
- **Seeds**: Odorant molecules (anisole, ethyl acetate, terpineol)
- **Objective**: Odorant quality score (ML-based)
- **Filters**: Volatility window (MW 85-250, LogP 0.5-4, TPSA <40)
- **Safety**: Allergen and sensitizer alerts

## 🤖 LLM Features

### **Setup OpenAI API**
```bash
export OPENAI_API_KEY="your-api-key"
# Or use our setup script:
python setup_llm.py
```

### **LLM Controller**
Makes intelligent optimization decisions:
- **Exploration vs Exploitation**: Adjusts parameters based on progress
- **Operation Selection**: Chooses between BRICS, attach, or LLM generation
- **Chemical Reasoning**: Understands when to increase diversity or focus

### **LLM Candidate Generator**
Generates novel molecules with reasoning:
- **Chemical Logic**: "Added formyl group to anisole to introduce polarity"
- **Property Awareness**: Considers MW, LogP, and target objectives
- **Fallback Safety**: Uses heuristic methods if LLM fails

## 📊 Outputs

All runs save comprehensive results:

```
runs/demo_[preset]_[timestamp]/
├── config.json          # Run configuration
├── history.json         # Best/avg scores per round
├── decisions.json       # LLM controller decisions
├── selections.csv       # Per-molecule data
├── report_curves.png    # Learning curves
├── report_qed_vs_sa.png # Property scatter
├── report_scaffolds.png # Scaffold diversity
└── top_grid.png        # Best molecules visualization
```

## 🔬 Evaluation & Analysis

**Learning curves and property analysis:**
```bash
python baseline/plot_csv.py --csv runs/qed_run.csv --out_prefix runs/analysis
```

**Compare optimization modes:**
```bash
python baseline/make_latest.py --rounds 6
open docs/figs/latest/compare_modes_index.html
```

**Prediction accuracy:**
```bash
python baseline/plot_pred_vs_true.py --csv runs/qed_run.csv --out pred_vs_true.png
```

## 🧪 Chemosensory Details

### **Odor Oracle (ML Model)**
- **Algorithm**: Random Forest Regressor (n_estimators=50, max_depth=8)
- **Features**: MW, LogP, TPSA, aromatic content, flexibility
- **Performance**: R² = 0.965 on curated odorant data
- **Extensible**: Ready for DREAM Olfaction, Pyrfume datasets

### **Odorant Filters**
- **MW**: 85-250 Da (typical odorant range)
- **LogP**: 0.5-4 (volatility window)
- **TPSA**: <40 (membrane permeability)
- **HBD/HBA**: ≤1/≤6 (hydrogen bonding)
- **Safety**: Allergen patterns, reactive groups

### **Curated Seeds**
```python
odorant_seeds = [
    "COc1ccccc1",          # anisole (sweet, floral)
    "CCOC(=O)C",           # ethyl acetate (fruity)
    "CC1=CCC(CC1)O",       # terpineol (citrus, floral)
    "O=CCCCCC",            # hexanal (green, grassy)
    "CC1=CC=C(C=C1)C(C)C"  # p-cymene (citrus, woody)
]
```

## 🏗️ Architecture

```
baseline/
├── demo.py              # One-command demos
├── baseline_opt.py      # Core optimizer
└── plot_csv.py         # Visualization

agent/
├── llm_controller.py    # LLM-based controller
├── llm_candidate_generator.py  # LLM molecular design
└── controller.py        # Heuristic fallback

├── chemosensory_filters.py    # Odorant-specific filters
├── odor_oracle.py            # ML odorant prediction
└── LLM_INTEGRATION.md        # LLM documentation
```

## 🎯 Use Cases

### **💊 Pharmaceutical**
- Lead optimization around known scaffolds
- Hit finding with diversity constraints
- ADMET-aware molecular design

### **🧪 Flavor & Fragrance**
- Odorant discovery with volatility constraints
- Safety-compliant fragrance design
- Novel aroma molecule generation

### **🔬 Research**
- AI-driven molecular hypothesis generation
- Chemical space exploration
- Benchmarking molecular optimization algorithms

## 🏆 Hackathon Features

**Perfect for chemistry competitions:**
- ✅ **LLM Integration**: GPT-powered molecular design
- ✅ **Chemosensory Focus**: Addresses flavor chemistry requirements
- ✅ **ML Pipeline**: Random Forest with molecular descriptors
- ✅ **Safety Filters**: Regulatory compliance awareness
- ✅ **Visualization**: Professional plots and reports
- ✅ **Extensible**: Easy to add new objectives/datasets

## 📈 Performance

**Typical Results:**
- **QED Optimization**: 0.4 → 0.9+ in 6 rounds
- **Odor Optimization**: Significant improvement in odorant-likeness
- **Speed**: ~5 seconds per round with caching
- **Diversity**: High scaffold coverage with intelligent exploration

## 🤝 Contributing

This project is designed for hackathons and research. Key extension points:
- Add new molecular objectives in `get_objective()`
- Implement custom filters in `chemosensory_filters.py`
- Train new oracles with your datasets
- Extend LLM prompts for domain-specific knowledge

---

**Ready to design better molecules with AI? Start with a simple demo:**

```bash
python baseline/demo.py --preset odor --llm --llm-candidates --rounds 5
```

*Watch as GPT generates novel odorant molecules with chemical reasoning!* 🧪✨