# SCALE: Scaffold‑Conscious Agent for Learning & Exploration

**AI-Powered Molecular Optimization with LLM Intelligence and Chemosensory Focus**

SCALE is a fast, agentic molecular optimizer that designs small molecules under guardrails. It now features **LLM-powered controllers** and **chemosensory optimization** for flavor/fragrance applications.
**Perfect for:** Drug discovery, fragrance design, chemistry research, AI agent experiments

## ✨ What Makes This Special?

### **🤖 AI-Powered**
- **Smart Decisions**: AI agent chooses the best optimization strategy each round
- **Molecule Design**: AI agent creates new molecules with chemical reasoning
- **Multiple Models**: Works with GPT-4.1, GPT-5, and other AI models

### **🧪 Real Chemistry**
- **Drug Discovery**: Finds molecules that could become medicines
- **Fragrance Design**: Creates molecules for perfumes and flavors
- **Safety First**: Automatically filters out dangerous molecules
- **Physics Aware**: Considers real molecular properties and constraints

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

## 🚀 Quick Start

### **🌐 Web Demo (Recommended)**
**The agentic chemist in your browser. Explores, evaluates, and evolves with diversity and safety by design:**

#### **Quick Launch:**
```bash
python launch_demo.py
```

#### **Manual Launch:**
```bash
# Install all dependencies (includes Flask)
pip install -r requirements.txt

# Start the web server
cd webapp
python app.py
```

#### **Access the Demo:**
- **URL**: `http://localhost:8080`
- **Features**: Interactive demos with real-time AI agent reasoning
- **Demos Available**:
  - **💊 Drug Discovery**: Design molecules that make it to lab
  - **🧪 Fragrance Design**: Odorant optimization with volatility & safety  
  - **🎭 Mixture Optimization**: Turn intent into candidates that stick ⭐

#### **Demo Features:**
- **Real-time Progress**: Watch agent formulate hypotheses and run experiments
- **Chemical Reasoning**: See agent's experimentation process step-by-step
- **Visual Results**: Elegant molecule cards and blend formulas
- **Tools Transparency**: See all computational tools being used (RDKit, GPT-5, MMFF94, etc.)
- **Interactive UI**: Click any demo to see AI agent in action

#### **Troubleshooting:**
- **Port 8080 in use**: The launcher automatically handles port conflicts
- **Flask not found**: Run `pip install -r requirements.txt` first
- **Browser not opening**: Manually navigate to `http://localhost:8080`
- **Demo not loading**: Check terminal for error messages and ensure all dependencies are installed

### **⚡ Command Line**
**For quick testing and development:**
```bash
# Find drug-like molecules
python baseline/demo.py --preset qed_sa --llm --llm-candidates --rounds 5

# Find fragrance molecules with AI agent
python baseline/demo.py --preset odor --llm --llm-candidates --rounds 5
```

**Results** are saved in `runs/` folder with plots and molecule data.

## 📚 Simple Concepts

### **What are QED and Odor Scores?**
- **QED Score (0-1)**: How "drug-like" a molecule is
  - 0.0 = Bad for drugs (too big, toxic, etc.)
  - 1.0 = Perfect drug candidate
  - Example: Aspirin = 0.65, Caffeine = 0.78

- **Odor Score (0-1)**: How good a molecule smells/tastes
  - 0.0 = No interesting smell
  - 1.0 = Perfect fragrance molecule
  - Example: Rose scent molecules = 0.8+

### **Two Search Strategies**
- **Explore**: Find completely new types of molecules
- **Optimize**: Improve one specific molecule family

## 🎯 What Can You Optimize?

SCALE can optimize molecules for different purposes. Just pick your goal:

### **💊 Drug Discovery (QED Score)**
**Goal**: Find drug-like molecules
```bash
python baseline/demo.py --preset qed_sa --llm --llm-candidates --rounds 5
```
- **QED Score**: 0-1 (higher = more drug-like)
- **Examples**: Aspirin (0.65), Caffeine (0.78)
- **Good for**: Pharmaceuticals, medicine development

### **🧪 Fragrance & Flavor (Odor Score)**
**Goal**: Find molecules that smell/taste good
```bash
python baseline/demo.py --preset odor --llm --llm-candidates --rounds 5
```
- **Odor Score**: 0-1 (higher = better odorant)
- **Examples**: Anisole (floral), Ethyl acetate (fruity)
- **Good for**: Perfumes, food flavoring, aromatherapy

### **⚗️ Lipophilicity (LogP Score)**
**Goal**: Control how molecules interact with fats/oils
```bash
python baseline/baseline_opt.py --objective pen_logp --rounds 5
```
- **LogP Score**: How well molecules dissolve in fats vs water
- **Good for**: Membrane permeability, drug absorption

## 🤖 Using AI Agent Features

### **Setup (First Time Only)**
```bash
export OPENAI_API_KEY="your-api-key"
```

### **What the AI Agent Does**
- **Smart Controller**: Decides when to explore vs optimize based on progress
- **Molecule Creator**: Designs new molecules with explanations like:
  - *"Added hydroxyl group to increase water solubility"*
  - *"Modified aromatic ring for better drug-likeness"*
- **Safe Fallback**: If AI agent fails, automatically uses backup methods

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
- AI agent-driven molecular hypothesis generation
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

## 🎯 Summary

**SCALE makes molecule design simple:**

1. **Pick your goal**: Drug discovery (`qed_sa`) or Fragrance (`odor`)
2. **Add AI agent**: `--llm --llm-candidates` 
3. **Run**: Results automatically saved with plots

**Quick Examples:**
```bash
# Find drug molecules
python baseline/demo.py --preset qed_sa --llm --llm-candidates --rounds 5

# Find fragrance molecules with AI agent help
python baseline/demo.py --preset odor --llm --llm-candidates --rounds 5

# Compare different strategies
python baseline/compare_modes.py --rounds 5
```

**Perfect for:** Drug discovery, fragrance design, chemistry research, AI agent experiments

*Ready to design better molecules with AI agent?* 🧪✨
