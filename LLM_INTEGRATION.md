# LLM Integration for SCALE

This document describes how to use the comprehensive LLM-based optimization system in SCALE that replaces all heuristic components.

## Quick Start

1. **Setup OpenAI API Key:**
   ```bash
   python setup_llm.py
   ```
   Or manually set the environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

2. **Run with Full LLM System:**
   ```bash
   # Demo with complete LLM system
   python baseline/demo.py --llm --llm-candidates
   
   # Full optimization with LLM controller and candidate generation
   python baseline/baseline_opt.py --llm --llm-candidates --agent --seed "c1ccccc1" --rounds 6
   ```

## What's New

### LLM Controller (`agent/llm_controller.py`)
- **Enhanced Decision Making**: Analyzes optimization progress with comprehensive context
- **Adaptive Parameters**: Intelligently adjusts exploration, diversity, and penalty parameters
- **Strategic Reasoning**: Considers stagnation, progress rate, and optimization phase
- **Fallback Safety**: Falls back to heuristic rules if LLM fails

### LLM Candidate Generator (`agent/llm_candidate_generator.py`)
- **Intelligent Generation**: Uses GPT to propose chemically valid molecular modifications
- **Context-Aware**: Considers current optimization state, objective, and molecular properties
- **Multi-Strategy**: Combines LLM reasoning with BRICS and attachment fallbacks
- **Property Optimization**: Focuses on improving specific molecular properties

### LLM Optimization Agent (`agent/llm_optimization_agent.py`)
- **Comprehensive Analysis**: Analyzes molecular patterns, limitations, and improvement areas
- **Strategic Modifications**: Proposes targeted modifications based on chemical reasoning
- **Multi-Objective**: Handles complex optimization objectives with chemical constraints
- **Adaptive Strategies**: Adjusts modification strategies based on optimization progress

## Usage Examples

### Basic Demo with LLM
```bash
python baseline/demo.py --llm --preset qed_sa --rounds 4
```

### Full LLM System (Controller + Candidate Generation)
```bash
python baseline/demo.py --llm --llm-candidates --preset qed_sa --rounds 4
```

### Custom Optimization with Complete LLM System
```bash
python baseline/baseline_opt.py \
  --llm --llm-candidates --agent \
  --seed "c1ccccc1,c1ccncc1" \
  --objective qed \
  --rounds 6 \
  --cands 800 \
  --topk 60
```

### Compare Different Approaches
```bash
# Heuristic system (original)
python baseline/demo.py --preset qed_sa --rounds 4

# LLM controller only
python baseline/demo.py --llm --preset qed_sa --rounds 4

# LLM candidate generation only
python baseline/demo.py --llm-candidates --preset qed_sa --rounds 4

# Complete LLM system
python baseline/demo.py --llm --llm-candidates --preset qed_sa --rounds 4
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: gpt-5)

### Command Line Options
- `--llm`: Use LLM-based controller instead of heuristic
- `--llm-candidates`: Use LLM-based candidate generation instead of BRICS/attach
- `--agent`: Enable agentic controller (works with both heuristic and LLM)

## How It Works

### LLM Controller Decision Process
1. **State Analysis**: Observes current optimization state (scores, progress, round info, stagnation)
2. **Context Building**: Creates comprehensive prompt with optimization history and chemical context
3. **Strategic Reasoning**: GPT analyzes state and recommends optimal parameters
4. **Parameter Adjustment**: Adjusts exploration, diversity, penalties, and operation type
5. **Fallback Safety**: Falls back to heuristic rules if LLM fails

### LLM Candidate Generation Process
1. **Molecular Analysis**: Analyzes current molecules and their properties
2. **Strategy Generation**: GPT proposes modification strategies based on optimization context
3. **Candidate Creation**: Generates chemically valid molecular modifications
4. **Validation**: Filters candidates for chemical validity and drug-likeness
5. **Fallback Methods**: Uses BRICS/attach methods if LLM generation fails

### Complete LLM System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   LLM Controller│───▶│  Optimization    │───▶│ LLM Candidate       │
│                 │    │     Loop         │    │ Generator           │
│ - Parameter     │    │                  │    │                     │
│   Adjustment    │    │ - Training       │    │ - Molecular         │
│ - Strategy      │    │ - Prediction     │    │   Analysis          │
│   Selection     │    │ - Selection      │    │ - Modification      │
└─────────────────┘    └──────────────────┘    │   Strategies        │
                                               │ - Chemical          │
                                               │   Validation        │
                                               └─────────────────────┘
```
4. System parses and applies the recommendations
5. Falls back to heuristic rules if LLM fails

### LLM Agent Modification Process
1. Receives input SMILES string
2. Sends chemical modification prompt to GPT
3. GPT proposes modified SMILES with explanation
4. System validates the proposed molecule
5. Falls back to rule-based modifications if LLM fails

## Cost Considerations

- Uses `gpt-5` by default (cost-effective)
- Each optimization round makes 1-2 API calls
- Can switch to `gpt-5` for better performance (higher cost)

## Troubleshooting

### Common Issues

1. **"OpenAI API key must be provided"**
   - Run `python setup_llm.py` to configure your API key
   - Or set `export OPENAI_API_KEY=your_key`

2. **"LLM API error"**
   - Check your internet connection
   - Verify your API key is valid
   - Check your OpenAI account has credits

3. **"Failed to parse LLM response"**
   - The system will automatically fall back to heuristic rules
   - This is normal behavior for robustness

### Fallback Behavior
- If LLM fails, the system automatically falls back to rule-based approaches
- No optimization runs will fail due to LLM issues
- All existing functionality remains unchanged

## Future Enhancements

- Support for other LLM providers (Claude, local models)
- More sophisticated prompting strategies
- LLM-based molecular generation (not just modification)
- Multi-objective optimization with LLM reasoning
