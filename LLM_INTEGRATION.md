# LLM Integration for SCALE

This document describes how to use the newly added LLM-based controllers and agents in SCALE.

## Quick Start

1. **Setup OpenAI API Key:**
   ```bash
   python setup_llm.py
   ```
   Or manually set the environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

2. **Run with LLM Controller:**
   ```bash
   # Demo with LLM controller
   python baseline/demo.py --llm
   
   # Full optimization with LLM controller
   python baseline/baseline_opt.py --llm --agent --seed "c1ccccc1" --rounds 6
   ```

## What's New

### LLM Controller (`agent/llm_controller.py`)
- Replaces the heuristic controller with GPT-based decision making
- Analyzes optimization progress and adjusts parameters intelligently
- Falls back to heuristic rules if LLM fails

### LLM Agent (`agent/llm_agent.py`)
- Uses GPT to propose molecular modifications
- Provides chemical reasoning for changes
- Validates proposed molecules with chemical constraints

## Usage Examples

### Basic Demo with LLM
```bash
python baseline/demo.py --llm --preset qed_sa --rounds 4
```

### Custom Optimization with LLM
```bash
python baseline/baseline_opt.py \
  --llm --agent \
  --seed "c1ccccc1,c1ccncc1" \
  --objective qed \
  --rounds 6 \
  --cands 800 \
  --topk 60
```

### Compare Heuristic vs LLM
```bash
# Heuristic controller
python baseline/demo.py --preset qed_sa --rounds 4

# LLM controller  
python baseline/demo.py --llm --preset qed_sa --rounds 4
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: gpt-5)

### Command Line Options
- `--llm`: Use LLM-based controller instead of heuristic
- `--agent`: Enable agentic controller (works with both heuristic and LLM)

## How It Works

### LLM Controller Decision Process
1. Observes current optimization state (scores, progress, round info)
2. Sends structured prompt to GPT with optimization context
3. GPT responds with JSON containing parameter recommendations
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
