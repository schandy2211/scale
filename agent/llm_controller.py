"""
LLM-based controller for molecular optimization decisions.
"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional
import openai

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")

from .controller import Observation, Action


class LLMController:
    """LLM-based controller that uses GPT to make optimization decisions."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        """
        Initialize the LLM controller.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var.
            model: OpenAI model to use (default: gpt-5 for cost efficiency)
        """
        self.model = model
        
        # Set up OpenAI client
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
    
    def decide(self, obs: Observation) -> Action:
        """
        Use LLM to decide on optimization parameters based on current observation.
        
        Args:
            obs: Current optimization state observation
            
        Returns:
            Action with recommended parameters
        """
        # Create prompt for the LLM
        prompt = self._create_prompt(obs)
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert in molecular optimization. Always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1000
            )
            
            # Parse LLM response
            response_text = response.choices[0].message.content.strip()
            
            # If empty response, use fallback immediately
            if not response_text:
                print("Empty LLM response, using fallback")
                return self._fallback_decision(obs)
                
            return self._parse_response(response_text, obs)
            
        except Exception as e:
            print(f"LLM API error: {e}. Falling back to heuristic controller.")
            # Fallback to simple heuristic
            return self._fallback_decision(obs)
    
    def _create_prompt(self, obs: Observation) -> str:
        """Create a comprehensive prompt for the LLM based on the observation."""
        improved = obs.best > obs.last_best + 1e-6
        progress_rate = (obs.best - obs.last_best) / max(obs.last_best, 0.001)
        stagnation_rounds = getattr(self, '_stagnation_count', 0)
        
        # Update stagnation tracking
        if improved:
            self._stagnation_count = 0
        else:
            self._stagnation_count = getattr(self, '_stagnation_count', 0) + 1
        
        prompt = f"""You are an expert molecular optimization controller. Analyze the current state and recommend optimal parameters.

OPTIMIZATION STATE:
- Round: {obs.round_index}/{obs.rounds_total}
- Current Best Score: {obs.best:.4f}
- Previous Best: {obs.last_best:.4f}
- Improvement: {'Yes' if improved else 'No'} ({progress_rate:+.1%})
- Stagnation Rounds: {stagnation_rounds}
- Training Set Size: {obs.train_size}
- Average Score: {obs.avg:.4f}

CONTROL PARAMETERS TO OPTIMIZE:
- op: Operation type ("brics" for fragment-based, "attach" for simple attachment, "llm" for LLM-based generation)
- k: Exploration factor (0.5-2.0, higher = more exploration)
- div: Diversity penalty (0.0-0.5, higher = more diverse molecules)
- lam: Strain penalty scaling (0.5-2.0, higher = more strain penalty)
- sa_beta: Synthetic accessibility penalty (0.0-0.3, higher = penalize complex molecules)
- scaf_alpha: Scaffold diversity penalty (0.0-0.5, higher = encourage scaffold diversity)

STRATEGY GUIDELINES:
- If stagnating (no improvement for 2+ rounds): increase exploration (k=1.5), try different operation
- If improving well: maintain current strategy, slight exploration increase
- If early rounds: focus on exploration and diversity
- If late rounds: focus on exploitation and refinement
- For QED optimization: prioritize drug-likeness and synthetic accessibility
- For pen_logp optimization: balance lipophilicity with other properties

Return JSON with your recommendations:
{{"op": "brics", "k": 1.2, "div": 0.25, "lam": 1.0, "sa_beta": 0.1, "scaf_alpha": 0.2}}"""
        return prompt
    
    def _parse_response(self, response_text: str, obs: Observation) -> Action:
        """Parse LLM response into Action object."""
        try:
            # Try to extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
            
            params = json.loads(json_text)
            
            # Create action with parsed parameters
            action = Action()
            action.op = params.get("op", "brics")
            action.k = float(params.get("k", 1.0))
            action.div = float(params.get("div", 0.2))
            action.lam = float(params.get("lam", 1.0))
            action.sa_beta = float(params.get("sa_beta", 0.0))
            action.scaf_alpha = float(params.get("scaf_alpha", 0.0))
            
            return action
            
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response_text}")
            return self._fallback_decision(obs)
    
    def _fallback_decision(self, obs: Observation) -> Action:
        """Fallback to simple heuristic if LLM fails."""
        improved = obs.best > obs.last_best + 1e-6
        
        action = Action(op="brics")
        if not improved:
            action.k = 1.2  # more exploration
            action.div = 0.25
            action.lam = 0.9  # reduce strain penalty
        else:
            action.k = 1.0
            action.div = 0.2
            action.lam = 1.0
        
        return action
