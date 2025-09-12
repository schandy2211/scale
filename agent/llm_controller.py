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
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1"):
        """
        Initialize the LLM controller.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var.
            model: OpenAI model to use (default: gpt-5)
        """
        self.model = model
        
        # Set up OpenAI client
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        print(f"🎯 LLM Controller initialized with model: {self.model}")
    
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
        print(f"📤 Sending prompt to {self.model}:")
        print(f"'{prompt}'")
        print("="*50)
        
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
            response_text = response.choices[0].message.content
            print(f"🤖 LLM Controller Raw Response: {repr(response_text)}")
            
            if response_text:
                response_text = response_text.strip()
                print(f"🧠 LLM Controller Cleaned Response: '{response_text}'")
            
            # If empty response, use fallback immediately
            if not response_text:
                print("❌ Empty LLM response, using fallback")
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
        
        prompt = f"""Molecular optimization controller. Round {obs.round_index}/{obs.rounds_total}.

Current best score: {obs.best:.3f}
Improved this round: {'Yes' if improved else 'No'}

Return JSON parameters:
{{"op": "brics", "k": 1.0, "div": 0.2, "lam": 1.0, "sa_beta": 0.0, "scaf_alpha": 0.0}}

Guidelines:
- If not improving: increase k (exploration) to 1.5, try "attach" or "llm" operation
- If improving: keep k around 1.0, use "brics" operation  
- For advanced generation: use "llm" operation for AI-powered molecular design
- div = diversity (0.0-0.5), lam = strain penalty (0.5-2.0)

Return only valid JSON:"""
        return prompt
    
    def _parse_response(self, response_text: str, obs: Observation) -> Action:
        """Parse LLM response into Action object."""
        print(f"🔍 Parsing LLM response: '{response_text}'")
        try:
            # Try to extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
                print(f"📝 Extracted JSON from code block: '{json_text}'")
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
                print(f"📝 Extracted JSON from braces: '{json_text}'")
            else:
                print(f"❌ No JSON found in response: '{response_text}'")
                raise ValueError("No JSON found in response")
            
            params = json.loads(json_text)
            print(f"✅ Successfully parsed JSON: {params}")
            
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
