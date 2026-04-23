#!/usr/bin/env python3
"""
prompt_builder.py
Dynamic prompt assembly for facet evaluation.
Provider-agnostic and token-optimized.
"""

import json
from typing import List, Dict, Optional


def _compact_facet(f: Dict) -> Dict:
    """Return a minimal facet dict for prompt injection."""
    return {
        "id": f["facet_id"],
        "name": f["facet_name"],
        "def": f["definition"],
        "rubric": {str(i): f[f"score_{i}"] for i in range(1, 6)},
        "ex": {
            "low": f.get("example_low", ""),
            "mid": f.get("example_mid", ""),
            "high": f.get("example_high", "")
        },
        "invert": f.get("invert_scale", False)
    }


def build_evaluation_prompt(
    turn_text: str,
    facet_batch: List[Dict],
    max_tokens: int = 12000,  # Leave room for response
    model_context_window: int = 32768  # Configurable per model
) -> str:
    """
    Build a token-efficient, context-aware evaluation prompt.
    
    Args:
        turn_text: The conversation turn to evaluate
        facet_batch: List of facet dicts from FacetRegistry
        max_tokens: Max tokens for the prompt (safety limit)
        model_context_window: Model's total context window
        
    Returns:
        Formatted prompt string
    """
    # Compact facet representation to save tokens
    facets_context = [_compact_facet(f) for f in facet_batch]
    
    # Estimate token count (rough: 4 chars ≈ 1 token)
    prompt_template = """You are an expert conversation evaluator. Score the turn on each facet.

TURN: {turn}

FACETS: {facets_json}

OUTPUT FORMAT (strict JSON):
{{
  "results": [
    {{"facet_id": "FACET-XXX", "score": 1-5, "reasoning": "1-sentence why", "confidence": 0.0-1.0}}
  ]
}}

RULES:
- If facet.invert=true, higher raw score = MORE negative trait (e.g., Dishonesty).
- Use examples (ex.low/mid/high) to anchor scoring.
- For multi_turn facets, assume full dialogue context is available.
- Confidence: 1.0=certain, 0.5=guessing."""

    # Build prompt incrementally to check length
    base_prompt = prompt_template.format(
        turn=turn_text[:500],  # Truncate very long turns
        facets_json=json.dumps(facets_context, separators=(',', ':'))  # Compact JSON
    )
    
    # Simple token estimate (adjust multiplier per model)
    estimated_tokens = len(base_prompt) // 4
    if estimated_tokens > (model_context_window - 500):  # Reserve 500 for response
        # Fallback: reduce batch size externally; log warning
        print(f"⚠️ Prompt too large ({estimated_tokens} tokens). Reduce facet batch size.")
    
    return base_prompt