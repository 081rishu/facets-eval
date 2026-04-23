#!/usr/bin/env python3
"""
parser.py
Pydantic schemas for facet evaluation parsing and validation.
"""

from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Optional, Literal


class FacetEvaluation(BaseModel):
    """Parsed evaluation for a single facet."""
    facet_id: str = Field(..., pattern=r"^FACET-\d{3,5}$")
    facet_name: str
    reasoning: str = Field(..., min_length=10)  # Ensure meaningful reasoning
    score: int = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=0.0, le=1.0)  # Numeric for calibration
    invert_scale: bool = Field(default=False)  # Track for post-processing
    schema_version: str = Field(default="1.0.0")
    
    @validator('reasoning')
    def reasoning_not_generic(cls, v):
        generic = ["good", "bad", "ok", "average", "neutral"]
        if v.lower().strip() in generic:
            raise ValueError("Reasoning must be specific, not generic")
        return v
    
    def get_final_score(self) -> int:
        """Return score inverted if invert_scale=True."""
        return 6 - self.score if self.invert_scale else self.score


class TurnEvaluationResult(BaseModel):
    """Complete evaluation result for a conversation turn."""
    statement: str
    total_facets_evaluated: int
    evaluations: List[FacetEvaluation]
    model_used: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    schema_version: str = Field(default="1.0.0")
    
    def get_facet_score(self, facet_id: str) -> Optional[int]:
        """Lookup final score (with invert_scale applied) by ID."""
        for ev in self.evaluations:
            if ev.facet_id == facet_id:
                return ev.get_final_score()
        return None
    
    def get_low_confidence_facets(self, threshold: float = 0.7) -> List[str]:
        """Return facet IDs with confidence below threshold."""
        return [ev.facet_id for ev in self.evaluations if ev.confidence < threshold]