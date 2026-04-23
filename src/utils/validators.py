"""
validators.py - Pydantic schemas for facet data validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Literal


class FacetSchema(BaseModel):
    """Strict schema for enriched facet data."""
    
    facet_id: str = Field(..., pattern=r'^FACET-\d{3,5}$')
    facet_name: str
    category: Literal['Linguistic', 'Pragmatics', 'Safety', 'Emotion']
    definition: str
    context_scope: Literal['single_turn', 'multi_turn']
    invert_scale: bool
    score_1: str
    score_2: str
    score_3: str
    score_4: str
    score_5: str
    example_low: str
    example_mid: str
    example_high: str
    schema_version: str = Field(default="1.0.0")
    
    @validator('facet_name')
    def name_not_empty(cls, v):
        if not v or v.strip() == '':
            raise ValueError('facet_name cannot be empty')
        return v.strip()
    
    @validator('definition')
    def definition_not_empty(cls, v, values, **kwargs):
        min_len = 10  # Configurable via main config
        if not v or len(v.strip()) < min_len:
            raise ValueError(f'definition must be at least {min_len} characters')
        return v.strip()
    
    @validator('score_1', 'score_2', 'score_3', 'score_4', 'score_5')
    def score_anchor_not_empty(cls, v):
        if not v or v.strip() == '':
            raise ValueError('score anchors cannot be empty')
        return v.strip()