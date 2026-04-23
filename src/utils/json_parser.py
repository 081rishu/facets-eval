"""
json_parser.py - Robust JSON extraction utilities for LLM responses.
"""

import re
import json
from typing import Union, Dict, List


def extract_json_from_response(content: str) -> Union[Dict, List]:
    """
    Extract JSON from LLM response, handling markdown wrappers and edge cases.
    
    Args:
        content: Raw response string from LLM
        
    Returns:
        Parsed JSON object (dict or list)
        
    Raises:
        ValueError: If no valid JSON can be extracted
    """
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try to extract from ```json ... ``` or ``` ... ``` blocks
    patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',  # Markdown code blocks (objects)
        r'```(?:json)?\s*(\[.*?\])\s*```',  # Markdown code blocks (arrays)
        r'(\{.*\})',  # Fallback: find first { to last }
        r'(\[.*\])',  # Fallback for arrays
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    # Last resort: try to find balanced braces/brackets
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = content.find(start_char)
        if start != -1:
            brace_count = 0
            for i, char in enumerate(content[start:], start):
                if char == start_char:
                    brace_count += 1
                elif char == end_char:
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(content[start:i+1])
                        except json.JSONDecodeError:
                            break
    
    raise ValueError(
        f"Could not extract valid JSON from response. "
        f"Content preview: {content[:200]}..."
    )