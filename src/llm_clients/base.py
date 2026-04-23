from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class LLMResponse(BaseModel):
    """Standardized response structure across all providers."""
    content: str
    usage: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens
    model: Optional[str] = None
    error: Optional[str] = None
    confidence: Optional[float] = None

class BaseLLMClient(ABC):
    """Abstract interface for all LLM providers."""
    
    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize client with provider-specific credentials/config."""
        pass
    
    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Generate structured JSON output from the LLM.
        
        All providers must support:
        - system/user prompt separation
        - temperature control
        - max_tokens limit
        - JSON-mode or structured output (if available)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if client is properly configured and reachable."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier (e.g., 'groq', 'openai', 'vllm')."""
        pass