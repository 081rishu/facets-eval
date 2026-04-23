from typing import Optional, Dict, Any
from .base import BaseLLMClient
from .groq_client import GroqClient
# from .openai_client import OpenAIClient  # Future: uncomment when implemented
# from .vllm_client import VLLMClient      # Future: for local ≤16B models

class LLMClientFactory:
    """Factory to create LLM clients based on configuration."""
    
    PROVIDERS = {
        "groq": GroqClient,
        # "openai": OpenAIClient,
        # "vllm": VLLMClient,
    }
    
    @classmethod
    def create(
        cls,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLLMClient:
        """
        Create an LLM client instance.
        
        Args:
            provider: One of the supported providers ('groq', 'openai', 'vllm')
            api_key: API key for the provider (optional, can use env vars)
            model: Model name/ID to use
            **kwargs: Provider-specific configuration
            
        Returns:
            Configured LLM client instance
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Available: {list(cls.PROVIDERS.keys())}"
            )
        
        client_class = cls.PROVIDERS[provider]
        
        # Inject model if provided and supported by client
        if model:
            kwargs["model"] = model
            
        return client_class(api_key=api_key, **kwargs)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """Return list of available provider names."""
        return list(cls.PROVIDERS.keys())