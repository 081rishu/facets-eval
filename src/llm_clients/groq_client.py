import os
import json
import re
import time
from typing import Optional, Dict, Any
from groq import Groq
from .base import BaseLLMClient, LLMResponse


class GroqClient(BaseLLMClient):
    """Groq API client implementation with retries, validation, and confidence hooks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        **kwargs
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")

        self.model = model
        self._kwargs = kwargs

        self.max_retries = kwargs.get("max_retries", 3)
        self.retry_delay = kwargs.get("retry_delay_sec", 2)
        self.enable_logprobs = kwargs.get("enable_logprobs", False)
        self.timeout = kwargs.get("timeout_sec", 30)

        self.client = Groq(
            api_key=self.api_key,
            timeout=self.timeout
        )

    def _safe_parse_json(self, content: str) -> Optional[dict]:
        """Attempt to parse JSON with cleanup fallback."""
        try:
            return json.loads(content)
        except:
            try:
                content = re.sub(r"```json|```", "", content).strip()
                return json.loads(content)
            except:
                return None

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:

        request_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        # optional params
        if self.enable_logprobs:
            request_kwargs["logprobs"] = True

        request_kwargs.update(self._kwargs)
        request_kwargs.update(kwargs)

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**request_kwargs)

                content = response.choices[0].message.content

                parsed = self._safe_parse_json(content)

                if parsed is None:
                    raise ValueError("Invalid JSON returned by model")

                # --- confidence placeholder (future calibration hook) ---
                confidence = None
                if self.enable_logprobs:
                    # you can later compute from token logprobs
                    confidence = 0.8  # placeholder

                return LLMResponse(
                    content=json.dumps(parsed),
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                    model=self.model,
                    confidence=confidence,
                )

            except Exception as e:
                last_error = str(e)

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return LLMResponse(
                        content="",
                        error=f"Failed after {self.max_retries} attempts: {last_error}",
                        model=self.model,
                    )

    def is_available(self) -> bool:
        """Lightweight health check."""
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except:
            return False

    @property
    def provider_name(self) -> str:
        return "groq"