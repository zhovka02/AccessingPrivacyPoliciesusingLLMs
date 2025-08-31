# llm_client.py
"""
Unified LLM client wrapper for multiple providers (OpenAI, Gemini, OpenRouter, Ollama).

This class standardizes access to LLM backends via AiSuite,
handling provider-specific API base URLs, authentication,
and optional rate limiting (for OpenRouter).

Example:
--------
from llm_client import LLMClient

client = LLMClient("openai:gpt-4.1-mini")
pred = client.classify(system_prompt, user_prompt)
print(pred)
"""

import os
import threading
from collections import deque
from typing import Optional
import aisuite as ai


class LLMClient:
    """
    Client wrapper for different LLM providers using AiSuite.
    Supports: OpenAI (default), Gemini, OpenRouter, and Ollama.

    Parameters
    ----------
    model : str, default="openai:gpt-4.1"
        Model identifier. Can be prefixed with a provider:
        - "openai:gpt-4.1-mini"
        - "gemini:gemini-2.0-flash"
        - "openrouter:gpt-4-32k"
        - "ollama:llama2"
    api_key : str, optional
        Override API key (otherwise read from environment).
    """

    def __init__(self, model: str = "openai:gpt-4.1", api_key: Optional[str] = None):
        # Extract provider prefix if present
        if ":" in model:
            provider, model_name = model.split(":", 1)
        else:
            provider, model_name = "openai", model

        self.model = model
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key

        # Rate limiting for OpenRouter (20 requests/minute)
        self._rate_limit_lock = threading.Lock()
        self._openrouter_timestamps = deque()

        # Provider-specific setup
        if self.provider == "gemini":
            provider_settings = {
                "openai": {
                    "api_key": os.getenv("GOOGLE_API_KEY"),
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                    "max_retries": 100,
                }
            }
            self.model = "openai:" + model_name
            self.client = ai.Client(provider_settings)

        elif self.provider == "openrouter":
            provider_settings = {
                "openai": {
                    "api_key": os.getenv("OPENROUTER_API_KEY"),
                    "base_url": "https://openrouter.ai/api/v1",
                    "max_retries": 100,
                }
            }
            self.model = "openai:" + model_name
            self.client = ai.Client(provider_settings)

        elif self.provider == "ollama":
            provider_settings = {
                "openai": {
                    "base_url": "https://f2ki-h100-1.f2.htw-berlin.de:11435/v1",
                    "max_retries": 100,
                }
            }
            self.model = "openai:" + model_name
            self.client = ai.Client(provider_settings)

        else:
            # Default: OpenAI
            self.client = ai.Client()
            if api_key:
                self.client.api_key = api_key

    def classify(self, system_prompt: str, user_prompt: str) -> str:
        """
        Run classification for a given span using system+user prompts.

        Parameters
        ----------
        system_prompt : str
            Instruction prompt (role = system).
        user_prompt : str
            Input span/context (role = user).

        Returns
        -------
        str
            Model output (final label, stripped of whitespace).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
