from __future__ import annotations
import os, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests
from .config import LLMConfig

class LLMError(RuntimeError):
    pass

@dataclass
class LLMResponse:
    content: str
    raw: Dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_elapsed: float = 0.0

class TokenUsageTracker:
    def __init__(self):
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_time: float = 0.0
        self.api_calls: int = 0

    def update(self, response: LLMResponse) -> None:
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        self.total_tokens += response.total_tokens
        self.total_time += response.time_elapsed
        self.api_calls += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_time_seconds": round(self.total_time, 2),
            "api_calls": self.api_calls,
            "avg_time_per_call_seconds": round(self.total_time / self.api_calls, 2) if self.api_calls > 0 else 0.0
        }

class OpenAICompatibleChatClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def _api_key(self) -> str:
        key = os.getenv(self.cfg.api_key_env, "").strip()
        if not key:
            raise LLMError(f"Missing API key. Please set env var {self.cfg.api_key_env}.")
        return key

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        url = f"{self.cfg.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
        payload = {"model": self.cfg.model, "messages": messages, "temperature": self.cfg.temperature}

        start_time = time.time()
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_seconds)
                if r.status_code >= 400:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                end_time = time.time()

                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                time_elapsed = end_time - start_time

                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    raw=data,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    time_elapsed=time_elapsed
                )
            except Exception as e:
                last_err = e
                time.sleep(1.0 + attempt * 0.5)
        raise LLMError(f"LLM request failed after retries: {last_err}")
