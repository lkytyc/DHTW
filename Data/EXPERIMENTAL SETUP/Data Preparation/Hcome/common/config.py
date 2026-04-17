from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key_env: str
    model: str
    temperature: float = 0.2
    timeout_seconds: int = 120
    max_retries: int = 2


@dataclass(frozen=True)
class RunConfig:
    chunk_size_lines: int = 0
    max_lines: int = 0
    interactive: bool = True
    enforce_no_hallucination: bool = True


@dataclass(frozen=True)
class PathsConfig:
    project_input: Path
    source_text: Path
    output_dir: Path


@dataclass(frozen=True)
class AppConfig:
    project_name: str = "hcome_reproduction"
    timezone: str = "Asia/Tokyo"


@dataclass(frozen=True)
class Config:
    app: AppConfig
    paths: PathsConfig
    llm: LLMConfig
    run: RunConfig
    root_dir: Path


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_base_url(base_url: str) -> str:
    """
    Accept both:
      https://api.xxx.com
      https://api.xxx.com/v1
    And normalize to:
      https://api.xxx.com
    Because the client will append /v1/chat/completions
    """
    u = (base_url or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3].rstrip("/")
    return u


def _looks_like_api_key(s: str) -> bool:
    """
    Heuristic: treat it as a literal key if it looks like common provider keys.
    You can extend this if your provider uses a different prefix/pattern.
    """
    if not s:
        return False
    s = s.strip()
    return (
        s.startswith("sk-")
        or s.startswith("rk-")
        or s.startswith("api-")
        or len(s) >= 32  # fallback heuristic
    )


def _ensure_env_key(api_key_env_or_literal: str, preferred_env_name: str = "OPENAI_API_KEY") -> str:
    """
    If user provided a literal key in config (api_key_env_or_literal looks like a key),
    inject it into an env var so the existing llm_client can read it as usual.

    Returns the env var name that llm_client should read from.
    """
    val = (api_key_env_or_literal or "").strip()

    # Case A: user provided a literal key directly
    if _looks_like_api_key(val):
        # Put it into preferred_env_name (or keep existing if already set)
        if not os.getenv(preferred_env_name):
            os.environ[preferred_env_name] = val
        return preferred_env_name

    # Case B: user provided an env var name
    # (llm_client will read it; we keep it)
    return val or preferred_env_name


def load_config(config_path: str | Path) -> Config:
    config_path = Path(config_path).resolve()
    root_dir = config_path.parent
    raw = load_yaml(config_path)

    app = AppConfig(**raw.get("app", {}))

    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        project_input=(root_dir / paths_raw.get("project_input", "input/project_input.yaml")).resolve(),
        source_text=(root_dir / paths_raw.get("source_text", "input/source_text.txt")).resolve(),
        output_dir=(root_dir / paths_raw.get("output_dir", "outputs")).resolve(),
    )

    llm_raw = raw.get("llm", {})

    # Normalize base_url (strip /v1 if user included it)
    base_url = _normalize_base_url(str(llm_raw.get("base_url", "https://api.openai.com")))

    # api_key_env can be either:
    #   - an env var name (recommended), e.g. "OPENAI_API_KEY"
    #   - a literal key, e.g. "sk-xxxx" (supported for convenience)
    api_key_env_or_literal = str(llm_raw.get("api_key_env", "OPENAI_API_KEY"))
    # If literal key, inject it into env and return the env var name for llm_client to read
    api_key_env = _ensure_env_key(api_key_env_or_literal, preferred_env_name="OPENAI_API_KEY")

    llm = LLMConfig(
        base_url=base_url,
        api_key_env=api_key_env,
        model=str(llm_raw.get("model", "gpt-4o")),
        temperature=float(llm_raw.get("temperature", 0.2)),
        timeout_seconds=int(llm_raw.get("timeout_seconds", 120)),
        max_retries=int(llm_raw.get("max_retries", 2)),
    )

    run_raw = raw.get("run", {})
    run = RunConfig(
        chunk_size_lines=int(run_raw.get("chunk_size_lines", 0)),
        max_lines=int(run_raw.get("max_lines", 0)),
        interactive=bool(run_raw.get("interactive", True)),
        enforce_no_hallucination=bool(run_raw.get("enforce_no_hallucination", True)),
    )

    return Config(app=app, paths=paths, llm=llm, run=run, root_dir=root_dir)
