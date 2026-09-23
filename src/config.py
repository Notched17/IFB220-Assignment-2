"""
Central configuration for the assistant.

Every value that could plausibly change between environments (which topic to
run, which model deployment to call, how aggressive the guardrails are) is
read from an environment variable with a sensible default. This is what lets
the whole assistant be re-topicked or re-tuned WITHOUT touching any code:
you only ever edit `.env` and/or the JSON file under topics/.

Required by the assignment brief: the API key must come from the
environment variable API_KEY, never hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    # python-dotenv is optional at runtime but convenient for local dev.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a dev convenience only
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Required secret -----------------------------------------------
    api_key: str = field(default_factory=lambda: os.environ["API_KEY"])

    # --- IFB220 Developer API Portal endpoint details -------------------
    # NOTE: these three values are portal/course-specific and are NOT
    # something we can know in advance. Fill them in from the IFB220
    # Developer API Portal dashboard in your .env file (see .env.example).
    azure_endpoint: str = field(
        default_factory=lambda: os.environ["AZURE_OPENAI_ENDPOINT"]
    )
    api_version: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    )
    chat_deployment: str = field(
        default_factory=lambda: os.getenv("CHAT_DEPLOYMENT", "gpt-4.1-mini")
    )
    embedding_deployment: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    )

    # --- Topic configuration --------------------------------------------
    topic_config_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("TOPIC_CONFIG", str(PROJECT_ROOT / "topics" / "climbing.json"))
        )
    )

    # --- Guardrail tuning -------------------------------------------------
    # Overrides the per-topic threshold from the JSON file if set.
    similarity_threshold_override: float | None = field(
        default_factory=lambda: (
            float(os.environ["SIMILARITY_THRESHOLD"])
            if os.getenv("SIMILARITY_THRESHOLD")
            else None
        )
    )
    max_input_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_INPUT_CHARS", "2000"))
    )
    check_output_topicality: bool = field(
        default_factory=lambda: _bool_env("CHECK_OUTPUT_TOPICALITY", True)
    )

    # --- Context window management ---------------------------------------
    max_context_turns: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_TURNS", "6"))
    )
    max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
    )
    summarize_with_llm: bool = field(
        default_factory=lambda: _bool_env("SUMMARIZE_WITH_LLM", False)
    )

    # --- Reliability -------------------------------------------------------
    request_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_S", "20"))
    )
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))

    # --- Cost estimation (placeholder rates; adjust to actual portal pricing) --
    chat_cost_per_1k_prompt: float = field(
        default_factory=lambda: float(os.getenv("CHAT_COST_PER_1K_PROMPT", "0.0"))
    )
    chat_cost_per_1k_completion: float = field(
        default_factory=lambda: float(os.getenv("CHAT_COST_PER_1K_COMPLETION", "0.0"))
    )
    embedding_cost_per_1k: float = field(
        default_factory=lambda: float(os.getenv("EMBEDDING_COST_PER_1K", "0.0"))
    )

    # --- Logging -----------------------------------------------------------
    log_dir: Path = field(
        default_factory=lambda: Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))
    )


def load_settings() -> Settings:
    """Build a Settings object, raising a clear error if required env vars
    are missing rather than failing deep inside the API client later."""
    missing = [name for name in ("API_KEY", "AZURE_OPENAI_ENDPOINT") if not os.getenv(name)]
    if missing:
        raise EnvironmentError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )
    return Settings()
