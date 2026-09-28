"""
Central configuration for the assistant.

The ONLY required value is the secret API_KEY, which must come from the
environment (or a `.env` file in the project folder) -- the assignment
brief requires it is never hardcoded. Everything else has a working
default, so the program runs with nothing but `API_KEY=...` in `.env`.

Precedence for every non-secret setting (highest first):

  1. environment variable (optional override, e.g. TOPIC_CONFIG=...)
  2. app_config.json in the project root (committed, no secrets)
  3. DEFAULTS below (so the program still runs if app_config.json is
     missing or broken)

Relative paths (topic_config, log_dir) are always resolved against the
project root, never the current working directory, so `python
/any/path/main.py` behaves the same from any folder.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_CONFIG_PATH = PROJECT_ROOT / "app_config.json"

# Setting name -> default value. Keep in sync with app_config.json.
DEFAULTS: dict = {
    "base_url": "https://qut-ai.azure-api.net/ifb220/openai/",
    # Verified live: 2025-03-01-preview works for both chat and embeddings
    # on the IFB220 portal; "2025-04-14" returns 404 (it is the model
    # version, not an API version). See docs/evidence/api_version_probe.txt.
    "api_version": "2025-03-01-preview",
    "chat_deployment": "gpt-4.1-mini",
    # The brief names Ada-002, but the portal returns 404 for it;
    # text-embedding-3-small (also 1536 dimensions) is deployed instead.
    "embedding_deployment": "text-embedding-3-small",
    "topic_config": "topics/climbing.json",
    "similarity_threshold_override": None,
    "max_input_chars": 2000,
    "check_output_topicality": True,
    "max_context_turns": 6,
    "max_context_tokens": 3000,
    "request_timeout_s": 20.0,
    "max_retries": 3,
    # Illustrative USD per 1k tokens (public list prices), not portal billing.
    "chat_cost_per_1k_prompt": 0.0004,
    "chat_cost_per_1k_completion": 0.0016,
    "embedding_cost_per_1k": 0.00002,
    "log_dir": "logs",
}

# Setting name -> optional environment variable that overrides it.
ENV_OVERRIDES: dict = {
    "base_url": "AZURE_OPENAI_BASE_URL",
    "api_version": "AZURE_OPENAI_API_VERSION",
    "chat_deployment": "CHAT_DEPLOYMENT",
    "embedding_deployment": "EMBEDDING_DEPLOYMENT",
    "topic_config": "TOPIC_CONFIG",
    "similarity_threshold_override": "SIMILARITY_THRESHOLD",
    "max_input_chars": "MAX_INPUT_CHARS",
    "check_output_topicality": "CHECK_OUTPUT_TOPICALITY",
    "max_context_turns": "MAX_CONTEXT_TURNS",
    "max_context_tokens": "MAX_CONTEXT_TOKENS",
    "request_timeout_s": "REQUEST_TIMEOUT_S",
    "max_retries": "MAX_RETRIES",
    "chat_cost_per_1k_prompt": "CHAT_COST_PER_1K_PROMPT",
    "chat_cost_per_1k_completion": "CHAT_COST_PER_1K_COMPLETION",
    "embedding_cost_per_1k": "EMBEDDING_COST_PER_1K",
    "log_dir": "LOG_DIR",
}

MISSING_KEY_MESSAGE = (
    "API_KEY is not set. Put your IFB220 Developer API Portal key in the "
    f"environment, or in a file called .env in the project folder ({PROJECT_ROOT}) "
    "containing one line:  API_KEY=your-key-here"
)


class ConfigError(EnvironmentError):
    """Raised for configuration problems the user can fix (e.g. no API_KEY)."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = DEFAULTS["base_url"]
    api_version: str = DEFAULTS["api_version"]
    chat_deployment: str = DEFAULTS["chat_deployment"]
    embedding_deployment: str = DEFAULTS["embedding_deployment"]
    topic_config_path: Path = PROJECT_ROOT / DEFAULTS["topic_config"]
    similarity_threshold_override: float | None = None
    max_input_chars: int = DEFAULTS["max_input_chars"]
    check_output_topicality: bool = DEFAULTS["check_output_topicality"]
    max_context_turns: int = DEFAULTS["max_context_turns"]
    max_context_tokens: int = DEFAULTS["max_context_tokens"]
    request_timeout_s: float = DEFAULTS["request_timeout_s"]
    max_retries: int = DEFAULTS["max_retries"]
    chat_cost_per_1k_prompt: float = DEFAULTS["chat_cost_per_1k_prompt"]
    chat_cost_per_1k_completion: float = DEFAULTS["chat_cost_per_1k_completion"]
    embedding_cost_per_1k: float = DEFAULTS["embedding_cost_per_1k"]
    log_dir: Path = PROJECT_ROOT / DEFAULTS["log_dir"]


# ----------------------------------------------------------------------
# .env / API key
# ----------------------------------------------------------------------
def clean_secret(raw: str | None) -> str:
    """Strip whitespace, Windows '\\r' and one layer of surrounding quotes."""
    if raw is None:
        return ""
    value = raw.replace("\r", "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def _read_api_key_line(env_file: Path) -> None:
    """Fallback used only if python-dotenv isn't installed: read just the
    API_KEY line from .env so the program still works."""
    if "API_KEY" in os.environ or not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if line.startswith("API_KEY="):
            os.environ["API_KEY"] = clean_secret(line.split("=", 1)[1])
            return


def load_env_file() -> None:
    """Load .env from the project folder first (so it works from any CWD),
    then from the CWD. Real environment variables always win."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        _read_api_key_line(PROJECT_ROOT / ".env")
        _read_api_key_line(Path.cwd() / ".env")
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(override=False)


def read_api_key() -> str:
    key = clean_secret(os.environ.get("API_KEY"))
    if not key:
        raise ConfigError(MISSING_KEY_MESSAGE)
    return key


# ----------------------------------------------------------------------
# Non-secret settings
# ----------------------------------------------------------------------
def load_app_config(path: Path = APP_CONFIG_PATH) -> dict:
    """Read app_config.json. Missing or malformed -> warn and use DEFAULTS."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Warning: {path.name} not found; using built-in defaults.", file=sys.stderr)
        return {}
    except (OSError, ValueError) as exc:
        print(f"Warning: could not read {path.name} ({exc}); using built-in defaults.",
              file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"Warning: {path.name} is not a JSON object; using built-in defaults.",
              file=sys.stderr)
        return {}
    return data


def _coerce(name: str, value):
    """Convert a raw value (a string from the environment, or JSON) to the
    type of the matching default."""
    default = DEFAULTS[name]
    if name == "similarity_threshold_override":
        return None if value in (None, "") else float(value)
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return str(value).strip()


def resolve_setting(name: str, file_config: dict):
    env_name = ENV_OVERRIDES[name]
    if os.environ.get(env_name, "").strip():
        source, raw = f"environment variable {env_name}", os.environ[env_name]
    elif name in file_config:
        source, raw = APP_CONFIG_PATH.name, file_config[name]
    else:
        return DEFAULTS[name]
    try:
        return _coerce(name, raw)
    except (TypeError, ValueError):
        print(f"Warning: invalid value for {name} in {source}; using default "
              f"{DEFAULTS[name]!r}.", file=sys.stderr)
        return DEFAULTS[name]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_writable_dir(path: Path, label: str) -> Path:
    """Return `path` if it can be created and written to, otherwise a
    temp-directory fallback (with a warning) instead of crashing."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return path
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / f"ifb220_{label}"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"Warning: {label} directory {path} is not writable ({exc}); "
              f"using {fallback} instead.", file=sys.stderr)
        return fallback


def env_overrides_in_use() -> list[str]:
    """Names of optional environment overrides currently set (for a
    one-line startup notice, so a stray override is never a mystery)."""
    return [env for env in ENV_OVERRIDES.values() if os.environ.get(env, "").strip()]


def load_settings() -> Settings:
    load_env_file()
    api_key = read_api_key()
    file_config = load_app_config()
    values = {name: resolve_setting(name, file_config) for name in DEFAULTS}
    return Settings(
        api_key=api_key,
        base_url=values["base_url"],
        api_version=values["api_version"],
        chat_deployment=values["chat_deployment"],
        embedding_deployment=values["embedding_deployment"],
        topic_config_path=resolve_path(values["topic_config"]),
        similarity_threshold_override=values["similarity_threshold_override"],
        max_input_chars=values["max_input_chars"],
        check_output_topicality=values["check_output_topicality"],
        max_context_turns=values["max_context_turns"],
        max_context_tokens=values["max_context_tokens"],
        request_timeout_s=values["request_timeout_s"],
        max_retries=values["max_retries"],
        chat_cost_per_1k_prompt=values["chat_cost_per_1k_prompt"],
        chat_cost_per_1k_completion=values["chat_cost_per_1k_completion"],
        embedding_cost_per_1k=values["embedding_cost_per_1k"],
        log_dir=ensure_writable_dir(resolve_path(values["log_dir"]), "logs"),
    )
