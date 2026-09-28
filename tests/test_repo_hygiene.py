"""
Guards against the real API key ever being committed again (it was, once:
see docs/AI_USAGE.md). Skipped automatically when run outside a git
checkout (e.g. from the submitted zip). The key value is never printed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.config import clean_secret

ROOT = Path(__file__).resolve().parent.parent


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or _git("rev-parse", "--is-inside-work-tree").returncode != 0
    or not (ROOT / ".git").exists(),
    reason="not a git checkout",
)


def _current_api_key() -> str:
    key = clean_secret(os.environ.get("API_KEY"))
    env_file = ROOT / ".env"
    if not key and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("API_KEY="):
                key = clean_secret(line.split("=", 1)[1])
    return key


def _tracked_files() -> list[str]:
    out = _git("ls-files", "-z").stdout.decode()
    return [f for f in out.split("\0") if f]


def test_env_file_is_not_tracked():
    assert ".env" not in _tracked_files(), "the real .env is tracked by git -- run: git rm --cached .env"


def test_env_file_is_gitignored():
    assert _git("check-ignore", "-q", ".env").returncode == 0


def test_no_tracked_file_contains_the_api_key():
    key = _current_api_key()
    if len(key) < 8:
        pytest.skip("API_KEY not set")
    needle = key.encode()
    leaks = [f for f in _tracked_files()
             if (ROOT / f).is_file() and needle in (ROOT / f).read_bytes()]
    # Only file NAMES are reported, never the key.
    assert not leaks, f"API key found in tracked file(s): {leaks}"
