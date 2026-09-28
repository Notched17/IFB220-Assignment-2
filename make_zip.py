#!/usr/bin/env python3
"""
Builds the submission archive ifb220-assignment2.zip (one top-level
folder) and then CHECKS it: the build fails loudly, and the zip is
deleted, if it contains a .env file (other than .env.example) or the
current API_KEY value anywhere. The key is never printed.

Usage (from any folder):  python make_zip.py
"""

from __future__ import annotations

import fnmatch
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "ifb220-assignment2.zip"
TOP = "ifb220-assignment2"

INCLUDE = [
    "README.md", "main.py", "app_config.json", "requirements.txt", ".env.example",
    ".gitignore", "make_zip.py", "logs/.gitkeep", "src", "topics", "tests", "docs",
]
EXCLUDE_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", ".git", "node_modules",
                ".vscode", "cache"}
EXCLUDE_FILES = ["*.pyc", ".DS_Store", "*.jsonl", "*.log", "*.zip", "package.json",
                 "package-lock.json"]


def is_forbidden_env(name: str) -> bool:
    return name.startswith(".env") and name != ".env.example"


def wanted(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]):
        return False
    if is_forbidden_env(path.name):
        return False
    if rel.as_posix() == "logs/.gitkeep":
        return True
    return not any(fnmatch.fnmatch(path.name, pat) for pat in EXCLUDE_FILES)


def collect() -> list[Path]:
    files: list[Path] = []
    for entry in INCLUDE:
        path = ROOT / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in sorted(path.rglob("*")) if p.is_file() and wanted(p))
        else:
            raise SystemExit(f"ERROR: required path missing: {entry}")
    # docs/evidence/*.jsonl are deliberate evidence samples, keep them.
    files.extend(sorted((ROOT / "docs" / "evidence").glob("*.jsonl")))
    return sorted(set(files))


def current_api_key() -> str:
    """API_KEY from the environment or the project .env (never printed)."""
    key = os.environ.get("API_KEY", "")
    env_file = ROOT / ".env"
    if not key.strip() and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            name, sep, value = line.partition("=")
            if sep and name.strip() == "API_KEY":
                key = value
    return key.replace("\r", "").strip().strip("\"'").strip()


def verify(zip_path: Path) -> list[str]:
    problems = []
    key = current_api_key().encode()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = name.rsplit("/", 1)[-1]
            if is_forbidden_env(base):
                problems.append(f"secret file in archive: {name}")
            if len(key) >= 8 and key in zf.read(name):
                problems.append(f"API key value found inside: {name}")  # name only, never the key
            if not name.startswith(TOP + "/"):
                problems.append(f"file outside top-level folder: {name}")
    return problems


def main() -> int:
    files = collect()
    ZIP_PATH.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, f"{TOP}/{path.relative_to(ROOT).as_posix()}")
    problems = verify(ZIP_PATH)
    if problems:
        ZIP_PATH.unlink()
        print("BUILD FAILED -- archive deleted:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    key_checked = "yes" if len(current_api_key()) >= 8 else "no (API_KEY not set)"
    print(f"Built {ZIP_PATH.name}: {len(files)} files, {ZIP_PATH.stat().st_size / 1024:.0f} KB. "
          f"No .env inside; checked for the API key value: {key_checked}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
