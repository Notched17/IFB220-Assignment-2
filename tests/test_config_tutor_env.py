"""
Simulates the marker's setup: a fresh copy of the project whose .env holds
ONLY `API_KEY=...`, run in a subprocess with a scrubbed environment (no
other variables) from a DIFFERENT working directory. No network is used.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src import config
from src.config import APP_CONFIG_PATH, DEFAULTS

ROOT = Path(__file__).resolve().parent.parent
IGNORE = shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", "cache",
                                ".env", "*.jsonl", "*.log", "*.zip", "live_*", "offline_*")

PRINT_SETTINGS = """
import dataclasses, json, sys
sys.path.insert(0, sys.argv[1])
from src.config import load_settings
s = load_settings()
print(json.dumps({k: str(v) for k, v in dataclasses.asdict(s).items()}))
"""


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "unzipped" / "project"
    shutil.copytree(ROOT, root, ignore=IGNORE)
    (tmp_path / "elsewhere").mkdir()
    return root


def scrubbed_env(tmp_path, **extra):
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    if "SYSTEMROOT" in os.environ:  # needed on Windows
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    env.update(extra)
    return env


def run_settings(project, tmp_path, **extra_env):
    proc = subprocess.run(
        [sys.executable, "-c", PRINT_SETTINGS, str(project)],
        cwd=tmp_path / "elsewhere", env=scrubbed_env(tmp_path, **extra_env),
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr


def expected_defaults(project) -> dict:
    exp = {k: str(v) for k, v in DEFAULTS.items() if k not in ("topic_config", "log_dir")}
    exp["topic_config_path"] = str(project / DEFAULTS["topic_config"])
    exp["log_dir"] = str(project / DEFAULTS["log_dir"])
    exp.pop("similarity_threshold_override")
    exp["similarity_threshold_override"] = "None"
    return exp


def check_all_defaults(settings: dict, project: Path):
    assert settings["api_key"] == "dummy"
    for name, value in expected_defaults(project).items():
        assert settings[name] == value, name


def test_env_file_with_only_api_key_is_enough(project, tmp_path):
    (project / ".env").write_text("API_KEY=dummy\n")
    settings, stderr = run_settings(project, tmp_path)
    check_all_defaults(settings, project)
    if (project / "app_config.json").exists():
        assert "Warning" not in stderr


def test_works_with_app_config_deleted(project, tmp_path):
    (project / ".env").write_text("API_KEY=dummy\n")
    (project / "app_config.json").unlink(missing_ok=True)
    settings, stderr = run_settings(project, tmp_path)
    check_all_defaults(settings, project)
    assert "app_config.json not found" in stderr


def test_malformed_app_config_falls_back_to_defaults(project, tmp_path):
    (project / ".env").write_text("API_KEY=dummy\n")
    (project / "app_config.json").write_text("{ this is not json")
    settings, stderr = run_settings(project, tmp_path)
    check_all_defaults(settings, project)
    assert "could not read app_config.json" in stderr


@pytest.mark.parametrize("line", [
    b'API_KEY="dummy"\n',
    b"API_KEY='dummy'\n",
    b"API_KEY=dummy   \n",
    b"API_KEY=dummy\r\n",
    b'API_KEY="dummy"\r\n',
    b"  API_KEY = dummy\n",
])
def test_api_key_line_variants(project, tmp_path, line):
    (project / ".env").write_bytes(b"# comment\r\n" + line)
    settings, _ = run_settings(project, tmp_path)
    assert settings["api_key"] == "dummy"


def test_environment_variable_overrides_file(project, tmp_path):
    (project / ".env").write_text("API_KEY=dummy\n")
    settings, _ = run_settings(project, tmp_path, MAX_RETRIES="5", TOPIC_CONFIG="topics/motor_vehicles.json")
    assert settings["max_retries"] == "5"
    assert settings["topic_config_path"] == str(project / "topics" / "motor_vehicles.json")


def test_missing_api_key_gives_friendly_error_not_traceback(project, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(project / "main.py")], cwd=tmp_path / "elsewhere",
        env=scrubbed_env(tmp_path), capture_output=True, text=True, timeout=60, input="",
    )
    assert proc.returncode != 0
    assert "API_KEY" in proc.stderr and ".env" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_blank_api_key_is_treated_as_missing(project, tmp_path):
    (project / ".env").write_text('API_KEY=""\n')
    proc = subprocess.run(
        [sys.executable, str(project / "main.py")], cwd=tmp_path / "elsewhere",
        env=scrubbed_env(tmp_path), capture_output=True, text=True, timeout=60, input="",
    )
    assert proc.returncode == 1 and "Traceback" not in proc.stderr


def test_main_starts_from_another_directory_with_absolute_path(project, tmp_path):
    (project / ".env").write_text("API_KEY=dummy\n")
    proc = subprocess.run(
        [sys.executable, str(project / "main.py")], cwd=tmp_path / "elsewhere",
        env=scrubbed_env(tmp_path), capture_output=True, text=True, timeout=60, input="/quit\n",
    )
    assert proc.returncode == 0, proc.stderr
    assert "=== Climbing Coach ===" in proc.stdout
    assert not (tmp_path / "elsewhere" / "logs").exists()  # never relative to the CWD


def test_bad_topic_path_gives_friendly_error(project, tmp_path):
    (project / ".env").write_text("API_KEY=dummy\n")
    proc = subprocess.run(
        [sys.executable, str(project / "main.py")], cwd=tmp_path / "elsewhere",
        env=scrubbed_env(tmp_path, TOPIC_CONFIG="topics/does_not_exist.json"),
        capture_output=True, text=True, timeout=60, input="",
    )
    assert proc.returncode == 1
    assert "Topic configuration error" in proc.stderr and "Traceback" not in proc.stderr


# --- in-process unit tests ------------------------------------------------------
@pytest.mark.skipif(not APP_CONFIG_PATH.exists(), reason="app_config.json deleted (defaults-only run)")
def test_defaults_match_committed_app_config():
    data = json.loads(APP_CONFIG_PATH.read_text())
    for name, default in DEFAULTS.items():
        assert data[name] == default, name


def test_fallback_parser_reads_only_api_key_when_dotenv_missing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_bytes(b'OTHER=1\r\nexport API_KEY="dummy"\r\n')
    # setenv first so monkeypatch restores the ORIGINAL state afterwards
    # (a bare delenv of an unset variable isn't undone, which would leak
    # API_KEY=dummy into later tests).
    for name in ("API_KEY", "OTHER"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    config._read_api_key_line(env_file)
    assert os.environ["API_KEY"] == "dummy"
    assert "OTHER" not in os.environ


def test_unwritable_log_dir_falls_back_to_temp(tmp_path, capsys):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")  # a FILE where the directory should be
    result = config.ensure_writable_dir(blocker / "logs", "logs_test")
    assert result != blocker / "logs" and result.exists()
    assert "not writable" in capsys.readouterr().err
