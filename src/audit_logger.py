"""
Three separate logs, deliberately kept apart because they serve different
audiences and have different retention/sensitivity considerations
(logs/usage.jsonl, the per-API-call token log, lives in usage_monitor.py):

  * logs/audit.jsonl  -- one structured JSON record per user turn, with
    the outcome of every guardrail layer. This is what you'd hand to a
    security reviewer to answer "did the guardrails work, and on what
    did they trigger". Includes the raw (sanitised) user text, since for
    a coaching assistant with no expected PII, seeing the actual flagged
    text is far more useful for auditing than a redacted placeholder --
    documented here as a deliberate trade-off; a production deployment
    handling sensitive user data would hash or truncate this field.

  * logs/errors.log -- a conventional rotating text log via the stdlib
    `logging` module, for operational issues (API errors, timeouts,
    unexpected exceptions). This is what you'd tail while running the
    service, separate from the security-focused audit trail.
"""

from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path


def build_error_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    # One logger per log directory, so a second session (or a test) writing
    # to a different folder doesn't silently log into the first one's file.
    logger = logging.getLogger(f"ifb220_a2.errors.{log_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:  # avoid duplicate handlers on repeated construction
        handler = RotatingFileHandler(
            log_dir / "errors.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


class AuditLogger:
    def __init__(self, log_dir: Path):
        self._path = log_dir / "audit.jsonl"
        log_dir.mkdir(parents=True, exist_ok=True)

    def log_turn(
        self,
        *,
        session_id: str,
        turn_index: int,
        user_text: str,
        action: str,
        layers: dict,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        record = {
            "timestamp": time.time(),
            "session_id": session_id,
            "turn_index": turn_index,
            "user_text": user_text,
            "action": action,  # answered | refused_input | refused_output | error
            "layers": layers,
            "latency_ms": round(latency_ms, 1),
            "error": error,
        }
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
