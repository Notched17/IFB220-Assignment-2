#!/usr/bin/env python3
"""
Entry point for the topic-constrained AI assistant.

Usage (from any folder):
    python main.py
    python /path/to/project/main.py

The only thing you need to provide is your IFB220 Developer API Portal key,
either as the environment variable API_KEY or in a file called .env in the
project folder:

    API_KEY=your-key-here

Everything else (portal URL, API version, model deployments, topic,
guardrail and context settings) is preconfigured in app_config.json. To
change the topic, edit "topic_config" in app_config.json to point at
another topics/*.json file (e.g. topics/motor_vehicles.json) -- no code
changes needed. Any setting can optionally be overridden by an
environment variable (see src/config.py).

In-chat commands:
    /usage   show a running total of API usage for this session
    /reset   clear conversation history (keeps the same topic)
    /quit    exit
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable however the script is launched (any working directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ConfigError, env_overrides_in_use, load_settings  # noqa: E402
from src.pipeline import GuardedChatSession  # noqa: E402
from src.topic import Topic  # noqa: E402


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        topic = Topic.load(settings.topic_config_path)
    except (OSError, ValueError) as exc:
        print(f"Topic configuration error: could not load {settings.topic_config_path}: {exc}",
              file=sys.stderr)
        return 1

    overrides = env_overrides_in_use()
    if overrides:
        print(f"(note: settings overridden by environment: {', '.join(overrides)})")

    print(f"=== {topic.display_name} ===")
    print("Ask me anything within scope. Type /quit to exit, /usage for usage stats.\n")

    session = GuardedChatSession.create(settings, topic)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/usage":
            print(f"[usage] {session.usage_summary()}")
            continue
        if user_input == "/reset":
            session = GuardedChatSession.create(settings, topic)
            print("[conversation reset]")
            continue

        response = session.handle_message(user_input)
        print(f"{topic.display_name}: {response}\n")

    print(f"\nSession usage summary: {session.usage_summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
