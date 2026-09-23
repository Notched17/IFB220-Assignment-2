#!/usr/bin/env python3
"""
Entry point for the topic-constrained AI assistant.

Usage:
    python main.py

Environment variables (see .env.example):
    API_KEY                 (required) IFB220 Developer API Portal key
    AZURE_OPENAI_BASE_URL   (required) IFB220 portal base URL, ending in /openai/
    TOPIC_CONFIG            (optional) path to a topics/*.json file;
                             defaults to topics/climbing.json.
                             Point this at topics/motor_vehicles.json or
                             topics/cinematography.json (or your own new
                             file) to retopic the assistant with ZERO
                             code changes.

In-chat commands:
    /usage   show a running total of API usage for this session
    /reset   clear conversation history (keeps the same topic)
    /quit    exit
"""

from __future__ import annotations

import sys

from src.config import load_settings
from src.pipeline import GuardedChatSession
from src.topic import Topic


def main() -> int:
    try:
        settings = load_settings()
    except EnvironmentError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        topic = Topic.load(settings.topic_config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Topic configuration error: {exc}", file=sys.stderr)
        return 1

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
