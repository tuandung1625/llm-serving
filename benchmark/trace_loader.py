from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.schemas import WorkloadTrace

ALLOWED_TRACE_KEYS = {
    "name",
    "description",
    "num_conversations",
    "user_turns_per_conversation",
    "total_request",
    "shared_system_prefix_tokens",
    "per_conversation_prefix_tokens",
    "new_user_tokens_per_turn",
    "output_tokens_per_turn_pinned",
    "arrival",
    "shared_system_prefix_text",
    "per_conversation_prefix_texts",
    "user_turn_texts",
    "add_per_conversation_prefix_to_first_turn",
}


def load_trace(path: str | Path) -> WorkloadTrace:
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file does not exist: {trace_path}")
    with trace_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Trace root must be a JSON object")
    return trace_from_dict(raw)


def trace_from_dict(raw: dict[str, Any]) -> WorkloadTrace:
    unknown_keys = sorted(set(raw) - ALLOWED_TRACE_KEYS)
    if unknown_keys:
        raise ValueError(f"Unsupported trace fields: {', '.join(unknown_keys)}")
    required = {
        "num_conversations",
        "user_turns_per_conversation",
        "total_request",
        "shared_system_prefix_tokens",
        "per_conversation_prefix_tokens",
        "new_user_tokens_per_turn",
        "output_tokens_per_turn_pinned",
        "arrival",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Missing required trace fields: {', '.join(missing)}")
    trace = WorkloadTrace(
        name=str(raw.get("name", "trace")),
        num_conversations=raw["num_conversations"],
        user_turns_per_conversation=raw["user_turns_per_conversation"],
        total_request=raw["total_request"],
        shared_system_prefix_tokens=raw["shared_system_prefix_tokens"],
        per_conversation_prefix_tokens=raw["per_conversation_prefix_tokens"],
        new_user_tokens_per_turn=raw["new_user_tokens_per_turn"],
        output_tokens_per_turn_pinned=raw["output_tokens_per_turn_pinned"],
        arrival=raw["arrival"],
        shared_system_prefix_text=raw.get("shared_system_prefix_text"),
        per_conversation_prefix_texts=raw.get("per_conversation_prefix_texts"),
        user_turn_texts=raw.get("user_turn_texts"),
        add_per_conversation_prefix_to_first_turn=bool(raw.get("add_per_conversation_prefix_to_first_turn", True)),
    )
    trace.validate()
    return trace

