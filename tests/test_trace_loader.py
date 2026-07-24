from __future__ import annotations

import pytest

from benchmark.trace_loader import trace_from_dict


def valid_trace() -> dict[str, object]:
    return {
        "name": "unit",
        "num_conversations": 2,
        "user_turns_per_conversation": 2,
        "total_request": 4,
        "shared_system_prefix_tokens": 10,
        "per_conversation_prefix_tokens": [3, 4],
        "new_user_tokens_per_turn": [5, 6],
        "output_tokens_per_turn_pinned": [7, 8],
        "arrival": {"type": "burst"},
    }


def test_valid_trace_loads() -> None:
    trace = trace_from_dict(valid_trace())
    assert trace.num_conversations == 2
    assert trace.conversation_prefix_tokens(1) == 4
    assert trace.new_user_tokens(1) == 6


def test_invalid_trace_field_fails() -> None:
    raw = valid_trace()
    raw["unexpected"] = 1
    with pytest.raises(ValueError, match="Unsupported trace fields"):
        trace_from_dict(raw)


def test_incorrect_total_request_count_fails() -> None:
    raw = valid_trace()
    raw["total_request"] = 5
    with pytest.raises(ValueError, match="total_request must equal"):
        trace_from_dict(raw)


def test_invalid_turn_list_length_fails() -> None:
    raw = valid_trace()
    raw["new_user_tokens_per_turn"] = [1]
    with pytest.raises(ValueError, match="new_user_tokens_per_turn"):
        trace_from_dict(raw)

