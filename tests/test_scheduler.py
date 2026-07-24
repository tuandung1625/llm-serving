from __future__ import annotations

from benchmark.scheduler import build_request_plans
from benchmark.trace_loader import trace_from_dict


def test_multi_turn_ordering() -> None:
    trace = trace_from_dict(
        {
            "num_conversations": 2,
            "user_turns_per_conversation": 3,
            "total_request": 6,
            "shared_system_prefix_tokens": 0,
            "per_conversation_prefix_tokens": 0,
            "new_user_tokens_per_turn": 4,
            "output_tokens_per_turn_pinned": 5,
            "arrival": {"type": "fixed_interval", "interval_ms": 100},
        }
    )
    plans = build_request_plans(trace)
    conv0_turns = [plan.turn_index for plan in plans if plan.conversation_id == 0]
    conv1_turns = [plan.turn_index for plan in plans if plan.conversation_id == 1]
    assert conv0_turns == [0, 1, 2]
    assert conv1_turns == [0, 1, 2]
    assert plans[0].conversation_id == 0
    assert plans[1].conversation_id == 1
    assert plans[2].conversation_id == 0


def test_cross_conversation_concurrency_is_schedulable() -> None:
    trace = trace_from_dict(
        {
            "num_conversations": 2,
            "user_turns_per_conversation": 2,
            "total_request": 4,
            "shared_system_prefix_tokens": 0,
            "per_conversation_prefix_tokens": 0,
            "new_user_tokens_per_turn": 4,
            "output_tokens_per_turn_pinned": 5,
            "arrival": {"type": "burst"},
        }
    )
    plans = build_request_plans(trace)
    first_turn_arrivals = [plan.scheduled_arrival_s for plan in plans if plan.turn_index == 0]
    assert first_turn_arrivals == [0.0, 0.0]


def test_explicit_arrival_validation() -> None:
    trace = trace_from_dict(
        {
            "num_conversations": 1,
            "user_turns_per_conversation": 2,
            "total_request": 2,
            "shared_system_prefix_tokens": 0,
            "per_conversation_prefix_tokens": 0,
            "new_user_tokens_per_turn": 4,
            "output_tokens_per_turn_pinned": 5,
            "arrival": {"type": "explicit", "times_ms": [0, 250]},
        }
    )
    plans = build_request_plans(trace)
    assert [plan.scheduled_arrival_s for plan in plans] == [0.0, 0.25]

