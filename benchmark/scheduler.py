from __future__ import annotations

import random
from typing import Any

from benchmark.schemas import RequestPlan, WorkloadTrace


def generate_arrival_times_s(arrival: dict[str, Any], total_request: int) -> list[float]:
    if total_request <= 0:
        raise ValueError("total_request must be > 0")
    arrival_type = str(arrival.get("type", "burst")).lower()
    if arrival_type == "burst":
        return [0.0 for _ in range(total_request)]
    if arrival_type in {"fixed", "fixed_interval"}:
        interval_s = _fixed_interval_s(arrival)
        return [index * interval_s for index in range(total_request)]
    if arrival_type == "poisson":
        rate = float(arrival.get("rate_per_second", 0.0))
        if rate <= 0:
            raise ValueError("arrival.rate_per_second must be > 0 for poisson arrival")
        rng = random.Random(int(arrival.get("seed", 0)))
        times = [0.0]
        for _ in range(1, total_request):
            times.append(times[-1] + rng.expovariate(rate))
        return times
    if arrival_type == "explicit":
        if "times_s" in arrival:
            times = [float(value) for value in arrival["times_s"]]
        elif "times_ms" in arrival:
            times = [float(value) / 1000.0 for value in arrival["times_ms"]]
        else:
            raise ValueError("explicit arrival requires times_s or times_ms")
        if len(times) != total_request:
            raise ValueError(f"explicit arrival length must equal total_request ({total_request})")
        if any(value < 0 for value in times):
            raise ValueError("arrival times must be non-negative")
        return times
    raise ValueError(f"Unsupported arrival.type: {arrival_type}")


def build_request_plans(trace: WorkloadTrace) -> list[RequestPlan]:
    trace.validate()
    arrivals = generate_arrival_times_s(trace.arrival, trace.total_request)
    plans: list[RequestPlan] = []
    request_index = 0
    for turn_index in range(trace.user_turns_per_conversation):
        for conversation_id in range(trace.num_conversations):
            output_tokens = trace.expected_output_tokens(turn_index)
            plans.append(
                RequestPlan(
                    request_id=request_index,
                    conversation_id=conversation_id,
                    turn_index=turn_index,
                    scheduled_arrival_s=arrivals[request_index],
                    new_user_tokens=trace.new_user_tokens(turn_index),
                    expected_output_tokens=output_tokens,
                    requested_output_tokens=output_tokens,
                )
            )
            request_index += 1
    if len(plans) != trace.total_request:
        raise ValueError(f"Scheduled {len(plans)} requests, expected {trace.total_request}")
    _validate_per_conversation_order(plans)
    return plans


def _fixed_interval_s(arrival: dict[str, Any]) -> float:
    if "interval_s" in arrival:
        interval = float(arrival["interval_s"])
    elif "interval_ms" in arrival:
        interval = float(arrival["interval_ms"]) / 1000.0
    elif "rate_per_second" in arrival:
        rate = float(arrival["rate_per_second"])
        if rate <= 0:
            raise ValueError("arrival.rate_per_second must be > 0")
        interval = 1.0 / rate
    else:
        raise ValueError("fixed arrival requires interval_s, interval_ms, or rate_per_second")
    if interval < 0:
        raise ValueError("fixed arrival interval must be non-negative")
    return interval


def _validate_per_conversation_order(plans: list[RequestPlan]) -> None:
    last_turn_by_conversation: dict[int, int] = {}
    last_arrival_by_conversation: dict[int, float] = {}
    for plan in plans:
        last_turn = last_turn_by_conversation.get(plan.conversation_id, -1)
        if plan.turn_index != last_turn + 1:
            raise ValueError(
                f"Conversation {plan.conversation_id} turn order is invalid: "
                f"got {plan.turn_index} after {last_turn}"
            )
        last_arrival = last_arrival_by_conversation.get(plan.conversation_id, -1.0)
        if plan.scheduled_arrival_s < last_arrival:
            raise ValueError(f"Conversation {plan.conversation_id} arrival times are not monotonic")
        last_turn_by_conversation[plan.conversation_id] = plan.turn_index
        last_arrival_by_conversation[plan.conversation_id] = plan.scheduled_arrival_s

