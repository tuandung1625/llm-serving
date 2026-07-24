from __future__ import annotations

import math

from benchmark.metrics import (
    aggregate_results,
    request_score,
    tpot_component_score,
    ttft_component_score,
)
from benchmark.schemas import RequestResult


def test_ttft_below_10_ms_scores_one() -> None:
    assert ttft_component_score(5.0) == 1.0


def test_ttft_exactly_10_ms_scores_one() -> None:
    assert ttft_component_score(10.0) == 1.0


def test_ttft_between_10_and_400_ms() -> None:
    assert math.isclose(ttft_component_score(205.0), 0.25)


def test_ttft_exactly_400_ms_scores_zero() -> None:
    assert ttft_component_score(400.0) == 0.0


def test_ttft_above_400_ms_scores_zero() -> None:
    assert ttft_component_score(500.0) == 0.0


def test_tpot_below_1_ms_scores_one() -> None:
    assert tpot_component_score(0.5) == 1.0


def test_tpot_exactly_1_ms_scores_one() -> None:
    assert tpot_component_score(1.0) == 1.0


def test_tpot_between_1_and_10_ms() -> None:
    assert math.isclose(tpot_component_score(5.5), 0.25)


def test_tpot_exactly_10_ms_scores_zero() -> None:
    assert tpot_component_score(10.0) == 0.0


def test_tpot_above_10_ms_scores_zero() -> None:
    assert tpot_component_score(20.0) == 0.0


def test_error_response_scores_zero() -> None:
    assert request_score(10.0, 1.0, 10, error=True) == 0.0


def test_timeout_response_scores_zero() -> None:
    assert request_score(10.0, 1.0, 10, timeout=True) == 0.0


def test_zero_token_response_scores_zero() -> None:
    assert request_score(10.0, 1.0, 0) == 0.0


def test_ers_aggregation() -> None:
    rows = [
        _result(0, score=request_score(10.0, 1.0, 8), output_tokens=8),
        _result(1, score=0.0, output_tokens=0, success=False, error_type="zero_token_response"),
    ]
    aggregate = aggregate_results("exp", rows, total_duration_s=2.0)
    assert aggregate.request_count == 2
    assert aggregate.successful_requests == 1
    assert aggregate.zero_token_responses == 1
    assert math.isclose(aggregate.ers, 0.5)


def _result(
    request_id: int,
    *,
    score: float,
    output_tokens: int,
    success: bool = True,
    error_type: str | None = None,
) -> RequestResult:
    return RequestResult(
        experiment_id="exp",
        request_id=request_id,
        conversation_id=0,
        turn_index=request_id,
        scheduled_arrival_s=0.0,
        actual_start_offset_s=0.0,
        http_status=200,
        success=success,
        error_type=error_type,
        error_message=None,
        timeout=False,
        ttft_ms=10.0,
        tpot_ms=1.0,
        latency_ms=20.0,
        input_token_count=10,
        output_token_count=output_tokens,
        expected_output_token_count=8,
        requested_output_token_count=8,
        stream_chunk_count=2,
        non_empty_delta_count=2,
        request_score=score,
        server_model_name="LFM2.5-1.2B-Instruct",
        benchmark_timestamp="2026-07-24T00:00:00Z",
        generated_text="hello",
    )

