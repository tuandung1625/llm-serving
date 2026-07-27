from __future__ import annotations

from dataclasses import asdict
from statistics import mean

from benchmark.schemas import AggregateMetrics, RequestResult

F_TTFT_MS = 10.0
C_TTFT_MS = 400.0
F_TPOT_MS = 1.0
C_TPOT_MS = 10.0
GAMMA = 2.0
WEIGHT_TTFT = 0.5
ACCURACY_DELTA_FREE = 0.10
ACCURACY_DELTA_ZERO = 0.16


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def ttft_component_score(ttft_ms: float) -> float:
    x_ttft = clamp((C_TTFT_MS - ttft_ms) / (C_TTFT_MS - F_TTFT_MS))
    return x_ttft**GAMMA


def tpot_component_score(tpot_ms: float) -> float:
    x_tpot = clamp((C_TPOT_MS - tpot_ms) / (C_TPOT_MS - F_TPOT_MS))
    return x_tpot**GAMMA


def request_score(
    ttft_ms: float | None,
    tpot_ms: float | None,
    output_token_count: int,
    *,
    error: bool = False,
    timeout: bool = False,
) -> float:
    if error or timeout or output_token_count <= 0 or ttft_ms is None or tpot_ms is None:
        return 0.0
    return WEIGHT_TTFT * ttft_component_score(ttft_ms) + (1.0 - WEIGHT_TTFT) * tpot_component_score(tpot_ms)


def accuracy_penalty(delta: float) -> float:
    if delta <= ACCURACY_DELTA_FREE:
        return 1.0
    if delta >= ACCURACY_DELTA_ZERO:
        return 0.0
    return 1.0 - ((delta - ACCURACY_DELTA_FREE) / (ACCURACY_DELTA_ZERO - ACCURACY_DELTA_FREE))


def final_submission_score(ers: float, baseline_accuracy: float, submission_accuracy: float) -> dict[str, float]:
    if not 0.0 <= ers <= 1.0:
        raise ValueError("ers must be in [0, 1]")
    if not 0.0 <= baseline_accuracy <= 1.0:
        raise ValueError("baseline_accuracy must be in [0, 1]")
    if not 0.0 <= submission_accuracy <= 1.0:
        raise ValueError("submission_accuracy must be in [0, 1]")
    delta = baseline_accuracy - submission_accuracy
    penalty = accuracy_penalty(delta)
    return {
        "ers": ers,
        "baseline_accuracy": baseline_accuracy,
        "submission_accuracy": submission_accuracy,
        "accuracy_delta": delta,
        "accuracy_penalty": penalty,
        "final_score": 100.0 * ers * penalty,
    }


def calculate_tpot_ms(
    first_token_time_s: float | None,
    last_token_time_s: float | None,
    output_token_count: int,
) -> float | None:
    if first_token_time_s is None or last_token_time_s is None or output_token_count <= 0:
        return None
    interval_ms = max((last_token_time_s - first_token_time_s) * 1000.0, 0.0)
    return interval_ms / max(output_token_count - 1, 1)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if not 0 <= q <= 100:
        raise ValueError("percentile q must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate_results(
    experiment_id: str,
    results: list[RequestResult],
    total_duration_s: float,
) -> AggregateMetrics:
    request_count = len(results)
    successes = [row for row in results if row.success]
    failures = [row for row in results if not row.success]
    timeouts = [row for row in results if row.timeout]
    zero_tokens = [row for row in results if row.output_token_count == 0]
    ttfts = [row.ttft_ms for row in successes if row.ttft_ms is not None]
    tpots = [row.tpot_ms for row in successes if row.tpot_ms is not None]
    latencies = [row.latency_ms for row in results if row.latency_ms is not None]
    total_output_tokens = sum(row.output_token_count for row in results)
    safe_duration = max(total_duration_s, 1e-9)
    ers = mean([row.request_score for row in results]) if results else 0.0
    return AggregateMetrics(
        experiment_id=experiment_id,
        ers=ers,
        request_count=request_count,
        successful_requests=len(successes),
        failed_requests=len(failures),
        timeout_requests=len(timeouts),
        zero_token_responses=len(zero_tokens),
        ttft_mean_ms=mean(ttfts) if ttfts else None,
        ttft_p50_ms=percentile(ttfts, 50),
        ttft_p95_ms=percentile(ttfts, 95),
        ttft_p99_ms=percentile(ttfts, 99),
        tpot_mean_ms=mean(tpots) if tpots else None,
        tpot_p50_ms=percentile(tpots, 50),
        tpot_p95_ms=percentile(tpots, 95),
        tpot_p99_ms=percentile(tpots, 99),
        latency_mean_ms=mean(latencies) if latencies else None,
        output_tokens_per_second=total_output_tokens / safe_duration,
        requests_per_second=request_count / safe_duration,
        total_benchmark_duration_s=total_duration_s,
    )


def aggregate_to_dict(metrics: AggregateMetrics) -> dict[str, object]:
    return asdict(metrics)
