from __future__ import annotations

import math

from benchmark.metrics import calculate_tpot_ms, percentile


def test_tpot_uses_generation_interval_after_first_token() -> None:
    assert math.isclose(calculate_tpot_ms(100.0, 100.009, 10), 1.0)


def test_tpot_single_token_uses_denominator_one() -> None:
    assert math.isclose(calculate_tpot_ms(100.0, 100.006, 1), 6.0)


def test_tpot_zero_tokens_is_none() -> None:
    assert calculate_tpot_ms(100.0, 100.006, 0) is None


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 3.0], 50) == 2.0

