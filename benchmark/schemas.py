from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class RequestPlan:
    request_id: int
    conversation_id: int
    turn_index: int
    scheduled_arrival_s: float
    new_user_tokens: int
    expected_output_tokens: int
    requested_output_tokens: int


@dataclass
class StreamMeasurement:
    http_status: int | None
    generated_text: str
    stream_chunk_count: int
    non_empty_delta_count: int
    ttft_ms: float | None
    tpot_ms: float | None
    latency_ms: float
    timeout: bool = False
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class RequestResult:
    experiment_id: str
    request_id: int
    conversation_id: int
    turn_index: int
    scheduled_arrival_s: float
    actual_start_offset_s: float
    http_status: int | None
    success: bool
    error_type: str | None
    error_message: str | None
    timeout: bool
    ttft_ms: float | None
    tpot_ms: float | None
    latency_ms: float
    input_token_count: int
    output_token_count: int
    expected_output_token_count: int
    requested_output_token_count: int
    stream_chunk_count: int
    non_empty_delta_count: int
    request_score: float
    server_model_name: str
    benchmark_timestamp: str
    generated_text: str = field(repr=False, default="")

    def to_public_dict(self, include_text: bool = False) -> dict[str, Any]:
        row = asdict(self)
        if not include_text:
            row.pop("generated_text", None)
        return row


@dataclass(frozen=True)
class WorkloadTrace:
    name: str
    num_conversations: int
    user_turns_per_conversation: int
    total_request: int
    shared_system_prefix_tokens: int
    per_conversation_prefix_tokens: int | list[int]
    new_user_tokens_per_turn: int | list[int]
    output_tokens_per_turn_pinned: int | list[int]
    arrival: dict[str, Any]
    shared_system_prefix_text: str | None = None
    per_conversation_prefix_texts: list[str] | None = None
    user_turn_texts: list[str] | None = None
    add_per_conversation_prefix_to_first_turn: bool = True

    def validate(self) -> None:
        if self.num_conversations <= 0:
            raise ValueError("num_conversations must be > 0")
        if self.user_turns_per_conversation <= 0:
            raise ValueError("user_turns_per_conversation must be > 0")
        expected_total = self.num_conversations * self.user_turns_per_conversation
        if self.total_request != expected_total:
            raise ValueError(
                "total_request must equal num_conversations * "
                f"user_turns_per_conversation ({expected_total}), got {self.total_request}"
            )
        _validate_non_negative_int("shared_system_prefix_tokens", self.shared_system_prefix_tokens)
        _validate_int_or_list(
            "per_conversation_prefix_tokens",
            self.per_conversation_prefix_tokens,
            expected_len=self.num_conversations,
        )
        _validate_int_or_list(
            "new_user_tokens_per_turn",
            self.new_user_tokens_per_turn,
            expected_len=self.user_turns_per_conversation,
        )
        _validate_int_or_list(
            "output_tokens_per_turn_pinned",
            self.output_tokens_per_turn_pinned,
            expected_len=self.user_turns_per_conversation,
            strictly_positive=True,
        )
        if not isinstance(self.arrival, dict):
            raise ValueError("arrival must be an object")
        if self.per_conversation_prefix_texts is not None and len(self.per_conversation_prefix_texts) != self.num_conversations:
            raise ValueError("per_conversation_prefix_texts length must match num_conversations")
        if self.user_turn_texts is not None and len(self.user_turn_texts) != self.user_turns_per_conversation:
            raise ValueError("user_turn_texts length must match user_turns_per_conversation")

    def conversation_prefix_tokens(self, conversation_id: int) -> int:
        return _value_for_index(self.per_conversation_prefix_tokens, conversation_id)

    def new_user_tokens(self, turn_index: int) -> int:
        return _value_for_index(self.new_user_tokens_per_turn, turn_index)

    def expected_output_tokens(self, turn_index: int) -> int:
        return _value_for_index(self.output_tokens_per_turn_pinned, turn_index)


@dataclass(frozen=True)
class BenchmarkConfig:
    base_url: str
    model: str
    model_path: str
    trace_path: str
    output_dir: str
    request_timeout_s: float
    connect_timeout_s: float
    max_connections: int
    temperature: float
    top_p: float
    include_output_text: bool
    tokenizer_required: bool

    def validate(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not self.model:
            raise ValueError("model must not be empty")
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be > 0")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")
        if self.max_connections <= 0:
            raise ValueError("max_connections must be > 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class AggregateMetrics:
    experiment_id: str
    ers: float
    request_count: int
    successful_requests: int
    failed_requests: int
    timeout_requests: int
    zero_token_responses: int
    ttft_mean_ms: float | None
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    ttft_p99_ms: float | None
    tpot_mean_ms: float | None
    tpot_p50_ms: float | None
    tpot_p95_ms: float | None
    tpot_p99_ms: float | None
    latency_mean_ms: float | None
    output_tokens_per_second: float
    requests_per_second: float
    total_benchmark_duration_s: float


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_int_or_list(
    name: str,
    value: int | list[int],
    expected_len: int,
    strictly_positive: bool = False,
) -> None:
    minimum = 1 if strictly_positive else 0
    if isinstance(value, int):
        if value < minimum:
            label = "positive" if strictly_positive else "non-negative"
            raise ValueError(f"{name} must be a {label} integer")
        return
    if not isinstance(value, list) or len(value) != expected_len:
        raise ValueError(f"{name} must be an integer or a list of length {expected_len}")
    for index, item in enumerate(value):
        if not isinstance(item, int) or item < minimum:
            label = "positive" if strictly_positive else "non-negative"
            raise ValueError(f"{name}[{index}] must be a {label} integer")


def _value_for_index(value: int | list[int], index: int) -> int:
    if isinstance(value, int):
        return value
    return value[index]

