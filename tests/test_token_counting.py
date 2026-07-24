from __future__ import annotations

import json

from benchmark.streaming_client import OpenAIStreamingClient
from benchmark.tokenizer_utils import WhitespaceTokenCounter


def test_output_token_counting_is_not_chunk_counting() -> None:
    counter = WhitespaceTokenCounter()
    chunks = ["hello world", " from vllm"]
    output_text = "".join(chunks)
    assert len(chunks) == 2
    assert counter.count_text(output_text) == 4


def test_streaming_chunks_with_empty_deltas_are_ignored_for_first_token() -> None:
    text_parts: list[str] = []
    empty_payload = {"choices": [{"delta": {}}]}
    result = OpenAIStreamingClient._process_sse_line(
        line=f"data: {json.dumps(empty_payload)}",
        text_parts=text_parts,
        now_s=10.0,
        first_token_holder=[None],
        last_token_holder=[None],
    )
    assert result["stream_chunk_count"] == 1
    assert result["non_empty_delta_count"] == 0
    assert result["first_token_s"] is None
    assert text_parts == []


def test_first_non_empty_delta_sets_first_token_time() -> None:
    text_parts: list[str] = []
    payload = {"choices": [{"delta": {"content": "hello"}}]}
    result = OpenAIStreamingClient._process_sse_line(
        line=f"data: {json.dumps(payload)}",
        text_parts=text_parts,
        now_s=12.0,
        first_token_holder=[None],
        last_token_holder=[None],
    )
    assert result["stream_chunk_count"] == 1
    assert result["non_empty_delta_count"] == 1
    assert result["first_token_s"] == 12.0
    assert result["last_token_s"] == 12.0
    assert text_parts == ["hello"]

