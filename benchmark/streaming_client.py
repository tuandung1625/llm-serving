from __future__ import annotations

import asyncio
import codecs
import json
import logging
import time
from typing import Any

import aiohttp

from benchmark.metrics import calculate_tpot_ms
from benchmark.schemas import ChatMessage, StreamMeasurement
from benchmark.tokenizer_utils import TokenCounter

LOGGER = logging.getLogger(__name__)


class OpenAIStreamingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        token_counter: TokenCounter,
        *,
        request_timeout_s: float,
        connect_timeout_s: float,
        max_connections: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.token_counter = token_counter
        self.temperature = temperature
        self.top_p = top_p
        timeout = aiohttp.ClientTimeout(total=request_timeout_s, connect=connect_timeout_s, sock_connect=connect_timeout_s)
        connector = aiohttp.TCPConnector(limit=max_connections, ttl_dns_cache=300)
        self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def close(self) -> None:
        await self._session.close()

    async def stream_chat_completion(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        request_id: int,
    ) -> StreamMeasurement:
        url = f"{self.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
        }
        start_s = time.perf_counter()
        first_token_s: float | None = None
        last_token_s: float | None = None
        text_parts: list[str] = []
        stream_chunk_count = 0
        non_empty_delta_count = 0
        http_status: int | None = None
        decoder = codecs.getincrementaldecoder("utf-8")()
        text_buffer = ""
        try:
            async with self._session.post(url, json=payload) as response:
                http_status = response.status
                if response.status >= 400:
                    body = await response.text()
                    return StreamMeasurement(
                        http_status=http_status,
                        generated_text="",
                        stream_chunk_count=0,
                        non_empty_delta_count=0,
                        ttft_ms=None,
                        tpot_ms=None,
                        latency_ms=(time.perf_counter() - start_s) * 1000.0,
                        error_type="http_error",
                        error_message=f"HTTP {response.status}: {body[:500]}",
                    )
                async for raw_chunk in response.content.iter_any():
                    decoded = decoder.decode(raw_chunk)
                    if not decoded:
                        continue
                    text_buffer += decoded
                    lines = text_buffer.split("\n")
                    text_buffer = lines.pop()
                    done = False
                    for line in lines:
                        parsed_done = self._process_sse_line(
                            line=line,
                            text_parts=text_parts,
                            now_s=time.perf_counter(),
                            first_token_holder=[first_token_s],
                            last_token_holder=[last_token_s],
                        )
                        first_token_s = parsed_done["first_token_s"]
                        last_token_s = parsed_done["last_token_s"]
                        stream_chunk_count += parsed_done["stream_chunk_count"]
                        non_empty_delta_count += parsed_done["non_empty_delta_count"]
                        done = done or parsed_done["done"]
                    if done:
                        break
                flush = decoder.decode(b"", final=True)
                if flush:
                    text_buffer += flush
                if text_buffer:
                    parsed_done = self._process_sse_line(
                        line=text_buffer,
                        text_parts=text_parts,
                        now_s=time.perf_counter(),
                        first_token_holder=[first_token_s],
                        last_token_holder=[last_token_s],
                    )
                    first_token_s = parsed_done["first_token_s"]
                    last_token_s = parsed_done["last_token_s"]
                    stream_chunk_count += parsed_done["stream_chunk_count"]
                    non_empty_delta_count += parsed_done["non_empty_delta_count"]
        except asyncio.TimeoutError:
            return StreamMeasurement(
                http_status=http_status,
                generated_text="".join(text_parts),
                stream_chunk_count=stream_chunk_count,
                non_empty_delta_count=non_empty_delta_count,
                ttft_ms=(first_token_s - start_s) * 1000.0 if first_token_s is not None else None,
                tpot_ms=None,
                latency_ms=(time.perf_counter() - start_s) * 1000.0,
                timeout=True,
                error_type="timeout",
                error_message=f"Request {request_id} exceeded configured timeout",
            )
        except aiohttp.ClientError as exc:
            return StreamMeasurement(
                http_status=http_status,
                generated_text="".join(text_parts),
                stream_chunk_count=stream_chunk_count,
                non_empty_delta_count=non_empty_delta_count,
                ttft_ms=(first_token_s - start_s) * 1000.0 if first_token_s is not None else None,
                tpot_ms=None,
                latency_ms=(time.perf_counter() - start_s) * 1000.0,
                error_type=exc.__class__.__name__,
                error_message=str(exc),
            )
        generated_text = "".join(text_parts)
        output_tokens = self.token_counter.count_text(generated_text)
        return StreamMeasurement(
            http_status=http_status,
            generated_text=generated_text,
            stream_chunk_count=stream_chunk_count,
            non_empty_delta_count=non_empty_delta_count,
            ttft_ms=(first_token_s - start_s) * 1000.0 if first_token_s is not None else None,
            tpot_ms=calculate_tpot_ms(first_token_s, last_token_s, output_tokens),
            latency_ms=(time.perf_counter() - start_s) * 1000.0,
        )

    @staticmethod
    def _process_sse_line(
        *,
        line: str,
        text_parts: list[str],
        now_s: float,
        first_token_holder: list[float | None],
        last_token_holder: list[float | None],
    ) -> dict[str, Any]:
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            return _empty_parse_result(first_token_holder[0], last_token_holder[0])
        if not stripped.startswith("data:"):
            return _empty_parse_result(first_token_holder[0], last_token_holder[0])
        payload = stripped[len("data:") :].strip()
        if payload == "[DONE]":
            result = _empty_parse_result(first_token_holder[0], last_token_holder[0])
            result["done"] = True
            return result
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            LOGGER.warning("invalid_sse_json", extra={"error": str(exc), "payload_prefix": payload[:120]})
            return _empty_parse_result(first_token_holder[0], last_token_holder[0])
        result = _empty_parse_result(first_token_holder[0], last_token_holder[0])
        result["stream_chunk_count"] = 1
        choices = event.get("choices") or []
        if not choices:
            return result
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if not content:
            return result
        text_parts.append(content)
        result["non_empty_delta_count"] = 1
        if result["first_token_s"] is None:
            result["first_token_s"] = now_s
        result["last_token_s"] = now_s
        return result


def _empty_parse_result(first_token_s: float | None, last_token_s: float | None) -> dict[str, Any]:
    return {
        "done": False,
        "first_token_s": first_token_s,
        "last_token_s": last_token_s,
        "stream_chunk_count": 0,
        "non_empty_delta_count": 0,
    }

