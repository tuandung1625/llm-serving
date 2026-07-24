#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import time
from typing import Any

import aiohttp


async def run_smoke_test(args: argparse.Namespace) -> int:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
    }
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    decoder = codecs.getincrementaldecoder("utf-8")()
    text_buffer = ""
    chunks = 0
    text_parts: list[str] = []
    first_content_s: float | None = None
    start_s = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as response:
            if response.status >= 400:
                body = await response.text()
                print(f"Smoke test failed with HTTP {response.status}: {body[:500]}")
                return 1
            async for raw_chunk in response.content.iter_any():
                text_buffer += decoder.decode(raw_chunk)
                lines = text_buffer.split("\n")
                text_buffer = lines.pop()
                for line in lines:
                    stripped = line.strip()
                    if not stripped.startswith("data:"):
                        continue
                    data = stripped[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    chunks += 1
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        if first_content_s is None:
                            first_content_s = time.perf_counter()
                        text_parts.append(content)
    text = "".join(text_parts)
    latency_ms = (time.perf_counter() - start_s) * 1000.0
    ttft_ms = (first_content_s - start_s) * 1000.0 if first_content_s is not None else None
    if not text:
        print("Smoke test failed: streaming completed with zero generated text.")
        return 1
    print("Smoke test passed.")
    print(f"  chunks: {chunks}")
    print(f"  ttft_ms: {ttft_ms:.3f}" if ttft_ms is not None else "  ttft_ms: null")
    print(f"  latency_ms: {latency_ms:.3f}")
    print(f"  response_prefix: {text[:200]!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a streaming OpenAI-compatible API smoke test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="LFM2.5-1.2B-Instruct")
    parser.add_argument("--prompt", default="Reply with exactly: baseline ready")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=60.0)
    return asyncio.run(run_smoke_test(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

