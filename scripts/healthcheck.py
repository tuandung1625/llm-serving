#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="HTTP health check for the vLLM OpenAI-compatible server.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/health")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    try:
        with urllib.request.urlopen(args.url, timeout=args.timeout) as response:
            if 200 <= response.status < 300:
                return 0
            print(f"Health check failed with HTTP {response.status}", file=sys.stderr)
            return 1
    except urllib.error.URLError as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

