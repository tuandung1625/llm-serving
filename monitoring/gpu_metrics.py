from __future__ import annotations

import csv
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event


FIELDS = [
    "sampling_timestamp",
    "gpu_index",
    "gpu_name",
    "gpu_utilization_percent",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_memory_utilization_percent",
    "power_usage_watts",
    "temperature_celsius",
]

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,utilization.memory,power.draw,temperature.gpu",
    "--format=csv,noheader,nounits",
]


def run_gpu_monitor(output_csv: str | Path, interval_s: float, duration_s: float | None = None) -> int:
    if interval_s <= 0:
        raise ValueError("interval_s must be > 0")
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stop_event = Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    start_s = time.monotonic()
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        while not stop_event.is_set():
            timestamp = datetime.now(timezone.utc).isoformat()
            rows = sample_gpu_metrics(timestamp)
            if not rows:
                return 1
            writer.writerows(rows)
            handle.flush()
            if duration_s is not None and time.monotonic() - start_s >= duration_s:
                break
            stop_event.wait(interval_s)
    return 0


def sample_gpu_metrics(timestamp: str) -> list[dict[str, object]]:
    try:
        result = subprocess.run(NVIDIA_SMI_QUERY, check=False, capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        print("nvidia-smi is not installed or is not on PATH.")
        return []
    except subprocess.TimeoutExpired:
        print("nvidia-smi timed out while sampling GPU metrics.")
        return []
    if result.returncode != 0:
        print(f"nvidia-smi failed while sampling GPU metrics: {result.stderr.strip()}")
        return []
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        rows.append(
            {
                "sampling_timestamp": timestamp,
                "gpu_index": _int_or_none(parts[0]),
                "gpu_name": parts[1],
                "gpu_utilization_percent": _float_or_none(parts[2]),
                "gpu_memory_used_mib": _float_or_none(parts[3]),
                "gpu_memory_total_mib": _float_or_none(parts[4]),
                "gpu_memory_utilization_percent": _float_or_none(parts[5]),
                "power_usage_watts": _float_or_none(parts[6]),
                "temperature_celsius": _float_or_none(parts[7]),
            }
        )
    return rows


def _float_or_none(value: str) -> float | None:
    cleaned = value.strip()
    if cleaned in {"", "N/A", "[N/A]"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _int_or_none(value: str) -> int | None:
    number = _float_or_none(value)
    return int(number) if number is not None else None

