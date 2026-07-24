#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
from pathlib import Path


DEFAULT_CUDA_CHECK_IMAGE = "vllm/vllm-openai:v0.22.1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a GPU Docker host before launching the vLLM baseline.")
    parser.add_argument("--model-dir", default="./model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--min-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--min-gpu-vram-gb", type=float, default=18.0)
    parser.add_argument("--cuda-check-image", default=DEFAULT_CUDA_CHECK_IMAGE)
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    gpu_info = _check_nvidia_smi(failures, warnings, args.min_gpu_vram_gb)
    _check_host_resources(failures, warnings, args.min_free_disk_gb)
    _check_model_dir(Path(args.model_dir), failures)
    _check_port(args.port, failures)
    _check_docker(failures)
    _check_docker_gpu(args.cuda_check_image, failures)

    print("Host resource summary:")
    print(f"  CPU cores visible: { _cpu_count() }")
    print(f"  RAM total: { _format_gb(_ram_total_bytes()) }")
    print(f"  Disk free at cwd: { _format_gb(shutil.disk_usage('.').free) }")
    if gpu_info:
        print("GPU summary:")
        for gpu in gpu_info:
            print(f"  {gpu['name']} | total={gpu['memory_total_mib']} MiB free={gpu['memory_free_mib']} MiB driver={gpu['driver_version']}")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if failures:
        print("Environment validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("Environment validation passed.")
    return 0


def _check_nvidia_smi(failures: list[str], warnings: list[str], min_gpu_vram_gb: float) -> list[dict[str, str]] | None:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        failures.append("No NVIDIA GPU is visible through nvidia-smi, or nvidia-smi is not installed.")
        return None
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            failures.append(f"Could not parse nvidia-smi GPU line: {line}")
            continue
        name, total_mib, free_mib, driver = parts[:4]
        try:
            total_mib_f = float(total_mib)
            free_mib_f = float(free_mib)
        except ValueError:
            failures.append(f"Could not parse GPU memory from nvidia-smi line: {line}")
            continue
        if total_mib_f / 1024.0 < min_gpu_vram_gb:
            failures.append(f"GPU total VRAM is below {min_gpu_vram_gb:.1f} GiB: {total_mib_f / 1024.0:.2f} GiB")
        if free_mib_f / 1024.0 < min_gpu_vram_gb * 0.80:
            warnings.append(f"GPU free VRAM is low before vLLM startup: {free_mib_f / 1024.0:.2f} GiB")
        if not driver.startswith("590."):
            warnings.append(f"Driver version is {driver}; official environment uses 590.x.")
        gpus.append(
            {
                "name": name,
                "memory_total_mib": str(int(total_mib_f)),
                "memory_free_mib": str(int(free_mib_f)),
                "driver_version": driver,
            }
        )
    return gpus


def _check_host_resources(failures: list[str], warnings: list[str], min_free_disk_gb: float) -> None:
    free_disk_gb = shutil.disk_usage(".").free / (1024**3)
    if free_disk_gb < min_free_disk_gb:
        failures.append(f"Available disk space is insufficient: {free_disk_gb:.2f} GiB free, need at least {min_free_disk_gb:.2f} GiB")
    if _cpu_count() < 3:
        warnings.append("Fewer than 3 CPU cores are visible; official environment has 3 cores.")
    ram_gb = _ram_total_bytes() / (1024**3)
    if ram_gb < 8:
        warnings.append(f"Less than 8 GiB RAM is visible: {ram_gb:.2f} GiB.")


def _check_model_dir(model_dir: Path, failures: list[str]) -> None:
    if not model_dir.exists():
        failures.append(f"Model directory is missing: {model_dir}")
        return
    if not (model_dir / "config.json").exists():
        failures.append(f"Model directory is missing config.json: {model_dir}")
    if not list(model_dir.glob("*.safetensors")) and not list(model_dir.glob("*.bin")) and not list(model_dir.glob("*.pt")):
        failures.append(f"Model directory does not contain model weights: {model_dir}")
    tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json"]
    if not any((model_dir / name).exists() for name in tokenizer_files):
        failures.append(f"Model directory does not contain recognizable tokenizer files: {model_dir}")


def _check_port(port: int, failures: list[str]) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            failures.append(f"Port {port} is already occupied on localhost.")


def _check_docker(failures: list[str]) -> None:
    docker_version = _run(["docker", "--version"], timeout=10)
    if docker_version.returncode != 0:
        failures.append("Docker is not installed or is not on PATH.")
        return
    compose_version = _run(["docker", "compose", "version"], timeout=10)
    if compose_version.returncode != 0:
        failures.append("Docker Compose v2 is not available through 'docker compose'.")
    docker_info = _run(["docker", "info"], timeout=15)
    if docker_info.returncode != 0:
        failures.append("Docker daemon is not reachable by the current user.")
    elif "nvidia" not in docker_info.stdout.lower() and "nvidia" not in docker_info.stderr.lower():
        failures.append("Docker runtime does not appear to expose the NVIDIA runtime. Install or reconfigure NVIDIA Container Toolkit.")


def _check_docker_gpu(cuda_check_image: str, failures: list[str]) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--entrypoint",
        "python3",
        cuda_check_image,
        "-c",
        "import torch; print('torch_cuda', torch.version.cuda); print('device_count', torch.cuda.device_count()); assert torch.cuda.is_available(), 'CUDA unavailable inside container'",
    ]
    result = _run(command, timeout=180)
    if result.returncode != 0:
        failures.append(
            "Docker cannot access CUDA inside the check container. "
            f"Image={cuda_check_image}. stderr={result.stderr.strip()[:500]}"
        )


def _run(args: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(args=args, returncode=127, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(args=args, returncode=124, stdout=exc.stdout or "", stderr=exc.stderr or "command timed out")


def _cpu_count() -> int:
    return os_cpu_count() or 0


def os_cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


def _ram_total_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return 0
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return 0


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


if __name__ == "__main__":
    raise SystemExit(main())

