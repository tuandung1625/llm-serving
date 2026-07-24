# Viettel AI Race 2026 LLM Inference Baseline

GPU-only vLLM baseline for `LiquidAI/LFM2.5-1.2B-Instruct`. The server uses the required OpenAI-compatible vLLM entry point and exposes streaming chat completions on port `8000`. There is no CPU inference path, mock server, or CPU fallback.

## Architecture Overview

- `docker-compose.yml` is the competition Compose file. It assumes the organizer provides the model at `/model`, starts `vllm.entrypoints.openai.api_server`, exposes `8000`, requests one NVIDIA GPU, and makes no startup network calls.
- `docker-compose.local.yml` overlays local development settings: it mounts `./model:/model:ro`, adds a health check, and constrains the container near the official `3 CPU / 8 GB RAM` envelope.
- `scripts/download_model.py` downloads the complete Hugging Face snapshot into `./model` before serving.
- `scripts/validate_gpu_environment.py` fails early on missing GPU, broken Docker GPU access, missing model files, insufficient disk, or occupied port `8000`.
- `benchmark/` contains the async streaming benchmark client, trace loader, scheduler, token counting, ERS metrics, and result writer.
- `monitoring/` records lightweight GPU metrics with `nvidia-smi` separately from the benchmark process.

## Important Assumptions And Risks

- The official model path is `/model`; local testing mounts `./model` there read-only.
- The benchmark uses the local official tokenizer through `transformers.AutoTokenizer` when available. Whitespace token counting exists only for explicit test/development fallback and should not be used for scored runs.
- Synthetic prompts are deterministic when traces provide token counts instead of literal text. Exact input token counts are measured after message construction with the tokenizer, so requested synthetic token budgets and measured chat-template token counts may differ.
- TPOT is measured as `(last_token_time - first_token_time) / max(output_token_count - 1, 1)`. `last_token_time` is the arrival time of the last non-empty text delta. Streaming APIs do not expose every internal token timestamp.
- H200 MIG performance can differ from rented GPUs because of memory bandwidth, MIG isolation, clocks, driver/runtime details, and contention.

## Project Tree

```text
llm-serving-baseline/
├── Dockerfile
├── docker-compose.yml
├── docker-compose.local.yml
├── .dockerignore
├── .gitignore
├── README.md
├── requirements-benchmark.txt
├── Makefile
├── configs/
│   ├── baseline.env
│   ├── sample_trace.json
│   └── benchmark.yaml
├── scripts/
│   ├── download_model.py
│   ├── validate_gpu_environment.py
│   ├── healthcheck.py
│   ├── smoke_test.py
│   ├── benchmark.py
│   ├── calculate_ers.py
│   └── collect_gpu_metrics.py
├── benchmark/
│   ├── __init__.py
│   ├── trace_loader.py
│   ├── workload_runner.py
│   ├── streaming_client.py
│   ├── scheduler.py
│   ├── tokenizer_utils.py
│   ├── metrics.py
│   ├── result_writer.py
│   └── schemas.py
├── monitoring/
│   ├── __init__.py
│   └── gpu_metrics.py
├── tests/
│   ├── test_ers.py
│   ├── test_metrics.py
│   ├── test_trace_loader.py
│   ├── test_scheduler.py
│   └── test_token_counting.py
└── results/
    └── .gitkeep
```

## ERS And Timing Definitions

All timing values are milliseconds.

```text
x_ttft = clamp((400 - TTFT) / (400 - 10), 0, 1)
s_ttft = x_ttft ** 2
x_tpot = clamp((10 - TPOT_mean) / (10 - 1), 0, 1)
s_tpot = x_tpot ** 2
request_score = 0.5 * s_ttft + 0.5 * s_tpot
ERS = mean(request_score for all requests)
```

Errors, timeouts, and zero-output-token responses score `0`.

## GPU Machine Setup Commands

Official references: Docker Engine on Ubuntu: <https://docs.docker.com/engine/install/ubuntu/>. NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>.

Inspect the machine:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
cat /proc/driver/nvidia/version
docker --version
docker compose version
nproc
free -h
df -h
```

Install Docker on Ubuntu 24.04:

```bash
sudo apt remove -y docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc || true
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
newgrp docker
```

Install and configure NVIDIA Container Toolkit:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -i nvidia
```

Verify GPU access inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
docker run --rm --gpus all --entrypoint python3 vllm/vllm-openai:v0.22.1 \
  -c "import torch; print(torch.version.cuda); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
```

## Project Setup

```bash
git clone <your-repo-url> llm-serving-baseline
cd llm-serving-baseline
python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-benchmark.txt
```

If you are creating the project directly on the rented machine, copy or create this directory there and run the same virtual environment commands.

## Model Download Commands

```bash
cd llm-serving-baseline
. .venv/bin/activate
export HF_TOKEN=<optional_hugging_face_token>
python scripts/download_model.py --repo-id LiquidAI/LFM2.5-1.2B-Instruct --local-dir ./model
python scripts/validate_gpu_environment.py --model-dir ./model --port 8000
```

The model is never downloaded during container startup. `model/` and common weight extensions are ignored by Git.

## Server Startup Commands

Use both Compose files for local GPU testing. Later files override earlier files.

```bash
docker compose --env-file configs/baseline.env -f docker-compose.yml -f docker-compose.local.yml up -d
docker compose -f docker-compose.yml -f docker-compose.local.yml logs -f lfm-vllm
```

Wait for health:

```bash
python scripts/healthcheck.py --url http://127.0.0.1:8000/health
docker compose -f docker-compose.yml -f docker-compose.local.yml ps
```

Stop the server:

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml down
```

## Smoke Test Commands

```bash
. .venv/bin/activate
python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --model LFM2.5-1.2B-Instruct
```

## Benchmark Commands

```bash
. .venv/bin/activate
python scripts/benchmark.py --config configs/benchmark.yaml --trace configs/sample_trace.json
```

Outputs are written to a unique directory such as `results/2026-07-24T120000Z_baseline/`:

- `requests.json` and `requests.csv`
- `aggregate.json` and `aggregate.csv`
- `metadata.json`

## ERS Calculation Commands

```bash
python scripts/calculate_ers.py results/<experiment_id>/requests.json
python scripts/calculate_ers.py results/<experiment_id>/requests.csv --write-json results/<experiment_id>/aggregate_recalculated.json
```

## GPU Monitoring Commands

Run monitoring in a separate shell:

```bash
. .venv/bin/activate
python scripts/collect_gpu_metrics.py --output results/gpu_metrics.csv --interval-s 1.0
```

Stop it with `Ctrl+C`. Or run for a fixed duration:

```bash
python scripts/collect_gpu_metrics.py --output results/gpu_metrics.csv --interval-s 1.0 --duration-s 300
```

## Docker Build And Push Commands

Build a final image that preserves the required entry point:

```bash
export DOCKERHUB_USER=<your_dockerhub_user>
docker build -t "$DOCKERHUB_USER/lfm-vllm-baseline:0.1.0" .
docker run --rm --gpus all -v "$PWD/model:/model:ro" -p 8000:8000 "$DOCKERHUB_USER/lfm-vllm-baseline:0.1.0"
```

In another shell:

```bash
python scripts/healthcheck.py --url http://127.0.0.1:8000/health
python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --model LFM2.5-1.2B-Instruct
```

Push:

```bash
docker login
docker push "$DOCKERHUB_USER/lfm-vllm-baseline:0.1.0"
```

Render the final competition Compose with your image if required by the portal:

```bash
VLLM_IMAGE="$DOCKERHUB_USER/lfm-vllm-baseline:0.1.0" docker compose -f docker-compose.yml config > docker-compose.submission.yml
```

## Baseline vLLM Arguments

The baseline server uses:

```text
--model=/model
--served-model-name=LFM2.5-1.2B-Instruct
--host=0.0.0.0
--port=8000
--max-model-len=32768
--gpu-memory-utilization=0.95
--tensor-parallel-size=1
--enable-prefix-caching
```

## Future Experiments After Baseline Validation

Do not enable these until the baseline is healthy, measured, and reproducible:

- BF16/model-default sanity sweeps for `max_num_seqs`, batching, and scheduler knobs.
- Prefix-cache workload sensitivity.
- KV cache dtype experiments if allowed by the final rules.
- Quantization or speculative decoding only if explicitly allowed and measured against correctness.
- Docker image pinning by digest after the exact competition runtime is known.

## Submission-Readiness Checklist

- `docker-compose.yml` keeps the required entry point and mandatory arguments.
- The image starts offline with model path `/model`.
- No Hugging Face token or secret appears in Compose, logs, or committed files.
- Port `8000` is exposed.
- Streaming `/v1/chat/completions` smoke test passes.
- `scripts/benchmark.py` produces JSON and CSV request and aggregate results.
- `scripts/calculate_ers.py` matches aggregate ERS.
- GPU metrics CSV is collected separately.
- `metadata.json` records image, GPU, driver, Docker, vLLM version, model path/revision, vLLM args, trace hash, timestamps, and config.

## Rented GPU vs Official H200 MIG Differences

- H200 MIG has fixed memory and isolation; rented full GPUs may have different bandwidth, clocks, and thermals.
- Official CPU/RAM limits are tight. Local Compose applies `3` CPUs and `8g` memory, but host scheduling and cgroups can still differ.
- Driver is expected to be `590.x` with CUDA `13.x` support. Validate rented machines before benchmarking.
- MIG memory available is `18 GB`, so avoid changes that only work on larger VRAM.

## Short Baseline Validation Checklist

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
python scripts/validate_gpu_environment.py --model-dir ./model --port 8000
docker compose --env-file configs/baseline.env -f docker-compose.yml -f docker-compose.local.yml up -d
python scripts/healthcheck.py --url http://127.0.0.1:8000/health
python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --model LFM2.5-1.2B-Instruct
python scripts/benchmark.py --config configs/benchmark.yaml --trace configs/sample_trace.json
python scripts/calculate_ers.py results/<experiment_id>/requests.json
```

