.PHONY: help venv install download validate up logs health smoke benchmark monitor ers down build test package-server server-setup server-start sample-benchmark

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
RESULTS_DIR ?= results
TRACE ?= configs/sample_trace.json

help:
	@printf '%s\n' \
		'Targets:' \
		'  package-server  Create tar.gz for uploading to a GPU server' \
		'  server-setup    Run one-shot GPU server setup on the server itself' \
		'  server-start    Start vLLM, wait for health, and run smoke test' \
		'  sample-benchmark Run sample trace and recalculate ERS' \
		'  venv       Create Python virtual environment' \
		'  install    Install benchmark dependencies' \
		'  download   Download LiquidAI/LFM2.5-1.2B-Instruct into ./model' \
		'  validate   Validate GPU, Docker GPU access, model dir, disk, and port' \
		'  up         Start local vLLM server with GPU and ./model mounted read-only' \
		'  logs       Follow vLLM container logs' \
		'  health     Run HTTP health check' \
		'  smoke      Run streaming smoke test' \
		'  benchmark  Run multi-turn benchmark' \
		'  monitor    Collect GPU metrics until interrupted' \
		'  ers        Recalculate ERS from REQUESTS_FILE=results/.../requests.json' \
		'  down       Stop local server' \
		'  build      Build local submission image' \
		'  test       Run unit tests'

package-server:
	bash scripts/package_for_server.sh

server-setup:
	bash scripts/setup_gpu_server.sh

server-start:
	bash scripts/start_vllm_server.sh

sample-benchmark:
	bash scripts/run_sample_benchmark.sh

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-benchmark.txt

download:
	$(PY) scripts/download_model.py --repo-id LiquidAI/LFM2.5-1.2B-Instruct --local-dir ./model

validate:
	$(PY) scripts/validate_gpu_environment.py --model-dir ./model --port 8000

up:
	docker compose --env-file configs/baseline.env -f docker-compose.yml -f docker-compose.local.yml up -d

logs:
	docker compose -f docker-compose.yml -f docker-compose.local.yml logs -f lfm-vllm

health:
	$(PY) scripts/healthcheck.py --url http://127.0.0.1:8000/health

smoke:
	$(PY) scripts/smoke_test.py --base-url http://127.0.0.1:8000 --model LFM2.5-1.2B-Instruct

benchmark:
	$(PY) scripts/benchmark.py --config configs/benchmark.yaml --trace $(TRACE)

monitor:
	$(PY) scripts/collect_gpu_metrics.py --output $(RESULTS_DIR)/gpu_metrics.csv --interval-s 1.0

ers:
	$(PY) scripts/calculate_ers.py $(REQUESTS_FILE)

down:
	docker compose -f docker-compose.yml -f docker-compose.local.yml down

build:
	docker build -t $${DOCKERHUB_USER:-local}/lfm-vllm-baseline:0.1.0 .

test:
	$(PY) -m pytest -q
