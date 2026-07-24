# Huong Dan Nhanh Chuyen Project Len Server GPU Va Chay Test

Runbook nay giu cac buoc co ban nhat. Ban co 2 file chinh:

1. `scripts/package_for_server.sh`
2. `scripts/setup_gpu_server.sh`

Gia dinh:

- Ban vao duoc server bang SSH.
- Server co NVIDIA GPU va `nvidia-smi` chay duoc.
- Ban khong dung `sudo`, chi dung `apt`.
- Ban chay server bang user co quyen cai package, thuong la `root`.

## 1. Dat bien o may local

```bash
export SERVER_USER=root
export SERVER_HOST="YOUR_SERVER_PUBLIC_IP_OR_DNS"
export SERVER_PORT=22
export PROJECT_NAME=llm-serving-baseline
```

## 2. Nen project o local

Tren may local:

```bash
cd /root/Project/Viettel
ARCHIVE="$(bash llm-serving-baseline/scripts/package_for_server.sh)"
ls -lh "$ARCHIVE"
scp -P "$SERVER_PORT" "$ARCHIVE" "$SERVER_USER@$SERVER_HOST:~/"
```

## 3. Kiem tra server GPU

SSH vao server:

```bash
ssh -p "$SERVER_PORT" "$SERVER_USER@$SERVER_HOST"
```

Tren server:

```bash
whoami
nvidia-smi
nproc
free -h
df -h
```

Neu `nvidia-smi` khong chay, doi server/image khac truoc khi lam tiep.

## 4. Giai nen project tren server

```bash
mkdir -p ~/viettel
tar -xzf ~/llm-serving-baseline_*.tar.gz -C ~/viettel
cd ~/viettel/llm-serving-baseline
```

## 5. Chay script setup mot lan

Neu can HF token:

```bash
export HF_TOKEN="YOUR_HF_TOKEN"
```

Chay setup:

```bash
bash scripts/setup_gpu_server.sh
```

Script nay se:

- cai Docker, Python, NVIDIA Container Toolkit
- start `dockerd`
- test Docker va GPU
- tao `.venv` va cai benchmark dependencies
- download model vao `./model`
- chay validation

## 6. Chay server vLLM

```bash
cd ~/viettel/llm-serving-baseline
docker compose \
  --env-file configs/baseline.env \
  -f docker-compose.yml \
  -f docker-compose.local.yml \
  up -d
```

Xem log:

```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml logs -f lfm-vllm
```

Health check:

```bash
cd ~/viettel/llm-serving-baseline
. .venv/bin/activate
python scripts/healthcheck.py --url http://127.0.0.1:8000/health
```

## 7. Smoke test

```bash
cd ~/viettel/llm-serving-baseline
. .venv/bin/activate
python scripts/smoke_test.py \
  --base-url http://127.0.0.1:8000 \
  --model LFM2.5-1.2B-Instruct
```

## 8. Chay benchmark

```bash
cd ~/viettel/llm-serving-baseline
. .venv/bin/activate
python scripts/benchmark.py \
  --config configs/benchmark.yaml \
  --trace configs/sample_trace.json
```

Tinh lai ERS:

```bash
EXP_DIR=$(ls -td results/*_baseline* | head -1)
python scripts/calculate_ers.py "$EXP_DIR/requests.json"
```

## 9. Lay results ve local

Tren server:

```bash
cd ~/viettel/llm-serving-baseline
tar -czf results.tar.gz results
```

Tren may local:

```bash
mkdir -p /root/Project/Viettel/results_from_server
scp -P "$SERVER_PORT" \
  "$SERVER_USER@$SERVER_HOST:~/viettel/llm-serving-baseline/results.tar.gz" \
  /root/Project/Viettel/results_from_server/
```

## 10. Stop server

```bash
cd ~/viettel/llm-serving-baseline
docker compose -f docker-compose.yml -f docker-compose.local.yml down
```
