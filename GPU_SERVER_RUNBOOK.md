# Huong Dan Nhanh Chay Tren Server GPU Bang GitHub

## 1. SSH vao server

Tren may local:

```bash
ssh root@YOUR_SERVER_PUBLIC_IP
```

Tren server, kiem tra GPU:

```bash
whoami
nvidia-smi
df -h
free -h
```

Neu `nvidia-smi` khong chay, doi image/server truoc.

## 2. Cai git va clone repo

Tren server:

```bash
export DEBIAN_FRONTEND=noninteractive
apt update
apt install -y git ca-certificates curl

mkdir -p ~/viettel
cd ~/viettel
git clone https://github.com/tuandung1625/llm-serving.git
cd llm-serving
```

Neu repo da clone roi va muon cap nhat code:

```bash
cd ~/viettel/llm-serving
git pull
```

## 3. Chay setup mot lan

Neu Hugging Face can token:

```bash
export HF_TOKEN="YOUR_HF_TOKEN"
```

Chay setup:

```bash
cd ~/viettel/llm-serving
bash scripts/setup_gpu_server.sh
```

Script nay se tu lam cac viec co ban:

- cai Python, Docker, Docker Compose plugin
- start `dockerd` tren may khong co `systemd`
- cai NVIDIA Container Toolkit
- test GPU trong Docker bang image `vllm/vllm-openai:v0.22.1`
- tao `.venv` va cai benchmark dependencies
- download `LiquidAI/LFM2.5-1.2B-Instruct` vao `./model`
- validate GPU, Docker, model dir, disk, port `8000`

Neu script fail o Docker daemon, xem log:

```bash
tail -100 /var/log/dockerd.log
```

Neu log co `Failed to create bridge docker0`, pull code moi roi chay lai:

```bash
cd ~/viettel/llm-serving
git pull
bash scripts/setup_gpu_server.sh
```

## 4. Start vLLM va smoke test

```bash
cd ~/viettel/llm-serving
bash scripts/start_vllm_server.sh
```

Script nay se:

- start container vLLM bang `docker compose`
- doi `/health`
- chay streaming smoke test

Xem log server:

```bash
cd ~/viettel/llm-serving
docker compose -f docker-compose.server.yml logs -f lfm-vllm
```

## 5. Chay benchmark mau

```bash
cd ~/viettel/llm-serving
bash scripts/run_sample_benchmark.sh
```

Ket qua nam trong:

```text
results/<timestamp>_baseline/
```

## 6. Lay results ve local

Tren server:

```bash
cd ~/viettel/llm-serving
tar -czf results.tar.gz results
```

Tren may local:

```bash
scp root@YOUR_SERVER_PUBLIC_IP:~/viettel/llm-serving/results.tar.gz .
```

## 7. Stop server

```bash
cd ~/viettel/llm-serving
docker compose -f docker-compose.server.yml down
```

## Ban copy nhanh

Tren server, sau khi SSH:

```bash
export DEBIAN_FRONTEND=noninteractive
apt update
apt install -y git ca-certificates curl
mkdir -p ~/viettel
cd ~/viettel
git clone https://github.com/tuandung1625/llm-serving.git
cd llm-serving
export HF_TOKEN="YOUR_HF_TOKEN"
bash scripts/setup_gpu_server.sh
bash scripts/start_vllm_server.sh
bash scripts/run_sample_benchmark.sh
```

Neu model khong can token thi bo dong `export HF_TOKEN=...`.
