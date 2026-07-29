# Huong Dan Nhanh Chay Tren Server GPU Bang GitHub

Runbook nay gia dinh ban dang SSH vao server bang `root`, nen cac lenh dung `apt` truc tiep, khong dung `sudo`.
Neu server cua ban khong phai root nhung co quyen sudo, them `sudo` truoc cac lenh ghi vao `/etc`, `apt`, `service`.

## 1. SSH vao server va kiem tra GPU

Tren may local:

```bash
ssh root@YOUR_SERVER_PUBLIC_IP
```

Tren server:

```bash
whoami
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
df -h
free -h
nproc
```

Neu `nvidia-smi` khong chay, dung lai va doi image/server truoc. Docker khong the thay GPU neu host khong thay GPU.

## 2. Cai Docker va NVIDIA Container Toolkit ngay tu dau

Chay nguyen khoi nay truoc khi clone/chay benchmark:

```bash
export DEBIAN_FRONTEND=noninteractive

apt update
apt install -y ca-certificates curl gnupg git python3 python3-venv python3-pip

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt update
apt install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
```

Restart Docker:

```bash
service docker restart || true
```

Neu server khong co `systemd/service` hoac Docker daemon chua chay, start thu cong:

```bash
mkdir -p /var/run /var/lib/docker /var/log
pkill dockerd 2>/dev/null || true
pkill containerd 2>/dev/null || true

nohup dockerd \
  --host=unix:///var/run/docker.sock \
  --iptables=false \
  --bridge=none \
  --ip-forward=false \
  --ip-masq=false \
  > /var/log/dockerd.log 2>&1 &

sleep 8
```

Kiem tra Docker va NVIDIA runtime:

```bash
docker --version
docker compose version
docker info | grep -i -A5 runtime || true
```

Verify GPU trong Docker:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

Neu CUDA 13 image khong pull duoc, chi de test runtime thi thu:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

Neu gap loi:

```text
could not select device driver "" with capabilities: [[gpu]]
```

thi NVIDIA Container Toolkit chua duoc cai/config dung. Chay lai tu dong `curl ... nvidia-container-toolkit`, `apt install -y nvidia-container-toolkit`, `nvidia-ctk runtime configure --runtime=docker`, roi restart Docker.

## 3. Clone repo

```bash
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

## 4. Cai Python env va download model

Neu Hugging Face can token:

```bash
export HF_TOKEN="hf_xxx_token_cua_ban"
```

Tao venv va cai dependencies:

```bash
cd ~/viettel/llm-serving
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-benchmark.txt
```

Download model vao `./model`:

```bash
python scripts/download_model.py \
  --repo-id LiquidAI/LFM2.5-1.2B-Instruct \
  --local-dir model
```

Validate nhanh:

```bash
python scripts/validate_gpu_environment.py --model-dir model --port 8000
```

## 5. Chay setup script neu muon tu dong validate lai

Buoc nay khong bat buoc neu ban da lam day du buoc 2 va 4, nhung nen chay de script check lai Docker/GPU/model/port:

```bash
cd ~/viettel/llm-serving
WITH_MODEL_DOWNLOAD=0 bash scripts/setup_gpu_server.sh
```

Neu chua download model o buoc 4, co the de script download:

```bash
cd ~/viettel/llm-serving
bash scripts/setup_gpu_server.sh
```

Neu script fail o Docker daemon, xem log:

```bash
tail -100 /var/log/dockerd.log
```

## 6. Start vLLM va smoke test

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
docker compose -f docker-compose/docker-compose.server.yml logs -f lfm-vllm
```

## 7. Chay golden benchmark cho mot compose

```bash
cd ~/viettel/llm-serving
bash run_golden_score.sh docker-compose/docker-compose-260725-101045-rtx4090-mimic-h200mig.yaml
```

Chay toan bo level1:

```bash
cd ~/viettel/llm-serving
bash run_level1_golden.sh
```

Gom ranking:

```bash
python scripts/summarize_all_golden_results.py
```

Ket qua:

```text
results/golden_runs/
results/golden_runs_ranking.csv
GOLDEN_RESULTS_ANALYSIS.md
```

## 8. Lay results ve local

Tren server:

```bash
cd ~/viettel/llm-serving
tar -czf results.tar.gz results GOLDEN_RESULTS_ANALYSIS.md
```

Tren may local:

```bash
scp root@YOUR_SERVER_PUBLIC_IP:~/viettel/llm-serving/results.tar.gz .
```

## 9. Stop server

```bash
cd ~/viettel/llm-serving
docker compose -f docker-compose/docker-compose.server.yml down || true
docker compose -f docker-compose/docker-compose-260725-101045-rtx4090-mimic-h200mig.yaml down || true
```

## Ban copy nhanh tu server moi

```bash
export DEBIAN_FRONTEND=noninteractive

apt update
apt install -y ca-certificates curl gnupg git python3 python3-venv python3-pip

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt update
apt install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

service docker restart || true
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi

mkdir -p ~/viettel
cd ~/viettel
git clone https://github.com/tuandung1625/llm-serving.git
cd llm-serving

export HF_TOKEN="hf_xxx_token_cua_ban"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-benchmark.txt
python scripts/download_model.py --repo-id LiquidAI/LFM2.5-1.2B-Instruct --local-dir model
python scripts/validate_gpu_environment.py --model-dir model --port 8000

bash run_golden_score.sh docker-compose/docker-compose-260725-101045-rtx4090-mimic-h200mig.yaml
```

Neu model khong can token thi bo dong `export HF_TOKEN=...`.
