#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/model}"
PORT="${PORT:-8000}"
REPO_ID="${REPO_ID:-LiquidAI/LFM2.5-1.2B-Instruct}"
WITH_MODEL_DOWNLOAD="${WITH_MODEL_DOWNLOAD:-1}"

log() {
  printf '[setup] %s\n' "$*"
}

die() {
  printf '[setup][error] %s\n' "$*" >&2
  exit 1
}

require_root_like() {
  if [[ "$(id -u)" -ne 0 ]]; then
    die "script nay can chay bang root hoac user co quyen apt va ghi /etc"
  fi
}

check_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || die "khong tim thay nvidia-smi"
  nvidia-smi >/dev/null 2>&1 || die "nvidia-smi khong chay duoc"
}

install_base_packages() {
  export DEBIAN_FRONTEND=noninteractive
  log "install base packages"
  apt update
  apt install -y ca-certificates curl git gnupg2 python3 python3-venv python3-pip
}

install_docker() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    log "docker va compose da co san"
    docker --version || true
    docker compose version || true
    return
  fi

  log "install docker"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc

  tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

  apt update
  apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

start_dockerd() {
  mkdir -p /var/run /var/lib/docker /var/log
  pkill dockerd 2>/dev/null || true
  pkill containerd 2>/dev/null || true

  log "start dockerd"
  nohup dockerd \
    --host=unix:///var/run/docker.sock \
    --iptables=false \
    --bridge=none \
    --ip-forward=false \
    --ip-masq=false \
    > /var/log/dockerd.log 2>&1 &
  sleep 8

  if docker info >/dev/null 2>&1; then
    log "dockerd da chay"
    return
  fi

  log "dockerd mode mac dinh fail, thu vfs"
  pkill dockerd 2>/dev/null || true
  pkill containerd 2>/dev/null || true
  nohup dockerd \
    --host=unix:///var/run/docker.sock \
    --iptables=false \
    --bridge=none \
    --ip-forward=false \
    --ip-masq=false \
    --storage-driver=vfs \
    > /var/log/dockerd.log 2>&1 &
  sleep 8

  docker info >/dev/null 2>&1 || {
    tail -100 /var/log/dockerd.log >&2 || true
    die "khong start duoc dockerd, xem /var/log/dockerd.log"
  }
}

install_nvidia_toolkit() {
  if docker info 2>/dev/null | grep -qi nvidia; then
    log "nvidia runtime da co san trong docker"
    return
  fi

  log "install nvidia container toolkit"
  rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

  apt update
  apt install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker

  start_dockerd
}

verify_docker_gpu() {
  log "verify gpu in docker"
  local pull_log="/tmp/vllm_image_pull.log"
  if ! docker pull vllm/vllm-openai:v0.22.1 2>&1 | tee "$pull_log"; then
    if grep -Eqi "unshare: operation not permitted|failed to extract layer.*operation not permitted|failed to mount.*operation not permitted|failed to unmount.*operation not permitted" "$pull_log"; then
      die "Docker pull fail vi server bi chan unshare/mount. Day thuong la GPU container/LXC khong du capability de chay Docker daemon. Hay thue VM/bare-metal co Docker support, hoac bat privileged/Docker-in-Docker tren provider."
    fi
    die "docker pull vllm/vllm-openai:v0.22.1 fail, xem $pull_log"
  fi

  docker run --rm --gpus all --network=none --entrypoint python3 vllm/vllm-openai:v0.22.1 \
    -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
}

setup_python_env() {
  log "setup python venv"
  cd "$PROJECT_ROOT"
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements-benchmark.txt
}

download_model_if_needed() {
  if [[ "$WITH_MODEL_DOWNLOAD" != "1" ]]; then
    log "bo qua download model"
    return
  fi

  log "download model vao $MODEL_DIR"
  cd "$PROJECT_ROOT"
  . .venv/bin/activate
  python scripts/download_model.py --repo-id "$REPO_ID" --local-dir "$MODEL_DIR"
}

validate_environment() {
  log "validate environment"
  cd "$PROJECT_ROOT"
  . .venv/bin/activate
  python scripts/validate_gpu_environment.py --model-dir "$MODEL_DIR" --port "$PORT"
}

summary() {
  cat <<EOF

Setup xong. Chay tiep:

cd $PROJECT_ROOT
. .venv/bin/activate
bash scripts/start_vllm_server.sh
bash scripts/run_sample_benchmark.sh
EOF
}

main() {
  require_root_like
  check_gpu
  install_base_packages
  install_docker
  start_dockerd
  install_nvidia_toolkit
  verify_docker_gpu
  setup_python_env
  download_model_if_needed
  validate_environment
  summary
}

main "$@"
