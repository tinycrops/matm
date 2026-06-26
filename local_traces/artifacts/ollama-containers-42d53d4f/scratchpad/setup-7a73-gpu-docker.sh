#!/usr/bin/env bash
#
# setup-7a73-gpu-docker.sh
# One-time setup to let Docker containers on ath-ms-7a73 (GTX 1060, Pascal sm_61)
# access the GPU. After this runs once, `ath` can audition containerized inference
# WITHOUT sudo, exactly like ath-ms-7a72.
#
# Run on 7a73:   bash setup-7a73-gpu-docker.sh
# It will prompt for your sudo password once.
#
set -euo pipefail

echo "==> 7a73 GPU-in-Docker setup (nvidia-container-toolkit)"

if [[ "$(hostname)" != *7A73* && "$(hostname)" != *7a73* ]]; then
  echo "!! This is meant for ath-ms-7a73 (got: $(hostname)). Continue anyway? Ctrl-C to abort."
  read -r _
fi

# 0. Sanity: GPU + docker present
command -v nvidia-smi >/dev/null || { echo "!! nvidia-smi missing — install the NVIDIA driver first"; exit 1; }
command -v docker     >/dev/null || { echo "!! docker missing"; exit 1; }
echo "--> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Already configured? Bail out early, idempotent.
if docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
  echo "--> nvidia runtime already present in docker. Skipping install, jumping to verify."
else
  # 1. Add the NVIDIA container-toolkit apt repo (signed)
  echo "--> Adding nvidia-container-toolkit apt repository"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

  # 2. Install
  echo "--> Installing nvidia-container-toolkit"
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit

  # 3. Wire it into the docker daemon and restart
  echo "--> Configuring docker runtime + restarting daemon"
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

# 4. Verify GPU passthrough actually works from inside a container
echo "--> Verifying GPU passthrough inside a container..."
if docker image inspect nvidia/cuda:12.2.0-base-ubuntu22.04 >/dev/null 2>&1; then
  IMG=nvidia/cuda:12.2.0-base-ubuntu22.04
else
  IMG=nvidia/cuda:12.2.0-base-ubuntu22.04
  docker pull "$IMG"
fi
if docker run --rm --gpus all "$IMG" nvidia-smi --query-gpu=name --format=csv,noheader; then
  echo
  echo "==> SUCCESS. Containers on 7a73 can now use the GPU (sudo-free for 'ath')."
  echo "    Tell Claude it can proceed with containerized inference on 7a73."
else
  echo "!! GPU container test failed — check 'docker info | grep -i runtime' and dmesg."
  exit 1
fi
