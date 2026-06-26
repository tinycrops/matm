#!/usr/bin/env bash
# Deploy VibeThinker-3B as a sudo-free, GPU-accelerated llama.cpp container.
# Fastest engine on these Pascal GPUs (beat ollama 42.8 vs 36.3 tok/s on the P4000).
# Usage: bash deploy-vibethinker.sh [PORT]   (default port 8080)
set -euo pipefail
PORT="${1:-8080}"
IMG="ghcr.io/ggml-org/llama.cpp:server-cuda"
GGUF="$HOME/models/VibeThinker-3B.Q4_K_M.gguf"
BLOB="/usr/share/ollama/.ollama/models/blobs/sha256-9782b918cc220fb81d59e21be3e45c3ae027e5d86fb56ce7c6d537a347c80d79"

echo "==> Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# 1. Stable, ollama-independent copy of the GGUF weights.
mkdir -p "$HOME/models"
if [[ ! -f "$GGUF" ]]; then
  echo "==> Copying VibeThinker GGUF out of ollama blob store -> $GGUF"
  cp "$BLOB" "$GGUF"
fi
echo "==> Weights: $GGUF ($(du -h "$GGUF" | cut -f1))"

# 2. Image (no-op if already present)
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "==> Pulling $IMG"; docker pull "$IMG"; }

# 3. Clean up any prior / ad-hoc containers, then run the persistent one.
docker rm -f vibethinker bench-ollama bench-llamacpp >/dev/null 2>&1 || true
echo "==> Starting 'vibethinker' container on :$PORT"
docker run -d --name vibethinker --restart unless-stopped --gpus all -p "${PORT}:8080" \
  -v "$HOME/models":/models:ro \
  "$IMG" \
  -m /models/VibeThinker-3B.Q4_K_M.gguf \
  -ngl 99 -c 16384 -fa on --jinja \
  --temp 1.0 --top-p 0.95 --top-k 0 \
  --host 0.0.0.0 --port 8080 -a vibethinker >/dev/null

# 4. Wait for health
echo -n "==> Waiting for model load"
for i in $(seq 1 40); do
  if curl -s "http://localhost:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; then echo " ... READY"; ok=1; break; fi
  echo -n "."; sleep 3
done
[[ "${ok:-}" == 1 ]] || { echo " FAILED"; docker logs --tail 20 vibethinker; exit 1; }

# 5. Smoke test: real reasoning + speed
echo "==> Smoke test (OpenAI /v1/chat/completions):"
curl -s --max-time 120 "http://localhost:${PORT}/v1/chat/completions" \
  -d '{"model":"vibethinker","messages":[{"role":"user","content":"What is 17*23? Reason step by step."}],"max_tokens":400}' \
| python3 -c '
import sys,json
d=json.load(sys.stdin)
c=d["choices"][0]["message"]["content"]; t=d.get("timings",{})
print("   reasoning present:", "<think>" in c or "boxed" in c)
print("   decode: %.1f tok/s (%d tok)" % (t.get("predicted_per_second",0), t.get("predicted_n",0)))
print("   answer tail:", repr(c[-80:]))
'
echo "==> DONE. VibeThinker @ http://$(hostname -I | awk "{print \$1}"):${PORT}  (OpenAI API: /v1/chat/completions, model=vibethinker)"
