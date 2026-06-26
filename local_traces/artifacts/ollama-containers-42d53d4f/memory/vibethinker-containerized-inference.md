---
name: vibethinker-containerized-inference
description: "VibeThinker-3B served sudo-free via llama.cpp containers on the two inference nodes; endpoints, engine choice, how to audition"
metadata: 
  node_type: memory
  type: project
  originSessionId: 42d53d4f-1922-4e19-934b-5f20cb9bcdab
---

VibeThinker-3B (the cluster's primary reasoning workload) runs as a sudo-free, GPU-accelerated **llama.cpp** container on both inference nodes. Set up 2026-06-25.

- Endpoints (OpenAI-compatible, `model=vibethinker`): `http://192.168.1.8:8080` (ath-ms-7a72 / Quadro P4000) and `http://192.168.1.15:8080` (ath-ms-7a73 / GTX 1060). Paths: `/v1/chat/completions`, `/health`.
- Container `vibethinker`, image `ghcr.io/ggml-org/llama.cpp:server-cuda`, `--restart unless-stopped`, `--gpus all`, flags `-ngl 99 -c 16384 -fa on --jinja --temp 1.0 --top-p 0.95 --top-k 0`.
- Weights live at `~/models/VibeThinker-3B.Q4_K_M.gguf` on each node (copied out of the ollama blob store so it's ollama-independent).
- To audition perf changes sudo-free: edit/run `~/deploy-vibethinker.sh [PORT]` on the node (it pulls image, copies GGUF, restarts the container, smoke-tests). `ath` is in the `docker` group on both nodes.
- Engine was chosen empirically: llama.cpp beat containerized ollama 42.8 vs 36.3 tok/s on the P4000 — see [[ollama-pascal-cuda-backend]].
- Native ollama (port 11434) still runs as a systemd fallback; its `vibethinker-q4km` model was re-templated to match (ChatML).
- 7a73 required a one-time `nvidia-container-toolkit` install (human-run, since no passwordless sudo); 7a72 already had the nvidia docker runtime.
