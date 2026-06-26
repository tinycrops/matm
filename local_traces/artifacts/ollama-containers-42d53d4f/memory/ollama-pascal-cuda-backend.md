---
name: ollama-pascal-cuda-backend
description: ollama:latest container ships cuda_v13 (no Pascal sm_61) and falls back to cuda_v12; llama.cpp is faster on these Pascal GPUs
metadata: 
  node_type: memory
  type: reference
  originSessionId: 42d53d4f-1922-4e19-934b-5f20cb9bcdab
---

On the Pascal GPUs in this cluster (Quadro P4000 / GTX 1060, both compute capability 6.1 / sm_61):

- The `ollama/ollama:latest` container's default `cuda_v13` runner is compiled only for sm_75+ and **logs "skipping CUDA device — compute capability not in compiled architectures" cc=610**. It then falls through to a bundled `cuda_v12` runner that *does* include sm_61, so it still runs on GPU — just via the fallback path.
- `ghcr.io/ggml-org/llama.cpp:server-cuda` is compiled with Pascal archs (ARCHS includes 610) and ran faster: **VibeThinker-3B Q4 decode ≈ 42.8 tok/s (llama.cpp) vs ≈ 36.3 tok/s (ollama)** on the P4000, exclusive GPU.
- Gotcha when benchmarking: with multiple model servers sharing an 8GB GPU, the loser gets squeezed to CPU (~3.4 tok/s) — bench one engine at a time and confirm GPU residency with `nvidia-smi --query-compute-apps`.

Consistent with the lil-blue PufferLib note that newer CUDA toolchains drop Pascal support. See [[vibethinker-containerized-inference]].
