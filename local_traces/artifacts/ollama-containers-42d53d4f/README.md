# Ollama Containers Trace Artifacts

Durable artifacts mirrored from the lil-blue Claude session:

- Session: `42d53d4f-1922-4e19-934b-5f20cb9bcdab`
- Theme: containerized, sudo-free VibeThinker inference on `ath-ms-7a72` and `ath-ms-7a73`
- Extracted trace: `../../extracted/ollama-containers-42d53d4f`

## Memory Notes

- `memory/vibethinker-containerized-inference.md`: endpoints, chosen engine, deployment/audition path.
- `memory/vibethinker-is-the-workload.md`: user feedback that VibeThinker is the real workload and must be benchmarked directly.
- `memory/ollama-pascal-cuda-backend.md`: Pascal CUDA backend behavior and llama.cpp vs ollama result.

## Scratchpad Scripts

- `scratchpad/deploy-vibethinker.sh`: deploys the llama.cpp CUDA VibeThinker server.
- `scratchpad/setup-7a73-gpu-docker.sh`: one-time GPU Docker setup helper for 7a73.
- `scratchpad/Modelfile.vibethinker`: ChatML/parameter reference for the ollama fallback model.
