# Agent Observations

Canonical form: [agent_observations.jsonl](./agent_observations.jsonl)

This is a downstream observation layer for notes contributed by an agent after
reading the trace corpus. It is intentionally separate from the raw trace
archive so the accumulated source material stays neutral while higher-level
takeaways can evolve.

## VibeThinker-3B Standup Note

Observed from the VibeThinker-3B trace: the simplest local path on this
machine is the GGUF release served through Ollama, with `Q4_K_M` as the
default fit and `Q6_K` as the next quality step if more download size is
acceptable.

Recommended one-liners:

```bash
/home/ath/matm/scripts/vibethinker.sh chat
/home/ath/matm/scripts/vibethinker.sh bench
```

The trace-backed model reference is:

```bash
hf.co/prithivMLmods/VibeThinker-3B-GGUF:Q4_K_M
```

## VibeCluster Containerized Inference Note

Observed from the containerization trace: VibeThinker-3B is the actual
workload for the cluster, so benchmark and deploy with VibeThinker itself.
The durable setup serves VibeThinker through sudo-free `llama.cpp` CUDA
containers on both inference nodes:

- `http://192.168.1.8:8080` for `ath-ms-7a72`
- `http://192.168.1.15:8080` for `ath-ms-7a73`

Use OpenAI-compatible `/v1/chat/completions` with `model=vibethinker`.
The trace-backed deployment scripts and memory notes are mirrored under:

```bash
/home/ath/matm/local_traces/artifacts/ollama-containers-42d53d4f
```

Key result: `llama.cpp` server-cuda beat containerized ollama on the P4000
for VibeThinker Q4, about `42.8 tok/s` versus `36.3 tok/s`.
