---
name: vibethinker-is-the-workload
description: "VibeThinker-3B is the actual reason for the cluster inference work — treat it as the target, not a disposable benchmark prop"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 42d53d4f-1922-4e19-934b-5f20cb9bcdab
---

When working on cluster inference, VibeThinker-3B (https://huggingface.co/WeiboAI/VibeThinker-3B) is the real workload — "it's the only reason we are doing this." The user pushed back hard when I kept swapping it out for qwen2.5 as a benchmark vehicle.

**Why:** The infra work (containers, fastest engine) exists to serve VibeThinker well; substituting another model defeats the purpose and reads as missing the point.

**How to apply:** Benchmark and deploy *with VibeThinker itself*. It's a Qwen2.5-Coder-3B reasoning fine-tune: needs the **ChatML template** + official sampling (temp 1.0, top_p 0.95, top_k -1) and a large token budget; it emits integrated `<think>…</think>` reasoning. A bare `{{ .Prompt }}` template makes it stop after ~2 tokens — that's a config bug, not a broken model. See [[vibethinker-containerized-inference]].
