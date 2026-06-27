# Gemma 4 Thinking Traces To VibeThinker LoRA Trace

Extracted from Claude session `e629efed-d36a-4b42-a149-35f421314e08`,
title `Find Gemma 4 31B thinking traces dataset`.

## Source And Artifacts

- Source transcript:
  `/home/ath/.claude/projects/-var-local-shared/e629efed-d36a-4b42-a149-35f421314e08.jsonl`
- MATM extraction:
  `/var/local/shared/matm/local_traces/extracted/gemma4-thinking-traces-e629efed`
- Project workspace:
  `/var/local/shared/ath-vibethinker-gemma4`
- Serving reference workspace:
  `/var/local/shared/gemma4-llama-dgx-spark`
- Preferred retrieval unit:
  `local_traces/extracted/gemma4-thinking-traces-e629efed/ao_chunks_l5.jsonl`

Extraction produced 397 normalized steps, 141 action-observation pairs, and
141 `l=5` action-observation chunks. The shared MATM index was rebuilt with
this trace included.

## Spoke Objective

Test whether VibeThinker can be finetuned to emit reasoning that is more
structurally and stylistically compatible with Gemma 4, so Gemma 4 can consume
that reasoning as a prefix and continue into a stronger final answer.

The practical hypothesis was not literal token-level speculative decoding.
VibeThinker and Gemma use different tokenizers, so the chosen mechanism was:

`VibeThinker produces Gemma-flavored thought text -> Gemma 4 consumes that text as thought-channel prefill -> Gemma 4 writes the final answer`

## Dataset Finding

The trace first disambiguated datasets from models. Most web hits were Gemma
student models distilled from other teachers, which were the opposite direction
for this task.

The useful public dataset found was:

- `MasonMac/CodeX-Thinking-Gemma-4-31B-IT`
- 138,683 coding examples
- top-level column: `messages`
- assistant turn includes `content` for the answer and `reasoning` for the
  Gemma-4-31B thinking trace
- useful for a proof of concept, but domain-biased toward code

The trace also records that no existing public VibeThinker finetune on Gemma 4
thinking traces was found.

## Workstation Rules Captured

The session contains an important procedural correction: the assistant had not
actually been conditioned on `~/CLAUDE.md` at the start. It explicitly read the
file before GPU/container work and then followed the relevant workstation rules:

- no host-level CUDA/library installs;
- use Docker containers for GPU work;
- do not touch `gary-backend-spark-*` production containers or network;
- check `nvidia-smi` before GPU work;
- keep GPU temperature guarded, later operationalized as a sweep cap below
  77 C and a stricter 74 C run gate;
- avoid production ports, especially host port 8080;
- remove only unused `ath-*` containers.

This makes the trace useful when a future task needs to do experimental model
work on the shared Gary workstation without colliding with production.

## Procedural Training Pattern

The trace built a complete experiment in stages:

- clean unused `ath-*` containers and inspect GPU state;
- identify a fast Gemma 4 31B serving path using QAT 4-bit GGUF plus MTP;
- avoid the existing repo's port 8080 because production owns it;
- serve Gemma 4 31B on a safe alternate port;
- convert Gemma reasoning traces into VibeThinker-compatible SFT data;
- build a pool, holdout, cluster split, and router;
- compare one diverse LoRA against six cluster-specialist LoRAs;
- validate training with a tiny proof-of-concept run before launching a sweep;
- repair OOM issues by moving to SDPA, gradient checkpointing, batch 2,
  accumulation 4, and `max_len 1024`;
- use serial LoRA training under a thermal gate.

The key training recipe that survived diagnostics was:

`sdpa + gradient checkpointing + batch 2 x accum 4 + max_len 1024`

The trace documents a real failure mode: eager attention or ineffective memory
settings caused absurd 72 GB OOM behavior for a 3B LoRA. The recovery was to
verify the active attention implementation inside the container and lock a lean
configuration before scaling.

## Handoff Eval Pattern

The eval design was selected as behavioral continuation plus judge:

`VibeThinker reasoning -> Gemma thought-channel prefill -> forced final channel -> Gemma judge`

The trace discovered that Gemma 4 does not use `<think>` tags for this path.
It uses a channel format with a thought channel, so the evaluator had to prefill
Gemma's thought channel and then force the final channel.

The session validated the handoff with a small raw-generation test, then built
`scripts/eval_handoff.py` and smoke-tested the base arm. A judge robustness bug
was found because Gemma spent too many tokens thinking before emitting a score;
the evaluator was then adjusted to give the judge enough budget.

## Runnable Artifacts

Durable files from the session include:

- `scripts/convert_codex_to_vibethinker.py`
- `scripts/build_pool.py`
- `scripts/cluster_split.py`
- `scripts/router.py`
- `train/train_lora.py`
- `scripts/eval_handoff.py`
- `run_sweep.sh`
- `data/pool.jsonl`
- `data/holdout.jsonl`
- `data/h1_diversity_train.jsonl`
- `data/h2/cluster_*_train.jsonl`
- `data/cluster_meta.json`
- `adapters/h1_diversity/`
- `results/eval_base.jsonl`
- `results/sweep_20260627_124349.log`

## Human Label

Reusable:

- Search for trace provenance before choosing training fuel; a model named
  Gemma is not necessarily a Gemma-generated trace source.
- For cross-tokenizer handoff, train style/prefix compatibility, not token-level
  speculative decoding.
- Treat code-only public thinking traces as a proof-of-concept fuel source and
  plan to generate math/STEM Gemma traces for the real run.
- On this shared host, always make the production-safety and thermal rules part
  of the experimental procedure, not an afterthought.
- Validate each pipeline stage with a tiny run before launching long training.
- Test the raw model chat/channel format before building an eval harness.

Do not reuse blindly:

- Do not assume `<think>` delimiters; inspect the target model's real template.
- Do not assume faster token generation means a better handoff. Measure the full
  continuation-and-judge behavior.
- Do not run the serving repo as-is if it binds production ports.
- Do not scale LoRA sweeps until the actual in-container attention
  implementation and memory behavior are verified.

## MATM Value

This is high-value procedural memory for tasks shaped like:

`find model-native reasoning traces -> convert to SFT data -> train small LoRAs -> serve target model safely on shared GPU -> evaluate prefix handoff behavior`

The trace is especially useful because it preserves the reasoning around
constraints: dataset provenance, cross-tokenizer mechanism choice, shared-host
safety, thermal gating, memory failures, and eval construction.
