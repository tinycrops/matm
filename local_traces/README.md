# Personal Trajectory Store Pilot

This directory is the first local pilot of a personal trajectory store built from
Claude/Codex work traces. The immediate goal is not to freeze the final schema;
it is to preserve real sessions in a form that can be searched, inspected, and
used as procedural memory while the schema becomes clearer.

## Current Sources

- Host: `lil-blue`
- First-pass source transcript: `~/.claude/projects/-home-ath/a8876efb-9c84-4bcd-9af5-dce8e2f61e83.jsonl`
- First-pass local copy: `claude/lil-blue/-home-ath/a8876efb-9c84-4bcd-9af5-dce8e2f61e83.jsonl`
- Second-pass source transcript: `~/.claude/projects/-home-ath/94967c56-e4f5-4267-804b-16cb3cd819ec.jsonl`
- Second-pass local copy: `claude/lil-blue/-home-ath/94967c56-e4f5-4267-804b-16cb3cd819ec.jsonl`
- Container source transcript: `~/.claude/projects/-home-ath-projects/42d53d4f-1922-4e19-934b-5f20cb9bcdab.jsonl`
- Container local copy: `claude/lil-blue/-home-ath-projects/42d53d4f-1922-4e19-934b-5f20cb9bcdab.jsonl`
- Theme: VibeThinker-3B local/containerized/distributed inference, a Qwen/Qwen-coder tool-caller duo, evaluation runs, LoRA/data thoughts, and recovery from orchestration bugs.

## Extracted Artifacts

The extracted store lives in:

- `extracted/vibe-stack-a8876efb/metadata.json`
- `extracted/vibe-stack-a8876efb/README.md`
- `extracted/vibe-stack-a8876efb/steps.jsonl`
- `extracted/vibe-stack-a8876efb/chunks_l5.jsonl`
- `extracted/vibe-stack-a8876efb/action_observations.jsonl`
- `extracted/vibe-stack-a8876efb/ao_chunks_l5.jsonl`
- `extracted/vibe-stack-94967c56/metadata.json`
- `extracted/vibe-stack-94967c56/README.md`
- `extracted/vibe-stack-94967c56/steps.jsonl`
- `extracted/vibe-stack-94967c56/chunks_l5.jsonl`
- `extracted/vibe-stack-94967c56/action_observations.jsonl`
- `extracted/vibe-stack-94967c56/ao_chunks_l5.jsonl`
- `extracted/ollama-containers-42d53d4f/metadata.json`
- `extracted/ollama-containers-42d53d4f/README.md`
- `extracted/ollama-containers-42d53d4f/steps.jsonl`
- `extracted/ollama-containers-42d53d4f/chunks_l5.jsonl`
- `extracted/ollama-containers-42d53d4f/action_observations.jsonl`
- `extracted/ollama-containers-42d53d4f/ao_chunks_l5.jsonl`

Durable project artifacts mirrored from the containerization session live in:

- `artifacts/ollama-containers-42d53d4f/memory/vibethinker-containerized-inference.md`
- `artifacts/ollama-containers-42d53d4f/memory/vibethinker-is-the-workload.md`
- `artifacts/ollama-containers-42d53d4f/memory/ollama-pascal-cuda-backend.md`
- `artifacts/ollama-containers-42d53d4f/scratchpad/deploy-vibethinker.sh`
- `artifacts/ollama-containers-42d53d4f/scratchpad/setup-7a73-gpu-docker.sh`
- `artifacts/ollama-containers-42d53d4f/scratchpad/Modelfile.vibethinker`

The most useful retrieval unit so far is `ao_chunks_l5.jsonl`: each row uses a
rolling recent action/observation history as the key and the next procedural
segment as the value. This matches the ALFWorld-style shape closely enough to
start experimenting, while preserving the richer messiness of real assistant
work.

## Index

The current first-stage retrieval index is:

- `index/chunks.jsonl`
- `index/vectors.npy`
- `index/tfidf.joblib`
- `index/svd.joblib`
- `index/metadata.json`

It was built with the local TF-IDF + SVD pilot indexer from the
`claude-matm-extraction` skill. This is intentionally simple and reproducible;
it should be treated as a swappable retrieval layer, not the final retriever.

Downstream, trace-specific observations belong in the index layer instead of
this archive. See `index/agent_observations.jsonl` for the first observer note
on the VibeThinker-3B standup path.

## What This Trace Teaches

- A personal trajectory is not just a task transcript. It contains user intent,
  assistant decisions, tool calls, tool outputs, bugs, fixes, status reports,
  and changing hypotheses.
- The valuable chunks are often recovery patterns: resume after a crash, parse
  partial results, patch tool-call formatting, re-run from saved state, and
  explain what changed.
- The VibeThinker session also contains training-data hints: thinking chains,
  self-correction behavior, tool-routing decisions, timeout failure modes, and
  cases where a smaller baseline fails but a thinking model succeeds.

## Open Schema Questions

- Should the store keep one canonical raw transcript plus derived views, or
  should durable project artifacts be mirrored alongside the transcript?
- Should trajectory rows be centered on tool actions, user feedback, assistant
  decisions, or completed task phases?
- Should "personal value" labels be explicit fields, such as reused, saved time,
  prevented bug, or worth training on?
- Should long thinking traces be stored verbatim, summarized, or split into a
  separate reasoning-signal layer?
- How should unfinished/live sessions be represented without pretending they are
  completed episodes?

## Rebuild Commands

```bash
python3 /home/ath/.codex/skills/claude-matm-extraction/scripts/extract_claude_matm.py \
  /home/ath/matm/local_traces/claude/lil-blue/-home-ath/a8876efb-9c84-4bcd-9af5-dce8e2f61e83.jsonl \
  /home/ath/matm/local_traces/extracted/vibe-stack-a8876efb

python3 /home/ath/.codex/skills/claude-matm-extraction/scripts/extract_claude_matm.py \
  /home/ath/matm/local_traces/claude/lil-blue/-home-ath/94967c56-e4f5-4267-804b-16cb3cd819ec.jsonl \
  /home/ath/matm/local_traces/extracted/vibe-stack-94967c56

python3 /home/ath/.codex/skills/claude-matm-extraction/scripts/extract_claude_matm.py \
  /home/ath/matm/local_traces/claude/lil-blue/-home-ath-projects/42d53d4f-1922-4e19-934b-5f20cb9bcdab.jsonl \
  /home/ath/matm/local_traces/extracted/ollama-containers-42d53d4f

python3 /home/ath/.codex/skills/claude-matm-extraction/scripts/build_matm_index.py \
  --root /home/ath/matm/local_traces/extracted \
  --out /home/ath/matm/local_traces/index

python3 /home/ath/.codex/skills/claude-matm-extraction/scripts/matm_search.py \
  --index /home/ath/matm/local_traces/index \
  -k 5 \
  "thinking model tool caller duo evaluation VibeThinker qwen resume bug"
```
