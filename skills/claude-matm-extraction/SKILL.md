---
name: claude-matm-extraction
description: Extract reproducible Multi-Agent Transactive Memory style datasets from raw Claude conversation JSONL logs in `~/.claude/projects/**/*.jsonl`. Use when the user asks to inspect Claude traces individually, convert agent conversations into action-observation trajectories, build MATM/Multi-Agent Transactive Memory datasets, chunk Claude sessions for retrieval, or preserve reusable procedural knowledge from prior Claude work.
---

# Claude MATM Extraction

## Workflow

1. Resolve the source conversation.
   - Prefer a canonical `~/.claude/projects/**/*.jsonl` transcript over distilled memory files.
   - Use `~/.claude/history.jsonl` or the `claude-memory-retrieval` skill only to locate candidate sessions.
   - Inspect the raw JSONL directly before making claims about the trajectory.

2. Run the bundled extractor.

```bash
python3 /home/ath/.codex/skills/claude-matm-extraction/scripts/extract_claude_matm.py \
  /path/to/source-session.jsonl \
  /path/to/output-dir
```

3. Review the output in this order.
   - `metadata.json`: counts and basic sanity checks.
   - `README.md`: reconstructed phases, feedback, and representative actions.
   - `action_observations.jsonl`: paired tool calls and tool results.
   - `ao_chunks_l5.jsonl`: preferred MATM retrieval units.
   - `steps.jsonl` and `chunks_l5.jsonl`: lower-level event stream if deeper reconstruction is needed.

4. Build or refresh the local MATM retrieval index.

```bash
python3 /home/ath/.codex/skills/claude-matm-extraction/scripts/build_matm_index.py \
  --root /home/ath/matm_claude_traces \
  --out /home/ath/matm_claude_traces/index
```

5. Smoke-test retrieval against the new index before reporting success.

```bash
python3 /home/ath/.codex/skills/claude-matm-extraction/scripts/matm_search.py \
  --index /home/ath/matm_claude_traces/index \
  -k 5 \
  "task description plus current agent state"
```

6. Report what was produced.
   - Include the source log path, session id, step count, action-observation count, and output directory.
   - Include the index directory, indexed chunk count, vector dimensionality, and one retrieval smoke-test result.
   - Mention any observed failures or recovery patterns in the trace.
   - Treat the dataset as procedural memory, not as proof of current project state unless durable files are verified separately.

## Extraction Rules

- Ignore hidden Claude `thinking` and signature fields.
- Treat assistant text and tool calls as actions.
- Treat tool results and user feedback as observations.
- Pair each tool call with its matching tool result by `tool_use_id`.
- Use rolling `l=5` windows for MATM-style chunks: recent state/history is the retrieval key, and the next procedural segment is the value.
- Embed `ao_chunks_l5.jsonl` keys for retrieval; keep values as the procedural payload to inject or inspect after retrieval.
- The default pilot index uses local TF-IDF + SVD dense vectors for reproducibility. Treat it as a swappable first-stage retriever, not as the final MATM retriever.
- Do not open credential/config files while searching unless the user explicitly asks for security/audit work.

## Output Schema

Read `references/output-schema.md` when you need exact field meanings, downstream indexing guidance, or a quick `jq` inspection command.

## Retrieval Index

The bundled index scripts create and query the first-stage MATM memory:

- `scripts/build_matm_index.py`: discovers `*/ao_chunks_l5.jsonl`, writes `index/chunks.jsonl`, `index/vectors.npy`, `index/tfidf.joblib`, `index/svd.joblib`, and `index/metadata.json`.
- `scripts/matm_search.py`: embeds a task/state query and returns top matching trajectory chunks.
- Convenience wrappers may exist at `~/.local/bin/matm_index` and `~/.local/bin/matm_search`.

## Next Layers

After extraction and first-stage indexing:

- Add marginal-utility labels only after replaying or reusing chunks on downstream tasks.
- Build rerankers from producer metadata, consumer metadata, retrieval scores, query features, trajectory features, and query-trajectory interaction features, following the MATM paper.
