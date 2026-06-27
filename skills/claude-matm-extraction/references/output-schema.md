# Output Schema

## Files

- `metadata.json`: source path, session id, record count, extracted step count, chunk count, action-observation count, tool counts, and event counts.
- `README.md`: human-readable audit summary generated from the trace.
- `steps.jsonl`: normalized visible event stream. One row per user feedback, assistant text, tool call, or tool result.
- `chunks_l5.jsonl`: rolling MATM chunks over all normalized steps.
- `action_observations.jsonl`: paired tool-call/tool-result rows. Prefer this file for procedural analysis.
- `ao_chunks_l5.jsonl`: rolling MATM chunks over action-observation pairs. Prefer this file for retrieval experiments.
- `index/`: optional retrieval index built from all discovered `ao_chunks_l5.jsonl` files.

## `steps.jsonl`

Important fields:

- `step_index`: zero-based normalized event index.
- `timestamp`, `cwd`, `uuid`, `parent_uuid`: source provenance.
- `event_type`: `assistant_text`, `tool_call`, `tool_result`, `user_feedback`, `user_command`, or `session_control`.
- `role`: `assistant`, `user`, or `environment`.
- `action`: visible assistant/user action text or tool invocation summary.
- `observation`: tool command payload or tool result text.
- `tool_use_id`, `tool_name`, `outcome`: populated for tool events when available.

## `action_observations.jsonl`

Important fields:

- `ao_index`: zero-based action-observation index.
- `source_step_index`: corresponding tool-call step in `steps.jsonl`.
- `action`: normalized tool call.
- `observation`: matching tool result.
- `outcome`: `ok`, `error`, or `missing_result`.
- `recent_context`: nearby assistant/user context before the action.

## Chunk Files

Each chunk row contains:

- `chunk_id`: stable id formed from session id and chunk index.
- `session_id`: Claude session id.
- `key`: rolling recent context for retrieval.
- `value`: subsequent procedural segment to inject or inspect.
- `metadata`: timestamp, cwd, event/tool type, and outcome.

## Quick Inspection

```bash
jq -r '.chunk_id + "\nKEY: " + .key + "\nVALUE: " + .value + "\n---"' ao_chunks_l5.jsonl | sed -n '1,120p'
```

Find chunks about a topic:

```bash
jq -r 'select(.key|test("label|keyboard|abstract|embedding"; "i")) | .chunk_id + "\n" + .key + "\n" + .value + "\n---"' ao_chunks_l5.jsonl
```

## Index Files

Build or refresh the local first-stage retrieval index:

```bash
python3 /home/ath/.codex/skills/claude-matm-extraction/scripts/build_matm_index.py \
  --root /home/ath/matm_claude_traces \
  --out /home/ath/matm_claude_traces/index
```

Important index files:

- `index/chunks.jsonl`: row-aligned chunk payloads and metadata.
- `index/vectors.npy`: normalized dense vectors for chunk keys.
- `index/tfidf.joblib`: fitted TF-IDF vectorizer.
- `index/svd.joblib`: fitted dense projection, when enough rows/features exist.
- `index/metadata.json`: build manifest, indexed chunk count, vector dimension, sessions, tools, and outcomes.

Search the index:

```bash
python3 /home/ath/.codex/skills/claude-matm-extraction/scripts/matm_search.py \
  --index /home/ath/matm_claude_traces/index \
  -k 5 \
  "task description plus recent agent state"
```
