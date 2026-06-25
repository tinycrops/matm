#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_DEV="${INSTALL_DEV:-0}"
SETUP_ALFWORLD="${SETUP_ALFWORLD:-1}"
STAGE_HF_TRACES="${STAGE_HF_TRACES:-1}"
BUILD_SMOKE_INDEX="${BUILD_SMOKE_INDEX:-1}"
SMOKE_INDEX_ENTRIES="${SMOKE_INDEX_ENTRIES:-500}"
BUILD_VECTOR_INDEX="${BUILD_VECTOR_INDEX:-1}"
FORCE_INDEX="${FORCE_INDEX:-0}"

log() {
  printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Could not find PYTHON_BIN=$PYTHON_BIN. Set PYTHON_BIN=/path/to/python3 and rerun." >&2
  exit 1
fi

log "Creating local virtualenv at $VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Some legacy ALFWorld/TextWorld build steps shell out to a literal `python`.
# Put the venv first so those subprocesses resolve to this project's Python.
export PATH="$ROOT_DIR/$VENV_DIR/bin:$PATH"
PY="$ROOT_DIR/$VENV_DIR/bin/python"

log "Upgrading venv packaging tools"
"$PY" -m pip install --upgrade pip setuptools wheel

log "Installing base dependencies"
"$PY" -m pip install -r requirements-base.txt

log "Installing ALFWorld dependencies"
"$PY" -m pip install -r requirements-alfworld.txt

if [[ "$INSTALL_DEV" == "1" ]]; then
  log "Installing development dependencies"
  "$PY" -m pip install -r requirements-dev.txt
fi

if [[ ! -f .env ]]; then
  log "Creating .env from .env.example"
  cp .env.example .env
  echo "Created .env. Add OPENAI_API_KEY before running live model evals."
else
  log ".env already exists"
fi

if ! grep -qE '^OPENAI_API_KEY=.+[^[:space:]]' .env 2>/dev/null; then
  echo "Warning: .env does not appear to contain OPENAI_API_KEY; live evals will fail until it is set."
fi

if [[ "$SETUP_ALFWORLD" == "1" ]]; then
  log "Downloading/verifying ALFWorld game data"
  "$PY" setup_environments.py --worlds alfworld --download-only
fi

if [[ "$STAGE_HF_TRACES" == "1" ]]; then
  log "Staging MATM HuggingFace ALFWorld prepopulation traces"
  "$PY" traj_retrieval/preprocess/alfworld/stage_hf_prepopulation_train.py
fi

INDEX_DIR="environments/train_only_lancedb/alfworld/lancedb_indices"
if [[ "$BUILD_SMOKE_INDEX" == "1" ]]; then
  if [[ "$FORCE_INDEX" == "1" || ! -d "$INDEX_DIR" ]]; then
    log "Building ALFWorld LanceDB smoke index ($SMOKE_INDEX_ENTRIES entries)"
    "$PY" traj_retrieval/preprocess/alfworld/create_seq_to_seq_indices_train_only.py \
      --force \
      --test-entries "$SMOKE_INDEX_ENTRIES"

    if [[ "$BUILD_VECTOR_INDEX" == "1" ]]; then
      log "Building vector index for smoke LanceDB table"
      "$PY" traj_retrieval/preprocess/alfworld/create_lancedb_index_train_only.py
    fi
  else
    log "LanceDB index already exists at $INDEX_DIR; set FORCE_INDEX=1 to rebuild"
  fi
else
  log "Skipping smoke index build"
fi

log "Running import/CLI smoke checks"
"$PY" -m py_compile \
  traj_retrieval/run_evaluation.py \
  traj_retrieval/run_webarena.py \
  traj_retrieval/preprocess/alfworld/stage_hf_prepopulation_train.py \
  traj_retrieval/preprocess/alfworld/create_seq_to_seq_indices_train_only.py
"$PY" -m traj_retrieval.run_evaluation --help >/dev/null

cat <<'EOF'

Setup complete.

Activate the environment:
  source .venv/bin/activate

Run a one-episode OpenAI smoke eval:
  .venv/bin/python -m traj_retrieval.run_evaluation \
    --environment-name alfworld \
    --strategy t0 \
    --start-idx 0 \
    --end-idx 1 \
    --max-steps 1 \
    --evaluation-run-id smoke_openai_gpt54mini_t0

Useful setup toggles:
  FORCE_INDEX=1 ./setup.sh                 # rebuild the local smoke index
  SMOKE_INDEX_ENTRIES=2000 ./setup.sh      # larger local index
  BUILD_SMOKE_INDEX=0 ./setup.sh           # deps/data/traces only
  INSTALL_DEV=1 ./setup.sh                 # include dev dependencies

For live evals, ensure .env contains:
  OPENAI_API_KEY=...
EOF
