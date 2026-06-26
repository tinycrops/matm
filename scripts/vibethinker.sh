#!/usr/bin/env bash
set -euo pipefail

MODEL_REF="${MODEL_REF:-hf.co/prithivMLmods/VibeThinker-3B-GGUF:Q4_K_M}"
BENCH_PROMPT="${BENCH_PROMPT:-Solve 17 * 23 and reply with the integer only.}"
HF_PAPERS_VENV="${HF_PAPERS_VENV:-$HOME/.cache/matm-hf-papers-venv}"
HF_PAPERS_PYPI_VERSION="${HF_PAPERS_PYPI_VERSION:-1.21.0}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") chat [prompt...]
  $(basename "$0") paper PAPER_ID QUESTION...
  $(basename "$0") bench
  $(basename "$0") info

Environment overrides:
  MODEL_REF=hf.co/prithivMLmods/VibeThinker-3B-GGUF:Q6_K
  HF_PAPERS_VENV=$HOME/.cache/matm-hf-papers-venv
  HF_PAPERS_PYPI_VERSION=1.21.0
  BENCH_PROMPT=...

The default model matches the trace-backed fit for this 6 GB GTX 1060:
Q4_K_M. Use Q6_K if you want a little more quality and do not mind the
larger download.
EOF
}

need_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama is not installed or not on PATH" >&2
    exit 1
  fi
}

need_hf_papers() {
  if command -v hf >/dev/null 2>&1 && hf papers --help >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is not installed or not on PATH" >&2
    exit 1
  fi

  if [[ ! -x "$HF_PAPERS_VENV/bin/hf" ]]; then
    python3 -m venv "$HF_PAPERS_VENV"
    "$HF_PAPERS_VENV/bin/python" -m pip install --upgrade pip >/dev/null
    "$HF_PAPERS_VENV/bin/python" -m pip install "huggingface_hub==$HF_PAPERS_PYPI_VERSION" >/dev/null
  fi

  export PATH="$HF_PAPERS_VENV/bin:$PATH"
  if ! hf papers --help >/dev/null 2>&1; then
    echo "hf papers is unavailable even after bootstrapping $HF_PAPERS_VENV" >&2
    exit 1
  fi
}

cmd="${1:-chat}"
shift || true

case "$cmd" in
  chat)
    need_ollama
    if [[ $# -gt 0 ]]; then
      exec ollama run "$MODEL_REF" "$*"
    fi
    exec ollama run "$MODEL_REF"
    ;;
  paper)
    need_hf_papers
    need_ollama
    if [[ $# -lt 2 ]]; then
      echo "Usage: $(basename "$0") paper PAPER_ID QUESTION..." >&2
      exit 1
    fi

    paper_id="$1"
    shift

    paper_markdown="$(mktemp)"
    prompt_file="$(mktemp)"
    cleanup() {
      rm -f "$paper_markdown" "$prompt_file"
    }
    trap cleanup EXIT

    hf papers read "$paper_id" >"$paper_markdown"
    {
      printf 'You are answering a question about a Hugging Face paper.\n'
      printf 'Use the paper context below as the primary source.\n'
      printf 'Paper ID: %s\n\n' "$paper_id"
      printf 'Paper context:\n'
      cat "$paper_markdown"
      printf '\nQuestion: %s\n' "$*"
    } >"$prompt_file"

    exec ollama run "$MODEL_REF" <"$prompt_file"
    ;;
  bench)
    need_ollama
    exec ollama run --verbose "$MODEL_REF" "$BENCH_PROMPT"
    ;;
  info)
    cat <<EOF
Model ref: $MODEL_REF
Benchmark prompt: $BENCH_PROMPT
Recommended default quant: Q4_K_M
Higher-quality option: Q6_K
HF papers venv: $HF_PAPERS_VENV
HF papers CLI version: $HF_PAPERS_PYPI_VERSION
EOF
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
