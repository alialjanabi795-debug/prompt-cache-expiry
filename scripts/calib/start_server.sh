#!/bin/sh
# Start llama-server inside memory cgroup $1 (created if missing). Extra args are passed through.
# Usage: start_server.sh <cgroup> <logfile> [llama-server args...]
# LLAMA_SERVER: path to the llama-server binary (default: llama-server on PATH);
# MODEL_GGUF: Qwen3.5-9B Q4_K_M weights (default: data/raw/gguf/Qwen3.5-9B-Q4_K_M.gguf).
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CG=/sys/fs/cgroup/memory/$1; LOG=$2; shift 2
mkdir -p "$CG"
echo $$ > "$CG/cgroup.procs"
exec "${LLAMA_SERVER:-llama-server}" \
  -m "${MODEL_GGUF:-$ROOT/data/raw/gguf/Qwen3.5-9B-Q4_K_M.gguf}" \
  -np 1 -fa on -t 4 --port 8099 --no-webui -cram 0 "$@" > "$LOG" 2>&1
