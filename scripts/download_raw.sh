#!/bin/sh
# Download the public raw inputs into data/raw/ and check their SHA-256 hashes.
#   SWE-rebench OpenHands trajectories (CC BY 4.0), SWE-smith trajectories shard (MIT),
#   Qwen3.5-9B tokenizer.json (Apache-2.0).
set -eu
cd "$(dirname "$0")/.."
HF=https://huggingface.co
get() {  # url target sha256
  mkdir -p "$(dirname "$2")"
  [ -f "$2" ] || curl -fL --retry 3 -o "$2" "$1"
  echo "$3  $2" | sha256sum -c -
}
get $HF/datasets/nebius/SWE-rebench-openhands-trajectories/resolve/main/trajectories.parquet \
  data/raw/openhands/trajectories.parquet 14048dd1fcd22ce094b6e85f8a38f223a9ef1327031aaaad052804870212efa1
get $HF/datasets/SWE-bench/SWE-smith-trajectories/resolve/main/data/tool-00004-of-00008.parquet \
  data/raw/swesmith/tool-00004-of-00008.parquet e41ffbd3f9d1571917ac5b9f4d0108ded4799d9776f597a054203467135a5715
get $HF/Qwen/Qwen3.5-9B/resolve/main/tokenizer.json \
  data/raw/tokenizer/tokenizer.json 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
