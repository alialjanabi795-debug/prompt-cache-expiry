#!/bin/bash
# Case E: weights in anonymous memory (--no-mmap, as with --mlock or no page cache to reclaim).
# Fill ~16k context, then a 1.2 GB pytest with limit = U_now + 0.6 GiB, no swap.
# Also logs the q8_0 KV allocation for the KV table cross-check.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)  # repository root
CG=/sys/fs/cgroup/memory/dev
cd $R
echo 15032385536 > $CG/memory.memsw.limit_in_bytes; echo 15032385536 > $CG/memory.limit_in_bytes
pkill -x llama-server; sleep 3
# q8_0 KV allocation check (load, read log, stop)
setsid nohup ./scripts/calib/start_server.sh dev results/calib/server_q8probe.log -c 65536 -ctk q8_0 -ctv q8_0 --no-repack -lv 4 >/dev/null 2>&1 </dev/null &
until grep -q "listening" results/calib/server_q8probe.log 2>/dev/null; do sleep 2; done
grep -E "KV buffer size|llama_kv_cache: size" results/calib/server_q8probe.log
pkill -x llama-server; sleep 3; sync; echo 3 > /proc/sys/vm/drop_caches
setsid nohup ./scripts/calib/start_server.sh dev results/calib/server_nommap.log -c 65536 -ctk f16 -ctv f16 --no-repack -lm none >/dev/null 2>&1 </dev/null &
until grep -q "listening" results/calib/server_nommap.log 2>/dev/null; do sleep 2; done
grep -E "^(rss|cache) " $CG/memory.stat
TID=$(head -1 results/calib/replay_main.jsonl | python3 -c "import sys,json; print(json.load(sys.stdin)['traj_id'])")
python3 -I scripts/calib/agent_replay.py --target-ctx 16384 --cgroup dev --tag nommap --traj-id $TID | tail -2
U=$(python3 -c "print(round($(cat $CG/memory.usage_in_bytes)/2**30, 3))"); L=$(python3 -c "print($U + 0.6)")
echo "case E: U_now=$U GiB limit=$L GiB"
python3 -I scripts/calib/mem_pressure.py --tag E_anon_weights_tight --limit-gib $L --swap-gib 0 --prompt-file results/calib/last_prompt_nommap.txt
echo 15032385536 > $CG/memory.memsw.limit_in_bytes; echo 15032385536 > $CG/memory.limit_in_bytes
dmesg 2>/dev/null | grep -iE "oom|killed process" | tail -5
echo CASE_E_DONE
