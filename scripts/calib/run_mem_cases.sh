#!/bin/bash
# Run the memory-pressure cases A-D against the running (mmap-weights) llama-server whose KV holds
# the long prompt. U = device-cgroup usage measured now (weights page cache + anon).
#   A ample        limit = U + 2.5 GiB, no swap, pytest 1.2 GB
#   B tight        limit = U + 0.6 GiB, no swap, pytest 1.2 GB
#   C tight+swap   limit = U + 0.6 GiB, 4 GiB swap, pytest 1.2 GB
#   D 8GB-class    limit = 6.5 GiB, no swap, no pytest (idle device that cannot hold the stack)
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)  # repository root
CG=/sys/fs/cgroup/memory/dev
PROMPT=${1:-$R/results/calib/last_prompt_main.txt}
cd $R
reset_limits() { echo 15032385536 > $CG/memory.memsw.limit_in_bytes; echo 15032385536 > $CG/memory.limit_in_bytes; }
U=$(( $(cat $CG/memory.usage_in_bytes) ))
UG=$(python3 -c "print(round($U/2**30, 3))")
echo "U = $UG GiB"
reset_limits
python3 -I scripts/calib/mem_pressure.py --tag A_ample --limit-gib $(python3 -c "print($UG+2.5)") --swap-gib 0 --prompt-file $PROMPT
reset_limits
python3 -I scripts/calib/mem_pressure.py --tag B_tight_noswap --limit-gib $(python3 -c "print($UG+0.6)") --swap-gib 0 --prompt-file $PROMPT
reset_limits
if ! grep -q calib_swap /proc/swaps; then
  fallocate -l 4G ${SWAPFILE:-/var/tmp/calib_swap} && chmod 600 ${SWAPFILE:-/var/tmp/calib_swap} && mkswap ${SWAPFILE:-/var/tmp/calib_swap} >/dev/null && swapon ${SWAPFILE:-/var/tmp/calib_swap}
fi
python3 -I scripts/calib/mem_pressure.py --tag C_tight_swap --limit-gib $(python3 -c "print($UG+0.6)") --swap-gib 4 --prompt-file $PROMPT
reset_limits
swapoff ${SWAPFILE:-/var/tmp/calib_swap} 2>/dev/null; rm -f ${SWAPFILE:-/var/tmp/calib_swap}
python3 -I scripts/calib/mem_pressure.py --tag D_8gb_idle --limit-gib 6.5 --swap-gib 0 --no-pytest --prompt-file $PROMPT
reset_limits
echo ALL_CASES_DONE
