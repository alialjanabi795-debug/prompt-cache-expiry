#!/bin/bash
# Re-run the tight cases with U (device-cgroup usage) measured immediately before each case.
#   B  tight        limit = U_now + 0.6 GiB, no swap, pytest 1.2 GB (needs 0.6 GiB more than free)
#   B2 very tight   limit = U_now + 0.2 GiB, no swap, pytest 1.2 GB
#   C  tight+swap   limit = U_now + 0.6 GiB, 4 GiB swap, pytest 1.2 GB
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)  # repository root
CG=/sys/fs/cgroup/memory/dev
PROMPT=${1:-$R/results/calib/last_prompt_main.txt}
cd $R
reset_limits() { echo 15032385536 > $CG/memory.memsw.limit_in_bytes; echo 15032385536 > $CG/memory.limit_in_bytes; }
u_now() { python3 -c "print(round($(cat $CG/memory.usage_in_bytes)/2**30, 3))"; }
run_case() {  # tag margin swap
  reset_limits
  python3 -I -c "import sys; sys.path.insert(0,'scripts/calib'); from llama_client import completion; completion(open('$PROMPT').read(), 4)"
  U=$(u_now); L=$(python3 -c "print($U + $2)")
  echo "case $1: U_now=$U GiB limit=$L GiB swap=$3 GiB"
  python3 -I scripts/calib/mem_pressure.py --tag $1 --limit-gib $L --swap-gib $3 --prompt-file $PROMPT
}
run_case B_tight_noswap_v2 0.6 0
run_case B2_verytight_noswap 0.2 0
if ! grep -q calib_swap /proc/swaps; then
  fallocate -l 4G ${SWAPFILE:-/var/tmp/calib_swap} && chmod 600 ${SWAPFILE:-/var/tmp/calib_swap} && mkswap ${SWAPFILE:-/var/tmp/calib_swap} >/dev/null && swapon ${SWAPFILE:-/var/tmp/calib_swap}
fi
run_case C_tight_swap_v2 0.6 4
reset_limits
swapoff ${SWAPFILE:-/var/tmp/calib_swap} 2>/dev/null; rm -f ${SWAPFILE:-/var/tmp/calib_swap}
echo ALL_CASES2_DONE
