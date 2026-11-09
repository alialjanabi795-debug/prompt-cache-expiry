"""Run a command and report wall time and peak RSS of its whole process tree (polling /proc)."""

from __future__ import annotations

import json
import subprocess
import sys
import time


def tree_rss_kb(pid: int) -> int:
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        try:
            with open(f"/proc/{p}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
            with open(f"/proc/{p}/task/{p}/children") as f:
                stack.extend(int(c) for c in f.read().split())
        except (FileNotFoundError, ProcessLookupError):
            pass
    return total


if __name__ == "__main__":
    cwd, cmd = sys.argv[1], sys.argv[2:]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = 0
    while proc.poll() is None:
        peak = max(peak, tree_rss_kb(proc.pid))
        time.sleep(0.05)
    print(
        json.dumps(
            {
                "cmd": " ".join(cmd),
                "cwd": cwd,
                "rc": proc.returncode,
                "wall_s": round(time.time() - t0, 2),
                "peak_rss_mb": round(peak / 1024, 1),
            }
        )
    )
