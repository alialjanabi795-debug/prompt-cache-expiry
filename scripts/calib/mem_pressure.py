"""Memory-pressure experiment: what happens to a running llama-server (Qwen3.5-9B, long context)
when a ~1.2 GB pytest run starts inside the same device-memory budget (cgroup v1)?

One case per invocation:
  --limit-gib L     memory.limit_in_bytes of the device cgroup (RAM for agent + LLM + tools)
  --swap-gib S      extra swap allowed (memory.memsw.limit = L + S); 0 = no swap
  --alloc-mb A      memory the synthetic pytest test allocates and touches (default 1200)
During the case a decode request runs concurrently on the cached long prompt; afterwards a probe
request checks whether the KV prefix is still cached (cache_n) and how fast decode is.
Everything is sampled every 0.5 s. Output: results/calib/mem_<tag>.json

Usage: python -I scripts/calib/mem_pressure.py --tag A_ample --limit-gib 10 --swap-gib 0 \
           --prompt-file results/calib/last_prompt.txt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_client import completion  # noqa: E402

CG = Path("/sys/fs/cgroup/memory")
TEST = """import os, time
def test_alloc():
    n = int(os.environ.get("ALLOC_MB", "1200")) * 1024 * 1024
    b = bytearray(n)
    for i in range(0, n, 4096):
        b[i] = 1
    time.sleep(float(os.environ.get("HOLD_S", "30")))
    assert len(b) == n
"""


def read_stat(cg: str) -> dict:
    d = CG / cg
    st = {
        k: int(v)
        for k, v in (ln.split() for ln in (d / "memory.stat").read_text().splitlines())
        if k in ("rss", "cache", "mapped_file", "swap", "pgmajfault")
    }
    oc = dict(ln.split() for ln in (d / "memory.oom_control").read_text().splitlines())
    st.update(
        usage=int((d / "memory.usage_in_bytes").read_text()),
        failcnt=int((d / "memory.failcnt").read_text()),
        under_oom=int(oc.get("under_oom", 0)),
        oom_kill=int(oc.get("oom_kill", 0)),
    )
    return st


def server_pid() -> int | None:
    out = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True).stdout
    return int(out.split()[0]) if out.strip() else None


def proc_mem(pid: int | None) -> dict:
    if pid is None:
        return {"alive": False}
    try:
        kv = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS", "VmSwap", "RssAnon", "RssFile")):
                k, v = line.split(":")
                kv[k] = int(v.split()[0]) * 1024
        return {"alive": True, **kv}
    except FileNotFoundError:
        return {"alive": False}


def set_limits(cg: str, limit: int, swap: int) -> None:
    d = CG / cg
    # memsw must always be >= limit: order the writes accordingly
    cur = int((d / "memory.limit_in_bytes").read_text())
    if limit + swap >= cur:
        (d / "memory.memsw.limit_in_bytes").write_text(str(limit + swap))
        (d / "memory.limit_in_bytes").write_text(str(limit))
    else:
        (d / "memory.limit_in_bytes").write_text(str(limit))
        (d / "memory.memsw.limit_in_bytes").write_text(str(limit + swap))


def probe(prompt: str, n: int = 16) -> dict:
    try:
        r = completion(prompt, n, timeout=3600)
        t = r["timings"]
        return {
            "ok": True,
            "cache_n": t.get("cache_n"),
            "prompt_n": t["prompt_n"],
            "prompt_ms": t["prompt_ms"],
            "predicted_n": t["predicted_n"],
            "tg_tps": t["predicted_n"] / max(t["predicted_ms"], 1e-9) * 1e3,
            "wall_s": r["wall_s"],
        }
    except Exception as e:  # server killed / connection refused
        return {"ok": False, "error": repr(e)[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cgroup", default="dev")
    ap.add_argument("--limit-gib", type=float, required=True)
    ap.add_argument("--swap-gib", type=float, default=0.0)
    ap.add_argument("--alloc-mb", type=int, default=1200)
    ap.add_argument("--hold-s", type=float, default=30.0)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--no-pytest", action="store_true", help="only apply the limit (idle case)")
    args = ap.parse_args()
    prompt = Path(args.prompt_file).read_text()
    out = {"args": vars(args), "samples": []}
    stop = threading.Event()

    def sampler():
        t0 = time.time()
        while not stop.is_set():
            out["samples"].append(
                {
                    "t": round(time.time() - t0, 2),
                    **read_stat(args.cgroup),
                    "server": proc_mem(server_pid()),
                }
            )
            time.sleep(0.5)

    out["before"] = {
        "stat": read_stat(args.cgroup),
        "server": proc_mem(server_pid()),
        "probe": probe(prompt),
    }
    gib = 2**30
    set_limits(args.cgroup, int(args.limit_gib * gib), int(args.swap_gib * gib))
    out["after_limit"] = {"stat": read_stat(args.cgroup), "server": proc_mem(server_pid())}
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    pyt = None
    if not args.no_pytest:
        tdir = ROOT / "results" / "calib" / "memtest"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "test_mem.py").write_text(TEST)
        env = {**os.environ, "ALLOC_MB": str(args.alloc_mb), "HOLD_S": str(args.hold_s)}
        pyt = subprocess.Popen(
            [
                "sh",
                "-c",
                f"echo $$ > {CG / args.cgroup}/cgroup.procs; exec python3 -m pytest -q "
                f"-p no:cacheprovider {tdir}/test_mem.py",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(5)  # let the test allocate
    out["during_probe"] = probe(prompt)  # decode on the long cached prompt under pressure
    if pyt is not None:
        so, _ = pyt.communicate(timeout=3600)
        out["pytest"] = {"rc": pyt.returncode, "tail": so[-600:], "wall_s": time.time() - t0}
    time.sleep(2)
    out["after_probe"] = probe(prompt)
    out["after_probe_2"] = probe(prompt)
    stop.set()
    th.join()
    out["final"] = {"stat": read_stat(args.cgroup), "server": proc_mem(server_pid())}
    path = ROOT / "results" / "calib" / f"mem_{args.tag}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    s = out["samples"]
    summary = {
        "tag": args.tag,
        "limit_gib": args.limit_gib,
        "swap_gib": args.swap_gib,
        "pytest_rc": out.get("pytest", {}).get("rc"),
        "server_alive_end": out["final"]["server"]["alive"],
        "oom_kills": out["final"]["stat"]["oom_kill"] - out["before"]["stat"]["oom_kill"],
        "min_cache_gib": min(x["cache"] for x in s) / gib if s else None,
        "max_swap_gib": max(x["swap"] for x in s) / gib if s else None,
        "majfault_delta": (
            out["final"]["stat"]["pgmajfault"] - out["before"]["stat"]["pgmajfault"]
        ),
        "before": out["before"]["probe"],
        "during": out["during_probe"],
        "after": out["after_probe"],
        "after2": out["after_probe_2"],
    }
    out["summary"] = summary
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
