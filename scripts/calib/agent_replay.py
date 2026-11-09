"""Calibration run: replay one real trajectory against llama-server, agent style.

Per step i the prompt is the rendered transcript up to step i (append-only, as an agent loop sends
it). We request min(output_tokens_i, decode_sample) decode tokens, so each request measures
  * prefill of the new suffix (previous output + observation) at the current context, and
  * decode tok/s at the current context.
The recorded output then enters the next prompt; the server rolls back to its checkpoint at the end
of the previous prompt (verified: prefix reuse survives divergent generations on Qwen3.5).
Between steps we idle for the step's tool time (tool-time model draw, capped) like an agent.
Memory of the device cgroup is sampled after every step. Results: results/calib/replay_*.jsonl

Usage: python -I scripts/calib/agent_replay.py --target-ctx 53248 --cgroup dev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402
from llama_client import completion, wait_ready  # noqa: E402

from cache_delegation.config import load_catalog  # noqa: E402
from cache_delegation.loader import (  # noqa: E402
    iter_raw,
    load_processed,
    render_assistant,
    render_observation,
    render_prefix,
    sample_indices,
)
from cache_delegation.timing import ToolTimer  # noqa: E402

CG = Path("/sys/fs/cgroup/memory")


def cg_mem(name: str) -> dict:
    d = CG / name
    stat = {}
    for line in (d / "memory.stat").read_text().splitlines():
        k, v = line.split()
        if k in ("rss", "cache", "mapped_file", "swap", "shmem", "active_file", "inactive_file"):
            stat[k] = int(v)
    stat["usage"] = int((d / "memory.usage_in_bytes").read_text())
    stat["failcnt"] = int((d / "memory.failcnt").read_text())
    return stat


def pick_trajectory(target_ctx: int):
    """Sampled trajectory whose context crosses target_ctx closest to its end (agent-typical)."""
    trajs = load_processed(ROOT / "data" / "processed", "openhands")
    best = None
    for t in trajs:
        if t.final_context_tokens < target_ctx * 1.05 or t.final_context_tokens > target_ctx * 1.4:
            continue
        score = abs(t.n_steps - 61)
        if best is None or score < best[0]:
            best = (score, t)
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-ctx", type=int, default=53248)
    ap.add_argument("--decode-sample", type=int, default=32)
    ap.add_argument("--tool-cap-s", type=float, default=20.0)
    ap.add_argument("--cgroup", default="dev")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--traj-id", default=None, help="replay this trajectory (default: auto-pick)")
    args = ap.parse_args()

    if args.traj_id:
        t = next(
            x
            for x in load_processed(ROOT / "data" / "processed", "openhands")
            if x.traj_id == args.traj_id
        )
    else:
        t = pick_trajectory(args.target_ctx)
    path = ROOT / "data" / "raw" / "openhands" / "trajectories.parquet"
    idx = sample_indices(pq.ParquetFile(path).metadata.num_rows, 2000, 20261019)
    raw = next(r for r in iter_raw(path, "openhands", idx) if r.traj_id == t.traj_id)
    timer = ToolTimer(load_catalog(ROOT / "configs").tool_time, seed=0)

    out = ROOT / "results" / "calib" / f"replay_{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    wait_ready(args.port)
    prompt = render_prefix(raw.system, raw.tools, raw.task)
    t_start = time.time()
    with open(out, "w") as fh:
        fh.write(
            json.dumps(
                {"traj_id": t.traj_id, "n_steps": t.n_steps, "final_ctx": t.final_context_tokens}
            )
            + "\n"
        )
        for i, s in enumerate(raw.steps):
            step = t.steps[i]
            n_dec = max(1, min(step.output_tokens, args.decode_sample))
            prompt_sent = prompt
            r = completion(prompt, n_dec, port=args.port)
            tm = r["timings"]
            rec = {
                "step": i,
                "action": step.action.value,
                "elapsed_s": time.time() - t_start,
                "ctx_before": tm.get("cache_n", 0),
                "prompt_n": tm["prompt_n"],
                "prompt_ms": tm["prompt_ms"],
                "predicted_n": tm["predicted_n"],
                "predicted_ms": tm["predicted_ms"],
                "wall_s": r["wall_s"],
                "ctx_tokens_model": step.context_tokens,
                "mem": cg_mem(args.cgroup),
            }
            tool_s = min(args.tool_cap_s, timer.source_seconds(t.traj_id, step))
            rec["tool_sleep_s"] = tool_s
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(
                f"step {i:3d} ctx {rec['ctx_before'] + rec['prompt_n']:6d} "
                f"pp {tm['prompt_n']:5d} @ {tm['prompt_n'] / max(tm['prompt_ms'], 1) * 1e3:6.1f}"
                f" t/s | tg {tm['predicted_n'] / max(tm['predicted_ms'], 1) * 1e3:5.2f} t/s | "
                f"rss {rec['mem']['rss'] / 2**30:5.2f} GiB cache "
                f"{rec['mem']['cache'] / 2**30:5.2f} GiB | {rec['elapsed_s'] / 60:5.1f} min",
                flush=True,
            )
            time.sleep(tool_s)
            prompt += render_assistant(s.content, s.calls) + render_observation(s.obs_parts)
            if tm.get("cache_n", 0) + tm["prompt_n"] >= args.target_ctx:
                (out.parent / f"last_prompt_{args.tag}.txt").write_text(prompt_sent)
                break


if __name__ == "__main__":
    main()
