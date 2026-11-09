"""Rebuild the rendered prompt that agent_replay sent at step k (for probes on the cached KV).

Usage: python -I scripts/calib/dump_prompt.py --replay results/calib/replay_main.jsonl --out FILE
(k defaults to the last step recorded in the replay file)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq  # noqa: E402

from cache_delegation.loader import (  # noqa: E402
    iter_raw,
    render_assistant,
    render_observation,
    render_prefix,
    sample_indices,
)


def build_prompt(traj_id: str, k: int) -> str:
    path = ROOT / "data" / "raw" / "openhands" / "trajectories.parquet"
    idx = sample_indices(pq.ParquetFile(path).metadata.num_rows, 2000, 20261019)
    raw = next(r for r in iter_raw(path, "openhands", idx) if r.traj_id == traj_id)
    p = render_prefix(raw.system, raw.tools, raw.task)
    for s in raw.steps[:k]:
        p += render_assistant(s.content, s.calls) + render_observation(s.obs_parts)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = [json.loads(x) for x in Path(a.replay).read_text().splitlines()]
    k = a.step if a.step is not None else lines[-1]["step"]
    Path(a.out).write_text(build_prompt(lines[0]["traj_id"], k))
    print(f"prompt for step {k} written to {a.out}")
