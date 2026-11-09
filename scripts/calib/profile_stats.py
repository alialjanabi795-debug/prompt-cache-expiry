"""KeyStep timeline statistics on the calibrated device profiles (stop-point 2; no K metrics).

For each profile / decode tps / arm: inter-delegation gaps (TTL clock), tool share, wall time,
memory events (OOM kills, thrashing time), edge fallbacks. Writes results/calib/profile_stats.csv.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cache_delegation.config import load_catalog  # noqa: E402
from cache_delegation.loader import load_processed  # noqa: E402
from cache_delegation.metrics import gaps, summarize  # noqa: E402
from cache_delegation.policies import KeyStep  # noqa: E402
from cache_delegation.preregistration import verify  # noqa: E402
from cache_delegation.simulator import Scenario, simulate  # noqa: E402

_S: dict = {}


def _init(trajs, cat):
    _S.update(trajs=trajs, cat=cat)


def _run(job):
    name, tps, dyn, thermal = job
    cat = _S["cat"]
    dev = cat.devices[name].with_decode_tps(tps)
    if not thermal:
        dev = replace(dev, thermal=replace(dev.thermal, enabled=False))
    sc = Scenario(
        dev,
        cat.networks["average"],
        cat.cloud["anthropic_5m"],
        cat.edge,
        cat.tool_time,
        dynamic=dyn,
    )
    runs, gs = [], []
    for t in _S["trajs"]:
        sim = simulate(t, KeyStep(), sc)
        runs.append(summarize(t, sim))
        gs += [asdict(g) for g in gaps(t, sim)]
    r, g = pd.DataFrame(runs), pd.DataFrame(gs)
    d = g.duration_s.to_numpy()
    return {
        "profile": name,
        "decode_tps": tps,
        "arm": "dynamic" if dyn else "nominal",
        "thermal": thermal,
        "wall_median_min": r.total_s.median() / 60,
        "wall_p90_min": r.total_s.quantile(0.9) / 60,
        "gap_median_s": np.median(d),
        "gap_p90_s": np.quantile(d, 0.9),
        "gaps_ge_300s": float(np.mean(d >= 300)),
        "gaps_ge_1800s": float(np.mean(d >= 1800)),
        "tool_share_of_gap": g.tool_s.sum() / d.sum(),
        "oom_kills_per_traj": r.oom_kills.mean(),
        "thrash_share_of_wall": r.thrash_s.sum() / r.total_s.sum(),
        "throttle_share_of_wall": r.throttle_s.sum() / r.total_s.sum(),
        "edge_ctx_share_of_steps": r.n_edge_oom.sum() / r.n_steps.sum(),
        "edge_fault_share_of_steps": r.n_edge_fault.sum() / r.n_steps.sum(),
    }


def main() -> None:
    verify(ROOT / "configs" / "kill_criteria.yaml", ROOT / "configs" / "kill_criteria.lock.json")
    cat = load_catalog(ROOT / "configs")
    trajs = load_processed(ROOT / "data" / "processed", "openhands")
    jobs = [
        (p, tps, dyn, True)
        for p in ("weak_edge", "weak12_fp16", "weak8_q8_16k")
        for tps in (5.0, 11.0, 30.0)
        for dyn in (True, False)
    ]
    jobs += [
        ("weak8_fp16", 11.0, True, True),
        ("weak12_q8", 11.0, True, True),
        ("weak12_q8", 11.0, False, True),
        ("weak_edge", 11.0, True, False),
        ("weak_edge", 5.0, True, False),
        ("weak_edge", 30.0, True, False),
        ("laptop_16gb", 11.0, True, True),
        ("dgx_spark", 11.19, True, True),
    ]
    with Pool(4, initializer=_init, initargs=(trajs, cat)) as pool:
        rows = pool.map(_run, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "calib" / "profile_stats.csv"
    df.to_csv(out, index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
