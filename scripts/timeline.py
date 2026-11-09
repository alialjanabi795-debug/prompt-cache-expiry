"""Timeline of one trajectory for the paper's illustration (no new experiment: one replay with the
frozen configuration's settings on the measured device model).

Setting: weak_edge with the calibrated slowdown and no assumed throttling, average network, full
context, key-step policy, the Claude 5-minute-TTL preset, local decode speed 11 tok/s.
Selection rule (fixed here, applied to the 2,000-trajectory sample in sample order): among the
trajectories with at least one delegation that finds its cached prefix expired, take the one whose
number of steps is closest to the sample median; ties go to the earlier trajectory.

Writes results/timeline/steps.csv, results/timeline/calls.csv and results/timeline/meta.json.
Usage: python -I scripts/timeline.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from cache_delegation.config import load_catalog  # noqa: E402
from cache_delegation.loader import load_processed  # noqa: E402
from cache_delegation.pings import replay  # noqa: E402
from cache_delegation.policies import make_policy  # noqa: E402
from cache_delegation.preregistration import verify  # noqa: E402
from cache_delegation.simulator import Scenario, simulate  # noqa: E402

OUT = ROOT / "results" / "timeline"
DEVICE, PRESET, NETWORK, POLICY, TPS = "weak_edge", "anthropic_5m", "average", "keystep", 11.0


def scenario() -> Scenario:
    crit = yaml.safe_load((ROOT / "configs" / "kill_criteria.yaml").read_text())
    cat = load_catalog(ROOT / "configs")
    dev = cat.devices[DEVICE].with_decode_tps(TPS)
    dev = replace(dev, thermal=replace(dev.thermal, enabled=False))  # measured device model
    return Scenario(
        dev,
        cat.networks[NETWORK],
        cat.cloud[PRESET],
        cat.edge,
        cat.tool_time,
        dynamic=True,
        context="full",
        seed=crit["primary_config"]["tool_time_seed"],
    )


def main() -> None:
    verify(ROOT / "configs" / "kill_criteria.yaml", ROOT / "configs" / "kill_criteria.lock.json")
    trajs = load_processed(ROOT / "data" / "processed", "openhands")
    sc = scenario()
    pol = make_policy(POLICY)
    median_steps = float(np.median([t.n_steps for t in trajs]))
    best = None
    for order, t in enumerate(trajs):
        sim = simulate(t, pol, sc)
        if not any(c.result.cold for c in sim.delegations):
            continue
        key = (abs(t.n_steps - median_steps), order)
        if best is None or key < best[0]:
            best = (key, t, sim)
    (_, order), traj, sim = best
    rep = replay(sim, sc.cloud)
    steps = pd.DataFrame(
        [
            {
                "index": s.index,
                "action": s.action,
                "tier": s.executed.name.lower(),
                "t_start_s": s.t_start,
                "t_llm_end_s": s.t_llm_end,
                "t_end_s": s.t_end,
                "context_tokens": s.context_tokens,
            }
            for s in sim.steps
        ]
    )
    calls = pd.DataFrame(
        [
            {
                "step_index": c.step_index,
                "t_start_s": c.t,
                "prompt_tokens": c.result.prompt_tokens,
                "hit_tokens": c.result.hit_tokens,
                "write_tokens": c.result.write_tokens,
                "expired_tokens": c.result.expired_tokens,
                "expired": c.result.cold,
                "usd": c.usd_cache,
                "usd_if_never_expired": c.usd_inf_ttl,
                "usd_without_cache": c.usd_nocache,
            }
            for c in rep.calls
            if c.kind == "delegation"
        ]
    )
    calls["gap_s"] = calls.t_start_s.diff()
    OUT.mkdir(parents=True, exist_ok=True)
    steps.to_csv(OUT / "steps.csv", index=False)
    calls.to_csv(OUT / "calls.csv", index=False)
    meta = {
        "traj_id": traj.traj_id,
        "sample_order": order,
        "n_steps": traj.n_steps,
        "sample_median_steps": median_steps,
        "device": DEVICE,
        "thermal": "off (measured device model)",
        "network": NETWORK,
        "preset": PRESET,
        "ttl_s": sc.cloud.ttl_s,
        "policy": POLICY,
        "decode_tps": TPS,
        "total_s": sim.total_s,
        "selection_rule": "fewest |n_steps - sample median| among trajectories with at least one "
        "expired-cache delegation; ties by sample order",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(calls.round(3).to_string())


if __name__ == "__main__":
    main()
