"""Per-run summaries, inter-delegation gap decomposition and the coupling comparison."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .simulator import SimResult
from .types import Tier, Trajectory


def summarize(traj: Trajectory, sim: SimResult) -> dict:
    dels = sim.delegations
    calls = sim.cloud_calls
    non_compulsory = [c for c in dels if not c.result.compulsory]
    steps = sim.steps
    return {
        "traj_id": traj.traj_id,
        "policy": sim.policy,
        "scenario": sim.scenario,
        "n_steps": traj.n_steps,
        "resolved": traj.resolved,
        # (a) call count = delegated steps (EdgeWise metric)
        "calls_a": sum(c.cost.calls for c in calls),
        "n_cloud_requests": len(calls),
        # (b) token pricing without cache, (c) with cache
        "usd_b": sum(c.cost.usd_nocache for c in calls),
        "usd_c": sum(c.cost.usd_cache for c in calls),
        "usd_c_inf_ttl": sum(c.cost.usd_cache_inf_ttl for c in calls),
        "usd_c_deleg": sum(c.cost.usd_cache for c in dels),
        "usd_c_inf_ttl_deleg": sum(c.cost.usd_cache_inf_ttl for c in dels),
        "usd_out": sum(c.cost.usd_output for c in calls),
        "edge_usd": sim.edge_usd,
        "prompt_tokens": sum(c.result.prompt_tokens for c in calls),
        "hit_tokens": sum(c.result.hit_tokens for c in calls),
        "write_tokens": sum(c.result.write_tokens for c in calls),
        "expired_tokens": sum(c.result.expired_tokens for c in dels),
        "n_deleg": len(dels),
        "n_deleg_noncompulsory": len(non_compulsory),
        "n_cold": sum(1 for c in dels if c.result.cold),
        "upload_bytes": sum(c.upload_bytes for c in calls),
        "deleg_latency_s": sum(c.latency_s for c in dels),
        "total_s": sim.total_s,
        "local_prefill_s": sum(s.local_prefill_s for s in steps),
        "local_decode_s": sum(s.local_decode_s for s in steps),
        "local_wasted_s": sum(s.local_wasted_s for s in steps),
        "local_prefill_tokens": sum(s.local_prefill_tokens for s in steps),
        "tool_s": sum(s.tool_s for s in steps),
        "edge_s": sum(s.remote_s for s in steps if s.executed == Tier.EDGE),
        "handoff_s": sum(s.handoff_s for s in steps),
        "n_local": sum(1 for s in steps if s.executed == Tier.LOCAL),
        "n_edge_fault": sum(1 for s in steps if s.fallback == "fault"),
        "n_edge_oom": sum(1 for s in steps if s.fallback == "oom"),
        "n_reprefill": sum(1 for s in steps if s.reprefill),
        "n_kv_evictions": sum(1 for s in steps if s.kv_evicted_after),
        "throttle_s": sim.throttle_s,
        "thrash_s": sim.thrash_s,
        "oom_kills": sim.oom_kills,
        "local_reload_s": sum(s.local_reload_s for s in steps),
        "energy_compute_j": sim.energy_compute_j,
        "energy_tool_j": sim.energy_tool_j,
        "energy_idle_j": sim.energy_idle_j,
        "energy_radio_j": sim.energy_radio_j,
    }


@dataclass(frozen=True, slots=True)
class Gap:
    traj_id: str
    from_step: int
    to_step: int
    duration_s: float
    cold: bool
    tool_s: float
    local_s: float  # prefill + decode + wasted
    edge_s: float
    cloud_s: float  # latency of the delegation that opens the gap
    handoff_s: float
    n_local_steps: int


def gaps(traj: Trajectory, sim: SimResult) -> list[Gap]:
    """Start-to-start intervals between consecutive delegations (the TTL clock)."""
    dels = sim.delegations
    out: list[Gap] = []
    for a, b in itertools.pairwise(dels):
        seg = sim.steps[a.step_index : b.step_index]
        nxt = sim.steps[b.step_index]
        out.append(
            Gap(
                traj_id=traj.traj_id,
                from_step=a.step_index,
                to_step=b.step_index,
                duration_s=b.t_start - a.t_start,
                cold=b.result.cold,
                tool_s=sum(s.tool_s for s in seg),
                local_s=sum(
                    s.local_prefill_s + s.local_decode_s + s.local_wasted_s + s.local_reload_s
                    for s in seg
                ),
                edge_s=sum(s.remote_s for s in seg if s.executed == Tier.EDGE),
                cloud_s=a.latency_s,
                handoff_s=nxt.handoff_s,
                n_local_steps=sum(1 for s in seg if s.executed == Tier.LOCAL),
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class Coupling:
    n_deleg: int
    cold_nominal: int
    cold_dynamic: int
    cold_only_dynamic: int  # turned cold only because local steps slowed down
    cold_only_nominal: int


def coupling(nominal: SimResult, dynamic: SimResult) -> Coupling:
    nom = {c.step_index: c.result for c in nominal.delegations}
    dyn = {c.step_index: c.result for c in dynamic.delegations}
    if nom.keys() != dyn.keys():
        raise ValueError("coupling requires identical delegation decisions in both arms")
    keys = [k for k in nom if not nom[k].compulsory]
    return Coupling(
        n_deleg=len(keys),
        cold_nominal=sum(nom[k].cold for k in keys),
        cold_dynamic=sum(dyn[k].cold for k in keys),
        cold_only_dynamic=sum(dyn[k].cold and not nom[k].cold for k in keys),
        cold_only_nominal=sum(nom[k].cold and not dyn[k].cold for k in keys),
    )
