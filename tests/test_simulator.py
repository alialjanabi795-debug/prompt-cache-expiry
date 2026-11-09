from dataclasses import replace

import pytest

from cache_delegation.context import STRATEGIES
from cache_delegation.metrics import coupling, gaps, summarize
from cache_delegation.policies import AllCloud, AllLocal, DelegateEarly, KeyStep, Sticky
from cache_delegation.simulator import Scenario, simulate
from cache_delegation.types import ActionType, Tier

from .conftest import make_traj, mixed_traj, quiet_device


def sc_(catalog, dynamic=True, tps=11.0, preset="anthropic_5m", context="full", device=None):
    dev = device or quiet_device(catalog.devices["weak_edge"])
    return Scenario(
        dev.with_decode_tps(tps),
        catalog.networks["average"],
        catalog.cloud[preset],
        catalog.edge,
        catalog.tool_time,
        dynamic=dynamic,
        context=context,
    )


def test_all_local_makes_no_cloud_calls(catalog):
    sim = simulate(make_traj(n=8), AllLocal(), sc_(catalog))
    assert sim.cloud_calls == []
    assert all(s.executed == Tier.LOCAL for s in sim.steps)
    assert sim.total_s > 0


def test_all_cloud_consecutive_calls_are_warm(catalog):
    sim = simulate(make_traj(n=8), AllCloud(), sc_(catalog))
    d = sim.delegations
    assert len(d) == 8
    assert d[0].result.compulsory
    assert all(not c.result.cold for c in d[1:])
    assert all(c.result.hit_tokens > 0 for c in d[1:])


def test_keystep_delegates_edits_and_long_reads(catalog):
    acts = [
        ActionType.READ_SEARCH,
        ActionType.READ_SEARCH,
        ActionType.WRITE_EDIT,
        ActionType.TEST_VALIDATE,
    ]
    traj = make_traj(actions=acts, obs_list=[200, 1500, 100, 3000])
    sim = simulate(traj, KeyStep(), sc_(catalog))
    assert [s.requested for s in sim.steps] == [Tier.LOCAL, Tier.CLOUD, Tier.CLOUD, Tier.LOCAL]


def test_deterministic(catalog):
    traj = make_traj(n=20, obs=2000)
    a = summarize(traj, simulate(traj, KeyStep(), sc_(catalog)))
    b = summarize(traj, simulate(traj, KeyStep(), sc_(catalog)))
    assert a == b


def test_slower_device_means_longer_gaps_and_more_cold(catalog):
    traj = mixed_traj()
    fast = simulate(traj, KeyStep(), sc_(catalog, tps=60.0))
    slow = simulate(traj, KeyStep(), sc_(catalog, tps=5.0))
    assert slow.total_s > fast.total_s
    assert sum(c.result.cold for c in slow.delegations) > sum(
        c.result.cold for c in fast.delegations
    )


def test_coupling_same_decisions_and_dynamic_not_faster(catalog):
    traj = mixed_traj()
    nom = simulate(traj, KeyStep(), sc_(catalog, dynamic=False))
    dyn = simulate(traj, KeyStep(), sc_(catalog, dynamic=True))
    cp = coupling(nom, dyn)
    assert cp.n_deleg == len(dyn.delegations) - 1
    assert dyn.total_s >= nom.total_s
    assert cp.cold_dynamic >= cp.cold_nominal - cp.cold_only_nominal


def test_oom_falls_back_to_edge(catalog):
    traj = make_traj(n=12, prefix=50_000, obs=2000)
    sim = simulate(traj, AllLocal(), sc_(catalog))
    assert any(s.fallback == "oom" and s.executed == Tier.EDGE for s in sim.steps)
    assert sim.edge_usd > 0


def test_faults_fall_back_to_edge(catalog):
    dev = catalog.devices["weak_edge"]
    dev = replace(dev, fault=replace(dev.fault, p_base=1.0, p_max=1.0))
    sim = simulate(make_traj(n=5), AllLocal(), sc_(catalog, device=dev))
    assert all(s.fallback == "fault" for s in sim.steps)


def _oom_on_tests(catalog):
    """Device whose budget fits everything except a test run (0.27 GiB) -> OOM kill on tests.

    Anonymous weights (as with -lm none / mlock) so nothing is reclaimable, one checkpoint.
    """
    p = quiet_device(catalog.devices["weak12_fp16"])
    m = p.memory
    ram = p.kv_reserved_gb + m.runtime_overhead_gb + m.checkpoint_gb + m.weights_gb + 0.1
    return replace(p, memory=replace(m, ram_usable_gb=ram, max_checkpoints=1, weights_mmap=False))


def test_tool_oom_kill_forces_reload_and_reprefill(catalog):
    acts = [ActionType.READ_SEARCH, ActionType.TEST_VALIDATE, ActionType.READ_SEARCH]
    traj = make_traj(actions=acts, prefix=45_000, obs_list=[800, 800, 800])
    sim = simulate(traj, AllLocal(), sc_(catalog, device=_oom_on_tests(catalog)))
    assert sim.steps[1].kv_evicted_after and sim.oom_kills == 1
    assert sim.steps[2].reprefill
    assert sim.steps[2].local_reload_s == pytest.approx(30.0)
    assert sim.steps[2].local_prefill_tokens == traj.steps[2].context_tokens
    nom = simulate(traj, AllLocal(), sc_(catalog, dynamic=False, device=_oom_on_tests(catalog)))
    assert not nom.steps[2].reprefill and nom.oom_kills == 0


def test_ample_memory_keeps_kv_across_tests(catalog):
    acts = [ActionType.READ_SEARCH, ActionType.TEST_VALIDATE, ActionType.READ_SEARCH]
    traj = make_traj(actions=acts, prefix=45_000, obs_list=[800, 800, 800])
    sim = simulate(traj, AllLocal(), sc_(catalog))
    assert sim.oom_kills == 0 and not sim.steps[2].reprefill
    assert sim.steps[2].local_prefill_tokens == 800  # own output is already in the KV


def test_delegate_early_moves_delegation_after_eviction(catalog):
    acts = [
        ActionType.READ_SEARCH,
        ActionType.TEST_VALIDATE,
        ActionType.NAVIGATE,
        ActionType.NAVIGATE,
        ActionType.WRITE_EDIT,
    ]
    traj = make_traj(actions=acts, prefix=45_000, obs_list=[300] * 5)
    dev = _oom_on_tests(catalog)
    sim = simulate(traj, DelegateEarly(KeyStep(), shift=3), sc_(catalog, device=dev))
    assert [s.requested for s in sim.steps] == [
        Tier.LOCAL,
        Tier.LOCAL,
        Tier.CLOUD,
        Tier.CLOUD,
        Tier.CLOUD,
    ]
    base = simulate(traj, KeyStep(), sc_(catalog, device=dev))
    assert base.steps[2].requested == Tier.LOCAL and base.steps[2].reprefill


def test_sticky_extends_cloud_runs(catalog):
    acts = [ActionType.WRITE_EDIT] + [ActionType.NAVIGATE] * 4
    sim = simulate(make_traj(actions=acts), Sticky(KeyStep(), 2), sc_(catalog))
    assert [s.requested for s in sim.steps] == [Tier.CLOUD] * 3 + [Tier.LOCAL] * 2


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_context_strategies_run(catalog, strategy):
    traj = mixed_traj(n=20, obs=2500)
    sim = simulate(traj, KeyStep(), sc_(catalog, context=strategy, tps=5.0))
    full = simulate(traj, KeyStep(), sc_(catalog, context="full", tps=5.0))
    tok = sum(c.result.prompt_tokens for c in sim.delegations)
    tok_full = sum(c.result.prompt_tokens for c in full.delegations)
    if strategy in ("traj_drop", "compact_pre", "compact_suf"):
        assert tok < tok_full
    if strategy == "compact_suf":
        assert any(c.kind == "summary" for c in sim.cloud_calls)
    if strategy == "compact_pre":
        assert any(s.handoff_s > 0 for s in sim.steps)


def test_gap_decomposition_sums(catalog):
    traj = mixed_traj(n=20, obs=2000)
    sim = simulate(traj, KeyStep(), sc_(catalog))
    gs = gaps(traj, sim)
    assert gs
    assert len(gs) == len(sim.delegations) - 1
    for g in gs:
        parts = g.tool_s + g.local_s + g.edge_s + g.cloud_s + g.handoff_s
        assert parts == pytest.approx(g.duration_s, rel=0.05, abs=0.5)
