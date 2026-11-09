"""Keep-warm pings: the optimized schedule never increases cost; replay matches the simulator."""

from dataclasses import replace

import numpy as np
import pytest

from cache_delegation.pings import fixed_keepwarm, optimal_pings, replay
from cache_delegation.policies import KeyStep, RandomP
from cache_delegation.simulator import Scenario, simulate
from cache_delegation.types import ActionType

from .conftest import make_traj, mixed_traj, quiet_device


def _scenario(catalog, preset="anthropic_5m", tps=5.0):
    dev = quiet_device(catalog.devices["weak_edge"]).with_decode_tps(tps)
    return Scenario(
        dev, catalog.networks["average"], catalog.cloud[preset], catalog.edge, catalog.tool_time
    )


def _random_traj(rng, i):
    n = int(rng.integers(8, 40))
    all_acts = list(ActionType)
    acts = [all_acts[k] for k in rng.integers(0, len(all_acts), size=n)]
    obs = [int(x) for x in rng.lognormal(6.5, 1.2, size=n)]
    return make_traj(actions=acts, obs_list=obs, traj_id=f"r{i}", out=int(rng.integers(50, 400)))


def test_replay_without_pings_matches_simulator(catalog):
    traj = make_traj(n=30, obs=3000)
    for preset in ("anthropic_5m", "anthropic_1h", "openai_30m"):
        sim = simulate(traj, KeyStep(), _scenario(catalog, preset))
        rep = replay(sim, catalog.cloud[preset])
        assert rep.usd_cache == pytest.approx(sum(c.cost.usd_cache for c in sim.cloud_calls))
        assert rep.cold_delegations == sum(c.result.cold for c in sim.delegations)


@pytest.mark.parametrize("preset", ["anthropic_5m", "anthropic_1h", "openai_30m"])
@pytest.mark.parametrize("mode", ["refresh", "extend", "best"])
def test_ping_schedule_never_increases_cost(catalog, preset, mode):
    rng = np.random.default_rng(123)
    for i in range(25):
        traj = _random_traj(rng, i)
        sc = _scenario(catalog, preset, tps=float(rng.choice([5.0, 11.0])))
        sim = simulate(traj, RandomP(0.3, seed=i), sc)
        base = replay(sim, sc.cloud).usd_cache
        pings, _ = optimal_pings(traj, sim, sc.cloud, mode=mode)
        with_p = replay(sim, sc.cloud, pings).usd_cache
        assert with_p <= base + 1e-12


def test_planned_saving_matches_replay(catalog):
    rng = np.random.default_rng(5)
    for i in range(15):
        traj = _random_traj(rng, i)
        sc = _scenario(catalog, "anthropic_5m", 5.0)
        sim = simulate(traj, RandomP(0.25, seed=i), sc)
        base = replay(sim, sc.cloud).usd_cache
        pings, plans = optimal_pings(traj, sim, sc.cloud, mode="best")
        got = base - replay(sim, sc.cloud, pings).usd_cache
        assert got == pytest.approx(sum(p.planned_saving_usd for p in plans), rel=1e-9, abs=1e-12)


def test_optimal_pings_remove_cold_delegations_when_cheap(catalog):
    # long gaps (slow device), expensive context: every cold gap should be kept warm
    traj = mixed_traj(n=25, obs=2500)
    sc = _scenario(catalog, "anthropic_5m", 5.0)
    sim = simulate(traj, KeyStep(), sc)
    base = replay(sim, sc.cloud)
    assert base.cold_delegations > 0
    pings, _plans = optimal_pings(traj, sim, sc.cloud)
    rep = replay(sim, sc.cloud, pings)
    assert rep.cold_delegations < base.cold_delegations
    assert rep.usd_cache < base.usd_cache
    assert all(p.t < sim.total_s for p in pings)


def test_fixed_keepwarm_keeps_everything_warm(catalog):
    traj = mixed_traj(n=25, obs=2500)
    sc = _scenario(catalog, "anthropic_5m", 5.0)
    sim = simulate(traj, KeyStep(), sc)
    rep = replay(sim, sc.cloud, fixed_keepwarm(sim, sc.cloud))
    assert rep.cold_delegations == 0
    assert rep.n_pings > 0


def test_no_pings_needed_with_long_ttl(catalog):
    traj = make_traj(n=12, obs=500)
    sc = _scenario(catalog, "anthropic_1h", 30.0)
    sim = simulate(traj, KeyStep(), sc)
    pings, plans = optimal_pings(traj, sim, sc.cloud)
    assert pings == [] and plans == []


def test_optimal_rejects_summary_calls(catalog):
    traj = mixed_traj(n=10, obs=2000)
    sc = replace(_scenario(catalog), context="compact_suf")
    sim = simulate(traj, KeyStep(), sc)
    with pytest.raises(ValueError):
        optimal_pings(traj, sim, sc.cloud)
