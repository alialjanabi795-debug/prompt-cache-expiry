from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cache_delegation.policies import DecisionContext, RandomP, make_policy
from cache_delegation.preregistration import freeze, holds_k4, pooled_ratio, ranking, verify
from cache_delegation.types import Tier

from .conftest import make_traj

CTX = DecisionContext(now=0.0, prev_tiers=(), local_prefill_needed=0, local_kv_evicted=False)


def test_random_policy_rate_and_determinism():
    traj = make_traj(n=2000)
    pol = RandomP(0.2, seed=1)
    d1 = [pol.decide(traj, i, CTX) for i in range(traj.n_steps)]
    d2 = [RandomP(0.2, seed=1).decide(traj, i, CTX) for i in range(traj.n_steps)]
    assert d1 == d2
    rate = np.mean([d == Tier.CLOUD for d in d1])
    assert 0.17 < rate < 0.23


def test_make_policy_specs():
    for spec in [
        "all_local",
        "all_cloud",
        "random_0.4",
        "keystep",
        "keystep_L2000",
        "sticky2_keystep",
        "early3_keystep",
    ]:
        assert make_policy(spec).name == spec


def test_pooled_ratio_and_ci():
    est = pooled_ratio(np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 10.0, 10.0, 10.0]))
    assert est.value == pytest.approx(0.25)
    assert est.lo <= 0.25 <= est.hi


def test_holds_k4():
    presets = ["a", "b", "c"]
    cells = {(p, t): True for p in presets for t in (5, 11, 30)}
    assert holds_k4(cells, presets, (5, 11, 30))
    cells[("a", 5)] = False
    assert holds_k4(cells, presets, (5, 11, 30))
    cells[("b", 5)] = False
    assert not holds_k4(cells, presets, (5, 11, 30))


def test_ranking_ties_use_list_order():
    df = pd.DataFrame({"policy": ["x", "y", "z"], "v": [2.0, 1.0, 2.0]})
    assert ranking(df, ["x", "y", "z"], "v") == ["y", "x", "z"]


def test_freeze_and_verify(tmp_path: Path):
    y = tmp_path / "k.yaml"
    y.write_text("a: 1\n")
    lock = tmp_path / "k.lock.json"
    freeze(y, lock)
    verify(y, lock)
    with pytest.raises(FileExistsError):
        freeze(y, lock)
    y.write_text("a: 2\n")
    with pytest.raises(RuntimeError):
        verify(y, lock)
