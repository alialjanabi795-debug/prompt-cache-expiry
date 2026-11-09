"""End-to-end smoke test of the sweep runner + verdict analysis on SYNTHETIC trajectories only
(no real-data results are produced before the sweep is approved)."""

import importlib.util
import sys

import pandas as pd
import yaml

from .conftest import ROOT, mixed_traj


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_k_grid_runner_and_analysis(catalog, tmp_path, monkeypatch):
    rs = _load("run_sweep")
    an = _load("analyze")
    crit = yaml.safe_load((ROOT / "configs" / "kill_criteria.yaml").read_text())
    trajs = [mixed_traj(n=25, obs=2000 + 300 * i, traj_id=f"s{i}") for i in range(6)]
    rs._init(catalog, {"openhands": trajs, "swesmith": trajs}, 0, 0)
    jobs = rs.k_grid(crit)
    assert len(jobs) == 9
    rows = [r for j in jobs for r in rs.run_job(j)]
    df = pd.DataFrame(rows)
    assert set(rs.K_POLICIES) <= set(df.policy)
    ks = df[df.policy == "keystep"]
    assert (ks.usd_c_ping_opt <= ks.usd_c + 1e-12).all()
    assert ks.gap_s.notna().all() and ks.nom_n_cold.notna().all()
    path = tmp_path / "k.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr(sys, "argv", ["analyze", "--input", str(path), "--out", str(tmp_path)])
    an.main()
    v = yaml.safe_load((tmp_path / "verdicts.json").read_text())
    fz = v["frozen"]
    for k in ("K1", "K2", "K3", "K5", "K6", "K7", "tool_rule"):
        assert k in fz
    assert len(fz["cells"]) == 9
    assert v["K2_status"].startswith(("PROVISIONAL", "FINAL"))
    assert (tmp_path / "k6_k7_three_arms.csv").exists()


def test_sens_grid_builds(catalog):
    rs = _load("run_sweep")
    crit = yaml.safe_load((ROOT / "configs" / "kill_criteria.yaml").read_text())
    jobs = rs.sens_grid(crit)
    assert {j.grid for j in jobs} == {"sens_ctx", "sens_preset", "robust_swesmith"}
    assert all(j.arm == "thermal_off" for j in jobs)
    reg = rs.regime_grid(crit)
    assert len(reg) == 2 * 3 * 3 * len(rs.REGIME_TPS)
    assert {j.arm for j in reg} == {"thermal_off", "thermal_on"}
    rs._init(catalog, {"openhands": [mixed_traj(n=12)], "swesmith": [mixed_traj(n=12)]}, 0, 0)
    for j in jobs[:: max(1, len(jobs) // 6)]:
        assert rs.run_job(j)
