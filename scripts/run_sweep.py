"""Phase-0 sweep.

Writes one row per (trajectory, configuration) to results/sweep/<grid>.parquet. Device arms
(deviation D5): "thermal_on" = profile as configured (assumed thermal model), "thermal_off" =
measured context slowdown and memory model only, "const" = constant nominal speed.
  k             FROZEN primary grid: weak_edge, thermal_on, presets x tps {5,11,30} x policies
  k_measured    same grid, thermal_off (basis of every hardware-coupling claim)
  k_const       same grid, constant speed (third arm for K6/K7)
  k_laptop_on / k_laptop_off / k_dgx   comparison profiles (same grid)
  regime        regime map: weak_edge KeyStep, tps 5..60 x TTL preset x network x {off, on}
  sens          context strategies (Handoff-Tax, component 8), Opus 5.5 pricing, SWE-smith
Overlay policies carry the constant-speed arm (coupling), ping overlays and gap aggregates.
Finished grids are skipped on rerun (resume).

Usage: python -I scripts/run_sweep.py --grid all|<name>[,<name>...] [--limit N] [--workers 4]
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from cache_delegation.config import Catalog, load_catalog  # noqa: E402
from cache_delegation.loader import load_processed  # noqa: E402
from cache_delegation.metrics import coupling, gaps, summarize  # noqa: E402
from cache_delegation.pings import fixed_keepwarm, optimal_pings, replay  # noqa: E402
from cache_delegation.policies import make_policy  # noqa: E402
from cache_delegation.preregistration import verify  # noqa: E402
from cache_delegation.simulator import Scenario, simulate  # noqa: E402

K3_POLICIES = ["random_0.2", "random_0.4", "keystep", "sticky2_keystep", "all_cloud"]
K_POLICIES = [
    *K3_POLICIES,
    "all_local",
    "early1_keystep",
    "early2_keystep",
    "early3_keystep",
    "keystep_L250",
]
# Policies that get ping overlays, the nominal arm and gap aggregates. keystep is the frozen
# verdict policy; keystep_L250 (~40% coverage) and random_0.4 are reporting-only (deviation D3).
OVERLAY_POLICIES = {"keystep", "keystep_L250", "random_0.4"}


@dataclass(frozen=True)
class Job:
    grid: str
    device: str
    decode_tps: float
    network: str
    preset: str
    context: str
    policies: tuple[str, ...]
    with_overlays: bool  # pings + nominal arm + gaps (KeyStep, full context)
    dataset: str = "openhands"
    arm: str = "thermal_on"  # thermal_on | thermal_off | const


_STATE: dict = {}


def _init(cat: Catalog, datasets: dict, seed_tool: int, seed_policy: int) -> None:
    _STATE.update(cat=cat, datasets=datasets, seed_tool=seed_tool, seed_policy=seed_policy)


def _scenario(job: Job, dynamic: bool) -> Scenario:
    cat: Catalog = _STATE["cat"]
    dev = cat.devices[job.device].with_decode_tps(job.decode_tps)
    if job.arm == "thermal_off":
        dev = replace(dev, thermal=replace(dev.thermal, enabled=False))
    return Scenario(
        dev,
        cat.networks[job.network],
        cat.cloud[job.preset],
        cat.edge,
        cat.tool_time,
        dynamic=dynamic,
        context=job.context,
        seed=_STATE["seed_tool"],
    )


def run_job(job: Job) -> list[dict]:
    trajs = _STATE["datasets"][job.dataset]
    sc_dyn = _scenario(job, job.arm != "const")
    sc_nom = _scenario(job, False)
    preset = sc_dyn.cloud
    rows = []
    for spec in job.policies:
        pol = make_policy(spec, seed=_STATE["seed_policy"])
        for t in trajs:
            sim = simulate(t, pol, sc_dyn)
            r = summarize(t, sim)
            r.update(
                grid=job.grid,
                dataset=job.dataset,
                device=job.device,
                decode_tps=job.decode_tps,
                network=job.network,
                preset=job.preset,
                context=job.context,
                dynamic=job.arm != "const",
                arm=job.arm,
            )
            if job.with_overlays and spec in OVERLAY_POLICIES:
                nom = simulate(t, pol, sc_nom)
                cp = coupling(nom, sim)
                r.update(
                    nom_n_cold=cp.cold_nominal,
                    nom_total_s=nom.total_s,
                    cold_only_dynamic=cp.cold_only_dynamic,
                    cold_only_nominal=cp.cold_only_nominal,
                    nom_usd_c=sum(c.cost.usd_cache for c in nom.cloud_calls),
                )
                base = replay(sim, preset)
                p_opt, plans = optimal_pings(t, sim, preset, mode="best")
                rep_opt = replay(sim, preset, p_opt)
                rep_fix = replay(sim, preset, fixed_keepwarm(sim, preset))
                r.update(
                    usd_c_ping_opt=rep_opt.usd_cache,
                    n_pings_opt=rep_opt.n_pings,
                    ping_usd_opt=rep_opt.ping_usd,
                    cold_after_opt=rep_opt.cold_delegations,
                    ping_upload_bytes_opt=rep_opt.ping_upload_bytes,
                    n_gaps_refresh=sum(p.mode == "refresh" for p in plans),
                    n_gaps_extend=sum(p.mode == "extend" for p in plans),
                    usd_c_ping_fixed=rep_fix.usd_cache,
                    n_pings_fixed=rep_fix.n_pings,
                    ttft_deleg_noping=base.delegation_ttft_s,
                    ttft_deleg_opt=rep_opt.delegation_ttft_s,
                )
                gs = gaps(t, sim)
                r.update(
                    gap_s=sum(g.duration_s for g in gs),
                    gap_tool_s=sum(g.tool_s for g in gs),
                    gap_local_s=sum(g.local_s for g in gs),
                    n_gaps=len(gs),
                    n_gaps_over_ttl=sum(g.duration_s >= preset.ttl_s for g in gs),
                )
            rows.append(r)
    return rows


def k_grid(crit: dict, name: str = "k", device: str | None = None, arm: str = "thermal_on"):
    pc = crit["primary_config"]
    dev = device or pc["device_profile"]
    return [
        Job(
            name,
            dev,
            11.19 if (dev == "dgx_spark" and tps == 11) else float(tps),
            pc["network"],
            preset,
            pc["context_strategy"],
            tuple(K_POLICIES),
            True,
            arm=arm,
        )
        for preset, tps in itertools.product(pc["presets"], pc["local_decode_tps_grid"])
    ]


REGIME_TPS = [5.0, 8.0, 11.0, 15.0, 20.0, 30.0, 45.0, 60.0]


def regime_grid(crit: dict) -> list[Job]:
    pc = crit["primary_config"]
    return [
        Job("regime", "weak_edge", tps, net, preset, "full", ("keystep",), True, arm=arm)
        for arm, preset, net, tps in itertools.product(
            ["thermal_off", "thermal_on"], pc["presets"], ["good", "average", "weak"], REGIME_TPS
        )
    ]


def sens_grid(crit: dict) -> list[Job]:
    pc = crit["primary_config"]
    presets = pc["presets"]
    jobs = []
    # context strategies (Handoff-Tax baselines, component 8)
    for preset, ctx in itertools.product(
        presets, ["traj_drop", "compact_pre", "compact_suf", "cache_aware"]
    ):
        jobs.append(
            Job(
                "sens_ctx",
                "weak_edge",
                11.0,
                "average",
                preset,
                ctx,
                ("keystep", "sticky2_keystep"),
                False,
            )
        )
    # sensitivity-only preset (Opus 5.5 pricing) and robustness dataset
    jobs.append(
        Job(
            "sens_preset",
            "weak_edge",
            11.0,
            "average",
            "anthropic_5m_opus55",
            "full",
            ("keystep",),
            True,
        )
    )
    for preset in presets:
        jobs.append(
            Job(
                "robust_swesmith",
                "weak_edge",
                11.0,
                "average",
                preset,
                "full",
                tuple(K3_POLICIES),
                True,
                dataset="swesmith",
            )
        )
    return [replace(job, arm="thermal_off") for job in jobs]  # measured arm (D5)


def main() -> None:
    verify(ROOT / "configs" / "kill_criteria.yaml", ROOT / "configs" / "kill_criteria.lock.json")
    crit = yaml.safe_load((ROOT / "configs" / "kill_criteria.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="k", help="all, or comma-separated grid names")
    ap.add_argument("--limit", type=int, default=None, help="first N trajectories (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=None, help="output directory (default results/sweep)")
    args = ap.parse_args()
    cat = load_catalog(ROOT / "configs")
    data = {
        n: load_processed(ROOT / "data" / "processed", n)[: args.limit]
        for n in ("openhands", "swesmith")
    }
    grids = {
        "k": k_grid(crit),
        "k_measured": k_grid(crit, "k_measured", arm="thermal_off"),
        "k_const": k_grid(crit, "k_const", arm="const"),
        "k_laptop_on": k_grid(crit, "k_laptop_on", "laptop_16gb", "thermal_on"),
        "k_laptop_off": k_grid(crit, "k_laptop_off", "laptop_16gb", "thermal_off"),
        "k_dgx": k_grid(crit, "k_dgx", "dgx_spark", "thermal_on"),
        "regime": regime_grid(crit),
        "sens": sens_grid(crit),
    }
    todo = list(grids) if args.grid == "all" else args.grid.split(",")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    pc = crit["primary_config"]
    for g in todo:
        suffix = f"_limit{args.limit}" if args.limit else ""
        target = out_dir / f"{g}{suffix}.parquet"
        if target.exists():
            print(f"grid {g}: exists, skipped", flush=True)
            continue
        t0 = time.time()
        # ProcessPoolExecutor raises BrokenProcessPool if a worker dies (Pool would hang)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init,
            initargs=(cat, data, pc["tool_time_seed"], pc["policy_seed"]),
        ) as ex:
            parts = list(ex.map(run_job, grids[g]))
        df = pd.DataFrame([r for p in parts for r in p])
        df.to_parquet(target, index=False)
        print(f"grid {g}: {len(df)} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
