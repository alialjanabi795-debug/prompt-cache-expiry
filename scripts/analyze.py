"""Verdicts and tables from results/sweep/*.parquet, exactly as pre-registered in
configs/kill_criteria.yaml (verified against the lock).

* FROZEN verdict: grid "k" (weak_edge, thermal on) with the frozen rules (K1-K7 + K4 + tool rule).
* The same rules on every other K grid (measured arm, constant speed, comparison profiles) -
  reported, not substituted (D5, D6).
* K6/K7 three-arm tables (D5), realism subset (D2), delegation-rate table (D3), regime-map table
  (D6), sensitivity summaries. K2 is PROVISIONAL until the provider check passes (D7).

Usage: python -I scripts/analyze.py [--sweep results/sweep] [--out results]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from cache_delegation.preregistration import (  # noqa: E402
    Estimate,
    holds_k4,
    pooled_ratio,
    ranking,
    verify,
)

B, SEED = 2000, 7
RATE_POLICIES = ["keystep", "keystep_L250", "random_0.4"]  # ~24% vs ~40% delegation (D3)
REALISM_MIN = 60.0
K_GRIDS = {  # grid -> (profile, arm)
    "k": ("weak_edge", "thermal_on"),
    "k_measured": ("weak_edge", "thermal_off"),
    "k_const": ("weak_edge", "const"),
    "k_laptop_on": ("laptop_16gb", "thermal_on"),
    "k_laptop_off": ("laptop_16gb", "thermal_off"),
    "k_dgx": ("dgx_spark", "thermal_on"),
}


def _boot_idx(n: int, b: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def diff_of_ratios(n1, d1, n2, d2, b, seed) -> Estimate:
    n1, d1, n2, d2 = (np.asarray(x, float) for x in (n1, d1, n2, d2))
    v = n1.sum() / d1.sum() - n2.sum() / d2.sum()
    idx = _boot_idx(len(n1), b, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = n1[idx].sum(1) / d1[idx].sum(1) - n2[idx].sum(1) / d2[idx].sum(1)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return Estimate(float(v), float(lo), float(hi))


def j_improvement(base: pd.DataFrame, early: dict[str, pd.DataFrame], b, seed) -> Estimate:
    """1 - min_s J(early_s), J = 0.5 U/U_base + 0.5 T/T_base (sums over trajectories)."""
    base = base.set_index("traj_id")
    u0 = (base.usd_c + base.edge_usd).to_numpy()
    t0 = base.total_s.to_numpy()
    us, ts = [], []
    for df in early.values():
        df = df.set_index("traj_id").loc[base.index]
        us.append((df.usd_c + df.edge_usd).to_numpy())
        ts.append(df.total_s.to_numpy())

    def imp(sel):
        js = [
            0.5 * u[sel].sum() / u0[sel].sum() + 0.5 * t[sel].sum() / t0[sel].sum()
            for u, t in zip(us, ts, strict=True)
        ]
        return 1.0 - min(js)

    v = imp(slice(None))
    bs = [imp(ix) for ix in _boot_idx(len(u0), b, seed)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return Estimate(float(v), float(lo), float(hi))


def verdicts_for(cells: dict, presets, grid, C: dict) -> dict:
    """Frozen verdict rules (K1-K7 with K4) over the given cells."""

    def k4(key: str) -> bool:
        return holds_k4({c: v[key] for c, v in cells.items()}, presets, grid)

    k3_raw_kill = any(
        all(not cells[(p, t)]["K3"] for p in presets if (p, t) in cells) for t in grid
    )
    return {
        "K1": "PASS" if k4("K1") else "KILL",
        "K2": ("STRONG" if k4("K2_strong") else "PASS") if k4("K2") else "KILL",
        "K3": "KILL" if (k3_raw_kill or not k4("K3")) else "PASS",
        "K3_raw_rule_kill": k3_raw_kill,
        "K5": "KEEP_COMPONENT_8" if k4("K5") else "DROP_COMPONENT_8",
        "K6": "COUPLING_STRONG" if k4("K6") else "COUPLING_WEAK_SENSITIVITY_ONLY",
        "K7": "KEEP_DELEGATE_EARLY" if k4("K7") else "DROP_DELEGATE_EARLY_CLAIM",
    }


def cell_metrics(sub: pd.DataFrame, C: dict) -> tuple[dict, dict]:
    ks = sub[sub.policy == "keystep"].sort_values("traj_id")
    k1 = pooled_ratio(ks.usd_c_deleg - ks.usd_c_inf_ttl_deleg, ks.usd_c_deleg, B, SEED)
    k2 = pooled_ratio(ks.usd_c - ks.usd_c_ping_opt, ks.usd_c, B, SEED)
    k5 = pooled_ratio(ks.n_cold, ks.n_deleg_noncompulsory, B, SEED)
    k6 = diff_of_ratios(
        ks.n_cold, ks.n_deleg_noncompulsory, ks.nom_n_cold, ks.n_deleg_noncompulsory, B, SEED
    )
    order_a = ranking(sub, C["K3"]["policies"], "calls_a")
    order_c = ranking(sub, C["K3"]["policies"], "usd_c")
    k7 = j_improvement(ks, {s: sub[sub.policy == s] for s in C["K7"]["policies"]}, B, SEED)
    m = {
        "K1": k1.value,
        "K1_lo": k1.lo,
        "K1_hi": k1.hi,
        "K2": k2.value,
        "K2_lo": k2.lo,
        "K2_hi": k2.hi,
        "K3_order_calls": ",".join(order_a),
        "K3_order_usd": ",".join(order_c),
        "K5": k5.value,
        "K5_lo": k5.lo,
        "K5_hi": k5.hi,
        "K6_pp": 100 * k6.value,
        "K6_lo_pp": 100 * k6.lo,
        "K6_hi_pp": 100 * k6.hi,
        "cold_share_const": float(ks.nom_n_cold.sum() / ks.n_deleg_noncompulsory.sum()),
        "K6_cold_only_dynamic": int(ks.cold_only_dynamic.sum()),
        "K7": k7.value,
        "K7_lo": k7.lo,
        "K7_hi": k7.hi,
        "tool_share": ks.gap_tool_s.sum() / ks.gap_s.sum(),
        "median_wall_min": float(ks.total_s.median() / 60),
        "coverage": float(ks.n_deleg.sum() / ks.n_steps.sum()),
        "usd_c_per_traj": float(ks.usd_c.mean()),
        "pings_per_traj": float(ks.n_pings_opt.mean()),
    }
    c = {
        "K1": k1.value >= C["K1"]["threshold"],
        "K2": k2.value >= C["K2"]["threshold"],
        "K2_strong": k2.value >= C["K2"]["strong_threshold"],
        "K3": order_a != order_c,
        "K5": k5.value >= C["K5"]["threshold"],
        "K6": abs(100 * k6.value) >= C["K6"]["threshold_abs_pp"],
        "K7": k7.value >= C["K7"]["threshold"],
        "borderline": {
            "K1": k1.straddles(C["K1"]["threshold"]),
            "K2": k2.straddles(C["K2"]["threshold"]),
            "K5": k5.straddles(C["K5"]["threshold"]),
            "K7": k7.straddles(C["K7"]["threshold"]),
        },
    }
    return m, c


def rate_rows(sub: pd.DataFrame, preset: str, tps: float, grid: str) -> list[dict]:
    out = []
    for pol in RATE_POLICIES:
        d = sub[sub.policy == pol].sort_values("traj_id")
        if d.empty or "usd_c_ping_opt" not in d or d.usd_c_ping_opt.isna().all():
            continue
        out.append(
            {
                "grid": grid,
                "preset": preset,
                "decode_tps": tps,
                "policy": pol,
                "coverage": d.n_deleg.sum() / d.n_steps.sum(),
                "cold_share": d.n_cold.sum() / d.n_deleg_noncompulsory.sum(),
                "K1_share": (d.usd_c_deleg - d.usd_c_inf_ttl_deleg).sum() / d.usd_c_deleg.sum(),
                "K2_saving": (d.usd_c - d.usd_c_ping_opt).sum() / d.usd_c.sum(),
                "median_wall_min": float(d.total_s.median() / 60),
            }
        )
    return out


def analyze_k_grid(df: pd.DataFrame, name: str, C: dict, presets) -> dict:
    grid = sorted(float(x) for x in df.decode_tps.unique())
    cells, rows, rates = {}, [], []
    for p in presets:
        for tps in grid:
            sub = df[(df.preset == p) & (df.decode_tps == tps)]
            if sub.empty:
                continue
            m, c = cell_metrics(sub, C)
            rows.append({"grid": name, "preset": p, "decode_tps": tps, **m})
            cells[(p, tps)] = c
            rates += rate_rows(sub, p, tps, name)
    verdicts = verdicts_for(cells, presets, grid, C)
    realistic = {
        (r["preset"], r["decode_tps"]) for r in rows if r["median_wall_min"] <= REALISM_MIN
    }
    kept = [t for t in grid if sum((p, t) in realistic for p in presets) >= 2]
    sub_cells = {c: v for c, v in cells.items() if c in realistic and c[1] in kept}
    verdicts["reporting_only_realistic_le_60min"] = (
        {
            **verdicts_for(sub_cells, presets, kept, C),
            "cells": sorted(f"{a}|{b:g}" for a, b in sub_cells),
        }
        if kept
        else "NOT_EVALUABLE: no tps value has >= 2 presets with median wall <= 60 min"
    )
    tool_cell = C["tool_rule"]["cell"]
    tool = [
        r["tool_share"]
        for r in rows
        if r["preset"] == tool_cell["preset"]
        and round(r["decode_tps"]) == tool_cell["local_decode_tps"]
    ]
    if tool:
        verdicts["tool_rule"] = {
            "tool_share": tool[0],
            "action": "ADD_RUN_TOOL_ON_EDGE"
            if tool[0] > C["tool_rule"]["threshold"]
            else "NO_ACTION",
        }
    verdicts["cells"] = {f"{p}|{t:g}": v for (p, t), v in cells.items()}
    return {"verdicts": verdicts, "rows": rows, "rates": rates}


def provider_status() -> str:
    """K2 status from the OpenRouter provider check (D9, D9-A1); see scripts/calib/settle_k2.py."""
    sys.path.insert(0, str(ROOT / "scripts" / "calib"))
    from settle_k2 import settle

    k2 = settle()
    return "PROVISIONAL (provider check not run)" if k2 is None else k2["status_line"]


def regime_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df.policy == "keystep"].groupby(["arm", "preset", "network", "decode_tps"])
    out = g.apply(
        lambda d: pd.Series(
            {
                "cold_share": d.n_cold.sum() / d.n_deleg_noncompulsory.sum(),
                "cold_share_const": d.nom_n_cold.sum() / d.n_deleg_noncompulsory.sum(),
                "K1_share": (d.usd_c_deleg - d.usd_c_inf_ttl_deleg).sum() / d.usd_c_deleg.sum(),
                "K2_saving": (d.usd_c - d.usd_c_ping_opt).sum() / d.usd_c.sum(),
                "median_wall_min": d.total_s.median() / 60,
                "usd_c_per_traj": d.usd_c.mean(),
                "usd_b_per_traj": d.usd_b.mean(),
                "n_traj": d.traj_id.nunique(),
            }
        ),
        include_groups=False,
    )
    return out.reset_index()


def sens_tables(sens: pd.DataFrame, kmeas: pd.DataFrame | None) -> dict:
    out = {}
    ctx = sens[sens.grid == "sens_ctx"]
    if kmeas is not None:
        base = kmeas[(kmeas.decode_tps == 11) & kmeas.policy.isin(["keystep", "sticky2_keystep"])]
        ctx = pd.concat([ctx, base.assign(grid="sens_ctx")], ignore_index=True)
    out["context"] = (
        ctx.groupby(["preset", "policy", "context"])
        .apply(
            lambda d: pd.Series(
                {
                    "usd_c_per_traj": d.usd_c.mean(),
                    "usd_b_per_traj": d.usd_b.mean(),
                    "prompt_tokens_per_traj": d.prompt_tokens.mean(),
                    "cold_share": d.n_cold.sum() / max(1, d.n_deleg_noncompulsory.sum()),
                    "median_wall_min": d.total_s.median() / 60,
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    for gname in ("sens_preset", "robust_swesmith"):
        d = sens[(sens.grid == gname) & (sens.policy == "keystep")]
        rows = []
        for p, dd in d.groupby("preset"):
            rows.append(
                {
                    "grid": gname,
                    "preset": p,
                    "cold_share": dd.n_cold.sum() / max(1, dd.n_deleg_noncompulsory.sum()),
                    "K1_share": (dd.usd_c_deleg - dd.usd_c_inf_ttl_deleg).sum()
                    / dd.usd_c_deleg.sum(),
                    "K2_saving": (dd.usd_c - dd.usd_c_ping_opt).sum() / dd.usd_c.sum(),
                }
            )
        sw = sens[sens.grid == gname]
        if gname == "robust_swesmith":
            for p, dd in sw.groupby("preset"):
                pols = ["random_0.2", "random_0.4", "keystep", "sticky2_keystep", "all_cloud"]
                ra, rc = ranking(dd, pols, "calls_a"), ranking(dd, pols, "usd_c")
                for r in rows:
                    if r["preset"] == p:
                        r["K3_rankings_differ"] = ra != rc
        out[gname] = pd.DataFrame(rows)
    return out


def main() -> None:
    verify(ROOT / "configs" / "kill_criteria.yaml", ROOT / "configs" / "kill_criteria.lock.json")
    crit = yaml.safe_load((ROOT / "configs" / "kill_criteria.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(ROOT / "results" / "sweep"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--input", default=None, help="analyze a single K-grid parquet as 'k'")
    args = ap.parse_args()
    C, presets = crit["criteria"], crit["primary_config"]["presets"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sweep = Path(args.sweep)
    files = (
        {"k": Path(args.input)}
        if args.input
        else {g: sweep / f"{g}.parquet" for g in K_GRIDS if (sweep / f"{g}.parquet").exists()}
    )
    k2_status = provider_status()
    results, all_rows, all_rates = {}, [], []
    for g, f in files.items():
        r = analyze_k_grid(pd.read_parquet(f), g, C, presets)
        prof, arm = K_GRIDS.get(g, ("weak_edge", "thermal_on"))
        r["verdicts"].update(profile=prof, arm=arm, K2_status=k2_status)
        results[g] = r["verdicts"]
        all_rows += r["rows"]
        all_rates += r["rates"]
    cells = pd.DataFrame(all_rows)
    cells.to_csv(out / "k_cells.csv", index=False)
    pd.DataFrame(all_rates).to_csv(out / "delegation_rate.csv", index=False)
    verdicts = {
        "frozen": results.get("k"),
        "K2_status": k2_status,
        "by_grid": results,
        "aggregation": crit["aggregation"]["estimator"],
    }
    # D5 three-arm tables for K6 and K7 (weak_edge)
    arms = [g for g in ("k_const", "k_measured", "k") if g in results]
    if arms:
        cols = [
            "grid",
            "preset",
            "decode_tps",
            "K5",
            "cold_share_const",
            "K6_pp",
            "K6_lo_pp",
            "K6_hi_pp",
            "K6_cold_only_dynamic",
            "K7",
            "K7_lo",
            "K7_hi",
            "median_wall_min",
        ]
        cells[cells.grid.isin(arms)][cols].to_csv(out / "k6_k7_three_arms.csv", index=False)
    if (sweep / "regime.parquet").exists():
        regime_table(pd.read_parquet(sweep / "regime.parquet")).to_csv(
            out / "regime_map.csv", index=False
        )
    if (sweep / "sens.parquet").exists():
        km = pd.read_parquet(files["k_measured"]) if "k_measured" in files else None
        for name, t in sens_tables(pd.read_parquet(sweep / "sens.parquet"), km).items():
            t.to_csv(out / f"sens_{name}.csv", index=False)
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=2, default=bool) + "\n")
    show = ["grid", "preset", "decode_tps", "K1", "K2", "K5", "K6_pp", "K7", "median_wall_min"]
    print(cells[show].round(3).to_string())
    for g, v in results.items():
        print(
            g,
            {k: v[k] for k in ("K1", "K2", "K3", "K5", "K6", "K7")},
            v.get("tool_rule", {}).get("action"),
        )
    print("K2 status:", k2_status)


if __name__ == "__main__":
    main()
