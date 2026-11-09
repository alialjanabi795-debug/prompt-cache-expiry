"""Reprice the K2 keep-warm ping schedules with the measured ping output cost (D9).

The sweep priced pings as prefill-only (max_tokens=0). Through OpenRouter a ping bills output
tokens (results/calib/provider_check_openrouter.json): Claude 1 token, GPT up to 16 (we use the
maximum). Each optimal (and fixed) ping schedule is repriced with
    extra_usd = pings_per_traj * output_tokens * price_out
and K2 is recomputed as (C_noping - C_ping - extra) / C_noping (pooled over the same 2,000
trajectories, so per-trajectory means suffice). Re-optimizing with the extra cost could only
recover part of it, so the true optimum lies between the old and the repriced saving; the
repriced value is the conservative one. K4 is re-applied to the repriced cells.

Writes results/k2_repriced.csv and results/k2_repriced.json.
Usage: python -I scripts/reprice_pings.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main() -> None:
    tiers = yaml.safe_load((ROOT / "configs" / "tiers.yaml").read_text())["cloud_presets"]
    crit = yaml.safe_load((ROOT / "configs" / "kill_criteria.yaml").read_text())
    thr = crit["criteria"]["K2"]["threshold"]
    strong = crit["criteria"]["K2"]["strong_threshold"]
    presets = crit["primary_config"]["presets"]
    tps_grid = crit["primary_config"]["local_decode_tps_grid"]
    pc = json.loads((RES / "calib" / "provider_check_openrouter.json").read_text())
    po = pc["evaluation"]["ping_output_tokens"]
    out_tok = {p: po[tiers[p]["provider"]]["max_output_tokens"] for p in presets}

    c = pd.read_csv(RES / "k_cells.csv")
    a = pd.read_csv(RES / "accounting_gap.csv").rename(columns={"tps": "decode_tps"})
    c = c.merge(
        a[["grid", "preset", "decode_tps", "usd_c_ping_opt", "usd_c_ping_fixed", "pings_fixed"]],
        on=["grid", "preset", "decode_tps"],
        how="left",
    )
    c["ping_output_tokens"] = c.preset.map(out_tok)
    c["usd_per_ping"] = c.ping_output_tokens * c.preset.map(
        lambda p: tiers[p]["price_out_per_mtok"] / 1e6
    )
    c["extra_usd_opt"] = c.pings_per_traj * c.usd_per_ping
    c["K2_old"] = c.K2
    c["K2_new"] = c.K2 - c.extra_usd_opt / c.usd_c_per_traj
    c["K2_change_pp"] = 100 * (c.K2_new - c.K2_old)
    c["usd_c_ping_opt_new"] = c.usd_c_ping_opt + c.extra_usd_opt
    c["usd_c_ping_fixed_new"] = c.usd_c_ping_fixed + c.pings_fixed * c.usd_per_ping
    cols = [
        "grid",
        "preset",
        "decode_tps",
        "usd_c_per_traj",
        "pings_per_traj",
        "ping_output_tokens",
        "usd_per_ping",
        "extra_usd_opt",
        "K2_old",
        "K2_new",
        "K2_change_pp",
        "usd_c_ping_opt",
        "usd_c_ping_opt_new",
        "pings_fixed",
        "usd_c_ping_fixed",
        "usd_c_ping_fixed_new",
    ]
    c[cols].to_csv(RES / "k2_repriced.csv", index=False)

    verdicts = {}
    for g, d in c.groupby("grid"):
        res = {}
        for col in ("K2_old", "K2_new"):
            hold = all(
                (
                    d[(d.decode_tps.round() == t)].set_index("preset")[col].reindex(presets) >= thr
                ).sum()
                >= 2
                for t in tps_grid
                if (d.decode_tps.round() == t).any()
            )
            res[col] = "HOLDS" if hold else "KILL"
        res["strong_cells_new"] = int((d.K2_new >= strong).sum())
        verdicts[g] = res
    summary = {
        "ping_output_tokens": out_tok,
        "max_abs_change_pp": float(c.K2_change_pp.abs().max()),
        "verdicts": verdicts,
        "note": "repriced optimal schedule (conservative); re-optimization lies in between",
    }
    (RES / "k2_repriced.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    show = c[c.grid.isin(["k", "k_measured"])][
        ["grid", "preset", "decode_tps", "pings_per_traj", "K2_old", "K2_new", "K2_change_pp"]
    ]
    print(show.round(5).to_string())


if __name__ == "__main__":
    main()
