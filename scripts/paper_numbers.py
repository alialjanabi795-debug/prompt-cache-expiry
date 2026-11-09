r"""Compute every number and table reported in the paper from results/ (no hand-typed numbers).

Sources: results/*.csv, results/verdicts.json, results/figures/*.csv, results/calib/*.json,
results/timeline/, plus the frozen inputs the numbers are defined by (configs/*.yaml,
data/processed/manifest.json). Writes output/numbers.json (key -> value), output/numbers.tex
(the same values as LaTeX macros) and output/table_*.tex.

Usage: python -I scripts/paper_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "output"
P = {"anthropic_5m": "5m", "openai_30m": "30m", "anthropic_1h": "1h"}
TTL_LABEL = {"5m": "5 min", "30m": "30 min", "1h": "1 h"}
V: dict[str, str] = {}


def pct(x: float, d: int = 1) -> str:
    if x != 0 and abs(100 * x) < 0.5 * 10**-d:  # nonzero but rounds to zero
        return f"$<$\\,{0.5 * 10**-d:g}" if x > 0 else f"$>$\\,{-0.5 * 10**-d:g}"
    return f"{100 * x:.{d}f}"


def num(x: float, d: int = 1) -> str:
    s = f"{x:,.{d}f}"
    return s.replace(",", "{,}")


def put(key: str, val: str) -> None:
    assert key not in V, key
    V[key] = val


def cells() -> pd.DataFrame:
    c = pd.read_csv(RES / "k_cells.csv")
    c["p"] = c.preset.map(P)
    c["t"] = c.decode_tps.round().astype(int)
    # K2 repriced with the measured ping output cost (scripts/reprice_pings.py)
    r = pd.read_csv(RES / "k2_repriced.csv")
    c = c.merge(r[["grid", "preset", "decode_tps", "K2_new"]], on=["grid", "preset", "decode_tps"])
    c["K2_old"], c["K2"] = c.K2, c.K2_new
    return c


def setup_numbers() -> None:
    man = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text())
    oh = next(m for m in man if m["name"] == "openhands")
    sm = next(m for m in man if m["name"] == "swesmith")
    put("nTraj", num(oh["kept"], 0))
    put("nRows", num(oh["rows_in_file"], 0))
    put("nSmith", num(sm["kept"], 0))
    crit = yaml.safe_load((ROOT / "configs" / "kill_criteria.yaml").read_text())
    C = crit["criteria"]
    put("thrKone", pct(C["K1"]["threshold"], 0))
    put("thrKtwo", pct(C["K2"]["threshold"], 0))
    put("thrKtwoStrong", pct(C["K2"]["strong_threshold"], 0))
    put("thrKfive", pct(C["K5"]["threshold"], 0))
    put("thrKsix", num(C["K6"]["threshold_abs_pp"], 0))
    put("thrKseven", pct(C["K7"]["threshold"], 0))
    put("thrTool", pct(C["tool_rule"]["threshold"], 0))
    put("bootB", num(2000, 0) if "B = 2000" in crit["aggregation"]["ci"] else "??")
    put("keystepLong", num(crit["primary_config"]["keystep_long_read_tokens"], 0))
    tps = crit["primary_config"]["local_decode_tps_grid"]
    put("tpsGrid", ", ".join(str(t) for t in tps))
    tiers = yaml.safe_load((ROOT / "configs" / "tiers.yaml").read_text())["cloud_presets"]
    a5, a1, o3 = tiers["anthropic_5m"], tiers["anthropic_1h"], tiers["openai_30m"]
    put("opusIn", num(a5["price_in_per_mtok"], 0))
    put("opusOut", num(a5["price_out_per_mtok"], 0))
    put("gptIn", num(o3["price_in_per_mtok"], 0))
    put("gptOut", num(o3["price_out_per_mtok"], 0))
    put("wFive", num(a5["write_mult"], 2))
    put("wHour", num(a1["write_mult"], 0))
    put("wGpt", num(o3["write_mult"], 2))
    put("readMult", num(a5["read_mult"], 1))
    put("minCache", num(a5["min_cacheable_tokens"], 0))
    net = yaml.safe_load((ROOT / "configs" / "network.yaml").read_text())["profiles"]
    for name in ("good", "average", "weak"):
        key = {"good": "Good", "average": "Avg", "weak": "Weak"}[name]
        put(f"net{key}Mbps", num(net[name]["down_mbps"], 0))
        put(f"net{key}Rtt", num(1000 * net[name]["rtt_s"], 0))
    put("pingMargin", num(crit["primary_config"]["pings"]["safety_margin_s"], 0))
    fit = json.loads((RES / "calib" / "device_fit.json").read_text())
    put("prefDouble", num(fit["prefill_ctx_double_tokens"] / 1000, 1))
    put("decDouble", num(fit["decode_ctx_double_tokens"] / 1000, 1))
    put("pdRatio", num(fit["prefill_decode_ratio"], 1))
    put("rTwoPref", num(fit["prefill_fit_r2"], 3))
    put("rTwoDec", num(fit["decode_fit_r2"], 2))
    put("calibDecode", num(fit["decode_tps_at_ctx0"], 2))
    put("calibMinutes", num(fit["duration_min"], 0))
    dr = pd.read_csv(RES / "delegation_rate.csv")
    d = dr[(dr.grid == "k_measured") & (dr.preset == "anthropic_5m") & (dr.decode_tps == 11)]
    put("covKeystep", pct(d[d.policy == "keystep"].coverage.iloc[0]))
    put("covLtwofifty", pct(d[d.policy == "keystep_L250"].coverage.iloc[0]))


def artifact_numbers() -> None:
    lock = json.loads((ROOT / "configs" / "kill_criteria.lock.json").read_text())
    sha = lock["sha256"]  # split so the 64 hex digits can break across lines
    put("lockSha", "\\allowbreak ".join(sha[i : i + 16] for i in range(0, len(sha), 16)))
    put("lockFrozen", lock["frozen_at_utc"][:10])


def verdict_numbers() -> None:
    v = json.loads((RES / "verdicts.json").read_text())
    f = v["frozen"]
    names = {
        "KILL": "KILL",
        "DROP_COMPONENT_8": "drop component 8",
        "COUPLING_WEAK_SENSITIVITY_ONLY": "coupling weak",
        "DROP_DELEGATE_EARLY_CLAIM": "drop claim",
        "NO_ACTION": "no action",
    }
    for k, w in (
        ("K1", "one"),
        ("K2", "two"),
        ("K3", "three"),
        ("K5", "five"),
        ("K6", "six"),
        ("K7", "seven"),
    ):
        put(f"verdict{w}", names.get(f[k], f[k]))
    put("verdictTool", names[f["tool_rule"]["action"]])
    put("toolShare", pct(f["tool_rule"]["tool_share"]))
    st = v["K2_status"]
    put("ktwoStatus", st.split(" ")[0])
    put("ktwoStatusLower", st.split(" ")[0].lower())
    same = all(
        all(g[k] == f[k] for k in ("K1", "K2", "K3", "K5", "K6", "K7"))
        for g in v["by_grid"].values()
    )
    put("nConfigs", str(len(v["by_grid"])))
    put("verdictsIdentical", "identical" if same else "NOT identical")


def cell_numbers(c: pd.DataFrame) -> None:
    for grid, gk in (
        ("k", "F"),
        ("k_measured", "M"),
        ("k_const", "C"),
        ("k_dgx", "D"),
        ("k_laptop_off", "L"),
    ):
        for _, r in c[c.grid == grid].iterrows():
            key = f"{gk}{r.p}{r.t}"
            put(f"Kone{key}", pct(r.K1))
            put(f"Ktwo{key}", pct(r.K2))
            put(f"cold{key}", pct(r.K5))
            put(f"Ksix{key}", num(r.K6_pp, 1))
            put(f"Kseven{key}", pct(r.K7))
            put(f"usd{key}", num(r.usd_c_per_traj, 2))
            put(f"pings{key}", num(r.pings_per_traj, 1))
            put(f"wall{key}", num(r.median_wall_min, 0))
    m = c[c.grid == "k_measured"].set_index(["p", "t"])
    k = c[c.grid == "k_const"].set_index(["p", "t"])
    put(
        "couplingBill",
        pct(m.loc[("5m", 5)].usd_c_per_traj / k.loc[("5m", 5)].usd_c_per_traj - 1, 0),
    )
    five = m.loc["5m"]
    put("KoneMeasLo", pct(five.K1.min(), 0))
    put("KoneMeasHi", pct(five.K1.max(), 0))
    put("KtwoMeasLo", pct(five.K2.min(), 1))
    put("KtwoMeasHi", pct(five.K2.max(), 1))
    put("KtwoMeasOldLo", pct(five.K2_old.min(), 1))
    put("KtwoMeasOldHi", pct(five.K2_old.max(), 1))
    for t, r in five.iterrows():
        put(f"KtwoOldTwoM{t}", pct(r.K2_old, 2))
        put(f"KtwoNewTwoM{t}", pct(r.K2, 2))
    rj = json.loads((RES / "k2_repriced.json").read_text())
    put("ktwoMaxChange", num(rj["max_abs_change_pp"], 2))
    put("pingTokClaude", str(rj["ping_output_tokens"]["anthropic_5m"]))
    put("pingTokGpt", str(rj["ping_output_tokens"]["openai_30m"]))
    put(
        "ktwoRepricedVerdicts",
        "unchanged"
        if all(v["K2_old"] == v["K2_new"] for v in rj["verdicts"].values())
        else "CHANGED",
    )
    put("KsixMeasLo", num(five.K6_pp.min(), 0))
    put("KsixMeasHi", num(five.K6_pp.max(), 0))
    put("dgxTps", num(c[c.grid == "k_dgx"].decode_tps.median(), 1))
    # why the DGX Spark keeps short gaps: its configured speed profile (configs/device.yaml)
    sys.path.insert(0, str(ROOT / "src"))
    from cache_delegation.config import load_catalog

    dgx = load_catalog(ROOT / "configs").devices["dgx_spark"]
    put("dgxPdRatio", num(dgx.prefill_decode_ratio, 0))
    put("dgxPrefDouble", num(dgx.prefill_ctx_double_tokens / 1000, 0))
    f = c[c.grid == "k"]
    half = pd.concat(
        [
            (f[h] - f[lo]) / 2 * s
            for h, lo, s in (
                ("K1_hi", "K1_lo", 100),
                ("K2_hi", "K2_lo", 100),
                ("K5_hi", "K5_lo", 100),
                ("K6_hi_pp", "K6_lo_pp", 1),
                ("K7_hi", "K7_lo", 100),
            )
        ]
    )
    put("ciHalfMax", num(half.max(), 1))


def accounting_numbers() -> None:
    a = pd.read_csv(RES / "accounting_gap.csv")
    nt = a[a.grid != "k"]  # claims rest on arms without the assumed thermal model (D5)
    put("bOverCLo", num(nt.b_over_c.min(), 1))
    put("bOverCHi", num(nt.b_over_c.max(), 1))
    put("bOverCFrozenLo", num(a[a.grid == "k"].b_over_c.min(), 1))
    rp = pd.read_csv(RES / "k2_repriced.csv")  # ping schedules repriced with measured output
    m = rp[rp.grid == "k_measured"].set_index(["preset", "decode_tps"])
    r = m.loc[("anthropic_5m", 11.0)]
    put("fixedElevenUsd", num(r.usd_c_ping_fixed_new, 3))
    put("optElevenUsd", num(r.usd_c_ping_opt_new, 3))
    for t in (5.0, 11.0, 30.0):
        put(f"pingOptFive{int(t)}", num(m.loc[("anthropic_5m", t)].usd_c_ping_opt_new, 2))
        put(f"pingOptHour{int(t)}", num(m.loc[("anthropic_1h", t)].usd_c_ping_opt_new, 2))
    f4 = pd.read_csv(RES / "figures" / "fig4_policy_rankings.csv").set_index(["preset", "policy"])
    calls = f4.xs("all_cloud", level=1).calls / f4.xs("keystep", level=1).calls
    usd_b = f4.xs("all_cloud", level=1).usd_b / f4.xs("keystep", level=1).usd_b
    usd_c = f4.xs("all_cloud", level=1).usd_c / f4.xs("keystep", level=1).usd_c
    put("allCloudCalls", num(calls.mean(), 1))
    put("allCloudB", num(usd_b.mean(), 1))
    put("allCloudCLo", num(usd_c.min(), 1))
    put("allCloudCHi", num(usd_c.max(), 1))


def regime_numbers() -> None:
    x = pd.read_csv(RES / "figures" / "fig1_regime_map.csv")
    avg = x[x.network == "average"].set_index(["metric", "preset", "arm"]).crossing_decode_tps
    put("crossKoneFive", num(avg[("K1_share", "anthropic_5m", "thermal_off")], 0))
    put("crossKtwoFive", num(avg[("K2_saving", "anthropic_5m", "thermal_off")], 0))
    put("crossKoneThirty", num(avg[("K1_share", "openai_30m", "thermal_off")], 1))
    put("crossKtwoThirty", num(avg[("K2_saving", "openai_30m", "thermal_off")], 1))
    put("crossColdFive", num(avg[("cold_share", "anthropic_5m", "thermal_off")], 1))
    put("crossKoneThirtyTh", num(avg[("K1_share", "openai_30m", "thermal_on")], 0))
    put("crossKoneHourTh", num(avg[("K1_share", "anthropic_1h", "thermal_on")], 0))
    spread = x.groupby(["metric", "preset", "arm"]).crossing_decode_tps.agg(
        lambda s: s.max() - s.min()
    )
    put("netSpread", num(spread.max(), 1))
    rm = pd.read_csv(RES / "regime_map.csv")
    h = rm[(rm.preset == "anthropic_1h") & (rm.arm == "thermal_off")]
    put("hourKoneMax", pct(h.K1_share.max()))
    put("regimeTpsMax", num(rm.decode_tps.max(), 0))
    put("regimeTpsMin", num(rm.decode_tps.min(), 0))


def provider_numbers() -> list[str]:
    f = RES / "calib" / "provider_check_openrouter.json"
    s = RES / "calib" / "provider_check_openrouter_supp_a1.json"
    s2 = RES / "calib" / "provider_check_openrouter_supp_a2.json"
    supp2 = json.loads(s2.read_text()) if s2.exists() else None
    if not f.exists():
        return []
    js = json.loads(f.read_text())
    supp = json.loads(s.read_text()) if s.exists() else None
    cnt = js["counts"]
    reps = [g["repeats"] for k, g in js["plan"].items() if k != "C5_min1500"]
    put("prRepMin", str(min(reps)))
    put("prRepMax", str(max(reps)))
    extra = [x for x in (supp, supp2) if x]
    put("prRequests", num(cnt["requests"] + sum(x["counts"]["requests"] for x in extra), 0))
    put("prValid", num(cnt["valid"] + sum(x["counts"]["valid"] for x in extra), 0))
    allc = [cnt] + [x["counts"] for x in extra]
    put("prDiscarded", num(sum(c["requests"] - c["valid"] for c in allc), 0))
    put("prWrongProvider", num(sum(c["wrong_provider"] for c in allc), 0))
    spend = json.loads((RES / "calib" / "provider_check_spend.json").read_text())
    put("prSpend", num(spend["total_usd"], 2))
    put("prBudget", num(spend["budget_usd"], 0))
    sys.path.insert(0, str(ROOT / "scripts" / "calib"))
    import provider_check as pc

    put("hitShare", pct(pc.OR_HIT_SHARE, 0))
    recs = [r for r in js["records"] if r.get("valid") and r["step"] == 0]
    for prov, key in (("anthropic", "prefixClaude"), ("openai", "prefixGpt")):
        toks = sorted(
            r["prompt_tokens"] for r in recs if r["provider"] == prov and r["blocks"] == "X"
        )
        put(key, num(toks[len(toks) // 2] / 1000, 1))
    ev = js["evaluation"]
    A = ev["assumptions"]
    ex = A["openai_30m"]["extend_reads_prefix"]
    put("sameMsgHits", f"{0 if ex['status'] != 'CONFIRMED' else ex['n']}/{ex['n']}")
    rows = []
    label = {
        "ping_hit_15s": "ping 15 s after write: hit",
        "hit_270s": "read 270 s after write: hit",
        "miss_330s": "read 330 s after write: miss",
        "refresh_read_240s": "read at 240 s: hit",
        "refresh_hit_420s": "read at 420 s (180 s after refresh): hit",
        "generate_after_ping_hit": "generation after pings: hit",
        "extend_writes_suffix_only": "extend ping: prefix read, suffix written",
        "hit_600s_beyond_5m": "read 600 s after write: hit",
        "hit_3300s": "read 55 min after write: hit",
        "miss_3900s": "read 65 min after write: miss",
        "refresh_hit_3900s": "read at 65 min (55 min after refresh): hit",
        "hit_20s": "read 20 s after write: hit",
        "hit_300s": "read 300 s after write: hit",
        "hit_1500s": "read 25 min after write: hit",
        "miss_2100s": "read 35 min after write: miss",
        "refresh_read_1200s": "read at 20 min: hit",
        "refresh_hit_2700s": "read at 45 min (25 min after refresh): hit",
        "extend_reads_prefix": "extend (same message): prefix read",
    }
    plan_offsets = {o for g in js["plan"].values() for o, _, _ in g["steps"]}
    for k, txt in label.items():  # every delay named in a label must be a planned offset
        for tok, unit in (("s ", 1), ("min ", 60)):
            for part in txt.split(" at ")[-1:] + txt.split("read ")[-1:]:
                w = part.split(" ")
                if len(w) > 1 and w[0].isdigit() and (w[1] + " ").startswith(tok):
                    assert int(w[0]) * unit in plan_offsets, (k, txt)
    for p in ("anthropic_5m", "anthropic_1h", "openai_30m"):
        first = True
        for key, a in A[p].items():
            if "mult" in key:
                continue
            outc = a.get("outcomes")
            if outc is not None:
                good = outc.get(a["want"], 0)
                res = f"{good}/{a['n']}"
            else:
                res = f"{a['n']}" if a["status"] == "CONFIRMED" else f"0/{a['n']}"
                res = f"{a['n']}/{a['n']}" if a["status"] == "CONFIRMED" else res
            st = a["status"].lower()
            rows.append(f"{TTL_LABEL[P[p]] if first else ''} & {label[key]} & {res} & {st} \\\\")
            first = False
        if p == "openai_30m" and supp:
            e = supp["evaluation"]["O_extend_msg"]
            rows.append(
                f" & extend (new messages, A1): prefix read & {e['hits']}/{e['n']} & "
                f"{e['status'].lower()} \\\\"
            )
        if p == "openai_30m" and supp2:
            for g, key in (("O_hit600", "A"), ("O_hit1740", "B"), ("O_miss2400", "C")):
                off, _, want = supp2["plan"][g]["steps"][1]
                put(f"aTwoDelay{key}", num(off / 60, 0))
                txt = f"read {off / 60:.0f} min after write: {want}"
                e = supp2["evaluation"][g]
                good = e["outcomes"].get(e["want"], 0)
                rows.append(f" & {txt} (A2) & {good}/{e['n']} & {e['status'].lower()} \\\\")
                put(f"aTwo{g[2:]}", f"{good}/{e['n']}")
        if p != "openai_30m":
            rows.append("\\midrule")
    # price multipliers
    for p, wkey in (
        ("anthropic_5m", "write_mult_1.25"),
        ("anthropic_1h", "write_mult_2.0"),
        ("openai_30m", "write_mult_1.25"),
    ):
        w, r = A[p][wkey], A[p]["read_mult_0.10"]
        tag = P[p]
        put(f"wMeas{tag}", num(w["median"], 2) if w["n"] else "n/a")
        put(f"rMeas{tag}", num(r["median"], 3) if r["n"] else "n/a")
        put(f"wMeasN{tag}", str(w["n"]))
        put(f"rMeasN{tag}", str(r["n"]))
    put("priceErrMax", pct(ev["price_rel_err_max"], 1))
    po = ev["ping_output_tokens"]
    put("pingOutClaude", num(po["anthropic"]["mean_output_tokens"], 1))
    put("pingOutGpt", num(po["openai"]["mean_output_tokens"], 1))
    put("pingOutGptMax", num(po["openai"]["max_output_tokens"], 0))
    if supp:
        e = supp["evaluation"]["O_extend_msg"]
        put("aOneHits", f"{e['hits']}/{e['n']}")
        d = e["detail"][0]
        put("aOnePrefix", num(d["prefix"], 0))
        put("aOneWritten", num(d["written"], 0))
        c = supp["evaluation"]["C5_extend_msg"]
        put("aOneClaudeHits", f"{c['hits']}/{c['n']}")
    k2 = json.loads((RES / "verdicts.json").read_text()).get("K2_provider_check", {})
    put("ktwoCheck", k2.get("check", "n/a"))
    return rows


def timeline_numbers() -> None:
    """The introduction's example: one replayed trajectory (scripts/timeline.py)."""
    d = RES / "timeline"
    meta = json.loads((d / "meta.json").read_text())
    steps = pd.read_csv(d / "steps.csv")
    calls = pd.read_csv(d / "calls.csv")
    put("tlSteps", num(meta["n_steps"], 0))
    put("tlTps", num(meta["decode_tps"], 0))
    put("tlTtl", num(meta["ttl_s"] / 60, 0))
    put("tlMinutes", num(meta["total_s"] / 60, 0))
    put("tlLocal", num((steps.tier == "local").sum(), 0))
    put("tlDelegated", num(len(calls), 0))
    put("tlExpiredN", num(int(calls.expired.sum()), 0))
    llm_local = ((steps.t_llm_end_s - steps.t_start_s)[steps.tier == "local"]).sum()
    put("tlLocalShare", pct(llm_local / meta["total_s"], 0))
    c = calls[calls.expired].iloc[-1]  # the last expired call is the worked example
    put("tlGap", num(c.gap_s / 60, 1))
    put("tlPrompt", num(c.prompt_tokens, 0))
    put("tlExpTok", num(c.expired_tokens, 0))
    put("tlUsdExp", num(c.usd, 2))
    put("tlUsdWarm", num(c.usd_if_never_expired, 2))
    put("tlUsdNo", num(c.usd_without_cache, 2))
    put("tlRatio", num(c.usd / c.usd_if_never_expired, 1))
    put("tlStep", num(c.step_index + 1, 0))


POLICY_ROWS = [
    ("random_0.2", "random, 20\\% of steps"),
    ("random_0.4", "random, 40\\% of steps"),
    ("keystep", "key steps only"),
    ("sticky2_keystep", "key steps + next 2 steps"),
    ("all_cloud", "every step (all-cloud)"),
]


def write_policy_table() -> None:
    """Table I: call counts vs cost without and with the cache (measured device, 11 tok/s)."""
    f = pd.read_csv(RES / "figures" / "fig4_policy_rankings.csv").set_index(["preset", "policy"])
    lines = []
    for pol, label in POLICY_ROWS:
        a5, a1 = f.loc[("anthropic_5m", pol)], f.loc[("anthropic_1h", pol)]
        assert abs(a5.usd_b - a1.usd_b) < 1e-9  # same tokens and price without a cache
        lines.append(
            f"{label} & {num(a5.calls, 1)} & {num(a5.usd_b, 2)} & {num(a5.usd_c, 2)} & "
            f"{num(a1.usd_c, 2)} \\\\"
        )
    (OUT / "table_policies.tex").write_text(
        "% generated by paper/tools/make_numbers.py\n" + "\n".join(lines) + "\n\\bottomrule\n"
    )


def write_ttl_table(c: pd.DataFrame) -> None:
    """Table II: results by cache lifetime and device speed (measured device model)."""
    m = c[c.grid == "k_measured"]
    lines = []
    for p, name in (("5m", "5 min"), ("30m", "30 min"), ("1h", "1 h")):
        for t in (5, 11, 30):
            r = m[(m.p == p) & (m.t == t)].iloc[0]
            lines.append(
                f"{name if t == 5 else ''} & {t} & {pct(r.K1)} & {pct(r.K2)} & {pct(r.K5)} & "
                f"{num(r.median_wall_min, 0)} \\\\"
            )
        if p != "1h":
            lines.append("\\midrule")
    (OUT / "table_ttl.tex").write_text(
        "% generated by paper/tools/make_numbers.py\n" + "\n".join(lines) + "\n\\bottomrule\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = cells()
    setup_numbers()
    artifact_numbers()
    verdict_numbers()
    cell_numbers(c)
    accounting_numbers()
    regime_numbers()
    provider_numbers()  # numbers only; the provider table is not in this paper
    timeline_numbers()
    write_policy_table()
    write_ttl_table(c)
    out = [
        "% generated by paper/tools/make_numbers.py from results/ -- do not edit",
        r"\providecommand{\V}[1]{\ifcsname cd:#1\endcsname\csname cd:#1\endcsname"
        r"\else\textbf{??#1}\fi}",
    ]
    out += [f"\\expandafter\\def\\csname cd:{k}\\endcsname{{{v}}}" for k, v in sorted(V.items())]
    (OUT / "numbers.tex").write_text("\n".join(out) + "\n")
    (OUT / "numbers.json").write_text(
        json.dumps(
            {k: v.replace("{,}", ",").replace("\\allowbreak ", "") for k, v in sorted(V.items())},
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {len(V)} numbers")


if __name__ == "__main__":
    main()
