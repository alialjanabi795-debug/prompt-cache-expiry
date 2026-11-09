"""Settle K2 from the OpenRouter provider check (D9, D9-A1) and record the spend.

Reads results/calib/provider_check_openrouter.json (+ _supp_a1.json), applies the rule fixed
before the run (provider_check.or_settle_k2), writes the K2 status into results/verdicts.json
and the spend summary into results/calib/provider_check_spend.json. With
PROVIDER_CHECK_OPENROUTER_KEY set, the key's total usage (all OpenRouter spending, including
probes and the aborted first run) is read from /api/v1/key; the key itself is never printed.

Usage: python -I scripts/calib/settle_k2.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provider_check as pc

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "results" / "calib"


def k2_status_line(k2: dict) -> str:
    if k2["status"] == "FINAL":
        return (
            "FINAL (provider check confirmed via OpenRouter with pinned providers, D9; "
            "post-hoc amendments D9-A1 message-level GPT extend, D9-A2 extra GPT delays)"
        )
    return f"PROVISIONAL ({k2['reason']})"


def settle() -> dict | None:
    f = CAL / "provider_check_openrouter.json"
    if not f.exists():
        return None
    main = json.loads(f.read_text())
    supp: dict = {}
    for name in (
        "provider_check_openrouter_supp_a1.json",
        "provider_check_openrouter_supp_a2.json",
    ):
        s = CAL / name
        if s.exists():
            supp.update(json.loads(s.read_text())["evaluation"])
    k2 = pc.or_settle_k2(main["evaluation"], supp or None)
    k2["status_line"] = k2_status_line(k2)
    return k2


def main() -> None:
    k2 = settle()
    if k2 is None:
        raise SystemExit("no provider check results")
    vf = ROOT / "results" / "verdicts.json"
    v = json.loads(vf.read_text())
    v["K2_status"] = k2["status_line"]
    v["K2_provider_check"] = k2
    for g in [v["frozen"], *v["by_grid"].values()]:
        g["K2_status"] = k2["status_line"]
    vf.write_text(json.dumps(v, indent=2) + "\n")

    runs = {}
    for name in (
        "provider_check_openrouter.json",
        "provider_check_openrouter_supp_a1.json",
        "provider_check_openrouter_supp_a2.json",
    ):
        p = CAL / name
        if p.exists():
            runs[name] = json.loads(p.read_text())["spend"]["run_usd_from_generations"]
    smoke = CAL / "provider_check_openrouter_smoke_other_session.jsonl"
    if smoke.exists():  # D9-B: same key, other session
        runs[smoke.name] = round(
            sum(json.loads(x)["total_cost"] for x in smoke.read_text().splitlines() if x), 6
        )
    spend_f = CAL / "provider_check_spend.json"
    spend = json.loads(spend_f.read_text()) if spend_f.exists() else {}
    if os.environ.get("PROVIDER_CHECK_OPENROUTER_KEY"):
        u = pc.or_key_usage()
        spend["total_usd"] = u["usage"]
        spend["key_limit_usd"] = u["limit"]
    spend.update(
        budget_usd=pc.OR_TOTAL_BUDGET_USD,
        runs_usd=runs,
        other_usd=round(spend["total_usd"] - sum(runs.values()), 4),
        other_note="probes (max_tokens, tokens per word) and the aborted first run "
        "(per-word tag tripled the prompt; stopped after 14 logged requests); see "
        "provider_check_openrouter_run1_aborted.jsonl",
    )
    spend_f.write_text(json.dumps(spend, indent=2) + "\n")
    print(json.dumps({"K2": k2["status_line"], "spend": spend}, indent=2))


if __name__ == "__main__":
    main()
