"""Real-API check of the cloud cache model (a few cents).

Verifies with real calls:
  A1  a ping with no generated output (max_tokens=0) is accepted and billed as prefill only
  A2  an identical ping within the TTL is a cache read (discounted), with no write
  A3  an "extend" ping (prefix + new suffix) reads the prefix and writes only the suffix
  A4  a real generation after pings reads the cache
  A5  TTL refresh-on-hit: entry X written at t0 and read at t0+240s is still a hit at t0+420s,
      while a control entry Y written at t0 and never touched is a miss at t0+420s (5-min TTL)
  O1-O3 the same for OpenAI (Responses API); if max_output_tokens=0 is rejected, the smallest
      accepted value is used and logged as a deviation.

Keys are read from PROVIDER_CHECK_ANTHROPIC_KEY / PROVIDER_CHECK_OPENAI_KEY (never from the
Claude Code session). The Anthropic base URL is pinned to https://api.anthropic.com.
Results: results/calib/provider_check.json

OpenRouter backend (deviation D9, ``--backend openrouter``): the same models through
https://openrouter.ai/api/v1/chat/completions with the provider pinned
({"order": [<provider>], "allow_fallbacks": false}); key from PROVIDER_CHECK_OPENROUTER_KEY.
Every request's usage (cached_tokens, cache_write_tokens) and its generation record
(/api/v1/generation: provider_name, native_tokens_cached, total_cost) are stored; a request not
served by the pinned provider at the default service tier is discarded and counted. The TTL
schedule (see ``OR_PLAN``) tests hit/miss on both sides of each TTL boundary and the read/write
price multipliers with independent repeats. Results: results/calib/provider_check_openrouter.json
(token counts and costs only; no prompt text).

Usage: python -I scripts/calib/provider_check.py [--skip-ttl] [--anthropic-model claude-opus-4-8]
       python -I scripts/calib/provider_check.py --backend openrouter [--budget-usd 4.5]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "calib" / "provider_check.json"


def corpus(tag: str, n_words: int = 2600) -> str:
    """Deterministic filler text (>= 2k tokens, above every model's minimum cacheable prefix)."""
    words = [f"{tag}-{i % 97}-{(i * 7919) % 1009}" for i in range(n_words)]
    return "Reference log " + tag + ":\n" + " ".join(words)


def anthropic_checks(model: str, skip_ttl: bool) -> dict:
    import anthropic

    client = anthropic.Anthropic(
        api_key=os.environ["PROVIDER_CHECK_ANTHROPIC_KEY"], base_url="https://api.anthropic.com"
    )
    res: dict = {"model": model, "steps": []}

    def call(name: str, blocks: list[dict], max_tokens: int) -> dict:
        t = time.time()
        try:
            r = client.messages.create(
                model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": blocks}]
            )
            rec = {
                "name": name,
                "t": t,
                "ok": True,
                "max_tokens": max_tokens,
                "stop_reason": r.stop_reason,
                "n_content": len(r.content),
                "usage": r.usage.model_dump(),
            }
        except anthropic.APIStatusError as e:
            rec = {
                "name": name,
                "t": t,
                "ok": False,
                "max_tokens": max_tokens,
                "status": e.status_code,
                "error": str(e)[:500],
            }
        res["steps"].append(rec)
        print(json.dumps(rec)[:400], flush=True)
        return rec

    def block(text: str, cache: bool) -> dict:
        b = {"type": "text", "text": text}
        if cache:
            b["cache_control"] = {"type": "ephemeral"}
        return b

    x, y, suffix = corpus("X"), corpus("Y"), corpus("S", 900)
    t0 = time.time()
    a1 = call("A1_ping_write_X", [block(x, True)], 0)
    if not a1["ok"]:
        call("A1b_ping_write_X_max1", [block(x, True)], 1)
        res["deviation"] = "max_tokens=0 rejected; used max_tokens=1"
    mt = 0 if a1["ok"] else 1
    call("A5_write_control_Y", [block(y, True)], mt)
    call("A2_ping_read_X", [block(x, True)], mt)
    call("A3_extend_ping_X+S", [block(x, True), block(suffix, True)], mt)
    call(
        "A4_generate_after_ping",
        [block(x, True), block(suffix, True), block("Reply with the single word OK.", False)],
        5,
    )
    if not skip_ttl:
        time.sleep(max(0.0, t0 + 240 - time.time()))
        call("A5_refresh_X_at_240s", [block(x, True)], mt)
        time.sleep(max(0.0, t0 + 420 - time.time()))
        call("A5_read_X_at_420s_expect_hit", [block(x, True)], mt)
        call("A5_read_Y_at_420s_expect_miss", [block(y, True)], mt)
    return res


def _openai_post(body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['PROVIDER_CHECK_OPENAI_KEY']}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def openai_checks(model: str) -> dict:
    res: dict = {"model": model, "steps": []}
    x, suffix = corpus("OX"), corpus("OS", 900)

    def call(name: str, text: str, max_out: int) -> dict:
        body = {
            "model": model,
            "input": text,
            "max_output_tokens": max_out,
            "prompt_cache_key": "phase0-cache-check",
            "store": False,
            "reasoning": {"effort": "none"},
        }
        t = time.time()
        status, js = _openai_post(body)
        rec = {
            "name": name,
            "t": t,
            "status": status,
            "max_output_tokens": max_out,
            "usage": js.get("usage"),
            "error": (js.get("error") or {}).get("message"),
        }
        res["steps"].append(rec)
        print(json.dumps(rec)[:400], flush=True)
        return rec

    r = call("O1_ping_write", x, 0)
    max_out = 0
    if r["status"] != 200:
        for m in (1, 16):
            r = call(f"O1b_ping_write_max{m}", x, m)
            if r["status"] == 200:
                max_out = m
                res["deviation"] = f"max_output_tokens=0 rejected; smallest accepted = {m}"
                break
    call("O2_ping_read", x, max_out)
    call("O3_extend_ping", x + "\n" + suffix, max_out)
    return res


# ------------------------------------------------------------------------------------------------
# OpenRouter backend (deviation D9). Same models, provider pinned, generation records audited.
# ------------------------------------------------------------------------------------------------
OR_BASE = "https://openrouter.ai/api/v1"
OR_OUT = ROOT / "results" / "calib" / "provider_check_openrouter.json"
OR_TOTAL_BUDGET_USD = 5.0  # hard cap for all OpenRouter spending on this key (D9)
OR_MODELS = {
    # prices: configs/tiers.yaml (= OpenRouter list prices for these endpoints, checked 2026-10-20)
    "anthropic": {
        "slug": "anthropic/claude-opus-4.8",
        "order": "anthropic",
        "provider_name": "Anthropic",
        "price_in": 5.0,
        "price_out": 25.0,
        "read_mult": 0.10,
        "write_mult": {"5m": 1.25, "1h": 2.0},
        "ping_max_tokens": 1,  # max_tokens=0 is ignored by OpenRouter (full generation)
    },
    "openai": {
        "slug": "openai/gpt-5.6-sol",
        "order": "openai",
        "provider_name": "OpenAI",
        "price_in": 4.0,
        "price_out": 20.0,
        "read_mult": 0.10,
        "write_mult": {"auto": 1.25},
        "ping_max_tokens": 16,  # OpenAI minimum; smaller values are raised by the router
    },
}
# 680 words = ~4.8k Claude / ~3.4k GPT tokens: above every minimum cacheable length (<= 4,096).
OR_PREFIX_WORDS, OR_SUFFIX_WORDS, OR_SHORT_WORDS = 680, 200, 215
OR_HIT_SHARE = 0.9  # hit = at least this share of the written prefix read from cache

# group -> (provider, ttl, repeats, steps); step = (offset_s from the chain's first request start,
# label, expectation, blocks, max_tokens or None for the ping minimum). Expectations:
# write = first contact; hit = cached >= 90% of the written prefix; miss = cached == 0;
# extend = prefix read, only the new suffix written.
OR_PLAN: dict[str, tuple[str, str, int, list[tuple]]] = {
    "C5_extend": (
        "anthropic",
        "5m",
        3,
        [
            (0, "write", "write", "X", None),
            (15, "ping", "hit", "X", None),
            (30, "extend", "extend", "XS", None),
            (45, "generate", "hit", "XSQ", 5),
        ],
    ),
    "C5_hit270": (
        "anthropic",
        "5m",
        6,
        [(0, "write", "write", "X", None), (270, "read", "hit", "X", None)],
    ),
    "C5_miss330": (
        "anthropic",
        "5m",
        6,
        [(0, "write", "write", "X", None), (330, "read", "miss", "X", None)],
    ),
    "C5_refresh": (
        "anthropic",
        "5m",
        6,
        [
            (0, "write", "write", "X", None),
            (240, "read", "hit", "X", None),
            (420, "read", "hit", "X", None),
        ],
    ),
    "C5_min1500": (
        "anthropic",
        "5m",
        2,
        [(0, "write", "write", "x", None), (20, "read", "hit", "x", None)],
    ),
    "C1h_hit3300": (
        "anthropic",
        "1h",
        5,
        [(0, "write", "write", "X", None), (3300, "read", "hit", "X", None)],
    ),
    "C1h_miss3900": (
        "anthropic",
        "1h",
        5,
        [(0, "write", "write", "X", None), (3900, "read", "miss", "X", None)],
    ),
    "C1h_refresh": (
        "anthropic",
        "1h",
        5,
        [
            (0, "write", "write", "X", None),
            (600, "read", "hit", "X", None),
            (3900, "read", "hit", "X", None),
        ],
    ),
    "O_quick": (
        "openai",
        "auto",
        5,
        [
            (0, "write", "write", "X", None),
            (20, "read", "hit", "X", None),
            (300, "read", "hit", "X", None),
        ],
    ),
    "O_extend": (
        "openai",
        "auto",
        3,
        [(0, "write", "write", "X", None), (20, "extend", "extend", "XS", None)],
    ),
    "O_hit1500": (
        "openai",
        "auto",
        5,
        [(0, "write", "write", "X", None), (1500, "read", "hit", "X", None)],
    ),
    "O_miss2100": (
        "openai",
        "auto",
        5,
        [(0, "write", "write", "X", None), (2100, "read", "miss", "X", None)],
    ),
    "O_refresh": (
        "openai",
        "auto",
        5,
        [
            (0, "write", "write", "X", None),
            (1200, "read", "hit", "X", None),
            (2700, "read", "hit", "X", None),
        ],
    ),
}
# Supplementary groups (amendment A1 to D9, added after the main run showed that GPT does not
# reuse a prefix when the suffix is appended inside the same user message): the agent case, where
# the next prompt appends whole messages (assistant reply + new user turn) to the previous prompt.
OR_PLAN_SUPP: dict[str, tuple[str, str, int, list[tuple]]] = {
    "O_extend_msg": (
        "openai",
        "auto",
        5,
        [(0, "write", "write", "X", None), (20, "extend", "extend", "X|A|S", None)],
    ),
    "C5_extend_msg": (
        "anthropic",
        "5m",
        3,
        [(0, "write", "write", "X", None), (20, "extend", "extend", "X|A|S", None)],
    ),
}
# Amendment A2 (requested by the PI during the main run): more GPT delays around the 30-min TTL.
OR_PLAN_SUPP.update(
    {
        "O_hit600": (
            "openai",
            "auto",
            4,
            [(0, "write", "write", "X", None), (600, "read", "hit", "X", None)],
        ),
        "O_hit1740": (
            "openai",
            "auto",
            4,
            [(0, "write", "write", "X", None), (1740, "read", "hit", "X", None)],
        ),
        "O_miss2400": (
            "openai",
            "auto",
            4,
            [(0, "write", "write", "X", None), (2400, "read", "miss", "X", None)],
        ),
    }
)
OR_MAIN_GROUPS = list(OR_PLAN)
OR_PLAN.update(OR_PLAN_SUPP)


def _or_request(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        OR_BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['PROVIDER_CHECK_OPENROUTER_KEY']}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except json.JSONDecodeError:
            return e.code, {}
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, {"error": {"message": str(e)[:200]}}


def or_key_usage() -> dict:
    status, js = _or_request("GET", "/key")
    d = js.get("data") or {}
    return {"status": status, "limit": d.get("limit"), "usage": d.get("usage")}


class _Budget:
    def __init__(self, cap: float) -> None:
        self.cap, self.spent, self.lock = cap, 0.0, threading.Lock()

    def reserve(self, est: float) -> bool:
        with self.lock:
            return self.spent + est <= self.cap

    def add(self, usd: float) -> None:
        with self.lock:
            self.spent += usd


def _or_spec(group: str, rep: int) -> dict:
    """Chain-unique texts: a random tag at the start keeps every chain's cache entries separate."""
    prov, ttl, _, _ = OR_PLAN[group]
    tag = f"{group}-{rep}-{os.urandom(4).hex()}"
    texts = {
        # the unique tag appears once (a per-word tag tripled the token count in an aborted run)
        "X": f"Session {tag}.\n" + corpus("W", OR_PREFIX_WORDS),
        "x": f"Session {tag}.\n" + corpus("W", OR_SHORT_WORDS),
        "S": corpus("S", OR_SUFFIX_WORDS),
        "Q": "Reply with the single word OK.",
    }
    return {"prov": prov, "ttl": ttl, "m": OR_MODELS[prov], "texts": texts}


def _or_messages(spec: dict, blocks: str) -> list[dict]:
    """Blocks in one user message; "|" starts a new message (roles alternate user/assistant)."""
    if "|" in blocks:
        msgs = []
        for i, part in enumerate(blocks.split("|")):
            if i % 2:
                msgs.append({"role": "assistant", "content": "OK."})
            else:
                msgs += _or_messages(spec, part)
        return msgs
    texts, prov, ttl = spec["texts"], spec["prov"], spec["ttl"]
    if prov == "openai":  # automatic caching: one string, the prefix is byte-identical
        return [{"role": "user", "content": "\n".join(texts[b] for b in blocks)}]
    cc = {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": "1h"}
    parts = []
    for b in blocks:
        p = {"type": "text", "text": texts[b]}
        if b != "Q":
            p["cache_control"] = cc
        parts.append(p)
    return [{"role": "user", "content": parts}]


def _or_chain(group: str, rep: int, start_at: float, budget: _Budget, records: list) -> None:
    prov, ttl, _, steps = OR_PLAN[group]
    spec = _or_spec(group, rep)
    m = spec["m"]
    time.sleep(max(0.0, start_at - time.time()))
    t0 = None
    for k, (offset, label, expect, blocks, max_tokens) in enumerate(steps):
        if t0 is not None:
            time.sleep(max(0.0, t0 + offset - time.time()))
        rec = {
            "group": group,
            "rep": rep,
            "step": k,
            "label": label,
            "expect": expect,
            "provider": prov,
            "ttl": ttl,
            "blocks": blocks,
            "planned_offset_s": offset,
        }
        mt = max_tokens if max_tokens is not None else m["ping_max_tokens"]
        body = {
            "model": m["slug"],
            "messages": _or_messages(spec, blocks),
            "max_tokens": mt,
            "provider": {"order": [m["order"]], "allow_fallbacks": False},
        }
        if prov == "openai":
            body["reasoning"] = {"effort": "none"}
        if not budget.reserve(0.08):
            rec.update(status=None, skipped="budget")
            records.append(rec)
            print(json.dumps(rec), flush=True)
            return
        for attempt in range(4):
            t_send = time.time()
            status, js = _or_request("POST", "/chat/completions", body)
            if status not in (0, 429, 500, 502, 503, 504):
                break
            time.sleep(3 * (attempt + 1))
        if t0 is None:
            t0 = t_send
        u = js.get("usage") or {}
        det = u.get("prompt_tokens_details") or {}
        cost = float(u.get("cost") or 0.0)
        budget.add(cost)
        rec.update(
            max_tokens=mt,
            status=status,
            t_send=t_send,
            offset_s=round(t_send - t0, 2),
            latency_s=round(time.time() - t_send, 2),
            gen_id=js.get("id"),
            resp_provider=js.get("provider"),
            service_tier=js.get("service_tier"),
            prompt_tokens=u.get("prompt_tokens"),
            completion_tokens=u.get("completion_tokens"),
            cached_tokens=det.get("cached_tokens"),
            cache_write_tokens=det.get("cache_write_tokens"),
            cost=cost,
            error=((js.get("error") or {}).get("message") or "")[:300] or None,
        )
        records.append(rec)
        print(
            json.dumps(
                {
                    k2: rec[k2]
                    for k2 in (
                        "group",
                        "rep",
                        "label",
                        "offset_s",
                        "status",
                        "resp_provider",
                        "prompt_tokens",
                        "cached_tokens",
                        "cache_write_tokens",
                        "completion_tokens",
                        "cost",
                    )
                }
            ),
            flush=True,
        )
        if status != 200:
            return  # the chain cannot continue meaningfully


def _or_fetch_generation(gen_id: str) -> dict:
    for attempt in range(10):
        status, js = _or_request("GET", f"/generation?id={gen_id}")
        if status == 200 and js.get("data"):
            d = js["data"]
            return {
                k: d.get(k)
                for k in (
                    "provider_name",
                    "native_tokens_prompt",
                    "native_tokens_cached",
                    "native_tokens_completion",
                    "native_tokens_reasoning",
                    "total_cost",
                    "service_tier",
                    "finish_reason",
                    "native_finish_reason",
                    "model",
                    "cache_discount",
                )
            }
        time.sleep(2 + 2 * attempt)
    return {"error": f"generation lookup failed (status {status})"}


def _or_validate(records: list) -> dict:
    """Keep a request only if the pinned provider served it at the default tier; once a step of a
    chain is invalid, the later steps of that chain are excluded (their cache state is unknown)."""
    counts = {
        "requests": 0,
        "valid": 0,
        "wrong_provider": 0,
        "non_default_tier": 0,
        "http_error": 0,
        "budget_skip": 0,
        "after_invalid_step": 0,
        "no_generation": 0,
    }
    by_chain: dict = {}
    for r in records:
        by_chain.setdefault((r["group"], r["rep"]), []).append(r)
    for chain in by_chain.values():
        chain.sort(key=lambda r: r["step"])
        broken = False
        for r in chain:
            counts["requests"] += 1
            g = r.get("generation") or {}
            reason = None
            if r.get("skipped"):
                reason = "budget_skip"
            elif broken:
                reason = "after_invalid_step"
            elif r.get("status") != 200:
                reason = "http_error"
            elif "provider_name" not in g:
                reason = "no_generation"
            elif g["provider_name"] != OR_MODELS[r["provider"]]["provider_name"]:
                reason = "wrong_provider"
            elif (g.get("service_tier") or r.get("service_tier")) != "default":
                reason = "non_default_tier"
            r["valid"], r["discard_reason"] = reason is None, reason
            counts["valid" if reason is None else reason] += 1
            broken = broken or reason is not None
    return counts


def _or_price(r: dict) -> dict:
    """Expected cost under configs/tiers.yaml and the measured multipliers it implies."""
    m = OR_MODELS[r["provider"]]
    wm = m["write_mult"][r["ttl"]]
    p, c = r["prompt_tokens"] or 0, r["cached_tokens"] or 0
    w, o = r["cache_write_tokens"] or 0, r["completion_tokens"] or 0
    base = (p - c - w) * m["price_in"] / 1e6
    out_usd = o * m["price_out"] / 1e6
    expected = (
        base + c * m["price_in"] * m["read_mult"] / 1e6 + w * m["price_in"] * wm / 1e6 + out_usd
    )
    cost = (r.get("generation") or {}).get("total_cost") or r["cost"]
    res = {
        "expected_usd": expected,
        "billed_usd": cost,
        "rel_err": (cost - expected) / expected if expected else None,
    }
    if c == 0 and w > 0:
        res["measured_write_mult"] = (cost - base - out_usd) / (w * m["price_in"] / 1e6)
    if w == 0 and c > 0:
        res["measured_read_mult"] = (cost - base - out_usd) / (c * m["price_in"] / 1e6)
    return res


def _or_outcome(r: dict, prefix_tokens: int) -> str:
    c = r["cached_tokens"] or 0
    if c == 0:
        return "miss"
    if c >= OR_HIT_SHARE * prefix_tokens:
        return "hit"
    return "partial"


def _agree(outcomes: list[str], want: str) -> str:
    n, k = len(outcomes), sum(o == want for o in outcomes)
    if n == 0:
        return "NOT_EVALUABLE"
    if k == n or (n >= 5 and k == n - 1):
        return "CONFIRMED"
    return "CONTRADICTED" if k <= n / 2 else "MIXED"


def or_evaluate(records: list) -> dict:
    """Score the assumptions the K2 ping overlay rests on (rules fixed before the run, D9)."""
    chains: dict = {}
    for r in records:
        chains.setdefault((r["group"], r["rep"]), []).append(r)
    obs: dict = {}  # (group, step) -> list of outcomes
    extend_ok: dict = {"anthropic": [], "openai": []}
    mults: dict = {}
    ping_out: dict = {"anthropic": [], "openai": []}
    price_err: list = []
    for (group, _), chain in chains.items():
        chain.sort(key=lambda r: r["step"])
        valid = [r for r in chain if r.get("valid")]
        if not valid or valid[0]["label"] != "write":
            continue
        w0 = valid[0]
        # cacheable prefix = what the first request wrote (OpenAI may not report writes)
        prefix = (w0["cache_write_tokens"] or 0) or (w0["prompt_tokens"] or 0)
        for r in valid:
            r["price"] = _or_price(r)
            if r["price"]["rel_err"] is not None:
                price_err.append(abs(r["price"]["rel_err"]))
            for key in ("measured_write_mult", "measured_read_mult"):
                if key in r["price"]:
                    mults.setdefault(f"{r['provider']}_{r['ttl']}_{key}", []).append(
                        r["price"][key]
                    )
            if (
                r["label"] in ("ping", "read", "extend")
                and r["max_tokens"] == OR_MODELS[r["provider"]]["ping_max_tokens"]
            ):
                ping_out[r["provider"]].append(r["completion_tokens"] or 0)
            if r["step"] == 0:
                continue
            out = _or_outcome(r, prefix)
            r["outcome"] = out
            if r["label"] == "extend":
                # prefix read, only the suffix written (Anthropic reports writes; OpenAI may not)
                okx = out == "hit"
                if r["provider"] == "anthropic":
                    suffix = (r["prompt_tokens"] or 0) - (r["cached_tokens"] or 0)
                    okx = okx and abs((r["cache_write_tokens"] or 0) - suffix) <= 0.05 * suffix + 8
                extend_ok[r["provider"]].append(okx)
            else:
                obs.setdefault((group, r["step"]), []).append(out)

    def a(group: str, step: int, want: str) -> dict:
        o = obs.get((group, step), [])
        return {
            "group": group,
            "step": step,
            "want": want,
            "n": len(o),
            "outcomes": {x: o.count(x) for x in sorted(set(o))},
            "status": _agree(o, want),
        }

    def mult_check(key: str, target: float, tol: float) -> dict:
        v = mults.get(key, [])
        if not v:
            return {"n": 0, "status": "NOT_EVALUABLE", "target": target}
        med = sorted(v)[len(v) // 2]
        return {
            "n": len(v),
            "median": med,
            "min": min(v),
            "max": max(v),
            "target": target,
            "status": "CONFIRMED" if abs(med - target) <= tol else "CONTRADICTED",
        }

    A: dict = {
        "anthropic_5m": {
            "ping_hit_15s": a("C5_extend", 1, "hit"),
            "hit_270s": a("C5_hit270", 1, "hit"),
            "miss_330s": a("C5_miss330", 1, "miss"),
            "refresh_read_240s": a("C5_refresh", 1, "hit"),
            "refresh_hit_420s": a("C5_refresh", 2, "hit"),
            "generate_after_ping_hit": a("C5_extend", 3, "hit"),
            "extend_writes_suffix_only": {
                "n": len(extend_ok["anthropic"]),
                "status": _agree(["y" if x else "n" for x in extend_ok["anthropic"]], "y"),
            },
            "write_mult_1.25": mult_check("anthropic_5m_measured_write_mult", 1.25, 0.03),
            "read_mult_0.10": mult_check("anthropic_5m_measured_read_mult", 0.10, 0.01),
        },
        "anthropic_1h": {
            "hit_600s_beyond_5m": a("C1h_refresh", 1, "hit"),
            "hit_3300s": a("C1h_hit3300", 1, "hit"),
            "miss_3900s": a("C1h_miss3900", 1, "miss"),
            "refresh_hit_3900s": a("C1h_refresh", 2, "hit"),
            "write_mult_2.0": mult_check("anthropic_1h_measured_write_mult", 2.0, 0.05),
            "read_mult_0.10": mult_check("anthropic_1h_measured_read_mult", 0.10, 0.01),
        },
        "openai_30m": {
            "hit_20s": a("O_quick", 1, "hit"),
            "hit_300s": a("O_quick", 2, "hit"),
            "hit_1500s": a("O_hit1500", 1, "hit"),
            "miss_2100s": a("O_miss2100", 1, "miss"),
            "refresh_read_1200s": a("O_refresh", 1, "hit"),
            "refresh_hit_2700s": a("O_refresh", 2, "hit"),
            "extend_reads_prefix": {
                "n": len(extend_ok["openai"]),
                "status": _agree(["y" if x else "n" for x in extend_ok["openai"]], "y"),
            },
            "write_mult_1.25": mult_check("openai_auto_measured_write_mult", 1.25, 0.03),
            "read_mult_0.10": mult_check("openai_auto_measured_read_mult", 0.10, 0.01),
        },
        "informational": {
            "anthropic_min_cacheable_1500tok": a("C5_min1500", 1, "hit"),
        },
    }
    ping = {
        p: {
            "n": len(v),
            "mean_output_tokens": (sum(v) / len(v)) if v else None,
            "max_output_tokens": max(v) if v else None,
            "max_tokens_sent": OR_MODELS[p]["ping_max_tokens"],
        }
        for p, v in ping_out.items()
    }
    return {
        "assumptions": A,
        "ping_output_tokens": ping,
        "price_rel_err_max": max(price_err) if price_err else None,
        "price_rel_err_median": sorted(price_err)[len(price_err) // 2] if price_err else None,
    }


def or_evaluate_extend(records: list) -> dict:
    """Supplementary groups (A1, A2): hit = cached >= OR_HIT_SHARE of the prefix, miss = 0."""
    _or_validate(records)
    res: dict = {}
    for group in sorted({r["group"] for r in records}):
        chains: dict = {}
        for r in records:
            if r["group"] == group and r.get("valid"):
                chains.setdefault(r["rep"], {})[r["step"]] = r
        outs, want, detail = [], None, []
        for c in chains.values():
            if 0 in c and 1 in c:
                prefix = (c[0]["cache_write_tokens"] or 0) or (c[0]["prompt_tokens"] or 0)
                out = _or_outcome(c[1], prefix)
                want = "miss" if c[1]["expect"] == "miss" else "hit"
                outs.append(out)
                detail.append(
                    {
                        "prefix": prefix,
                        "cached": c[1]["cached_tokens"],
                        "written": c[1]["cache_write_tokens"],
                        "prompt": c[1]["prompt_tokens"],
                        "offset_s": c[1]["offset_s"],
                    }
                )
        res[group] = {
            "n": len(outs),
            "want": want,
            "hits": outs.count("hit"),
            "outcomes": {x: outs.count(x) for x in sorted(set(outs))},
            "status": _agree(outs, want) if want else "NOT_EVALUABLE",
            "detail": detail,
        }
    return res


def or_settle_k2(ev: dict, supp: dict | None = None) -> dict:
    """K2 rule fixed before the run (docs/deviations.md D9).

    The frozen K2 verdict comes from the simulated ping overlay. The check decides whether that
    overlay may be trusted:
      * TTL semantics (hit inside, miss outside, refresh on hit, 1 h outlives 5 min) are what
        creates or removes expiry; if any is CONTRADICTED/MIXED the overlay must be re-run with the
        measured behaviour -> K2 stays PROVISIONAL.
      * Price multipliers off by more than the tolerance -> PROVISIONAL (re-price).
      * A ping that must bill a few output tokens (max_tokens=0 not available) only raises the
        cost of every ping schedule while C_noping is unchanged, so the optimal saving can only
        fall; a KILL therefore stands, a PASS would need a re-run.
    """
    ttl_keys = {
        "anthropic_5m": [
            "hit_270s",
            "miss_330s",
            "refresh_read_240s",
            "refresh_hit_420s",
            "ping_hit_15s",
            "generate_after_ping_hit",
            "extend_writes_suffix_only",
        ],
        "anthropic_1h": ["hit_600s_beyond_5m", "hit_3300s", "miss_3900s", "refresh_hit_3900s"],
        "openai_30m": [
            "hit_20s",
            "hit_300s",
            "hit_1500s",
            "miss_2100s",
            "refresh_read_1200s",
            "refresh_hit_2700s",
            "extend_reads_prefix",
        ],
    }
    price_keys = {
        "anthropic_5m": ["write_mult_1.25", "read_mult_0.10"],
        "anthropic_1h": ["write_mult_2.0", "read_mult_0.10"],
        "openai_30m": ["write_mult_1.25", "read_mult_0.10"],
    }
    A = ev["assumptions"]
    failed = [
        f"{p}.{k}={A[p][k]['status']}"
        for p, ks in ttl_keys.items()
        for k in ks
        if A[p][k]["status"] != "CONFIRMED"
    ]
    failed_price = [
        f"{p}.{k}={A[p][k]['status']}"
        for p, ks in price_keys.items()
        for k in ks
        if A[p][k]["status"] != "CONFIRMED"
    ]
    out_dev = {
        p: v["mean_output_tokens"]
        for p, v in ev["ping_output_tokens"].items()
        if v["mean_output_tokens"]
    }
    amended = []
    supp = dict(supp or {})
    for key, want in (("O_hit600", "hit"), ("O_hit1740", "hit"), ("O_miss2400", "miss")):
        if key in supp:  # A2: a GPT entry that dies early (or lives on) blocks a FINAL verdict
            st = supp[key]["status"]
            amended.append(f"A2 openai_30m {key} ({want}) = {st}")
            if st != "CONFIRMED":
                failed.append(f"openai_30m.{key}={st} (A2)")
    if supp and "openai_30m.extend_reads_prefix=CONTRADICTED" in failed:
        # A1 (post hoc): the within-message test does not match how an agent's prompt grows;
        # the message-level test decides, and the substitution is reported.
        st = (supp.get("O_extend_msg") or {}).get("status")
        if st == "CONFIRMED":
            failed.remove("openai_30m.extend_reads_prefix=CONTRADICTED")
            amended.append("openai_30m.extend_reads_prefix replaced by message-level test (A1)")
    if failed or failed_price:
        return {
            "status": "PROVISIONAL",
            "check": "CONTRADICTED (re-run ping overlay)",
            "reason": "assumptions not confirmed: " + ", ".join(failed + failed_price),
            "ping_output_tokens_deviation": out_dev,
            "amendments": amended,
        }
    return {
        "status": "FINAL",
        "check": "CONFIRMED",
        "reason": (
            "TTL semantics and cache prices confirmed for all three presets; pings bill "
            + ", ".join(f"{p} {v:.1f}" for p, v in out_dev.items())
            + " output tokens instead of 0, which can only lower the ping saving"
        )
        if out_dev
        else "all assumptions confirmed",
        "ping_output_tokens_deviation": out_dev,
        "amendments": amended,
    }


def openrouter_checks(args: argparse.Namespace) -> None:
    import random

    if not os.environ.get("PROVIDER_CHECK_OPENROUTER_KEY"):
        raise SystemExit("PROVIDER_CHECK_OPENROUTER_KEY not set")
    key0 = or_key_usage()
    already = float(key0.get("usage") or 0.0)
    cap = min(args.budget_usd, OR_TOTAL_BUDGET_USD - already - 0.25)
    groups = args.groups.split(",") if args.groups else OR_MAIN_GROUPS
    groups = [g for g in groups if not (args.skip_ttl and max(s[0] for s in OR_PLAN[g][3]) > 60)]
    chains = [(g, r) for g in groups for r in range(OR_PLAN[g][2])]
    random.Random(20261020).shuffle(chains)  # interleave groups in time
    print(f"key usage before: ${already:.4f}; run cap ${cap:.2f}; {len(chains)} chains", flush=True)
    budget, records = _Budget(cap), []
    start = time.time() + 2
    threads = [
        threading.Thread(target=_or_chain, args=(g, r, start + 2.0 * i, budget, records))
        for i, (g, r) in enumerate(chains)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in records:
        if r.get("gen_id"):
            r["generation"] = _or_fetch_generation(r["gen_id"])
    for r in records:
        r["t_send"] = round(r.get("t_send", 0) - start, 2) if r.get("t_send") else None
    counts = _or_validate(records)
    ev = or_evaluate(records)
    key1 = or_key_usage()
    out = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
        "backend": "openrouter",
        "endpoint": OR_BASE + "/chat/completions",
        "models": {
            p: {k: m[k] for k in ("slug", "order", "provider_name", "ping_max_tokens")}
            for p, m in OR_MODELS.items()
        },
        "provider_routing": {
            p: {"order": [m["order"]], "allow_fallbacks": False} for p, m in OR_MODELS.items()
        },
        "max_tokens_0_probe": "accepted but ignored by OpenRouter (full generation billed); "
        "smallest honoured value: anthropic 1, openai 16",
        "plan": {
            g: {"provider": v[0], "ttl": v[1], "repeats": v[2], "steps": [s[:3] for s in v[3]]}
            for g, v in OR_PLAN.items()
        },
        "counts": counts,
        "spend": {
            "run_usd_from_usage": round(budget.spent, 6),
            "run_usd_from_generations": round(
                sum((r.get("generation") or {}).get("total_cost") or 0 for r in records), 6
            ),
            "key_usage_before": already,
            "key_usage_after": key1.get("usage"),
        },
        "evaluation": ev,
        "k2": or_settle_k2(ev),
        "groups": groups,
        "records": sorted(records, key=lambda r: (r["group"], r["rep"], r["step"])),
    }
    if args.groups:  # supplementary run: no K2 decision here (see or_settle_k2)
        out.pop("k2")
        out["evaluation"] = or_evaluate_extend(records)
    dest = Path(args.out) if args.out else OR_OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(
        json.dumps({k: out[k] for k in ("counts", "spend", "evaluation") if k in out}, indent=2)[
            :3000
        ]
    )
    print(f"wrote {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["direct", "openrouter"], default="direct")
    ap.add_argument("--anthropic-model", default="claude-opus-4-8")
    ap.add_argument("--openai-model", default="gpt-5.6-sol")
    ap.add_argument("--skip-ttl", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=4.5)
    ap.add_argument("--groups", default=None, help="comma-separated OR_PLAN groups (supplementary)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.backend == "openrouter":
        openrouter_checks(args)
        return
    out: dict = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if os.environ.get("PROVIDER_CHECK_ANTHROPIC_KEY"):
        out["anthropic"] = anthropic_checks(args.anthropic_model, args.skip_ttl)
    else:
        out["anthropic"] = "skipped: PROVIDER_CHECK_ANTHROPIC_KEY not set"
    if os.environ.get("PROVIDER_CHECK_OPENAI_KEY"):
        out["openai"] = openai_checks(args.openai_model)
    else:
        out["openai"] = "skipped: PROVIDER_CHECK_OPENAI_KEY not set"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
