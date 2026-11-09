"""Success-neutral keep-warm pings (component 7) and the fixed server-style keep-warm baseline.

A ping is a ``max_tokens=0`` request whose prompt is a prefix of the next delegation's prompt. It
runs prefill only: it refreshes the TTL of the entry it reads and (in "extend" mode) writes the new
transcript suffix. It changes no agent decision, so pings are overlaid on the fixed timeline of a
simulation run and evaluated by an exact replay of the cloud cache. Pings are asynchronous: they do
not move the agent timeline (the TTFT gain of a warm delegation is reported, not fed back, which is
conservative for cost).

Gaps between consecutive cloud calls are independent under the full-history strategy: every
delegation rewrites/refreshes its full prompt, so the cache state after a delegation does not depend
on earlier pings. The optimizer therefore solves each gap separately and keeps the cheapest of
{no pings, refresh chain, extend chain (DP)}; "no pings" is always feasible, so the optimized
schedule never costs more than the baseline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .accounting import cloud_call_cost, cloud_ttft_s
from .cache import CacheResult, Prompt, PromptCache
from .config import CloudPreset
from .context import TranscriptIndex
from .simulator import SimResult
from .types import Trajectory

DEFAULT_MARGIN_S = 10.0


@dataclass(frozen=True, slots=True)
class Ping:
    t: float
    prompt: Prompt


@dataclass
class ReplayCall:
    step_index: int
    kind: str
    t: float
    result: CacheResult
    usd_cache: float
    usd_nocache: float
    usd_inf_ttl: float
    upload_bytes: int
    ttft_s: float


@dataclass
class ReplayResult:
    calls: list[ReplayCall] = field(default_factory=list)

    def _sum(self, kinds: Sequence[str], attr: str) -> float:
        return sum(getattr(c, attr) for c in self.calls if c.kind in kinds)

    @property
    def usd_cache(self) -> float:  # everything billed, pings included
        return self._sum(("delegation", "summary", "ping"), "usd_cache")

    @property
    def ping_usd(self) -> float:
        return self._sum(("ping",), "usd_cache")

    @property
    def n_pings(self) -> int:
        return sum(1 for c in self.calls if c.kind == "ping")

    @property
    def ping_upload_bytes(self) -> int:
        return int(self._sum(("ping",), "upload_bytes"))

    @property
    def cold_delegations(self) -> int:
        return sum(1 for c in self.calls if c.kind == "delegation" and c.result.cold)

    @property
    def delegation_ttft_s(self) -> float:
        return self._sum(("delegation",), "ttft_s")


def replay(sim: SimResult, preset: CloudPreset, pings: Sequence[Ping] = ()) -> ReplayResult:
    cache = PromptCache(
        ttl_s=preset.ttl_s,
        min_tokens=preset.min_cacheable_tokens,
        static_breakpoint=preset.cache_static_prefix,
    )
    events: list[tuple[float, int, object]] = [(c.t_start, 0, c) for c in sim.cloud_calls]
    events += [(p.t, 1, p) for p in pings]
    events.sort(key=lambda e: (e[0], e[1]))
    out = ReplayResult()
    for t, is_ping, ev in events:
        if is_ping:
            assert isinstance(ev, Ping)
            prompt, n_out, kind, idx = ev.prompt, preset.ping_output_tokens, "ping", -1
        else:
            prompt, n_out, kind, idx = ev.prompt, ev.out_tokens, ev.kind, ev.step_index
        res = cache.request(prompt, t)
        cost = cloud_call_cost(res, n_out, preset, is_delegation=(kind == "delegation"))
        out.calls.append(
            ReplayCall(
                idx,
                kind,
                t,
                res,
                cost.usd_cache,
                cost.usd_nocache,
                cost.usd_cache_inf_ttl,
                prompt.nbytes,
                cloud_ttft_s(res, preset),
            )
        )
    return out


def fixed_keepwarm(
    sim: SimResult, preset: CloudPreset, margin_s: float = DEFAULT_MARGIN_S
) -> list[Ping]:
    """Server-style keep-alive: after every cloud call, re-send its prompt every (TTL - margin)
    until the next cloud call or the end of the session (no knowledge of the future)."""
    calls = sim.cloud_calls
    step = preset.ttl_s - margin_s
    pings: list[Ping] = []
    for j, c in enumerate(calls):
        until = calls[j + 1].t_start if j + 1 < len(calls) else sim.total_s
        n = 1
        while c.t_start + n * step < until:
            pings.append(Ping(c.t_start + n * step, c.prompt))
            n += 1
    return pings


# ------------------------------------------------------------------------------------------------
# Offline optimal schedule
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GapPlan:
    call_index: int  # index into sim.cloud_calls of the call that ends the gap
    mode: str  # "none" | "refresh" | "extend"
    pings: tuple[Ping, ...]
    planned_saving_usd: float


def _available_steps(step_ends: Sequence[float], t: float, upto: int) -> int:
    """Number of completed steps (<= upto) whose observation is available at time t."""
    m = 0
    while m < upto and step_ends[m] <= t:
        m += 1
    return m


def optimal_pings(
    traj: Trajectory,
    sim: SimResult,
    preset: CloudPreset,
    mode: str = "best",
    margin_s: float = DEFAULT_MARGIN_S,
    grid_s: float | None = None,
) -> tuple[list[Ping], list[GapPlan]]:
    """Per-gap optimal keep-warm schedule (oracle: knows the next delegation time)."""
    if mode not in ("refresh", "extend", "best"):
        raise ValueError(mode)
    calls = sim.cloud_calls
    if any(c.kind != "delegation" for c in calls):
        raise ValueError("optimal_pings assumes the full-history strategy (no summary calls)")
    base = replay(sim, preset)
    index = TranscriptIndex(traj)
    step_ends = [s.t_end for s in sim.steps]
    ttl, rm, wm = preset.ttl_s, preset.read_mult, preset.write_mult
    pin = preset.in_price() * preset.tokenizer_ratio
    hop = ttl - margin_s
    g = grid_s if grid_s is not None else max(1.0, min(15.0, ttl / 20.0))
    pings: list[Ping] = []
    plans: list[GapPlan] = []
    for k in range(1, len(calls)):
        cj, ck = calls[k - 1], calls[k]
        bk = base.calls[k]
        if not bk.result.cold or cj.prompt.tokens < preset.min_cacheable_tokens:
            continue
        gap_start, gap_end = cj.t_start, ck.t_start
        tok_j, tok_k = cj.prompt.tokens, ck.prompt.tokens
        base_in = bk.usd_cache - ck.out_tokens * preset.out_price() * preset.tokenizer_ratio
        best_cost, best_mode, best_p = base_in, "none", ()

        # refresh chain: re-send call j's prompt just in time, read-only
        if mode in ("refresh", "best"):
            times = []
            tt = gap_start
            while gap_end >= tt + hop:
                tt += hop
                times.append(tt)
            cost = len(times) * tok_j * rm * pin + (tok_j * rm + (tok_k - tok_j) * wm) * pin
            if cost < best_cost - 1e-12:
                best_cost, best_mode = cost, "refresh"
                best_p = tuple(Ping(x, cj.prompt) for x in times)

        # extend chain: DP over candidate ping times, prompt = transcript available at that time
        if mode in ("extend", "best"):
            cands = sorted(
                {x for x in step_ends if gap_start < x < gap_end}
                | {gap_start + n * g for n in range(1, int((gap_end - gap_start) / g))}
            )
            ms = [_available_steps(step_ends, x, ck.step_index) for x in cands]
            # before step j's observation exists the only valid content is call j's own prompt
            prompts = [cj.prompt if m <= cj.step_index else index.full_prompt(m) for m in ms]
            toks = [pr.tokens for pr in prompts]
            n = len(cands)
            f = [math.inf] * n
            prev = [-1] * n
            lo = 0
            for c in range(n):
                if cands[c] - gap_start <= hop:  # reachable from call j's entry
                    f[c] = (tok_j * rm + max(0, toks[c] - tok_j) * wm) * pin
                while cands[c] - cands[lo] > hop:
                    lo += 1
                for p in range(lo, c):
                    if f[p] == math.inf:
                        continue
                    v = f[p] + (toks[p] * rm + max(0, toks[c] - toks[p]) * wm) * pin
                    if v < f[c]:
                        f[c], prev[c] = v, p
            for c in range(n):
                if f[c] == math.inf or gap_end - cands[c] > hop:
                    continue
                v = f[c] + (toks[c] * rm + (tok_k - toks[c]) * wm) * pin
                if v < best_cost - 1e-12:
                    chain = []
                    x = c
                    while x != -1:
                        chain.append(x)
                        x = prev[x]
                    best_cost, best_mode = v, "extend"
                    best_p = tuple(Ping(cands[x], prompts[x]) for x in reversed(chain))
        if best_mode != "none":
            pings.extend(best_p)
            plans.append(GapPlan(k, best_mode, best_p, base_in - best_cost))
    return pings, plans
