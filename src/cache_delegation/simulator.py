"""Replay one trajectory under one policy and one scenario.

Per step i (in order): decide a tier; execute the LLM call on that tier (local with fallback to the
edge on a fault or when the prompt does not fit; edge; or cloud through the context strategy and
the TTL cache); then run the step's tool on the device. Time, energy, cache state and costs are
tracked throughout. Nothing is generated: outputs and observations come from the recorded
trajectory, so every policy sees the same agent behaviour (success-neutral replay).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .accounting import CallCost, cloud_call_cost, cloud_ttft_s, edge_call_cost
from .cache import CacheResult, Prompt, PromptCache
from .config import CloudPreset, DeviceProfile, EdgeTier, NetworkProfile, ToolTimeParams
from .context import CloudContext, TranscriptIndex
from .device import DeviceState
from .policies import DecisionContext, Policy
from .timing import ToolTimer
from .types import Tier, Trajectory

EDGE_J_PER_BYTE = 1.0e-7  # LAN radio energy per byte (ASSUMPTION; Wi-Fi order of magnitude)


@dataclass(frozen=True)
class Scenario:
    device: DeviceProfile
    network: NetworkProfile
    cloud: CloudPreset
    edge: EdgeTier
    tool_time: ToolTimeParams
    dynamic: bool = True
    context: str = "full"
    seed: int = 0
    summary_tokens: int = 600
    retrieval_k: int = 5
    retrieval_budget_tokens: int = 8000
    warm_fraction: float = 0.5

    @property
    def label(self) -> str:
        return (
            f"{self.device.name}|{self.network.name}|{self.cloud.name}|"
            f"{'dyn' if self.dynamic else 'nom'}|{self.context}"
        )


@dataclass(slots=True)
class CloudCall:
    step_index: int
    kind: str  # "delegation" | "summary" | "ping"
    t_start: float  # request start at the provider (TTL clock)
    prompt: Prompt
    out_tokens: int
    result: CacheResult
    cost: CallCost
    latency_s: float
    upload_bytes: int


@dataclass(slots=True)
class StepRecord:
    index: int
    action: str
    requested: Tier
    executed: Tier
    fallback: str  # "" | "fault" | "oom"
    t_start: float
    t_llm_end: float
    t_end: float
    context_tokens: int
    output_tokens: int
    local_prefill_tokens: int = 0
    local_prefill_s: float = 0.0
    local_decode_s: float = 0.0
    local_wasted_s: float = 0.0
    local_reload_s: float = 0.0  # runtime restart after an OOM kill
    reprefill: bool = False  # local step had to re-prefill because the KV had been evicted
    kv_evicted_after: bool = False
    tool_s: float = 0.0
    remote_s: float = 0.0  # edge or cloud LLM latency
    handoff_s: float = 0.0  # extra time spent on a handoff (local summary, cloud summary call)
    cloud_mode: str = ""
    theta: float = 0.0


@dataclass
class SimResult:
    traj_id: str
    policy: str
    scenario: str
    steps: list[StepRecord] = field(default_factory=list)
    cloud_calls: list[CloudCall] = field(default_factory=list)
    edge_usd: float = 0.0
    total_s: float = 0.0
    energy_compute_j: float = 0.0
    energy_tool_j: float = 0.0
    energy_idle_j: float = 0.0
    energy_radio_j: float = 0.0
    throttle_s: float = 0.0
    thrash_s: float = 0.0
    oom_kills: int = 0

    @property
    def delegations(self) -> list[CloudCall]:
        return [c for c in self.cloud_calls if c.kind == "delegation"]


def simulate(traj: Trajectory, policy: Policy, sc: Scenario) -> SimResult:
    dev = DeviceState(sc.device, dynamic=sc.dynamic)
    net = sc.network
    preset = sc.cloud
    timer = ToolTimer(sc.tool_time, seed=sc.seed)
    index = TranscriptIndex(traj)
    ctxb = CloudContext(
        traj,
        index,
        sc.context,
        sc.summary_tokens,
        sc.retrieval_k,
        sc.retrieval_budget_tokens,
        sc.warm_fraction,
    )
    cloud_cache = PromptCache(
        ttl_s=preset.ttl_s,
        min_tokens=preset.min_cacheable_tokens,
        static_breakpoint=preset.cache_static_prefix,
    )
    edge_cache = PromptCache(
        ttl_s=None,
        min_tokens=0,
        static_breakpoint=True,
        capacity_tokens=sc.edge.cache_capacity_tokens,
    )
    res = SimResult(traj.traj_id, policy.name, sc.label)
    policy.reset(traj)
    t = 0.0
    tiers: list[Tier] = []
    executed_prev: Tier | None = None
    kv_evicted = False

    def cloud_request(
        prompt: Prompt, out_tokens: int, out_bytes: int, kind: str, i: int, t0: float
    ) -> CloudCall:
        _, ghost = cloud_cache.peek(prompt, t0)
        up = prompt.nbytes
        if preset.upload_mode == "incremental" and prompt.tokens > 0:
            up = int(prompt.nbytes * (1.0 - ghost / prompt.tokens))
        t_req = t0 + net.upload_s(up) + net.rtt_s / 2.0
        cres = cloud_cache.request(prompt, t_req)
        decode_s = out_tokens * preset.tokenizer_ratio / preset.decode_tps
        lat = (
            net.upload_s(up)
            + net.rtt_s
            + cloud_ttft_s(cres, preset)
            + max(decode_s, net.download_s(out_bytes))
        )
        cost = cloud_call_cost(cres, out_tokens, preset, is_delegation=(kind == "delegation"))
        dev.energy.radio_j += up * net.j_per_byte_up + out_bytes * net.j_per_byte_down + net.tail_j
        call = CloudCall(i, kind, t_req, prompt, out_tokens, cres, cost, lat, up)
        res.cloud_calls.append(call)
        return call

    for i, step in enumerate(traj.steps):
        L = step.context_tokens
        need = dev.tokens_to_prefill(L)
        dctx = DecisionContext(
            now=t, prev_tiers=tuple(tiers), local_prefill_needed=need, local_kv_evicted=kv_evicted
        )
        req = policy.decide(traj, i, dctx)
        tiers.append(req)
        rec = StepRecord(
            index=i,
            action=step.action.value,
            requested=req,
            executed=req,
            fallback="",
            t_start=t,
            t_llm_end=t,
            t_end=t,
            context_tokens=L,
            output_tokens=step.output_tokens,
            theta=dev.theta(),
        )
        executed = req

        if req == Tier.LOCAL:
            if not dev.fits(L, step.output_tokens):
                executed, rec.fallback = Tier.EDGE, "oom"
                dev.kv_tokens = 0
            elif dev.draws_fault(sc.seed, traj.traj_id, i, L):
                executed, rec.fallback = Tier.EDGE, "fault"
                # part of the attempted prefill is lost before the runtime fails
                wasted_tokens = int(need * sc.device.fault.wasted_fraction)
                rec.local_wasted_s = dev.prefill(wasted_tokens, L - need)
                t += rec.local_wasted_s
                dev.kv_tokens = 0
                kv_evicted = True
            else:
                rec.reprefill = kv_evicted and L > 0
                rec.local_reload_s = dev.reload_if_needed()
                need = dev.tokens_to_prefill(L)
                rec.local_prefill_tokens = need
                rec.local_prefill_s = dev.prefill(need, L - need)
                rec.local_decode_s = dev.decode(step.output_tokens, L)
                t += rec.local_reload_s + rec.local_prefill_s + rec.local_decode_s
                dev.kv_tokens = L + step.output_tokens
                dev.note_local_call()
                kv_evicted = False

        if executed == Tier.EDGE:
            prompt = index.full_prompt(i)
            eres = edge_cache.request(prompt, t)
            lat = (
                sc.edge.rtt_s
                + prompt.nbytes * 8 / (sc.edge.lan_mbps * 1e6)
                + (eres.prompt_tokens - eres.hit_tokens) / sc.edge.prefill_tps
                + step.output_tokens / sc.edge.decode_tps
            )
            res.edge_usd += edge_call_cost(eres, step.output_tokens, sc.edge)
            dev.energy.radio_j += (prompt.nbytes + step.output_bytes) * EDGE_J_PER_BYTE
            dev.idle(lat)
            rec.remote_s = lat
            t += lat

        elif executed == Tier.CLOUD:
            prompt, extras = ctxb.prompt_for(i, executed_prev == Tier.CLOUD, cloud_cache, t)
            rec.cloud_mode = extras.mode
            if extras.local_summary_tokens:
                pre = dev.tokens_to_prefill(L)
                hs = dev.prefill(pre, L - pre) + dev.decode(extras.local_summary_tokens, L)
                dev.kv_tokens = L
                rec.handoff_s += hs
                t += hs
            if extras.cloud_summary_prompt is not None:
                sc_call = cloud_request(
                    extras.cloud_summary_prompt,
                    extras.cloud_summary_out_tokens,
                    extras.cloud_summary_out_tokens * 4,
                    "summary",
                    i,
                    t,
                )
                dev.idle(sc_call.latency_s)
                rec.handoff_s += sc_call.latency_s
                t += sc_call.latency_s
            call = cloud_request(prompt, step.output_tokens, step.output_bytes, "delegation", i, t)
            dev.idle(call.latency_s)
            rec.remote_s = call.latency_s
            t += call.latency_s

        rec.executed = executed
        rec.t_llm_end = t
        # ---- tool execution on the device ----
        kills = dev.oom_kills
        rec.tool_s = dev.tool(timer.source_seconds(traj.traj_id, step), step.action)
        t += rec.tool_s
        if dev.oom_kills > kills:  # the tool's memory pushed the runtime into an OOM kill
            rec.kv_evicted_after = True
            kv_evicted = True
        rec.t_end = t
        executed_prev = executed
        res.steps.append(rec)

    res.total_s = t
    e = dev.energy
    res.energy_compute_j, res.energy_tool_j = e.compute_j, e.tool_j
    res.energy_idle_j, res.energy_radio_j = e.idle_j, e.radio_j
    res.throttle_s = dev.throttle_time_s
    res.thrash_s = dev.thrash_s
    res.oom_kills = dev.oom_kills
    return res
