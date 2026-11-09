"""Weak-device model: context-dependent prefill/decode, thermal throttling, memory, faults.

Nominal arm (``dynamic=False``): constant prefill/decode throughput, no throttling, no memory
effects. Dynamic arm: everything on. Faults and the n_ctx limit are identical in both arms (they
depend only on prompt length and a keyed RNG), so tier decisions never differ.

Memory (calibrated against llama.cpp, see MemoryParams): anonymous memory = reserved KV (n_ctx) +
runtime overhead + context checkpoints (+ the running tool). With mmap weights the kernel keeps
only ram_usable - anon of the weights resident; every pass over the weights (one per decoded token,
one per prefill ubatch) re-reads the missing part at ``refault_mb_s`` (thrashing). Pages dropped
while a tool runs are re-read once by the next LLM call. If anonymous memory alone exceeds the
budget, the kernel OOM-kills the runtime: the KV is lost and the next local call pays a reload
and a full re-prefill.

Thermal model: normalized heat h with dh/dt = (u - h) / tau, where u = 1 during LLM compute,
``tool_heat`` during tool execution and ``idle_heat`` otherwise (h = 1 is the steady state under
sustained LLM compute). Throttle level theta = clip((h - onset) / (1 - onset), 0, 1) and each
workload runs at 1 - theta * (1 - min_factor) of its cool throughput. Integration is exact for
constant u within chunks of at most ``CHUNK_S`` seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import DeviceProfile
from .timing import keyed_rng
from .types import ActionType

CHUNK_S = 2.0


@dataclass
class EnergyLedger:
    compute_j: float = 0.0
    tool_j: float = 0.0
    idle_j: float = 0.0
    radio_j: float = 0.0

    @property
    def total_j(self) -> float:
        return self.compute_j + self.tool_j + self.idle_j + self.radio_j


class DeviceState:
    def __init__(self, profile: DeviceProfile, dynamic: bool = True) -> None:
        self.p = profile
        self.dynamic = dynamic
        self.heat = 0.0
        self.kv_tokens = 0  # length of the transcript prefix whose KV is resident (0 = none)
        self.n_checkpoints = 0
        self.refault_debt_gb = 0.0
        self.needs_reload = False
        self.oom_kills = 0
        self.thrash_s = 0.0
        self.energy = EnergyLedger()
        self.throttle_time_s = 0.0  # time spent with theta > 0 while computing

    # -- thermal ----------------------------------------------------------------------------
    @property
    def thermal_on(self) -> bool:
        return self.dynamic and self.p.thermal.enabled

    def theta(self) -> float:
        if not self.thermal_on:
            return 0.0
        th = self.p.thermal
        return min(1.0, max(0.0, (self.heat - th.onset) / (1.0 - th.onset)))

    def _heat(self, dt: float, u: float) -> None:
        if not self.thermal_on or dt <= 0:
            return
        self.heat = u + (self.heat - u) * math.exp(-dt / self.p.thermal.tau_s)

    def _factor(self, min_factor: float) -> float:
        return 1.0 - self.theta() * (1.0 - min_factor)

    # -- compute ------------------------------------------------------------------------------
    def _llm_work(
        self,
        n: int,
        ctx0: int,
        tps: float,
        ctx_double: float,
        min_factor: float,
        passes: float = 0.0,
    ) -> float:
        if n <= 0:
            return 0.0
        extra = self._weight_reads_s(passes)
        if not self.dynamic:
            dt = n / tps
            self.energy.compute_j += dt * self.p.power.compute_w
            return dt
        if extra > 0:
            self.thrash_s += extra
            self._heat(extra, self.p.thermal.idle_heat)  # waiting on storage, not computing
            self.energy.idle_j += extra * self.p.power.idle_w
        total, ctx, left = 0.0, float(ctx0), n
        while left > 0:
            f = self._factor(min_factor) if self.thermal_on else 1.0
            per_tok = (1.0 + ctx / ctx_double) / (tps * f)
            m = min(left, max(1, int(CHUNK_S / per_tok)))
            dt = m * (1.0 + (ctx + m / 2.0) / ctx_double) / (tps * f)
            if f < 1.0:
                self.throttle_time_s += dt
            self._heat(dt, 1.0)
            total += dt
            ctx += m
            left -= m
        self.energy.compute_j += total * self.p.power.compute_w
        return total + extra

    def prefill(self, n_new: int, ctx_start: int) -> float:
        th = self.p.thermal
        passes = math.ceil(n_new / self.p.memory.ubatch_tokens) if n_new > 0 else 0
        return self._llm_work(
            n_new,
            ctx_start,
            self.p.prefill_tps,
            self.p.prefill_ctx_double_tokens,
            th.min_prefill_factor,
            passes,
        )

    def decode(self, n_out: int, ctx: int) -> float:
        th = self.p.thermal
        return self._llm_work(
            n_out,
            ctx,
            self.p.decode_tps,
            self.p.decode_ctx_double_tokens,
            th.min_decode_factor,
            float(n_out),
        )

    def tool(self, source_s: float, action: ActionType | None = None) -> float:
        """Run a tool whose source-container runtime is ``source_s``; returns device seconds.

        Applies the tool's memory footprint: weight pages dropped (re-read later) or an OOM kill.
        """
        kills = self.oom_kills
        if action is not None:
            self._tool_memory(self.p.tool_mem(action))
        work = source_s * self.p.tool_speed_factor
        if self.oom_kills > kills and self.p.memory.oom_kills_tool:
            work *= 2.0  # the tool dies with the runtime and the agent reruns it
        if work <= 0:
            return 0.0
        if not self.thermal_on:
            self.energy.tool_j += work * self.p.power.tool_w
            return work
        th = self.p.thermal
        total, left = 0.0, work
        while left > 1e-12:
            f = self._factor(th.min_tool_factor)
            dt = min(CHUNK_S, left / f)
            self._heat(dt, th.tool_heat)
            left -= dt * f
            total += dt
        self.energy.tool_j += total * self.p.power.tool_w
        return total

    def idle(self, dt: float) -> None:
        if dt <= 0:
            return
        self._heat(dt, self.p.thermal.idle_heat)
        self.energy.idle_j += dt * self.p.power.idle_w

    # -- memory ---------------------------------------------------------------------------------
    def anon_gb(self, tool_gb: float = 0.0) -> float:
        m = self.p.memory
        a = (
            self.p.kv_reserved_gb
            + m.runtime_overhead_gb
            + self.n_checkpoints * m.checkpoint_gb
            + tool_gb
        )
        return a if m.weights_mmap else a + m.weights_gb

    def resident_weights_gb(self, tool_gb: float = 0.0) -> float:
        m = self.p.memory
        if not m.weights_mmap:
            return m.weights_gb
        return min(m.weights_gb, max(0.0, m.ram_usable_gb - self.anon_gb(tool_gb)))

    def _weight_reads_s(self, passes: float) -> float:
        """Storage time to re-read missing weight pages (thrashing + debt from tool pressure)."""
        if not self.dynamic:
            return 0.0
        m = self.p.memory
        missing = m.weights_gb - self.resident_weights_gb()
        debt, self.refault_debt_gb = self.refault_debt_gb, 0.0
        return passes * missing * 1024.0 / m.refault_mb_s + debt * 1024.0 / m.refault_seq_mb_s

    def _tool_memory(self, tool_gb: float) -> None:
        if not self.dynamic or tool_gb <= 0:
            return
        m = self.p.memory
        if self.anon_gb(tool_gb) > m.ram_usable_gb + m.swap_gb:
            self.oom_kill()
            return
        dropped = self.resident_weights_gb() - self.resident_weights_gb(tool_gb)
        self.refault_debt_gb += max(0.0, dropped)

    def oom_kill(self) -> None:
        self.kv_tokens = 0
        self.n_checkpoints = 0
        self.needs_reload = True
        self.oom_kills += 1

    def note_local_call(self) -> None:
        """llama.cpp keeps one context checkpoint per prompt, up to max_checkpoints."""
        self.n_checkpoints = min(self.p.memory.max_checkpoints, self.n_checkpoints + 1)

    def reload_if_needed(self) -> float:
        if not self.needs_reload:
            return 0.0
        self.needs_reload = False
        dt = self.p.memory.reload_s
        self.energy.compute_j += dt * self.p.power.tool_w
        return dt

    def runtime_viable(self) -> bool:
        """The runtime cannot even start if its anonymous memory exceeds the budget."""
        m = self.p.memory
        return not self.dynamic or self.anon_gb() <= m.ram_usable_gb + m.swap_gb

    def tokens_to_prefill(self, context_tokens: int) -> int:
        if 0 < self.kv_tokens <= context_tokens:
            return context_tokens - self.kv_tokens
        return context_tokens

    # -- faults ---------------------------------------------------------------------------------
    def fits(self, prompt_tokens: int, out_tokens: int) -> bool:
        return (
            prompt_tokens + out_tokens <= self.p.max_local_prompt_tokens and self.runtime_viable()
        )

    def fault_probability(self, prompt_tokens: int) -> float:
        f = self.p.fault
        if not f.enabled:
            return 0.0
        z = (prompt_tokens - f.len50_tokens) / f.scale_tokens
        return f.p_base + (f.p_max - f.p_base) / (1.0 + math.exp(-z))

    def draws_fault(self, seed: int, traj_id: str, step_index: int, prompt_tokens: int) -> bool:
        u = keyed_rng("fault", seed, traj_id, step_index).random()
        return u < self.fault_probability(prompt_tokens)
