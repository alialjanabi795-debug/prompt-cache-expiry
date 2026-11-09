"""Typed configuration objects and YAML loading.

Every numeric default lives in ``configs/*.yaml`` (marked there as ASSUMPTION / MEASURED / DOC);
these dataclasses only define structure and derived quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from .types import ActionType


@dataclass(frozen=True)
class ThermalParams:
    enabled: bool = True
    tau_s: float = 240.0  # first-order heating/cooling time constant
    onset: float = 0.35  # normalized heat (1 = steady state under sustained full LLM compute)
    min_prefill_factor: float = 0.5  # throughput multiplier at full throttle
    min_decode_factor: float = 0.75
    min_tool_factor: float = 0.75
    tool_heat: float = 0.6  # heat input of tool execution relative to LLM compute
    idle_heat: float = 0.05


@dataclass(frozen=True)
class FaultParams:
    enabled: bool = True
    p_base: float = 0.002
    p_max: float = 0.3
    len50_tokens: float = 48_000.0
    scale_tokens: float = 8_000.0
    wasted_fraction: float = 0.5  # fraction of the attempted local prefill lost before failure


@dataclass(frozen=True)
class PowerParams:
    compute_w: float = 6.0
    tool_w: float = 4.0
    idle_w: float = 0.8


@dataclass(frozen=True)
class MemoryParams:
    """Device memory budget for the local runtime (llama.cpp semantics, calibrated 2026-10-19).

    llama.cpp reserves (and touches) the whole KV cache for n_ctx at start-up, keeps up to
    ``max_checkpoints`` copies of the recurrent state for prefix reuse on hybrid models, and maps
    the weights from the GGUF file (reclaimable page cache). It never evicts the KV by itself:
    under pressure the kernel first drops weight pages (thrashing), and kills a process only when
    anonymous memory alone no longer fits.
    """

    ram_usable_gb: float = 10.0  # RAM available to agent + runtime + tools (GiB)
    weights_gb: float = 5.29  # MEASURED page cache of Qwen3.5-9B Q4_K_M
    weights_mmap: bool = True
    runtime_overhead_gb: float = 0.26  # MEASURED anon besides KV/checkpoints (RS, compute buffers)
    checkpoint_gb: float = 0.049  # MEASURED per context checkpoint (recurrent-state copy)
    max_checkpoints: int = 32  # llama.cpp --ctx-checkpoints default
    refault_mb_s: float = 54.0  # MEASURED effective weight re-read rate under steady thrashing
    refault_seq_mb_s: float = 1200.0  # MEASURED lower bound: one-time re-read after a tool
    ubatch_tokens: int = 512  # weights are streamed once per ubatch during prefill
    reload_s: float = 30.0  # MEASURED model load to ready after a kill
    swap_gb: float = 0.0
    oom_kills_tool: bool = True  # MEASURED (case E2): the tool was killed right after the runtime


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    decode_tps: float
    prefill_decode_ratio: float
    prefill_ctx_double_tokens: float  # context at which per-token prefill cost doubles
    decode_ctx_double_tokens: float
    kv_bytes_per_token: int
    max_ctx_tokens: int  # runtime n_ctx (KV reserved for all of it)
    tool_speed_factor: float  # tool runtime multiplier vs the trajectory's source container
    tool_mem_gb: dict[str, float] = field(default_factory=dict)  # by ActionType value (GiB)
    memory: MemoryParams = field(default_factory=MemoryParams)
    thermal: ThermalParams = field(default_factory=ThermalParams)
    fault: FaultParams = field(default_factory=FaultParams)
    power: PowerParams = field(default_factory=PowerParams)

    @property
    def prefill_tps(self) -> float:
        return self.decode_tps * self.prefill_decode_ratio

    @property
    def kv_reserved_gb(self) -> float:
        return self.kv_bytes_per_token * self.max_ctx_tokens / 2**30

    @property
    def max_local_prompt_tokens(self) -> int:
        """Largest prompt+output the runtime accepts (n_ctx)."""
        return self.max_ctx_tokens

    def tool_mem(self, action: ActionType) -> float:
        return self.tool_mem_gb.get(action.value, 0.0)

    def with_decode_tps(self, tps: float) -> DeviceProfile:
        return replace(self, decode_tps=tps, name=f"{self.name}@{tps:g}tps")


@dataclass(frozen=True)
class NetworkProfile:
    name: str
    down_mbps: float
    up_mbps: float
    rtt_s: float
    j_per_byte_up: float
    j_per_byte_down: float
    tail_j: float  # radio tail energy per request burst

    def upload_s(self, nbytes: float) -> float:
        return nbytes * 8.0 / (self.up_mbps * 1e6)

    def download_s(self, nbytes: float) -> float:
        return nbytes * 8.0 / (self.down_mbps * 1e6)


@dataclass(frozen=True)
class CloudPreset:
    name: str
    provider: str
    model: str
    ttl_s: float
    write_mult: float
    read_mult: float
    min_cacheable_tokens: int
    price_in_per_mtok: float
    price_out_per_mtok: float
    prefill_tps: float  # uncached prompt processing at the provider
    cached_prefill_tps: float  # effective rate for cache-hit tokens (TTFT contribution)
    decode_tps: float
    ttft_base_s: float
    upload_mode: str = "full"  # "full": every request re-uploads the prompt; "incremental"
    tokenizer_ratio: float = 1.0  # cloud tokens per Qwen token
    ping_output_tokens: int = 0  # max_tokens=0 pre-warm requests bill no output
    cache_static_prefix: bool = True  # explicit breakpoint after system+tools+task
    prices_source: str = ""
    prices_verified_on: str = ""

    def in_price(self) -> float:
        return self.price_in_per_mtok / 1e6

    def out_price(self) -> float:
        return self.price_out_per_mtok / 1e6


@dataclass(frozen=True)
class EdgeTier:
    name: str
    model: str
    prefill_tps: float
    decode_tps: float
    rtt_s: float
    lan_mbps: float
    price_in_per_mtok: float
    price_cached_per_mtok: float
    price_out_per_mtok: float
    cache_capacity_tokens: int


@dataclass(frozen=True)
class LogNormal:
    median_s: float
    sigma: float


@dataclass(frozen=True)
class ToolTimeParams:
    priors: dict[str, LogNormal]
    default: LogNormal
    use_measured: bool = True
    measured_overhead_s: float = 0.8  # interpreter start + collection not included in "in X s"
    measured_jitter_sigma: float = 0.1
    # a reported duration is trusted only for shell tools and up to this bound (mocked clocks,
    # e.g. freezegun, print absurd "in 1672354266s"; file views can quote old test logs)
    measured_max_s: float = 1800.0
    measured_tools: tuple[str, ...] = ("execute_bash", "bash")


# ------------------------------------------------------------------------------------------------
# YAML -> dataclass
# ------------------------------------------------------------------------------------------------
def _build(cls: type, data: Any) -> Any:
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        typ = hints[f.name]
        if is_dataclass(typ):
            kwargs[f.name] = _build(typ, val)
        elif f.name == "priors":
            kwargs[f.name] = {k: LogNormal(**v) for k, v in val.items()}
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def _strip_meta(d: Any) -> Any:
    """Drop documentation keys (``_note``, ``_source`` ...) before building dataclasses."""
    if isinstance(d, dict):
        return {k: _strip_meta(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [_strip_meta(v) for v in d]
    return d


def load_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Catalog:
    devices: dict[str, DeviceProfile]
    networks: dict[str, NetworkProfile]
    cloud: dict[str, CloudPreset]
    edge: EdgeTier
    tool_time: ToolTimeParams


def load_catalog(config_dir: Path) -> Catalog:
    dev = _strip_meta(load_yaml(config_dir / "device.yaml"))
    net = _strip_meta(load_yaml(config_dir / "network.yaml"))
    tiers = _strip_meta(load_yaml(config_dir / "tiers.yaml"))
    tt = _strip_meta(load_yaml(config_dir / "tool_time.yaml"))
    base = dev.get("defaults", {})
    devices = {}
    for name, d in dev["profiles"].items():
        merged = _deep_merge(base, d)
        devices[name] = _build(DeviceProfile, {"name": name, **merged})
    networks = {n: _build(NetworkProfile, {"name": n, **v}) for n, v in net["profiles"].items()}
    cloud = {n: _build(CloudPreset, {"name": n, **v}) for n, v in tiers["cloud_presets"].items()}
    edge = _build(EdgeTier, tiers["edge"])
    tool = _build(ToolTimeParams, tt)
    return Catalog(devices, networks, cloud, edge, tool)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
