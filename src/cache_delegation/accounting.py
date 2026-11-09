"""Cost accounting: (a) call count, (b) token pricing without cache, (c) token pricing with cache.

(c) is the provider bill. ``usd_cache_inf_ttl`` is the same request priced as if no cache entry had
ever expired; ``usd_cache - usd_cache_inf_ttl`` is the cold re-prefill excess used by K1.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cache import CacheResult
from .config import CloudPreset, EdgeTier


@dataclass(frozen=True, slots=True)
class CallCost:
    calls: int  # (a)
    usd_nocache: float  # (b)
    usd_cache: float  # (c)
    usd_cache_inf_ttl: float  # (c) with infinite TTL (same history)
    usd_output: float

    @property
    def reprefill_excess(self) -> float:
        return self.usd_cache - self.usd_cache_inf_ttl


def cloud_call_cost(
    res: CacheResult, out_tokens: int, preset: CloudPreset, is_delegation: bool = True
) -> CallCost:
    r = preset.tokenizer_ratio
    pin = preset.in_price() * r
    out_usd = out_tokens * preset.out_price() * r
    nocache = res.prompt_tokens * pin + out_usd
    cached = (
        res.hit_tokens * preset.read_mult
        + res.write_tokens * preset.write_mult
        + res.uncached_tokens
    ) * pin + out_usd
    # Infinite-TTL counterfactual: the ghost hit is read, the remainder is written.
    if res.uncached_tokens == res.prompt_tokens and res.hit_tokens == 0 and res.write_tokens == 0:
        inf = cached  # not cacheable at all (below minimum)
    else:
        g = res.ghost_hit_tokens
        inf = (g * preset.read_mult + (res.prompt_tokens - g) * preset.write_mult) * pin + out_usd
        inf = min(inf, cached)
    return CallCost(int(is_delegation), nocache, cached, inf, out_usd)


def edge_call_cost(res: CacheResult, out_tokens: int, edge: EdgeTier) -> float:
    uncached = res.prompt_tokens - res.hit_tokens
    return (
        uncached * edge.price_in_per_mtok
        + res.hit_tokens * edge.price_cached_per_mtok
        + out_tokens * edge.price_out_per_mtok
    ) / 1e6


def cloud_ttft_s(res: CacheResult, preset: CloudPreset) -> float:
    r = preset.tokenizer_ratio
    uncached = (res.prompt_tokens - res.hit_tokens) * r
    return (
        preset.ttft_base_s
        + uncached / preset.prefill_tps
        + res.hit_tokens * r / (preset.cached_prefill_tps)
    )
