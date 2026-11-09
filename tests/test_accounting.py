"""Cost accounting against hand-computed cases (anthropic_5m: $5 in / $25 out per MTok,
write 1.25x, read 0.1x)."""

import pytest

from cache_delegation.accounting import cloud_call_cost, edge_call_cost
from cache_delegation.cache import CacheResult, PromptCache, build_prompt


def P(*segs):
    return build_prompt([(s, t, t * 4) for s, t in segs], static_segs=1)


def test_three_call_sequence_by_hand(catalog):
    preset = catalog.cloud["anthropic_5m"]
    c = PromptCache(ttl_s=preset.ttl_s, min_tokens=preset.min_cacheable_tokens)
    # call 1: 10,000 tokens, all written; 100 output tokens
    r1 = c.request(P(("P", 8000), ("x", 2000)), 0.0)
    k1 = cloud_call_cost(r1, 100, preset)
    assert k1.calls == 1
    assert k1.usd_nocache == pytest.approx(10_000 * 5e-6 + 100 * 25e-6)  # 0.0525
    assert k1.usd_cache == pytest.approx(10_000 * 1.25 * 5e-6 + 100 * 25e-6)  # 0.065
    # call 2 at t=30s: 12,000 tokens, 10,000 read + 2,000 written
    r2 = c.request(P(("P", 8000), ("x", 2000), ("y", 2000)), 30.0)
    k2 = cloud_call_cost(r2, 100, preset)
    assert k2.usd_cache == pytest.approx(
        10_000 * 0.1 * 5e-6 + 2_000 * 1.25 * 5e-6 + 100 * 25e-6
    )  # 0.02
    assert k2.reprefill_excess == pytest.approx(0.0)
    # call 3 at t=430s: everything expired (last refresh at 30 -> 330); 13,000 written
    r3 = c.request(P(("P", 8000), ("x", 2000), ("y", 2000), ("z", 1000)), 430.0)
    k3 = cloud_call_cost(r3, 100, preset)
    assert r3.cold
    assert k3.usd_cache == pytest.approx(13_000 * 1.25 * 5e-6 + 100 * 25e-6)  # 0.08375
    inf = 12_000 * 0.1 * 5e-6 + 1_000 * 1.25 * 5e-6 + 100 * 25e-6  # 0.01475
    assert k3.usd_cache_inf_ttl == pytest.approx(inf)
    assert k3.reprefill_excess == pytest.approx(
        13_000 * 1.25 * 5e-6 - (12_000 * 0.1 * 5e-6 + 1_000 * 1.25 * 5e-6)
    )


def test_one_hour_write_premium(catalog):
    preset = catalog.cloud["anthropic_1h"]
    r = CacheResult(
        prompt_tokens=10_000,
        hit_tokens=0,
        write_tokens=10_000,
        uncached_tokens=0,
        ghost_hit_tokens=0,
    )
    k = cloud_call_cost(r, 0, preset)
    assert k.usd_cache == pytest.approx(10_000 * 2.0 * 5e-6)


def test_ping_bills_no_output(catalog):
    preset = catalog.cloud["openai_30m"]
    r = CacheResult(20_000, 20_000, 0, 0, 20_000)
    k = cloud_call_cost(r, preset.ping_output_tokens, preset, is_delegation=False)
    assert k.calls == 0
    assert k.usd_cache == pytest.approx(20_000 * 0.1 * 4e-6)


def test_uncacheable_prompt_priced_at_base(catalog):
    preset = catalog.cloud["anthropic_5m"]
    r = CacheResult(500, 0, 0, 500, 0)
    k = cloud_call_cost(r, 10, preset)
    assert k.usd_cache == pytest.approx(k.usd_nocache)
    assert k.usd_cache_inf_ttl == pytest.approx(k.usd_cache)


def test_edge_cost(catalog):
    r = CacheResult(10_000, 8_000, 2_000, 0, 8_000)
    assert edge_call_cost(r, 100, catalog.edge) == pytest.approx((2000 * 0.15 + 100 * 0.60) / 1e6)
