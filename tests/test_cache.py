from cache_delegation.cache import PromptCache, build_prompt


def P(*segs, static=1):
    """Prompt from (id, tokens) pairs."""
    return build_prompt([(s, t, t * 4) for s, t in segs], static_segs=static)


BASE = [("P", 4000), ("o0", 500), ("b0", 1500)]


def test_first_request_writes_everything():
    c = PromptCache(ttl_s=300, min_tokens=1024)
    r = c.request(P(*BASE), t=0.0)
    assert (r.hit_tokens, r.write_tokens, r.uncached_tokens) == (0, 6000, 0)
    assert r.compulsory and not r.cold


def test_hit_within_ttl_bills_only_delta():
    c = PromptCache(ttl_s=300, min_tokens=1024)
    c.request(P(*BASE), t=0.0)
    r = c.request(P(*BASE, ("o1", 200), ("b1", 800)), t=100.0)
    assert r.hit_tokens == 6000
    assert r.write_tokens == 1000
    assert not r.cold


def test_expiry_after_ttl():
    c = PromptCache(ttl_s=300, min_tokens=1024)
    c.request(P(*BASE), t=0.0)
    r = c.request(P(*BASE, ("o1", 200)), t=300.0)  # exactly at expiry -> expired
    assert r.hit_tokens == 0
    assert r.write_tokens == 6200
    assert r.ghost_hit_tokens == 6000
    assert r.cold and r.expired_tokens == 6000


def test_refresh_on_hit_extends_lifetime():
    c = PromptCache(ttl_s=300, min_tokens=1024)
    p = P(*BASE)
    c.request(p, t=0.0)
    c.request(p, t=250.0)  # read refreshes -> expiry 550
    r = c.request(p, t=500.0)
    assert r.hit_tokens == 6000 and r.write_tokens == 0
    r2 = c.request(p, t=801.0)  # 500 + 300 = 800 < 801 -> expired
    assert r2.hit_tokens == 0 and r2.cold


def test_ttl_clock_starts_at_request_start_not_end():
    c = PromptCache(ttl_s=300, min_tokens=0)
    c.request(P(*BASE), t=10.0)
    assert c.peek(P(*BASE), 309.9)[0] == 6000
    assert c.peek(P(*BASE), 310.0)[0] == 0


def test_static_prefix_breakpoint_survives_divergence():
    c = PromptCache(ttl_s=300, min_tokens=1024)
    c.request(P(*BASE), t=0.0)
    r = c.request(P(("P", 4000), ("sum", 600), ("b0", 1500)), t=10.0)
    assert r.hit_tokens == 4000  # only the system+task prefix matches
    assert r.write_tokens == 2100


def test_below_minimum_is_not_cached():
    c = PromptCache(ttl_s=300, min_tokens=1024)
    small = P(("P", 500), ("o0", 100))
    r1 = c.request(small, 0.0)
    r2 = c.request(small, 1.0)
    assert r1.uncached_tokens == 600 and r2.hit_tokens == 0 and r2.uncached_tokens == 600


def test_prefix_mismatch_is_not_a_hit():
    c = PromptCache(ttl_s=300, min_tokens=0, static_breakpoint=False)
    c.request(P(("A", 3000), ("B", 1000)), 0.0)
    r = c.request(P(("A2", 3000), ("B", 1000)), 1.0)
    assert r.hit_tokens == 0


def test_longest_live_prefix_is_used():
    c = PromptCache(ttl_s=300, min_tokens=0)
    c.request(P(*BASE[:2]), 0.0)
    c.request(P(*BASE), 100.0)
    r = c.request(P(*BASE, ("o1", 10)), 350.0)  # first entry expired at 300, second alive
    assert r.hit_tokens == 6000


def test_edge_cache_never_expires():
    c = PromptCache(ttl_s=None, min_tokens=0)
    c.request(P(*BASE), 0.0)
    r = c.request(P(*BASE, ("o1", 10)), 1e9)
    assert r.hit_tokens == 6000 and not r.cold
