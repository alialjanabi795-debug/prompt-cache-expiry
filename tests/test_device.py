from dataclasses import replace

import pytest

from cache_delegation.device import DeviceState
from cache_delegation.types import ActionType


def test_prefill_time_grows_with_context(catalog):
    d = DeviceState(catalog.devices["weak_edge"], dynamic=True)
    d_cool = replace(
        catalog.devices["weak_edge"],
        thermal=replace(catalog.devices["weak_edge"].thermal, enabled=False),
    )
    a = DeviceState(d_cool).prefill(1000, 0)
    b = DeviceState(d_cool).prefill(1000, 40_000)
    assert b > a * 1.8  # per-token cost doubles at 40k context
    assert d.prefill(10, 0) > 0


def test_nominal_is_constant_speed(catalog):
    p = catalog.devices["weak_edge"]
    d = DeviceState(p, dynamic=False)
    assert d.prefill(1100, 50_000) == pytest.approx(1100 / p.prefill_tps)
    assert d.decode(110, 50_000) == pytest.approx(110 / p.decode_tps)


def test_thermal_throttling_slows_sustained_prefill_and_recovers(catalog):
    p = catalog.devices["weak_edge"]
    d = DeviceState(p, dynamic=True)
    cold_rate = 2000 / d.prefill(2000, 0)
    for _ in range(20):  # ~6 min of sustained compute
        d.prefill(2000, 0)
    assert d.theta() > 0.5
    hot_rate = 2000 / DeviceState.prefill(d, 2000, 0)
    assert hot_rate < 0.8 * cold_rate
    d.idle(3600)
    assert d.theta() == 0.0


def test_decode_less_sensitive_than_prefill(catalog):
    p = catalog.devices["weak_edge"]
    d = DeviceState(p, dynamic=True)
    d.heat = 1.0
    th = p.thermal
    assert th.min_decode_factor > th.min_prefill_factor
    assert d._factor(th.min_decode_factor) > d._factor(th.min_prefill_factor)


def _tight(p, ram):
    return replace(p, memory=replace(p.memory, ram_usable_gb=ram))


def test_kv_reserved_and_checkpoints(catalog):
    p = catalog.devices["weak12_fp16"]
    d = DeviceState(p, dynamic=True)
    assert p.kv_reserved_gb == pytest.approx(65536 * 32768 / 2**30)  # 2.0 GiB, fill-independent
    a0 = d.anon_gb()
    for _ in range(40):
        d.note_local_call()
    assert d.n_checkpoints == 32
    assert d.anon_gb() - a0 == pytest.approx(32 * 0.049)


def test_ample_memory_tool_has_no_effect(catalog):
    d = DeviceState(catalog.devices["weak_edge"], dynamic=True)
    d.kv_tokens = 50_000
    d.tool(1.0, ActionType.TEST_VALIDATE)
    assert d.kv_tokens == 50_000 and d.oom_kills == 0 and d.refault_debt_gb == 0.0
    assert d.resident_weights_gb() == pytest.approx(5.29)


def test_thrashing_matches_calibration_case_d(catalog):
    # case D: 6.5 GiB usable, anon 2.43 GiB -> 4.07 GiB of 5.29 resident; decode 1.41 -> ~0.04
    p = catalog.devices["weak12_fp16"]
    p = replace(
        p,
        thermal=replace(p.thermal, enabled=False),
        memory=replace(p.memory, ram_usable_gb=6.5, runtime_overhead_gb=2.43 - p.kv_reserved_gb),
    )
    d = DeviceState(p, dynamic=True)
    assert d.resident_weights_gb() == pytest.approx(4.07, abs=0.01)
    extra_per_token = d._weight_reads_s(1.0)
    assert 1.0 / (1.0 / 1.41 + extra_per_token) == pytest.approx(0.041, rel=0.1)


def test_tool_oom_kill_forces_reload_and_full_reprefill(catalog):
    p = catalog.devices["weak12_fp16"]
    d = DeviceState(p, dynamic=True)
    p2 = _tight(p, d.anon_gb() + p.memory.weights_gb + 0.1)
    d = DeviceState(p2, dynamic=True)
    d.kv_tokens = 30_000
    d.tool(1.0, ActionType.WRITE_EDIT)  # 0.05 GiB: fits, no kill
    assert d.oom_kills == 0
    p3 = _tight(p, d.anon_gb() + 0.1)  # anonymous memory alone overflows with a 0.27 GiB test
    d = DeviceState(p3, dynamic=True)
    d.kv_tokens = 30_000
    d.tool(1.0, ActionType.TEST_VALIDATE)
    assert d.oom_kills == 1 and d.kv_tokens == 0
    assert d.tokens_to_prefill(31_000) == 31_000
    assert d.reload_if_needed() == pytest.approx(p.memory.reload_s)
    assert d.reload_if_needed() == 0.0


def test_tool_pressure_drops_weight_pages_once(catalog):
    p = catalog.devices["weak12_fp16"]
    base = DeviceState(p, dynamic=True).anon_gb()
    d = DeviceState(_tight(p, base + p.memory.weights_gb + 0.1), dynamic=True)
    d.tool(1.0, ActionType.TEST_VALIDATE)  # needs 0.27, only 0.1 spare -> 0.17 GiB dropped
    assert d.oom_kills == 0
    assert d.refault_debt_gb == pytest.approx(0.17)
    assert d._weight_reads_s(1.0) == pytest.approx(0.17 * 1024 / 1200.0)
    assert d._weight_reads_s(1.0) == 0.0  # debt paid once


def test_no_memory_effects_in_nominal_arm(catalog):
    p = _tight(catalog.devices["weak8_fp16"], 1.0)
    d = DeviceState(p, dynamic=False)
    d.kv_tokens = 60_000
    d.tool(1.0, ActionType.TEST_VALIDATE)
    assert d.oom_kills == 0 and d.runtime_viable()
    assert d.tokens_to_prefill(61_000) == 1_000
    assert d.decode(10, 0) == pytest.approx(10 / p.decode_tps)


def test_8gb_fp16_profile_cannot_hold_the_stack(catalog):
    d = DeviceState(catalog.devices["weak8_fp16"], dynamic=True)
    for _ in range(32):
        d.note_local_call()
    assert d.anon_gb() + 5.29 > 6.5
    assert d.resident_weights_gb() < 5.29 * 0.6
    fit = DeviceState(catalog.devices["weak8_q8_16k"], dynamic=True)
    for _ in range(32):
        fit.note_local_call()
    assert fit.anon_gb() + 5.29 <= 6.5


def test_resident_kv_avoids_reprefill(catalog):
    d = DeviceState(catalog.devices["weak_edge"], dynamic=True)
    d.kv_tokens = 20_000
    assert d.tokens_to_prefill(23_000) == 3_000


def test_fault_probability_monotone_and_keyed(catalog):
    d = DeviceState(catalog.devices["weak_edge"])
    ps = [d.fault_probability(x) for x in (0, 20_000, 48_000, 80_000)]
    assert ps == sorted(ps) and ps[2] == pytest.approx((0.002 + 0.25) / 2, rel=1e-6)
    a = d.draws_fault(0, "t", 3, 50_000)
    b = DeviceState(catalog.devices["weak_edge"], dynamic=False).draws_fault(0, "t", 3, 50_000)
    assert a == b  # identical in both arms


def test_kv_bytes_per_token_from_model(catalog):
    assert catalog.devices["weak12_fp16"].kv_bytes_per_token == 2 * 8 * 4 * 256 * 2
    assert catalog.devices["weak12_q8"].kv_bytes_per_token == 2 * 8 * 4 * 256 * 34 // 32


def test_thrashing_slows_real_decode_and_prefill_calls(catalog):
    slow = DeviceState(catalog.devices["weak8_fp16"], dynamic=True)
    ok = DeviceState(catalog.devices["weak12_fp16"], dynamic=True)
    assert slow.decode(10, 1000) > 20 * ok.decode(10, 1000)
    assert slow.thrash_s > 0 and ok.thrash_s == 0
    assert slow.prefill(1024, 0) > ok.prefill(1024, 0) + 2 * 19.0  # 2 ubatches re-read weights


def test_measured_duration_plausibility_filter(catalog):
    from dataclasses import replace as rep

    from cache_delegation.timing import ToolTimer

    from .conftest import make_traj

    t = make_traj(n=2, measured=1.672e9)  # freezegun-style mocked clock
    timer = ToolTimer(catalog.tool_time, seed=0)
    assert timer.source_seconds("t", t.steps[0]) < 100  # falls back to the prior
    ok = rep(t.steps[0], tool_name="execute_bash", measured_tool_s=12.0)
    assert 10 < timer.source_seconds("t", ok) < 16
    view = rep(ok, tool_name="str_replace_editor")  # a file view quoting an old test log
    assert timer.source_seconds("t", view) < 5
