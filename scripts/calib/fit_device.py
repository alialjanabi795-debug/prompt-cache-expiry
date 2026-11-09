"""Fit the device-model throughput parameters from an agent_replay run.

Model (same as src/cache_delegation/device.py, cool device):
  prefill of n new tokens starting at context c0:  t = o + n/P0 * (1 + (c0 + n/2)/C2)
  decode, per token at context c:                  t_tok = 1/D0 * (1 + c/C2d)
Least squares on the per-step timings; drift = trend of the fit residual ratio over elapsed time.
Writes results/calib/device_fit.json.

Usage: python -I scripts/calib/fit_device.py results/calib/replay_main.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def fit(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines()[1:]]
    n = np.array([r["prompt_n"] for r in rows], float)
    c0 = np.array([r["ctx_before"] for r in rows], float)
    tp = np.array([r["prompt_ms"] for r in rows], float) / 1e3
    keep = n >= 32
    X = np.column_stack([np.ones(keep.sum()), n[keep], n[keep] * (c0[keep] + n[keep] / 2)])
    (o, a, b), *_ = np.linalg.lstsq(X, tp[keep], rcond=None)
    P0, C2 = 1 / a, a / b if b > 0 else float("inf")
    pred_p = X @ np.array([o, a, b])

    dn = np.array([r["predicted_n"] for r in rows], float)
    dms = np.array([r["predicted_ms"] for r in rows], float) / 1e3
    ctx = c0 + n
    kd = dn >= 8
    per_tok = dms[kd] / dn[kd]
    Xd = np.column_stack([np.ones(kd.sum()), ctx[kd]])
    (ad, bd), *_ = np.linalg.lstsq(Xd, per_tok, rcond=None)
    D0, C2d = 1 / ad, ad / bd if bd > 0 else float("inf")
    pred_d = Xd @ np.array([ad, bd])

    el = np.array([r["elapsed_s"] for r in rows], float)
    rp = tp[keep] / pred_p
    rd = per_tok / pred_d
    drift_p = np.polyfit(el[keep] / 60, rp, 1)[0]
    drift_d = np.polyfit(el[kd] / 60, rd, 1)[0]
    r2p = 1 - ((tp[keep] - pred_p) ** 2).sum() / ((tp[keep] - tp[keep].mean()) ** 2).sum()
    r2d = 1 - ((per_tok - pred_d) ** 2).sum() / ((per_tok - per_tok.mean()) ** 2).sum()
    mem = rows[-1]["mem"]
    return {
        "source": str(path),
        "n_steps": len(rows),
        "duration_min": float(el[-1] / 60),
        "final_ctx": int(ctx[-1]),
        "prefill_tps_at_ctx0": float(P0),
        "prefill_ctx_double_tokens": float(C2),
        "prefill_overhead_s": float(o),
        "prefill_fit_r2": float(r2p),
        "decode_tps_at_ctx0": float(D0),
        "decode_ctx_double_tokens": float(C2d),
        "decode_fit_r2": float(r2d),
        "prefill_decode_ratio": float(P0 / D0),
        "drift_prefill_per_min": float(drift_p),
        "drift_decode_per_min": float(drift_d),
        "prefill_tps_observed": {
            "p10": float(np.percentile(n[keep] / tp[keep], 10)),
            "median": float(np.median(n[keep] / tp[keep])),
            "p90": float(np.percentile(n[keep] / tp[keep], 90)),
        },
        "decode_tps_observed": {
            "first_5": float(np.mean(1 / per_tok[:5])),
            "last_5": float(np.mean(1 / per_tok[-5:])),
        },
        "mem_final_gib": {k: v / 2**30 for k, v in mem.items() if k != "failcnt"},
    }


if __name__ == "__main__":
    p = Path(sys.argv[1])
    res = fit(p)
    (p.parent / "device_fit.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
