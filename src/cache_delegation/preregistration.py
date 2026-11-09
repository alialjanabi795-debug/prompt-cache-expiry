"""Freeze / verify the pre-registered kill criteria and compute verdicts from sweep tables."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(yaml_path: Path, lock_path: Path, note: str = "") -> dict:
    if lock_path.exists():
        raise FileExistsError(f"{lock_path} exists; criteria are already frozen")
    lock = {
        "file": yaml_path.name,
        "sha256": sha256_file(yaml_path),
        "frozen_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": note,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def verify(yaml_path: Path, lock_path: Path) -> None:
    lock = json.loads(lock_path.read_text())
    got = sha256_file(yaml_path)
    if got != lock["sha256"]:
        raise RuntimeError(f"kill criteria changed after freezing: {got} != {lock['sha256']}")


# ------------------------------------------------------------------------------------------------
# Verdict helpers. Inputs are per-trajectory tables (one row per trajectory and cell).
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Estimate:
    value: float
    lo: float
    hi: float

    def straddles(self, threshold: float) -> bool:
        return self.lo < threshold <= self.hi or self.lo <= threshold < self.hi


def pooled_ratio(num: np.ndarray, den: np.ndarray, b: int = 2000, seed: int = 7) -> Estimate:
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    value = num.sum() / den.sum() if den.sum() else float("nan")
    if len(num) < 2:
        return Estimate(value, value, value)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(b, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return Estimate(float(value), float(lo), float(hi))


def holds_k4(
    cells: dict[tuple[str, float], bool],
    presets: Sequence[str],
    tps_grid: Sequence[float],
    min_presets: int = 2,
) -> bool:
    """K4: for every tps, the finding holds in >= min_presets presets."""
    for tps in tps_grid:
        if sum(bool(cells.get((p, tps), False)) for p in presets) < min_presets:
            return False
    return True


def ranking(df: pd.DataFrame, policies: Sequence[str], col: str) -> list[str]:
    totals = df.groupby("policy")[col].sum()
    order = {p: i for i, p in enumerate(policies)}
    return sorted(policies, key=lambda p: (float(totals.get(p, np.inf)), order[p]))


def evaluate_cells(
    df: pd.DataFrame,
    presets: Sequence[str],
    tps_grid: Sequence[float],
    cell_fn: Callable[[pd.DataFrame], tuple[float, bool]],
) -> dict[tuple[str, float], tuple[float, bool]]:
    out = {}
    for p in presets:
        for tps in tps_grid:
            sub = df[(df["preset"] == p) & (df["decode_tps"] == tps)]
            out[(p, tps)] = cell_fn(sub)
    return out
