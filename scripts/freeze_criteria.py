"""Freeze configs/kill_criteria.yaml (records its SHA-256; refuses to re-freeze)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache_delegation.preregistration import freeze  # noqa: E402

if __name__ == "__main__":
    lock = freeze(
        ROOT / "configs" / "kill_criteria.yaml",
        ROOT / "configs" / "kill_criteria.lock.json",
        note="Frozen before any simulation on real trajectories.",
    )
    print(json.dumps(lock, indent=2))
