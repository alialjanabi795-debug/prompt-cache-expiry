"""Tool execution time on the trajectory's source container (before device scaling).

Deterministic per (seed, trajectory, step): the same draw is used by every policy and by both the
nominal and the dynamic device arms, so differences between arms come only from the device model.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .config import ToolTimeParams
from .types import ActionType, Step

_TEST_SUBTYPES = {"test_runner", "test_script"}


def stable_seed(*parts: object) -> int:
    h = hashlib.blake2b("\x1f".join(map(str, parts)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def keyed_rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


class ToolTimer:
    def __init__(self, params: ToolTimeParams, seed: int = 0) -> None:
        self.params = params
        self.seed = seed

    def source_seconds(self, traj_id: str, step: Step) -> float:
        p = self.params
        rng = keyed_rng("tool", self.seed, traj_id, step.index)
        z = rng.standard_normal()
        if (
            p.use_measured
            and step.measured_tool_s is not None
            and step.tool_name in p.measured_tools
            and step.measured_tool_s <= p.measured_max_s
        ):
            base = step.measured_tool_s
            if step.subtype in _TEST_SUBTYPES or step.action == ActionType.TEST_VALIDATE:
                base += p.measured_overhead_s
            return base * math.exp(p.measured_jitter_sigma * z)
        prior = p.priors.get(step.subtype, p.default)
        if prior.median_s <= 0:
            return 0.0
        return prior.median_s * math.exp(prior.sigma * z)
