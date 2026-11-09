"""Delegation policies replayed on fixed trajectories (decisions only, no generation).

All policies are oracle labelers over the recorded trajectory (as in EdgeWise's analysis of
recorded trajectories): KeyStep sees the step's action type and observation length.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .timing import keyed_rng
from .types import ActionType, Tier, Trajectory


@dataclass(frozen=True, slots=True)
class DecisionContext:
    now: float
    prev_tiers: tuple[Tier, ...]  # tiers decided for steps < i (requested, before fallback)
    local_prefill_needed: int  # tokens the local model would have to prefill for step i
    local_kv_evicted: bool  # True if the local KV was dropped since the last local step


class Policy(ABC):
    name: str = "policy"

    def reset(self, traj: Trajectory) -> None:  # noqa: B027 - optional hook
        pass

    @abstractmethod
    def decide(self, traj: Trajectory, i: int, ctx: DecisionContext) -> Tier: ...


class AllLocal(Policy):
    name = "all_local"

    def decide(self, traj: Trajectory, i: int, ctx: DecisionContext) -> Tier:
        return Tier.LOCAL


class AllCloud(Policy):
    name = "all_cloud"

    def decide(self, traj: Trajectory, i: int, ctx: DecisionContext) -> Tier:
        return Tier.CLOUD


class RandomP(Policy):
    def __init__(self, p: float, seed: int = 0) -> None:
        self.p = p
        self.seed = seed
        self.name = f"random_{p:g}"

    def decide(self, traj: Trajectory, i: int, ctx: DecisionContext) -> Tier:
        u = keyed_rng("random_policy", self.seed, self.p, traj.traj_id, i).random()
        return Tier.CLOUD if u < self.p else Tier.LOCAL


class KeyStep(Policy):
    """EdgeWise key-step rule: delegate Write/Edit steps and long Read/Search steps.

    EdgeWise's judge is prompted with "Write/Edit and complex Read/Search operations" and gives no
    numeric threshold; here "long" = the step's observation has >= ``long_read_tokens`` tokens
    (pre-registered in configs/kill_criteria.yaml).
    """

    def __init__(self, long_read_tokens: int = 1000) -> None:
        self.long_read_tokens = long_read_tokens
        self.name = "keystep" if long_read_tokens == 1000 else f"keystep_L{long_read_tokens}"

    def is_key(self, traj: Trajectory, i: int) -> bool:
        s = traj.steps[i]
        if s.action == ActionType.WRITE_EDIT:
            return True
        return s.action == ActionType.READ_SEARCH and s.obs_tokens >= self.long_read_tokens

    def decide(self, traj: Trajectory, i: int, ctx: DecisionContext) -> Tier:
        return Tier.CLOUD if self.is_key(traj, i) else Tier.LOCAL


class Sticky(Policy):
    """Session-aware / sticky routing: after the base policy delegates, stay on the cloud for
    ``k`` more steps (keeps the cloud session and its cache hot, like vLLM session affinity)."""

    def __init__(self, base: Policy, k: int = 2) -> None:
        self.base = base
        self.k = k
        self.name = f"sticky{k}_{base.name}"
        self._last_base_cloud = -(10**9)

    def reset(self, traj: Trajectory) -> None:
        self.base.reset(traj)
        self._last_base_cloud = -(10**9)

    def decide(self, traj: Trajectory, i: int, ctx: DecisionContext) -> Tier:
        if self.base.decide(traj, i, ctx) == Tier.CLOUD:
            self._last_base_cloud = i
            return Tier.CLOUD
        return Tier.CLOUD if i - self._last_base_cloud <= self.k else Tier.LOCAL


class DelegateEarly(Policy):
    """K7: when the local KV has been evicted (the next local step would re-prefill at least
    ``min_reprefill_tokens``) and the base policy delegates within the next ``shift`` steps, start
    delegating now, so the cloud absorbs the steps that would otherwise pay the local re-prefill."""

    def __init__(self, base: KeyStep, shift: int, min_reprefill_tokens: int = 8000) -> None:
        self.base = base
        self.shift = shift
        self.min_reprefill_tokens = min_reprefill_tokens
        self.name = f"early{shift}_{base.name}"
        self._labels: list[bool] = []

    def reset(self, traj: Trajectory) -> None:
        self._labels = [self.base.is_key(traj, j) for j in range(traj.n_steps)]

    def decide(self, traj: Trajectory, i: int, ctx: DecisionContext) -> Tier:
        if self._labels[i]:
            return Tier.CLOUD
        if ctx.local_kv_evicted and ctx.local_prefill_needed >= self.min_reprefill_tokens:
            upcoming = self._labels[i + 1 : i + 1 + self.shift]
            if any(upcoming):
                return Tier.CLOUD
        return Tier.LOCAL


def make_policy(spec: str, seed: int = 0) -> Policy:
    """Parse a policy spec: all_local | all_cloud | random_<p> | keystep[_L<n>] |
    sticky<k>_<base> | early<s>_<keystep spec>."""
    if spec == "all_local":
        return AllLocal()
    if spec == "all_cloud":
        return AllCloud()
    if spec.startswith("random_"):
        return RandomP(float(spec.split("_", 1)[1]), seed)
    if spec.startswith("keystep"):
        rest = spec[len("keystep") :]
        return KeyStep(int(rest[2:]) if rest.startswith("_L") else 1000)
    if spec.startswith("sticky"):
        k, base = spec[len("sticky") :].split("_", 1)
        return Sticky(make_policy(base, seed), int(k))
    if spec.startswith("early"):
        s, base = spec[len("early") :].split("_", 1)
        b = make_policy(base, seed)
        assert isinstance(b, KeyStep)
        return DelegateEarly(b, int(s))
    raise ValueError(f"unknown policy spec {spec!r}")
