from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cache_delegation.config import Catalog, load_catalog
from cache_delegation.types import ActionType, Step, Trajectory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return load_catalog(ROOT / "configs")


def make_traj(
    actions: list[ActionType] | None = None,
    n: int = 10,
    prefix: int = 5000,
    out: int = 150,
    obs: int = 800,
    traj_id: str = "t0",
    obs_list: list[int] | None = None,
    measured: float | None = None,
) -> Trajectory:
    actions = actions or [ActionType.READ_SEARCH, ActionType.WRITE_EDIT] * (n // 2) + (
        [ActionType.NAVIGATE] if n % 2 else []
    )
    steps = []
    ctx = prefix
    for i, a in enumerate(actions):
        o = obs_list[i] if obs_list else obs
        subtype = {
            ActionType.READ_SEARCH: "view_file",
            ActionType.WRITE_EDIT: "str_replace",
            ActionType.TEST_VALIDATE: "test_runner",
            ActionType.DIAGNOSE_REPRODUCE: "run_script",
            ActionType.NAVIGATE: "navigate",
            ActionType.WAIT: "think",
        }[a]
        steps.append(
            Step(
                index=i,
                action=a,
                subtype=subtype,
                tool_name="x",
                context_tokens=ctx,
                new_tokens=prefix if i == 0 else (obs_list[i - 1] if obs_list else obs),
                output_tokens=out,
                obs_tokens=o,
                output_bytes=out * 4,
                obs_bytes=o * 4,
                measured_tool_s=measured,
                files=(f"/repo/f{i % 3}.py",),
                sig_text=f"step {i} file f{i % 3} {a.value}",
            )
        )
        ctx += out + o
    return Trajectory(
        traj_id=traj_id,
        source="synthetic",
        instance_id="i",
        repo="r",
        resolved=True,
        exit_status="submit",
        prefix_tokens=prefix,
        prefix_bytes=prefix * 4,
        task_text="fix the bug in f0",
        steps=tuple(steps),
    )


def quiet_device(dev):
    """Device without faults (deterministic tiers) for focused tests."""
    return replace(dev, fault=replace(dev.fault, enabled=False))


MIXED = [
    ActionType.WRITE_EDIT,
    ActionType.NAVIGATE,
    ActionType.DIAGNOSE_REPRODUCE,
    ActionType.TEST_VALIDATE,
    ActionType.NAVIGATE,
]


def mixed_traj(n: int = 30, obs: int = 3000, prefix: int = 5000, traj_id: str = "m0") -> Trajectory:
    """KeyStep delegates 1 step in 5 (the Write/Edit), leaving local gaps in between."""
    return make_traj(
        actions=[MIXED[i % 5] for i in range(n)], obs=obs, prefix=prefix, traj_id=traj_id
    )
