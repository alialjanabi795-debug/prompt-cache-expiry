"""Core data types shared by the loader, the simulator and the analysis code."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ActionType(StrEnum):
    """EdgeWise six-way step taxonomy (arXiv:2610.17560, Sec. on step types)."""

    WRITE_EDIT = "write_edit"
    READ_SEARCH = "read_search"
    DIAGNOSE_REPRODUCE = "diagnose_reproduce"
    TEST_VALIDATE = "test_validate"
    NAVIGATE = "navigate"
    WAIT = "wait"


class Tier(StrEnum):
    LOCAL = "local"
    EDGE = "edge"
    CLOUD = "cloud"


@dataclass(frozen=True, slots=True)
class Step:
    """One agent step = one LLM call (assistant turn) + the tool execution it triggers.

    Token accounting (all in Qwen3.5 tokens, rendered with the Qwen chat format):
      context_tokens[i] = prompt length of the LLM call of step i
                        = context_tokens[i-1] + output_tokens[i-1] + obs_tokens[i-1]
      new_tokens[i]     = tokens appended since the previous LLM call that the previous call did
                          not generate itself (= obs_tokens[i-1]; = prefix for step 0)
    """

    index: int
    action: ActionType
    subtype: str
    tool_name: str
    context_tokens: int
    new_tokens: int
    output_tokens: int
    obs_tokens: int
    output_bytes: int
    obs_bytes: int
    measured_tool_s: float | None = None
    files: tuple[str, ...] = ()
    sig_text: str = ""


@dataclass(frozen=True, slots=True)
class Trajectory:
    traj_id: str
    source: str
    instance_id: str
    repo: str
    resolved: bool
    exit_status: str
    prefix_tokens: int  # system + tool schemas + task
    prefix_bytes: int
    task_text: str
    steps: tuple[Step, ...] = field(default_factory=tuple)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def final_context_tokens(self) -> int:
        if not self.steps:
            return self.prefix_tokens
        last = self.steps[-1]
        return last.context_tokens + last.output_tokens + last.obs_tokens
