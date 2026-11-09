"""What the cloud receives on a delegated step.

Transcript segments: ``("P",)`` = system + tool schemas + task (static prefix), then for every step
j ``("o", j)`` = the assistant output and ``("b", j)`` = the observation.

Strategies (a "handoff" is the first cloud step of a run of consecutive cloud steps):
  full         every cloud call sees the full transcript (EdgeWise "Full History", Handoff-Tax Raw).
  traj_drop    Handoff-Tax Traj-drop: P + fixed "a previous agent worked on this task" message +
               the previous step; then the run appends natively.
  compact_pre  Handoff-Tax Compact_pre: the LOCAL model writes an S-token summary (device decode
               time); P + summary + previous step; then append.
  compact_suf  Handoff-Tax Compact_suf: the CLOUD writes the summary in an extra call over the full
               transcript (billed), then P + summary + previous step.
  cache_aware  Component 8: if the full-history prompt is warm in the cloud cache (live hit >=
               warm_fraction of it) send full history, otherwise send a retrieved compact context:
               P + top-k relevant earlier steps + latest view of each touched file (token budget) +
               previous step. Decided once per handoff; the run then appends natively.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .cache import Prompt, PromptBuilder, PromptCache
from .types import ActionType, Trajectory

DROP_MSG_TOKENS = 40
DROP_MSG_BYTES = 170
STRATEGIES = ("full", "traj_drop", "compact_pre", "compact_suf", "cache_aware")


class TranscriptIndex:
    """Precomputed full-history prompts for one trajectory (O(1) slicing per step)."""

    def __init__(self, traj: Trajectory) -> None:
        self.traj = traj
        b = PromptBuilder()
        b.append(("P",), traj.prefix_tokens, traj.prefix_bytes).mark_static()
        for s in traj.steps:
            b.append(("o", s.index), s.output_tokens, s.output_bytes)
            b.append(("b", s.index), s.obs_tokens, s.obs_bytes)
        self.full = b.build()

    def full_prompt(self, i: int) -> Prompt:
        """Prompt of step i's LLM call under full history (segments P, o0, b0, ..., b_{i-1})."""
        return self.full.prefix(1 + 2 * i)


@dataclass(frozen=True, slots=True)
class HandoffExtras:
    local_summary_tokens: int = 0  # tokens the device must decode (compact_pre)
    cloud_summary_prompt: Prompt | None = None  # extra billed cloud call (compact_suf)
    cloud_summary_out_tokens: int = 0
    mode: str = "full"
    retrieved_tokens: int = 0


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _tf(text: str) -> Counter:
    return Counter(w.lower() for w in _WORD.findall(text))


class Retriever:
    """Deterministic lexical stand-in for an embedding retriever (TF-IDF cosine over step text).

    Phase 0 only needs the *token count* of the retrieved context; the selection rule is kept
    simple and documented. A sentence-embedding model can replace ``_tf`` without other changes.
    """

    def __init__(self, traj: Trajectory) -> None:
        self.traj = traj
        self.tfs = [_tf(s.sig_text) for s in traj.steps]
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = max(1, len(self.tfs))
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
        self.task_tf = _tf(traj.task_text)

    def _vec(self, tf: Counter) -> dict[str, float]:
        return {w: c * self.idf.get(w, 1.0) for w, c in tf.items()}

    @staticmethod
    def _cos(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(v * b.get(w, 0.0) for w, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def select(self, h: int, k: int, budget_tokens: int) -> list[int]:
        """Steps (< h-1) to include verbatim (output+observation) for a handoff at step h."""
        if h <= 1:
            return []
        steps = self.traj.steps
        q = self._vec(self.task_tf + self.tfs[h - 1])
        cands = list(range(h - 1))
        ranked = sorted(cands, key=lambda j: (-self._cos(q, self._vec(self.tfs[j])), -j))[:k]
        # latest full-file view of every file touched before h
        latest_view: dict[str, int] = {}
        for j in cands:
            s = steps[j]
            if s.action == ActionType.READ_SEARCH and s.subtype == "view_file":
                for f in s.files:
                    latest_view[f] = j
        touched = {f for j in cands for f in steps[j].files}
        file_steps = sorted({latest_view[f] for f in touched if f in latest_view}, reverse=True)
        chosen: list[int] = []
        used = 0
        for j in [*ranked, *file_steps]:
            if j in chosen:
                continue
            cost = steps[j].output_tokens + steps[j].obs_tokens
            if used + cost > budget_tokens:
                continue
            chosen.append(j)
            used += cost
        return sorted(chosen)


class CloudContext:
    """Stateful builder of cloud prompts for one simulated trajectory."""

    def __init__(
        self,
        traj: Trajectory,
        index: TranscriptIndex,
        strategy: str,
        summary_tokens: int = 600,
        retrieval_k: int = 5,
        retrieval_budget_tokens: int = 8000,
        warm_fraction: float = 0.5,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(strategy)
        self.traj = traj
        self.index = index
        self.strategy = strategy
        self.summary_tokens = summary_tokens
        self.retrieval_k = retrieval_k
        self.retrieval_budget = retrieval_budget_tokens
        self.warm_fraction = warm_fraction
        self._run: PromptBuilder | None = None
        self._run_end = -1  # last step index whose (o,b) is in the run builder
        self._retriever: Retriever | None = None

    def _append_steps(self, b: PromptBuilder, lo: int, hi: int) -> None:
        for j in range(lo, hi):
            s = self.traj.steps[j]
            b.append(("o", j), s.output_tokens, s.output_bytes)
            b.append(("b", j), s.obs_tokens, s.obs_bytes)

    def prompt_for(
        self, i: int, prev_was_cloud: bool, cache: PromptCache, t: float
    ) -> tuple[Prompt, HandoffExtras]:
        full = self.index.full_prompt(i)
        if self.strategy == "full":
            return full, HandoffExtras()
        if i == 0:  # nothing to hand off yet: the run starts with the static prefix
            self._run = PromptBuilder()
            self._run.append(("P",), self.traj.prefix_tokens, self.traj.prefix_bytes).mark_static()
            self._run_end = -1
            return full, HandoffExtras(mode="start")
        if prev_was_cloud and self._run is not None and self._run_end == i - 2:
            # continue the run natively: append step i-1
            self._append_steps(self._run, i - 1, i)
            self._run_end = i - 1
            return self._run.build(), HandoffExtras(mode="continue")
        # ---- handoff at step i ----
        s = self.traj.steps
        b = PromptBuilder()
        b.append(("P",), self.traj.prefix_tokens, self.traj.prefix_bytes).mark_static()
        extras = HandoffExtras(mode=self.strategy)
        if self.strategy == "traj_drop":
            b.append(("drop", i), DROP_MSG_TOKENS, DROP_MSG_BYTES)
        elif self.strategy in ("compact_pre", "compact_suf"):
            b.append(("sum", i), self.summary_tokens, self.summary_tokens * 4)
            if self.strategy == "compact_pre":
                extras = HandoffExtras(local_summary_tokens=self.summary_tokens, mode="compact_pre")
            else:
                extras = HandoffExtras(
                    cloud_summary_prompt=full,
                    cloud_summary_out_tokens=self.summary_tokens,
                    mode="compact_suf",
                )
        elif self.strategy == "cache_aware":
            live, _ = cache.peek(full, t)
            if live >= self.warm_fraction * full.tokens:
                self._run = None
                return full, HandoffExtras(mode="warm_full")
            if self._retriever is None:
                self._retriever = Retriever(self.traj)
            chosen = self._retriever.select(i, self.retrieval_k, self.retrieval_budget)
            ret_tokens = sum(s[j].output_tokens + s[j].obs_tokens for j in chosen)
            ret_bytes = sum(s[j].output_bytes + s[j].obs_bytes for j in chosen)
            b.append(("ret", i), ret_tokens, ret_bytes)
            extras = HandoffExtras(mode="cold_compact", retrieved_tokens=ret_tokens)
        self._append_steps(b, i - 1, i)
        self._run = b
        self._run_end = i - 1
        return b.build(), extras
