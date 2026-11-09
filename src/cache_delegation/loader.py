"""Load public coding-agent trajectories and normalize them into ``Trajectory`` / ``Step``.

Primary source: ``nebius/SWE-rebench-openhands-trajectories`` (OpenHands v0.54, native function
calling, Qwen3-Coder-480B, 67,074 trajectories over 1,823 real GitHub repos; CC-BY-4.0).
Robustness source: ``SWE-bench/SWE-smith-trajectories`` (SWE-agent, Claude 3.7 Sonnet, XML tool
calls inside assistant text).

Token counts use the Qwen3.5-9B tokenizer (``Qwen/Qwen3.5-9B/tokenizer.json``) and a renderer that
mirrors the model's chat template (tools block in the system turn, ``<tool_call><function=..>``
XML for calls, ``<tool_response>`` wrapped in a user turn for observations).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from .classify import classify_call, measured_duration, touched_files
from .types import ActionType, Step, Trajectory

_TOOLS_PREAMBLE = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_TOOLS_SUFFIX = (
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is no "
    "function call available, answer the question like normal with your current knowledge and do "
    "not tell the user about function calls\n</IMPORTANT>"
)
SIG_CHARS = 600


def _strip_none(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_strip_none(v) for v in x]
    return x


def render_prefix(system: str, tools: Sequence[dict] | None, task: str) -> str:
    out = ["<|im_start|>system\n"]
    if tools:
        out.append(_TOOLS_PREAMBLE)
        for t in tools:
            out.append("\n" + json.dumps(t, sort_keys=True, ensure_ascii=False))
        out.append(_TOOLS_SUFFIX)
        if system.strip():
            out.append("\n\n" + system.strip())
    else:
        out.append(system.strip())
    out.append("<|im_end|>\n<|im_start|>user\n" + task.strip() + "<|im_end|>\n")
    return "".join(out)


def render_assistant(content: str, calls: Sequence[tuple[str, dict]]) -> str:
    """Rendered assistant turn incl. generation header (counted as output of the step)."""
    content = (content or "").strip()
    s = "<|im_start|>assistant\n<think>\n\n</think>\n\n" + content
    for j, (name, args) in enumerate(calls):
        if j == 0:
            s += ("\n\n" if content else "") + f"<tool_call>\n<function={name}>\n"
        else:
            s += f"\n<tool_call>\n<function={name}>\n"
        for k, v in args.items():
            vs = json.dumps(v, ensure_ascii=False) if isinstance(v, dict | list) else str(v)
            s += f"<parameter={k}>\n{vs}\n</parameter>\n"
        s += "</function>\n</tool_call>"
    return s + "<|im_end|>\n"


def render_observation(parts: Sequence[tuple[str, str]]) -> str:
    """Render the messages between two assistant turns (tool results and/or user messages)."""
    out: list[str] = []
    in_tool = False
    for role, content in parts:
        content = (content or "").strip()
        if role == "tool":
            if not in_tool:
                out.append("<|im_start|>user")
                in_tool = True
            out.append("\n<tool_response>\n" + content + "\n</tool_response>")
        else:
            if in_tool:
                out.append("<|im_end|>\n")
                in_tool = False
            out.append("<|im_start|>user\n" + content + "<|im_end|>\n")
    if in_tool:
        out.append("<|im_end|>\n")
    return "".join(out)


@dataclass(frozen=True, slots=True)
class RawStep:
    content: str
    calls: tuple[tuple[str, dict], ...]
    obs_parts: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class RawTrajectory:
    traj_id: str
    source: str
    instance_id: str
    repo: str
    resolved: bool
    exit_status: str
    system: str
    tools: tuple[dict, ...]
    task: str
    steps: tuple[RawStep, ...]


# ------------------------------------------------------------------------------------------------
# Source adapters
# ------------------------------------------------------------------------------------------------
def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(raw or "{}")
        return val if isinstance(val, dict) else {"_raw": raw}
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw}


def _openhands_row(row: dict) -> RawTrajectory | None:
    msgs = row["trajectory"]
    if len(msgs) < 3 or msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
        return None
    steps: list[RawStep] = []
    cur: dict | None = None
    obs: list[tuple[str, str]] = []
    for m in msgs[2:]:
        if m["role"] == "assistant":
            if cur is not None:
                steps.append(RawStep(cur["content"], cur["calls"], tuple(obs)))
            calls = tuple(
                (tc["function"]["name"], _parse_args(tc["function"]["arguments"]))
                for tc in (m.get("tool_calls") or [])
            )
            cur = {"content": m.get("content") or "", "calls": calls}
            obs = []
        elif cur is not None:
            obs.append((m["role"], m.get("content") or ""))
    if cur is not None:
        steps.append(RawStep(cur["content"], cur["calls"], tuple(obs)))
    tools = tuple(_strip_none(t) for t in (row.get("tools") or []))
    return RawTrajectory(
        traj_id=row["trajectory_id"],
        source="openhands",
        instance_id=row["instance_id"],
        repo=row["repo"],
        resolved=bool(row["resolved"]),
        exit_status=row["exit_status"] or "",
        system=msgs[0]["content"] or "",
        tools=tools,
        task=msgs[1]["content"] or "",
        steps=tuple(steps),
    )


_XML_CALL = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|$)", re.S)
_XML_PARAM = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)


def _content_text(c: Any) -> str:
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c or ""


def _swesmith_row(row: dict) -> RawTrajectory | None:
    msgs = json.loads(row["messages"])
    if len(msgs) < 3 or msgs[0]["role"] != "system":
        return None
    steps: list[RawStep] = []
    cur: dict | None = None
    obs: list[tuple[str, str]] = []
    for m in msgs[2:]:
        text = _content_text(m.get("content"))
        if m["role"] == "assistant":
            if cur is not None:
                steps.append(RawStep(cur["content"], cur["calls"], tuple(obs)))
            calls = []
            first = None
            if m.get("tool_calls"):  # structured function calling
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    calls.append((fn.get("name", ""), _parse_args(fn.get("arguments"))))
            else:  # XML calls inline in the assistant text
                first = _XML_CALL.search(text)
                for name, body in _XML_CALL.findall(text):
                    calls.append((name, dict(_XML_PARAM.findall(body))))
            content = text[: first.start()] if first else text
            cur = {"content": content, "calls": tuple(calls)}
            obs = []
        elif cur is not None:
            # SWE-agent returns tool output as a user turn prefixed with OBSERVATION:
            obs.append(("tool", text.removeprefix("OBSERVATION:\n")))
    if cur is not None:
        steps.append(RawStep(cur["content"], cur["calls"], tuple(obs)))
    iid = row["instance_id"]
    return RawTrajectory(
        traj_id=row["traj_id"],
        source="swesmith",
        instance_id=iid,
        repo=iid.rsplit(".", 1)[0],
        resolved=bool(row["resolved"]),
        exit_status="",
        system=_content_text(msgs[0]["content"]),
        tools=(),
        task=_content_text(msgs[1]["content"]),
        steps=tuple(steps),
    )


def sample_indices(n_rows: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = min(n, n_rows)
    return np.sort(rng.choice(n_rows, size=n, replace=False))


def iter_raw(
    path: Path, source: str, indices: Iterable[int] | None = None
) -> Iterator[RawTrajectory]:
    """Stream raw trajectories (optionally a subset of global row indices) from a parquet file."""
    pf = pq.ParquetFile(path)
    adapter = _openhands_row if source == "openhands" else _swesmith_row
    wanted = None if indices is None else set(int(i) for i in indices)
    offset = 0
    for rg in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        local = None
        if wanted is not None:
            local = sorted(i - offset for i in wanted if offset <= i < offset + n)
            if not local:
                offset += n
                continue
        table = pf.read_row_group(rg)
        if local is not None:
            table = table.take(local)
        for row in table.to_pylist():
            raw = adapter(row)
            if raw is not None and raw.steps:
                yield raw
        offset += n


# ------------------------------------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------------------------------------
_ACTION_PRIORITY = [
    ActionType.WRITE_EDIT,
    ActionType.TEST_VALIDATE,
    ActionType.DIAGNOSE_REPRODUCE,
    ActionType.READ_SEARCH,
    ActionType.NAVIGATE,
    ActionType.WAIT,
]


def normalize(raw: RawTrajectory, tokenizer: Tokenizer) -> Trajectory:
    prefix_text = render_prefix(raw.system, raw.tools, raw.task)
    out_texts = [render_assistant(s.content, s.calls) for s in raw.steps]
    obs_texts = [render_observation(s.obs_parts) for s in raw.steps]
    enc = tokenizer.encode_batch([prefix_text, *out_texts, *obs_texts], add_special_tokens=False)
    n = len(raw.steps)
    prefix_tok = len(enc[0].ids)
    out_tok = [len(e.ids) for e in enc[1 : 1 + n]]
    obs_tok = [len(e.ids) for e in enc[1 + n :]]

    steps: list[Step] = []
    ctx = prefix_tok
    for i, s in enumerate(raw.steps):
        obs_joined = "\n".join(c for _, c in s.obs_parts)
        labels = []
        files: list[str] = []
        for name, args in s.calls or ((None, {}),):
            labels.append(classify_call(name, args, obs_joined))
            files.extend(touched_files(name, args))
        action, subtype = min(labels, key=lambda a: _ACTION_PRIORITY.index(a[0]))
        measured, _kind = measured_duration(obs_joined)
        call_desc = " ".join(f"{n_}:{json.dumps(a, ensure_ascii=False)[:300]}" for n_, a in s.calls)
        steps.append(
            Step(
                index=i,
                action=action,
                subtype=subtype,
                tool_name=s.calls[0][0] if s.calls else "",
                context_tokens=ctx,
                new_tokens=prefix_tok if i == 0 else obs_tok[i - 1],
                output_tokens=out_tok[i],
                obs_tokens=obs_tok[i],
                output_bytes=len(out_texts[i].encode()),
                obs_bytes=len(obs_texts[i].encode()),
                measured_tool_s=measured,
                files=tuple(dict.fromkeys(files)),
                sig_text=(call_desc + "\n" + obs_joined)[:SIG_CHARS],
            )
        )
        ctx += out_tok[i] + obs_tok[i]
    return Trajectory(
        traj_id=raw.traj_id,
        source=raw.source,
        instance_id=raw.instance_id,
        repo=raw.repo,
        resolved=raw.resolved,
        exit_status=raw.exit_status,
        prefix_tokens=prefix_tok,
        prefix_bytes=len(prefix_text.encode()),
        task_text=raw.task[:4000],
        steps=tuple(steps),
    )


# ------------------------------------------------------------------------------------------------
# Persistence: one row per step + one row per trajectory (parquet)
# ------------------------------------------------------------------------------------------------
STEP_COLUMNS = [
    "traj_id",
    "index",
    "action",
    "subtype",
    "tool_name",
    "context_tokens",
    "new_tokens",
    "output_tokens",
    "obs_tokens",
    "output_bytes",
    "obs_bytes",
    "measured_tool_s",
    "files",
    "sig_text",
]
TRAJ_COLUMNS = [
    "traj_id",
    "source",
    "instance_id",
    "repo",
    "resolved",
    "exit_status",
    "prefix_tokens",
    "prefix_bytes",
    "task_text",
    "n_steps",
    "final_context_tokens",
]


def to_frames(trajs: Sequence[Trajectory]):
    import pandas as pd

    srows, trows = [], []
    for t in trajs:
        trows.append(
            {
                "traj_id": t.traj_id,
                "source": t.source,
                "instance_id": t.instance_id,
                "repo": t.repo,
                "resolved": t.resolved,
                "exit_status": t.exit_status,
                "prefix_tokens": t.prefix_tokens,
                "prefix_bytes": t.prefix_bytes,
                "task_text": t.task_text,
                "n_steps": t.n_steps,
                "final_context_tokens": t.final_context_tokens,
            }
        )
        for s in t.steps:
            srows.append(
                {
                    "traj_id": t.traj_id,
                    "index": s.index,
                    "action": s.action.value,
                    "subtype": s.subtype,
                    "tool_name": s.tool_name,
                    "context_tokens": s.context_tokens,
                    "new_tokens": s.new_tokens,
                    "output_tokens": s.output_tokens,
                    "obs_tokens": s.obs_tokens,
                    "output_bytes": s.output_bytes,
                    "obs_bytes": s.obs_bytes,
                    "measured_tool_s": s.measured_tool_s,
                    "files": list(s.files),
                    "sig_text": s.sig_text,
                }
            )
    return pd.DataFrame(trows, columns=TRAJ_COLUMNS), pd.DataFrame(srows, columns=STEP_COLUMNS)


def from_frames(traj_df, step_df) -> list[Trajectory]:
    out: list[Trajectory] = []
    groups = {k: g.sort_values("index") for k, g in step_df.groupby("traj_id", sort=False)}
    for r in traj_df.itertuples(index=False):
        g = groups.get(r.traj_id)
        steps: tuple[Step, ...] = ()
        if g is not None:
            steps = tuple(
                Step(
                    index=int(s.index),
                    action=ActionType(s.action),
                    subtype=s.subtype,
                    tool_name=s.tool_name,
                    context_tokens=int(s.context_tokens),
                    new_tokens=int(s.new_tokens),
                    output_tokens=int(s.output_tokens),
                    obs_tokens=int(s.obs_tokens),
                    output_bytes=int(s.output_bytes),
                    obs_bytes=int(s.obs_bytes),
                    measured_tool_s=None
                    if s.measured_tool_s != s.measured_tool_s or s.measured_tool_s is None
                    else float(s.measured_tool_s),
                    files=tuple(s.files) if s.files is not None else (),
                    sig_text=s.sig_text or "",
                )
                for s in g.itertuples(index=False)
            )
        out.append(
            Trajectory(
                traj_id=r.traj_id,
                source=r.source,
                instance_id=r.instance_id,
                repo=r.repo,
                resolved=bool(r.resolved),
                exit_status=r.exit_status or "",
                prefix_tokens=int(r.prefix_tokens),
                prefix_bytes=int(r.prefix_bytes),
                task_text=r.task_text or "",
                steps=steps,
            )
        )
    return out


def load_processed(directory: Path, name: str) -> list[Trajectory]:
    import pandas as pd

    tdf = pd.read_parquet(directory / f"{name}_trajectories.parquet")
    sdf = pd.read_parquet(directory / f"{name}_steps.parquet")
    return from_frames(tdf, sdf)
