"""Map agent tool calls onto the EdgeWise six-way step taxonomy.

The rules are deterministic and documented here because the KeyStep policy and the tool-time
model both depend on them. Each step gets a coarse ``ActionType`` and a fine ``subtype`` that
the tool-time model keys on.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from typing import Any

from .types import ActionType

# ---------------------------------------------------------------------------------------------
# Measured durations found inside tool observations (the source container's own timings).
# ---------------------------------------------------------------------------------------------
_PYTEST_RE = re.compile(
    r"=+ [^\n=]*?(?:passed|failed|errors?|skipped|deselected|xfailed|xpassed|no tests ran)"
    r"[^\n=]*? in (\d+(?:\.\d+)?)s"
)
_UNITTEST_RE = re.compile(r"Ran \d+ tests? in (\d+(?:\.\d+)?)s")
_NO_OUTPUT_RE = re.compile(r"no new output after (\d+(?:\.\d+)?) seconds", re.I)
_TIMED_OUT_RE = re.compile(r"timed out after (\d+(?:\.\d+)?) seconds", re.I)


def measured_duration(obs: str) -> tuple[float | None, str | None]:
    """Return (seconds, kind) if the observation reports how long the tool ran."""
    if not obs:
        return None, None
    for regex, kind in (
        (_TIMED_OUT_RE, "timeout"),
        (_NO_OUTPUT_RE, "no_output"),
        (_PYTEST_RE, "pytest"),
        (_UNITTEST_RE, "unittest"),
    ):
        found = regex.findall(obs)
        if found:
            return float(found[-1]), kind
    return None, None


# ---------------------------------------------------------------------------------------------
# Bash command classification
# ---------------------------------------------------------------------------------------------
_TEST_PAT = re.compile(
    r"(?:^|[\s/])(?:pytest|py\.test|tox|nox|unittest|nosetests|trial)\b"
    r"|manage\.py\s+test|go\s+test|cargo\s+test|npm\s+(?:run\s+)?test|yarn\s+test"
    r"|make\s+(?:test|check)|mvn\s+test|gradle\w*\s+test|rspec|phpunit|ctest|jest|mocha"
)
_BUILD_PAT = re.compile(
    r"pip3?\s+install|conda\s+install|poetry\s+install|setup\.py\s+(?:build|develop|install)"
    r"|npm\s+(?:install|ci|run\s+build)|yarn\s+(?:install|build)|cargo\s+build|go\s+build"
    r"|(?:^|\s)make(?:\s|$)|cmake|mvn\s+(?:compile|package|install)|gradle\w*\s+build"
    r"|maturin|cythonize|python3?\s+-m\s+build|uv\s+pip\s+install|apt(?:-get)?\s+install"
)
_WRITE_PAT = re.compile(
    r"sed\s+-i|perl\s+-pi|(?:^|[^<>2&])>{1,2}\s*(?!/dev/null)[\w./~-]+|\btee\b|\bpatch\b"
    r"|git\s+apply"
    r"|git\s+(?:checkout|stash|reset|commit|add)\b|(?:^|\s)(?:mv|cp|rm|mkdir|touch|chmod)\s"
)
_SEARCH_CMDS = {"grep", "rg", "ag", "ack", "egrep", "fgrep", "git grep"}
_READ_CMDS = {"cat", "head", "tail", "less", "more", "nl", "wc", "awk", "diff", "cmp", "xxd"}
_GIT_READ = re.compile(r"git\s+(?:log|show|diff|status|blame|branch|rev-parse|ls-files)\b")
_NAV_CMDS = {"ls", "cd", "pwd", "tree", "find", "which", "whereis", "du", "file", "stat", "fd"}
_PY_RUN = re.compile(r"(?:^|\s)(?:python3?|python3\.\d+)(?:\s+-u)?\s+(?!-m\b|-c\b)([^\s;|&]+)")
_PY_C = re.compile(r"(?:^|\s)python3?(?:\.\d+)?\s+-c\b")
_PY_M = re.compile(r"(?:^|\s)python3?(?:\.\d+)?\s+-m\s+([\w.]+)")
_SLEEP = re.compile(r"^\s*sleep\s+(\d+(?:\.\d+)?)")


def _segments(cmd: str) -> list[str]:
    parts = re.split(r"\s*(?:&&|\|\||;|\n)\s*", cmd)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"^(?:\w+=\S+\s+)+", "", p)  # leading env assignments
        p = re.sub(r"^timeout\s+\d+[smh]?\s+", "", p)
        out.append(p)
    return out


def _first_word(seg: str) -> str:
    try:
        toks = shlex.split(seg, posix=True)
    except ValueError:
        toks = seg.split()
    return toks[0] if toks else ""


def classify_bash(command: str, is_input: bool = False) -> tuple[ActionType, str]:
    """Classify a bash command. Priority: wait > test > build > write > diagnose > read > nav."""
    cmd = (command or "").strip()
    if is_input or cmd == "" or cmd in {"C-c", "C-d", "C-z"}:
        return ActionType.WAIT, "wait_process"
    if _SLEEP.match(cmd) and len(_segments(cmd)) == 1:
        return ActionType.WAIT, "sleep"
    if _TEST_PAT.search(cmd):
        return ActionType.TEST_VALIDATE, "test_runner"
    if _BUILD_PAT.search(cmd):
        return ActionType.TEST_VALIDATE, "build_install"
    py_m = _PY_M.search(cmd)
    if py_m and py_m.group(1) in {"pytest", "unittest", "nose", "tox"}:
        return ActionType.TEST_VALIDATE, "test_runner"
    run = _PY_RUN.search(cmd)
    if run:
        script = run.group(1).rsplit("/", 1)[-1].lower()
        if script.startswith("test") or any(k in script for k in ("verify", "check", "valid")):
            return ActionType.TEST_VALIDATE, "test_script"
        return ActionType.DIAGNOSE_REPRODUCE, "run_script"
    if _PY_C.search(cmd) or py_m:
        return ActionType.DIAGNOSE_REPRODUCE, "python_inline"
    if _WRITE_PAT.search(cmd) and not cmd.lstrip().startswith(("grep", "find", "ls")):
        return ActionType.WRITE_EDIT, "shell_write"
    words = [_first_word(s) for s in _segments(cmd)]
    if any(w in _SEARCH_CMDS for w in words) or "git grep" in cmd:
        return ActionType.READ_SEARCH, "search"
    if any(w in _READ_CMDS for w in words) or re.search(r"sed\s+-n", cmd) or _GIT_READ.search(cmd):
        return ActionType.READ_SEARCH, "read"
    if any(w in _NAV_CMDS for w in words):
        return ActionType.NAVIGATE, "navigate"
    if any(w in {"echo", "printf", "export", "source", "true"} for w in words):
        return ActionType.NAVIGATE, "shell_misc"
    return ActionType.DIAGNOSE_REPRODUCE, "shell_other"


_DIR_VIEW_MARKERS = ("files and directories up to", "Here's the files and directories")


def classify_call(
    tool_name: str | None, args: Mapping[str, Any], obs: str
) -> tuple[ActionType, str]:
    """Classify one tool call given its parsed arguments and observation text."""
    if not tool_name:
        return ActionType.WAIT, "message"
    name = tool_name.lower()
    if name in {"execute_bash", "bash", "execute_ipython_cell"}:
        if name == "execute_ipython_cell":
            return ActionType.DIAGNOSE_REPRODUCE, "python_inline"
        is_input = str(args.get("is_input", "false")).lower() == "true"
        return classify_bash(str(args.get("command", "")), is_input)
    if name in {"str_replace_editor", "edit", "str_replace_based_edit_tool"}:
        command = str(args.get("command", "")).lower()
        if command == "view":
            if any(m in (obs or "")[:200] for m in _DIR_VIEW_MARKERS):
                return ActionType.NAVIGATE, "view_dir"
            return ActionType.READ_SEARCH, "view_file"
        if command == "create":
            return ActionType.WRITE_EDIT, "create"
        if command in {"str_replace", "insert", "undo_edit"}:
            return ActionType.WRITE_EDIT, command
        return ActionType.WRITE_EDIT, "edit_other"
    if name in {"finish", "submit"}:
        return ActionType.WAIT, "finish"
    if name == "think":
        return ActionType.WAIT, "think"
    if name == "task_tracker":
        return ActionType.WAIT, "plan"
    if name in {"search", "find_file", "search_dir", "search_file"}:
        return ActionType.READ_SEARCH, "search"
    return ActionType.DIAGNOSE_REPRODUCE, f"tool_{name}"


_PATH_RE = re.compile(
    r"(?:/[\w.@+-]+)+\.(?:py|pyx|pyi|js|ts|tsx|go|rs|java|c|h|cc|cpp|rb|toml|cfg|"
    r"ini|yaml|yml|json|md|rst|txt)\b"
)


def touched_files(tool_name: str | None, args: Mapping[str, Any]) -> tuple[str, ...]:
    if not tool_name:
        return ()
    if "path" in args and isinstance(args["path"], str):
        return (args["path"],)
    cmd = args.get("command")
    if isinstance(cmd, str):
        return tuple(dict.fromkeys(_PATH_RE.findall(cmd)))
    return ()
