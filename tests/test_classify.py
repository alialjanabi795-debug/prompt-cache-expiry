import pytest

from cache_delegation.classify import classify_bash, classify_call, measured_duration
from cache_delegation.types import ActionType as A


@pytest.mark.parametrize(
    "cmd,expected",
    [
        ("cd /workspace/x && python -m pytest tests/ -v", (A.TEST_VALIDATE, "test_runner")),
        ("pytest -q tests/test_a.py::test_b", (A.TEST_VALIDATE, "test_runner")),
        ("cd /repo && pip install -e .", (A.TEST_VALIDATE, "build_install")),
        ("python reproduce_issue.py", (A.DIAGNOSE_REPRODUCE, "run_script")),
        ("python test_fix.py", (A.TEST_VALIDATE, "test_script")),
        ('python -c "import x; print(x)"', (A.DIAGNOSE_REPRODUCE, "python_inline")),
        ("grep -rn 'foo' src/", (A.READ_SEARCH, "search")),
        ("cat setup.py", (A.READ_SEARCH, "read")),
        ("sed -n '10,40p' a.py", (A.READ_SEARCH, "read")),
        ("git log --oneline -10", (A.READ_SEARCH, "read")),
        ("ls -la", (A.NAVIGATE, "navigate")),
        ("find . -name '*.py' | head -20", (A.NAVIGATE, "navigate")),
        ("find . -name '*.py' 2>/dev/null", (A.NAVIGATE, "navigate")),
        ("sed -i 's/a/b/' x.py", (A.WRITE_EDIT, "shell_write")),
        ("cat > /tmp/t.py << 'EOF'\nprint(1)\nEOF", (A.WRITE_EDIT, "shell_write")),
        ("", (A.WAIT, "wait_process")),
        ("C-c", (A.WAIT, "wait_process")),
        ("sleep 30", (A.WAIT, "sleep")),
    ],
)
def test_classify_bash(cmd, expected):
    assert classify_bash(cmd) == expected


def test_editor_calls():
    assert classify_call(
        "str_replace_editor",
        {"command": "view", "path": "/a.py"},
        "Here's the result of running `cat -n`",
    ) == (A.READ_SEARCH, "view_file")
    assert classify_call(
        "str_replace_editor",
        {"command": "view", "path": "/repo"},
        "Here's the files and directories up to 2 levels deep in /repo",
    ) == (A.NAVIGATE, "view_dir")
    assert classify_call("str_replace_editor", {"command": "str_replace"}, "")[0] == A.WRITE_EDIT
    assert classify_call("str_replace_editor", {"command": "create"}, "")[0] == A.WRITE_EDIT
    assert classify_call("think", {}, "") == (A.WAIT, "think")
    assert classify_call("finish", {}, "") == (A.WAIT, "finish")
    assert classify_call(None, {}, "") == (A.WAIT, "message")
    assert classify_call("execute_bash", {"command": "", "is_input": "true"}, "")[0] == A.WAIT


def test_measured_duration():
    obs = "....\n========== 5 passed, 1 warning in 2.31s ==========\n"
    assert measured_duration(obs) == (2.31, "pytest")
    assert measured_duration("Ran 12 tests in 0.004s\n\nOK") == (0.004, "unittest")
    assert measured_duration("[The command has no new output after 30 seconds.]")[0] == 30.0
    assert measured_duration("hello") == (None, None)
