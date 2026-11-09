import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from cache_delegation.loader import (
    RawStep,
    RawTrajectory,
    _openhands_row,
    _swesmith_row,
    from_frames,
    normalize,
    render_assistant,
    render_observation,
    to_frames,
)
from cache_delegation.types import ActionType


@pytest.fixture(scope="module")
def ws_tokenizer():
    """Whitespace word-level tokenizer: token count = number of whitespace-separated chunks."""
    tok = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tok


def _raw():
    steps = (
        RawStep(
            "Let me look",
            (("str_replace_editor", {"command": "view", "path": "/r/a.py"}),),
            (("tool", "Here's the result of running `cat -n` on /r/a.py\n 1 x = 1"),),
        ),
        RawStep(
            "",
            (("execute_bash", {"command": "python -m pytest -q"}),),
            (("tool", "== 3 passed in 0.50s =="),),
        ),
        RawStep("done", (("finish", {"message": "ok"}),), ()),
    )
    return RawTrajectory(
        "id1",
        "openhands",
        "inst",
        "repo",
        True,
        "submit",
        "sys prompt",
        ({"type": "function", "function": {"name": "finish"}},),
        "fix it",
        steps,
    )


def test_context_accounting_identity(ws_tokenizer):
    t = normalize(_raw(), ws_tokenizer)
    assert t.n_steps == 3
    assert t.steps[0].context_tokens == t.prefix_tokens
    for a, b in zip(t.steps, t.steps[1:], strict=False):
        assert b.context_tokens == a.context_tokens + a.output_tokens + a.obs_tokens
        assert b.new_tokens == a.obs_tokens
    assert [s.action for s in t.steps] == [
        ActionType.READ_SEARCH,
        ActionType.TEST_VALIDATE,
        ActionType.WAIT,
    ]
    assert t.steps[1].measured_tool_s == 0.5
    assert t.steps[0].files == ("/r/a.py",)
    assert t.steps[2].obs_tokens == 0


def test_frames_roundtrip(ws_tokenizer):
    t = normalize(_raw(), ws_tokenizer)
    tdf, sdf = to_frames([t])
    (back,) = from_frames(tdf, sdf)
    assert back == t


def test_render_formats():
    a = render_assistant("hi", [("execute_bash", {"command": "ls"})])
    assert a.startswith("<|im_start|>assistant\n<think>\n\n</think>\n\nhi\n\n<tool_call>")
    assert "<parameter=command>\nls\n</parameter>" in a and a.endswith("<|im_end|>\n")
    o = render_observation([("tool", "out1"), ("tool", "out2")])
    assert o == (
        "<|im_start|>user\n<tool_response>\nout1\n</tool_response>"
        "\n<tool_response>\nout2\n</tool_response><|im_end|>\n"
    )


def test_openhands_adapter_groups_observations():
    row = {
        "trajectory_id": "x",
        "instance_id": "i",
        "repo": "r",
        "resolved": 1,
        "exit_status": "submit",
        "tools": [],
        "trajectory": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "a",
                "tool_calls": [{"function": {"name": "think", "arguments": '{"thought": "t"}'}}],
            },
            {"role": "tool", "content": "logged", "name": "think"},
            {"role": "user", "content": "please continue"},
            {"role": "assistant", "content": "b", "tool_calls": None},
        ],
    }
    raw = _openhands_row(row)
    assert len(raw.steps) == 2
    assert raw.steps[0].obs_parts == (("tool", "logged"), ("user", "please continue"))
    assert raw.steps[0].calls == (("think", {"thought": "t"}),)
    assert raw.steps[1].calls == ()


def test_swesmith_adapter_structured_and_xml():
    import json

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "look",
            "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
        },
        {"role": "tool", "content": [{"type": "text", "text": "a.py"}]},
        {
            "role": "assistant",
            "content": "edit\n<function=str_replace_editor>\n<parameter=command>create</parameter>"
            "\n</function>",
        },
        {"role": "user", "content": "OBSERVATION:\nFile created"},
    ]
    row = {"messages": json.dumps(msgs), "instance_id": "o__r.1", "resolved": True, "traj_id": "t"}
    raw = _swesmith_row(row)
    assert raw.steps[0].calls == (("bash", {"command": "ls"}),)
    assert raw.steps[0].obs_parts == (("tool", "a.py"),)
    assert raw.steps[1].calls == (("str_replace_editor", {"command": "create"}),)
    assert raw.steps[1].content == "edit\n"
    assert raw.steps[1].obs_parts == (("tool", "File created"),)
