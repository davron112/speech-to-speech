import json

from speech_to_speech.LLM.language_model import LanguageModelHandler, StreamContext
from speech_to_speech.LLM.tool_call.function_call import (
    alias_native_tool_markers,
    extract_function_calls_from_text,
    native_function_calls_to_python,
)
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.tool_prompt import END_CODE, ENTER_CODE, build_block_regex

QWEN_DANCE = (
    "<tool_call>\n<function=dance>\n<parameter=move>\nheadbanger_combo\n</parameter>\n"
    "<parameter=repeat>\n2\n</parameter>\n</function>\n</tool_call>"
)


def _dance_context():
    tool = FunctionTool(
        type="function",
        name="dance",
        description="Dance.",
        parameters={
            "type": "object",
            "properties": {"move": {"type": "string"}, "repeat": {"type": "integer"}},
        },
    )
    return StreamContext(
        function_tools=[tool], block_regex=build_block_regex(), enter_code=ENTER_CODE, end_code=END_CODE
    )


def test_native_function_is_rewritten_as_python_call():
    text = (
        "<function=dance>\n<parameter=move>\npendulum\n</parameter>\n<parameter=repeat>\n2\n</parameter>\n</function>"
    )
    assert native_function_calls_to_python(text) == "dance(move='pendulum', repeat=2)"


def test_native_function_without_parameters():
    assert native_function_calls_to_python("<function=stop_dance>\n</function>") == "stop_dance()"


def test_native_markers_alias_to_code_block():
    assert alias_native_tool_markers(QWEN_DANCE, ENTER_CODE, END_CODE).startswith(ENTER_CODE)
    assert alias_native_tool_markers(QWEN_DANCE, ENTER_CODE, END_CODE).endswith(END_CODE)


def test_extract_parses_native_call_inside_code_block():
    block = alias_native_tool_markers(QWEN_DANCE, ENTER_CODE, END_CODE)
    outside, calls = extract_function_calls_from_text(block, build_block_regex())
    assert outside == ""
    assert [(call.function_name, call.parameters) for call in calls] == [
        ("dance", {"move": "headbanger_combo", "repeat": 2})
    ]


def test_native_tool_call_is_never_spoken():
    handler = object.__new__(LanguageModelHandler)
    handler.stream_batch_sentences = 1
    chunks, tools, remaining = handler._process_printable_text(
        f"Ja, ich tanze! {QWEN_DANCE}", None, [], _dance_context()
    )
    spoken = " ".join(chunk.text for chunk in chunks)
    assert "tool_call" not in spoken and "function=" not in spoken
    assert "Ja, ich tanze!" in spoken
    assert [(tool.name, json.loads(tool.arguments)) for tool in tools] == [
        ("dance", {"move": "headbanger_combo", "repeat": 2})
    ]
    assert remaining == ""


def test_prompted_code_block_still_works():
    handler = object.__new__(LanguageModelHandler)
    handler.stream_batch_sentences = 1
    _chunks, tools, _ = handler._process_printable_text(
        f"{ENTER_CODE}dance(move='pendulum'){END_CODE}", None, [], _dance_context()
    )
    assert [tool.name for tool in tools] == ["dance"]
