from speech_to_speech.LLM.utils import insert_before_last_user_message

NOTE = "Please reply to my message in English."


def test_note_is_placed_before_latest_user_message():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    result = insert_before_last_user_message(messages, NOTE)
    assert result[-2] == {"role": "system", "content": NOTE}
    assert result[-1] == {"role": "user", "content": "c"}
    assert len(messages) == 4


def test_note_precedes_tool_outputs_after_the_user_message():
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "x"},
    ]
    assert insert_before_last_user_message(messages, NOTE)[0] == {"role": "system", "content": NOTE}


def test_without_note_or_user_message_input_is_unchanged():
    messages = [{"role": "system", "content": "S"}]
    assert insert_before_last_user_message(messages, NOTE) is messages
    assert insert_before_last_user_message([{"role": "user", "content": "a"}], None) == [
        {"role": "user", "content": "a"}
    ]


def test_non_list_input_is_passed_through():
    assert insert_before_last_user_message("raw", NOTE) == "raw"


def test_leading_bracketed_asides_are_stripped():
    from speech_to_speech.LLM.utils import strip_stage_directions

    assert strip_stage_directions("(Music is playing, so keep it short)") == ""
    assert strip_stage_directions("(leise) [lacht] Bitte schön!") == "Bitte schön!"
    assert strip_stage_directions("Ein Hase (klein) hüpft.") == "Ein Hase (klein) hüpft."
