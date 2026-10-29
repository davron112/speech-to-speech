from speech_to_speech.LLM.mlx_prompt_cache import MLXPromptCheckpoint


def _checkpoint(prefill_step=4):
    fed = []

    def make_cache(model):
        return [{"tokens": []}]

    def forward(model, tokens, cache):
        fed.append(list(tokens))
        cache[0]["tokens"].extend(tokens)

    return MLXPromptCheckpoint(object(), make_cache, forward, prefill_step=prefill_step), fed


def test_first_turn_prefills_stable_prefix_in_chunks():
    checkpoint, fed = _checkpoint(prefill_step=4)
    remaining, cache = checkpoint.prepare(list(range(10)), stable_len=8)
    assert fed == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert remaining == [8, 9]
    assert cache[0]["tokens"] == list(range(8))
    assert checkpoint.reused_tokens == 0


def test_next_turn_only_prefills_new_tokens():
    checkpoint, fed = _checkpoint(prefill_step=100)
    checkpoint.prepare([1, 2, 3, 90, 91], stable_len=3)
    fed.clear()
    remaining, cache = checkpoint.prepare([1, 2, 3, 4, 5, 6, 90, 91], stable_len=6)
    assert fed == [[4, 5, 6]]
    assert remaining == [90, 91]
    assert cache[0]["tokens"] == [1, 2, 3, 4, 5, 6]
    assert checkpoint.reused_tokens == 3


def test_generation_does_not_leak_into_checkpoint():
    checkpoint, _ = _checkpoint()
    _, cache = checkpoint.prepare([1, 2, 3, 90], stable_len=3)
    cache[0]["tokens"].extend([77, 78])
    _, next_cache = checkpoint.prepare([1, 2, 3, 4, 90], stable_len=4)
    assert next_cache[0]["tokens"] == [1, 2, 3, 4]


def test_changed_prefix_starts_from_scratch():
    checkpoint, fed = _checkpoint(prefill_step=100)
    checkpoint.prepare([1, 2, 3, 90], stable_len=3)
    fed.clear()
    _, cache = checkpoint.prepare([9, 2, 3, 4, 90], stable_len=4)
    assert fed == [[9, 2, 3, 4]]
    assert cache[0]["tokens"] == [9, 2, 3, 4]
    assert checkpoint.reused_tokens == 0


def test_at_least_one_token_is_left_for_generation():
    checkpoint, _ = _checkpoint()
    remaining, cache = checkpoint.prepare([1, 2, 3], stable_len=3)
    assert remaining == [3]
    assert cache[0]["tokens"] == [1, 2]


def test_reset_forgets_checkpoint():
    checkpoint, fed = _checkpoint(prefill_step=100)
    checkpoint.prepare([1, 2, 3, 90], stable_len=3)
    checkpoint.reset()
    fed.clear()
    checkpoint.prepare([1, 2, 3, 4, 90], stable_len=4)
    assert fed == [[1, 2, 3, 4]]


def test_prompt_variants_do_not_evict_each_other():
    checkpoint, fed = _checkpoint(prefill_step=100)
    checkpoint.prepare([1, 2, 3, 90], stable_len=3)
    checkpoint.prepare([7, 8, 90], stable_len=2)
    fed.clear()
    _, cache = checkpoint.prepare([1, 2, 3, 4, 90], stable_len=4)
    assert fed == [[4]]
    assert checkpoint.reused_tokens == 3
    fed.clear()
    checkpoint.prepare([7, 8, 9, 90], stable_len=3)
    assert fed == [[9]]


def test_oldest_checkpoint_is_evicted_beyond_limit():
    fed = []

    def forward(model, tokens, cache):
        fed.append(list(tokens))

    checkpoint = MLXPromptCheckpoint(object(), lambda model: [], forward, prefill_step=100, max_checkpoints=2)
    for first in (1, 2, 3):
        checkpoint.prepare([first, 0, 90], stable_len=2)
    fed.clear()
    checkpoint.prepare([1, 0, 5, 90], stable_len=3)
    assert fed == [[1, 0, 5]]
    fed.clear()
    checkpoint.prepare([3, 0, 5, 90], stable_len=3)
    assert fed == [[5]]


def test_revised_user_message_reuses_history_before_it():
    checkpoint, fed = _checkpoint(prefill_step=100)
    checkpoint.prepare([1, 2, 3, 4, 90], stable_len=4, extra_checkpoints=[2])
    fed.clear()
    _, cache = checkpoint.prepare([1, 2, 5, 6, 90], stable_len=4, extra_checkpoints=[2])
    assert checkpoint.reused_tokens == 2
    assert fed == [[5, 6]]
    assert cache[0]["tokens"] == [1, 2, 5, 6]


def test_extra_checkpoint_outside_range_is_ignored():
    checkpoint, fed = _checkpoint(prefill_step=100)
    checkpoint.prepare([1, 2, 3, 90], stable_len=3, extra_checkpoints=[0, 3, 9])
    assert fed == [[1, 2, 3]]


def test_repeated_revisions_keep_reusing_pre_user_checkpoint():
    checkpoint, fed = _checkpoint(prefill_step=100)
    checkpoint.prepare([1, 2, 3, 4, 90], stable_len=4, extra_checkpoints=[2])
    for revision in (5, 6, 7):
        fed.clear()
        checkpoint.prepare([1, 2, revision, 8, 90], stable_len=4, extra_checkpoints=[2])
        assert checkpoint.reused_tokens == 2
        assert fed == [[revision, 8]]


def _handler_with_template(template):
    from types import SimpleNamespace

    from speech_to_speech.LLM.language_model import LanguageModelHandler

    handler = object.__new__(LanguageModelHandler)
    handler.tokenizer = SimpleNamespace(
        apply_chat_template=template, encode=lambda text, add_special_tokens=False: list(range(len(text)))
    )
    return handler


def _render(messages, **kwargs):
    return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


def _char_tokens(handler, messages):
    return handler.tokenizer.encode(_render(messages))


def _char_handler():
    from types import SimpleNamespace

    from speech_to_speech.LLM.language_model import LanguageModelHandler

    handler = object.__new__(LanguageModelHandler)
    handler.tokenizer = SimpleNamespace(
        apply_chat_template=_render, encode=lambda text, add_special_tokens=False: [ord(c) for c in text]
    )
    return handler


def test_first_turn_checkpoints_before_user_text():
    handler = _char_handler()
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    assert handler._before_last_user_turn(messages, _char_tokens(handler, messages)) == [
        len("<system>S</system><user>")
    ]


def test_later_turn_checkpoints_before_latest_user_text():
    handler = _char_handler()
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ]
    prefix = "<system>S</system><user>hi</user><assistant>hello</assistant><user>"
    assert handler._before_last_user_turn(messages, _char_tokens(handler, messages)) == [len(prefix)]


def test_boundary_is_shared_by_revised_transcripts():
    handler = _char_handler()
    first = [{"role": "system", "content": "S"}, {"role": "user", "content": "Tanz"}]
    revised = [{"role": "system", "content": "S"}, {"role": "user", "content": "Tanz bitte"}]
    assert handler._before_last_user_turn(first, _char_tokens(handler, first)) == handler._before_last_user_turn(
        revised, _char_tokens(handler, revised)
    )


def test_no_boundary_when_last_message_is_not_user():
    handler = _char_handler()
    messages = [{"role": "system", "content": "S"}, {"role": "assistant", "content": "a"}]
    assert handler._before_last_user_turn(messages, _char_tokens(handler, messages)) == []


def test_first_turn_revision_reuses_system_prompt_checkpoint():
    handler = _char_handler()
    checkpoint, fed = _checkpoint(prefill_step=1000)
    first = [{"role": "system", "content": "LONG SYSTEM PROMPT"}, {"role": "user", "content": "Tanz"}]
    revised = [{"role": "system", "content": "LONG SYSTEM PROMPT"}, {"role": "user", "content": "Tanz bitte"}]
    for messages in (first, revised):
        tokens = _char_tokens(handler, messages) + [0]
        checkpoint.prepare(tokens, len(tokens) - 1, handler._before_last_user_turn(messages, tokens))
    assert checkpoint.reused_tokens == len("<system>LONG SYSTEM PROMPT</system><user>")
