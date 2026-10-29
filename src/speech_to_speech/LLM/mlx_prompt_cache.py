from __future__ import annotations

import copy
from typing import Any, Callable, Sequence

DEFAULT_PREFILL_STEP = 2048
DEFAULT_MAX_CHECKPOINTS = 4


class MLXPromptCheckpoint:
    """
    Keep the model state of conversation prefixes between turns.

    Hybrid models such as Qwen3.5 have recurrent layers whose cache cannot be
    trimmed, and chat templates render past assistant turns differently from
    the generation prompt. Instead of trimming, the state is checkpointed at
    the end of the stable prefix (everything before the generation prompt),
    which every later turn of the same conversation starts with. A few
    checkpoints are kept so prompt variants (audio vs text-only responses)
    do not evict each other.
    """

    def __init__(
        self,
        model: Any,
        make_cache: Callable[[Any], list[Any]],
        forward: Callable[[Any, Sequence[int], list[Any]], None],
        prefill_step: int = DEFAULT_PREFILL_STEP,
        max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
    ) -> None:
        self._model = model
        self._make_cache = make_cache
        self._forward = forward
        self._prefill_step = prefill_step
        self._max_checkpoints = max(1, max_checkpoints)
        self._entries: list[tuple[list[int], list[Any]]] = []
        self.reused_tokens = 0

    def reset(self) -> None:
        self._entries = []

    def _best_entry(self, tokens: list[int], stable_len: int) -> int | None:
        best, best_len = None, 0
        for index, (cached, _) in enumerate(self._entries):
            if best_len < len(cached) <= stable_len and tokens[: len(cached)] == cached:
                best, best_len = index, len(cached)
        return best

    def prepare(
        self, tokens: Sequence[int], stable_len: int, extra_checkpoints: Sequence[int] = ()
    ) -> tuple[list[int], list[Any]]:
        """
        Return the tokens still to feed and a cache already holding ``tokens[:stable_len]``.

        ``extra_checkpoints`` are earlier boundaries (e.g. before the latest user
        message) that are also saved, so a revised transcript of the same turn
        still reuses the conversation before it.
        """
        tokens = list(tokens)
        stable_len = max(0, min(stable_len, len(tokens) - 1))
        index = self._best_entry(tokens, stable_len)
        if index is not None:
            self._entries.append(self._entries.pop(index))
            cached, state = self._entries[-1]
            cache = copy.deepcopy(state)
            start = len(cached)
        else:
            cache = self._make_cache(self._model)
            start = 0
        self.reused_tokens = start
        boundaries = sorted({b for b in extra_checkpoints if start < b < stable_len} | {stable_len})
        position = start
        for boundary in boundaries:
            for offset in range(position, boundary, self._prefill_step):
                self._forward(self._model, tokens[offset : min(offset + self._prefill_step, boundary)], cache)
            position = boundary
            if boundary:
                self._save(tokens[:boundary], cache)
        return tokens[stable_len:], cache

    def _save(self, tokens: list[int], cache: list[Any]) -> None:
        self._entries = [entry for entry in self._entries if entry[0] != tokens]
        self._entries.append((tokens, copy.deepcopy(cache)))
        del self._entries[: -self._max_checkpoints]
