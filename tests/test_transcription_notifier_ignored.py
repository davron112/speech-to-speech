from queue import Queue
from threading import Event

from speech_to_speech.pipeline.events import TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier


def _notifier(ignored=""):
    events, listen = Queue(), Event()
    notifier = TranscriptionNotifier(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"text_output_queue": events, "should_listen": listen, "ignored_transcripts": ignored},
    )
    return notifier, events, listen


def _completed(events):
    items = []
    while not events.empty():
        item = events.get()
        if isinstance(item, TranscriptionCompletedEvent):
            items.append(item)
    return items


def test_noise_filler_is_treated_as_no_speech():
    notifier, events, listen = _notifier("yeah, mm-hmm, thank you")
    list(notifier.process(Transcription(text="Mm-hmm.", turn_id="t1", turn_revision=0)))
    assert [e.transcript for e in _completed(events)] == [""]
    assert listen.is_set()


def test_matching_ignores_case_and_punctuation():
    notifier, events, _ = _notifier("thank you")
    list(notifier.process(Transcription(text="Thank you!", turn_id="t1", turn_revision=0)))
    assert [e.transcript for e in _completed(events)] == [""]


def test_real_speech_containing_a_filler_passes():
    notifier, events, listen = _notifier("yeah")
    list(notifier.process(Transcription(text="Yeah, tanz bitte!", turn_id="t1", turn_revision=0)))
    assert [e.transcript for e in _completed(events)] == ["Yeah, tanz bitte!"]
    assert not listen.is_set()


def test_nothing_is_ignored_by_default():
    notifier, events, _ = _notifier()
    list(notifier.process(Transcription(text="Yeah.", turn_id="t1", turn_revision=0)))
    assert [e.transcript for e in _completed(events)] == ["Yeah."]
