from dataclasses import dataclass, field

DEFAULT_MAX_SECONDS_PER_CHAR = 0.25
DEFAULT_MAX_EXTRA_SECONDS = 2.0


@dataclass
class PocketTTSHandlerArguments:
    pocket_tts_device: str = field(
        default="cpu",
        metadata={
            "help": "The device type on which the Pocket TTS model will run. Options: 'auto' (first available of CUDA, NPU, XPU, MPS, CPU), 'cuda', 'npu', 'xpu', 'mps', or 'cpu'. Default is 'cpu'."
        },
    )
    pocket_tts_voice: str = field(
        default="jean",
        metadata={
            "help": "Voice to use for Pocket TTS. Can be a preset name ('alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma'), a local audio file path, or a HuggingFace path like 'hf://kyutai/tts-voices/...'. Default is 'jean'."
        },
    )
    pocket_tts_sample_rate: int = field(
        default=16000,
        metadata={
            "help": "Output sample rate in Hz for Pocket TTS. Pocket TTS uses 24kHz internally but will be resampled to this rate to match the audio output. Default is 16000 to match the pipeline's audio streamer."
        },
    )
    pocket_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Size of audio blocks to yield for streaming in Pocket TTS. Default is 512."},
    )
    pocket_tts_max_tokens: int = field(
        default=50,
        metadata={"help": "Maximum number of tokens to generate per sentence in Pocket TTS. Default is 50."},
    )
    pocket_tts_language: str = field(
        default="english",
        metadata={"help": "PocketTTS language/model configuration to load."},
    )
    pocket_tts_extra_languages: str = field(
        default="",
        metadata={
            "help": "Comma-separated Pocket TTS languages (e.g. 'english') loaded next to pocket_tts_language. Each response uses the model matching its language code, falling back to pocket_tts_language."
        },
    )
    pocket_tts_model_name: str | None = field(
        default=None,
        metadata={
            "help": "Optional Farsi model selection. Use 'mehdi-hf/pocket-tts-farsi-v2' for Persian with automatic phoneme conversion. Overrides pocket_tts_language."
        },
    )
    pocket_tts_temperature: float | None = field(
        default=None,
        metadata={"help": "Sampling temperature. Uses the model config default when unset, 0.3 for Farsi v2."},
    )
    pocket_tts_max_seconds_per_char: float = field(
        default=DEFAULT_MAX_SECONDS_PER_CHAR,
        metadata={
            "help": "Audio length cap per input character. A sentence whose end of speech the model misses stops at "
            "len(text) * this + pocket_tts_max_extra_seconds instead of streaming for a minute. Default is 0.25."
        },
    )
    pocket_tts_max_extra_seconds: float = field(
        default=DEFAULT_MAX_EXTRA_SECONDS,
        metadata={"help": "Fixed seconds added to the per-character audio length cap. Default is 2.0."},
    )
    pocket_tts_eos_threshold: float | None = field(
        default=None,
        metadata={
            "help": "End-of-speech threshold. Defaults to -2 for Farsi v2 and the Pocket TTS default for other models."
        },
    )
