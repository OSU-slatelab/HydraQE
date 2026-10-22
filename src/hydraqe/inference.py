from pathlib import Path

import librosa
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

from .configuration import HydraQEConfig
from .model import HydraQEModel


def load_audio(audio, config):
    source_rate = config.sample_rate
    if isinstance(audio, (str, Path)):
        waveform, source_rate = librosa.load(
            str(audio), sr=None, mono=True, dtype=np.float32
        )
    elif isinstance(audio, tuple):
        waveform, source_rate = audio
    else:
        waveform = audio

    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().float().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0 if waveform.shape[0] <= 8 else 1)
    if waveform.ndim != 1 or not waveform.size:
        raise ValueError("Audio must be a non-empty mono waveform")
    if source_rate != config.sample_rate:
        waveform = librosa.resample(
            waveform, orig_sr=source_rate, target_sr=config.sample_rate
        )
    max_samples = int(config.max_audio_seconds * config.sample_rate)
    return np.ascontiguousarray(waveform[:max_samples], dtype=np.float32)


def build_batch(
    processor,
    config,
    audio,
    translations,
):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]
    prefix = processor.apply_chat_template(
        [messages], add_generation_prompt=True, tokenize=False
    )[0]
    eos = processor.tokenizer.eos_token or ""
    prompts = [f"{prefix}{translation}{eos}" for translation in translations]
    waveforms = [load_audio(item, config) for item in audio]
    encoded = processor(
        text=prompts,
        audio=waveforms,
        return_tensors="pt",
        text_kwargs={
            "padding": True,
            "truncation": True,
            "max_length": config.max_text_length,
        },
        audio_kwargs={"padding": True, "return_attention_mask": True},
    )
    speech = {
        "input_features": encoded["input_features"],
        "feature_attention_mask": encoded["feature_attention_mask"],
    }
    text = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
    return speech, text


def _resolve_dtype(value, device):
    if isinstance(value, torch.dtype):
        return value
    names = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if value not in names:
        raise ValueError(
            f"Unsupported dtype {value!r}; choose auto, float32, bfloat16, or float16"
        )
    return names[value]


class HydraQE:
    """High-level inference API returning all available quality scores."""

    def __init__(self, model, processor, config, device):
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path,
        *,
        revision=None,
        token=None,
        device=None,
        dtype="auto",
        merge_lora=True,
    ):
        config = HydraQEConfig.from_pretrained(
            model_id_or_path, revision=revision, token=token
        )
        resolved_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        resolved_dtype = _resolve_dtype(dtype, resolved_device)
        local_path = Path(model_id_or_path)
        if local_path.is_dir():
            weights_path = local_path / config.weights_filename
        else:
            weights_path = Path(
                hf_hub_download(
                    repo_id=str(model_id_or_path),
                    filename=config.weights_filename,
                    revision=revision,
                    token=token,
                )
            )

        model = HydraQEModel(config, dtype=resolved_dtype)
        model.load_release_weights(weights_path)
        if merge_lora:
            model.merge_lora()
        model.to(device=resolved_device, dtype=resolved_dtype).eval()

        processor = Qwen3ASRProcessor.from_pretrained(
            config.base_model,
            revision=config.base_model_revision,
            fix_mistral_regex=True,
        )
        return cls(model, processor, config, resolved_device)

    @torch.inference_mode()
    def score_batch(
        self,
        audio,
        translations,
        *,
        heads=None,
    ):
        selected = tuple(heads or self.config.heads)
        unknown = sorted(set(selected) - set(self.config.heads))
        if unknown:
            raise ValueError(f"Unknown prediction heads: {unknown}")
        speech, text = build_batch(
            self.processor, self.config, list(audio), list(translations)
        )
        model_dtype = next(self.model.parameters()).dtype
        speech = {
            key: value.to(
                self.device,
                dtype=model_dtype if key == "input_features" else None,
            )
            for key, value in speech.items()
        }
        text = {key: value.to(self.device) for key, value in text.items()}
        outputs = self.model(speech, text)
        return {
            head: (outputs[head] * self.config.score_scale).float().cpu().tolist()
            for head in selected
        }

    def score(
        self,
        audio,
        translation,
        *,
        heads=None,
    ):
        batch = self.score_batch([audio], [translation], heads=heads)
        return {head: values[0] for head, values in batch.items()}
