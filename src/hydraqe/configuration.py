from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

CONFIG_FILENAME = "qe_config.json"


@dataclass(frozen=True)
class HydraQEConfig:
    base_model: str
    base_model_revision: str
    architecture: str
    hidden_size: int
    num_hidden_layers: int
    num_fusion_layers: int
    num_fusion_heads: int
    num_pooler_heads: int
    dropout: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    heads: tuple[str, ...]
    primary_head: str
    secondary_head: str
    max_audio_seconds: float
    max_text_length: int
    sample_rate: int
    score_scale: float
    weights_filename: str
    weights_dtype: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> HydraQEConfig:
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - known)
        missing = sorted(known - set(raw))
        if unknown or missing:
            raise ValueError(
                f"Invalid {CONFIG_FILENAME}: unknown={unknown}, missing={missing}"
            )
        converted = dict(raw)
        for key in ("lora_target_modules", "heads"):
            converted[key] = tuple(converted[key])
        return cls(**converted)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        revision: str | None = None,
        token: str | bool | None = None,
    ) -> HydraQEConfig:
        local_path = Path(model_id_or_path)
        if local_path.is_dir():
            config_path = local_path / CONFIG_FILENAME
        else:
            config_path = Path(
                hf_hub_download(
                    repo_id=str(model_id_or_path),
                    filename=CONFIG_FILENAME,
                    revision=revision,
                    token=token,
                )
            )
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing model configuration: {config_path}")
        return cls.from_dict(json.loads(config_path.read_text()))
