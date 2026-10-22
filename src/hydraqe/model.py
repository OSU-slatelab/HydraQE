import torch
import torch.nn.functional as F
from entmax import sparsemax
from peft import LoraConfig, get_peft_model
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
from safetensors.torch import load_file
from torch import nn


class LayerwiseAttention(nn.Module):
    def __init__(self, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.scalar_parameters = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_layers)]
        )
        self.gamma = nn.Parameter(torch.ones(1))
        self.register_buffer("dropout_mask", torch.zeros(num_layers))
        self.register_buffer("dropout_fill", torch.full((num_layers,), -1e20))

    def forward(self, tensors):
        if len(tensors) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} hidden states, received {len(tensors)}"
            )
        weights = torch.cat(list(self.scalar_parameters))
        if self.training and self.dropout:
            weights = torch.where(
                self.dropout_mask.uniform_() > self.dropout,
                weights,
                self.dropout_fill,
            )
        weights = sparsemax(weights, dim=0)
        mixed = sum(weight * tensor for weight, tensor in zip(weights, tensors))
        return self.gamma * mixed


class EncoderConfig:
    def __init__(
        self,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        dropout_prob,
        layer_norm_eps=1e-7,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_prob = dropout_prob
        self.layer_norm_eps = layer_norm_eps


class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.drop_p = config.dropout_prob
        self.n_heads = config.num_attention_heads
        self.d_head = config.hidden_size // config.num_attention_heads
        dim = config.hidden_size
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.att_proj_linear = nn.Linear(dim, dim, bias=False)
        self.resid_dropout = nn.Dropout(self.drop_p)
        self.attn_norm_x = nn.RMSNorm(dim, eps=config.layer_norm_eps)
        self.attn_norm_kv = nn.RMSNorm(dim, eps=config.layer_norm_eps)

    def forward(self, x, attn_mask):
        batch_size = x.shape[0]
        x = self.attn_norm_x(x)
        xq, xk, xv = self.q(x), self.k(x), self.v(x)
        xq = xq.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        xk = xk.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        xv = xv.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attn_mask,
            dropout_p=self.drop_p if self.training else 0.0,
            is_causal=False,
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_head)
        )
        return self.resid_dropout(self.att_proj_linear(out))


class FeedForwardBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.ff_dropout = nn.Dropout(config.dropout_prob)
        self.ff_linear_1 = nn.Linear(dim, config.intermediate_size, bias=False)
        self.ff_linear_2 = nn.Linear(config.intermediate_size, dim, bias=False)
        self.ff_linear_3 = nn.Linear(dim, config.intermediate_size, bias=False)
        self.ff_norm = nn.RMSNorm(dim, eps=config.layer_norm_eps)

    def forward(self, x):
        x = self.ff_norm(x)
        return self.ff_dropout(
            self.ff_linear_2(F.silu(self.ff_linear_1(x)) * self.ff_linear_3(x))
        )


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiheadAttention(config)
        self.ff = FeedForwardBlock(config)
        self.self_attn_scale = nn.Parameter(torch.ones(1))
        self.ff_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, attn_mask):
        x = x + self.self_attn_scale * self.self_attention(x, attn_mask)
        return x + self.ff_scale * self.ff(x)


def _extended_attention_mask(mask, dtype):
    extended = mask.unsqueeze(1).unsqueeze(2)
    extended = extended * extended.squeeze(-2).unsqueeze(-1)
    min_value = -10000.0 if dtype == torch.float16 else torch.finfo(dtype).min
    extended = (1.0 - extended.float()) * min_value
    fully_masked = torch.all(extended == min_value, dim=-1, keepdim=True)
    if fully_masked.any():
        extended = extended.masked_fill(fully_masked, min_value)
        extended[..., 0] = extended[..., 0].masked_fill(fully_masked.squeeze(-1), 0.0)
    return extended.to(dtype)


class FusionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        mask = _extended_attention_mask(attention_mask, hidden_states.dtype)
        x = hidden_states
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)


class AttentionPooler(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        query = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(query, x, x, key_padding_mask=~mask.bool())
        return self.ln(out).squeeze(1)


class RegressionHead(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 3072),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(3072, 1024),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.ff(x).view(-1)


class AllScorerHeads(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.primary_head = RegressionHead(hidden_size, dropout)
        self.secondary_head = RegressionHead(hidden_size, dropout)
        self.da_head = RegressionHead(hidden_size, dropout)

    def forward(self, x):
        return {
            "xcomet_ref": self.primary_head(x),
            "metricx_ref": self.secondary_head(x),
            "da": self.da_head(x),
        }


def audio_output_lengths(input_lengths):
    input_lengths = input_lengths.long()
    remainder = input_lengths % 100
    feature_lengths = (remainder - 1) // 2 + 1
    return (
        ((feature_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    ).long()


class HydraQEModel(nn.Module):
    def __init__(self, config, *, dtype):
        super().__init__()
        self.config = config
        full_model = Qwen3ASRForConditionalGeneration.from_pretrained(
            config.base_model,
            revision=config.base_model_revision,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        if not hasattr(full_model, "thinker"):
            raise RuntimeError("The configured base model has no Qwen3-ASR thinker")
        self.thinker = full_model.thinker
        del full_model

        self.audio_tower = self.thinker.audio_tower
        for parameter in self.audio_tower.parameters():
            parameter.requires_grad_(False)
        self.audio_tower.eval()

        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=list(config.lora_target_modules),
            lora_dropout=config.lora_dropout,
            bias="none",
        )
        self.text_model = get_peft_model(self.thinker.model, peft_config)
        self.thinker.model = self.text_model

        self.layerwise_attention = LayerwiseAttention(
            num_layers=config.num_hidden_layers,
            dropout=config.dropout,
        )
        fusion_config = EncoderConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_fusion_layers,
            num_attention_heads=config.num_fusion_heads,
            intermediate_size=config.hidden_size * 4,
            dropout_prob=config.dropout,
        )
        self.fusion_encoder = FusionEncoder(fusion_config)
        self.attention_pooler = AttentionPooler(
            d_model=config.hidden_size,
            num_heads=config.num_pooler_heads,
        )
        self.estimator = AllScorerHeads(config.hidden_size, config.dropout)

    def train(self, mode=True):
        super().train(mode)
        self.audio_tower.eval()
        return self

    def load_release_weights(self, weights_path):
        state = load_file(str(weights_path), device="cpu")
        model_keys = set(self.state_dict())
        unexpected = sorted(set(state) - model_keys)
        required_prefixes = (
            "thinker.model.base_model.model.",
            "layerwise_attention.",
            "fusion_encoder.",
            "attention_pooler.",
            "estimator.primary_head.",
            "estimator.secondary_head.",
            "estimator.da_head.",
        )
        missing_groups = [
            prefix
            for prefix in required_prefixes
            if not any(k.startswith(prefix) for k in state)
        ]
        non_adapter_backbone = [
            key
            for key in state
            if key.startswith("thinker.model.") and "lora_" not in key
        ]
        if unexpected or missing_groups or non_adapter_backbone:
            raise RuntimeError(
                "Invalid release weights: "
                f"unexpected={unexpected[:10]}, missing_groups={missing_groups}, "
                f"non_adapter_backbone={non_adapter_backbone[:10]}"
            )
        self.load_state_dict(state, strict=False)

    def merge_lora(self):
        if not hasattr(self.text_model, "merge_and_unload"):
            return
        merged = self.text_model.merge_and_unload(safe_merge=True)
        self.text_model = merged
        self.thinker.model = merged

    def forward(self, speech, text):
        feature_mask = speech["feature_attention_mask"]
        lengths = audio_output_lengths(feature_mask.sum(dim=1))
        input_ids = text["input_ids"]
        inputs_embeds = self.thinker.get_input_embeddings()(input_ids)

        audio_features = self.thinker.get_audio_features(
            speech["input_features"],
            feature_attention_mask=feature_mask,
        ).to(inputs_embeds.device, inputs_embeds.dtype)
        audio_token_id = self.thinker.config.audio_token_id
        offset = 0
        for index in range(input_ids.shape[0]):
            feature_count = lengths[index].item()
            positions = (input_ids[index] == audio_token_id).nonzero(as_tuple=True)[0]
            take = min(feature_count, positions.numel())
            if take:
                inputs_embeds[index, positions[:take]] = audio_features[
                    offset : offset + take
                ]
            offset += feature_count

        output = self.thinker(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=text["attention_mask"],
            input_features=None,
            feature_attention_mask=None,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        mixed = self.layerwise_attention(list(output.hidden_states))
        mixed = self.fusion_encoder(mixed, text["attention_mask"].long())
        pooled = self.attention_pooler(mixed, text["attention_mask"])
        return self.estimator(pooled)
