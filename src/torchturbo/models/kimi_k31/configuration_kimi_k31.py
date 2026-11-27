from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class KimiK31TextConfig(PretrainedConfig):
    """Text backbone dimensions and layer schedules from Kimi's text_config."""

    model_type = "kimi_k31_text"

    def __init__(self, dsa_indexer_loss_coeff: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if dsa_indexer_loss_coeff < 0:
            raise ValueError("dsa_indexer_loss_coeff must be non-negative")
        self.dsa_indexer_loss_coeff = dsa_indexer_loss_coeff

    def is_dsa_layer(self, layer_idx: int) -> bool:
        """The published schedule numbers decoder layers from one."""
        return layer_idx + 1 in self.dsa_config["dsa_layers"]

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_k_dense_replace and layer_idx % self.moe_layer_freq == 0


class KimiK31Config(PretrainedConfig):
    """Top-level config for the released Kimi K3.1 checkpoint."""

    model_type = "kimi_k31"
    sub_configs = {"text_config": KimiK31TextConfig, "vision_config": PretrainedConfig}

    def __init__(
        self,
        text_config: dict[str, Any] | KimiK31TextConfig | None = None,
        vision_config: dict[str, Any] | PretrainedConfig | None = None,
        media_placeholder_token_id: int = 163605,
        **kwargs: Any,
    ) -> None:
        if isinstance(text_config, dict) and text_config.get("model_type") == "kimi_linear":
            text_config = dict(text_config)
            linear = dict(text_config["linear_attn_config"])
            linear.setdefault("checkpoint_num_heads", ((linear["num_heads"] + 127) // 128) * 128)
            text_config["linear_attn_config"] = linear
        self.text_config = (
            text_config if isinstance(text_config, KimiK31TextConfig)
            else KimiK31TextConfig(**(text_config or {}))
        )
        self.vision_config = (
            vision_config if isinstance(vision_config, PretrainedConfig)
            else PretrainedConfig(**(vision_config or {}))
        )
        self.media_placeholder_token_id = media_placeholder_token_id
        super().__init__(**kwargs)
