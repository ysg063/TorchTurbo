from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from torchturbo.engine.config import EngineConfig
from torchturbo.models.base import TurboModel, TurboVLMModel
from torchturbo.models.modules.attention.ulysses import SequenceParallelInputSlicingSpec
from torchturbo.models.registry import register_model
from torchturbo.models.weight_converter import BaseWeightConverter
from torchturbo.parallel import FSDPParallelPlan

from .configuration_kimi_k31 import KimiK31Config, KimiK31TextConfig
from .modeling_kimi_k31 import KimiK31Router, KimiK31TextModel
from .vision import KimiK31VisionProjector, KimiK31VisionTower


class KimiK31WeightConverter(BaseWeightConverter):
    """Map BF16 routed expert tensors using the latent width and HF names."""

    def __init__(self, model: nn.Module, parallel_context: dict[str, Any], model_path: str) -> None:
        super().__init__(model, parallel_context, model_path)
        # The base converter uses hidden_size solely for routed expert matrices.
        self.config = copy(self.config.get_text_config())
        self.config.hidden_size = self.config.routed_expert_hidden_size
        tensor_names = self.checkpoint_info.get_tensor_infos()
        self._hf_moe_name = (
            "block_sparse_moe" if any(".block_sparse_moe.experts." in name for name in tensor_names) else "mlp"
        )

    def get_hf_expert_mlp_name(self) -> str:
        return self._hf_moe_name

    def convert_router_param_to_hf(self, name: str) -> str:
        name = super().convert_router_param_to_hf(name)
        return name.replace(".mlp.gate.", f".{self._hf_moe_name}.gate.")

    def convert_persist_buffer(self, name: str) -> str:
        name = super().convert_persist_buffer(name)
        return name.replace(".mlp.gate.", f".{self._hf_moe_name}.gate.")

    def to_hf_name(self, name: str) -> str:
        name = super().to_hf_name(name)
        for child in ("shared_experts", "routed_expert_down_proj", "routed_expert_norm", "routed_expert_up_proj"):
            name = name.replace(f".mlp.{child}.", f".{self._hf_moe_name}.{child}.")
        return name

    def get_expert_gate_name(self) -> str:
        return "w1"

    def get_expert_down_name(self) -> str:
        return "w2"

    def get_expert_up_name(self) -> str:
        return "w3"


@dataclass
class KimiK31CausalLMOutput(CausalLMOutputWithPast):
    indexer_loss_sum: torch.Tensor | None = None
    indexer_num_queries: int | None = None
    indexer_num_layers: int | None = None
    indexer_loss_coeff: float | None = None


@register_model("kimi_k31_text", config_cls=KimiK31TextConfig, auto_class=AutoModelForCausalLM)
class TurboKimiK31ForCausalLM(TurboModel, PreTrainedModel):
    """Packed text training entry point for the Kimi K3.1 decoder."""

    config_class = KimiK31TextConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: KimiK31TextConfig, *, train_indexer: bool | None = None) -> None:
        super().__init__(config)
        # 默认根据 KL 系数决定是否训练 indexer。冻结时仍会用它为主 Attention 选 top-k。
        if train_indexer is None:
            train_indexer = config.dsa_indexer_loss_coeff > 0
        self.model = KimiK31TextModel(config, train_indexer=train_indexer)
        self.uses_indexer_loss = train_indexer and config.dsa_indexer_loss_coeff > 0
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    @contextmanager
    def init_context(cls, config: KimiK31TextConfig, engine_config: EngineConfig) -> Iterator[None]:
        checkpointing = engine_config.activation_checkpointing
        if checkpointing.enabled and checkpointing.use_reentrant:
            raise ValueError("Kimi K3.1 AttnRes requires non-reentrant activation checkpointing")
        config._attn_implementation = "eager"
        state = cls._force_config_dtype_float32(config)
        try:
            yield
        finally:
            cls._restore_config_dtype(config, state)

    @classmethod
    def get_default_parallel_plan(cls, **kwargs: Any) -> FSDPParallelPlan:
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan()

    def on_parallelized(self, parallel_context: dict[str, Any]) -> None:
        self.model.sp_group = parallel_context["sp"]["sp_group"]

    def on_optimizer_created(self, optimizer: torch.optim.Optimizer) -> None:
        routers = [
            module for module in self.modules()
            if isinstance(module, KimiK31Router)
            and module.weight.requires_grad
            and not module.config.moe_aux_loss_coeff
        ]
        if not routers:
            return
        for router in routers:
            if not 0 < router.topk < router.num_experts or router.score_function != "sigmoid":
                raise ValueError("Kimi quantile balancing requires sigmoid routing and top-k below the expert count")
            router.quantile_balancing = True
        # SP shards the tokens, while DP shards the batch. Every rank supplies
        # distinct source tokens, including when the experts are split by EP.
        group = dist.group.WORLD if dist.is_initialized() else None

        def finish_quantile_step(*_args: Any) -> None:
            for router in routers:
                router.finish_quantile_step(group)

        optimizer.register_step_post_hook(finish_quantile_step)

    def get_weight_converter(
        self, parallel_context: dict[str, Any], model_path: str,
    ) -> KimiK31WeightConverter:
        return KimiK31WeightConverter(self, parallel_context, model_path)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> KimiK31CausalLMOutput | tuple:
        if use_cache:
            raise ValueError("Kimi K3.1 training does not use a KV cache")
        if kwargs.get("attention_mask") is not None:
            raise ValueError("Kimi K3.1 training uses cu_seq_lens_q for packed inputs")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        hidden_states, indexer_loss, num_queries = self.model(inputs_embeds=inputs_embeds, cu_seqlens=cu_seq_lens_q)
        selection = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, selection])

        loss = None
        if labels is not None:
            if logits.shape[:2] != labels.shape or self.model.sp_group is not None and self.model.sp_group.size() > 1:
                raise ValueError("Pass complete labels without sequence sharding to model.forward")
            targets = F.pad(labels[:, 1:], (0, 1), value=-100)
            if cu_seq_lens_q is not None:
                positions = torch.arange(labels.size(1), device=labels.device)
                ends = cu_seq_lens_q[1:]
                segment = torch.bucketize(positions, ends, right=True).clamp(max=ends.numel() - 1)
                targets = targets.masked_fill(positions + 1 >= ends[segment], -100)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction="sum")
            loss = loss / targets.ne(-100).sum().clamp_min(1)
        with_indexer_loss = self.uses_indexer_loss and indexer_loss is not None
        num_layers = self.model.num_dsa_layers
        indexer_coeff = self.config.dsa_indexer_loss_coeff
        # 直接传 labels 给模型时，按当前 batch 的有效 query 数和 DSA 层数平均 KL。
        # 分布式训练由 engine 使用下面的原始 KL 总和与计数，按全局训练步归一化。
        if with_indexer_loss and loss is not None:
            loss = loss + indexer_loss * (indexer_coeff / max(1, num_queries * num_layers))
        output = KimiK31CausalLMOutput(
            loss=loss,
            logits=logits,
            indexer_loss_sum=indexer_loss,
            indexer_num_queries=num_queries if with_indexer_loss else None,
            indexer_num_layers=num_layers if with_indexer_loss else None,
            indexer_loss_coeff=indexer_coeff if with_indexer_loss else None,
        )
        if return_dict is False:
            return output.to_tuple()
        return output


@register_model("kimi_k31", config_cls=KimiK31Config, auto_class=AutoModelForCausalLM)
class TurboKimiK31ForConditionalGeneration(TurboVLMModel, PreTrainedModel):
    """Packed text and image training for the released Kimi K3.1 model."""

    config_class = KimiK31Config
    base_model_prefix = "language_model"
    multimodal_input_keys = ("pixel_values", "grid_thws")

    def __init__(self, config: KimiK31Config) -> None:
        if not isinstance(config, KimiK31Config):
            config = KimiK31Config(**config.to_dict())
        super().__init__(config)
        self.language_model = TurboKimiK31ForCausalLM(config.text_config)
        self.vision_tower = KimiK31VisionTower(config.vision_config)
        self.mm_projector = KimiK31VisionProjector(config.vision_config, config.text_config.hidden_size)
        self.uses_indexer_loss = self.language_model.uses_indexer_loss
        self.post_init()

    @classmethod
    @contextmanager
    def init_context(cls, config: KimiK31Config, engine_config: EngineConfig) -> Iterator[None]:
        state = cls._force_config_dtype_float32(config)
        config._attn_implementation = "eager"
        try:
            with TurboKimiK31ForCausalLM.init_context(config.text_config, engine_config):
                yield
        finally:
            cls._restore_config_dtype(config, state)

    @classmethod
    def get_default_parallel_plan(cls, **kwargs: Any) -> FSDPParallelPlan:
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(multimodal=True)

    def on_parallelized(self, parallel_context: dict[str, Any]) -> None:
        self.language_model.on_parallelized(parallel_context)

    def on_optimizer_created(self, optimizer: torch.optim.Optimizer) -> None:
        self.language_model.on_optimizer_created(optimizer)

    def get_sequence_parallel_input_slicing_spec(self) -> SequenceParallelInputSlicingSpec:
        return SequenceParallelInputSlicingSpec(
            self.language_model.model, KimiK31TextModel.forward, sliced_inputs=(("inputs_embeds", 1),),
        )

    def get_norm(self) -> nn.Module:
        return self.language_model.model.norm

    def get_lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.language_model.set_input_embeddings(value)

    def get_weight_converter(
        self, parallel_context: dict[str, Any], model_path: str,
    ) -> KimiK31WeightConverter:
        return KimiK31WeightConverter(self, parallel_context, model_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thws: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> KimiK31CausalLMOutput | tuple:
        embeddings = self.get_input_embeddings()(input_ids)
        if pixel_values is None:
            # Keep vision FSDP collectives in step when only some ranks receive images.
            kernel_h, kernel_w = self.vision_tower.merge_kernel
            patch = self.config.vision_config.patch_size
            pixels = embeddings.new_zeros((kernel_h * kernel_w, 3, patch, patch))
            grid = torch.tensor([[1, kernel_h, kernel_w]], device=input_ids.device)
            features = self.mm_projector(self.vision_tower(pixels, grid))
            embeddings = embeddings + features.sum().to(embeddings.dtype) * 0
        else:
            if grid_thws is None:
                raise ValueError("Kimi K3.1 image inputs require grid_thws")
            features = self.mm_projector(self.vision_tower(pixel_values, grid_thws))
            mask = input_ids == self.config.media_placeholder_token_id
            if mask.sum().item() != features.size(0):
                raise ValueError("Kimi image tokens and projected patches must match")
            embeddings = embeddings.masked_scatter(mask.unsqueeze(-1), features.to(embeddings.dtype))
        return self.language_model(inputs_embeds=embeddings, **kwargs)
