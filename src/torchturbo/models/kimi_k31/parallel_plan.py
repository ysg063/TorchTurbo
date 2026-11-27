from __future__ import annotations

from typing import Any

from torch import nn
from torch.distributed.tensor import Shard

from torchturbo.engine.config import EngineConfig
from torchturbo.parallel import FSDPPlacementSpec, FSDPPlanSpec, FSDPParallelPlan

from .configuration_kimi_k31 import KimiK31Config, KimiK31TextConfig
from .modeling_kimi_k31 import KimiK31MoE


def _place_moe(
    module: nn.Module,
    hf_config: KimiK31TextConfig | KimiK31Config,
    ep_info: dict[str, Any],
    sp_info: dict[str, Any],
    engine_config: EngineConfig,
) -> nn.Module:
    if not isinstance(module, KimiK31MoE):
        return module
    return KimiK31MoE(
        hf_config.get_text_config(),
        ep_group=ep_info["ep_group"],
        sp_group=sp_info["sp_group"],
        aux_loss_coeff=engine_config.moe.aux_loss_coeff,
        z_loss_coeff=engine_config.moe.z_loss_coeff,
    )


def get_parallel_plan(*, multimodal: bool = False) -> FSDPParallelPlan:
    # Replicate these small parameters to preserve their FP32 values. The short
    # convolution casts its weight slice to the projection dtype for the kernel.
    fp32_kda_parameters = (
        "A_log", "dt_bias", "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "o_norm.weight",
    )
    plan = {
        "model.layers.*": FSDPPlanSpec(activation_checkpoint=True),
        "model.layers.*.mlp": FSDPPlanSpec(replace_module=_place_moe),
        "model.layers.*.mlp.experts": FSDPPlanSpec(placement=FSDPPlacementSpec(Shard(0), "edp")),
        "model.embed_tokens": FSDPPlanSpec(standalone_world_shard=True),
        "lm_head": FSDPPlanSpec(standalone_world_shard=True),
        **{
            f"model.layers.*.self_attn.{name}": FSDPPlanSpec(no_shard=True)
            for name in fp32_kda_parameters
        },
    }
    if multimodal:
        plan = {f"language_model.{name}": spec for name, spec in plan.items()}
        plan["vision_tower.encoder.blocks.*"] = FSDPPlanSpec(activation_checkpoint=True)
    return FSDPParallelPlan(plan=plan, sp_size=1, ep_size=1)
