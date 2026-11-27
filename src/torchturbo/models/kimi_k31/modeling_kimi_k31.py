from __future__ import annotations

from dataclasses import replace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import ProcessGroup
from transformers import PretrainedConfig

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoERuntimeInputs
from torchturbo.models.modules.moe.router import TopKRouter
from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.ops import ops

from .attention import KimiK31DSAAttention, KimiK31DSALayout, KimiK31KDA
from .configuration_kimi_k31 import KimiK31TextConfig


class KimiK31Router(TopKRouter):
    """Sigmoid router with Kimi's global-batch quantile-balancing estimator."""

    num_histogram_bins: int = 1000

    def __init__(self, config: MoEConfig, sp_group: ProcessGroup | None) -> None:
        super().__init__(config, sp_group=sp_group)
        self.quantile_balancing = False
        self._histogram: torch.Tensor | None = None

    def routing(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        accumulate_tokens_per_expert: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.quantile_balancing:
            return super().routing(logits, input_ids, padding_mask, False)

        logits = self.apply_z_loss(logits.reshape(-1, self.num_experts), padding_mask=padding_mask)
        scores = logits.float().sigmoid()
        with torch.no_grad():
            selected = (scores + self.expert_bias).topk(self.topk + 1, dim=-1)
            indices = selected.indices[:, :-1]
            if self.training and accumulate_tokens_per_expert and logits.requires_grad:
                self._record_margins(scores, selected.values[:, -1:], padding_mask)

        selected_scores = scores.gather(-1, indices)
        if self.topk > 1:
            selected_scores = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
        selected_scores = selected_scores * self.config.moe_router_topk_scaling_factor
        probs = torch.zeros_like(scores).scatter(1, indices, selected_scores)
        routes = torch.zeros_like(scores, dtype=torch.bool).scatter(1, indices, True)
        return probs, routes, indices

    def _record_margins(
        self, scores: torch.Tensor, cutoff: torch.Tensor, padding_mask: torch.Tensor | None,
    ) -> None:
        num_bins = self.num_histogram_bins
        if self._histogram is None:
            self._histogram = scores.new_zeros((self.num_experts, num_bins), dtype=torch.int32)
        bias_min, bias_max = self.expert_bias.aminmax()
        lower, upper = bias_min - 1.0, bias_max + 1.0
        offsets = torch.arange(self.num_experts, device=scores.device) * num_bins
        valid = torch.ones(scores.size(0), dtype=torch.int32, device=scores.device)
        if padding_mask is not None:
            valid = (~padding_mask.reshape(-1)).to(torch.int32)

        # Bound the temporary index tensor independently of sequence length.
        chunk_size = max(1, 2**20 // self.num_experts)
        for start in range(0, scores.size(0), chunk_size):
            rows = slice(start, start + chunk_size)
            margins = cutoff[rows] - scores[rows]
            bins = ((margins - lower) * (num_bins / (upper - lower))).floor().long()
            bins = (bins.clamp_(0, num_bins - 1) + offsets).flatten()
            counts = valid[rows, None].expand(-1, self.num_experts).reshape(-1)
            self._histogram.view(-1).scatter_add_(0, bins, counts)

    @torch.no_grad()
    def finish_quantile_step(self, group: ProcessGroup | None) -> None:
        """Reduce the histogram and replace the bias with centered interpolated quantiles."""
        histogram = self._histogram
        if histogram is None:
            return
        if group is not None and group.size() > 1:
            dist.all_reduce(histogram, group=group)
        cumulative = histogram.cumsum(dim=-1, dtype=torch.int64)
        num_tokens = cumulative[0, -1]
        quantile = num_tokens.float() * (self.topk / self.num_experts)
        rank = quantile.ceil().long().clamp_min(1)
        bins = (cumulative >= rank).long().argmax(dim=-1, keepdim=True)
        counts = histogram.gather(1, bins).squeeze(1)
        before = cumulative.gather(1, bins).squeeze(1) - counts
        fraction = ((quantile - before) / counts.clamp_min(1)).clamp(0, 1)
        bias_min, bias_max = self.expert_bias.aminmax()
        width = (bias_max - bias_min + 2.0) / self.num_histogram_bins
        updated = bias_min - 1.0 + (bins.squeeze(1) + fraction) * width
        centered = updated - updated.mean()
        self.expert_bias.copy_(torch.where(num_tokens > 0, centered, self.expert_bias))
        histogram.zero_()


def situ_glu(gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    """Apply SiTU to the gate and bounded linear activation to the up branch."""
    gate_fp32 = gate.float()
    activated_gate = beta * (gate_fp32 / beta).tanh() * gate_fp32.sigmoid()
    activated_up = linear_beta * (up.float() / linear_beta).tanh()
    return (activated_gate * activated_up).to(gate.dtype)


class KimiK31MLP(nn.Module):
    def __init__(self, config: PretrainedConfig, intermediate_size: int) -> None:
        super().__init__()
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(situ_glu(gate, up, self.beta, self.linear_beta))


class KimiK31Experts(GroupedMLP):
    """Grouped latent experts with the standard packed weight layout."""

    def __init__(self, num_local_experts: int, config: MoEConfig, beta: float, linear_beta: float) -> None:
        super().__init__(num_local_experts, use_te=False, config=config)
        self.beta = beta
        self.linear_beta = linear_beta
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.config.init_method(self.weight1)
        self.config.init_method(self.weight2)

    def activation_func_with_probs(self, hidden_states: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Apply SiTU and the routing weight before the expert down projection."""
        gate, up = hidden_states.chunk(2, dim=-1)
        # Round once after incorporating the FP32 routing probability.
        activated = situ_glu(gate.float(), up.float(), self.beta, self.linear_beta)
        return (activated * probs).to(hidden_states.dtype)


class KimiK31MoE(MoELayer):
    """Route at model width, run experts at latent width, then combine and lift."""

    def __init__(
        self,
        hf_config: PretrainedConfig,
        *,
        ep_group: ProcessGroup | None = None,
        sp_group: ProcessGroup | None = None,
        aux_loss_coeff: float = 0.0,
        z_loss_coeff: float = 0.0,
    ) -> None:
        # The router and the experts intentionally use different hidden sizes.
        nn.Module.__init__(self)
        ep_size = ep_group.size() if ep_group is not None else 1
        ep_rank = ep_group.rank() if ep_group is not None else 0
        self.num_local_experts, remainder = divmod(hf_config.num_experts, ep_size)
        if remainder:
            raise ValueError("Kimi expert count must be divisible by EP size")
        self.expert_parallel_size = ep_size
        first_expert = ep_rank * self.num_local_experts
        self.local_expert_indices = list(range(first_expert, first_expert + self.num_local_experts))
        self._runtime_inputs = MoERuntimeInputs()
        self.config = MoEConfig(
            hidden_size=hf_config.hidden_size,
            num_moe_experts=hf_config.num_experts,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            expert_model_parallel_size=ep_size,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,
            moe_router_topk=hf_config.num_experts_per_token,
            moe_router_score_function=hf_config.moe_router_activation_func,
            moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
            moe_router_dtype="fp32",
            moe_router_enable_expert_bias=True,
            moe_aux_loss_coeff=aux_loss_coeff,
            moe_z_loss_coeff=z_loss_coeff,
            init_method_std=hf_config.initializer_range,
        )
        if hf_config.num_expert_group != 1 or not hf_config.moe_renormalize:
            raise ValueError("Kimi K3.1 MoE expects ungrouped, renormalized routing")
        self.router = KimiK31Router(self.config, sp_group=sp_group)
        self.token_dispatcher = MoEAlltoAllTokenDispatcher(
            self.num_local_experts, self.local_expert_indices, self.config, ep_group,
        )
        latent_size = hf_config.routed_expert_hidden_size
        expert_config = replace(self.config, hidden_size=latent_size)
        self.experts = KimiK31Experts(
            self.num_local_experts, expert_config,
            hf_config.activation_situ_beta, hf_config.activation_situ_linear_beta,
        )
        self.routed_expert_down_proj = nn.Linear(hf_config.hidden_size, latent_size, bias=False)
        self.routed_expert_norm = (
            nn.RMSNorm(latent_size, eps=hf_config.rms_norm_eps)
            if hf_config.latent_moe_use_norm else nn.Identity()
        )
        self.routed_expert_up_proj = nn.Linear(latent_size, hf_config.hidden_size, bias=False)
        self.shared_expert = None
        self.shared_expert_gate = None
        self.shared_experts = KimiK31MLP(
            hf_config, hf_config.num_shared_experts * hf_config.moe_intermediate_size,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None,
        *, padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run routed experts at latent width and shared experts at model width."""
        runtime_inputs = self._runtime_inputs
        if padding_mask is None:
            padding_mask = runtime_inputs.padding_mask
        probs, routing_map, expert_indices, logits = self.router(
            hidden_states, input_ids, padding_mask, runtime_inputs.accumulate_tokens_per_expert,
        )
        logits = logits.view(-1, logits.shape[-1])

        dispatcher = self.token_dispatcher
        use_ep = self.expert_parallel_size > 1
        expert_tokens, expert_probs = dispatcher.dispatch_preprocess(
            self.routed_expert_down_proj(hidden_states), routing_map, probs,
            expert_indices=expert_indices,
        )
        if use_ep:
            expert_tokens, expert_probs = dispatcher.token_dispatch(expert_tokens, expert_probs)
        expert_tokens, tokens_per_expert, expert_probs = dispatcher.dispatch_postprocess(
            expert_tokens, expert_probs,
        )
        expert_tokens, bias = self.experts(expert_tokens, tokens_per_expert, expert_probs)
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"
        expert_tokens = dispatcher.combine_preprocess(expert_tokens)
        if use_ep:
            expert_tokens = dispatcher.token_combine(expert_tokens)
        expert_tokens = dispatcher.combine_postprocess(expert_tokens)

        routed = self.routed_expert_norm(expert_tokens)
        routed = self.routed_expert_up_proj(routed)
        output = routed + self.shared_experts(hidden_states)
        return output, logits


def mix_attention_residuals(
    current: torch.Tensor,
    completed: tuple[torch.Tensor, ...],
    score_norm: nn.RMSNorm,
    score_proj: nn.Linear,
) -> torch.Tensor:
    """Mix completed blocks and the current sum using token-wise FP32 scores."""
    if not completed:
        return current

    states = torch.stack((*completed, current), dim=-2).float()
    inv_rms = (states.square().mean(dim=-1) + score_norm.eps).rsqrt()
    score_weight = score_norm.weight.float() * score_proj.weight.flatten().float()
    scores = (states * score_weight).sum(dim=-1) * inv_rms
    result = (states * scores.softmax(dim=-1).unsqueeze(-1)).sum(dim=-2)
    return result.to(current.dtype)


class KimiK31DecoderLayer(nn.Module):
    def __init__(self, config: KimiK31TextConfig, layer_idx: int, *, train_indexer: bool) -> None:
        super().__init__()
        self.block_start = layer_idx % config.attn_res_block_size == 0
        # 按官方 dsa_layers 列表选 DSA 层。列表以 1 为起点，is_dsa_layer 负责转换层号。
        self.is_dsa = config.is_dsa_layer(layer_idx)
        self.is_moe = config.is_moe_layer(layer_idx)
        self.self_attn = (
            KimiK31DSAAttention(config, train_indexer=train_indexer) if self.is_dsa else KimiK31KDA(config)
        )
        self.mlp = KimiK31MoE(config) if self.is_moe else KimiK31MLP(config, config.intermediate_size)
        dim = config.hidden_size
        self.input_layernorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
        self.self_attention_res_norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
        self.self_attention_res_proj = nn.Linear(dim, 1, bias=False)
        self.mlp_res_norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
        self.mlp_res_proj = nn.Linear(dim, 1, bias=False)

    def forward(
        self,
        current: torch.Tensor,
        completed: tuple[torch.Tensor, ...],
        dsa_layout: KimiK31DSALayout,
        cu_seqlens: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        sp_group: ProcessGroup | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor | None]:
        attn_input = mix_attention_residuals(
            current, completed, self.self_attention_res_norm, self.self_attention_res_proj,
        )
        attn_input = self.input_layernorm(attn_input)
        if self.is_dsa:
            # DSA 接收本 rank 的 Q/KV 边界，返回主 Attention 输出和本层可选的 indexer KL 总和。
            attn_output, indexer_loss = self.self_attn(attn_input, dsa_layout, sp_group=sp_group)
        else:
            attn_output = self.self_attn(attn_input, cu_seqlens, sp_group=sp_group)
            indexer_loss = None
        if self.block_start:
            completed = (*completed, current)
            current = attn_output
        else:
            current = current + attn_output
        mlp_input = mix_attention_residuals(current, completed, self.mlp_res_norm, self.mlp_res_proj)
        mlp_input = self.post_attention_layernorm(mlp_input)
        if self.is_moe:
            mlp_output, _ = self.mlp(mlp_input, padding_mask=padding_mask)
        else:
            mlp_output = self.mlp(mlp_input)
        return current + mlp_output, completed, indexer_loss


class KimiK31TextModel(nn.Module):
    """KDA/DSA decoder shared by text and multimodal entry points."""

    def __init__(self, config: KimiK31TextConfig, *, train_indexer: bool = False) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            KimiK31DecoderLayer(config, layer_idx, train_indexer=train_indexer)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.num_dsa_layers = sum(layer.is_dsa for layer in self.layers)
        self._dsa_config = config.dsa_config
        self.output_attn_res_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sp_group: ProcessGroup | None = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Consume local embeddings and global packed document boundaries."""
        batch, length, _ = inputs_embeds.shape
        sp_rank = self.sp_group.rank() if self.sp_group is not None else 0
        # 文档边界和 SP 分片在各 DSA 层相同，每个 microbatch 构造一次布局供各层复用。
        layout = KimiK31DSALayout.build(
            batch, length, inputs_embeds.device, cu_seqlens=cu_seqlens, sp_rank=sp_rank,
        )
        if (
            self.num_dsa_layers
            and layout.num_query_tokens
            and ops.get_impl_name("lightning_indexer") == "npu_lightning_indexer"
        ):
            # All DSA layers use the same indexer dimensions and local sequence boundaries.
            metadata = ops.lightning_indexer_metadata(
                num_heads_q=self._dsa_config["index_num_heads"],
                num_heads_k=1,
                head_dim=self._dsa_config["index_head_dim"],
                topk=self._dsa_config["index_topk"],
                cu_seqlens_q=layout.cu_seqlens_q,
                cu_seqlens_k=layout.cu_seqlens_k,
                cmp_residual_k=None,
                batch_size=layout.cu_seqlens_q.numel() - 1,
                max_seqlen_q=-1,
                max_seqlen_k=-1,
            )
            layout = replace(layout, indexer_metadata=metadata)
        padding_mask = None
        if cu_seqlens is not None:
            positions = torch.arange(length, device=inputs_embeds.device)
            padding_mask = (positions >= layout.num_query_tokens).unsqueeze(0)

        current = inputs_embeds
        completed = ()
        # 各 DSA 层返回本 rank 的 KL 总和。从各层汇总后，再由 loss 入口按 query 数归一化。
        total_indexer_loss = None
        for layer in self.layers:
            current, completed, indexer_loss = layer(
                current, completed, layout, cu_seqlens, padding_mask, self.sp_group,
            )
            if indexer_loss is not None:
                total_indexer_loss = (
                    indexer_loss if total_indexer_loss is None else total_indexer_loss + indexer_loss
                )
        mixed = mix_attention_residuals(current, completed, self.output_attn_res_norm, self.output_attn_res_proj)
        return self.norm(mixed), total_indexer_loss, layout.num_query_tokens
