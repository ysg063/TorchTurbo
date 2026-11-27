from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import ProcessGroup
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig

from torchturbo.models.modules.attention.ulysses.ulysses import (
    all_gather_sp_with_grad_reduce,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
)
from torchturbo.ops import ops
from torchturbo.ops.packed_params import get_packed_params


@dataclass(frozen=True)
class KimiK31DSALayout:
    """Local queries and the visible KV prefixes, prepared once per microbatch."""

    # Q、KV 各自使用从 0 开始的累计长度。两组边界按文档一一对应，长度可以不同。
    # kv_start 和 kv_end 则是 SP 拼回全序列后的全局位置，用于裁出当前 rank 的 KV。
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    num_query_tokens: int
    kv_start: int
    kv_end: int
    indexer_metadata: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        batch_size: int,
        local_length: int,
        device: torch.device,
        *,
        cu_seqlens: torch.Tensor | None = None,
        sp_rank: int = 0,
    ) -> KimiK31DSALayout:
        """Use leading-zero global packed offsets, excluding right padding.

        With no offsets, each batch row is an unpadded sequence. Packed input
        uses one batch row. SP assigns contiguous equal-length sequence shards.
        """
        start = local_length * sp_rank
        end = start + local_length
        if cu_seqlens is None:
            # 每个 batch 行是一篇完整文档。Q 只取当前分片，KV 从各行开头取到分片末尾。
            batches = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
            return cls(batches * local_length, batches * end, batch_size * local_length, 0, end)

        if batch_size != 1:
            raise ValueError("Kimi packed DSA expects a single flattened batch row")
        offsets = cu_seqlens.tolist()
        end = min(end, offsets[-1])
        query_offsets, key_offsets = [0], [0]
        key_start = end
        # 只保留与当前 Q 分片相交的文档。若分片从文档中间开始，KV 要回溯到该文档开头。
        # 例如文档边界为 [0, 3, 7]、本 rank 覆盖 [4, 8)，有效 Q 为 [4, 7)，KV 为 [3, 7)。
        # 此时 Q/KV 的累计长度分别为 [0, 3] 和 [0, 4]。右对齐因果掩码会让第一个 Q 看见 2 个 K。
        for left, right in pairwise(offsets):
            if right <= start:
                continue
            if left >= end:
                break
            if len(query_offsets) == 1:
                key_start = left
            query_offsets.append(min(right, end) - start)
            key_offsets.append(min(right, end) - key_start)

        return cls(
            torch.tensor(query_offsets, dtype=torch.int32, device=device),
            torch.tensor(key_offsets, dtype=torch.int32, device=device),
            query_offsets[-1],
            key_start,
            end,
        )


class KimiK31Indexer(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        dsa = config.dsa_config
        if dsa["index_k_norm_type"] != "layernorm" or dsa["index_rope_dim"] != 0:
            raise ValueError("Kimi K3.1 expects LayerNorm and NoPE for the DSA indexer")
        self.num_heads = dsa["index_num_heads"]
        self.head_dim = dsa["index_head_dim"]
        self.topk = dsa["index_topk"]
        self.score_scale = (self.head_dim * self.num_heads) ** -0.5
        # Indexer 和主 Attention 使用各自的投影。每个 indexer Q 有多个头，K 只有一个共享头。
        # weights_proj 为每个 query 产生一组头权重，供算子汇总各 indexer 头的打分。
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        local_states: torch.Tensor,
        q_latent: torch.Tensor,
        kv_start: int,
        kv_end: int,
        sp_group: ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project indexer inputs and isolate its gradients from the text backbone."""
        # 复用主 Attention 的低秩 Q 和层输入，同时切断这两处到主干的梯度。
        # 开启 KL 训练时，梯度仍会经过下面的 indexer 投影与 K 的 LayerNorm。
        index_query = self.wq_b(q_latent.detach()).unflatten(-1, (self.num_heads, self.head_dim))
        index_key = self.k_norm(self.wk(hidden_states.detach()))
        index_weights = self.weights_proj(local_states.detach()).float() * self.score_scale
        sp_size = sp_group.size() if sp_group is not None else 1
        # 拼齐各 rank 的 K，再取本 rank 所需的文档前缀。反向把对远端 K 的梯度加回所属 rank。
        index_key = all_gather_sp_with_grad_reduce(
            index_key, dim=1, sp_group=sp_group, sp_size=sp_size
        )
        index_key = index_key[:, kv_start:kv_end].flatten(0, 1).unsqueeze(1)
        return index_query, index_key, index_weights


def _kl_chunk(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    index_weights: torch.Tensor,
    indices: torch.Tensor,
    key_offsets: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_key: torch.Tensor,
    teacher_query_extra: torch.Tensor,
    teacher_key_extra: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    # top-k 下标以各自的 KV 文档为起点。加上该文档的偏移后，才能从拼接的 K 张量取行。
    # -1 表示未填满的槽位，暂时指向第 0 行以便 gather，随后用 valid 掩掉。
    valid = indices >= 0
    rows = (key_offsets[:, None] + indices).masked_fill(~valid, 0).long()
    with torch.autocast(device_type=index_query.device.type, enabled=False):
        # 学生分布只在已选 token 上重算。先对各头的 QK 点积取 ReLU，再用头权重求和。
        selected_keys = index_key[rows, 0].float()
        scores = (index_query.float() @ selected_keys.transpose(1, 2)).relu()
        scores = (scores * index_weights[:, :, None]).sum(dim=1)
        log_probs = scores.masked_fill(~valid, -torch.inf).log_softmax(dim=-1)

        # 教师来自主 Attention。用算子返回的每头 softmax 最大值 m 和指数和 s，
        # 恢复所选 token 的概率 exp(score - m) / s，再跨主 Attention 的头取平均。
        selected_teacher_keys = teacher_key[rows, 0].float()
        selected_teacher_extra = teacher_key_extra[rows, 0].float()
        teacher_scores = teacher_query.float() @ selected_teacher_keys.transpose(1, 2)
        teacher_scores = teacher_scores + (
            teacher_query_extra.float() @ selected_teacher_extra.transpose(1, 2)
        )
        teacher_scores.mul_(softmax_scale)
        head_probs = (teacher_scores - softmax_max[:, :, None]).masked_fill(
            ~valid[:, None], -torch.inf
        ).exp() / softmax_sum[:, :, None]
        target = head_probs.mean(dim=1)

        # 返回 KL(教师 || 学生) 在本块所有有效 query 上的总和，归一化留给 loss 入口。
        return (target.xlogy(target) - target * log_probs.masked_fill(~valid, 0)).sum()


def sparse_indexer_kl_sum(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    index_weights: torch.Tensor,
    indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    query_extra: torch.Tensor,
    key_extra: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    softmax_scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> torch.Tensor:
    """Sum KL from detached, head-averaged attention on the selected tokens."""
    # 每个 Q 对应一个 KV 文档起点。按该文档在本 rank 的 query 数重复该起点。
    key_offsets = torch.repeat_interleave(
        cu_seqlens_k[:-1], cu_seqlens_q.diff(), output_size=index_query.size(0)
    )
    loss = index_weights.new_zeros(())
    # 每次只处理 16 个 Q。反向重算所选 K 的 gather 和打分矩阵，减少需要保留的激活。
    for start in range(0, index_query.size(0), 16):
        rows = slice(start, start + 16)
        loss = loss + checkpoint(
            _kl_chunk,
            index_query[rows], index_key, index_weights[rows], indices[rows, 0], key_offsets[rows],
            query[rows], key, query_extra[rows], key_extra,
            softmax_max[0, rows], softmax_sum[0, rows], softmax_scale,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return loss


class KimiK31DSAAttention(nn.Module):
    """Gated sparse MLA in the shared KV latent space."""

    def __init__(self, config: PretrainedConfig, *, train_indexer: bool = False) -> None:
        super().__init__()
        dropout = vars(config).get("attention_dropout", 0.0)
        if not config.mla_use_nope or not config.mla_use_output_gate or dropout:
            raise ValueError("Kimi K3.1 DSA requires NoPE, the MLA output gate, and zero dropout")
        if config.num_key_value_heads != config.num_attention_heads:
            raise ValueError("Kimi K3.1 DSA expects one KV projection per query head")
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        # 权重吸收改变了算子的输入宽度，softmax 缩放仍按吸收前的 Q/K 头维度计算。
        self.softmax_scale = self.q_head_dim**-0.5
        self.train_indexer = train_indexer

        hidden_size = config.hidden_size
        self.q_a_proj = nn.Linear(hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank, eps=1e-6)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.g_proj = nn.Linear(hidden_size, self.num_heads * self.v_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden_size, bias=False)
        self.indexer = KimiK31Indexer(config)
        self.indexer.requires_grad_(train_indexer)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layout: KimiK31DSALayout,
        *,
        sp_group: ProcessGroup | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return local attention output and the optional sum of valid-query KL losses."""
        batch_size, local_length, hidden_size = hidden_states.shape
        sp_size = sp_group.size() if sp_group is not None else 1
        num_queries = layout.num_query_tokens
        # Q 保留在当前 rank，只为有效 token 展开各头的 Q，并计算 indexer 权重和输出门。
        local_states = hidden_states.reshape(-1, hidden_size)[:num_queries]

        q_latent = self.q_a_layernorm(self.q_a_proj(hidden_states)).flatten(0, 1)[:num_queries]
        query = self.q_b_proj(q_latent).unflatten(-1, (self.num_heads, self.q_head_dim))
        # extra 是权重和后端接口中沿用的额外打分通道，本模型直接使用投影结果。
        query_main, query_extra = query.split((self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)
        kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, kv_extra = kv.split((self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        kv = torch.cat((self.kv_a_layernorm(kv_latent), kv_extra), dim=-1)
        # SP 交换压缩后的单头 KV。拼回完整序列后，裁出本 rank 需要的 KV 前缀。
        kv = all_gather_sp_with_grad_reduce(kv, dim=1, sp_group=sp_group, sp_size=sp_size)
        kv = kv[:, layout.kv_start : layout.kv_end].flatten(0, 1).unsqueeze(1)
        key, key_extra = kv.split((self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)

        # 原本每个头用 Wk 将共享 latent c 展开成 K。把点积 q · (Wk c) 改写成 (q Wk) · c，
        # 就可以直接用共享 latent 算分数。Wv 留到 attention 聚合完成后再展开每个头的 V。
        kv_weight = self.kv_b_proj.weight.unflatten(0, (self.num_heads, -1))
        key_weight, value_weight = kv_weight.split((self.qk_nope_head_dim, self.v_head_dim), dim=1)
        query = torch.einsum("thd,hdc->thc", query_main, key_weight)
        index_query, index_key, index_weights = self.indexer(
            hidden_states, local_states, q_latent, layout.kv_start, layout.kv_end, sp_group
        )
        train_indexer = self.train_indexer and self.training and torch.is_grad_enabled()
        indexer_loss = None
        if num_queries:
            # 两种训练模式都先用 indexer 从因果可见的 K 中选 top-k。输出是各 KV 文档内的下标，
            # 形状为 [有效 query 数, 1, topk]，供所有主 Attention 头共享。
            indexer_kwargs = {}
            if layout.indexer_metadata is not None:
                indexer_kwargs["metadata"] = layout.indexer_metadata
            with torch.no_grad():
                indices, _ = ops.lightning_indexer(
                    index_query, index_key, index_weights, self.indexer.topk,
                    cu_seqlens_q=layout.cu_seqlens_q,
                    cu_seqlens_k=layout.cu_seqlens_k,
                    **indexer_kwargs,
                )
            # 主 Attention 用自己的 Q/K 在选中的位置重新打分，并加权汇总共享 latent。
            # 所以这里的 K 和 V 传入同一个张量。返回的 softmax 统计量供可选 KL 分支复用。
            latent_output, softmax_max, softmax_sum = ops.sparse_flash_attention(
                query, key, key, indices, self.softmax_scale,
                query_extra=query_extra,
                key_extra=key_extra,
                cu_seqlens_q=layout.cu_seqlens_q,
                cu_seqlens_k=layout.cu_seqlens_k,
            )
            if train_indexer:
                # 主 Attention 作为教师只提供目标分布。KL 的梯度通过 indexer 重新计算的分数回传。
                indexer_loss = sparse_indexer_kl_sum(
                    index_query, index_key, index_weights, indices,
                    query.detach(), key.detach(), query_extra.detach(), key_extra.detach(),
                    softmax_max, softmax_sum, self.softmax_scale,
                    layout.cu_seqlens_q, layout.cu_seqlens_k,
                )
        else:
            # 全 padding 的 rank 跳过两个稀疏算子。保留连到 KV 的零值计算图，
            # 让本 rank 仍参加 all-gather 的反向通信，接收其他 rank 对它持有的 K/V 的梯度。
            latent_output = query + (kv.sum() + query_extra.sum()) * 0
            if train_indexer:
                indexer_loss = (index_query.sum() + index_key.sum() + index_weights.sum()) * 0

        # 在各头展开 V，乘以对应 token 的 sigmoid 输出门，再投影回模型宽度。
        # 末尾补回零行，保持输出与本 rank 的残差张量形状一致。
        values = torch.einsum("thc,hvc->thv", latent_output, value_weight).flatten(1)
        output = self.o_proj(values * self.g_proj(local_states).sigmoid())
        output = F.pad(output, (0, 0, 0, batch_size * local_length - num_queries))
        return output.reshape_as(hidden_states), indexer_loss


class KimiK31KDA(nn.Module):
    """Kimi Delta Attention with a low-rank decay and a full-rank output gate."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        linear = config.linear_attn_config
        if not linear["use_full_rank_gate"]:
            raise ValueError("Kimi K3.1 KDA requires a full-rank output gate")
        self.num_heads = linear["num_heads"]
        self.head_dim = linear["head_dim"]
        self.lower_bound = linear["gate_lower_bound"]
        channels = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, channels, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, channels, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, channels, bias=False)
        conv_size = linear["short_conv_kernel_size"]
        self.q_conv1d = nn.Conv1d(channels, channels, conv_size, groups=channels, bias=False)
        self.k_conv1d = nn.Conv1d(channels, channels, conv_size, groups=channels, bias=False)
        self.v_conv1d = nn.Conv1d(channels, channels, conv_size, groups=channels, bias=False)
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, channels, bias=False)
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        decay_heads = linear.get("checkpoint_num_heads", self.num_heads)
        self.A_log = nn.Parameter(torch.empty(decay_heads, dtype=torch.float32).uniform_(1, 16).log_())
        self.dt_bias = nn.Parameter(torch.zeros(channels, dtype=torch.float32))
        self.g_proj = nn.Linear(config.hidden_size, channels, bias=False)
        self.o_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(channels, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        *,
        sp_group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Run the convolutions and recurrence after gathering sequence shards."""
        sp_size = sp_group.size() if sp_group is not None else 1
        sp_rank = sp_group.rank() if sp_group is not None else 0
        heads_per_rank, remainder = divmod(self.num_heads, sp_size)
        if remainder:
            raise ValueError("Kimi KDA head count must be divisible by SP size")
        head_slice = slice(sp_rank * heads_per_rank, (sp_rank + 1) * heads_per_rank)
        channels = slice(head_slice.start * self.head_dim, head_slice.stop * self.head_dim)

        packed = get_packed_params(cu_seqlens, hidden_states.size(1) * sp_size, device=hidden_states.device)

        def gather(tensor: torch.Tensor) -> torch.Tensor:
            if sp_size == 1:
                return tensor
            return gather_seq_scatter_heads(tensor, 1, 2, sp_group, sp_size)

        def project_and_convolve(projection: nn.Linear, conv: nn.Conv1d) -> torch.Tensor:
            projected = gather(projection(hidden_states))
            # Keep the replicated parameter in FP32 and match the kernel input dtype.
            weight = conv.weight[channels, 0].to(projected.dtype)
            convolved, _ = ops.causal_conv1d(
                projected, weight, activation="silu", packed_params=packed,
            )
            return convolved

        q = project_and_convolve(self.q_proj, self.q_conv1d)
        k = project_and_convolve(self.k_proj, self.k_conv1d)
        v = project_and_convolve(self.v_proj, self.v_conv1d)
        decay = gather(self.f_b_proj(self.f_a_proj(hidden_states)))
        beta = gather(self.b_proj(hidden_states).float())
        output, _ = ops.kda(
            q, k, v, decay.unflatten(-1, (heads_per_rank, self.head_dim)), beta,
            A_log=self.A_log[head_slice], dt_bias=self.dt_bias[channels],
            use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True, safe_gate=True, lower_bound=self.lower_bound,
            cu_seqlens=packed.cu_seqlens if packed is not None else None,
        )
        if sp_size > 1:
            output = gather_heads_scatter_seq(output, 2, 1, sp_group, sp_size)

        gate = self.g_proj(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        output = F.rms_norm(output.float(), (self.head_dim,), self.o_norm.weight.float(), self.o_norm.eps)
        output = (output * gate.float().sigmoid()).to(hidden_states.dtype)
        return self.o_proj(output.flatten(-2))
