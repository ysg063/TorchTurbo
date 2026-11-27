from __future__ import annotations

import torch
import torch.nn.functional as F

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="sparse_flash_attention",
    impl_name="npu_sparse_flash_attention",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuSparseFlashAttention(OpImpl):
    def available(self) -> bool:
        try:
            from torch_npu import npu_sparse_flash_attention
        except (ImportError, OSError, RuntimeError):
            return False
        return callable(npu_sparse_flash_attention)

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sparse_indices: torch.Tensor,
        softmax_scale: float,
        *,
        query_extra: torch.Tensor,
        key_extra: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torch_npu import npu_sparse_flash_attention

        num_heads = query.size(1)
        if num_heads == 96:
            # 此 CANN 训练路径将 Kimi 的 96 个 Q 头补到 128，适配反向算子的头数规格。
            # Q 的主通道和额外打分通道一起补零。共享的单头 K/V 和 top-k 下标沿用原布局。
            head_padding = (0, 0, 0, 128 - num_heads)
            query = F.pad(query, head_padding)
            query_extra = F.pad(query_extra, head_padding)

        output, softmax_max, softmax_sum, *_ = npu_sparse_flash_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            sparse_indices.contiguous(),
            softmax_scale,
            # 公共接口传带前导 0 的累计长度。CANN TND 接口接收每段的累计结束位置。
            actual_seq_lengths_query=cu_seqlens_q[1:].to(torch.int32).contiguous(),
            actual_seq_lengths_kv=cu_seqlens_k[1:].to(torch.int32).contiguous(),
            query_rope=query_extra.contiguous(),
            key_rope=key_extra.contiguous(),
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=True,
        )
        # 模型侧只消费真实的头，切片的反向会给补出的 32 个头传零梯度。
        return (
            output[:, :num_heads].contiguous(),
            softmax_max[..., :num_heads].detach().contiguous(),
            softmax_sum[..., :num_heads].detach().contiguous(),
        )
