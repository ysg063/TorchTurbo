from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


class _FlashMlaSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from flash_mla import flash_mla_sparse_fwd

        heads = query.size(1)
        q_kernel = F.pad(query, (0, 0, 0, 32)) if heads == 96 else query
        output, maximum, lse = flash_mla_sparse_fwd(
            q_kernel, kv, indices.unsqueeze(1), scale, d_v=512, topk_length=lengths,
        )
        output = output[:, :heads].contiguous()
        maximum = maximum[:, :heads].unsqueeze(0).contiguous()
        lse = lse[:, :heads].contiguous()
        denominator = (lse.unsqueeze(0) - maximum).exp()
        ctx.save_for_backward(query, kv.squeeze(1), output, lse, indices, lengths)
        ctx.scale = scale
        ctx.mark_non_differentiable(maximum, denominator)
        return output, maximum, denominator

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor, _grad_max: torch.Tensor, _grad_sum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        from cudnn import DSA

        query, kv, output, lse, indices, lengths = ctx.saved_tensors
        sink = torch.full((query.size(1),), -torch.inf, dtype=torch.float32, device=query.device)
        grads = DSA.sparse_attention_backward_wrapper(
            query, kv, output, grad_output.contiguous(), lse, sink, indices.clamp_min(0),
            softmax_scale=ctx.scale, topk_length=lengths,
        )
        return grads["dq"], grads["dkv"].unsqueeze(1), None, None, None


@register_op(
    op_name="sparse_flash_attention",
    impl_name="flash_mla_cudnn_sparse_attention",
    kind=OpImplKind.CUSTOM,
    priority=-10,
    devices=(TurboDevice.CUDA,),
)
class FlashMlaCudnnSparseAttention(OpImpl):
    """FlashMLA prefill and cuDNN native-head backward for shared latent KV."""

    def available(self) -> bool:
        try:
            from cudnn import DSA
            from flash_mla import flash_mla_sparse_fwd
            backward = DSA.sparse_attention_backward_wrapper
        except (AttributeError, ImportError, OSError, RuntimeError):
            return False
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] == 10
            and callable(backward)
            and callable(flash_mla_sparse_fwd)
        )

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
        if key is not value or key.shape[1:] != (1, 512) or key_extra.size(-1) != 64:
            raise ValueError("CUDA sparse MLA requires a shared 512-channel KV latent and 64 extra channels")
        if query.dtype != torch.bfloat16 or query.size(1) not in (64, 96, 128):
            raise ValueError("CUDA sparse MLA requires BF16 queries with 64, 96, or 128 heads")

        rows = torch.arange(query.size(0), dtype=torch.int32, device=query.device)
        documents = torch.bucketize(rows, cu_seqlens_q[1:], right=True)
        remaining = cu_seqlens_q[documents + 1] - rows
        kv_end = cu_seqlens_k[documents + 1]
        local_indices = sparse_indices.squeeze(1)
        global_indices = local_indices + cu_seqlens_k[documents].unsqueeze(-1)
        visible = (local_indices >= 0) & (global_indices <= (kv_end - remaining)[:, None])
        lengths = visible.sum(dim=-1, dtype=torch.int32)
        sentinel = key.size(0)
        indices = global_indices.masked_fill(~visible, sentinel).sort(dim=-1).values
        indices = indices.masked_fill(indices == sentinel, -1).to(torch.int32)
        indices = F.pad(indices, (0, (-indices.size(-1)) % 512), value=-1)
        query = torch.cat((query, query_extra), dim=-1)
        kv = torch.cat((key, key_extra), dim=-1)
        return _FlashMlaSparseAttention.apply(query, kv, indices, lengths, softmax_scale)
