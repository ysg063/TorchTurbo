from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="sparse_flash_attention",
    impl_name="torch_sparse_flash_attention",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchSparseFlashAttention(OpImpl):
    """PyTorch reference that gathers the selected KV rows explicitly."""

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
        query_rows = torch.arange(query.size(0), device=query.device)
        sequence_ids = torch.bucketize(query_rows, cu_seqlens_q[1:], right=True)
        kv_start = cu_seqlens_k[sequence_ids]
        kv_end = cu_seqlens_k[sequence_ids + 1]
        queries_remaining = cu_seqlens_q[sequence_ids + 1] - query_rows
        last_visible = kv_end - queries_remaining

        indices = sparse_indices.squeeze(1)
        kv_rows = kv_start[:, None] + indices
        visible = (indices >= 0) & (kv_rows <= last_visible[:, None])
        kv_rows = kv_rows.masked_fill(~visible, 0).long()

        compute_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=query.device.type, enabled=False):
            selected_keys = key.squeeze(1).to(compute_dtype)[kv_rows]
            selected_extra = key_extra.squeeze(1).to(compute_dtype)[kv_rows]
            scores = torch.bmm(query.to(compute_dtype), selected_keys.transpose(1, 2))
            scores = scores + torch.bmm(
                query_extra.to(compute_dtype), selected_extra.transpose(1, 2)
            )
            scores = (scores * softmax_scale).masked_fill(~visible[:, None, :], -torch.inf)
            softmax_max = scores.detach().amax(dim=-1, keepdim=True)
            weights = (scores - softmax_max).exp()
            softmax_sum = weights.sum(dim=-1, keepdim=True)
            selected_values = value.squeeze(1).to(compute_dtype)[kv_rows]
            output = torch.bmm(weights / softmax_sum, selected_values)

        return (
            output.to(value.dtype),
            softmax_max.squeeze(-1).unsqueeze(0).float(),
            softmax_sum.detach().squeeze(-1).unsqueeze(0).float(),
        )
