from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="lightning_indexer",
    impl_name="cudnn_lightning_indexer",
    kind=OpImplKind.CUSTOM,
    priority=-10,
    devices=(TurboDevice.CUDA,),
)
class CudnnLightningIndexer(OpImpl):
    """Use cuDNN's FP32 head-reduced score workspace for uncompressed DSA."""

    def available(self) -> bool:
        try:
            from cudnn import DSA
            forward = DSA.indexer_forward_wrapper
        except (AttributeError, ImportError, OSError, RuntimeError):
            return False
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10 and callable(forward)

    @torch.no_grad()
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        topk: int,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        layout_q: str = "TND",
        layout_k: str = "TND",
        mask_mode: int = 3,
        cmp_ratio: int = 1,
        return_value: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from cudnn import DSA

        if layout_q != "TND" or layout_k != "TND" or mask_mode != 3 or cmp_ratio != 1:
            raise ValueError("cuDNN lightning indexer requires uncompressed causal TND inputs")
        if query.shape[1:] not in ((32, 128), (64, 128)) or query.dtype != torch.bfloat16:
            raise ValueError("cuDNN indexer requires BF16 queries with 32 or 64 heads of size 128")

        indices, values = [], []
        weights = weights.to(torch.bfloat16)
        for (q_start, q_end), (k_start, k_end) in zip(
            pairwise(cu_seqlens_q.tolist()), pairwise(cu_seqlens_k.tolist()), strict=True,
        ):
            key_length = k_end - k_start
            count = min(topk, key_length)
            doc_keys = key[k_start:k_end].contiguous()
            cu_k = torch.tensor([0, key_length], dtype=torch.int32, device=query.device)
            key_rows = torch.arange(key_length, device=query.device)
            chunk_size = max(1, 2**24 // key_length)
            for start in range(q_start, q_end, chunk_size):
                stop = min(start + chunk_size, q_end)
                cu_q = torch.tensor([0, stop - start], dtype=torch.int32, device=query.device)
                offset = key_length - q_end + start
                causal_offset = torch.tensor([offset], dtype=torch.int32, device=query.device)
                scores = DSA.indexer_forward_wrapper(
                    query[start:stop].contiguous(), doc_keys, weights[start:stop].contiguous(),
                    ratio=1, sm_scale=1.0, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=stop - start, max_seqlen_k=key_length,
                    q_causal_offsets=causal_offset,
                )["scores"]
                visible = offset + torch.arange(stop - start, device=query.device)
                scores = scores[:, :key_length].masked_fill(key_rows > visible[:, None], -torch.inf)
                selected, rows = scores.topk(count, dim=-1)
                rows = rows.to(torch.int32).masked_fill_(selected.isneginf(), -1)
                indices.append(F.pad(rows, (0, topk - count), value=-1))
                if return_value:
                    values.append(F.pad(selected, (0, topk - count), value=-torch.inf))

        scores = torch.cat(values).unsqueeze(1) if return_value else query.new_empty(0, dtype=torch.float32)
        return torch.cat(indices).unsqueeze(1), scores
