from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="lightning_indexer",
    impl_name="torch_lightning_indexer",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchLightningIndexer(OpImpl):
    """Reference for uncompressed, bottom-right causal TND index selection."""

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
        if layout_q != "TND" or layout_k != "TND" or mask_mode != 3 or cmp_ratio != 1:
            raise ValueError("torch_lightning_indexer requires uncompressed causal TND inputs")

        indices, values = [], []
        q_spans = pairwise(cu_seqlens_q.tolist())
        k_spans = pairwise(cu_seqlens_k.tolist())
        with torch.autocast(device_type=query.device.type, enabled=False):
            for (q_start, q_end), (k_start, k_end) in zip(q_spans, k_spans, strict=True):
                key_length = k_end - k_start
                selected_count = min(topk, key_length)
                key_t = key[k_start:k_end, 0].float().T
                key_positions = torch.arange(key_length, device=query.device)
                # Bound the temporary per-head score matrix for the reference path.
                chunk_size = max(1, 2**20 // (key_length * query.size(1)))
                for start in range(q_start, q_end, chunk_size):
                    stop = min(start + chunk_size, q_end)
                    per_head = query[start:stop].float() @ key_t
                    scores = (per_head.relu_() * weights[start:stop, :, None].float()).sum(dim=1)
                    last_visible = key_length - q_end + torch.arange(start, stop, device=query.device)
                    scores.masked_fill_(key_positions > last_visible[:, None], -torch.inf)
                    scores, rows = scores.topk(selected_count, dim=-1)
                    rows = rows.to(torch.int32).masked_fill_(scores.isneginf(), -1)
                    indices.append(F.pad(rows, (0, topk - selected_count), value=-1))
                    if return_value:
                        values.append(F.pad(scores, (0, topk - selected_count), value=-torch.inf))

        scores = torch.cat(values).unsqueeze(1) if return_value else query.new_empty(0, dtype=torch.float32)
        return torch.cat(indices).unsqueeze(1), scores
