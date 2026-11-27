from __future__ import annotations

import pytest
import torch

from torchturbo.ops import OpDispatcher
from torchturbo.ops.registry import registry
from torchturbo.runtime.npu import NpuRuntime


def test_uncompressed_packed_indexer_selects_kimi_topk():
    torch_npu = pytest.importorskip("torch_npu")
    pytest.importorskip("cann_ops_transformer")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is required")

    # In the long row, keys in the prefix have lower scores than later keys.
    # Keep a separate short sequence to exercise index origins and -1 padding.
    query = torch.zeros(4, 32, 128, dtype=torch.bfloat16)
    query[..., 0] = 1
    key = torch.zeros(2055, 1, 128, dtype=torch.bfloat16)
    key[..., 0] = 1
    key[:3, 0, 0] = torch.tensor([0.125, 0.25, 0.5])
    weights = torch.full((4, 32), 1 / 32, dtype=torch.float32)
    dispatcher = OpDispatcher(device_runtime=NpuRuntime(), registry=registry)
    cu_seqlens_q = torch.tensor([0, 2, 4], dtype=torch.int32).npu()
    cu_seqlens_k = torch.tensor([0, 2051, 2055], dtype=torch.int32).npu()
    metadata = dispatcher.lightning_indexer_metadata(
        num_heads_q=32,
        num_heads_k=1,
        head_dim=128,
        topk=2048,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cmp_residual_k=None,
        batch_size=2,
        max_seqlen_q=-1,
        max_seqlen_k=-1,
    )
    indices, _ = dispatcher.lightning_indexer(
        query.npu(), key.npu(), weights.npu(), 2048,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        metadata=metadata,
        impl="npu_lightning_indexer",
    )
    indices = indices.cpu()[:, 0]
    expected = [range(2, 2050), range(3, 2051), range(3), range(4)]
    for row, keys in zip(indices, expected, strict=True):
        assert sorted(row[row >= 0].tolist()) == list(keys)
        assert torch.all(row[len(keys):] == -1)
