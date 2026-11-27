from __future__ import annotations

import pytest
import torch

from torchturbo.ops.providers.cuda.lightning_indexer import CudnnLightningIndexer
from torchturbo.ops.providers.cuda.sparse_flash_attention import FlashMlaCudnnSparseAttention
from torchturbo.ops.providers.native.lightning_indexer import TorchLightningIndexer
from torchturbo.ops.providers.native.sparse_flash_attention import TorchSparseFlashAttention


def test_kimi_cudnn_indexer_packed_suffix():
    provider = CudnnLightningIndexer()
    if not provider.available():
        pytest.skip("requires cuDNN Frontend DSA and SM100")
    torch.manual_seed(31)
    query = torch.randn(7, 32, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(18, 1, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 4, 7], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 11, 18], dtype=torch.int32, device="cuda")
    kwargs = {"cu_seqlens_q": cu_q, "cu_seqlens_k": cu_k, "return_value": 1}
    indices, scores = provider(query, key, weights, 4, **kwargs)
    ref_indices, ref_scores = TorchLightningIndexer()(query, key, weights, 4, **kwargs)
    torch.testing.assert_close(indices.sort(dim=-1).values, ref_indices.sort(dim=-1).values)
    torch.testing.assert_close(scores, ref_scores, atol=0.2, rtol=0.01)


def test_kimi_flash_mla_forward_and_cudnn_backward_96_heads():
    provider = FlashMlaCudnnSparseAttention()
    if not provider.available():
        pytest.skip("requires FlashMLA, cuDNN Frontend DSA, and SM100")
    torch.manual_seed(31)
    options = {"device": "cuda", "dtype": torch.bfloat16, "requires_grad": True}
    query = torch.randn(4, 96, 512, **options)
    key = torch.randn(10, 1, 512, **options)
    q_extra = torch.randn(4, 96, 64, **options)
    k_extra = torch.randn(10, 1, 64, **options)
    indices = torch.tensor(
        [[[0, 3, 1]], [[4, 0, 2]], [[1, 0, 2]], [[2, 4, -1]]],
        dtype=torch.int32, device="cuda",
    )
    cu_q = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 5, 10], dtype=torch.int32, device="cuda")
    kwargs = {
        "query_extra": q_extra, "key_extra": k_extra, "cu_seqlens_q": cu_q, "cu_seqlens_k": cu_k,
    }
    inputs = (query, key, q_extra, k_extra)
    ref = TorchSparseFlashAttention()(query, key, key, indices, 192**-0.5, **kwargs)
    result = provider(query, key, key, indices, 192**-0.5, **kwargs)
    torch.testing.assert_close(result, ref, atol=0.02, rtol=0.03)
    grad = torch.randn_like(result[0])
    grads = torch.autograd.grad(result[0], inputs, grad)
    ref_grads = torch.autograd.grad(ref[0], inputs, grad)
    torch.testing.assert_close(grads, ref_grads, atol=0.1, rtol=0.05)
