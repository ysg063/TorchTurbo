from __future__ import annotations

import pytest
import torch

from torchturbo.ops import OpDispatcher
from torchturbo.ops.registry import registry
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.npu import NpuRuntime


def test_sparse_flash_attention_kimi_shape_matches_reference():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is required")

    generator = torch.Generator().manual_seed(42)
    shapes = [(4, 96, 512), (2054, 1, 512), (4, 96, 64), (2054, 1, 64)]
    inputs = [torch.randn(shape, generator=generator, dtype=torch.bfloat16) for shape in shapes]
    indices = torch.full((4, 1, 2048), -1, dtype=torch.int32)
    indices[0, 0] = torch.arange(2048, dtype=torch.int32)
    indices[1, 0] = torch.arange(1, 2049, dtype=torch.int32)
    indices[2, 0, :4] = torch.arange(4, dtype=torch.int32)
    indices[3, 0, :5] = torch.arange(5, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 2, 4], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 2049, 2054], dtype=torch.int32)
    upstream = torch.randn(shapes[0], generator=generator, dtype=torch.bfloat16)

    def run(device, runtime, impl):
        local_inputs = [tensor.to(device).requires_grad_() for tensor in inputs]
        query, latent_kv, query_extra, key_extra = local_inputs
        dispatcher = OpDispatcher(device_runtime=runtime, registry=registry)
        result = dispatcher.sparse_flash_attention(
            query,
            latent_kv,
            latent_kv,
            indices.to(device),
            576**-0.5,
            query_extra=query_extra,
            key_extra=key_extra,
            cu_seqlens_q=cu_seqlens_q.to(device),
            cu_seqlens_k=cu_seqlens_k.to(device),
            impl=impl,
        )
        grads = torch.autograd.grad(result[0], local_inputs, upstream.to(device))
        return tuple(tensor.detach().cpu() for tensor in (*result, *grads))

    expected = run("cpu", CpuRuntime(), "torch_sparse_flash_attention")
    actual = run("npu:0", NpuRuntime(), "npu_sparse_flash_attention")
    torch.testing.assert_close(actual[:3], expected[:3], rtol=0.02, atol=0.02)
    torch.testing.assert_close(actual[3:], expected[3:], rtol=0.03, atol=0.03)
