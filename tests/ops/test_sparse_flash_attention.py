from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from torchturbo.ops import OpDispatcher
from torchturbo.ops.registry import registry
from torchturbo.runtime import TurboDevice
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture
def packed_inputs():
    generator = torch.Generator().manual_seed(42)
    tensors = [
        torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
        for shape in [(3, 96, 4), (6, 1, 4), (6, 1, 4), (3, 96, 2), (6, 1, 2)]
    ]
    query, key, value, query_extra, key_extra = tensors
    indices = torch.tensor([[[2, 1, -1]], [[2, 0, 1]], [[1, 0, -1]]], dtype=torch.int32)
    kwargs = dict(
        query_extra=query_extra,
        key_extra=key_extra,
        cu_seqlens_q=torch.tensor([0, 2, 3]),
        cu_seqlens_k=torch.tensor([0, 3, 6]),
    )
    return tensors, (query, key, value, indices, 0.5), kwargs


def test_native_sparse_flash_attention_matches_selected_token_reference(packed_inputs):
    inputs, args, kwargs = packed_inputs
    query, key, value, query_extra, key_extra = inputs
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    output, softmax_max, softmax_sum = dispatcher.sparse_flash_attention(*args, **kwargs)

    # The first query is right-aligned at local position 1 and cannot see key 2.
    visible_keys = [[1], [2, 0, 1], [4, 3]]
    expected_output, expected_max, expected_sum = [], [], []
    for row, keys in enumerate(visible_keys):
        scores = query[row] @ key[keys, 0].T + query_extra[row] @ key_extra[keys, 0].T
        scores = scores * 0.5
        maxima = scores.amax(dim=-1)
        expected_output.append(scores.softmax(dim=-1) @ value[keys, 0])
        expected_max.append(maxima)
        expected_sum.append((scores - maxima[:, None]).exp().sum(dim=-1))

    expected = torch.stack(expected_output)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(softmax_max, torch.stack(expected_max)[None].float())
    torch.testing.assert_close(softmax_sum, torch.stack(expected_sum)[None].float())
    assert not softmax_max.requires_grad
    assert not softmax_sum.requires_grad

    upstream = torch.linspace(-1, 1, output.numel(), dtype=output.dtype).reshape_as(output)
    grads = torch.autograd.grad(output, inputs, upstream)
    expected_grads = torch.autograd.grad(expected, inputs, upstream)
    torch.testing.assert_close(grads, expected_grads)


def test_npu_sparse_flash_attention_pads_only_query_heads(packed_inputs, monkeypatch):
    inputs, args, kwargs = packed_inputs
    cpu = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    expected = cpu.sparse_flash_attention(*args, **kwargs)
    padded_output_grads = []

    def fake_sparse_flash_attention(query, key, value, indices, scale, **options):
        assert query.shape == (3, 128, 4)
        assert options["query_rope"].shape == (3, 128, 2)
        assert key is args[1] and value is args[2] and indices is args[3]
        assert options["key_rope"] is kwargs["key_extra"]
        assert options["actual_seq_lengths_query"].tolist() == [2, 3]
        assert options["actual_seq_lengths_kv"].tolist() == [3, 6]
        assert options["actual_seq_lengths_query"].dtype == torch.int32
        assert options["actual_seq_lengths_kv"].dtype == torch.int32
        assert options["layout_query"] == options["layout_kv"] == "TND"
        assert options["sparse_mode"] == 3 and options["attention_mode"] == 2
        assert options["sparse_block_size"] == 1 and options["return_softmax_lse"]
        assert torch.count_nonzero(query[:, 96:]).item() == 0
        assert torch.count_nonzero(options["query_rope"][:, 96:]).item() == 0

        result = cpu.sparse_flash_attention(
            query, key, value, indices, scale, **(kwargs | {"query_extra": options["query_rope"]})
        )
        result[0].register_hook(padded_output_grads.append)
        return result

    torch_npu = ModuleType("torch_npu")
    torch_npu.npu_sparse_flash_attention = fake_sparse_flash_attention
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    npu = OpDispatcher(device_runtime=SimpleNamespace(turbo_device=TurboDevice.NPU), registry=registry)
    actual = npu.sparse_flash_attention(*args, **kwargs, impl="npu_sparse_flash_attention")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual[0].sum(), inputs),
        torch.autograd.grad(expected[0].sum(), inputs),
    )
    assert len(padded_output_grads) == 1
    assert torch.count_nonzero(padded_output_grads[0][:, 96:]).item() == 0
