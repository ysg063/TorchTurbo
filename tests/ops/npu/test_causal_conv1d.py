"""Public flat convolution matches the internal head-major training path."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("fla_npu")

from torchturbo.ops.packed_params import build_packed_params
from torchturbo.ops.providers.npu.causal_conv1d import AscendCCausalConv1d
from torchturbo.ops.providers.npu._fla_gdn import AscendCCausalConv1dFunction


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tokens", [75, 128])
def test_flat_conv_matches_internal_output_and_backward(packed, tokens):
    torch.manual_seed(23)
    x = torch.randn(1, tokens, 256, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(256, 4, device="npu", dtype=torch.bfloat16) * 0.1
    bias = torch.randn(256, device="npu", dtype=torch.bfloat16) * 0.1
    params = None
    if packed:
        params = build_packed_params(torch.tensor([0, 65, tokens], device="npu"))
    upstream = torch.randn_like(x)
    provider = AscendCCausalConv1d()

    def run(internal):
        inputs = [t.detach().clone().requires_grad_(True) for t in (x, weight, bias)]
        if internal:
            y, _ = AscendCCausalConv1dFunction.apply(
                inputs[0], inputs[1], 4, inputs[2], None, None, "silu",
                params.cu_seqlens if params is not None else None,
                False,
                params.cu_seqlens_list if params is not None else None,
            )
            assert y.shape == (1, 4, tokens, 64)
            y = y.transpose(1, 2).flatten(-2)
        else:
            y, _ = provider(*inputs, activation="silu", packed_params=params)
        assert y.shape == x.shape
        grads = torch.autograd.grad(y, inputs, upstream)
        return y.detach(), *grads

    reference = run(False)
    # A different head grouping must still preserve every channel's convolution.
    actual = run(True)
    for name, expected, result in zip(("output", "dx", "dw", "db"), reference, actual):
        error = (result.float() - expected.float()).norm() / expected.float().norm()
        assert error.item() < 0.02, f"{name}: relative L2 error {error.item()}"
