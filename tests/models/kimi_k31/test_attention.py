from __future__ import annotations

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from transformers import PretrainedConfig

from torchturbo.models.kimi_k31.attention import KimiK31DSAAttention, KimiK31DSALayout
from torchturbo.ops import get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


def _make_attention(train_indexer=False):
    config = PretrainedConfig(
        hidden_size=16, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=7, kv_lora_rank=5, qk_nope_head_dim=4, qk_rope_head_dim=2,
        v_head_dim=3, rms_norm_eps=1e-5, mla_use_nope=True, mla_use_output_gate=True,
        dsa_config={
            "index_head_dim": 4, "index_num_heads": 2, "index_topk": 3,
            "index_k_norm_type": "layernorm", "index_rope_dim": 0,
        },
    )
    return KimiK31DSAAttention(config, train_indexer=train_indexer)


def _dense_reference(layer, hidden_states, lengths):
    # Expand K and V explicitly, with each sample processed independently.
    outputs, losses = [], []
    for x in hidden_states[0, :sum(lengths)].split(lengths):
        latent_q = layer.q_a_layernorm(layer.q_a_proj(x))
        query = layer.q_b_proj(latent_q).reshape(len(x), 4, 6).transpose(0, 1)
        latent_kv, key_extra = layer.kv_a_proj_with_mqa(x).split((5, 2), dim=-1)
        kv = layer.kv_b_proj(layer.kv_a_layernorm(latent_kv)).reshape(len(x), 4, 7)
        key_main, value = kv.transpose(0, 1).split((4, 3), dim=-1)
        key = torch.cat((key_main, key_extra.unsqueeze(0).expand(4, -1, -1)), dim=-1)
        scores = query @ key.transpose(1, 2) * 6**-0.5

        indexer = layer.indexer
        index_query = indexer.wq_b(latent_q.detach()).reshape(len(x), 2, 4)
        index_key = indexer.k_norm(indexer.wk(x.detach()))
        index_weights = indexer.weights_proj(x.detach()) * 8**-0.5
        logits = ((index_query @ index_key.T).relu() * index_weights[..., None]).sum(dim=1)
        causal = torch.ones_like(logits, dtype=torch.bool).tril()
        indices = logits.detach().masked_fill(~causal, -torch.inf).topk(min(3, len(x))).indices
        selected = torch.zeros_like(causal).scatter_(1, indices, True) & causal
        probabilities = scores.masked_fill(~selected, -torch.inf).softmax(dim=-1)
        target = probabilities.detach().mean(dim=0)
        log_probs = logits.masked_fill(~selected, -torch.inf).log_softmax(dim=-1)
        losses.append((target.xlogy(target) - target * log_probs.masked_fill(~selected, 0)).sum())
        output = (probabilities @ value).transpose(0, 1).reshape(len(x), 12)
        outputs.append(layer.o_proj(output * layer.g_proj(x).sigmoid()))
    return torch.cat(outputs), sum(losses)


@pytest.mark.parametrize("train_indexer", [False, True])
def test_gated_dsa_matches_unabsorbed_attention(train_indexer):
    get_or_create_ops(device_runtime=CpuRuntime())
    torch.manual_seed(7)
    layer = _make_attention(train_indexer)
    hidden_states = torch.randn(1, 9, 16, requires_grad=True)
    cu_seqlens = torch.tensor([0, 5, 7], dtype=torch.int32)
    layout = KimiK31DSALayout.build(1, 9, hidden_states.device, cu_seqlens=cu_seqlens)
    actual, loss = layer(hidden_states, layout)
    expected, expected_loss = _dense_reference(layer, hidden_states, [5, 2])
    torch.testing.assert_close(actual[0, :7], expected)
    assert actual[0, 7:].count_nonzero() == 0

    main_inputs = (hidden_states, *(p for name, p in layer.named_parameters() if not name.startswith("indexer.")))
    grad_output = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, main_inputs, grad_output)
    expected_grads = torch.autograd.grad(expected, main_inputs, grad_output[0, :7])
    torch.testing.assert_close(actual_grads, expected_grads)
    if train_indexer:
        torch.testing.assert_close(loss, expected_loss)
        indexer_parameters = tuple(layer.indexer.parameters())
        actual_loss_grads = torch.autograd.grad(loss, (*indexer_parameters, *main_inputs), allow_unused=True)
        assert all(grad is None for grad in actual_loss_grads[len(indexer_parameters):])
        expected_loss_grads = torch.autograd.grad(expected_loss, indexer_parameters)
        torch.testing.assert_close(actual_loss_grads[:len(indexer_parameters)], expected_loss_grads)
    else:
        assert loss is None
        assert not any(parameter.requires_grad for parameter in layer.indexer.parameters())


def _sp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    get_or_create_ops(device_runtime=CpuRuntime())
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        layer = _make_attention(train_indexer=True)
        hidden_states = torch.randn(1, 8, 16)
        for offsets in ([0, 3, 7], [0, 3]):
            cu_seqlens = torch.tensor(offsets, dtype=torch.int32)
            ref_layout = KimiK31DSALayout.build(1, 8, hidden_states.device, cu_seqlens=cu_seqlens)
            reference, reference_kl = layer(hidden_states, ref_layout)
            reference_grads = torch.autograd.grad(reference.square().sum() + reference_kl, tuple(layer.parameters()))

            layout = KimiK31DSALayout.build(
                1, 4, hidden_states.device, cu_seqlens=cu_seqlens, sp_rank=rank,
            )
            local_slice = slice(rank * 4, (rank + 1) * 4)
            output, kl = layer(hidden_states[:, local_slice], layout, sp_group=dist.group.WORLD)
            torch.testing.assert_close(output, reference[:, local_slice])
            grads = torch.autograd.grad(output.square().sum() + kl, tuple(layer.parameters()))
            for grad in grads:
                dist.all_reduce(grad)
            torch.testing.assert_close(grads, reference_grads, atol=1e-5, rtol=1e-5)
            total_kl = kl.detach().clone()
            dist.all_reduce(total_kl)
            torch.testing.assert_close(total_kl, reference_kl)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required for CPU SP")
def test_packed_dsa_sp_matches_single_rank(tmp_path):
    mp.spawn(_sp_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
