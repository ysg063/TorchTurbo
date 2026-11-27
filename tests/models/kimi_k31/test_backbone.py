from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from datetime import timedelta
from importlib import import_module
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM

from torchturbo.engine.backends.fsdp2.activation_checkpointing import restore_moe_runtime_inputs
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import EngineConfig
from torchturbo.models.kimi_k31.attention import KimiK31KDA
from torchturbo.models.kimi_k31.configuration_kimi_k31 import KimiK31Config, KimiK31TextConfig
from torchturbo.models.kimi_k31.model import (
    KimiK31WeightConverter, TurboKimiK31ForCausalLM, TurboKimiK31ForConditionalGeneration,
)
from torchturbo.models.kimi_k31.modeling_kimi_k31 import KimiK31MoE, situ_glu
from torchturbo.models.kimi_k31.parallel_plan import _place_moe
from torchturbo.models.modules.moe.moe_layer import MoERuntimeInputs
from torchturbo.ops import OpDispatcher, get_or_create_ops, registry
from torchturbo.ops.providers.native.lightning_indexer import TorchLightningIndexer
from torchturbo.runtime import TurboDevice
from torchturbo.runtime.cpu import CpuRuntime


def tiny_config():
    return KimiK31TextConfig(
        vocab_size=32, pad_token_id=0, tie_word_embeddings=False, num_hidden_layers=4,
        hidden_size=16, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=7, kv_lora_rank=5, qk_nope_head_dim=4, qk_rope_head_dim=2, v_head_dim=3,
        rms_norm_eps=1e-5, mla_use_nope=True, mla_use_output_gate=True,
        linear_attn_config={
            "kda_layers": [1, 3], "num_heads": 4, "head_dim": 4,
            "short_conv_kernel_size": 3, "use_full_rank_gate": True, "gate_lower_bound": -5.0,
        },
        dsa_config={
            "dsa_layers": [2, 4], "index_head_dim": 4, "index_num_heads": 2,
            "index_topk": 3, "index_k_norm_type": "layernorm", "index_rope_dim": 0,
        },
        attn_res_block_size=2, first_k_dense_replace=1, moe_layer_freq=1,
        intermediate_size=24, moe_intermediate_size=8, num_experts=4,
        num_experts_per_token=2, num_shared_experts=1, num_expert_group=1,
        routed_expert_hidden_size=8, latent_moe_use_norm=True, moe_renormalize=True,
        moe_router_activation_func="sigmoid", routed_scaling_factor=1.0,
        initializer_range=0.02, activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
    )


def _recurrent_kda(q, k, v, g, beta, **kwargs):
    assert kwargs["safe_gate"] and kwargs["use_beta_sigmoid_in_kernel"]
    assert kwargs["use_gate_in_kernel"] and kwargs["use_qk_l2norm_in_kernel"]
    num_heads, dim = g.shape[-2:]
    q = F.normalize(q.unflatten(-1, (num_heads, dim)), dim=-1) * dim**-0.5
    k = F.normalize(k.unflatten(-1, (num_heads, dim)), dim=-1)
    v = v.unflatten(-1, (num_heads, dim))
    bias = kwargs["dt_bias"].view(num_heads, dim)
    log_decay = kwargs["lower_bound"] * ((g + bias) * kwargs["A_log"].exp()[:, None]).sigmoid()
    cu_seqlens = kwargs["cu_seqlens"]
    boundaries = [0, q.size(1)] if cu_seqlens is None else cu_seqlens.tolist()
    rows = []
    for start, end in pairwise(boundaries):
        state = q.new_zeros(q.size(0), num_heads, dim, dim)
        for pos in range(start, end):
            key = k[:, pos]
            state = state * log_decay[:, pos, :, :, None].exp()
            delta = v[:, pos] - torch.einsum("bhd,bhde->bhe", key, state)
            delta = delta * beta[:, pos, :, None].sigmoid()
            state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
            rows.append(torch.einsum("bhd,bhde->bhe", q[:, pos], state))
    return torch.stack(rows, dim=1), None


def _causal_conv(x, weight, *, activation, packed_params):
    assert activation == "silu"
    boundaries = [0, x.size(1)] if packed_params is None else packed_params.cu_seqlens_list
    chunks = []
    for start, end in pairwise(boundaries):
        seq = x[:, start:end].transpose(1, 2)
        conv = F.conv1d(seq, weight.unsqueeze(1), padding=weight.size(1) - 1, groups=seq.size(1))
        chunks.append(F.silu(conv[..., :end - start]).transpose(1, 2))
    return torch.cat(chunks, dim=1), None


def test_kda_short_conv_uses_projection_dtype_and_fp32_parameter_grads(monkeypatch):
    dispatcher = get_or_create_ops(device_runtime=CpuRuntime())

    def short_conv(x, weight, **kwargs):
        assert x.dtype == weight.dtype == torch.bfloat16
        output, state = _causal_conv(x.float(), weight.float(), **kwargs)
        return output.to(x.dtype), state

    def recurrent_kda(q, k, v, g, beta, **kwargs):
        output, state = _recurrent_kda(q.float(), k.float(), v.float(), g.float(), beta, **kwargs)
        return output.to(q.dtype), state

    monkeypatch.setattr(dispatcher, "causal_conv1d", short_conv)
    monkeypatch.setattr(dispatcher, "kda", recurrent_kda)
    layer = KimiK31KDA(tiny_config())
    # Mirror FSDP2's forward dtypes: BF16 projections and replicated FP32 weights.
    for module in layer.modules():
        if isinstance(module, torch.nn.Linear):
            module.bfloat16()

    hidden = torch.randn(1, 7, 16, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)
    layer(hidden, cu_seqlens).float().square().mean().backward()

    for conv in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d):
        assert conv.weight.dtype == conv.weight.grad.dtype == torch.float32


@pytest.mark.parametrize("native_override", [False, True])
def test_kimi_reuses_public_indexer_metadata_within_a_microbatch(monkeypatch, native_override):
    calls = []
    native_indexer = TorchLightningIndexer()

    def make_metadata(**kwargs):
        assert kwargs["num_heads_q"] == 2 and kwargs["num_heads_k"] == 1
        assert kwargs["head_dim"] == 4 and kwargs["topk"] == 3
        assert kwargs["batch_size"] == 2
        assert kwargs["cu_seqlens_q"].tolist() == [0, 4, 7]
        assert kwargs["cu_seqlens_k"].tolist() == [0, 4, 7]
        metadata = torch.empty(1024, dtype=torch.int32)
        calls.append(("prepare", metadata))
        return metadata

    def run_indexer(query, key, weights, topk, *, metadata, cu_seqlens_q, cu_seqlens_k, **kwargs):
        calls.append(("select", metadata))
        return native_indexer(query, key, weights, topk, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)

    cann_ops = SimpleNamespace(lightning_indexer_metadata=make_metadata, lightning_indexer=run_indexer)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", SimpleNamespace(ops=cann_ops))
    overrides = {"lightning_indexer": "torch_lightning_indexer"} if native_override else {}
    dispatcher = OpDispatcher(
        device_runtime=SimpleNamespace(turbo_device=TurboDevice.NPU), registry=registry, overrides=overrides,
    )
    monkeypatch.setattr(import_module("torchturbo.ops.ops"), "_OPS", dispatcher)
    monkeypatch.setattr(dispatcher, "kda", _recurrent_kda)
    monkeypatch.setattr(dispatcher, "causal_conv1d", _causal_conv)

    model = TurboKimiK31ForCausalLM(tiny_config())
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]])
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)
    for _ in range(2):
        model(ids, cu_seq_lens_q=cu_seqlens)

    if native_override:
        assert calls == []
    else:
        assert [name for name, _ in calls] == ["prepare", "select", "select"] * 2
        for start in (0, 3):
            assert calls[start][1] is calls[start + 1][1] is calls[start + 2][1]
        assert calls[0][1] is not calls[3][1]


def test_hybrid_backbone_packing_backward_and_recompute(monkeypatch):
    dispatcher = get_or_create_ops(device_runtime=CpuRuntime())
    monkeypatch.setattr(dispatcher, "kda", _recurrent_kda)
    monkeypatch.setattr(dispatcher, "causal_conv1d", _causal_conv)
    torch.manual_seed(41)
    config = tiny_config()
    model = AutoModelForCausalLM.from_config(config, train_indexer=True)
    assert isinstance(model, TurboKimiK31ForCausalLM)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]])
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)
    outputs = model(ids, cu_seq_lens_q=cu_seqlens, labels=ids)
    split_outputs = torch.cat((model(ids[:, :4]).logits, model(ids[:, 4:7]).logits), dim=1)
    torch.testing.assert_close(outputs.logits[:, :7], split_outputs, atol=2e-6, rtol=1e-5)
    assert outputs.indexer_loss_sum.isfinite()

    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    loss = outputs.loss + outputs.indexer_loss_sum / (7 * 2)
    expected = torch.autograd.grad(loss, parameters, allow_unused=True)
    for layer in model.model.layers:
        original = layer.forward

        def wrapped(*args, forward=original):
            return checkpoint(forward, *args, use_reentrant=False)

        monkeypatch.setattr(layer, "forward", wrapped)
    replay = model(ids, cu_seq_lens_q=cu_seqlens, labels=ids)
    actual = torch.autograd.grad(
        replay.loss + replay.indexer_loss_sum / (7 * 2), parameters, allow_unused=True,
    )
    torch.testing.assert_close(actual, expected)


def test_latent_moe_matches_explicit_experts_and_routes_at_model_width():
    get_or_create_ops(device_runtime=CpuRuntime())
    torch.manual_seed(17)
    layer = KimiK31MoE(tiny_config())
    hidden = torch.randn(1, 5, 16)
    output, logits = layer(hidden)

    scores = logits.sigmoid()
    indices = (scores + layer.router.expert_bias).topk(2, dim=-1).indices
    weights = scores.gather(-1, indices)
    weights = weights / weights.sum(-1, keepdim=True)
    latent = layer.routed_expert_down_proj(hidden).squeeze(0)
    w1 = layer.experts.weight1.unflatten(0, (4, 8))
    w2 = layer.experts.weight2.unflatten(0, (4, 8))
    values = []
    for row in range(latent.size(0)):
        experts = indices[row]
        gate, up = (latent[row] @ w1[experts]).chunk(2, dim=-1)
        features = situ_glu(gate, up, 4.0, 25.0) * weights[row, :, None]
        values.append((features.unsqueeze(1) @ w2[experts]).sum(dim=0).squeeze(0))
    summed = torch.stack(values).unsqueeze(0)
    expected = layer.routed_expert_up_proj(layer.routed_expert_norm(summed)) + layer.shared_experts(hidden)
    torch.testing.assert_close(output, expected)


def test_indexer_loss_uses_step_query_count_instead_of_label_count(monkeypatch):
    dispatcher = get_or_create_ops(device_runtime=CpuRuntime())
    monkeypatch.setattr(dispatcher, "kda", _recurrent_kda)
    monkeypatch.setattr(dispatcher, "causal_conv1d", _causal_conv)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    config = tiny_config()
    config.dsa_indexer_loss_coeff = 0.2
    model = TurboKimiK31ForCausalLM(config)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]])
    cu = torch.tensor([0, 4, 7], dtype=torch.int32)
    output = model(ids, cu_seq_lens_q=cu)
    assert output.indexer_num_queries == 7 and output.indexer_num_layers == 2
    assert model.uses_indexer_loss
    loss = FSDP2Backend._normalize_indexer_loss(
        output, sp_size=1, sp_group=None, num_items_in_batch=4,
        step_num_queries=torch.tensor(11), accumulation_steps=2,
    )
    torch.testing.assert_close(loss, output.indexer_loss_sum / (11 * 2))
    loss.backward()
    assert model.model.layers[1].self_attn.indexer.wq_b.weight.grad is not None
    assert model.model.layers[1].self_attn.q_b_proj.weight.grad is None
    assert not TurboKimiK31ForCausalLM(tiny_config()).model.layers[1].self_attn.indexer.wq_b.weight.requires_grad


def test_packed_image_forward_matches_separate_images(tmp_path, monkeypatch):
    dispatcher = get_or_create_ops(device_runtime=CpuRuntime())
    monkeypatch.setattr(dispatcher, "kda", _recurrent_kda)
    monkeypatch.setattr(dispatcher, "causal_conv1d", _causal_conv)
    config = KimiK31Config(
        text_config=tiny_config(),
        vision_config={
            "init_pos_emb_height": 2, "init_pos_emb_width": 2, "pos_emb_interpolation_mode": "bilinear",
            "vt_hidden_size": 8, "mm_hidden_size": 8, "vt_num_hidden_layers": 2,
            "vt_num_attention_heads": 2, "qkv_hidden_size": 8, "vt_intermediate_size": 16,
            "linear_bias": False, "attn_bias": False, "patch_embed_proj_bias": False,
            "patch_size": 2, "merge_kernel_size": [2, 2], "projector_ln_eps": 1e-5,
        },
        media_placeholder_token_id=31,
    )
    config.save_pretrained(tmp_path)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(tmp_path))
    assert isinstance(model, TurboKimiK31ForConditionalGeneration)
    ids = torch.tensor([[31, 1, 2, 31, 31, 3, 4]])
    grid = torch.tensor([[1, 2, 2], [1, 2, 4]])
    pixels = torch.randn(12, 3, 2, 2)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    logits = model(ids, pixel_values=pixels, grid_thws=grid, cu_seq_lens_q=cu).logits
    first = model(ids[:, :3], pixel_values=pixels[:4], grid_thws=grid[:1]).logits
    second = model(ids[:, 3:], pixel_values=pixels[4:], grid_thws=grid[1:]).logits
    torch.testing.assert_close(logits, torch.cat((first, second), dim=1), atol=2e-6, rtol=1e-5)
    logits.square().mean().backward()
    assert model.vision_tower.patch_embed.proj.weight.grad.norm() > 0
    assert model.mm_projector.proj[0].weight.grad.norm() > 0
    spec = model.get_sequence_parallel_input_slicing_spec()
    assert spec.module is model.language_model.model
    assert spec.sliced_inputs == (("inputs_embeds", 1),)


def test_quantile_balancing_updates_once_per_adamw_step():
    get_or_create_ops(device_runtime=CpuRuntime())
    torch.manual_seed(29)
    model = TurboKimiK31ForCausalLM(tiny_config())
    moe = model.model.layers[1].mlp
    assert isinstance(moe, KimiK31MoE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.on_optimizer_created(optimizer)

    hidden = torch.randn(1, 4, 16, requires_grad=True)
    padding = torch.tensor([[False, False, True, True]])
    replay = ((moe, MoERuntimeInputs(accumulate_tokens_per_expert=False)),)
    losses = []
    for _ in range(2):
        output = checkpoint(
            lambda x: moe(x, padding_mask=padding)[0], hidden,
            use_reentrant=False,
            context_fn=lambda: (nullcontext(), restore_moe_runtime_inputs(replay)),
        )
        losses.append(output.square().sum())
    sum(losses).backward()

    histogram = moe.router._histogram
    torch.testing.assert_close(histogram.sum(dim=-1), torch.full((4,), 4, dtype=torch.long))
    assert not moe.router.expert_bias.any()
    optimizer.step()
    assert not histogram.any()
    assert moe.router.expert_bias.abs().max() > 0
    torch.testing.assert_close(moe.router.expert_bias.mean(), torch.zeros(()), atol=1e-7, rtol=0)
    updated = moe.router.expert_bias.clone()
    optimizer.step()
    torch.testing.assert_close(moe.router.expert_bias, updated)


def test_latent_expert_bf16_mapping_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    model = TurboKimiK31ForCausalLM(tiny_config())
    save_file({"unused": torch.ones(1)}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"unused": "model.safetensors"}})
    )
    converter = KimiK31WeightConverter(
        model, {"ep": {"ep_group": None, "ep_rank": 1, "ep_size": 2}}, str(tmp_path),
    )
    assert model.config.hidden_size == 16
    assert converter.get_gate_weight_shape() == (8, 8)
    prefix = "model.layers.1.mlp.experts."
    generator = torch.Generator().manual_seed(8)
    for name, shape, export, assemble in (
        ("weight1", (16, 16), converter.get_local_ep_w1_hf_state_dict, converter.assemble_weight1_for_broadcast),
        ("weight2", (16, 8), converter.get_local_ep_w2_hf_state_dict, converter.assemble_weight2_for_broadcast),
    ):
        weight = torch.randn(shape, generator=generator, dtype=torch.bfloat16)
        tensors = export(prefix + name, weight)
        assert all(key.startswith((prefix + "2.", prefix + "3.")) for key in tensors)
        restored = torch.empty_like(weight)
        assemble(restored, tensors, prefix + name)
        torch.testing.assert_close(restored, weight, rtol=0, atol=0)


def _joint_parallel_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dispatcher = get_or_create_ops(device_runtime=CpuRuntime())
    dispatcher.kda = _recurrent_kda
    dispatcher.causal_conv1d = _causal_conv
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        group = dist.group.WORLD
        torch.manual_seed(21)
        config = tiny_config()
        reference = TurboKimiK31ForCausalLM(config, train_indexer=True)
        sharded = TurboKimiK31ForCausalLM(config, train_indexer=True)
        for layer in sharded.model.layers:
            layer.mlp = _place_moe(layer.mlp, config, {"ep_group": group}, {"sp_group": group}, EngineConfig())
        sharded.on_parallelized({"sp": {"sp_group": group}})
        state = {
            name: param.chunk(2, dim=0)[rank] if ".mlp.experts.weight" in name else param
            for name, param in reference.state_dict().items()
        }
        sharded.load_state_dict(state)

        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]])
        cu_seqlens = torch.tensor([0, 5, 7], dtype=torch.int32)
        expected = reference(ids, cu_seq_lens_q=cu_seqlens)
        local = slice(4 * rank, 4 * (rank + 1))
        actual = sharded(ids[:, local], cu_seq_lens_q=cu_seqlens)
        num_valid = min(4, 7 - 4 * rank)
        torch.testing.assert_close(actual.logits[:, :num_valid], expected.logits[:, local][:, :num_valid])
        # The final padded token has no loss or auxiliary contribution.
        (expected.logits[:, :7].square().sum() + expected.indexer_loss_sum).backward()
        (actual.logits[:, :num_valid].square().sum() + actual.indexer_loss_sum).backward()
        ref_parameters = dict(reference.named_parameters())
        for name, parameter in sharded.named_parameters():
            expected_grad = ref_parameters[name].grad
            if expected_grad is None:
                assert parameter.grad is None
            elif ".mlp.experts.weight" in name:
                torch.testing.assert_close(parameter.grad, expected_grad.chunk(2, dim=0)[rank], atol=2e-5, rtol=1e-4)
            else:
                dist.all_reduce(parameter.grad)
                torch.testing.assert_close(parameter.grad, expected_grad, atol=2e-5, rtol=1e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required for CPU SP/EP")
def test_hybrid_backbone_joint_sp_ep_matches_single_rank(tmp_path):
    mp.spawn(_joint_parallel_worker, args=(str(tmp_path / "init"),), nprocs=2, join=True)
