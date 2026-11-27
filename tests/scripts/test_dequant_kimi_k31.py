from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def test_text_export_decodes_mxfp4_and_preserves_kda_layout(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "dequant" / "kimik31.py"
    source, output = tmp_path / "original", tmp_path / "text_bf16"
    source.mkdir()
    weight = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"
    decay = "language_model.model.layers.0.self_attn.A_log"
    tensors = {
        f"{weight}_packed": torch.tensor([[0x91] * 32], dtype=torch.uint8),
        f"{weight}_scale": torch.tensor([[127, 129]], dtype=torch.uint8),
        decay: torch.zeros(128),
        "vision_tower.encoder.final_layernorm.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    save_file(tensors, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": dict.fromkeys(tensors, "model.safetensors"),
    }))
    (source / "config.json").write_text(json.dumps({
        "model_type": "kimi_k31",
        "text_config": {
            "linear_attn_config": {"num_heads": 96},
            "quantization_config": {"format": "mxfp4-pack-quantized"},
        },
    }))

    subprocess.run([
        sys.executable, str(script), "--input-hf-path", str(source),
        "--output-hf-path", str(output), "--text-only",
    ], check=True)

    index = json.loads((output / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    output_weight = weight.removeprefix("language_model.")
    output_decay = decay.removeprefix("language_model.")
    assert set(weight_map) == {output_weight, output_decay}
    expected = torch.tensor([[0.5, -0.5] * 16 + [2.0, -2.0] * 16], dtype=torch.bfloat16)
    torch.testing.assert_close(load_file(output / weight_map[output_weight])[output_weight], expected)
    torch.testing.assert_close(load_file(output / weight_map[output_decay])[output_decay], tensors[decay])

    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "kimi_k31_text"
    assert config["linear_attn_config"]["checkpoint_num_heads"] == 128
    assert "quantization_config" not in config
