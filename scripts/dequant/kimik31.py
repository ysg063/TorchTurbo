#!/usr/bin/env python3
"""Kimi K3.1 MXFP4 dequantization and offline CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

import dequantize
from dequantize import DequantizationFormat, Dequantizer, TensorSource


def _load_converter() -> type:
    # The shared converter imports this helper by its legacy package name.
    # Make the sibling module available under that name only during import.
    legacy_name = "torchturbo.scripts.dequant.dequantize"
    added_alias = legacy_name not in sys.modules
    if added_alias:
        sys.modules[legacy_name] = dequantize
    try:
        from converter import Converter
    finally:
        if added_alias:
            del sys.modules[legacy_name]
    return Converter


Converter = _load_converter()


class KimiK31Format(DequantizationFormat):
    """Decode compressed-tensors MXFP4 matrices with uint8 E8M0 scales."""

    GROUP_SIZE = 32

    def __init__(self, *, text_only: bool = False) -> None:
        self.text_only = text_only

    def resolve_sources(self, weight_map: Mapping[str, str]) -> dict[str, TensorSource]:
        keys = set(weight_map)
        consumed = set()
        sources = {}
        for packed_key in sorted(keys):
            if not packed_key.endswith(".weight_packed"):
                continue
            weight_key = packed_key.removesuffix("_packed")
            scale_key = f"{weight_key}_scale"
            if scale_key not in keys:
                raise ValueError(f"Missing MXFP4 scale tensor {scale_key!r}")
            if weight_key in keys:
                raise ValueError(f"Both packed and BF16 weights exist for {weight_key!r}")
            sources[weight_key] = TensorSource(weight_key, (packed_key, scale_key), quantized=True)
            consumed.update((packed_key, scale_key))
        for key in sorted(keys - consumed):
            sources[key] = TensorSource(key, (key,))
        if self.text_only:
            prefix = "language_model."
            sources = {
                key.removeprefix(prefix): TensorSource(
                    key.removeprefix(prefix), source.physical_keys, source.quantized,
                )
                for key, source in sources.items()
                if key.startswith(prefix)
            }
        return sources

    def dequantize(
        self, source: TensorSource, tensors: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        packed_key, scale_key = source.physical_keys
        packed, scale = tensors[packed_key], tensors[scale_key]
        if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
            raise ValueError(f"MXFP4 weights and scales must both be uint8 for {source.logical_key}")
        if packed.ndim != 2 or scale.ndim != 2:
            raise ValueError(f"MXFP4 weights and scales must be matrices for {source.logical_key}")
        if packed.shape != (scale.size(0), scale.size(1) * (self.GROUP_SIZE // 2)):
            raise ValueError(f"Incompatible MXFP4 shapes for {source.logical_key}: {packed.shape}, {scale.shape}")

        weight = self._unpack_e2m1(packed).unflatten(-1, (-1, self.GROUP_SIZE))
        scales = torch.exp2(scale.float() - 127).unsqueeze(-1).expand_as(weight)
        return Dequantizer.apply_scales_to_bf16(weight, scales).flatten(-2)

    @staticmethod
    def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
        """Decode E2M1 values stored low nibble first."""
        values = packed.new_tensor(
            (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6),
            dtype=torch.float32,
        )
        low = packed & 0x0F
        high = packed >> 4
        nibbles = torch.stack((low, high), dim=-1)
        return values[nibbles.long()].flatten(-2)


class KimiK31TextConverter(Converter):
    """Export the released decoder and LM head as a standalone TorchTurbo model."""

    @staticmethod
    def _mark_config_as_bf16(output_dir: Path) -> None:
        path = output_dir / "config.json"
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)["text_config"]
        config.pop("quantization_config", None)
        config.pop("auto_map", None)
        config.update(
            model_type="kimi_k31_text",
            architectures=["TurboKimiK31ForCausalLM"],
            dtype="bfloat16",
            torch_dtype="bfloat16",
            use_cache=False,
        )
        linear = config["linear_attn_config"]
        linear["checkpoint_num_heads"] = ((linear["num_heads"] + 127) // 128) * 128
        with path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Kimi K3.1 MXFP4 checkpoint weights to BF16.")
    parser.add_argument("--input-hf-path", required=True, help="Source Hugging Face checkpoint directory.")
    parser.add_argument("--output-hf-path", required=True, help="Destination BF16 checkpoint directory.")
    parser.add_argument("--text-only", action="store_true", help="Export a standalone TorchTurbo decoder.")
    args = parser.parse_args(argv)
    converter_cls = KimiK31TextConverter if args.text_only else Converter
    summary = converter_cls(KimiK31Format(text_only=args.text_only)).convert(
        args.input_hf_path, args.output_hf_path,
    )
    print(
        f"[complete] Kimi K3.1 BF16 checkpoint: converted={summary.converted_tensors}, "
        f"copied={summary.copied_tensors}, shards={summary.output_shards}, total_size={summary.total_size}"
    )


if __name__ == "__main__":
    main()
