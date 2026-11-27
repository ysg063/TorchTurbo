# Kimi K3.1 BF16 SFT

Convert a local copy of the released checkpoint. This preserves the tokenizer and image processor assets. The output needs space for BF16 routed experts, and the converter processes one shard at a time.

```bash
python scripts/dequant/kimik31.py \
  --input-hf-path /path/to/Kimi-K3.1-Pro \
  --output-hf-path /path/to/Kimi-K3.1-Pro-BF16
```

Add `--text-only` to export a standalone decoder and LM head. Use `MODEL_TYPE=turbo_kimi_k31_text` for that export and `turbo_kimi_k31` for the full export. Both use the same ms-swift launcher. The Kimi K3.1 template uses the checkpoint's image processor to expand placeholders before packing.

```bash
MODEL=/path/to/Kimi-K3.1-Pro-BF16 DATASET=/path/to/sft.jsonl \
  OUTPUT_DIR=/path/to/run NPROC_PER_NODE=16 \
  bash scripts/runs/kimi_k31/swift_sft.sh --tt_sp_size 8 --tt_ep_size 16
```

The example YAML starts at SP=EP=1 for reduced checkpoints. Pick SP and EP for your topology and model capacity. SP must divide the 96 KDA heads and EP must divide the 896 routed experts. Use the repository's ms-swift dependencies plus `tiktoken` for the released tokenizer.

The DSA indexer starts frozen. Set `dsa_indexer_loss_coeff` to a positive value in the converted checkpoint's `text_config` or at the root for a text-only export to train it. Swift weights the KL by valid queries across the optimizer step, including prompt tokens. Use `RESUME_FROM_CHECKPOINT` to resume from a TorchTurbo training checkpoint. The final checkpoint's `huggingface/` directory contains the BF16 export.

The default config uses the installed NPU providers on Ascend. For an SM100 GPU with FlashMLA's `nv_dev` sparse prefill and cuDNN Frontend 1.30 or later, set `TT_CONFIG=scripts/runs/kimi_k31/swift_sft_cuda.yaml` to select the CUDA DSA implementations explicitly. The model uses FLA for CUDA KDA.

For the hardware smoke test, run `tests/ops/npu/test_sparse_flash_attention.py` on NPU or `tests/ops/test_kimi_cuda_dsa.py` on CUDA. For multi-rank SFT use a reduced checkpoint with real attention dimensions and at least one document longer than 2048 tokens. Cover text and packed images, resume the saved training checkpoint, and reload the final BF16 export.
