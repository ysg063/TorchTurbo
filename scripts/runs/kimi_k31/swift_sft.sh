#!/bin/bash
# Run text or image SFT with a converted Kimi K3.1 BF16 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${MODEL:?Set MODEL to the converted BF16 checkpoint directory}"
: "${DATASET:?Set DATASET to the Swift SFT dataset or data file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the training output directory}"

TT_CONFIG=${TT_CONFIG:-"${SCRIPT_DIR}/swift_sft.yaml"}
export PYTHONPATH="${REPO_ROOT}/src${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
    sft
    --model "${MODEL}"
    --model_type "${MODEL_TYPE:-turbo_kimi_k31}"
    --template turbo_kimi_k31
    --dataset "${DATASET}"
    --output_dir "${OUTPUT_DIR}"
    --tuner_type full
    --optim adamw_torch
    --attn_impl flash_attn
    --packing true
    --packing_length "${SEQ_LEN:-4096}"
    --max_length "${SEQ_LEN:-4096}"
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
    --max_steps "${MAX_STEPS:-10}"
    --save_steps "${SAVE_STEPS:-5}"
    --eval_strategy no
    --bf16 true
    --tt_config "${TT_CONFIG}"
)

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${PYTHON:-python3}" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE:-1}" --nnodes "${NNODES:-1}" \
    --node_rank "${NODE_RANK:-0}" --master_addr "${MASTER_ADDR:-127.0.0.1}" \
    --master_port "${MASTER_PORT:-23576}" \
    "${REPO_ROOT}/scripts/runs/run_swift.py" "${ARGS[@]}" "$@"
