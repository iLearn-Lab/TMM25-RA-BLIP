#!/usr/bin/env bash

MODEL_DIR="${MODEL_DIR:-./output}"
DATA_JSON_PATH="${DATA_JSON_PATH:-./webqa_dataset/WebQA_train_val.json}"
IMAGE_DATA_PATH="${IMAGE_DATA_PATH:-./webqa_dataset/imgs.tsv}"
IMAGE_DATA_IDX="${IMAGE_DATA_IDX:-./webqa_dataset/imgs.lineidx}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

mkdir -p "${MODEL_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" train_webqa.py \
    --batch_size=2 \
    --num_epochs=10 \
    --warm_up_steps=1000 \
    --lr=1e-6 \
    --min_lr=5e-8 \
    --init_lr=1e-7 \
    --model_dir="${MODEL_DIR}" \
    --dataset_json_path="${DATA_JSON_PATH}" \
    --image_data_path="${IMAGE_DATA_PATH}" \
    --image_data_idx="${IMAGE_DATA_IDX}"
