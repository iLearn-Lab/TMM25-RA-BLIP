#!/usr/bin/env bash

MODEL_DIR="${MODEL_DIR:-./output}"
DATA_JSON_PATH="${DATA_JSON_PATH:-./webqa_dataset/webqa_test_retrieval_89.json}"
IMAGE_DATA_PATH="${IMAGE_DATA_PATH:-./webqa_dataset/imgs.tsv}"
IMAGE_DATA_IDX="${IMAGE_DATA_IDX:-./webqa_dataset/imgs.lineidx}"
MODEL_NAME="${MODEL_NAME:-model_epoch_9.pth}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python eval_webqa.py \
    --batch_size=16 \
    --model_dir="${MODEL_DIR}" \
    --dataset_json_path="${DATA_JSON_PATH}" \
    --image_data_path="${IMAGE_DATA_PATH}" \
    --image_data_idx="${IMAGE_DATA_IDX}" \
    --model_name="${MODEL_NAME}"
