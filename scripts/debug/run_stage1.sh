#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python src/hgr/training/stage1.py \
    --data_path src/hgr/data/parallel.csv \
    --output_dir ./stage1_output \
    --epochs 10

 