#!/bin/bash
# Test the refactored workflow

set -e

MODEL="meta-llama/Llama-3.2-1B-Instruct"
DATASET="../data/example_dataset.jsonl"
MAX_K=3

echo "Testing refactored workflow with max_k=${MAX_K}"
echo ""

# Clean up
rm -rf test_datasets test_probes
mkdir -p test_datasets test_probes

echo "0. Checking model..."
python -m look_ahead_probe.check_model \
    --model_name $MODEL \
    --max_new_tokens 10

echo ""
echo "1. Building dataset..."
python -m look_ahead_probe.build_look_ahead_activation_dataset \
    --model_name $MODEL \
    --dataset_path $DATASET \
    --split_value train \
    --max_k $MAX_K \
    --max_new_tokens 20 \
    --max_prompts 5 \
    --output_path test_datasets/train.pt

echo ""
echo "2. Training probe on layer 6, k=2..."
python -m look_ahead_probe.main \
    --train_dataset test_datasets/train.pt \
    --layer 6 \
    --k 2 \
    --probe_type linear \
    --num_epochs 1 \
    --batch_size 10 \
    --save_path test_probes/probe.pt

echo ""
echo "Test complete!"
