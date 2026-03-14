#!/bin/bash
source /home/ubuntu/look-ahead/.venv/bin/activate
cd /home/ubuntu/look-ahead/activation-patching
mkdir -p logs
python baseline_experiment_gemma.py > logs/baseline_gemma.log 2>&1
echo "Gemma done, starting Qwen..."
python baseline_experiment_qwen.py > logs/baseline_qwen.log 2>&1
echo "All done."
