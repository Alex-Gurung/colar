#!/bin/bash
# CoLaR SFT ablations on Flawed Fictions with Qwen3-4B
# Ablation axis: max_compression_factor (MCF)
#   MCF=1  -> no compression (always r=1)
#   MCF=5  -> moderate (r sampled from {1..5})
#   MCF=10 -> aggressive (r sampled from {1..10})

PYTHON=/mnt/disk/litereason_anon/new/bin/python

for CF in 1 5 10; do
  echo "=========================================="
  echo "Running MCF=${CF}"
  echo "=========================================="
  $PYTHON run.py \
    --devices=0 \
    --model=colar_ff_qwen3 \
    --dataset=converted_sft \
    --do_test \
    --log_suffix=coconut_mcf${CF}_ep5 \
    max_compression_factor=${CF} \
    max_epochs=5
done
