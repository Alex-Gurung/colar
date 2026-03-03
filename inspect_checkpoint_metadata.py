#!/usr/bin/env python3
"""
Inspect checkpoint metadata to understand training history.
"""
import torch
import sys

ckpt1_path = "/mnt/disk/baseline_colar/colar/oldlogs/colar/colar-experiments/5jh51trt/checkpoints/pt_checkpoint_best"
ckpt2_path = "/mnt/disk/baseline_colar/colar/oldlogs/colar/colar-experiments/b7fyxsug/checkpoints/pt_epoch1__step451__monitor3.139.ckpt"

print("="*70)
print("CHECKPOINT 1: 5jh51trt/pt_checkpoint_best")
print("="*70)
ckpt1 = torch.load(ckpt1_path, map_location="cpu", weights_only=False)
print("Top-level keys:", list(ckpt1.keys()))
print()

for key in ckpt1.keys():
    if key != "state_dict":
        val = ckpt1[key]
        print(f"{key}: {val}")
print()

print("="*70)
print("CHECKPOINT 2: b7fyxsug/pt_epoch1__step451__monitor3.139.ckpt")
print("="*70)
ckpt2 = torch.load(ckpt2_path, map_location="cpu", weights_only=False)
print("Top-level keys:", list(ckpt2.keys()))
print()

for key in ckpt2.keys():
    if key != "state_dict":
        val = ckpt2[key]
        print(f"{key}: {val}")
