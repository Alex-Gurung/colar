#!/usr/bin/env python3
"""
Quick script to compare two checkpoints and see if they have the same weights.
"""
import torch
import sys

ckpt1_path = "/mnt/disk/baseline_colar/colar/oldlogs/colar/colar-experiments/5jh51trt/checkpoints/pt_checkpoint_best"
ckpt2_path = "/mnt/disk/baseline_colar/colar/oldlogs/colar/colar-experiments/b7fyxsug/checkpoints/pt_epoch1__step451__monitor3.139.ckpt"

print("Loading checkpoint 1...")
try:
    ckpt1 = torch.load(ckpt1_path, map_location="cpu", weights_only=False)
except Exception as e:
    print(f"Error loading checkpoint 1: {e}")
    sys.exit(1)

print("Loading checkpoint 2...")
try:
    ckpt2 = torch.load(ckpt2_path, map_location="cpu", weights_only=False)
except Exception as e:
    print(f"Error loading checkpoint 2: {e}")
    sys.exit(1)

print("\nCheckpoint 1 keys:", list(ckpt1.keys())[:10])
print("Checkpoint 2 keys:", list(ckpt2.keys())[:10])

# Get state dicts
sd1 = ckpt1.get("state_dict", ckpt1)
sd2 = ckpt2.get("state_dict", ckpt2)

print(f"\nState dict 1 has {len(sd1)} keys")
print(f"State dict 2 has {len(sd2)} keys")

# Compare a few parameters
compare_keys = [
    "latent_policy.mlp.0.weight",
    "latent_policy.mlp.0.bias",
    "llm.model.layers.0.self_attn.q_proj.weight",
]

print("\nComparing some key parameters:")
for key in compare_keys:
    if key in sd1 and key in sd2:
        param1 = sd1[key]
        param2 = sd2[key]
        diff = (param1 - param2).abs().max().item()
        mean_val = param1.abs().mean().item()
        print(f"\n{key}:")
        print(f"  Shape: {param1.shape}")
        print(f"  Max absolute difference: {diff:.6e}")
        print(f"  Mean absolute value: {mean_val:.6e}")
        print(f"  Are they identical? {torch.allclose(param1, param2)}")
    else:
        print(f"\n{key}: NOT FOUND in one or both checkpoints")

# Check if they're exactly the same
print("\n" + "="*60)
print("Checking if ALL parameters are identical...")
all_same = True
for key in sd1.keys():
    if key in sd2:
        if not torch.equal(sd1[key], sd2[key]):
            all_same = False
            # Print first few that differ
            if all_same == False:
                print(f"  First differing key: {key}")
                break

if all_same:
    print("WARNING: All parameters are IDENTICAL! These are the same checkpoint!")
else:
    print("Good: Checkpoints have different weights as expected.")
