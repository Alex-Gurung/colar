#!/usr/bin/env python3
# test_load_colar.py
import os
import sys
import json
import math
import argparse
from pathlib import Path

import torch

def human_bytes(n):
    units = ["B","KB","MB","GB","TB"]
    if n <= 0: return "0 B"
    i = min(int(math.log(n, 1024)), len(units)-1)
    return f"{n/(1024**i):.2f} {units[i]}"

def summarize_state_dict(sd):
    n_params = 0
    n_tensors = 0
    total_bytes = 0
    sample = []
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            n_tensors += 1
            n_params += v.numel()
            total_bytes += v.numel() * v.element_size()
            if len(sample) < 8:
                sample.append(f"{k} {tuple(v.shape)} {str(v.dtype).replace('torch.', '')}")
    return {
        "tensors": n_tensors,
        "params": n_params,
        "size_bytes": total_bytes,
        "sample": sample,
    }

def load_deepspeed_folder(ckpt_dir: Path):
    # Expect .../checkpoint/mp_rank_00_model_states.pt
    ds_file = ckpt_dir / "checkpoint" / "mp_rank_00_model_states.pt"
    if not ds_file.exists():
        return None, "No DeepSpeed file found", None
    print(f"[info] Detected DeepSpeed/Lightning checkpoint: {ds_file}")
    state = torch.load(str(ds_file), map_location="cpu", weights_only=True)
    # Normalize to a plain state_dict
    if isinstance(state, dict):
        sd = state.get("module") or state.get("state_dict") or state
    else:
        sd = state
    summary = summarize_state_dict(sd)
    return sd, None, summary

def is_hf_folder(p: Path):
    # A HF folder has the index json or a pytorch_model.bin / safetensors files
    idx = p / "pytorch_model.bin.index.json"
    return idx.exists()

def load_hf_folder(hf_dir: Path):
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except Exception as e:
        return None, f"transformers not available: {e}", None

    print(f"[info] Detected HF sharded folder: {hf_dir}")
    # Load config first (cheap)
    cfg = AutoConfig.from_pretrained(str(hf_dir))
    print(f"[info] HF config: {cfg.__class__.__name__} | vocab_size={getattr(cfg, 'vocab_size', 'NA')}")

    # Load model weights on CPU with reduced peak memory
    # (low_cpu_mem_usage streams shards; device_map='cpu' keeps it off GPU)
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_dir),
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        device_map="cpu",
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ok] Loaded HF model: {model.__class__.__name__} | params={n_params:,}")
    # Return a tiny summary (don’t print whole model)
    return {"_hf_model_loaded": True, "params": n_params}, None, {
        "tensors": sum(1 for _ in model.state_dict().values()),
        "params": n_params,
        "size_bytes": sum(v.numel()*v.element_size() for v in model.state_dict().values()),
        "sample": list(list(model.state_dict().keys())[:8])
    }

def load_single_file(pt_file: Path):
    print(f"[info] Loading single file: {pt_file}")
    state = torch.load(str(pt_file), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        sd = state.get("state_dict") or state.get("module") or state
    else:
        sd = state
    if not isinstance(sd, dict):
        return None, "Unrecognized checkpoint structure", None
    summary = summarize_state_dict(sd)
    return sd, None, summary

def main():
    ap = argparse.ArgumentParser(description="Tiny CoLaR loader sanity check")
    ap.add_argument("path", help="Path to HF folder, DS/PL checkpoint folder, or single .pt/.ckpt file")
    args = ap.parse_args()
    p = Path(args.path)

    if not p.exists():
        print(f"[err] Path not found: {p}", file=sys.stderr)
        sys.exit(1)

    try:
        if p.is_dir():
            # Prefer DS/PL if present; otherwise HF
            ds_file = p / "checkpoint" / "mp_rank_00_model_states.pt"
            if ds_file.exists():
                sd, err, summary = load_deepspeed_folder(p)
            elif is_hf_folder(p):
                sd, err, summary = load_hf_folder(p)
            else:
                # maybe user passed the parent folder (epoch dir); try inner subfolders
                hf_dir = p / "colar_1epoch"
                if hf_dir.exists() and is_hf_folder(hf_dir):
                    sd, err, summary = load_hf_folder(hf_dir)
                else:
                    sd, err, summary = None, "Directory isn't a HF folder or DS checkpoint", None
        else:
            sd, err, summary = load_single_file(p)

        if err:
            print(f"[err] {err}")
            sys.exit(2)

        # Print compact summary
        print("\n=== SUMMARY ===")
        if summary:
            print(f"tensors: {summary['tensors']}")
            print(f"params:  {summary['params']:,}")
            if 'size_bytes' in summary:
                print(f"size:    {human_bytes(summary['size_bytes'])}")
            print("sample keys:")
            for s in summary["sample"]:
                print(f"  - {s}")

        print("\n[ok] Load sanity check completed.")
    except RuntimeError as e:
        # Commonly OOM or dtype/device mismatches
        print(f"[runtime error] {e}")
        sys.exit(3)
    except Exception as e:
        print(f"[unexpected] {type(e).__name__}: {e}")
        sys.exit(4)

if __name__ == "__main__":
    main()
