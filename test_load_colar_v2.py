#!/usr/bin/env python3
# test_load_colar_v2.py
import argparse, json, os, sys, math
from pathlib import Path

import torch

def human_bytes(n):
    if n <= 0: return "0 B"
    u = ["B","KB","MB","GB","TB"]
    i = min(int(math.log(n, 1024)), len(u)-1)
    return f"{n/(1024**i):.2f} {u[i]}"

def summarize(sd):
    n_params, n_tensors, n_bytes = 0, 0, 0
    sample = []
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            n_tensors += 1
            n_params += v.numel()
            n_bytes += v.numel() * v.element_size()
            if len(sample) < 12:
                sample.append(f"{k}  {tuple(v.shape)}  {v.dtype}")
    return dict(tensors=n_tensors, params=n_params, size_bytes=n_bytes, sample=sample)

def try_load_ds_folder(root: Path):
    """Load a DeepSpeed/Lightning epoch folder → state_dict (CPU)."""
    ds_file = root / "checkpoint" / "mp_rank_00_model_states.pt"
    if not ds_file.exists():
        return None, "no deepspeed file"
    print(f"[info] DS/PL checkpoint detected: {ds_file}")

    # Allow OmegaConf pickles safely (we trust our own artifact here).
    try:
        from omegaconf import DictConfig  # noqa
        torch.serialization.add_safe_globals([DictConfig])  # PyTorch 2.6+
    except Exception:
        pass

    try:
        state = torch.load(str(ds_file), map_location="cpu", weights_only=False)
    except Exception as e:
        return None, f"torch.load failed: {e}"

    if isinstance(state, dict):
        sd = state.get("module") or state.get("state_dict") or state
    else:
        sd = state
    if not isinstance(sd, dict):
        return None, "unrecognized DS/PL structure"
    return sd, None

def is_hf_sharded_folder(p: Path):
    return (p / "pytorch_model.bin.index.json").exists()

def try_load_hf_shards(hf_dir: Path):
    """Merge HF sharded *.bin using the index → state_dict (CPU). Works w/o config.json."""
    idx = hf_dir / "pytorch_model.bin.index.json"
    if not idx.exists():
        return None, "no HF index json"
    print(f"[info] HF sharded folder detected (no AutoModel): {hf_dir}")

    with open(idx, "r") as f:
        index = json.load(f)
    shard_map = index.get("weight_map") or {}

    # Map shard filename -> list of param names
    files_to_keys = {}
    for k, fname in shard_map.items():
        files_to_keys.setdefault(fname, []).append(k)

    merged = {}
    for fname, keys in files_to_keys.items():
        shard_path = hf_dir / fname
        print(f"[info] loading shard: {fname}  ({len(keys)} tensors)")
        # shards are plain tensors dictionary; safe with weights_only=True
        # but older shards might be plain dict, so fallback.
        try:
            part = torch.load(str(shard_path), map_location="cpu", weights_only=True)
        except Exception:
            part = torch.load(str(shard_path), map_location="cpu")
        # Some shards save full dicts; only copy expected keys
        for k in keys:
            if k not in part:
                return None, f"key {k} not found in {fname}"
            merged[k] = part[k]

    if not merged:
        return None, "empty merged state_dict"
    return merged, None

def try_load_single_file(p: Path):
    print(f"[info] single file: {p}")
    # Allow OmegaConf if present
    try:
        from omegaconf import DictConfig  # noqa
        torch.serialization.add_safe_globals([DictConfig])
    except Exception:
        pass
    try:
        state = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:
        return None, f"torch.load failed: {e}"
    if isinstance(state, dict):
        sd = state.get("state_dict") or state.get("module") or state
    else:
        sd = state
    if not isinstance(sd, dict):
        return None, "unrecognized single-file structure"
    return sd, None

def main():
    ap = argparse.ArgumentParser("CoLaR checkpoint loader (CPU, safe on memory)")
    ap.add_argument("path", help="HF shards dir / DS epoch dir / single .pt|.ckpt")
    args = ap.parse_args()
    p = Path(args.path)
    if not p.exists():
        print(f"[err] path not found: {p}", file=sys.stderr)
        sys.exit(1)

    sd, err = None, None
    try:
        if p.is_dir():
            # Prefer DS/PL if available; else plain HF shards; else try nested 'colar_1epoch'
            sd, err = try_load_ds_folder(p)
            if err:
                sd, err = try_load_hf_shards(p)
                if err:
                    nested = p / "colar_1epoch"
                    if nested.exists() and is_hf_sharded_folder(nested):
                        sd, err = try_load_hf_shards(nested)
        else:
            sd, err = try_load_single_file(p)
    except Exception as e:
        print(f"[unexpected] {type(e).__name__}: {e}")
        sys.exit(4)

    if err or sd is None:
        print(f"[err] {err or 'failed to load'}")
        sys.exit(2)

    # Print a minimal summary (prevents massive prints)
    s = summarize(sd)
    print("\n=== STATE_DICT SUMMARY ===")
    print(f"tensors:   {s['tensors']}")
    print(f"params:    {s['params']:,}")
    if 'size_bytes' in s:
        print(f"size:      {human_bytes(s['size_bytes'])}")
    print("sample keys:")
    for k in s["sample"]:
        print("  -", k)

    # Optional: show top-level prefixes (helps you see 'llm.' vs 'latent_policy.')
    prefixes = {}
    for k in sd.keys():
        root = k.split('.', 1)[0]
        prefixes[root] = prefixes.get(root, 0) + 1
    print("\nkey prefixes (counts):", dict(sorted(prefixes.items(), key=lambda x: -x[1])))

    print("\n[ok] Loaded on CPU. To load into your model:")
    print("  model.load_state_dict(sd, strict=False)  # after you instantiate LitCoLaR")
    print("Or only the base LLM part:")
    print("  llm_sd = {k.split('llm.',1)[1]: v for k,v in sd.items() if k.startswith('llm.')}")
    print("  model.llm.load_state_dict(llm_sd, strict=False)")

if __name__ == "__main__":
    main()
