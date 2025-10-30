#!/usr/bin/env python3
"""
Convert training checkpoints into compact model files without optimizer states.

Inputs supported (via run.load_weights_memory_safe):
- A Lightning .ckpt file
- A DeepSpeed Stage-2 checkpoint directory (with checkpoint/mp_rank_00_model_states.pt)
- A HuggingFace shard directory (with pytorch_model.bin.index.json)

Outputs:
- format=pt: a single PyTorch weights file (state_dict) in bf16
- format=hf: a HuggingFace directory for the LLM; non-LLM modules are saved to extra_state.pt

Example:
  python scripts/convert_checkpoint_to_model.py \
      --ckpt /path/to/logs/.../checkpoints/epoch1__step451__monitor3.139.ckpt/ \
      --out out/model-export \
      --format hf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from omegaconf import OmegaConf

from run import load_weights_memory_safe
from src.utils.utils import instantiate_from_config


def _create_config(model_name: str, dataset_name: str, trainer_name: str, workspace_path: str, devices: str):
    trainer_cfg = OmegaConf.load(f"src/configs/trainer/{trainer_name}.yaml")
    model_cfg = OmegaConf.load(f"src/configs/models/{model_name}.yaml")
    dataset_cfg = OmegaConf.load(f"src/configs/datasets/{dataset_name}.yaml")
    config = OmegaConf.merge(trainer_cfg, model_cfg, dataset_cfg)
    # minimal args stub
    config.args = OmegaConf.create(
        {
            "model": model_name,
            "dataset": dataset_name,
            "trainer": trainer_name,
            "devices": devices,
            "no_log": True,
            "workspace_path": workspace_path,
        }
    )
    return config


def main():
    ap = argparse.ArgumentParser(description="Export weights-only model from checkpoint")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint directory or file")
    ap.add_argument("--out", required=True, help="Output file/dir path")
    ap.add_argument("--format", choices=["pt", "hf"], default="pt", help="Export format")
    ap.add_argument("--model", default="colar", help="Model config name")
    ap.add_argument("--dataset", default="qsa", help="Dataset config name (used only to build model)")
    ap.add_argument("--trainer", default="default", help="Trainer config name (used only to build model)")
    ap.add_argument("--workspace-path", default="/mnt/disk/baseline_colar/colar", help="Workspace path referenced by configs")
    args = ap.parse_args()

    config = _create_config(args.model, args.dataset, args.trainer, args.workspace_path, devices="0")
    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})

    print(load_weights_memory_safe(model, args.ckpt, cast_bf16=True))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "pt":
        state = model.state_dict()
        torch.save(state, out_path)
        print(f"Saved weights-only state_dict to {out_path}")
        return

    # HF directory export
    out_path.mkdir(parents=True, exist_ok=True)
    llm_dir = out_path / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    # Save the LLM in HF format (sharded if large)
    model.llm.save_pretrained(str(llm_dir))
    try:
        model.tokenizer.save_pretrained(str(llm_dir))
    except Exception:
        pass

    # Save non-LLM modules (e.g., latent policy)
    extra = {}
    for k, v in model.state_dict().items():
        if not k.startswith("llm."):
            extra[k] = v
    if extra:
        torch.save(extra, out_path / "extra_state.pt")
    meta = {
        "source": str(args.ckpt),
        "format": "hf+extra",
        "llm_dir": str(llm_dir),
        "extra_file": str(out_path / "extra_state.pt") if extra else None,
    }
    with (out_path / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved HF model to {llm_dir} and extras to {out_path}")


if __name__ == "__main__":
    main()

