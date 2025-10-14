#!/usr/bin/env python3
"""Generate model outputs for a dataset split without running the Lightning test loop."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from run import load_weights_memory_safe
from src.utils.utils import instantiate_from_config

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _create_config(model_name: str, dataset_name: str, trainer_name: str, workspace_path: str, devices: str):
    trainer_cfg = OmegaConf.load(f"src/configs/trainer/{trainer_name}.yaml")
    model_cfg = OmegaConf.load(f"src/configs/models/{model_name}.yaml")
    dataset_cfg = OmegaConf.load(f"src/configs/datasets/{dataset_name}.yaml")

    config = OmegaConf.merge(trainer_cfg, model_cfg, dataset_cfg)

    config.args = OmegaConf.create(
        {
            "model": model_name,
            "dataset": dataset_name,
            "trainer": trainer_name,
            "devices": devices,
            "no_log": False,
            "log_suffix": "",
            "resume_ckpt_path": None,
            "load_ckpt_path": None,
            "workspace_path": workspace_path,
            "do_test": False,
            "test_ckpt_path": "",
            "test_times": 1,
            "seed": 0,
        }
    )
    return config


def _ensure_list(obj):
    if obj is None:
        return []
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, str):
        return [obj]
    return list(obj)


def main():
    parser = argparse.ArgumentParser(description="Generate answers from a checkpoint without running trainer.test().")
    parser.add_argument(
        "--ckpt",
        default="/mnt/disk/baseline_colar/colar/logs/colar/colar-experiments/b7fyxsug/checkpoints/epoch1__step451__monitor3.139.ckpt/",
        help="Path to the checkpoint directory or file to load (default: current best checkpoint).",
    )
    parser.add_argument("--dataset", default="rl_data", help="Dataset config name (e.g. rl_data, qsa).")
    parser.add_argument("--model", default="colar", help="Model config name (default: colar).")
    parser.add_argument("--trainer", default="default", help="Trainer config name (default: default).")
    parser.add_argument(
        "--workspace-path",
        default="/mnt/disk/baseline_colar/colar",
        help="Workspace path referenced by configs (default: /mnt/disk/baseline_colar/colar).",
    )
    parser.add_argument("--devices", default="all", help="Value for args.devices (ignored for standalone generation).")
    parser.add_argument("--device", default=_default_device(), help="Torch device to run generation on (default: cuda:0 if available).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override validation batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers to use for inference (default: 0).")
    parser.add_argument("--output", default=None, help="Optional output path (JSONL). Defaults to logs/manual_generate_<timestamp>.jsonl.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples to generate.")
    args = parser.parse_args()

    config = _create_config(
        model_name=args.model,
        dataset_name=args.dataset,
        trainer_name=args.trainer,
        workspace_path=args.workspace_path,
        devices=args.devices,
    )

    if args.batch_size is not None:
        config.dataloader.val_batch_size = args.batch_size
    config.dataloader.num_workers = args.num_workers
    config.dataloader.persistent_workers = False

    data_module = instantiate_from_config(config.data_module, extra_kwargs={"all_config": config})
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    device = torch.device(args.device)
    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    print(load_weights_memory_safe(model, args.ckpt, cast_bf16=True))
    model.to(device)

    if hasattr(model.llm, "parameters"):
        try:
            target_dtype = next(model.llm.parameters()).dtype
        except StopIteration:
            target_dtype = None
    else:
        target_dtype = None

    if target_dtype is not None:
        model.latent_policy.to(device=device, dtype=target_dtype)

    model.eval()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.output) if args.output else Path("logs") / f"manual_generate_{timestamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with torch.no_grad(), out_path.open("w", encoding="utf-8") as fp:
        progress = tqdm(test_loader, desc="Generating", unit="batch")
        for batch in progress:
            questions = _ensure_list(batch["question"])
            idxs = _ensure_list(batch["idx"])
            steps = _ensure_list(batch.get("steps", [""] * len(questions)))
            answers = _ensure_list(batch.get("answer", [""] * len(questions)))

            token_ids, n_latents = model.latent_generate(questions=questions)
            token_ids = token_ids.cpu()
            latent_counts = n_latents.squeeze(1).detach().cpu().tolist()
            completions = model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            pred_answers = [model.extract_answer_from_output(text) for text in completions]

            for idx, q, st, gt_ans, completion, pred_ans, latent_steps in zip(
                idxs,
                questions,
                steps,
                answers,
                completions,
                pred_answers,
                latent_counts,
            ):
                record = {
                    "idx": int(idx),
                    "question": q,
                    "steps": st,
                    "ground_truth_answer": gt_ans,
                    "output_string": completion,
                    "pred_answer": pred_ans,
                    "n_latent_forward": int(latent_steps),
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1
                if args.limit and total_written >= args.limit:
                    break

            progress.set_postfix({"written": total_written})
            if args.limit and total_written >= args.limit:
                break

    print(f"Wrote {total_written} completions to {out_path}")


if __name__ == "__main__":
    main()
