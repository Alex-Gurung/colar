#!/usr/bin/env python3
"""
Unified conversion script for flawed-fiction (FF) data.

Handles both SFT and RL modes for any model (Qwen 3, Gemma 3, etc.).
Applies the model's chat template to raw questions.

Usage:
  # SFT – Qwen 3
  python scripts/convert_ff_data.py \
    --model-id Qwen/Qwen3-4B-Instruct-2507 \
    --mode sft \
    --out-dir /mnt/disk/baseline_colar/ff_sft_qwen3

  # SFT – Gemma 3
  python scripts/convert_ff_data.py \
    --model-id google/gemma-3-4b-it \
    --mode sft \
    --out-dir /mnt/disk/baseline_colar/ff_sft_gemma3

  # RL – Qwen 3
  python scripts/convert_ff_data.py \
    --model-id Qwen/Qwen3-4B-Instruct-2507 \
    --mode rl \
    --out-dir /mnt/disk/baseline_colar/ff_rl_qwen3

  # RL – Gemma 3
  python scripts/convert_ff_data.py \
    --model-id google/gemma-3-4b-it \
    --mode rl \
    --out-dir /mnt/disk/baseline_colar/ff_rl_gemma3
"""

import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

# ── Source paths ──────────────────────────────────────────────────────────────

# SFT sources: per-model train/val from coconut, shared test
SFT_SOURCES = {
    "qwen": {
        "train": Path("/mnt/disk/coconut/ff_data/qwen_train.json"),
        "val":   Path("/mnt/disk/coconut/ff_data/qwen_val.json"),
        "test":  Path("/mnt/disk/coconut/ff_data/test_litereason.json"),
    },
    "gemma": {
        "train": Path("/mnt/disk/coconut/ff_data/gemma_train.json"),
        "val":   Path("/mnt/disk/coconut/ff_data/gemma_val.json"),
        "test":  Path("/mnt/disk/coconut/ff_data/test_litereason.json"),
    },
}

# RL sources: shared across models (raw, no chat template)
RL_SOURCE_DIR = Path("/mnt/disk/baseline_colar/ff_rl_data")


def model_family(model_id: str) -> str:
    """Determine model family from HF model id."""
    lower = model_id.lower()
    if "qwen" in lower:
        return "qwen"
    elif "gemma" in lower:
        return "gemma"
    else:
        raise ValueError(f"Unknown model family for {model_id!r}. Expected 'qwen' or 'gemma' in the id.")


def apply_chat_template(tokenizer, question: str) -> str:
    """Wrap raw question with chat template (system + user + assistant prefix)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# ── SFT conversion ───────────────────────────────────────────────────────────

def convert_sft_split(tokenizer, src_file: Path, dst_file: Path, split: str):
    """Convert a coconut JSON array file to CoLaR JSONL format."""
    with open(src_file, "r") as f:
        data = json.load(f)

    print(f"  {split}: {len(data)} examples from {src_file.name}")

    with open(dst_file, "w") as out:
        for idx, item in enumerate(data):
            question = apply_chat_template(tokenizer, item["question"])

            # steps: join list to newline-separated string
            raw_steps = item.get("steps", [])
            if isinstance(raw_steps, list):
                steps = "\n".join(raw_steps) if raw_steps else ""
            else:
                steps = str(raw_steps)

            record = {
                "idx": idx,
                "question": question,
                "steps": steps,
                "answer": item["answer"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"    -> wrote {len(data)} records to {dst_file}")


def convert_sft(tokenizer, model_id: str, out_dir: Path):
    """Convert all SFT splits for the given model family."""
    family = model_family(model_id)
    sources = SFT_SOURCES[family]

    out_dir.mkdir(parents=True, exist_ok=True)

    for split, src_file in sources.items():
        if not src_file.exists():
            print(f"  WARNING: {src_file} not found, skipping {split}")
            continue
        dst_file = out_dir / f"{split}_colar_format.jsonl"
        convert_sft_split(tokenizer, src_file, dst_file, split)


# ── RL conversion ────────────────────────────────────────────────────────────

def convert_rl_split(tokenizer, src_file: Path, dst_file: Path, split: str):
    """Convert an RL JSONL file by wrapping questions with chat template."""
    if not src_file.exists():
        print(f"  WARNING: {src_file} not found, skipping {split}")
        return

    count = 0
    with open(src_file, "r") as fin, open(dst_file, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["question"] = apply_chat_template(tokenizer, obj["question"])
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    print(f"  {split}: {count} examples -> {dst_file}")


def convert_rl(tokenizer, out_dir: Path):
    """Convert all RL splits."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        src_file = RL_SOURCE_DIR / f"{split}.jsonl"
        dst_file = out_dir / f"{split}.jsonl"
        convert_rl_split(tokenizer, src_file, dst_file, split)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Convert FF data with model-specific chat template")
    ap.add_argument("--model-id", required=True, help="HF model id (e.g. Qwen/Qwen3-4B-Instruct-2507)")
    ap.add_argument("--mode", required=True, choices=["sft", "rl"], help="Conversion mode")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if args.mode == "sft":
        convert_sft(tokenizer, args.model_id, out_dir)
    else:
        convert_rl(tokenizer, out_dir)

    # Sanity check: print first record
    first_file = sorted(out_dir.glob("*.jsonl"))
    if first_file:
        with open(first_file[0]) as f:
            first = json.loads(f.readline())
        print(f"\nSanity check ({first_file[0].name} record 0):")
        print(f"  question starts: {first['question'][:80]!r}")
        print(f"  answer: {first['answer']!r}")

    print("\nDone!")


if __name__ == "__main__":
    main()
