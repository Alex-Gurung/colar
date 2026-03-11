#!/usr/bin/env python3
"""
Convert RL dataset JSONL files by wrapping the `question` with the model's
chat template via `tokenizer.apply_chat_template`.

Usage examples:

  python scripts/convert_ff_rl_data_chat.py \
    --in /mnt/disk/baseline_colar/ff_rl_data \
    --out /mnt/disk/baseline_colar/ff_rl_data_chat \
    --model-id Qwen/Qwen2.5-7B-Instruct

  # In-place overwrite (create a backup copy per file with .bak)
  python scripts/convert_ff_rl_data_chat.py --in /mnt/disk/baseline_colar/ff_rl_data --inplace
"""

import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tok


def to_chat_prompt(tokenizer, question: str, add_system: bool = False) -> str:
    messages = []
    if add_system:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({"role": "user", "content": question})
    # add_generation_prompt=True leaves the assistant role open for generation
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def process_split(tokenizer, in_file: Path, out_file: Path, inplace: bool = False, add_system: bool = False) -> int:
    if not in_file.exists():
        return 0

    if inplace:
        # Write to a temp then replace
        tmp_file = in_file.with_suffix(in_file.suffix + ".tmp")
        backup_file = in_file.with_suffix(in_file.suffix + ".bak")
        writer_path = tmp_file
    else:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        writer_path = out_file

    count = 0
    with in_file.open("r", encoding="utf-8") as fin, writer_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question", "")
            if not isinstance(q, str):
                q = str(q)
            chat_q = to_chat_prompt(tokenizer, q, add_system=add_system)
            obj["question"] = chat_q
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    if inplace:
        # Backup original and move tmp into place
        if backup_file.exists():
            backup_file.unlink()
        in_file.rename(backup_file)
        tmp_file.rename(in_file)
    return count


def main():
    ap = argparse.ArgumentParser(description="Wrap RL dataset 'question' with chat template")
    ap.add_argument("--in", dest="in_dir", default="/mnt/disk/baseline_colar/ff_rl_data", help="Input directory containing train/val/test.jsonl")
    ap.add_argument("--out", dest="out_dir", default=None, help="Output directory (defaults to <in>_chat if not inplace)")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct", help="HF model id for tokenizer")
    ap.add_argument("--inplace", action="store_true", help="Overwrite files in place (creates .bak backups)")
    ap.add_argument("--add-system", action="store_true", help="Prepend a simple system message")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    if args.inplace:
        out_dir = in_dir
    else:
        if args.out_dir is not None:
            out_dir = Path(args.out_dir)
        else:
            out_dir = in_dir.parent / f"{in_dir.name}_chat"
        out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_id)

    total = 0
    for split in ["train", "val", "test"]:
        in_file = in_dir / f"{split}.jsonl"
        out_file = out_dir / f"{split}.jsonl"
        n = process_split(tokenizer, in_file, out_file, inplace=args.inplace, add_system=args.add_system)
        if n > 0:
            print(f"Converted {n} examples in {split}.jsonl")
        total += n

    print(f"Done. Total converted: {total}")


if __name__ == "__main__":
    main()

