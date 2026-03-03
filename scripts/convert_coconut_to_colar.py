"""
Convert coconut FF data to CoLaR JSONL format with Qwen3 chat template.

Sources:
  Train: /mnt/disk/coconut/ff_data/train.json
  Val:   /mnt/disk/coconut/ff_data/val.json
  Test:  /mnt/disk/coconut/ff_data/test_litereason.json

Output: /mnt/disk/baseline_colar/coconut_ff_data/{train,val,test}_colar_format.jsonl
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SRC_DIR = Path("/mnt/disk/coconut/ff_data")
DST_DIR = Path("/mnt/disk/baseline_colar/coconut_ff_data")

SPLITS = {
    "train": "train.json",
    "val": "val.json",
    "test": "test_litereason.json",
}


def apply_chat_template(tokenizer, question_text: str) -> str:
    """Wrap the raw question in a Qwen3 chat template (system + user + assistant prefix)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question_text},
    ]
    # add_generation_prompt=True appends the assistant header so the model can continue
    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return templated


def convert_split(tokenizer, split_name: str, src_file: Path, dst_file: Path):
    with open(src_file, "r") as f:
        data = json.load(f)

    print(f"  {split_name}: {len(data)} examples from {src_file}")

    with open(dst_file, "w") as out:
        for idx, item in enumerate(data):
            question = apply_chat_template(tokenizer, item["question"])

            # steps: join list with newlines (CoLaR expects a single string)
            raw_steps = item.get("steps", [])
            if isinstance(raw_steps, list):
                steps = "\n".join(raw_steps) if raw_steps else ""
            else:
                steps = str(raw_steps)

            answer = item["answer"]

            record = {
                "idx": idx,
                "question": question,
                "steps": steps,
                "answer": answer,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"    -> wrote {len(data)} records to {dst_file}")


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    for split_name, src_filename in SPLITS.items():
        src_file = SRC_DIR / src_filename
        dst_file = DST_DIR / f"{split_name}_colar_format.jsonl"
        convert_split(tokenizer, split_name, src_file, dst_file)

    # Quick sanity check on first record
    first_file = DST_DIR / "train_colar_format.jsonl"
    with open(first_file) as f:
        first = json.loads(f.readline())
    print("\nSanity check (train[0]):")
    print(f"  question starts with: {first['question'][:60]!r}")
    print(f"  steps length: {len(first['steps'])} chars")
    print(f"  answer: {first['answer']!r}")
    print(f"  idx: {first['idx']}")
    print("\nDone!")


if __name__ == "__main__":
    main()
