#!/usr/bin/env python3
"""
Summarize model outputs saved by this repo.

Supports two formats:
- JSON produced by JsonLogger (dict keyed by sample id, values contain lists like
  'acc', 'output_length', 'n_latent_forward', 'output_string', etc.).
- JSONL produced by scripts/generate_answers.py (one record per example).

Metrics reported:
- number of samples and completions
- average accuracy
- average output tokens (assistant text)
- average latent tokens
- average total generated tokens (output + latent)
- totals for the token counts

If token lengths are not logged, they are computed using the
Qwen 2.5 7B Instruct tokenizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _maybe_float(x) -> Union[float, None]:
    try:
        return float(x)
    except Exception:
        return None


_HF_TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"
_tokenizer = None
_FORCE_RECOMPUTE = False


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        # Lazy import to avoid dependency unless needed
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(_HF_TOKENIZER_ID, trust_remote_code=True)
    return _tokenizer


def _count_tokens(texts: List[str]) -> List[int]:
    tok = _get_tokenizer()
    # add_special_tokens=False to match generation decode lengths
    enc = tok(texts, add_special_tokens=False, return_attention_mask=False)
    lengths = [len(ids) for ids in enc["input_ids"]]
    return lengths


def _summarize_logger_json(obj: Dict) -> Tuple[int, int, float, float, float, int, int, int]:
    """Summarize logs saved by JsonLogger.

    Returns a tuple:
      (n_samples, n_completions, avg_acc, avg_out_tokens, avg_latent_tokens,
       total_out_tokens, total_latent_tokens, total_tokens)
    """
    n_samples = 0
    n_completions = 0
    acc_vals: List[float] = []
    out_lens: List[int] = []
    latent_lens: List[int] = []

    # obj is a dict keyed by sample ids (as strings when serialized)
    for _sid, rec in obj.items():
        if not isinstance(rec, dict):
            # Sometimes there can be non-sample fields; skip them gracefully
            continue
        n_samples += 1
        # Collect values (lists per sample)
        acc_vals.extend([float(a) for a in rec.get("acc", []) if a is not None])
        out_lens.extend([int(x) for x in rec.get("output_length", []) if x is not None])
        latent_lens.extend([int(x) for x in rec.get("n_latent_forward", []) if x is not None])

        # If output_length missing but output_string present, compute lengths
        if (_FORCE_RECOMPUTE or not rec.get("output_length")) and rec.get("output_string"):
            to_count = [str(s) for s in rec.get("output_string", [])]
            counted = _count_tokens(to_count)
            if _FORCE_RECOMPUTE:
                out_lens.extend(counted)
            else:
                # If some output_length existed, we might have appended a few already
                out_lens.extend(counted)

        n_completions += max(
            len(rec.get("acc", [])),
            len(rec.get("output_length", [])),
            len(rec.get("output_string", [])),
            len(rec.get("n_latent_forward", [])),
        )

    # Avoid division by zero
    def _avg(lst: List[Union[int, float]]) -> float:
        return float(sum(lst) / len(lst)) if lst else 0.0

    avg_acc = _avg(acc_vals)
    avg_out = _avg(out_lens)
    avg_latent = _avg(latent_lens)
    total_out = int(sum(out_lens))
    total_latent = int(sum(latent_lens))
    total_tokens = total_out + total_latent
    return (
        n_samples,
        n_completions,
        avg_acc,
        avg_out,
        avg_latent,
        total_out,
        total_latent,
        total_tokens,
    )


def _summarize_jsonl(records: Iterable[dict]) -> Tuple[int, int, float, float, float, int, int, int]:
    """Summarize JSONL from generate_answers.py.

    Returns the same tuple as _summarize_logger_json.
    """
    n_samples = 0
    n_completions = 0
    acc_vals: List[float] = []
    out_lens: List[int] = []
    latent_lens: List[int] = []

    batch_texts_to_count: List[str] = []
    batch_texts_indices: List[int] = []  # map back to out_lens positions

    for rec in records:
        n_samples += 1
        n_completions += 1

        # Accuracy from pred_answer vs ground_truth_answer when available
        pred = rec.get("pred_answer")
        gt = rec.get("ground_truth_answer")
        acc = None
        if pred is not None and gt is not None:
            pf = _maybe_float(pred)
            gf = _maybe_float(gt)
            if pf is not None and gf is not None:
                acc = 1.0 if pf == gf else 0.0
            else:
                acc = 1.0 if str(pred).strip() == str(gt).strip() else 0.0
        if acc is not None:
            acc_vals.append(float(acc))

        # Latent length
        if rec.get("n_latent_forward") is not None:
            try:
                latent_lens.append(int(rec["n_latent_forward"]))
            except Exception:
                pass

        # Output length, compute if missing
        if (not _FORCE_RECOMPUTE) and rec.get("output_length") is not None:
            try:
                out_lens.append(int(rec["output_length"]))
            except Exception:
                pass
        else:
            text = rec.get("output_string")
            if isinstance(text, str):
                batch_texts_indices.append(len(out_lens))
                out_lens.append(0)  # placeholder
                batch_texts_to_count.append(text)

    if batch_texts_to_count:
        counts = _count_tokens(batch_texts_to_count)
        for idx, val in zip(batch_texts_indices, counts):
            out_lens[idx] = int(val)

    def _avg(lst: List[Union[int, float]]) -> float:
        return float(sum(lst) / len(lst)) if lst else 0.0

    avg_acc = _avg(acc_vals)
    avg_out = _avg(out_lens)
    avg_latent = _avg(latent_lens)
    total_out = int(sum(out_lens))
    total_latent = int(sum(latent_lens))
    total_tokens = total_out + total_latent
    return (
        n_samples,
        n_completions,
        avg_acc,
        avg_out,
        avg_latent,
        total_out,
        total_latent,
        total_tokens,
    )


def summarize_path(path: Path) -> Tuple[int, int, float, float, float, int, int, int]:
    if path.suffix == ".json":
        obj = _load_json(path)
        return _summarize_logger_json(obj)
    elif path.suffix == ".jsonl":
        return _summarize_jsonl(_iter_jsonl(path))
    else:
        raise ValueError(f"Unsupported file type: {path}")


def main():
    ap = argparse.ArgumentParser(description="Summarize COLAR outputs/logs.")
    ap.add_argument("path", type=str, help="Path to a .json or .jsonl file, or a directory containing such files.")
    ap.add_argument("--recompute-tokens", action="store_true", help="Force recomputing token counts from text outputs.")
    args = ap.parse_args()

    in_path = Path(args.path)
    global _FORCE_RECOMPUTE
    _FORCE_RECOMPUTE = bool(args.recompute_tokens)
    if not in_path.exists():
        raise SystemExit(f"Path not found: {in_path}")

    files: List[Path] = []
    if in_path.is_dir():
        files.extend(sorted(in_path.rglob("*.json")))
        files.extend(sorted(in_path.rglob("*.jsonl")))
        if not files:
            raise SystemExit("No .json or .jsonl files found in directory")
    else:
        files = [in_path]

    grand_totals = {
        "samples": 0,
        "completions": 0,
        "acc_sum": 0.0,
        "acc_count": 0,
        "out_sum": 0,
        "out_count": 0,
        "latent_sum": 0,
        "latent_count": 0,
    }

    for fp in files:
        n_samples, n_completions, avg_acc, avg_out, avg_latent, total_out, total_latent, total_tokens = summarize_path(fp)
        print(f"File: {fp}")
        print(f"  samples: {n_samples}")
        print(f"  completions: {n_completions}")
        print(f"  avg_acc: {avg_acc:.4f}")
        print(f"  avg_output_tokens: {avg_out:.2f}")
        print(f"  avg_latent_tokens: {avg_latent:.2f}")
        print(f"  total_output_tokens: {total_out}")
        print(f"  total_latent_tokens: {total_latent}")
        print(f"  total_tokens: {total_tokens}")

        # accumulate for a global summary
        grand_totals["samples"] += n_samples
        grand_totals["completions"] += n_completions
        if avg_acc > 0 or avg_acc == 0:
            grand_totals["acc_sum"] += avg_acc * (n_completions or n_samples or 1)
            grand_totals["acc_count"] += (n_completions or n_samples or 1)
        if avg_out > 0 or avg_out == 0:
            grand_totals["out_sum"] += total_out
            grand_totals["out_count"] += max(n_completions, 1)
        if avg_latent > 0 or avg_latent == 0:
            grand_totals["latent_sum"] += total_latent
            grand_totals["latent_count"] += max(n_completions, 1)

    if len(files) > 1:
        def _safe_div(a, b):
            return a / b if b else 0.0

        print("\nAggregate summary:")
        print(f"  files: {len(files)}")
        print(f"  samples: {grand_totals['samples']}")
        print(f"  completions: {grand_totals['completions']}")
        print(f"  avg_acc: {_safe_div(grand_totals['acc_sum'], grand_totals['acc_count']):.4f}")
        print(f"  avg_output_tokens: {_safe_div(grand_totals['out_sum'], grand_totals['out_count']):.2f}")
        print(f"  avg_latent_tokens: {_safe_div(grand_totals['latent_sum'], grand_totals['latent_count']):.2f}")
        print(f"  total_output_tokens: {grand_totals['out_sum']}")
        print(f"  total_latent_tokens: {grand_totals['latent_sum']}")
        print(f"  total_tokens: {grand_totals['out_sum'] + grand_totals['latent_sum']}")


if __name__ == "__main__":
    main()
