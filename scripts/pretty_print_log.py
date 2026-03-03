#!/usr/bin/env python3
"""Quick helper to inspect JSON logs produced by JsonLogger/test runs."""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_entries(
    logs: Dict[str, Any], sample_ids: Iterable[str]
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for sid in sample_ids:
        rec = logs.get(sid)
        if not isinstance(rec, dict):
            continue
        yield sid, rec


def _short_question(text: str, width: int = 140) -> str:
    flat = " ".join(str(text).split())
    return textwrap.shorten(flat, width=width, placeholder="…")


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _value_or_missing(seq: List[Any], idx: int, placeholder: str = "<missing>") -> str:
    if idx < len(seq):
        return str(seq[idx])
    return placeholder


def main():
    parser = argparse.ArgumentParser(description="Pretty-print a few samples from a JSON log file.")
    parser.add_argument("log_path", type=Path, help="Path to test_*.json produced by JsonLogger.")
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        help="Optional list of specific sample ids to show. Defaults to the first N numeric ids.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of samples to display when --sample-ids is omitted (default: 5).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Maximum number of repetitions per sample to show (default: 3).",
    )
    args = parser.parse_args()

    logs = _load_json(args.log_path)

    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        numeric = sorted((int(k), k) for k in logs.keys() if k.isdigit())
        sample_ids = [k for _, k in numeric[: args.limit]]
        # Fall back to arbitrary keys if none are numeric
        if not sample_ids:
            sample_ids = list(logs.keys())[: args.limit]

    console = Console()
    for sid, rec in _iter_entries(logs, sample_ids):
        question = rec.get("question", "<missing>")
        steps = rec.get("steps", "<missing>")
        answer = rec.get("answer", "<missing>")
        preds = _ensure_list(rec.get("pred_answer"))
        outputs = _ensure_list(rec.get("output_string"))
        lengths = _ensure_list(rec.get("output_length"))
        latents = _ensure_list(rec.get("n_latent_forward"))
        accs = _ensure_list(rec.get("acc"))
        rewards = _ensure_list(rec.get("reward"))

        console.rule(f"Sample {sid}")
        header = (
            f"[bold]Question:[/bold] {_short_question(question)}\n"
            f"[bold]Steps:[/bold] {steps}\n"
            f"[bold]Answer:[/bold] {answer}"
        )
        console.print(Panel(header, expand=False))

        series = [preds, outputs, lengths, latents, accs, rewards]
        total_runs = max((len(s) for s in series), default=0)
        if total_runs == 0:
            console.print("[italic grey62](no generations logged)[/]")
            continue

        max_runs = min(args.runs, total_runs)
        table = Table(box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Run", style="cyan", no_wrap=True)
        table.add_column("Prediction", style="green")
        table.add_column("Output String", style="magenta")
        table.add_column("Output Len", style="yellow")
        table.add_column("Latent", style="yellow")
        table.add_column("Acc", style="bright_blue")
        table.add_column("Reward", style="bright_blue")

        for idx in range(max_runs):
            row = [
                str(idx + 1),
                _value_or_missing(preds, idx),
                _value_or_missing(outputs, idx),
                _value_or_missing(lengths, idx),
                _value_or_missing(latents, idx),
                _value_or_missing(accs, idx),
                _value_or_missing(rewards, idx),
            ]
            table.add_row(*row)

        console.print(table)


if __name__ == "__main__":
    main()

