#!/usr/bin/env python3
"""
Summarize evaluation logs.

Key ideas:
  * JsonLogger test files already contain every repetition per sample, so we
    only need to read the last file in a run (unless --include-all-logs is set).
  * For each sample/question we collect its list of accuracies / lengths,
    compute the mean per sample, and then report the dataset mean + SEM.
  * For all metrics: SEM = std(run_means) / sqrt(R)
    This estimates uncertainty due to model stochasticity (finite R samples),
    NOT inter-question variation.
  * A few example rows are shown so it is obvious how the aggregates were built.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from rich import box
from rich.console import Console
from rich.table import Table


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sem(values: List[float]) -> float:
    """Standard SEM = std / sqrt(N). Used for token counts."""
    if len(values) < 2:
        return float("nan")
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var) / math.sqrt(len(values))


def _try_int(text: str) -> str:
    try:
        return f"{int(text):06d}"
    except ValueError:
        return text


def _short(values: List[float], limit: int = 5) -> str:
    if not values:
        return "-"
    if len(values) <= limit:
        return ", ".join(f"{v:.3f}" for v in values)
    head = ", ".join(f"{v:.3f}" for v in values[:limit])
    return f"{head}, … ({len(values)} runs)"


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass
class SampleRecord:
    sample_id: str
    accuracy: List[float]
    output_tokens: List[int]
    latent_tokens: List[int]

    def completions(self) -> int:
        return max(len(self.accuracy), len(self.output_tokens), len(self.latent_tokens))

    def total_output_tokens(self) -> int:
        return sum(self.output_tokens)

    def total_latent_tokens(self) -> int:
        return sum(self.latent_tokens)

    def total_alltogether_tokens(self) -> int:
        return self.total_output_tokens() + self.total_latent_tokens()

    def mean_accuracy(self) -> float | None:
        return _mean(self.accuracy) if self.accuracy else None

    def mean_output_tokens(self) -> float | None:
        return _mean([float(t) for t in self.output_tokens]) if self.output_tokens else None

    def mean_latent_tokens(self) -> float | None:
        return _mean([float(t) for t in self.latent_tokens]) if self.latent_tokens else None

    def mean_alltogether_tokens(self) -> float | None:
        combined = [o + l for o, l in zip(self.output_tokens, self.latent_tokens)]
        return _mean([float(t) for t in combined]) if combined else None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_logger_json(path: Path) -> List[SampleRecord]:
    data = json.loads(path.read_text())
    samples: List[SampleRecord] = []
    for sid, rec in data.items():
        if not isinstance(rec, dict):
            continue
        samples.append(
            SampleRecord(
                sample_id=str(sid),
                accuracy=[float(x) for x in rec.get("acc", []) if x is not None],
                output_tokens=[int(x) for x in rec.get("output_length", []) if x is not None],
                latent_tokens=[int(x) for x in rec.get("n_latent_forward", []) if x is not None],
            )
        )
    samples.sort(key=lambda s: _try_int(s.sample_id))
    return samples


def _load_jsonl(path: Path) -> List[SampleRecord]:
    grouped: Dict[str, SampleRecord] = {}
    auto_idx = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("idx")
            if sid is None:
                sid = f"auto_{auto_idx}"
                auto_idx += 1
            sid = str(sid)
            bucket = grouped.setdefault(
                sid,
                SampleRecord(sample_id=sid, accuracy=[], output_tokens=[], latent_tokens=[]),
            )
            pred = rec.get("pred_answer")
            gt = rec.get("ground_truth_answer")
            if pred is not None and gt is not None:
                try:
                    acc = 1.0 if float(pred) == float(gt) else 0.0
                except Exception:
                    acc = 1.0 if str(pred).strip() == str(gt).strip() else 0.0
                bucket.accuracy.append(acc)
            if rec.get("output_length") is not None:
                try:
                    bucket.output_tokens.append(int(rec["output_length"]))
                except Exception:
                    pass
            if rec.get("n_latent_forward") is not None:
                try:
                    bucket.latent_tokens.append(int(rec["n_latent_forward"]))
                except Exception:
                    pass
    samples = list(grouped.values())
    samples.sort(key=lambda s: _try_int(s.sample_id))
    return samples


def load_samples(path: Path) -> List[SampleRecord]:
    if path.suffix == ".json":
        return _load_logger_json(path)
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    raise ValueError(f"Unsupported file: {path}")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass
class DatasetSummary:
    file: Path
    samples: List[SampleRecord]

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_completions(self) -> int:
        return sum(sample.completions() for sample in self.samples)

    @property
    def total_output_tokens(self) -> int:
        return sum(sample.total_output_tokens() for sample in self.samples)

    @property
    def total_latent_tokens(self) -> int:
        return sum(sample.total_latent_tokens() for sample in self.samples)

    @property
    def total_alltogether_tokens(self) -> int:
        return sum(sample.total_alltogether_tokens() for sample in self.samples)

    @property
    def total_tokens(self) -> int:
        return self.total_output_tokens + self.total_latent_tokens

    def per_sample_means(self, attr: str) -> List[float]:
        means: List[float] = []
        for sample in self.samples:
            getter = getattr(sample, f"mean_{attr}")
            value = getter()
            if value is not None:
                means.append(value)
        return means

    def run_level_sem(self, attr: str) -> float:
        """
        Compute run-level SEM for any attribute:
        SEM = std(run_means) / sqrt(R)

        This estimates uncertainty due to model stochasticity (finite R samples per question).
        Works for accuracy, output_tokens, latent_tokens, alltogether_tokens.
        """
        if not self.samples:
            return float("nan")

        # Determine number of runs from first sample
        first_sample = self.samples[0]
        if attr == "accuracy":
            n_runs = len(first_sample.accuracy)
        elif attr == "output_tokens":
            n_runs = len(first_sample.output_tokens)
        elif attr == "latent_tokens":
            n_runs = len(first_sample.latent_tokens)
        elif attr == "alltogether_tokens":
            n_runs = min(len(first_sample.output_tokens), len(first_sample.latent_tokens))
        else:
            return float("nan")

        if n_runs < 2:
            return float("nan")

        # Compute mean for each run across all samples
        run_means = []
        for r in range(n_runs):
            run_values = []
            for sample in self.samples:
                if attr == "accuracy" and len(sample.accuracy) > r:
                    run_values.append(sample.accuracy[r])
                elif attr == "output_tokens" and len(sample.output_tokens) > r:
                    run_values.append(float(sample.output_tokens[r]))
                elif attr == "latent_tokens" and len(sample.latent_tokens) > r:
                    run_values.append(float(sample.latent_tokens[r]))
                elif attr == "alltogether_tokens" and len(sample.output_tokens) > r and len(sample.latent_tokens) > r:
                    run_values.append(float(sample.output_tokens[r] + sample.latent_tokens[r]))

            if run_values:
                run_means.append(_mean(run_values))

        if len(run_means) < 2:
            return float("nan")

        # SEM = std(run_means) / sqrt(R)
        mu = _mean(run_means)
        var = sum((x - mu) ** 2 for x in run_means) / (len(run_means) - 1)
        return math.sqrt(var) / math.sqrt(len(run_means))


def summarize_dataset(samples: List[SampleRecord], file_path: Path) -> DatasetSummary:
    return DatasetSummary(file=file_path, samples=samples)


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def display_counts(console: Console, summaries: List[DatasetSummary]):
    table = Table(title="Counts", box=box.SIMPLE_HEAD, header_style="bold")
    table.add_column("File", overflow="fold")
    table.add_column("Samples", justify="right")
    table.add_column("Completions", justify="right")
    table.add_column("Tot Output", justify="right")
    table.add_column("Tot Latent", justify="right")
    table.add_column("Tot Alltogether", justify="right")
    table.add_column("Tot Tokens", justify="right")

    total_samples = 0
    total_completions = 0
    total_output = 0
    total_latent = 0
    total_alltogether = 0

    for summary in summaries:
        table.add_row(
            str(summary.file),
            str(summary.n_samples),
            str(summary.n_completions),
            f"{summary.total_output_tokens:,}",
            f"{summary.total_latent_tokens:,}",
            f"{summary.total_alltogether_tokens:,}",
            f"{summary.total_tokens:,}",
        )
        total_samples += summary.n_samples
        total_completions += summary.n_completions
        total_output += summary.total_output_tokens
        total_latent += summary.total_latent_tokens
        total_alltogether += summary.total_alltogether_tokens

    if len(summaries) > 1:
        table.add_section()
        table.add_row(
            "[bold]ALL[/bold]",
            str(total_samples),
            str(total_completions),
            f"[bold]{total_output:,}[/bold]",
            f"[bold]{total_latent:,}[/bold]",
            f"[bold]{total_alltogether:,}[/bold]",
            f"[bold]{total_output + total_latent:,}[/bold]",
        )

    console.print(table)


def display_metrics(console: Console, summaries: List[DatasetSummary]):
    table = Table(title="Means ± SEM", box=box.SIMPLE_HEAD, header_style="bold")
    table.add_column("File", overflow="fold")
    table.add_column("Accuracy", justify="right", no_wrap=True)
    table.add_column("Output Tokens", justify="right", no_wrap=True)
    table.add_column("Latent Tokens", justify="right", no_wrap=True)
    table.add_column("Alltogether Tokens", justify="right", no_wrap=True)

    def fmt_with_sem(values: List[float], sem: float, digits: int = 4, sem_digits: int = 3) -> str:
        """Format mean ± SEM using run-level SEM."""
        if not values:
            return "–"
        mean = _mean(values)
        mean_fmt = f"{{:.{digits}f}}"
        sem_fmt = f"{{:.{sem_digits}f}}"
        sem_str = "–" if math.isnan(sem) else sem_fmt.format(sem)
        return f"{mean_fmt.format(mean)} ± {sem_str}"

    all_samples: List[SampleRecord] = []

    for summary in summaries:
        acc = summary.per_sample_means("accuracy")
        out = summary.per_sample_means("output_tokens")
        latent = summary.per_sample_means("latent_tokens")
        alltogether = summary.per_sample_means("alltogether_tokens")

        # Use run-level SEM for all metrics
        acc_sem = summary.run_level_sem("accuracy")
        out_sem = summary.run_level_sem("output_tokens")
        latent_sem = summary.run_level_sem("latent_tokens")
        alltogether_sem = summary.run_level_sem("alltogether_tokens")

        table.add_row(
            str(summary.file),
            fmt_with_sem(acc, acc_sem),
            fmt_with_sem(out, out_sem, digits=2, sem_digits=2),
            fmt_with_sem(latent, latent_sem, digits=2, sem_digits=2),
            fmt_with_sem(alltogether, alltogether_sem, digits=2, sem_digits=2)
        )
        all_samples.extend(summary.samples)

    if len(summaries) > 1:
        # Compute aggregate run-level SEM for all metrics
        agg_summary = DatasetSummary(file=Path("ALL"), samples=all_samples)
        agg_acc = agg_summary.per_sample_means("accuracy")
        agg_out = agg_summary.per_sample_means("output_tokens")
        agg_latent = agg_summary.per_sample_means("latent_tokens")
        agg_alltogether = agg_summary.per_sample_means("alltogether_tokens")

        agg_acc_sem = agg_summary.run_level_sem("accuracy")
        agg_out_sem = agg_summary.run_level_sem("output_tokens")
        agg_latent_sem = agg_summary.run_level_sem("latent_tokens")
        agg_alltogether_sem = agg_summary.run_level_sem("alltogether_tokens")

        table.add_section()
        table.add_row(
            "[bold]ALL[/bold]",
            f"[bold]{fmt_with_sem(agg_acc, agg_acc_sem)}[/bold]",
            f"[bold]{fmt_with_sem(agg_out, agg_out_sem, digits=2, sem_digits=2)}[/bold]",
            f"[bold]{fmt_with_sem(agg_latent, agg_latent_sem, digits=2, sem_digits=2)}[/bold]",
            f"[bold]{fmt_with_sem(agg_alltogether, agg_alltogether_sem, digits=2, sem_digits=2)}[/bold]",
        )

    console.print(table)


def display_examples(console: Console, summary: DatasetSummary, limit: int):
    if limit <= 0 or not summary.samples:
        return
    table = Table(title=f"Sample examples (first {min(limit, summary.n_samples)})", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Sample ID", justify="right")
    table.add_column("#runs", justify="right")
    table.add_column("Acc mean", justify="right")
    table.add_column("Acc values", overflow="fold")
    table.add_column("Output mean", justify="right")
    table.add_column("Output values", overflow="fold")

    for sample in summary.samples[:limit]:
        acc_mean = sample.mean_accuracy()
        out_mean = sample.mean_output_tokens()
        table.add_row(
            sample.sample_id,
            str(sample.completions()),
            f"{acc_mean:.4f}" if acc_mean is not None else "–",
            _short(sample.accuracy),
            f"{out_mean:.2f}" if out_mean is not None else "–",
            _short([float(x) for x in sample.output_tokens]),
        )

    console.print(table)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def gather_files(in_path: Path, include_all: bool) -> List[Path]:
    if in_path.is_file():
        return [in_path]

    files = sorted(in_path.rglob("*.json")) + sorted(in_path.rglob("*.jsonl"))
    if not files:
        raise SystemExit("No .json or .jsonl files found")

    if include_all:
        return files

    latest_per_dir: Dict[Path, Path] = {}
    for fp in files:
        parent = fp.parent
        current = latest_per_dir.get(parent)
        if current is None or fp.name > current.name:
            latest_per_dir[parent] = fp
    return sorted(latest_per_dir.values())


def main():
    parser = argparse.ArgumentParser(description="Summarize COLAR evaluation logs.")
    parser.add_argument("path", type=Path, help="Path to a log file or directory.")
    parser.add_argument(
        "--include-all-logs",
        action="store_true",
        help="Include every log file found (instead of only the latest per directory).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=2,
        help="Number of per-sample example rows to print from each dataset (default: 2).",
    )
    args = parser.parse_args()

    files = gather_files(args.path, include_all=args.include_all_logs)
    summaries = [summarize_dataset(load_samples(fp), fp) for fp in files]

    console = Console()
    display_counts(console, summaries)
    display_metrics(console, summaries)
    for summary in summaries:
        display_examples(console, summary, args.examples)


if __name__ == "__main__":
    main()

