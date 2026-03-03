"""Custom Lightning callbacks for CoLaR training."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class HFCheckpoint(Callback):
    """Export lightweight HF-format model after every validation epoch.

    Saves ``llm/`` via ``save_pretrained`` + tokenizer + ``extra_state.pt``
    (latent_policy, embedding, etc.) + ``export_meta.json``.  The layout is
    identical to ``convert_checkpoint_to_model.py --format hf`` so that
    ``load_weights_memory_safe`` can reload it.

    Parameters
    ----------
    monitor : str
        Metric key logged during validation (e.g. ``"val/loss"``).
    save_top_k : int
        How many best exports to keep.  ``-1`` keeps all.  ``0`` disables
        top-k pruning (keeps everything).  Default ``-1``.
    mode : str
        ``"min"`` or ``"max"`` – whether smaller or larger *monitor* is better.
    """

    def __init__(
        self,
        monitor: str = "val/loss",
        save_top_k: int = -1,
        mode: str = "min",
    ):
        super().__init__()
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.mode = mode
        # (path, metric_value) pairs kept so far – oldest first
        self._kept: list[tuple[str, float]] = []

    # ------------------------------------------------------------------
    def _is_rank0(self, trainer: pl.Trainer) -> bool:
        return trainer.is_global_zero

    def _export_root(self, trainer: pl.Trainer) -> Path | None:
        """Derive ``hf_exports/`` directory next to ``checkpoints/``."""
        # WandbLogger puts checkpoints under <save_dir>/<name>/<id>/checkpoints
        # TensorBoardLogger uses <save_dir>/<name>/<version>/checkpoints
        # In both cases trainer.log_dir (or logger.log_dir / logger.save_dir)
        # gives us the run directory.
        log_dir = None
        if trainer.logger is not None:
            log_dir = getattr(trainer.logger, "log_dir", None)
            if log_dir is None:
                log_dir = getattr(trainer.logger, "save_dir", None)
        if log_dir is None:
            # no_log mode – fall back to cwd
            return None
        return Path(log_dir) / "hf_exports"

    # ------------------------------------------------------------------
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Skip sanity-check validation (runs before training starts)
        if trainer.sanity_checking:
            return
        if not self._is_rank0(trainer):
            return

        export_root = self._export_root(trainer)
        if export_root is None:
            return

        # Collect the monitored metric value
        metric_val = trainer.callback_metrics.get(self.monitor)
        metric_float = float(metric_val) if metric_val is not None else 0.0

        epoch = trainer.current_epoch
        step = trainer.global_step

        dir_name = f"epoch{epoch:02d}-step{step}-{self.monitor.replace('/', '_')}={metric_float:.4f}"
        final_path = export_root / dir_name

        # Atomic write: export to tmp dir then rename
        tmp_dir = export_root / f".tmp_{dir_name}"
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # 1) Save LLM via HF save_pretrained (bin format for compat)
            llm_dir = tmp_dir / "llm"
            llm_dir.mkdir(parents=True, exist_ok=True)
            pl_module.llm.save_pretrained(str(llm_dir), safe_serialization=False)
            try:
                pl_module.tokenizer.save_pretrained(str(llm_dir))
            except Exception:
                pass

            # 2) Save non-LLM parameters (latent_policy, embedding, etc.)
            extra = {}
            for k, v in pl_module.state_dict().items():
                if not k.startswith("llm."):
                    extra[k] = v.cpu()
            if extra:
                torch.save(extra, tmp_dir / "extra_state.pt")

            # 3) Write metadata
            meta = {
                "epoch": epoch,
                "global_step": step,
                "monitor": self.monitor,
                "monitor_value": metric_float,
                "format": "hf+extra",
                "llm_dir": "llm",
                "extra_file": "extra_state.pt" if extra else None,
            }
            with (tmp_dir / "export_meta.json").open("w") as f:
                json.dump(meta, f, indent=2)

            # Atomic rename
            if final_path.exists():
                shutil.rmtree(final_path)
            tmp_dir.rename(final_path)

        except Exception:
            # Clean up partial write
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        self._kept.append((str(final_path), metric_float))
        self._prune(export_root)

        print(f"[HFCheckpoint] Exported to {final_path}")

    # ------------------------------------------------------------------
    def _prune(self, export_root: Path) -> None:
        """Remove old exports beyond ``save_top_k``."""
        if self.save_top_k <= 0:
            # -1 = keep all, 0 = keep all (no pruning)
            return
        if len(self._kept) <= self.save_top_k:
            return

        # Sort: best last so we pop worst first
        reverse = self.mode == "max"
        self._kept.sort(key=lambda t: t[1], reverse=reverse)

        while len(self._kept) > self.save_top_k:
            worst_path, _ = self._kept.pop(0)
            p = Path(worst_path)
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
                print(f"[HFCheckpoint] Pruned old export: {p.name}")
