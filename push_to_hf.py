"""
Upload selected CoLaR exports to HuggingFace Hub and verify round-trip integrity.

Each checkpoint is uploaded as a separate commit to main, tagged with the epoch name.

Usage:
    python push_to_hf.py --upload          # upload all selected checkpoints
    python push_to_hf.py --verify          # download and verify weights match
    python push_to_hf.py --upload --verify # do both
"""

import argparse
import tempfile
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
from safetensors.torch import load_file as load_safetensors

HF_USER = "agurung"
BASE_DIR = Path(__file__).parent

# ── What to upload ─────────────────────────────────────────────────────────────

REPOS = {
    f"{HF_USER}/colar-qwen3-4b-ff-rl": {
        "checkpoints": [
            ("best-epoch24-val_reward=0.6719", "logs/colar_ff_qwen3_rl/hf_exports/epoch24-step14528-val_reward=0.6719"),
            ("second-epoch08-val_reward=0.6406", "logs/colar_ff_qwen3_rl/hf_exports/epoch08-step5184-val_reward=0.6406"),
            ("last-epoch28-val_reward=0.5781", "logs/colar_ff_qwen3_rl/hf_exports/epoch28-step16864-val_reward=0.5781"),
        ],
        "readme": """---
license: apache-2.0
tags:
  - colar
  - latent-reasoning
  - qwen3
  - reinforcement-learning
base_model: Qwen/Qwen3-4B-Instruct-2507
---

# CoLaR Qwen3-4B Flawed Fictions RL

Compressed Latent Reasoning (CoLaR) model fine-tuned with reinforcement learning on the Flawed Fictions dataset.

**Base model:** Qwen/Qwen3-4B-Instruct-2507
**WandB run:** `hqaakpve`

## Checkpoints

| Tag | Epoch | Step | val/reward |
|-----|-------|------|------------|
| `best-epoch24-val_reward=0.6719` | 24 | 14528 | **0.6719** |
| `second-epoch08-val_reward=0.6406` | 8 | 5184 | 0.6406 |
| `last-epoch28-val_reward=0.5781` | 28 | 16864 | 0.5781 |

Each checkpoint is stored as a tagged commit on `main`. Use:
```python
from huggingface_hub import snapshot_download
snapshot_download("agurung/colar-qwen3-4b-ff-rl", revision="best-epoch24-val_reward=0.6719")
```

## File Structure

- `model.safetensors` — LLM weights (merged LoRA if applicable)
- `extra_state.pt` — Latent policy network weights
- `export_meta.json` — Export metadata

## Training Config

```yaml
model:
  target: src.models.colar.LitCoLaR
  model_kwargs:
    model_id: Qwen/Qwen3-4B-Instruct-2507
    sft_method: colar
    chat_template: True
    do_lora: False
    latent_cot_config:
      ce_weight: 1
      embed_modeling_weight: 1
      embed_modeling_loss: mse
      entropy_weight: 0
      pred_embed_forward_weight: 0
      max_compression_factor: 5
      pred_compressed_cot: True
      sqrt_mean: True
    latent_policy_config:
      lp_determinisitc: False
      lp_intermediate_size: 2560
    latent_generation_config:
      max_n_latent_forward: 64
      latent_temperature: 1.0
      compression_factor: 5
    answer_generation_config:
      max_new_tokens: 2048
      do_sample: True
      top_p: 0.9
      temperature: 1.0
    do_rl: True
    rl_config:
      use_reference_reward: False
      average_per_token_loss: False
      random_speed_in_group: False
      filter_dataset: False
      punish_latent_length: False
      clip_grad_norm: 1.0
      clip_eps: 0.2
      use_latent_loss: True
      use_answer_loss: True
      group_size: 16
      exp_batch_size: 2
      n_train_samples_per_epoch: 290
  training_kwargs:
    optimizer:
      target: torch.optim.AdamW
      lr: 5e-7
      weight_decay: 0.01
    use_scheduler: True
    scheduler:
      target: cosine_with_min_lr
      warmup_steps: 870
      lr_scheduler_kwargs:
        min_lr_rate: 0.1

trainer:
  precision: bf16
  max_epochs: 30
  accumulate_grad_batches: 1
  strategy: ddp
  val_check_interval: 16

dataloader:
  batch_size: 16
  val_batch_size: 16
  num_workers: 4
  drop_last: True

dataset:
  target: src.datasets.rl_data.ConvertedRLDataModule
  dataset_name: ff_rl_data
```
""",
    },
    f"{HF_USER}/colar-qwen3-4b-ff-sft": {
        "checkpoints": [
            ("best-epoch02-val_loss=3.1664", "logs/colar_ff_qwen3/hf_exports/epoch02-step19-val_loss=3.1664"),
            ("second-epoch03-val_loss=3.4093", "logs/colar_ff_qwen3/hf_exports/epoch03-step31-val_loss=3.4093"),
            ("last-epoch04-val_loss=4.0637", "logs/colar_ff_qwen3/hf_exports/epoch04-step39-val_loss=4.0637"),
        ],
        "readme": """---
license: apache-2.0
tags:
  - colar
  - latent-reasoning
  - qwen3
  - supervised-finetuning
base_model: Qwen/Qwen3-4B-Instruct-2507
---

# CoLaR Qwen3-4B Flawed Fictions SFT

Compressed Latent Reasoning (CoLaR) model fine-tuned with supervised learning on the Flawed Fictions dataset.

**Base model:** Qwen/Qwen3-4B-Instruct-2507

## Checkpoints

| Tag | Epoch | Step | val/loss |
|-----|-------|------|----------|
| `best-epoch02-val_loss=3.1664` | 2 | 19 | **3.1664** |
| `second-epoch03-val_loss=3.4093` | 3 | 31 | 3.4093 |
| `last-epoch04-val_loss=4.0637` | 4 | 39 | 4.0637 |

Each checkpoint is stored as a tagged commit on `main`. Use:
```python
from huggingface_hub import snapshot_download
snapshot_download("agurung/colar-qwen3-4b-ff-sft", revision="best-epoch02-val_loss=3.1664")
```

## File Structure

- `model.safetensors` — LLM weights (merged LoRA if applicable)
- `extra_state.pt` — Latent policy network weights
- `export_meta.json` — Export metadata

## Training Config

```yaml
model:
  target: src.models.colar.LitCoLaR
  model_kwargs:
    model_id: Qwen/Qwen3-4B-Instruct-2507
    sft_method: colar
    chat_template: True
    do_lora: False
    latent_cot_config:
      ce_weight: 1
      embed_modeling_weight: 1
      embed_modeling_loss: mse
      entropy_weight: 0
      pred_embed_forward_weight: 0
      max_compression_factor: 5
      pred_compressed_cot: True
      sqrt_mean: True
    latent_policy_config:
      lp_determinisitc: False
      lp_intermediate_size: 2560
    latent_generation_config:
      max_n_latent_forward: 64
      latent_temperature: 1.0
      compression_factor: 5
    answer_generation_config:
      max_new_tokens: 2048
      do_sample: True
      top_p: 0.9
      temperature: 1.0
    do_rl: False
  training_kwargs:
    optimizer:
      target: torch.optim.AdamW
      lr: 1e-4
      weight_decay: 0.01
    use_scheduler: False

trainer:
  precision: bf16
  max_epochs: 5
  accumulate_grad_batches: 12
  strategy: auto
  val_check_interval: 0.5

dataloader:
  batch_size: 4
  val_batch_size: 4
  num_workers: 4
  drop_last: True

dataset:
  target: src.datasets.converted_sft.ConvertedSFTDataModule
  dataset_name: ff_sft_qwen3
```
""",
    },
}


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_all(dry_run=False):
    api = HfApi()

    for repo_id, repo_info in REPOS.items():
        print(f"\n{'=' * 60}")
        print(f"Repo: {repo_id}")
        print(f"{'=' * 60}")

        if not dry_run:
            api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)

        # Upload README first
        if not dry_run:
            print("  Uploading README.md...")
            api.upload_file(
                path_or_fileobj=repo_info["readme"].encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add model card with training config",
            )

        for tag, rel_path in repo_info["checkpoints"]:
            local_path = BASE_DIR / rel_path
            if not local_path.exists():
                print(f"  SKIP {tag}: {local_path} not found")
                continue

            print(f"\n  Uploading '{tag}' from {local_path}")
            if dry_run:
                continue

            # Collect all files to upload in one commit
            operations = []

            # LLM files from llm/ subdir -> repo root
            llm_dir = local_path / "llm"
            if llm_dir.exists():
                for f in llm_dir.iterdir():
                    if f.is_file():
                        operations.append(CommitOperationAdd(
                            path_in_repo=f.name,
                            path_or_fileobj=str(f),
                        ))

            # Extra files at export root
            for fname in ["extra_state.pt", "export_meta.json"]:
                fpath = local_path / fname
                if fpath.exists():
                    operations.append(CommitOperationAdd(
                        path_in_repo=fname,
                        path_or_fileobj=str(fpath),
                    ))

            print(f"    Uploading {len(operations)} files in one commit...")
            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=operations,
                commit_message=f"Upload checkpoint: {tag}",
            )

            # Tag this commit
            try:
                refs = api.list_repo_refs(repo_id, repo_type="model")
                main_sha = None
                for branch in refs.branches:
                    if branch.name == "main":
                        main_sha = branch.target_commit
                        break
                if main_sha:
                    api.create_tag(
                        repo_id=repo_id,
                        repo_type="model",
                        tag=tag,
                        revision=main_sha,
                    )
                    print(f"    Tagged: {tag}")
            except Exception as e:
                print(f"    Warning: could not create tag: {e}")

            print(f"    Done: https://huggingface.co/{repo_id}")


# ── Verify ─────────────────────────────────────────────────────────────────────

def verify_all():
    all_ok = True

    for repo_id, repo_info in REPOS.items():
        print(f"\n{'=' * 60}")
        print(f"Verifying: {repo_id}")
        print(f"{'=' * 60}")

        for tag, rel_path in repo_info["checkpoints"]:
            local_path = BASE_DIR / rel_path
            if not local_path.exists():
                print(f"  SKIP {tag}: local path not found")
                continue

            print(f"\n  Tag '{tag}':")

            with tempfile.TemporaryDirectory() as tmp_dir:
                print(f"    Downloading from HF (revision={tag})...")
                downloaded = snapshot_download(
                    repo_id=repo_id,
                    revision=tag,
                    local_dir=tmp_dir,
                    repo_type="model",
                )
                dl_path = Path(downloaded)

                # Compare LLM weights
                local_st = local_path / "llm" / "model.safetensors"
                remote_st = dl_path / "model.safetensors"

                if local_st.exists() and remote_st.exists():
                    print(f"    Comparing model.safetensors...")
                    local_weights = load_safetensors(str(local_st))
                    remote_weights = load_safetensors(str(remote_st))
                    ok = compare_state_dicts(local_weights, remote_weights, "LLM")
                    all_ok = all_ok and ok
                else:
                    local_shards = sorted((local_path / "llm").glob("model-*.safetensors"))
                    remote_shards = sorted(dl_path.glob("model-*.safetensors"))
                    if local_shards and remote_shards:
                        print(f"    Comparing {len(local_shards)} sharded safetensors...")
                        local_weights = {}
                        for s in local_shards:
                            local_weights.update(load_safetensors(str(s)))
                        remote_weights = {}
                        for s in remote_shards:
                            remote_weights.update(load_safetensors(str(s)))
                        ok = compare_state_dicts(local_weights, remote_weights, "LLM")
                        all_ok = all_ok and ok
                    else:
                        print(f"    WARNING: Could not find safetensors to compare")
                        all_ok = False

                # Compare extra_state.pt
                local_extra = local_path / "extra_state.pt"
                remote_extra = dl_path / "extra_state.pt"
                if local_extra.exists() and remote_extra.exists():
                    print(f"    Comparing extra_state.pt...")
                    local_ex = torch.load(str(local_extra), map_location="cpu", weights_only=True)
                    remote_ex = torch.load(str(remote_extra), map_location="cpu", weights_only=True)
                    ok = compare_state_dicts(local_ex, remote_ex, "extra_state")
                    all_ok = all_ok and ok
                elif local_extra.exists():
                    print(f"    WARNING: extra_state.pt missing from remote!")
                    all_ok = False

    print(f"\n{'=' * 60}")
    if all_ok:
        print("ALL CHECKS PASSED - local and remote weights are identical")
    else:
        print("SOME CHECKS FAILED - see above for details")
    print(f"{'=' * 60}")
    return all_ok


def compare_state_dicts(local_sd, remote_sd, label):
    ok = True
    local_keys = set(local_sd.keys())
    remote_keys = set(remote_sd.keys())

    if local_keys != remote_keys:
        missing = local_keys - remote_keys
        extra = remote_keys - local_keys
        if missing:
            print(f"      {label}: MISSING keys in remote: {missing}")
        if extra:
            print(f"      {label}: EXTRA keys in remote: {extra}")
        ok = False

    mismatched = []
    for key in sorted(local_keys & remote_keys):
        if not torch.equal(local_sd[key], remote_sd[key]):
            mismatched.append(key)

    if mismatched:
        print(f"      {label}: {len(mismatched)} tensors DIFFER: {mismatched[:5]}...")
        ok = False
    else:
        n_keys = len(local_keys & remote_keys)
        total_params = sum(t.numel() for t in local_sd.values())
        print(f"      {label}: {n_keys} tensors, {total_params:,} params - MATCH")

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload CoLaR models to HuggingFace Hub")
    parser.add_argument("--upload", action="store_true", help="Upload models")
    parser.add_argument("--verify", action="store_true", help="Download and verify weights")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    if not args.upload and not args.verify:
        parser.print_help()
        exit(1)

    if args.upload:
        upload_all(dry_run=args.dry_run)

    if args.verify:
        verify_all()
