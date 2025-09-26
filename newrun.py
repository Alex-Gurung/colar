import os
import ast
import json
import random
import shutil
import argparse
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig
import torch
import lightning.pytorch as pl
import torch.distributed as dist

from src.utils.utils import instantiate_from_config, get_timestamp, get_metric_statistics
from src.utils.log import setup_logger 


logger = setup_logger(__name__)
start_time = get_timestamp()


def do_test(model: pl.LightningModule, trainer: pl.Trainer, ckpt_path: str, data_module: pl.LightningDataModule, args):
    results = defaultdict(list)
    if ckpt_path == "best":
        state_dict = torch.load(trainer.checkpoint_callback.best_model_path, weights_only=False)["state_dict"]
    elif ckpt_path == "last":
        state_dict = torch.load(trainer.checkpoint_callback.last_model_path, weights_only=False)["state_dict"]
    else:
        state_dict = torch.load(ckpt_path)["state_dict"]
        logger.info(f"Loading ckpt from {ckpt_path}")
    logger.info(model.load_state_dict(state_dict=state_dict, strict=False))
    for i in range(args.test_times):
        pl.seed_everything(args.seed + i)
        res = trainer.test(model=model, datamodule=data_module)[0]
        for k, v in res.items():
            results[k].append(v)
    statistics = {k: get_metric_statistics(v, args.test_times) for k, v in results.items()}
    test_result_in_text = f"Test results: {results}\nTest statistics with {args.test_times} replications: {statistics}"
    model.text_logger.log(test_result_in_text)
    model.sample_logs["test_result"] = test_result_in_text
    model.json_logger.log(model.sample_logs)
    return results, statistics


def instantiate_callbacks(callback_configs: ListConfig):
    callbacks = []
    for callback_cfg in callback_configs:
        callbacks.append(instantiate_from_config(callback_cfg))

    return callbacks


def _preprocess_config(config, args, unknown_args):
    def set_config_key_value(inplace_dict, key_path, value):
        flag = False

        def bfs_set_config_key_value(inplace_dict, key, value):
            nonlocal flag
            if key in inplace_dict.keys():
                inplace_dict[key] = value
                flag = True
            for v in inplace_dict.values():
                if isinstance(v, (DictConfig, dict)):
                    bfs_set_config_key_value(inplace_dict=v, key=key, value=value)
                elif isinstance(v, ListConfig):
                    for item in v:
                        if isinstance(item, (DictConfig, dict)):
                            bfs_set_config_key_value(inplace_dict=item, key=key, value=value)

        keys = key_path.split(".")  # dataset.a.b = 1
        len_keys = len(keys)
        if len_keys == 1:
            bfs_set_config_key_value(inplace_dict, key=key_path, value=value)
            if flag:
                return
            else:
                raise ValueError(f"{key_path} is not found in config")

        for key_idx in range(len_keys - 1):  #
            inplace_dict = inplace_dict[keys[key_idx]]

            if isinstance(inplace_dict, ListConfig):
                for item in inplace_dict:
                    for sub_key_idx in range(key_idx + 1, len_keys - 1):
                        item = item[keys[sub_key_idx]]
                    item[keys[-1]] = value
                return

        inplace_dict[keys[-1]] = value

    is_test = False
    if p := args.test_ckpt_path:
        # load test model config
        config = OmegaConf.load(Path(p).parent.parent / "hparams.yaml").all_config
        is_test = True
    elif p := args.load_ckpt_path:
        # load pretrained ckpt config
        # config.model = OmegaConf.load(Path(p).parent.parent / 'hparams.yaml').all_config.model
        pass

    # set unknown args to config
    for unknown in unknown_args:
        k, v = unknown.split("=")
        v = v.strip("'")
        vlower = v.lower()
        if vlower == "none" or vlower == "~":
            v = None
        else:
            try:
                v = json.loads(vlower)
            except json.decoder.JSONDecodeError:
                pass  # v = v, the str itself
        set_config_key_value(config, k, v)

    # devices
    if (devices := args.devices) is not None:
        if devices == "all":
            devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
        config.trainer.devices = [int(rank) for rank in devices.split(",")]

    if is_test:
        return config

    # ++ begin of training configuration ++#

    # set project name and signature for logging
    if args.no_log:
        config.trainer.logger = False
    else:
        config.trainer.logger.save_dir = f"logs/{args.model}"
        config.trainer.logger.name = f"colar-{args.dataset}-{args.log_suffix}"
        # For WandbLogger, we don't need version, but add it for compatibility
        if hasattr(config.trainer.logger, 'version'):
            config.trainer.logger.version = (
                start_time
                + "_"
                + str(random.randint(100000, 999999))
                + (f"_{args.log_suffix}" if args.log_suffix != "" else "")
            )

    # batch size for ddp
    total_bs = config.dataloader.batch_size
    num_devices = len(config.trainer.devices)
    bs_per_device = total_bs // num_devices
    real_bs = bs_per_device * num_devices
    if real_bs != total_bs:
        logger.warning(f"real batch size is {real_bs}")
    config.dataloader.batch_size = bs_per_device

    # epoch scaling
    epoch_scaling = config.data_module.get("epoch_scaling")
    if epoch_scaling is not None and epoch_scaling != 1:
        config.trainer.max_epochs = int(config.trainer.max_epochs / epoch_scaling)
        logger.info(
            f"Training epoch length is scaled by {epoch_scaling}, thus the num of epochs is decreased to {config.trainer.max_epochs}"
        )

    # customize anything here
    config = preprocess_config_hook(config)

    return config


def preprocess_config_hook(config):
    return config


def get_processed_args_and_config():
    args, unknown_args = get_args()

    OmegaConf.register_new_resolver("eval", ast.literal_eval)

    # load trainer config
    trainer_config = OmegaConf.load(f"src/configs/trainer/{args.trainer}.yaml")
    OmegaConf.resolve(trainer_config)

    # load model config
    model_config = OmegaConf.load(f"src/configs/models/{args.model}.yaml")
    OmegaConf.resolve(model_config)
    config = OmegaConf.merge(trainer_config, model_config)

    # load dataset config
    dataset_config = OmegaConf.load(f"src/configs/datasets/{args.dataset}.yaml")
    OmegaConf.resolve(dataset_config)
    config = OmegaConf.merge(config, DictConfig(dataset_config))

    config = _preprocess_config(config, args, unknown_args)

    # merge args into config
    config = OmegaConf.merge(
        config,
        OmegaConf.create({"args": vars(args), "unkown_args": {x.split("=")[0]: x.split("=")[1] for x in unknown_args}}),
    )

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        logger.info(f"running with config: {config}")

    return args, config

import os, json, glob
import torch

def _load_ckpt_into_model_minimal(model, ckpt_path: str):
    """
    Accepts either:
      - a Lightning .ckpt FILE (with ['state_dict']), or
      - a DIR containing HF shard files: pytorch_model-00001-of-XXXX.bin + pytorch_model.bin.index.json
    Loads into the inner HF model at `model.model`.
    """
    if os.path.isdir(ckpt_path):
        # HF shard dir path
        idx = os.path.join(ckpt_path, "pytorch_model.bin.index.json")
        if not os.path.exists(idx):
            # also allow the case where shards live one level down (e.g., .../last.ckpt/colar_5_epoch)
            # and user passed .../last.ckpt; try to find the child with an index
            cand = glob.glob(os.path.join(ckpt_path, "**", "pytorch_model.bin.index.json"), recursive=True)
            if cand:
                idx = cand[0]
                ckpt_path = os.path.dirname(idx)
        if not os.path.exists(idx):
            raise FileNotFoundError(
                f"Checkpoint dir has no HF shard index file: {ckpt_path}\n"
                f"Expected: {ckpt_path}/pytorch_model.bin.index.json"
            )

        # Merge shards per index (no need for transformers/config.json)
        with open(idx, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        merged = {}
        for shard in shard_files:
            shard_path = os.path.join(ckpt_path, shard)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Missing shard file listed in index: {shard_path}")
            sd = torch.load(shard_path, map_location="cpu")
            merged.update(sd)

        # Load into the wrapped HF model inside the Lightning module
        # (change 'model.model' here if your attribute is named differently)
        missing, unexpected = model.model.load_state_dict(merged, strict=False)
        return f"Loaded HF shards from {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})"

    # File path -> expect Lightning .ckpt with ['state_dict']
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in sd:
        raise KeyError(f"File does not look like a Lightning .ckpt (missing 'state_dict'): {ckpt_path}")
    missing, unexpected = model.load_state_dict(sd["state_dict"], strict=False)
    return f"Loaded Lightning ckpt: {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})"

def _find_ds_ckpt_root(path: str) -> str:
    """
    Walk upward from `path` until we find a directory that looks like a DeepSpeed ZeRO
    checkpoint root (contains a 'latest' file or 'checkpoint' subdir).
    Returns that directory, or raises if not found.
    """
    path = os.path.abspath(path)
    cur = path
    while True:
        latest = os.path.join(cur, "latest")
        ckpt_subdir = os.path.join(cur, "checkpoint")
        if os.path.isfile(latest) or os.path.isdir(ckpt_subdir):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError(
        f"Could not locate a DeepSpeed ZeRO checkpoint root starting from: {path}\n"
        f"Hint: pass the directory like .../logs/.../checkpoints/last.ckpt"
    )

# def _load_weights_from_ds_zero_dir(model, maybe_dir: str) -> str:
#     """
#     Load ONLY model weights from a DeepSpeed ZeRO checkpoint directory into the LightningModule.
#     Keeps optimizer/scheduler fresh (perfect for starting RL).
#     """
#     from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
#     ds_root = _find_ds_ckpt_root(maybe_dir)
#     logger.info(f"Loading DeepSpeed ZeRO checkpoint from: {ds_root}")
#     fp32_sd = get_fp32_state_dict_from_zero_checkpoint(ds_root)  # returns a flat state_dict
#     # Load into the LightningModule (no need to know inner attribute names)
#     missing, unexpected = model.load_state_dict(fp32_sd, strict=False)
#     return (f"Loaded DS ZeRO weights (fresh init) from {ds_root}; "
#             f"missing={len(missing)}, unexpected={len(unexpected)}")

def _rank0_only_zero_to_fp32_then_load(model, ds_zero_dir: str, out_file: str = None, tag: str = None):
    """
    Rank-0 converts a ZeRO checkpoint to a single FP32 state_dict file once.
    All ranks then load that file (cheap) and we warm-start the LightningModule.
    """
    import torch
    import torch.distributed as dist
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

    # pick a default destination
    if out_file is None:
        out_file = os.path.join(ds_zero_dir, "fp32_state_dict.pth")

    is_dist = dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    if rank == 0:
        if not os.path.exists(out_file):
            # One-shot offline conversion to a single .pth file
            # (This avoids 6 concurrent conversions hammering the same shards)
            convert_zero_checkpoint_to_fp32_state_dict(
                checkpoint_dir=ds_zero_dir,
                output_dir=os.path.dirname(out_file),
                max_shard_size="0GB",      # produce a single file
                safe_serialization=False,  # standard torch .pth
                tag=tag
            )
        else:
            logger.info(f"Reusing existing FP32 file: {out_file}")

    if is_dist:
        dist.barrier()  # wait for the file

    # All ranks: load from the consolidated file (fast)
    sd = torch.load(out_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return f"Warm-started from {out_file} (missing={len(missing)}, unexpected={len(unexpected)})"

import os, json, glob, torch
from collections import Counter

import os, json, glob, gc, torch
from collections import defaultdict

def _discover_hf_index_dir(path: str) -> str:
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "pytorch_model.bin.index.json")):
        return path
    # search one level down
    cand = glob.glob(os.path.join(path, "**", "pytorch_model.bin.index.json"), recursive=True)
    if not cand:
        raise FileNotFoundError(f"No HF index json under: {path}")
    return os.path.dirname(cand[0])

def _remap_key_prefix(k: str, target_prefix: str) -> str:
    # collapse duplicates like model.model. -> model.
    if k.startswith("model.model."):
        k = "model." + k[len("model.model."):]
    if target_prefix and not k.startswith(target_prefix + "."):
        k = f"{target_prefix}.{k}"
    return k

def _choose_target_prefix(model) -> str:
    # If your LitCoLaR exposes the backbone with a specific name, prefer it here:
    for name in ["model", "backbone", "language_model", "transformer", "llm", "base_model"]:
        if hasattr(model, name) and isinstance(getattr(model, name), torch.nn.Module):
            return name
    # Otherwise load into the root module (no prefix)
    return ""

def load_hf_shards_streaming(model, ckpt_dir: str, cast_shards_to_bf16: bool = True, only_rank0: bool = True):
    import os, json, glob, gc, torch
    from collections import defaultdict

    def _discover_hf_index_dir(path: str) -> str:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "pytorch_model.bin.index.json")):
            return path
        cand = glob.glob(os.path.join(path, "**", "pytorch_model.bin.index.json"), recursive=True)
        if not cand:
            raise FileNotFoundError(f"No HF index json under: {path}")
        return os.path.dirname(cand[0])

    def _remap_key_prefix(k: str, target_prefix: str) -> str:
        if k.startswith("model.model."):
            k = "model." + k[len("model.model."):]
        if target_prefix and not k.startswith(target_prefix + "."):
            k = f"{target_prefix}.{k}"
        return k

    def _choose_target_prefix(model) -> str:
        for name in ["model", "backbone", "language_model", "transformer", "llm", "base_model"]:
            if hasattr(model, name) and isinstance(getattr(model, name), torch.nn.Module):
                return name
        return ""

    # rank-0 only disk I/O (but always return a 3-tuple)
    if only_rank0 and int(os.environ.get("RANK", "0")) != 0:
        return ("Skipping shard IO on non-zero rank; will receive params via broadcast.",
                [], [])

    ckpt_dir = _discover_hf_index_dir(ckpt_dir)
    index = json.load(open(os.path.join(ckpt_dir, "pytorch_model.bin.index.json")))
    weight_map = index["weight_map"]

    by_shard = defaultdict(list)
    for pname, shard in weight_map.items():
        by_shard[shard].append(pname)

    target_prefix = _choose_target_prefix(model)

    total_missing, total_unexpected = set(), set()
    shard_files = sorted(set(weight_map.values()))
    for shard in shard_files:
        shard_path = os.path.join(ckpt_dir, shard)
        sd = torch.load(shard_path, map_location="cpu")  # only this shard in RAM

        part = {}
        for pname in by_shard[shard]:
            if pname in sd:
                t = sd[pname]
                if cast_shards_to_bf16 and torch.is_floating_point(t):
                    t = t.to(torch.bfloat16)
                part[_remap_key_prefix(pname, target_prefix)] = t

        missing, unexpected = model.load_state_dict(part, strict=False)
        total_missing.update(missing)
        total_unexpected.update(unexpected)

        del sd, part
        gc.collect()

    msg = (f"Streaming-loaded HF shards from {ckpt_dir} -> target_prefix='{target_prefix}'; "
           f"missing={len(total_missing)}, unexpected={len(total_unexpected)}")
    return (msg, list(total_missing)[:20], list(total_unexpected)[:20])





def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="colar")

    parser.add_argument("--dataset", type=str, default="qsa")

    parser.add_argument("--trainer", type=str, default="default")

    parser.add_argument("--devices", type=str, default="0")

    parser.add_argument("--no_log", help="disable training log", action="store_true")

    parser.add_argument("--log_suffix", type=str, help="add suffix to log dir", default="")

    parser.add_argument("--resume_ckpt_path", type=str, help="resume training from ckpt", default=None)

    parser.add_argument("--load_ckpt_path", type=str, help="load ckpt as initialization", default=None)

    parser.add_argument("--workspace_path", type=str, help="assign the path of user workspace directory", default="/workspace/images-ks3-starfs/workspace/wenhui")

    parser.add_argument("--do_test", help="test after training", action="store_true")

    parser.add_argument("--test_ckpt_path", default="")

    parser.add_argument("--test_times", type=int, default=5)

    parser.add_argument("--seed", type=int, default=0)

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main():
    args, config = get_processed_args_and_config()

    pl.seed_everything(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_module: pl.LightningDataModule = instantiate_from_config(
        config.data_module, extra_kwargs={"all_config": config}
    )

    model: pl.LightningModule = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    if p := args.load_ckpt_path:
        logger.info(model.load_state_dict(state_dict=torch.load(p, map_location="cpu", weights_only=False)["state_dict"], strict=False))
    #     # replace the original logger.info(...) with:
    #     # logger.info(_load_ckpt_into_model_minimal(model, p))
    #     # logger.info(_load_weights_from_ds_zero_dir(model, p))
    #     # logger.info(_rank0_only_zero_to_fp32_then_load(model, p))
    #     # Only true rank-0 does the disk IO; the rest will get weights via DDP broadcast.
    #     if int(os.environ.get("RANK", "0")) == 0:
    #         logger.info(_load_ckpt_into_model_minimal(model, p))
    #     # If dist is already up, sync; if not, Lightning will broadcast on setup.
    #     if dist.is_available() and dist.is_initialized():
    #         dist.barrier()

    # if p := args.load_ckpt_path:
    #     # let only true rank-0 touch disk (reduces 6×200GB reads on NFS)
    #     if int(os.environ.get("RANK", "0")) == 0:
    #         # if the user pointed at /.../last.ckpt/, auto-find the child with index.json
    #         msg, miss, unexp = load_hf_shards_streaming(model, p)
    #         logger.info(msg)
    #         if miss or unexp:
    #             logger.info(f"Missing sample: {miss}")
    #             logger.info(f"Unexpected sample: {unexp}")
    #     # sync others; Lightning/strategy will broadcast params after setup anyway
    #     if torch.distributed.is_available() and torch.distributed.is_initialized():
    #         torch.distributed.barrier()

    trainer: pl.Trainer = instantiate_from_config(
        config.trainer, extra_kwargs={"callbacks": instantiate_callbacks(config.callbacks)}
    )

    # test only
    if p := args.test_ckpt_path:
        print(do_test(model=model, trainer=trainer, ckpt_path=p, data_module=data_module, args=args))
        return

    # training
    try:
        if trainer.global_rank == 0:
            # Handle different logger types for backup
            backup_dir = None
            if hasattr(trainer.logger, 'log_dir') and trainer.logger.log_dir:
                backup_dir = os.path.join(trainer.logger.log_dir, "src_backup")
            elif hasattr(trainer.logger, 'save_dir') and trainer.logger.save_dir:
                # For WandbLogger, create backup in save_dir
                os.makedirs(trainer.logger.save_dir, exist_ok=True)
                backup_dir = os.path.join(trainer.logger.save_dir, "src_backup")

            if backup_dir:
                # Remove existing backup and create new one
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree("src", backup_dir)
    except (AttributeError, TypeError, OSError):
        pass
    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

    # test after training
    if args.do_test:
        print(do_test(model=model, trainer=trainer, ckpt_path="best", data_module=data_module, args=args))


if __name__ == "__main__":
    main()
