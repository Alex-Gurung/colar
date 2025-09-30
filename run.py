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


# def do_test(model: pl.LightningModule, trainer: pl.Trainer, ckpt_path: str, data_module: pl.LightningDataModule, args):
#     results = defaultdict(list)
#     if ckpt_path == "best":
#         state_dict = torch.load(trainer.checkpoint_callback.best_model_path, weights_only=False)["state_dict"]
#     elif ckpt_path == "last":
#         state_dict = torch.load(trainer.checkpoint_callback.last_model_path, weights_only=False)["state_dict"]
#     else:
#         state_dict = torch.load(ckpt_path)["state_dict"]
#         logger.info(f"Loading ckpt from {ckpt_path}")
#     logger.info(model.load_state_dict(state_dict=state_dict, strict=False))
def do_test(model: pl.LightningModule, trainer: pl.Trainer, ckpt_path: str, data_module: pl.LightningDataModule, args):
    results = defaultdict(list)
    print(f"data_module", data_module)
    # Load weights first (supports DS/PL dir, HF shards dir, or Lightning .ckpt)
    # if ckpt_path in ("best", "last"):
    #     # Keep original behavior for trainer-managed checkpoints
    #     path = trainer.checkpoint_callback.best_model_path if ckpt_path == "best" else trainer.checkpoint_callback.last_model_path
    #     state_dict = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    #     logger.info(f"Loading ckpt from {path}")
    #     logger.info(model.load_state_dict(state_dict=state_dict, strict=False))
    # else:
    #     # New smart path
    #     msg, miss, unexp = load_colar_ckpt_smart(model, ckpt_path, cast_bf16=True)
    #     logger.info(msg)
    #     if miss:
    #         logger.info(f"Missing (sample): {miss}")
    #     if unexp:
    #         logger.info(f"Unexpected (sample): {unexp}")
    logger.info(load_weights_memory_safe(model, ckpt_path, cast_bf16=True))

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
        # config = OmegaConf.load(Path(p).parent.parent / "hparams.yaml").all_config
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

# --- helpers (top of file is fine) ---
import os, json, glob, gc, torch

def _rank0():
    import torch.distributed as dist
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

def _broadcast_model_from_rank0(model):
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        for p in model.parameters(): dist.broadcast(p.data, src=0)
        for b in model.buffers():    dist.broadcast(b.data, src=0)

def _normalize_key(k: str) -> str:
    if k.startswith("latent_policy."): return k
    if not k.startswith("llm."): k = "llm." + k
    if k.startswith("llm.model.model."):
        k = "llm.model." + k[len("llm.model.model."):]
    if k == "llm.embedding.weight":
        k = "llm.model.embed_tokens.weight"
    return k

def _named_tensors(model):
    d = {}
    for n,p in model.named_parameters(): d[n] = p.data
    for n,b in model.named_buffers():    d[n] = b.data
    return d

def _find_hf_index_dir(path: str) -> str:
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "pytorch_model.bin.index.json")):
        return path
    hits = glob.glob(os.path.join(path, "**", "pytorch_model.bin.index.json"), recursive=True)
    if not hits: raise FileNotFoundError(f"No HF index under {path}")
    return os.path.dirname(hits[0])

@torch.no_grad()
def load_hf_shards_inplace(model, path: str, cast_bf16: bool = True) -> str:
    idx_dir = _find_hf_index_dir(path)
    with open(os.path.join(idx_dir, "pytorch_model.bin.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard = {}
    for pname, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(pname)

    target = _named_tensors(model); allowed = set(target.keys())
    copied = 0
    for shard, keys in by_shard.items():
        shard_path = os.path.join(idx_dir, shard)
        try:
            part = torch.load(shard_path, map_location="cpu", weights_only=True)
        except Exception:
            part = torch.load(shard_path, map_location="cpu")
        for k in keys:
            t = part.get(k); 
            if t is None: continue
            nk = _normalize_key(k)
            if nk not in allowed or target[nk].shape != t.shape: continue
            if cast_bf16 and torch.is_floating_point(t): t = t.to(torch.bfloat16)
            target[nk].copy_(t)
            copied += 1
        del part; gc.collect()
    return f"HF shards loaded in-place from {idx_dir} (copied {copied} tensors)"

@torch.no_grad()
def load_ds_states_weights_only(model, epoch_dir: str, cast_bf16: bool = True) -> str:
    # Only the model file; ignores optimizer shards entirely.
    mp_file = os.path.join(epoch_dir, "checkpoint", "mp_rank_00_model_states.pt")
    if not os.path.exists(mp_file):
        raise FileNotFoundError(mp_file)
    # allowlist OmegaConf in PyTorch 2.6 safe loader
    try:
        from omegaconf import DictConfig
        torch.serialization.add_safe_globals([DictConfig])
    except Exception: pass

    state = torch.load(mp_file, map_location="cpu", weights_only=False)
    sd = state.get("module", state.get("state_dict", state))

    target = model.state_dict()
    keep = {}
    for k, v in sd.items():
        nk = _normalize_key(k)
        if nk in target and target[nk].shape == v.shape:
            keep[nk] = v.to(torch.bfloat16) if (cast_bf16 and torch.is_floating_point(v)) else v
    # This loads only model tensors; no optimizer, no sched, nothing else.
    missing, unexpected = model.load_state_dict(keep, strict=False)
    del state, sd, keep; gc.collect()
    return f"DS model weights loaded (missing={len(missing)}, unexpected={len(unexpected)})"

def load_weights_memory_safe(model, load_path: str, cast_bf16: bool = True) -> str:
    # Prefer HF shards (lowest peak). If not present, fall back to DS model file.
    if os.path.isdir(load_path):
        try:
            if _rank0():
                msg = load_hf_shards_inplace(model, load_path, cast_bf16=cast_bf16)
            _broadcast_model_from_rank0(model)
            return msg if _rank0() else "HF shards received via broadcast"
        except Exception:
            # maybe it's an epoch dir without index.json; try DS file
            pass
        if _rank0():
            msg = load_ds_states_weights_only(model, load_path, cast_bf16=cast_bf16)
        _broadcast_model_from_rank0(model)
        return msg if _rank0() else "DS weights received via broadcast"
    else:
        # Single Lightning .ckpt with 'state_dict' → filter weights only
        try:
            from omegaconf import DictConfig
            torch.serialization.add_safe_globals([DictConfig])
        except Exception: pass
        blob = torch.load(load_path, map_location="cpu", weights_only=False)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        target = model.state_dict()
        keep = {}
        for k, v in sd.items():
            nk = _normalize_key(k)
            if nk in target and target[nk].shape == v.shape:
                keep[nk] = v.to(torch.bfloat16) if (cast_bf16 and torch.is_floating_point(v)) else v
        missing, unexpected = model.load_state_dict(keep, strict=False)
        del blob, sd, keep; gc.collect()
        return f"Single-file weights loaded (missing={len(missing)}, unexpected={len(unexpected)})"


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
        # logger.info(model.load_state_dict(state_dict=torch.load(p, map_location="cpu", weights_only=False)["state_dict"], strict=False))
        logger.info(load_weights_memory_safe(model, p, cast_bf16=True))
        # pure = model.state_dict()
        # torch.save(pure, "/mnt/disk/baseline_colar/colar_postsft_weights_only.bf16.pt")  # ~ model size
        # logger.info(f"Saved model to /mnt/disk/baseline_colar/colar_postsft_weights_only.bf16.pt")
        # msg, miss, unexp = load_colar_ckpt_smart(model, p, cast_bf16=True)
        # logger.info(msg)
        # if miss:
        #     logger.info(f"Missing (sample): {miss}")
        # if unexp:
        #     logger.info(f"Unexpected (sample): {unexp}")

        # logger.info(load_any_ckpt_into_model(model, p, cast_bf16=True))

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
        print(do_test(model=model, trainer=trainer, ckpt_path="last", data_module=data_module, args=args))


if __name__ == "__main__":
    main()
