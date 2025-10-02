import tqdm
import random
import time
from typing import List, Tuple
import torch.nn.functional as F
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from liger_kernel.transformers import AutoLigerKernelForCausalLM

from .model_base import LitCoTModelBase
from ..modules.projector import LatentPolicy
from ..modules import grpo
from ..utils.utils import get_position_ids_from_attention_mask, sample_indices_from_attention_mask_3d
from ..utils.constants import MODEL_EMB_STD

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _HAS_LIGER = True
    _FUSED_CE_SUM = LigerFusedLinearCrossEntropyLoss(reduction="sum")  # sum now, mean later
except Exception:
    _HAS_LIGER = False
    _FUSED_CE_SUM = None

import pickle

with open("/mnt/disk/new_nrl_ncp/prompt_to_datapoint_with_baseline_ppl_qwen7B_cpu_new.pkl", "rb") as f:
    prompt_to_datapoint_with_baseline_ppl = pickle.load(f)

def ce_on_selected_tokens(
    last_hidden: torch.Tensor,     # [B, T, D]
    labels: torch.Tensor,          # [B, T]  (-100 = ignore)
    lm_head,                       # nn.Linear
    chunk_size: int = 512,
    backend: str = "liger",        # "liger" or "torch"
) -> torch.Tensor:
    """
    Computes CE only on positions where labels != -100, in chunks to avoid [*, V] blowups.

    backend="liger": uses fused linear+CE (no logits materialized)
    backend="torch": computes logits via lm_head(s) and uses torch CE
    """
    mask = labels.ne(-100)
    if not mask.any():
        return last_hidden.new_zeros(())

    sel_states = last_hidden[mask]  # [N, D]
    sel_labels = labels[mask]       # [N]
    # Ensure proper dtype for CE targets
    if sel_labels.dtype != torch.long:
        sel_labels = sel_labels.long()

    N = sel_states.size(0)
    loss_sum = sel_states.new_zeros(())

    use_liger = (backend == "liger") and _HAS_LIGER

    # Pull weights once
    weight = lm_head.weight if use_liger else None
    bias = getattr(lm_head, "bias", None) if use_liger else None

    for i in range(0, N, chunk_size):
        s = sel_states[i:i + chunk_size]  # [n_i, D]
        y = sel_labels[i:i + chunk_size]  # [n_i]

        if use_liger:
            # Fused matmul + CE (no [n_i, V] logits tensor)
            if bias is None:
                loss_sum = loss_sum + _FUSED_CE_SUM(weight, s, y)
            else:
                loss_sum = loss_sum + _FUSED_CE_SUM(weight, s, y, bias)
        else:
            # Torch fallback: project then CE on this small slice
            logits = lm_head(s)  # [n_i, V]
            # Upcast only this tiny slice for numeric stability
            loss_sum = loss_sum + F.cross_entropy(logits.float(), y, reduction="sum")

    return loss_sum / float(N)

class LitCoLaR(LitCoTModelBase):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)

        latent_policy_config = model_kwargs.latent_policy_config
        self.latent_policy = LatentPolicy(
            feature_size=self.llm.config.hidden_size,
            intermediate_size=latent_policy_config.get("lp_intermediate_size", self.llm.config.hidden_size),
            deterministic=latent_policy_config.get("lp_determinisitc", False),
        )
        self.embeds_std = MODEL_EMB_STD[model_kwargs.model_id]

        # FROZEN-LLM-SFT: Freeze base LLM if requested
        if model_kwargs.get("freeze_base_llm", False):
            self._freeze_base_llm_parameters()

        if model_kwargs.do_rl:
            self.init_rl()

        # model_class = AutoLigerKernelForCausalLM
        # self.baseline_llm = model_class.from_pretrained(model_kwargs.model_id, 
        #     attn_implementation="flash_attention_3", 
        #     trust_remote_code=True, 
        #     dtype=torch.bfloat16, device_map="auto")

        # self.fused_ce_sum = LigerFusedLinearCrossEntropyLoss(reduction="sum")

    def _freeze_base_llm_parameters(self):  # FROZEN-LLM-SFT: Freeze base LLM parameters
        """
        Freeze base LLM parameters while keeping CoLaR-specific parameters trainable.

        Frozen:
        - llm.* (all base model parameters)

        Trainable:
        - latent_policy.* (LatentPolicy network)
        - llm.embed_tokens (only if new tokens were added)
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info("Freezing base LLM parameters for CoLaR SFT")

        # FROZEN-LLM-SFT: Get original vocab size to detect added tokens
        original_vocab_size = getattr(self.llm.config, 'vocab_size', len(self.tokenizer) - 10)
        current_vocab_size = self.llm.get_input_embeddings().weight.shape[0]
        added_tokens = current_vocab_size > original_vocab_size

        frozen_count = 0
        trainable_count = 0

        for name, param in self.named_parameters():
            if name.startswith('llm.'):
                # FROZEN-LLM-SFT: Keep embedding of added tokens trainable
                if added_tokens and 'embed_tokens' in name:
                    # Only freeze original token embeddings, keep new ones trainable
                    if hasattr(param, 'data'):
                        param.data[:original_vocab_size].requires_grad_(False)
                        param.data[original_vocab_size:].requires_grad_(True)
                    param.requires_grad = True
                    trainable_count += param.numel()
                    logger.info(f"Keeping added token embeddings trainable: {name}")
                else:
                    # FROZEN-LLM-SFT: Freeze all other LLM parameters
                    param.requires_grad = False
                    frozen_count += param.numel()

            elif name.startswith('latent_policy.'):
                # FROZEN-LLM-SFT: Keep LatentPolicy trainable
                param.requires_grad = True
                trainable_count += param.numel()
                logger.info(f"Keeping CoLaR parameter trainable: {name}")

            else:
                # FROZEN-LLM-SFT: Other parameters (if any) - keep trainable by default
                param.requires_grad = True
                trainable_count += param.numel()
                if param.numel() > 0:  # Only log if parameter exists
                    logger.info(f"Keeping other parameter trainable: {name}")

        logger.info(f"Frozen parameters: {frozen_count:,}")
        logger.info(f"Trainable parameters: {trainable_count:,}")
        logger.info(f"Frozen percentage: {frozen_count / (frozen_count + trainable_count) * 100:.1f}%")

    # ++ basic methods implemenration begins ++#
    def limit_rl_train_epoch_length(self):
        n_indices = self.model_kwargs.rl_config.n_train_samples_per_epoch
        all_indices = self.trainer.datamodule.get_all_train_indices()
        indices = random.choices(all_indices, k=n_indices)
        self.trainer.datamodule.set_train_indices(indices)

    def on_fit_start(self):
        if self.model_kwargs.do_rl:
            self.limit_rl_train_epoch_length()
        return super().on_fit_start()

    def on_train_epoch_start(self):
        if self.model_kwargs.do_rl:
            self.limit_rl_train_epoch_length()
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx=None, dataloader_idx=0):
        if self.model_kwargs.do_rl:
            return self.rl_training_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        else:
            return self.sft_training_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
    # -- basic methods implemenration ends --#

    # ++ sft implementation begins ++#
    # def sft_training_step(self, batch, batch_idx, dataloader_idx=0):
    #     step_start_time = time.time()

    #     log_dict = self.forward(batch=batch)
    #     log_dict = {f'train/{k}': v for k, v in log_dict.items()}

    #     # Add performance and memory metrics
    #     step_time = time.time() - step_start_time
    #     if torch.cuda.is_available():
    #         memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    #         memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
    #         memory_peak = torch.cuda.max_memory_allocated() / 1024**3  # GB

    #         log_dict.update({
    #             'train/step_time': step_time,
    #             'train/memory_allocated_gb': memory_allocated,
    #             'train/memory_reserved_gb': memory_reserved,
    #             'train/memory_peak_gb': memory_peak,
    #         })

    #     # Add gradient norms and learning rate
    #     if hasattr(self, 'trainer') and self.trainer.optimizers:
    #         optimizer = self.trainer.optimizers[0]
    #         # Learning rate
    #         current_lr = optimizer.param_groups[0]['lr']
    #         log_dict['train/learning_rate'] = current_lr

    #         # Gradient norms for different components
    #         latent_policy_grad_norm = 0.0
    #         llm_grad_norm = 0.0

    #         for name, param in self.named_parameters():
    #             if param.grad is not None:
    #                 param_norm = param.grad.data.norm(2).item()
    #                 if 'latent_policy' in name:
    #                     latent_policy_grad_norm += param_norm ** 2
    #                 elif 'llm' in name:
    #                     llm_grad_norm += param_norm ** 2

    #         log_dict.update({
    #             'train/latent_policy_grad_norm': latent_policy_grad_norm ** 0.5,
    #             'train/llm_grad_norm': llm_grad_norm ** 0.5,
    #         })

    #     self.log_dict(log_dict, sync_dist=True, prog_bar=True, batch_size=len(batch['idx']))
    #     return log_dict['train/total_loss']

    def sft_training_step(self, batch, batch_idx, dataloader_idx=0):
        step_start_time = time.time()

        # forward returns a dict with 'total_loss' and many metrics
        out = self.forward(batch=batch)
        loss = out["total_loss"]                      # keep graph for backward

        # -------- helpers --------
        def _scalar(x):
            """Make a 0-d float tensor on this device (safe for sync_dist)."""
            if torch.is_tensor(x):
                t = x.detach().to(self.device).float()
                return t if t.ndim == 0 else t.mean()
            return torch.tensor(float(x), device=self.device, dtype=torch.float32)

        # Ensure fields that may be omitted by forward() are ALWAYS present
        steps_len = int(out.get("steps_length", 0))
        if "original_steps_length" not in out:
            out["original_steps_length"] = steps_len
        if "compressed_steps_length" not in out:
            out["compressed_steps_length"] = steps_len
        if "actual_compression_ratio" not in out:
            out["actual_compression_ratio"] = 1.0
        if "pred_embed_forward_loss" not in out:
            out["pred_embed_forward_loss"] = 0.0

        # Stable schema: EVERY rank logs these keys EVERY step
        schema = [
            # losses
            "total_loss",
            "ce_loss",
            "pred_embed_forward_loss",
            "embed_modeling_loss",
            "entropy",
            # lengths
            "question_length",
            "steps_length",
            "answer_length",
            "total_sequence_length",
            # compression
            "compression_factor",
            "original_steps_length",
            "compressed_steps_length",
            "actual_compression_ratio",
            # latent stats / norms
            "latent_policy_mean_std",
            "latent_policy_entropy",
            "embedding_norm",
        ]

        # Build log dict with stable keys → 0-d tensors on self.device
        logs = {f"train/{k}": _scalar(out.get(k, 0.0)) for k in schema}

        # Perf + memory (also as 0-d tensors)
        step_time = time.time() - step_start_time
        logs["train/step_time"] = _scalar(step_time)

        if torch.cuda.is_available():
            dev_idx = self.device.index if isinstance(self.device, torch.device) else torch.cuda.current_device()
            logs["train/memory_allocated_gb"] = _scalar(torch.cuda.memory_allocated(dev_idx) / 1024**3)
            logs["train/memory_reserved_gb"]  = _scalar(torch.cuda.memory_reserved(dev_idx)  / 1024**3)
            logs["train/memory_peak_gb"]      = _scalar(torch.cuda.max_memory_allocated(dev_idx) / 1024**3)
        else:
            logs["train/memory_allocated_gb"] = _scalar(0.0)
            logs["train/memory_reserved_gb"]  = _scalar(0.0)
            logs["train/memory_peak_gb"]      = _scalar(0.0)

        # Learning rate (grad norms are better in on_after_backward)
        if getattr(self, "trainer", None) and self.trainer.optimizers:
            opt = self.trainer.optimizers[0]
            logs["train/learning_rate"] = _scalar(opt.param_groups[0]["lr"])
        else:
            logs["train/learning_rate"] = _scalar(0.0)

        # Optional: a 'monitor' metric if your checkpoint callback expects it
        logs["train/monitor"] = -_scalar(out["total_loss"])

        # ✅ True cross-GPU reductions: identical keys & scalar shapes on all ranks
        bs = len(batch["question"]) if "question" in batch else (len(batch.get("idx", [])) or 1)
        self.log_dict(
            logs,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )

        return loss


    # (Optional, recommended) Move grad-norm logging here so grads exist.
    def on_after_backward(self):
        # throttle if you like
        if self.global_step % 10 != 0:
            return

        lp_sq, llm_sq = 0.0, 0.0
        for name, p in self.named_parameters():
            if p.grad is not None:
                g = p.grad.data.norm(2).item()
                if "latent_policy" in name:
                    lp_sq += g * g
                elif "llm" in name:
                    llm_sq += g * g

        # 0-d tensors on device + sync across ranks
        self.log("train/latent_policy_grad_norm", torch.tensor(lp_sq**0.5, device=self.device), sync_dist=True)
        self.log("train/llm_grad_norm",           torch.tensor(llm_sq**0.5, device=self.device), sync_dist=True)

    def forward(self, batch):
        latent_cot_config = self.model_kwargs.latent_cot_config
        max_compression_factor = latent_cot_config.max_compression_factor
        if isinstance(max_compression_factor, int):
            r = random.randint(1, max_compression_factor)
        elif isinstance(max_compression_factor, str):
            max_compression_factor = max_compression_factor.strip(",").split(",")
            r = int(random.choice(max_compression_factor))
        else:
            raise ValueError("max_compression_factor should be int or str")
        # 0: prepare inputs
        question = batch["question"]
        steps = batch["steps"]
        answer = batch["answer"]
        batch_size = len(question)

        # Detailed logging for data inspection
        if not hasattr(self, '_batch_count'):
            self._batch_count = 0
        self._batch_count += 1

        # Log every 10th batch and first few batches
        should_log = (self._batch_count <= 3) or (self._batch_count % 10 == 0)

        if should_log:
            print(f"\n{'='*60}")
            print(f"📊 BATCH {self._batch_count} ANALYSIS")
            print(f"{'='*60}")
            print(f"🔢 Batch size: {batch_size}")
            print(f"🗜️  Compression factor (r): {r}")

            # Sample content inspection
            for i in range(min(2, batch_size)):  # Show first 2 samples
                def smart_truncate(text, max_len):
                    if len(text) <= max_len:
                        return text
                    half = max_len // 2 - 10
                    return f"{text[:half]}...[TRUNCATED]...{text[-half:]}"

                q_display = smart_truncate(question[i], 200)
                s_display = smart_truncate(steps[i], 400)
                a_display = smart_truncate(answer[i], 150)

                print(f"\n📝 Sample {i+1}:")
                print(f"   Question ({len(question[i])} chars):")
                print(f"     {q_display}")
                print(f"   Steps ({len(steps[i])} chars):")
                print(f"     {s_display}")
                print(f"   Answer ({len(answer[i])} chars):")
                print(f"     {a_display}")

            avg_q_len = sum(len(q) for q in question) // batch_size
            avg_s_len = sum(len(s) for s in steps) // batch_size
            avg_a_len = sum(len(a) for a in answer) // batch_size
            print(f"\n📏 Average lengths: Q={avg_q_len}, S={avg_s_len}, A={avg_a_len}")
            print(f"{'='*60}\n")

        # question: [pad, question, speed]
        auto_prob = latent_cot_config.get("replace_r_with_auto_prob", 0)
        if random.random() < auto_prob:
            speed = "auto"
        else:
            speed = r
        question_input_ids, question_attention_mask = self.prepare_inputs(
            question, padding_side="left", part="question", suffix=self.speed_template.format(speed)
        )
        question_inputs_embeds = self.embedding(question_input_ids)

        if should_log:
            print(f"🔍 TOKENIZATION CHECK:")
            print(f"   Question tokens shape: {question_input_ids.shape}")
            # Check if any sequences hit max length (potential truncation)
            seq_lengths = question_attention_mask.sum(dim=1)
            max_seq_len = seq_lengths.max().item()
            min_seq_len = seq_lengths.min().item()
            print(f"   Question sequence lengths: min={min_seq_len}, max={max_seq_len}")
            # Check for actual truncation - only warn if using full tokenizer max length
            if max_seq_len >= 100000:  # Close to tokenizer's 131k limit
                print(f"   ⚠️  WARNING: Question sequences may be approaching tokenizer limit!")
            else:
                print(f"   ✅ No truncation detected")

        # steps: [pad, ###, steps]
        steps_input_ids, steps_attention_mask = self.prepare_inputs(
            steps, padding_side="left", part="steps", prefix=self.thinking_separator
        )

        if should_log:
            steps_seq_lengths = steps_attention_mask.sum(dim=1)
            steps_max_len = steps_seq_lengths.max().item()
            steps_min_len = steps_seq_lengths.min().item()
            print(f"   Steps tokens shape: {steps_input_ids.shape}")
            print(f"   Steps sequence lengths: min={steps_min_len}, max={steps_max_len}")
            # Check for actual truncation - only warn if using full tokenizer max length
            if steps_max_len >= 100000:  # Close to tokenizer's 131k limit
                print(f"   ⚠️  WARNING: Steps sequences may be approaching tokenizer limit!")
            else:
                print(f"   ✅ No steps truncation detected")

            # Show sample tokenized inputs for first sample
            if batch_size > 0:
                print(f"\n🔤 TOKENIZED SAMPLE (first in batch):")
                sample_q_tokens = question_input_ids[0][question_attention_mask[0] == 1]
                sample_s_tokens = steps_input_ids[0][steps_attention_mask[0] == 1]
                decoded_q = self.tokenizer.decode(sample_q_tokens, skip_special_tokens=False)
                decoded_s = self.tokenizer.decode(sample_s_tokens, skip_special_tokens=False)
                print(f"   Tokenized question: {smart_truncate(decoded_q, 200)}")
                print(f"   Tokenized steps: {smart_truncate(decoded_s, 300)}")
        # Store original steps length for compression tracking
        original_steps_length = steps_input_ids.shape[1]
        self._last_original_steps_length = original_steps_length

        if r == 1:
            steps_inputs_embeds = self.embedding(steps_input_ids)
            steps_labels = steps_input_ids
            if should_log:
                print(f"🚀 NO COMPRESSION (r=1): Using original steps")
                print(f"   Steps shape: {steps_input_ids.shape}")
        else:
            if should_log:
                print(f"🗜️  APPLYING COMPRESSION (r={r})...")
                print(f"   Original steps shape: {steps_input_ids.shape}")
            # better skip this else branch as it is really complex
            steps_pad_lengths = -(steps_attention_mask - 1).sum(dim=-1)
            # make sure there are $k*r - 1$ pad tokens before first token ('###'), so that the first token will be at position $k*r$
            # left pad the steps_input_ids and steps_attention_mask
            # suppose r=3, and there are already 4 pad tokens, to make '###' the 6th token, we need to pad 1 more
            n_extra_left_pad_length = r - 1 - steps_pad_lengths % r
            steps_length_left_padded = steps_attention_mask.shape[1] + n_extra_left_pad_length.max()
            # make the whole seq divisible to r
            min_right_pad_length = r - steps_length_left_padded % r
            all_steps_input_ids = []
            all_steps_attention_mask = []
            for b, l_length in enumerate(n_extra_left_pad_length):
                r_length = min_right_pad_length + (r - 1 - l_length)
                if r_length == r:  # if we should pad r extra tokens to the right
                    l_length += r  # we pad it left instead
                    r_length = 0  # to keep the last compressed token not a pad token
                s_ids = steps_input_ids[b]
                s_attn_mask = steps_attention_mask[b]
                if l_length > 0:
                    s_ids = torch.cat(
                        [
                            torch.ones(l_length, device=s_ids.device, dtype=s_ids.dtype) * self.tokenizer.pad_token_id,
                            s_ids,
                        ]
                    )
                    s_attn_mask = torch.cat(
                        [torch.zeros(l_length, device=s_attn_mask.device, dtype=s_attn_mask.dtype), s_attn_mask]
                    )
                if r_length > 0:
                    s_ids = torch.cat(
                        [
                            s_ids,
                            torch.ones(r_length, device=s_ids.device, dtype=s_ids.dtype) * self.tokenizer.pad_token_id,
                        ]
                    )
                    s_attn_mask = torch.cat(
                        [s_attn_mask, torch.zeros(r_length, device=s_attn_mask.device, dtype=s_attn_mask.dtype)]
                    )
                all_steps_input_ids.append(s_ids)
                all_steps_attention_mask.append(s_attn_mask)
            padded_steps_input_ids = torch.stack(all_steps_input_ids, dim=0)
            padded_steps_attention_mask = torch.stack(all_steps_attention_mask, dim=0)
            padded_steps_inputs_embeds = self.embedding(padded_steps_input_ids)
            padded_steps_inputs_embeds *= padded_steps_attention_mask.unsqueeze(-1)

            padded_steps_length = padded_steps_inputs_embeds.shape[1]
            compressed_steps_length = padded_steps_length // r
            compressed_steps_inputs_embeds = padded_steps_inputs_embeds.reshape(
                batch_size, compressed_steps_length, r, padded_steps_inputs_embeds.shape[-1]
            ).sum(dim=2)
            compressed_steps_attention_mask = padded_steps_attention_mask.reshape(
                batch_size, compressed_steps_length, r
            ).sum(dim=2)
            if latent_cot_config.get("sqrt_mean", False):
                compressed_steps_attention_mask = compressed_steps_attention_mask.sqrt()
            compressed_steps_inputs_embeds /= compressed_steps_attention_mask.unsqueeze(-1) + 1e-5
            compressed_steps_attention_mask = (compressed_steps_attention_mask != 0).long()
            compressed_steps_labels = padded_steps_input_ids.reshape(batch_size, compressed_steps_length, r)
            # rand_steps_indices = torch.randint(0, r, (batch_size, compressed_steps_length, 1), device=steps_input_ids.device)
            rand_steps_indices = sample_indices_from_attention_mask_3d(
                padded_steps_attention_mask.view(batch_size, compressed_steps_length, r)
            )
            compressed_steps_labels = compressed_steps_labels.gather(dim=2, index=rand_steps_indices).squeeze(dim=2)

            # finally we are here:
            steps_inputs_embeds = compressed_steps_inputs_embeds
            steps_attention_mask = compressed_steps_attention_mask
            steps_labels = compressed_steps_labels

            if should_log:
                print(f"✅ COMPRESSION COMPLETE:")
                print(f"   Original length: {padded_steps_length}")
                print(f"   Compressed length: {compressed_steps_length}")
                print(f"   Compression ratio: {padded_steps_length/compressed_steps_length:.2f}x")
                print(f"   Final steps shape: {steps_inputs_embeds.shape}")

        # answer: [###, answer, eos, pad]
        answer_input_ids, answer_attention_mask = self.prepare_inputs(
            answer,
            padding_side="right",
            part="answer",
            prefix=self.thinking_separator,
            suffix=self.tokenizer.eos_token,
        )
        answer_inputs_embeds = self.embedding(answer_input_ids)

        question_length = question_inputs_embeds.shape[1]
        steps_length = steps_inputs_embeds.shape[1]

        inputs_embeds = torch.cat([question_inputs_embeds, steps_inputs_embeds, answer_inputs_embeds], dim=1)
        attention_mask = torch.cat([question_attention_mask, steps_attention_mask, answer_attention_mask], dim=1)
        position_ids = get_position_ids_from_attention_mask(attention_mask)
        labels = torch.cat([question_input_ids, steps_labels, answer_input_ids], dim=1)
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, :question_length] = -100

        # print(f"inputs_embeds: {inputs_embeds.shape}")
        # print(f"labels: {labels.shape}")

        outputs = self.llm.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True,
        )
        ce_loss = outputs.loss
        # transformer_out = self.llm.model(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     # use_cache=False,
        #     output_hidden_states=True,
        # )
        # last_hidden = transformer_out.last_hidden_state  # [B, T, D]

        # 2) what you already do:
        # steps_outputs = last_hidden[:, question_length : question_length + steps_length, :]

        # 4) selected-token CE (small, safe upcast on chunks only)
        # ce_loss = ce_on_selected_tokens(last_hidden, 
        #     labels, self.llm.lm_head, chunk_size=512, backend="liger")

        # latent loss
        # steps_outputs = transformer_out.hidden_states[-1][:, question_length : question_length + steps_length, :]
        steps_outputs = outputs.hidden_states[-1][:, question_length : question_length + steps_length, :]
        distributions = self.latent_policy.forward(steps_outputs)
        gold_embeds = inputs_embeds[:, question_length + 1 : question_length + steps_length + 1, :]
        pred_embeds = distributions.rsample()
        if latent_cot_config.get("embed_modeling_loss", "nll") == "nll":
            embed_modeling_loss = -distributions.log_prob(gold_embeds.detach() / self.embeds_std).mean(dim=-1)
        else:
            embed_modeling_loss = F.mse_loss(
                pred_embeds, gold_embeds.detach() / self.embeds_std, reduction="none"
            ).mean(dim=-1)
        embed_modeling_loss = (embed_modeling_loss * steps_attention_mask).sum() / steps_attention_mask.sum()

        entropy = distributions.entropy().mean(dim=-1)
        entropy = (entropy * steps_attention_mask).sum() / steps_attention_mask.sum()

        # pred_embed_forward  # only used in NLL loss for faster convergence
        if latent_cot_config.pred_embed_forward_weight != 0:
            second_input_embeds = torch.cat(
                [
                    question_inputs_embeds,  # question: [pad, question]
                    answer_inputs_embeds[:, 0:1, :],  #: ['###']
                    pred_embeds[:, 1:, :],  # [pad, steps]
                    answer_inputs_embeds,  # [###, answer, eos, pad]
                ],
                dim=1,
            )
            second_attention_mask = torch.cat(
                [
                    question_attention_mask,
                    torch.ones_like(answer_attention_mask[:, 0:1]),
                    steps_attention_mask[:, 1:],
                    answer_attention_mask,
                ],
                dim=1,
            )
            second_position_ids = get_position_ids_from_attention_mask(second_attention_mask)
            # only supervise the answer
            second_outputs = self.llm.forward(
                inputs_embeds=second_input_embeds,
                attention_mask=second_attention_mask,
                position_ids=second_position_ids,
                labels=labels,
            )
            pred_embed_forward_loss = second_outputs.loss
        else:
            pred_embed_forward_loss = 0.0

        # total loss
        total_loss = 0
        if latent_cot_config.get("ce_weight", 1) != 0:
            total_loss += ce_loss * latent_cot_config.ce_weight
        if latent_cot_config.get("embed_modeling_weight", 0) != 0:
            total_loss += embed_modeling_loss * latent_cot_config.embed_modeling_weight
        if latent_cot_config.get("entropy_weight", 0) != 0:
            total_loss += entropy * latent_cot_config.entropy_weight
        if latent_cot_config.get("pred_embed_forward_weight", 0) != 0:
            total_loss += pred_embed_forward_loss * latent_cot_config.pred_embed_forward_weight

        # Add comprehensive logging metrics
        log_metrics = {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "pred_embed_forward_loss": pred_embed_forward_loss,
            "embed_modeling_loss": embed_modeling_loss,
            "entropy": entropy,
            # Sequence length tracking
            "question_length": question_length,
            "steps_length": steps_length,
            "answer_length": answer_inputs_embeds.shape[1],
            "total_sequence_length": inputs_embeds.shape[1],
            # Compression metrics
            "compression_factor": r,
        }

        # Add compression-specific metrics
        if r > 1:
            original_steps_length = getattr(self, '_last_original_steps_length', steps_length)
            compression_ratio = original_steps_length / steps_length if steps_length > 0 else 1.0
            log_metrics.update({
                "original_steps_length": original_steps_length,
                "compressed_steps_length": steps_length,
                "actual_compression_ratio": compression_ratio,
            })

        # Latent policy statistics
        if hasattr(self, 'latent_policy'):
            with torch.no_grad():
                if 'distributions' in locals():
                    log_metrics.update({
                        "latent_policy_mean_std": distributions.scale.mean().item(),
                        "latent_policy_entropy": distributions.entropy().mean().item(),
                    })
                # Embedding norms
                embedding_norm = torch.norm(inputs_embeds, dim=-1).mean().item()
                log_metrics["embedding_norm"] = embedding_norm

        return log_metrics

    # -- sft training ends --#

    # ++ rl implementation begins ++#
    def init_rl(self):
        self.grpo_loss = grpo.GRPOLoss(rl_config=self.model_kwargs.rl_config)
        self.replay_buffer = grpo.ReplayBuffer()
        self.automatic_optimization = False

    @torch.no_grad()
    def filter_train_indices(self, dataloader_to_filter_indices):
        """
        this function is not used in our paper, but might be helpful for future work
        """
        train_indices = []
        for batch in tqdm.tqdm(dataloader_to_filter_indices, desc="filtering train indices"):
            q = batch["question"]
            a = batch["answer"]
            idx = batch["idx"]
            batch_size = idx.shape[0]  # (batch_size, 1)
            exp = self.batch_rollout(questions=q, gt_answers=a)
            mean_acc = exp.accuracies.reshape(batch_size, -1).mean(-1)  # (batch_size, 1)
            # remove too easy questions
            train_indices.extend(idx[mean_acc.cpu() < 1.0].tolist())
        self.text_logger.log(f"filtered {len(train_indices)} train indices")
        return train_indices

    def rl_training_step(self, batch, batch_idx, dataloader_idx=0):
        rl_config = self.model_kwargs.rl_config
        questions = batch["question"]
        answers = batch["answer"]
        self.replay_buffer.clear()
        optimizer = self.optimizers()

        experience = self.rollout(questions=questions, gt_answers=answers)
        self.replay_buffer.append(experience.to("cpu"))

        self.log_dict(
            {
                "train/rewards": experience.rewards.mean(),
                "train/accuracies": experience.accuracies.mean(),
                "train/n_latent_forward": experience.n_latent_forward.float().mean(),
            }
        )
        torch.cuda.empty_cache()
        experience_dataloader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=rl_config.exp_batch_size,
            shuffle=True,
            collate_fn=grpo.join_experience_batch,
        )

        for experience in experience_dataloader:
            experience: grpo.Experience = experience.to(self.device)
            latent_logprobs, answer_logprobs = self.get_logprobs(e=experience)
            loss_dict = self.grpo_loss(
                latent_logprobs=latent_logprobs,
                answer_logprobs=answer_logprobs,
                experience=experience,
            )
            optimizer.zero_grad()
            self.manual_backward(loss_dict["total_loss"])
            grad_norm = clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            log_dict["train/grad_norm"] = grad_norm
            self.log_dict(log_dict)

    @torch.no_grad()
    def rollout(self, questions: List[str], gt_answers) -> grpo.Experience:
        # 0: prepare variables
        rl_config = self.model_kwargs.rl_config
        batch_size = len(questions)
        group_size = rl_config.group_size

        group_questions = []
        for q in questions:
            group_questions.extend([q] * group_size)

        print(f"group_questions: {group_questions}")
        print(f"len(group_questions): {len(group_questions)}")

        # 1: sample
        (question_input_ids, question_attention_mask, latent_inputs_embeds, latent_attention_mask, pred_ids) = (
            self.latent_generate(
                questions=group_questions,
                rl_mode=True,
            )
        )
        torch.cuda.empty_cache()
        pred_answer_strings = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        question_strings = self.tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)

        # 2: calculate rewards
        n_latent_forward = latent_attention_mask.sum(dim=1)
        all_rewards = []
        all_accuracies = []
        all_advantages = []
        for sample_idx in range(batch_size):
            group_answers = pred_answer_strings[sample_idx * group_size : (sample_idx + 1) * group_size]
            group_n_latent_forward = n_latent_forward[sample_idx * group_size : (sample_idx + 1) * group_size]
            group_question_strings = question_strings[sample_idx * group_size : (sample_idx + 1) * group_size]
            gt_answer = gt_answers[sample_idx]
            rewards, accuracies = self.get_group_rewards_and_acc(
                pred_answers=group_answers, gt_answer=gt_answer, n_latent_forward=group_n_latent_forward, question_strings=group_question_strings
            )
            advantages = grpo.group_advantages(rewards)
            all_rewards.append(rewards)
            all_accuracies.append(accuracies)
            all_advantages.append(advantages)
        print(f"len(all_rewards): {len(all_rewards)}")
        print(f"len(all_accuracies): {len(all_accuracies)}")
        print(f"len(all_advantages): {len(all_advantages)}")
        print(f"group_size: {group_size}")
        print(f"batch_size: {batch_size}")
    
        rewards = torch.cat(all_rewards, dim=0)
        accuracies = torch.cat(all_accuracies, dim=0)
        advantages = torch.cat(all_advantages, dim=0)

        # 3: calculate logprobs
        experience = grpo.Experience(
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            latent_inputs_embeds=latent_inputs_embeds,
            latent_attention_mask=latent_attention_mask,
            answer_input_ids=pred_ids,
            answer_attention_mask=pred_ids.ne(self.tokenizer.pad_token_id).long(),
            n_latent_forward=n_latent_forward.unsqueeze(1),
            rewards=rewards,
            accuracies=accuracies,
            advantages=advantages,
        )

        latent_logprobs, answer_logprobs = self.get_logprobs(experience)
        experience.latent_logprobs = latent_logprobs
        experience.answer_logprobs = answer_logprobs

        return experience

    def get_group_rewards_and_acc(
        self, pred_answers: List[str], gt_answer: str, n_latent_forward: torch.Tensor, question_strings: List[str]=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print(f"question_strings: {question_strings}")
        print(f"pred_answers: {pred_answers}")
        print(f"gt_answer: {gt_answer}")
        print(f"n_latent_forward: {n_latent_forward}")
        unique_question_strings = list(set(question_strings))
        print(f"number of unique question strings: {len(unique_question_strings)}")
        num_unique_question_strings = len(unique_question_strings)
        if num_unique_question_strings > 1:
            x = 1/0

        rl_config = self.model_kwargs.rl_config
        group_size = len(pred_answers)

        accuracies = torch.zeros(size=(group_size, 1), device=self.device, dtype=torch.float32)
        for i, pred_answer in enumerate(pred_answers):
            # pred_a = self.extract_answer_from_output(pred_answer)
            # accuracies[i] = self.verify_answer(gt_answer=gt_answer, pred_answer=pred_a)
            model_response = pred_answer.split("In summary:")[-1].strip()
            model_response = model_response.split("In summary,")[-1].strip()
            model_response = model_response.split("Detailed Plan:")[-1].strip()
            question_string = question_strings[i]
            # accuracies[i] = self.verify_answer(gt_answer=gt_answer, pred_answer=pred_a)

        rewards = accuracies.detach().clone()
        if rl_config.punish_latent_length:
            rewards /= n_latent_forward.unsqueeze(1)

        # x = 1/0

        return rewards, accuracies

    # def get_logprobs(self, e: grpo.Experience):
    #     question_length = e.question_input_ids.shape[1]
    #     latent_length = e.latent_inputs_embeds.shape[1]
    #     answer_length = e.answer_input_ids.shape[1]

    #     question_inputs_embeds = self.embedding(e.question_input_ids)
    #     answer_inputs_embeds = self.embedding(e.answer_input_ids)

    #     all_inputs_embeds = torch.cat([question_inputs_embeds, e.latent_inputs_embeds, answer_inputs_embeds], dim=1)
    #     all_attention_mask = torch.cat(
    #         [e.question_attention_mask, e.latent_attention_mask, e.answer_attention_mask], dim=1
    #     )

    #     all_position_ids = get_position_ids_from_attention_mask(all_attention_mask)
    #     print(f"all_inputs_embeds.shape: {all_inputs_embeds.shape}")
    #     print(f"all_attention_mask.shape: {all_attention_mask.shape}")
    #     print(f"all_position_ids.shape: {all_position_ids.shape}")
    #     print(f"question_length: {question_length}")
    #     print(f"latent_length: {latent_length}")
    #     print(f"answer_length: {answer_length}")
    #     print(f"e.n_latent_forward.shape: {e.n_latent_forward.shape}")
    #     # instead of all at once, let's do it one by one
    #     all_last_hidden_states = []
    #     all_answer_logits = []
    #     all_output_logits = []
    #     for i in range(all_inputs_embeds.shape[0]):
    #         inputs_embeds = all_inputs_embeds[i, :, :].unsqueeze(0)
    #         attention_mask = all_attention_mask[i, :].unsqueeze(0)
    #         position_ids = all_position_ids[i, :].unsqueeze(0)
    #         print(f"{i}; inputs_embeds.shape: {inputs_embeds.shape}")
    #         print(f"{i}; attention_mask.shape: {attention_mask.shape}")
    #         print(f"{i}; position_ids.shape: {position_ids.shape}")
    #         all_outputs = self.llm.forward(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             output_hidden_states=True,
    #         )
    #         print(f"{i}; all_outputs.logits.shape: {all_outputs.logits.shape}")
            
    #     # all_outputs = self.llm.forward(
    #     #     inputs_embeds=all_inputs_embeds,
    #     #     attention_mask=all_attention_mask,
    #     #     position_ids=all_position_ids,
    #     #     output_hidden_states=True,
    #     # )
        
    #         last_hidden_states_for_latents = all_outputs.hidden_states[-1][
    #             :, question_length - 1 : question_length + latent_length - 1
    #         ]

    #         all_last_hidden_states.append(last_hidden_states_for_latents)
    #         all_output_logits.append(all_outputs.logits)
    #         answer_logits = all_outputs.logits[:, -answer_length:-1, :]
    #         all_answer_logits.append(answer_logits)
            
    #         print(f"last_hidden_states_for_latents.shape: {last_hidden_states_for_latents.shape}")
    #         print(f"output_logits.shape: {all_outputs.logits.shape}")
    #         print(f"answer_logits.shape: {answer_logits.shape}")

    #     all_last_hidden_states = torch.cat(all_last_hidden_states, dim=0)
    #     all_output_logits = torch.cat(all_output_logits, dim=0)
    #     all_answer_logits = torch.cat(all_answer_logits, dim=0)
    #     print(f"all_last_hidden_states.shape: {all_last_hidden_states.shape}")
    #     print(f"all_output_logits.shape: {all_output_logits.shape}")
    #     print(f"all_answer_logits.shape: {all_answer_logits.shape}")

    #     distributions = self.latent_policy.forward(all_last_hidden_states)
    #     latent_logprobs = distributions.log_prob(e.latent_inputs_embeds / self.embeds_std).mean(dim=-1)

    #     # logits for end of think
    #     logits_for_eol = []
    #     for b, latent_length in enumerate(e.n_latent_forward):
    #         logits_for_eol.append(all_output_logits[b, question_length + latent_length - 1])
    #     logits_for_eol = torch.stack(logits_for_eol, dim=0)
    #     print(f"logits_for_eol.shape: {logits_for_eol.shape}")
    #     print(f"all_answer_logits.shape: {all_answer_logits.shape}")
    #     # answer_logprobs
    #     answer_logits = torch.cat([logits_for_eol, all_answer_logits], dim=1)
    #     answer_logprobs = F.log_softmax(answer_logits, dim=-1)
    #     answer_logprobs = answer_logprobs.gather(dim=-1, index=e.answer_input_ids.unsqueeze(-1)).squeeze(-1)
    #     print(f"latent_logprobs.shape: {latent_logprobs.shape}")
    #     print(f"answer_logprobs.shape: {answer_logprobs.shape}")
    #     # return latent_logprobs, answer_logprobs
    #     return latent_logprobs, answer_logprobs

    # -- rl ends --#


    def get_logprobs(self, e: grpo.Experience):
        # lens
        B = e.question_input_ids.size(0)
        question_length = e.question_input_ids.size(1)
        latent_length   = e.latent_inputs_embeds.size(1)
        answer_length   = e.answer_input_ids.size(1)

        # embed (these are modest-sized)
        question_inputs_embeds = self.embedding(e.question_input_ids)
        answer_inputs_embeds   = self.embedding(e.answer_input_ids)

        # containers to accumulate minimal things
        latent_hiddens = []   # [sum(latent_len_i), D] after cat
        eol_hiddens    = []   # [B, D]
        ans_hiddens    = []   # [sum(answer_len_i-1), D]
        ans_targets    = []   # same length as ans_hiddens (flattened targets)
        # Note: we do not store any logits tensors

        # Per-sample forward to keep activation memory low
        for i in range(B):
            inputs_embeds = torch.cat([
                question_inputs_embeds[i:i+1], 
                e.latent_inputs_embeds[i:i+1], 
                answer_inputs_embeds[i:i+1]
            ], dim=1)                                 # [1, T, D]

            attn_mask = torch.cat([
                e.question_attention_mask[i:i+1],
                e.latent_attention_mask[i:i+1],
                e.answer_attention_mask[i:i+1]
            ], dim=1)                                  # [1, T]

            pos_ids = get_position_ids_from_attention_mask(attn_mask)

            # Only last_hidden_state; avoids huge hidden_states list + logits
            out = self.llm.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                use_cache=False,              # be explicit
                output_hidden_states=False,   # critical for memory
                return_dict=True,
            )
            last_h = out.last_hidden_state    # [1, T, D]; keep graph

            # 1) latent positions: we need states that *predict* each latent token
            #    Those are the states right before each latent token -> shift by -1
            #    range: [q_len-1, q_len+latent_len-2]
            latent_states = last_h[:, (question_length-1):(question_length+latent_length-1), :]  # [1, L_latent, D]
            latent_hiddens.append(latent_states.squeeze(0))  # [L_latent, D]

            # 2) EOL position (single token)
            this_latent_len = int(e.n_latent_forward[i].item())  # scalar
            eol_pos = question_length + this_latent_len - 1
            eol_hiddens.append(last_h[:, eol_pos, :].squeeze(0))  # [D]

            # 3) Answer positions that *predict* answer tokens:
            # we need the states for tokens that predict answer ids:
            # positions: last_h[:, -answer_length:-1, :]
            if answer_length > 1:
                ans_pred_states = last_h[:, -answer_length:-1, :].squeeze(0)   # [(A-1), D]
                ans_hiddens.append(ans_pred_states)

                # targets are the next tokens in the answer
                ans_targets.append(e.answer_input_ids[i, 1:])                  # [(A-1)]
            # else: no answer tokens to score besides EOL

            # free per-sample big tensors quickly
            del out, last_h, inputs_embeds, attn_mask, pos_ids

        # --- stack & compute logprobs with tiny projections ---
        # Latent logprobs: distribution over embeds (no vocab blow-up)
        latent_hiddens = torch.cat(latent_hiddens, dim=0)                           # [sum L_lat, D]
        distributions = self.latent_policy.forward(latent_hiddens)                  # keeps grad
        # compare against provided latent_inputs_embeds (flattened the same way)
        latent_targets = torch.cat([e.latent_inputs_embeds[i, :, :] 
                                    for i in range(B)], dim=0)                       # [sum L_lat, D]
        latent_logprobs = distributions.log_prob(latent_targets / self.embeds_std).mean(dim=-1)  # [sum L_lat]
        # reshape back to [B, L_latent]
        latent_logprobs = latent_logprobs.view(B, latent_length)

        # Answer logprobs:
        # First token logprob uses EOL hidden -> target is the *first* answer token
        eol_hiddens  = torch.stack(eol_hiddens, dim=0)                               # [B, D]
        first_targets = e.answer_input_ids[:, 0]                                     # [B]

        first_lp = chunked_project_and_log_softmax(
            eol_hiddens, self.llm.lm_head, first_targets, chunk_size=2048
        )                                                                             # [B]

        # Remaining answer tokens (if any)
        if ans_hiddens:
            ans_hiddens  = torch.cat(ans_hiddens, dim=0)                              # [sum(A_i-1), D]
            ans_targets  = torch.cat(ans_targets, dim=0).long()                       # [sum(A_i-1)]
            rest_lp = chunked_project_and_log_softmax(
                ans_hiddens, self.llm.lm_head, ans_targets, chunk_size=2048
            )                                                                         # [sum(A_i-1)]

            # stitch per-sample into [B, A]
            answer_logprobs = []
            cursor = 0
            for i in range(B):
                A = answer_length
                if A > 1:
                    n = A - 1
                    answer_logprobs.append(torch.cat([first_lp[i:i+1], rest_lp[cursor:cursor+n]], dim=0))  # [A]
                    cursor += n
                else:
                    answer_logprobs.append(first_lp[i:i+1])  # [1]
            answer_logprobs = torch.stack(answer_logprobs, dim=0)                     # [B, A]
        else:
            answer_logprobs = first_lp[:, None]                                        # [B, 1]

        return latent_logprobs, answer_logprobs
