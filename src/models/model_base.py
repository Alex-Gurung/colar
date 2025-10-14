import numpy as np
from collections import defaultdict, OrderedDict
from os.path import join as opj
from typing import List
import torch
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.models.llama import LlamaForCausalLM
from peft import LoraConfig, get_peft_model

from ..utils.utils import instantiate_from_config, get_timestamp, get_position_ids_from_attention_mask
from ..utils.log import JsonLogger, TextLogger

from liger_kernel.transformers import AutoLigerKernelForCausalLM, apply_liger_kernel_to_qwen2
from torch.nn.utils.rnn import pad_sequence
from deepspeed.ops.adam import DeepSpeedCPUAdam

class LitCoTModelBase(pl.LightningModule):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__()  # this must be called before save hparams

        self.all_config = all_config
        self.training_kwargs = training_kwargs
        self.model_kwargs = model_kwargs
        self.save_hyperparameters()

        # llm_path = opj(all_config.args.workspace_path, "models", "llms", model_kwargs.model_id)
        llm_path = model_kwargs.model_id
        ### IMPORTANT: replace the llm path to YOUR OWN llm path ###

        # tokenizer
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        if model_kwargs.get("set_pad_as_last_token", False):  # we don't use this, but might help
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.pad_token_id = len(self.tokenizer) - 1
        else:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # prompt templates
        if model_kwargs.get('chat_template'):
            self.question_template = \
"""<|start_header_id|>system<|end_header_id|>

Task:
Think, and then answer a quesiton, split thinkings and answer with ### token.

Example:
Question:[A question here] Let's think step by step:###[reasoning here]###Answer:[Your answer here]
<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {} Let's think step by step:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            self.question_template = "Question: {} Let's think step by step:"
        self.speed_template = "(Thinking speed: {})"
        # self.thinking_separator = "###"
        self.thinking_separator = "\n"
        # self.thinking_separator_id = self.tokenizer.convert_tokens_to_ids(self.thinking_separator)
        self.thinking_separator_id = self.tokenizer.encode(self.thinking_separator)[0]
        self.steps_template = "{}"
        self.answer_template = "Answer:{}"

        # llm
        model_class = AutoLigerKernelForCausalLM
        # apply_liger_kernel_to_qwen2(
        #     rope=True,
        #     swiglu=True,
        #     # cross_entropy=True,
        #     cross_entropy=False,
        #     # fused_linear_cross_entropy=False,
        #     fused_linear_cross_entropy=True,
        #     # rms_norm=False
        #     rms_norm=True
        # )

        # model_class = AutoModelForCausalLM
        self.llm = model_class.from_pretrained(llm_path, 
            attn_implementation="flash_attention_3", 
            trust_remote_code=True, 
            dtype=torch.bfloat16)

        if not model_kwargs.get("set_pad_as_last_token", False):  # not used, but might help
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.embedding = self.llm.get_input_embeddings()

        # lora (applied after all the configurations readied)
        if model_kwargs.do_lora:
            self.llm = get_peft_model(self.llm, peft_config=LoraConfig(**model_kwargs.lora_config))
            self.llm.print_trainable_parameters()


        self.llm.gradient_checkpointing_enable() 
        # log
        self.sample_logs = defaultdict(dict)

    def configure_optimizers(self):
        kwargs = self.all_config.model.training_kwargs

        self.trainable_parameter_names = []
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.trainable_parameter_names.append(name)
                trainable_params.append(param)

        # optimizer = instantiate_from_config(kwargs.optimizer, extra_kwargs={"params": trainable_params})
        # optimizer = instantiate_from_config(kwargs.optimizer, trainable_params)
        optimizer = DeepSpeedCPUAdam(trainable_params, lr=kwargs.optimizer.lr, weight_decay=kwargs.optimizer.weight_decay)

        if not kwargs.get("use_scheduler", False):
            return {"optimizer": optimizer}
        else:
            scheduler_config = kwargs.scheduler

        if scheduler_config.target == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_config.warmup_steps,
                num_training_steps=scheduler_config.num_training_steps,
            )
        else:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=scheduler_config.warmup_steps)

        self.lr_scheduler = scheduler
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_fit_start(self):
        self.text_logger = TextLogger(self, log_file_name="log", tmp_log=self.all_config.args.no_log)
        self.text_logger.log(f"Start training with model:\n {self}\nconfig:\n{self.all_config}")
        self.json_logger = JsonLogger(self, log_file_name="train", tmp_log=self.all_config.args.no_log)

        # Training setup verification box
        print("\n" + "🚀" + "="*78 + "🚀")
        print("🔥 TRAINING STARTING")
        print("🚀" + "="*78 + "🚀")
        print(f"🎯 Model: {type(self).__name__}")
        print(f"🏋️  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print(f"🧊 Frozen params: {sum(p.numel() for p in self.parameters() if not p.requires_grad):,}")
        print(f"⚡ Device count: {self.trainer.num_devices}")
        print(f"📊 Effective batch size: {self.all_config.dataloader.batch_size * self.trainer.accumulate_grad_batches}")
        print(f"🎲 Gradient accumulation: {self.trainer.accumulate_grad_batches}")
        print("🚀" + "="*78 + "🚀" + "\n")

        return super().on_fit_start()

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        log_dict = self.get_log_dict(batch, "train", batch_idx, dataloader_idx=dataloader_idx)
        log_dict.update(self.extra_training_step(batch=batch, batch_idx=batch_idx))
        self.log_dict(log_dict, sync_dist=True, prog_bar=True, batch_size=self.all_config.dataloader.batch_size)
        return log_dict["train/total_loss"]

    def get_log_dict(self, batch, split, batch_idx, dataloader_idx):
        log_dict = self.forward(batch=batch)
        log_dict = {f"{split}/{k}": v for k, v in log_dict.items()}
        return log_dict
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def extra_training_step(self, batch, batch_idx):
        return {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Compute validation loss using the same forward pass as training
        with torch.no_grad():
            # Get loss from forward pass (same as training but no gradients)
            if hasattr(self, 'forward'):
                loss_output = self.forward(batch=batch)
                val_loss = loss_output.get("total_loss", 0.0)
                val_ce_loss = loss_output.get("ce_loss", 0.0)
                val_embed_loss = loss_output.get("embed_modeling_loss", 0.0)
                val_entropy = loss_output.get("entropy", 0.0)

                # Log validation losses
                loss_dict = {
                    "monitor": val_loss,
                    "val/loss": val_loss,
                    "val/ce_loss": val_ce_loss,
                    "val/embed_modeling_loss": val_embed_loss,
                    "val/entropy": val_entropy
                }
            else:
                loss_dict = {}

        # UNCOMMENT FOR RL TRAINING
        # Evaluate the generation of the model on the validation data
        if self.model_kwargs.do_rl:
            generation_dict = self.eval_generation(batch=batch, split="val", batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        else:
            generation_dict = {}

        # Combine loss and generation metrics
        log_dict = {**loss_dict, **generation_dict}
        # log_dict = loss_dict

        self.log_dict(
            log_dict,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=len(batch["idx"]),
        )
        return log_dict

    def on_validation_epoch_end(self):
        self.json_logger.log(self.sample_logs)
        return super().on_validation_epoch_end()

    def on_save_checkpoint(self, checkpoint):
        # only save the trainable parameters
        new_state_dict = OrderedDict()
        for k in self.trainable_parameter_names:
            new_state_dict[k] = checkpoint["state_dict"][k]
        checkpoint["state_dict"] = new_state_dict

    def on_test_start(self):
        self.text_logger = TextLogger(self, log_file_name="log")
        self.text_logger.log(f"Start testing with model:\n{self}\nconfig:\n{self.all_config}.")
        self.json_logger = JsonLogger(self, log_file_name=f"test_{get_timestamp()}")
        return super().on_test_start()

    def test_step(self, batch, batch_idx=None, dataloader_idx=0):
        log_dict = self.eval_generation(batch=batch, split="test", batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        self.log_dict(
            log_dict,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=len(batch["idx"]),
        )

    def on_test_end(self):
        merged_logs = self._gather_sample_logs(self.sample_logs)
        if merged_logs is not None:
            self.json_logger.log(merged_logs)
        return super().on_test_end()

    def _gather_sample_logs(self, local_logs):
        if not (dist.is_available() and dist.is_initialized()):
            return local_logs

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_logs)

        if rank != 0:
            return None

        merged = {}
        for shard in gathered:
            if not shard:
                continue
            for key, value in shard.items():
                if key not in merged:
                    merged[key] = value
                else:
                    for subkey, subval in value.items():
                        if subkey not in merged[key]:
                            merged[key][subkey] = subval
                        elif isinstance(subval, list):
                            merged[key][subkey].extend(subval)
                        else:
                            merged[key][subkey] = subval
        return merged

    # -- basic methods implemenration ends --#

    # ++ sft implementation begins ++#
    def prepare_inputs(self, text_list, padding_side, part, prefix="", suffix=""):
        if isinstance(text_list, str):
            text_list = [text_list]

        batch_size = len(text_list)
        if isinstance(prefix, str):
            prefix = [prefix] * batch_size
        if isinstance(suffix, str):
            suffix = [suffix] * batch_size

        # base_template = getattr(self, f"{part}_template")
        # text_list = [prefix[i] + base_template.format(text) + suffix[i] for i, text in enumerate(text_list)]
        text_list = [prefix[i] + text + "\n" + suffix[i] for i, text in enumerate(text_list)]
        # print(f"text_list:")
        # for text in text_list:
        #     print(text)
        #     print("--------------------"*10)
        #     print("--------------------"*10)
        # x = 1/0
        inputs = self.tokenizer.batch_encode_plus(
            text_list, return_tensors="pt", add_special_tokens=False, padding="longest", padding_side=padding_side
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        return input_ids, attention_mask

    # -- sft training ends --#

    # ++ evaluation begins ++#
    @torch.no_grad()
    def text_generate(self, questions: List[str]):
        answer_generation_config = self.model_kwargs.answer_generation_config
        batch_size = len(questions)
        # question: [pad, question] or [pad, question, speed, ###]
        input_ids, attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=(self.speed_template.format(1) + self.thinking_separator)
            if self.model_kwargs.sft_method == "cot"
            else "",
        )
        outputs = self.llm.generate(inputs=input_ids, attention_mask=attention_mask, **answer_generation_config)[
            :, input_ids.shape[1] :
        ]
        n_latent_forward = []
        for b in range(batch_size):
            try:
                o = outputs[b].tolist()
                if self.thinking_separator_id in o:
                    length = o.index(self.thinking_separator_id) - 1
                else:
                    length = o.index(self.tokenizer.encode(":", add_special_tokens=False)[0]) - 3
            except ValueError:  # no thinking separator
                length = outputs[b].shape[0]
            n_latent_forward.append(length)
        return outputs, torch.tensor(n_latent_forward, device=self.device, dtype=torch.long).unsqueeze(1)

    @torch.no_grad()
    def latent_generate(
        self,
        questions,
        rl_mode=False,
        return_latent_hidden_states=False,  # for evaluation
    ):
        latent_generation_config = self.model_kwargs.latent_generation_config
        answer_generation_config = self.model_kwargs.answer_generation_config
        max_n_latent_forward = latent_generation_config.max_n_latent_forward
        latent_temperature = latent_generation_config.get("latent_temperature", 1.0)

        batch_size = len(questions)
        n_latent_forward = torch.zeros(size=(batch_size, 1), device=self.device, dtype=torch.long)
        all_inputs_embeds = []

        # 1: question forward
        # question: [pad, question, speed, ###]
        speed = latent_generation_config["compression_factor"]
        suffix = self.speed_template.format(speed) + self.thinking_separator

        # print(f"🎬 QUESTION PROCESSING DEBUG:")
        # print(f"   Speed suffix: '{suffix}'")
        # first_question = questions[0]
        # print(f"   First question: '{first_question}'")
        # x = 1/0

        question_input_ids, question_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=suffix,
        )
        question_position_ids = get_position_ids_from_attention_mask(question_attention_mask)
        question_embeds = self.embedding(question_input_ids)

        need_hidden_states = return_latent_hidden_states

        # DEBUG: Check question processing
        # print(f"🎬 QUESTION PROCESSING DEBUG:")
        # print(f"   Speed suffix: '{suffix}'")
        # print(f"   Question input_ids shape: {question_input_ids.shape}")
        # print(f"   Question decoded (sample 0): '{self.tokenizer.decode(question_input_ids[0], skip_special_tokens=False)[:200]}...'")
        # print(f"   Question embedding norms: {question_embeds.norm(dim=-1)[0][:10].tolist()}")

        base_model = getattr(self.llm, "model", getattr(self.llm, "transformer", None))
        if base_model is None:
            raise RuntimeError("Could not find base model submodule (tried .model and .transformer)")

        outputs = base_model(
            inputs_embeds=question_embeds,
            attention_mask=question_attention_mask,
            position_ids=question_position_ids,
            output_hidden_states=need_hidden_states,
            use_cache=True,
            return_dict=True,
        )

        # DEBUG: Check initial LLM output
        # print(f"   Initial LLM logits shape: {outputs.logits.shape}")
        question_token_hidden = outputs.last_hidden_state[:, -1, :]
        initial_logits = self.llm.lm_head(question_token_hidden)
        initial_probs = torch.softmax(initial_logits[0], dim=-1)
        top_initial = torch.topk(initial_probs, 5)
        initial_tokens = [self.tokenizer.decode([tid]) for tid in top_initial.indices]
        # print(f"   Initial top tokens: {list(zip(initial_tokens, top_initial.values.tolist()))}")

        all_inputs_embeds.append(question_embeds)

        # 2: latent forward
        # 2.1: prepare containers to collect all the intermediate states
        all_attention_mask = question_attention_mask
        current_position_ids = question_position_ids[:, -1:]
        past_key_values = outputs.past_key_values
        is_done = torch.zeros(size=(batch_size, 1), device=self.device, dtype=torch.bool)

        return_latent_inputs_embeds = []
        return_latent_attention_mask = []
        all_latent_hidden_states = []

        # 2.2: latent generation
        for _ in range(max_n_latent_forward):
            print(f"latent generation step {_}")
            if return_latent_hidden_states:
                all_latent_hidden_states.append(torch.stack(outputs.hidden_states, dim=1)[:, :, -1:, :])
                token_state = outputs.hidden_states[-1][:, -1:, :]
            else:
                token_state = outputs.last_hidden_state[:, -1:, :]

            distributions = self.latent_policy.forward(
                token_state, temperature=latent_temperature
            )  # outputs from last loops

            # FIX: Scale latent embeddings to match question embedding norms
            # Instead of using the hardcoded embeds_std (0.0136), use the actual
            # norm of the question embeddings to prevent norm discontinuity
            if _ == 0:  # Calculate target norm only once
                target_embedding_norm = question_embeds.norm(dim=-1).mean().item()
            #     print(f"🔧 EMBEDDING NORM FIX:")
            #     print(f"   Old scaling (embeds_std): {self.embeds_std}")
            #     print(f"   New scaling (question norm): {target_embedding_norm:.4f}")
            #     print(f"   Ratio change: {target_embedding_norm / self.embeds_std:.2f}x")

            current_inputs_embeds = distributions.rsample() * target_embedding_norm
            return_latent_inputs_embeds.append(current_inputs_embeds)
            all_inputs_embeds.append(current_inputs_embeds)

            not_is_done_long = (~is_done).long()
            all_attention_mask = torch.cat(
                [
                    all_attention_mask,
                    not_is_done_long,
                ],
                dim=1,
            )
            return_latent_attention_mask.append(not_is_done_long)

            current_position_ids = current_position_ids + not_is_done_long
            n_latent_forward += not_is_done_long

            # print(f"inside latent generate, input_embeds shape: {current_inputs_embeds.shape}")
            # print(f"inside latent generate, past_key_values: {past_key_values}")

            outputs = base_model(
                inputs_embeds=current_inputs_embeds,
                attention_mask=all_attention_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                output_hidden_states=need_hidden_states,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            # print(f"past_key_values: {past_key_values}")

            current_last_hidden = outputs.last_hidden_state[:, -1, :]
            last_logits = self.llm.lm_head(current_last_hidden)
            probs = torch.softmax(last_logits / latent_generation_config.get("eol_temperature", 1.0), dim=-1)
            batch_next_token = torch.multinomial(probs, num_samples=1)  # [n, 1]

            # DEBUG: Show what tokens are being predicted
            # print(f"🎯 LATENT STEP {_} DEBUG:")
            predicted_tokens = self.tokenizer.batch_decode(batch_next_token, skip_special_tokens=False)
            # print(f"   Predicted next tokens: {predicted_tokens}")
            # print(f"   thinking_separator_id: {self.thinking_separator_id} ('{self.thinking_separator}')")

            # Show top 10 most likely tokens for first sample
            top_k = torch.topk(probs[0], 10)
            top_token_ids = top_k.indices.tolist()
            top_token_probs = top_k.values.tolist()
            top_token_strs = [self.tokenizer.decode([tid]) for tid in top_token_ids]
            # print(f"   Top 10 tokens for sample 0:")
            # for i, (token_str, prob, tid) in enumerate(zip(top_token_strs, top_token_probs, top_token_ids)):
                # marker = " ← SELECTED" if tid == batch_next_token[0].item() else ""
                # print(f"     {i+1:2d}. '{token_str}' (id={tid}) prob={prob:.4f}{marker}")

            is_eol = batch_next_token == self.thinking_separator_id
            is_done = is_done | is_eol
            # print(f"   is_eol: {is_eol.tolist()}")
            # print(f"   is_done: {is_done.tolist()}")
            print(f"   is_done: {is_done.tolist()}")
            if is_done.all():
                break
            # if _ > 10:
            #     break

        # all_latent_hidden_states = torch.cat(all_latent_hidden_states, dim=2)

        # 3: add end of thinking
        end_of_thinking_ids = (
            torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long) * self.thinking_separator_id
        )
        end_of_thinking_embeds = self.embedding(end_of_thinking_ids)
        all_inputs_embeds.append(end_of_thinking_embeds)
        all_attention_mask = torch.cat(
            [
                all_attention_mask,
                torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long),
            ],
            dim=1,
        )

        # 4: answer generation
        all_inputs_embeds = torch.cat(all_inputs_embeds, dim=1)

        # DEBUG: Show what's being fed to final generation
        # print(f"🚀 FINAL GENERATION DEBUG:")
        # print(f"   Input embeddings shape: {all_inputs_embeds.shape}")
        # print(f"   Input attention_mask shape: {all_attention_mask.shape}")
        # print(f"   Input sequence length: {all_inputs_embeds.shape[1]}")
        # print(f"   answer_generation_config: {answer_generation_config}")

        # DEEP DEBUG: Check the structure of input embeddings
        # print(f"🔍 DEEP EMBEDDING ANALYSIS:")
        # print(f"   Question length: {question_input_ids.shape[1]}")
        # print(f"   Latent embeddings count: {len(return_latent_inputs_embeds)}")
        # print(f"   End-of-thinking: 1 token")

        # Check embedding norms at different positions
        emb_norms = all_inputs_embeds.norm(dim=-1)[0]  # First sample
        # print(f"   Embedding norms (first 10): {emb_norms[:10].tolist()}")
        # print(f"   Embedding norms (last 10): {emb_norms[-10:].tolist()}")
        # print(f"   Question embeddings norm mean: {emb_norms[:question_input_ids.shape[1]].mean().item():.4f}")
        if len(return_latent_inputs_embeds) > 0:
            latent_start = question_input_ids.shape[1]
            latent_end = latent_start + len(return_latent_inputs_embeds)
            # print(f"   Latent embeddings norm mean: {emb_norms[latent_start:latent_end].mean().item():.4f}")

        # Check attention mask pattern
        attn_first = all_attention_mask[0]
        # print(f"   Attention mask sum: {attn_first.sum().item()}/{attn_first.shape[0]}")
        # print(f"   First 20 attention: {attn_first[:20].tolist()}")
        # print(f"   Last 20 attention: {attn_first[-20:].tolist()}")

        # Check for any NaN or inf values
        # if torch.isnan(all_inputs_embeds).any():
        #     print("   ❌ WARNING: NaN values in input embeddings!")
        # if torch.isinf(all_inputs_embeds).any():
        #     print("   ❌ WARNING: Inf values in input embeddings!")

        # IMPORTANT: Need to verify generate() behavior with inputs_embeds
        # Testing hypothesis: generate() with inputs_embeds returns ONLY new tokens
        # OR does it return [input_length + new_tokens]?
        pred_ids = self.llm.generate(
            inputs_embeds=all_inputs_embeds, attention_mask=all_attention_mask, **answer_generation_config
        )

        # print(f"   OUTPUT pred_ids shape: {pred_ids.shape}")
        input_length = all_inputs_embeds.shape[1]
        output_length = pred_ids.shape[1]

        if pred_ids.shape[1] > 0:
            decoded_full = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
            # print(f"   Full output (sample 0): '{decoded_full[0][:200]}{'...' if len(decoded_full[0]) > 200 else ''}'")
            print(f"   Full output (sample 0): {decoded_full[0]}")

            # Key test: Does output length > input length?
            if output_length > input_length:
                print(f"   CASE 1: Output includes input + new tokens ({output_length} > {input_length})")
                new_tokens = pred_ids[:, input_length:]
                if new_tokens.shape[1] > 0:
                    decoded_new = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
                    print(f"   NEW tokens only: '{decoded_new[0][:200]}{'...' if len(decoded_new[0]) > 200 else ''}'")
            elif output_length == input_length:
                print(f"   CASE 2: No new tokens generated (output == input length)")
            else:
                print(f"   CASE 3: Output is ONLY new tokens ({output_length} tokens)")
                print(f"   These ARE the new tokens: '{decoded_full[0][:200]}{'...' if len(decoded_full[0]) > 200 else ''}'")
        else:
            print(f"   ERROR: Empty output!")

        if rl_mode:
            res = (
                question_input_ids,
                question_attention_mask,
                torch.cat(return_latent_inputs_embeds, dim=1),
                torch.cat(return_latent_attention_mask, dim=1),
                torch.cat(
                    [
                        torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long)
                        * self.thinking_separator_id,
                        pred_ids,
                    ],
                    dim=1,
                ),  # cat '###' with answer
            )
        elif return_latent_hidden_states:
            res = (
                pred_ids,
                n_latent_forward,
                all_latent_hidden_states,
            )
        else:
            res = (
                pred_ids,
                n_latent_forward,
            )

        return res

    @torch.no_grad()
    def fixed_length_latent_generate(self, questions: List[str]):
        max_n_latent_forward = 6  # this is the hyper-parameter used in Coconut and distill
        answer_generation_config = self.model_kwargs.answer_generation_config
        batch_size = len(questions)
        all_inputs_embeds = []

        # 1: question forward
        # question: [pad, question, speed, ###]
        question_input_ids, attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format("auto") + self.thinking_separator,
        )
        question_embeds = self.embedding(question_input_ids)
        outputs = self.llm.forward(
            inputs_embeds=question_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_inputs_embeds.append(question_embeds)

        # 2: latent forward
        for i in range(max_n_latent_forward):
            inputs_embeds = outputs.hidden_states[-1][:, -1:, :]  # outputs from last loops
            if hasattr(self, "latent_proj"):
                inputs_embeds = self.latent_proj(inputs_embeds)
            all_inputs_embeds.append(inputs_embeds)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long),
                ],
                dim=1,
            )
            outputs = self.llm.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # 3: add ### token
        end_of_thinking_ids = (
            torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long) * self.thinking_separator_id
        )
        end_of_thinking_embeds = self.embedding(end_of_thinking_ids)
        all_inputs_embeds.append(end_of_thinking_embeds)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long),
            ],
            dim=1,
        )

        # 4: answer generation
        all_inputs_embeds = torch.cat(all_inputs_embeds, dim=1)
        pred_ids = self.llm.generate(
            inputs_embeds=all_inputs_embeds, attention_mask=attention_mask, **answer_generation_config
        )
        return pred_ids, torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long) * max_n_latent_forward

    def extract_answer_from_output(self, output_string: str):
        try:
            return output_string.strip('#').split(self.answer_template.format(""))[-1]
        except (ValueError, IndexError):
            return output_string

    def verify_answer(self, gt_answer: str, pred_answer: str) -> float:
        def get_pure_string(s: str):
            return s.strip("#\n ").rstrip(".").replace(",", "").lower()
        gt_answer = get_pure_string(gt_answer)
        pred_answer = get_pure_string(pred_answer)
        try:  # some answers may be like '10.0' but predicted as '10'
            gt_answer = float(gt_answer)
            pred_answer = float(pred_answer)
        except ValueError:
            pass
        return float(gt_answer == pred_answer)

    def eval_generation(self, batch, split="val", batch_idx=None, dataloader_idx=0):
        indices = batch["idx"].tolist()
        questions = batch["question"]
        answers = batch["answer"]
        steps = batch["steps"]

        # print(f"questions: {questions}")
        print(f"len(questions): {len(questions)}")
        print(f"len(answers): {len(answers)}")
        print(f"len(steps): {len(steps)}")
        print(f"len(indices): {len(indices)}")

        with torch.no_grad():
            # predict answers
            if (sft_method := self.model_kwargs.sft_method.lower()) == "colar":
                outputs_token_ids, n_latent_forward = self.latent_generate(questions=questions)
            elif sft_method == "coconut" or sft_method == "distill":
                outputs_token_ids, n_latent_forward = self.fixed_length_latent_generate(questions=questions)
            elif sft_method == "cot" or sft_method == "icot":
                outputs_token_ids, n_latent_forward = self.text_generate(questions=questions)
            else:
                raise NotImplementedError(f"Unknown sft_method: {sft_method}")

        output_strings = self.tokenizer.batch_decode(outputs_token_ids, skip_special_tokens=True)

        # metric and log
        all_acc = []
        all_output_length = []
        all_latent_forward = []
        all_reward = []
        for i, q, s, a, o_ids, o_str, nlf in zip(
            indices, questions, steps, answers, outputs_token_ids, output_strings, n_latent_forward
        ):
            if i not in self.sample_logs:
                self.sample_logs[i]["question"] = q
                self.sample_logs[i]["steps"] = s
                self.sample_logs[i]["answer"] = a

                self.sample_logs[i]["pred_answer"] = []
                self.sample_logs[i]["output_string"] = []
                self.sample_logs[i]["output_length"] = []
                self.sample_logs[i]["n_latent_forward"] = []
                self.sample_logs[i]["acc"] = []

            # check if in RL mode
            o_length = (o_ids != self.tokenizer.pad_token_id).sum().item()
            acc = 0
            reward = 0
            if self.model_kwargs.do_rl:
                # calculate reward
                pass
            else:
                pred_a = self.extract_answer_from_output(o_str)
                acc = self.verify_answer(gt_answer=a, pred_answer=pred_a)
                self.sample_logs[i]["pred_answer"].append(pred_a)
                self.sample_logs[i]["output_string"].append(o_str)
                self.sample_logs[i]["output_length"].append(o_length)
                self.sample_logs[i]["n_latent_forward"].append(nlf.item())
                self.sample_logs[i]["acc"].append(acc)

            all_acc.append(acc)
            all_output_length.append(o_length)
            all_latent_forward.append(nlf.item())
            all_reward.append(reward)

        acc_count = sum(all_acc)
        acc_forward_count = sum([a * alf for a, alf in zip(all_acc, all_latent_forward)])
        mean_n_latent_forward_on_acc = np.mean(acc_forward_count / (acc_count + 1e-8))
        mean_acc = np.mean(all_acc)
        mean_n_latent_forward = np.mean(all_latent_forward)
        mean_output_length = np.mean(all_output_length)

        res = {
            # "monitor": mean_acc,
            f"{split}/acc": mean_acc,
            f"{split}/n_latent_forward": mean_n_latent_forward,
            f"{split}/n_latent_forward_on_acc": mean_n_latent_forward_on_acc,
            f"{split}/output_length": mean_output_length,
        }
        return res

    # -- evaluation ends --#
