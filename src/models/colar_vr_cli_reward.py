"""
CoLaR model with unified reference model for both KL divergence and VR-CLI rewards

UNIFIED REFERENCE MODEL ARCHITECTURE:
1. GRPO Reference Policy: Built-in to GRPO, uses logprobs_old from experience collection time
   - Changes over time as we collect new experiences
   - Used for PPO clipping in policy gradient loss

2. Unified Frozen Reference Model: Single model serving dual purposes
   - Weights NEVER change during training (stable baseline)
   - Used for KL divergence: KL(π_current || π_reference) in PPO loss
   - Used for VR-CLI rewards: log P_ref(target | question, answer) - log P_ref(target | question)
   - More memory efficient than separate models

This implementation adds KL divergence loss to GRPO while using the same frozen
reference model for both KL computation and VR-CLI reward computation.

All modifications are tagged with: # UNIFIED-REFERENCE
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from .colar import LitCoLaR

logger = logging.getLogger(__name__)


class LitCoLaRVRCLIReward(LitCoLaR):  # UNIFIED-REFERENCE: Unified reference model for KL + VR-CLI
    """
    CoLaR model with unified reference model for both KL divergence and VR-CLI rewards.

    Key features:
    - Single frozen reference model serves dual purposes
    - KL divergence loss: KL(π_current || π_reference) added to GRPO loss
    - VR-CLI rewards: likelihood improvement measured against same reference
    - More memory efficient than separate reference models
    - Maintains GRPO's existing PPO clipping mechanism
    """

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(model_kwargs, training_kwargs, all_config)

        # UNIFIED-REFERENCE: Initialize unified reference model for both KL and VR-CLI
        self.unified_ref_config = model_kwargs.get('unified_reference_config', {})
        self.reference_model = None
        self.reference_tokenizer = None

        # UNIFIED-REFERENCE: KL divergence configuration
        self.kl_config = model_kwargs.get('kl_config', {})
        self.kl_coef = self.kl_config.get('init_kl_coef', 0.0)
        self.target_kl = self.kl_config.get('target_kl', 0.01)
        self.kl_adapt_rate = self.kl_config.get('adapt_rate', 1.1)

        # UNIFIED-REFERENCE: Load reference model if enabled
        if self.unified_ref_config.get('enabled', False):
            self._load_unified_reference_model()

    def _load_unified_reference_model(self):  # UNIFIED-REFERENCE: Load unified reference model for both KL and VR-CLI
        """
        Load the unified reference model for both KL divergence and VR-CLI reward computation.

        DUAL PURPOSE:
        - KL Divergence: Provides stable baseline for KL(π_current || π_reference) in PPO loss
        - VR-CLI Rewards: Measures likelihood improvement log P_ref(target | context)
        - Memory Efficient: Single model serves both purposes

        This model is completely frozen and never updated during training.
        """
        ref_model_name = self.unified_ref_config.get('model_name', self.model_kwargs.model_id)

        logger.info(f"Loading UNIFIED reference model for KL + VR-CLI: {ref_model_name}")

        try:
            # UNIFIED-REFERENCE: Load on CPU/separate device to save memory
            device_map = self.unified_ref_config.get('device_map', 'cpu')

            self.reference_model = AutoModelForCausalLM.from_pretrained(
                ref_model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
            )

            self.reference_tokenizer = AutoTokenizer.from_pretrained(
                ref_model_name,
                trust_remote_code=True,
            )

            # Add pad token if needed
            if self.reference_tokenizer.pad_token is None:
                self.reference_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.reference_model.resize_token_embeddings(len(self.reference_tokenizer))

            # UNIFIED-REFERENCE: FREEZE the reference model - weights never change
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False  # Explicitly freeze all parameters

            logger.info(f"Unified reference model loaded and FROZEN on {device_map}")
            logger.info("Reference model serves both KL divergence and VR-CLI rewards")

        except Exception as e:
            logger.error(f"Failed to load unified reference model: {e}")
            self.reference_model = None
            self.reference_tokenizer = None

    def get_group_rewards_and_acc(  # VR-CLI-INTEGRATION: Only modify reward computation
        self, pred_answers: List[str], gt_answer: str, n_latent_forward: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute VR-CLI style rewards while keeping everything else the same

        This is the ONLY method that needs to change - everything else in GRPO stays intact
        """
        group_size = len(pred_answers)
        rewards = torch.zeros(size=(group_size, 1), device=self.device, dtype=torch.float32)
        accuracies = torch.zeros(size=(group_size, 1), device=self.device, dtype=torch.float32)

        # UNIFIED-REFERENCE: Use VR-CLI rewards if available, otherwise fallback
        if self.reference_model is not None and self.unified_ref_config.get('vr_cli_enabled', False):
            for i, pred_answer in enumerate(pred_answers):
                try:
                    # VR-CLI-INTEGRATION: Compute VR-CLI style reward
                    vr_cli_reward = self._compute_vr_cli_reward(pred_answer, gt_answer)
                    rewards[i] = vr_cli_reward

                    # VR-CLI-INTEGRATION: Still compute binary accuracy for monitoring
                    pred_a = self.extract_answer_from_output(pred_answer)
                    accuracies[i] = self.verify_answer(gt_answer=gt_answer, pred_answer=pred_a)

                except Exception as e:
                    logger.warning(f"VR-CLI reward computation failed for sample {i}: {e}")
                    # Fallback to binary accuracy
                    pred_a = self.extract_answer_from_output(pred_answer)
                    acc = self.verify_answer(gt_answer=gt_answer, pred_answer=pred_a)
                    rewards[i] = acc
                    accuracies[i] = acc
        else:
            # VR-CLI-INTEGRATION: Fallback to original binary accuracy
            for i, pred_answer in enumerate(pred_answers):
                pred_a = self.extract_answer_from_output(pred_answer)
                accuracy = self.verify_answer(gt_answer=gt_answer, pred_answer=pred_a)
                rewards[i] = accuracy
                accuracies[i] = accuracy

        # VR-CLI-INTEGRATION: Apply length penalty if configured (unchanged)
        if self.model_kwargs.rl_config.get('punish_latent_length', False):
            rewards = rewards / (n_latent_forward.unsqueeze(1) + 1.0)

        return rewards, accuracies

    def _compute_vr_cli_reward(self, pred_answer: str, gt_answer: str) -> float:  # VR-CLI-INTEGRATION: Core VR-CLI reward
        """
        Compute VR-CLI style reward: improvement in reference model likelihood

        reward = log P_ref(target | question, answer) - log P_ref(target | question)
        """
        try:
            # VR-CLI-INTEGRATION: Get current question (you'll need to set this during rollout)
            current_question = getattr(self, '_current_question', '')
            if not current_question:
                logger.warning("No current question set for VR-CLI reward computation")
                return 0.0

            # VR-CLI-INTEGRATION: Define target based on your needs
            target = self._get_vr_cli_target(current_question, gt_answer)

            # VR-CLI-INTEGRATION: Compute baseline likelihood P(target | question)
            baseline_logprob = self._compute_reference_conditional_logprob(
                context=current_question,
                target=target
            )

            # VR-CLI-INTEGRATION: Compute likelihood with answer P(target | question, answer)
            context_with_answer = self._format_question_answer_context(current_question, pred_answer)
            answer_logprob = self._compute_reference_conditional_logprob(
                context=context_with_answer,
                target=target
            )

            # VR-CLI-INTEGRATION: VR-CLI reward is the likelihood improvement
            raw_reward = answer_logprob - baseline_logprob

            # VR-CLI-INTEGRATION: Scale and normalize
            reward_scale = self.vr_cli_config.get('reward_scale', 1.0)
            reward_clip = self.vr_cli_config.get('reward_clip', 5.0)

            scaled_reward = raw_reward * reward_scale
            clipped_reward = torch.clamp(torch.tensor(scaled_reward), -reward_clip, reward_clip).item()

            # VR-CLI-INTEGRATION: Apply sigmoid to get [0,1] range
            if self.vr_cli_config.get('use_sigmoid', True):
                final_reward = torch.sigmoid(torch.tensor(clipped_reward)).item()
            else:
                # Linear normalization to [0,1]
                final_reward = (clipped_reward + reward_clip) / (2 * reward_clip)

            return final_reward

        except Exception as e:
            logger.error(f"Error computing VR-CLI reward: {e}")
            return 0.0

    def _get_vr_cli_target(self, question: str, gt_answer: str) -> str:  # VR-CLI-INTEGRATION: Define VR-CLI target
        """
        Define what the reference model should become more confident about

        This is where you customize what "good" means for your task
        """
        target_type = self.vr_cli_config.get('target_type', 'correct_answer')

        if target_type == 'correct_answer':
            return f"The answer is {gt_answer}"

        elif target_type == 'answer_correctness':
            return "This answer is correct."

        elif target_type == 'reasoning_quality':
            return "This reasoning is sound and leads to the correct conclusion."

        elif target_type == 'custom':
            # VR-CLI-INTEGRATION: Use custom target from config
            custom_target = self.vr_cli_config.get('custom_target', '')
            return custom_target.format(question=question, gt_answer=gt_answer)

        else:
            return gt_answer

    def _format_question_answer_context(self, question: str, answer: str) -> str:  # VR-CLI-INTEGRATION: Format context
        """Format question and answer for reference model evaluation"""
        format_type = self.vr_cli_config.get('context_format', 'qa_format')

        if format_type == 'qa_format':
            return f"Question: {question}\nAnswer: {answer}"

        elif format_type == 'simple':
            return f"{question}\n{answer}"

        elif format_type == 'conversation':
            return f"Human: {question}\nAssistant: {answer}"

        elif format_type == 'reasoning':
            return f"Problem: {question}\nSolution: {answer}\nEvaluation:"

        else:
            return f"{question}\n{answer}"

    @torch.no_grad()
    def _compute_reference_conditional_logprob(self, context: str, target: str) -> float:  # UNIFIED-REFERENCE: Compute P(target|context)
        """
        Compute log P(target | context) using the unified reference model for VR-CLI rewards

        This model's weights never change during training - it serves as a stable
        baseline for measuring likelihood improvements.
        """
        try:
            # UNIFIED-REFERENCE: Tokenize context and target
            context_tokens = self.reference_tokenizer.encode(context, return_tensors='pt')
            full_text = context + " " + target
            full_tokens = self.reference_tokenizer.encode(full_text, return_tensors='pt')

            # UNIFIED-REFERENCE: Move to reference model device
            ref_device = next(self.reference_model.parameters()).device
            context_tokens = context_tokens.to(ref_device)
            full_tokens = full_tokens.to(ref_device)

            # UNIFIED-REFERENCE: Get target token positions
            context_len = context_tokens.shape[1]
            target_tokens = full_tokens[:, context_len:]

            if target_tokens.shape[1] == 0:
                logger.warning("Empty target tokens, returning 0.0")
                return 0.0

            # UNIFIED-REFERENCE: Forward pass through reference model
            with torch.no_grad():
                outputs = self.reference_model(full_tokens)
                logits = outputs.logits

            # UNIFIED-REFERENCE: Compute log probabilities for target tokens
            target_logits = logits[:, context_len-1:-1, :]  # Shift for next-token prediction
            target_logprobs = F.log_softmax(target_logits, dim=-1)

            # UNIFIED-REFERENCE: Gather log probabilities for actual target tokens
            target_logprobs_gathered = target_logprobs.gather(
                dim=-1,
                index=target_tokens.unsqueeze(-1)
            ).squeeze(-1)

            # UNIFIED-REFERENCE: Average log probability across tokens
            avg_logprob = target_logprobs_gathered.mean().item()

            return avg_logprob

        except Exception as e:
            logger.error(f"Error computing reference conditional log probability: {e}")
            return 0.0

    @torch.no_grad()
    def _compute_reference_logprobs(self, experience: 'grpo.Experience') -> Tuple[torch.Tensor, torch.Tensor]:  # UNIFIED-REFERENCE: Compute reference logprobs for KL
        """
        Compute reference model logprobs for KL divergence computation

        Returns:
            reference_latent_logprobs: Reference logprobs for latent tokens
            reference_answer_logprobs: Reference logprobs for answer tokens
        """
        if self.reference_model is None:
            logger.warning("Reference model not loaded, returning zeros for KL computation")
            return torch.zeros_like(experience.latent_logprobs), torch.zeros_like(experience.answer_logprobs)

        try:
            # UNIFIED-REFERENCE: Move to reference model device
            ref_device = next(self.reference_model.parameters()).device

            # UNIFIED-REFERENCE: Prepare inputs for reference model
            question_length = experience.question_input_ids.shape[1]
            latent_length = experience.latent_inputs_embeds.shape[1]
            answer_length = experience.answer_input_ids.shape[1]

            # Move experience to reference device
            question_input_ids = experience.question_input_ids.to(ref_device)
            latent_inputs_embeds = experience.latent_inputs_embeds.to(ref_device)
            answer_input_ids = experience.answer_input_ids.to(ref_device)
            question_attention_mask = experience.question_attention_mask.to(ref_device)
            latent_attention_mask = experience.latent_attention_mask.to(ref_device)
            answer_attention_mask = experience.answer_attention_mask.to(ref_device)

            # Get embeddings from reference model
            question_inputs_embeds = self.reference_model.get_input_embeddings()(question_input_ids)
            answer_inputs_embeds = self.reference_model.get_input_embeddings()(answer_input_ids)

            # Concatenate all inputs
            all_inputs_embeds = torch.cat([question_inputs_embeds, latent_inputs_embeds, answer_inputs_embeds], dim=1)
            all_attention_mask = torch.cat([question_attention_mask, latent_attention_mask, answer_attention_mask], dim=1)

            # UNIFIED-REFERENCE: Forward pass through reference model
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    inputs_embeds=all_inputs_embeds,
                    attention_mask=all_attention_mask,
                    output_hidden_states=True,
                )

            # UNIFIED-REFERENCE: Compute reference latent logprobs (using latent policy from main model)
            ref_last_hidden_states_for_latents = ref_outputs.hidden_states[-1][
                :, question_length - 1 : question_length + latent_length - 1
            ]
            ref_distributions = self.latent_policy.forward(ref_last_hidden_states_for_latents)
            ref_latent_logprobs = ref_distributions.log_prob(latent_inputs_embeds / self.embeds_std).mean(dim=-1)

            # UNIFIED-REFERENCE: Compute reference answer logprobs
            ref_logits_for_eol = []
            for b, latent_len in enumerate(experience.n_latent_forward):
                ref_logits_for_eol.append(ref_outputs.logits[b, question_length + latent_len - 1])
            ref_logits_for_eol = torch.stack(ref_logits_for_eol, dim=0)

            ref_answer_logits = torch.cat([ref_logits_for_eol, ref_outputs.logits[:, -answer_length:-1, :]], dim=1)
            ref_answer_logprobs = F.log_softmax(ref_answer_logits, dim=-1)
            ref_answer_logprobs = ref_answer_logprobs.gather(dim=-1, index=answer_input_ids.unsqueeze(-1)).squeeze(-1)

            # Move back to main device
            ref_latent_logprobs = ref_latent_logprobs.to(self.device)
            ref_answer_logprobs = ref_answer_logprobs.to(self.device)

            return ref_latent_logprobs, ref_answer_logprobs

        except Exception as e:
            logger.error(f"Error computing reference logprobs for KL: {e}")
            return torch.zeros_like(experience.latent_logprobs), torch.zeros_like(experience.answer_logprobs)

    def rollout(self, questions: List[str], gt_answers) -> 'grpo.Experience':  # UNIFIED-REFERENCE: Enhanced rollout with reference logprobs
        """
        Override rollout to compute both VR-CLI rewards and reference logprobs for KL divergence

        This computes:
        1. Standard GRPO experience with current policy logprobs
        2. Reference model logprobs for KL divergence computation
        3. Question context for VR-CLI reward computation
        """
        # UNIFIED-REFERENCE: Store current questions for VR-CLI reward computation
        self._current_questions = questions
        self._current_gt_answers = gt_answers

        # UNIFIED-REFERENCE: Call parent rollout to get standard experience
        experience = super().rollout(questions, gt_answers)

        # UNIFIED-REFERENCE: Compute reference logprobs for KL divergence if enabled
        if self.reference_model is not None and self.kl_config.get('enabled', False):
            logger.info("Computing reference logprobs for KL divergence")
            ref_latent_logprobs, ref_answer_logprobs = self._compute_reference_logprobs(experience)

            # Store reference logprobs in experience for KL computation
            experience.ref_latent_logprobs = ref_latent_logprobs
            experience.ref_answer_logprobs = ref_answer_logprobs
        else:
            # Set to None if KL not enabled
            experience.ref_latent_logprobs = None
            experience.ref_answer_logprobs = None

        return experience

    def get_group_rewards_and_acc_with_question(  # VR-CLI-INTEGRATION: Extended interface with question context
        self,
        pred_answers: List[str],
        gt_answer: str,
        n_latent_forward: torch.Tensor,
        question: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute VR-CLI rewards with explicit question context

        This version receives the question directly for cleaner usage
        """
        # VR-CLI-INTEGRATION: Set current question for VR-CLI reward computation
        self._current_question = question

        # VR-CLI-INTEGRATION: Call standard reward computation
        return self.get_group_rewards_and_acc(pred_answers, gt_answer, n_latent_forward)

    def grpo_loss(self, latent_logprobs, answer_logprobs, experience):  # UNIFIED-REFERENCE: Enhanced GRPO loss with KL divergence
        """
        Compute GRPO loss with optional KL divergence penalty

        This combines:
        1. Standard GRPO loss (PPO clipping)
        2. KL divergence penalty: KL(π_current || π_reference)
        """
        # UNIFIED-REFERENCE: Compute standard GRPO loss
        base_loss_dict = super().grpo_loss(latent_logprobs, answer_logprobs, experience)

        # UNIFIED-REFERENCE: Add KL divergence if enabled and reference logprobs available
        if (self.kl_config.get('enabled', False) and
            hasattr(experience, 'ref_latent_logprobs') and
            experience.ref_latent_logprobs is not None):

            # Compute KL divergence for latent tokens
            kl_latent = self._compute_kl_divergence(
                current_logprobs=latent_logprobs,
                reference_logprobs=experience.ref_latent_logprobs,
                attention_mask=experience.latent_attention_mask
            )

            # Compute KL divergence for answer tokens
            kl_answer = self._compute_kl_divergence(
                current_logprobs=answer_logprobs,
                reference_logprobs=experience.ref_answer_logprobs,
                attention_mask=experience.answer_attention_mask
            )

            # Total KL divergence
            total_kl = kl_latent + kl_answer
            kl_loss = self.kl_coef * total_kl

            # Add to total loss
            base_loss_dict["kl_loss"] = kl_loss
            base_loss_dict["kl_divergence"] = total_kl.detach()
            base_loss_dict["kl_coef"] = self.kl_coef
            base_loss_dict["total_loss"] = base_loss_dict["total_loss"] + kl_loss

            # Update KL coefficient if adaptive
            if self.kl_config.get('adaptive', True):
                self._update_kl_coefficient(total_kl.detach())

        else:
            # No KL loss
            base_loss_dict["kl_loss"] = torch.tensor(0.0, device=self.device)
            base_loss_dict["kl_divergence"] = torch.tensor(0.0, device=self.device)
            base_loss_dict["kl_coef"] = self.kl_coef

        return base_loss_dict

    def _compute_kl_divergence(self, current_logprobs: torch.Tensor, reference_logprobs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # UNIFIED-REFERENCE: KL computation
        """
        Compute KL divergence between current and reference log probabilities

        KL(π_current || π_reference) = Σ π_current * log(π_current / π_reference)
        """
        try:
            # Convert to probabilities
            current_probs = torch.exp(current_logprobs)

            # KL divergence: KL(p||q) = Σ p * log(p/q) = Σ p * (log(p) - log(q))
            kl_per_token = current_probs * (current_logprobs - reference_logprobs)

            # Mask and average
            if attention_mask is not None:
                kl_per_token = kl_per_token * attention_mask.float()
                kl_div = kl_per_token.sum() / attention_mask.sum().clamp(min=1)
            else:
                kl_div = kl_per_token.mean()

            return kl_div

        except Exception as e:
            logger.error(f"Error computing KL divergence: {e}")
            return torch.tensor(0.0, device=current_logprobs.device)

    def _update_kl_coefficient(self, kl_divergence: torch.Tensor):  # UNIFIED-REFERENCE: Adaptive KL coefficient
        """
        Update KL coefficient based on current KL divergence magnitude

        If KL is too high, increase penalty. If too low, decrease penalty.
        """
        try:
            kl_value = kl_divergence.item()

            if kl_value > self.target_kl:
                # KL too high, increase penalty
                self.kl_coef *= self.kl_adapt_rate
            else:
                # KL acceptable, decrease penalty
                self.kl_coef /= self.kl_adapt_rate

            # Clamp to reasonable bounds
            min_kl_coef = self.kl_config.get('min_kl_coef', 1e-8)
            max_kl_coef = self.kl_config.get('max_kl_coef', 1.0)
            self.kl_coef = torch.clamp(torch.tensor(self.kl_coef), min_kl_coef, max_kl_coef).item()

        except Exception as e:
            logger.error(f"Error updating KL coefficient: {e}")


# UNIFIED-REFERENCE: Helper functions for easy configuration
def create_unified_reference_config(
    enabled: bool = True,
    model_name: str = None,
    device_map: str = 'cpu',
    vr_cli_enabled: bool = True,
    target_type: str = 'correct_answer',
    context_format: str = 'qa_format',
    reward_scale: float = 1.0,
    reward_clip: float = 3.0,
    use_sigmoid: bool = True,
    custom_target: str = "",
    **kwargs
) -> Dict[str, Any]:
    """
    Create unified reference model configuration for both KL and VR-CLI

    Args:
        enabled: Whether to load reference model
        model_name: Reference model name (defaults to same as main model)
        device_map: Device for reference model ('cpu', 'cuda:0', etc.)
        vr_cli_enabled: Whether to use VR-CLI rewards
        target_type: What to use as VR-CLI target ('correct_answer', 'answer_correctness', etc.)
        context_format: How to format context ('qa_format', 'simple', 'conversation')
        reward_scale: Scale factor for raw rewards
        reward_clip: Clipping value for rewards before sigmoid
        use_sigmoid: Whether to apply sigmoid normalization
        custom_target: Custom target string (use {question} and {gt_answer} placeholders)

    Returns:
        Configuration dictionary
    """
    config = {
        'enabled': enabled,
        'model_name': model_name,
        'device_map': device_map,
        'vr_cli_enabled': vr_cli_enabled,
        'target_type': target_type,
        'context_format': context_format,
        'reward_scale': reward_scale,
        'reward_clip': reward_clip,
        'use_sigmoid': use_sigmoid,
        'custom_target': custom_target,
    }
    config.update(kwargs)
    return config

def create_kl_config(
    enabled: bool = True,
    init_kl_coef: float = 1e-6,
    target_kl: float = 0.01,
    adaptive: bool = True,
    adapt_rate: float = 1.1,
    min_kl_coef: float = 1e-8,
    max_kl_coef: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Create KL divergence configuration

    Args:
        enabled: Whether to use KL divergence penalty
        init_kl_coef: Initial KL coefficient (like OpenRLHF's init_kl_coef)
        target_kl: Target KL divergence value
        adaptive: Whether to adapt KL coefficient based on KL value
        adapt_rate: Rate at which to adapt KL coefficient
        min_kl_coef: Minimum KL coefficient
        max_kl_coef: Maximum KL coefficient

    Returns:
        Configuration dictionary
    """
    config = {
        'enabled': enabled,
        'init_kl_coef': init_kl_coef,
        'target_kl': target_kl,
        'adaptive': adaptive,
        'adapt_rate': adapt_rate,
        'min_kl_coef': min_kl_coef,
        'max_kl_coef': max_kl_coef,
    }
    config.update(kwargs)
    return config


# UNIFIED-REFERENCE: Example usage and configuration templates
EXAMPLE_UNIFIED_CONFIGS = {
    'kl_only': {
        'unified_reference_config': create_unified_reference_config(
            enabled=True,
            vr_cli_enabled=False,
            device_map='cpu'
        ),
        'kl_config': create_kl_config(
            enabled=True,
            init_kl_coef=1e-6,
            target_kl=0.01
        )
    },

    'vr_cli_only': {
        'unified_reference_config': create_unified_reference_config(
            enabled=True,
            vr_cli_enabled=True,
            target_type='correct_answer',
            device_map='cpu'
        ),
        'kl_config': create_kl_config(enabled=False)
    },

    'kl_plus_vr_cli': {
        'unified_reference_config': create_unified_reference_config(
            enabled=True,
            vr_cli_enabled=True,
            target_type='correct_answer',
            context_format='qa_format',
            device_map='cpu'
        ),
        'kl_config': create_kl_config(
            enabled=True,
            init_kl_coef=1e-6,
            target_kl=0.01,
            adaptive=True
        )
    },

    'openrlhf_equivalent': {
        'unified_reference_config': create_unified_reference_config(
            enabled=True,
            vr_cli_enabled=True,
            target_type='correct_answer',
            device_map='cpu'
        ),
        'kl_config': create_kl_config(
            enabled=True,
            init_kl_coef=1e-6,  # Matches OpenRLHF --init_kl_coef
            target_kl=0.01,
            adaptive=True,
            adapt_rate=1.1
        )
    },
}