"""
CoLaR model with reference model-based reward (VR-CLI style)

All modifications are tagged with: # VR-CLI-REWARD-MOD
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from .colar import LitCoLaR

logger = logging.getLogger(__name__)


class LitCoLaRReferenceReward(LitCoLaR):  # VR-CLI-REWARD-MOD: New class for reference model rewards
    """
    CoLaR model that uses a reference model to compute rewards based on
    how much the generated answer increases the reference model's likelihood
    of a target outcome.

    This implements the VR-CLI style reward where:
    reward = log P_ref(target | question, generated_answer) - log P_ref(target | question)
    """

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(model_kwargs, training_kwargs, all_config)

        # VR-CLI-REWARD-MOD: Initialize reference model components
        self.reference_config = model_kwargs.get('reference_reward_config', {})
        self.reference_model = None
        self.reference_tokenizer = None

        # VR-CLI-REWARD-MOD: Load reference model if specified
        if self.reference_config.get('enabled', False):
            self._load_reference_model()

    def _load_reference_model(self):  # VR-CLI-REWARD-MOD: Method to load reference model
        """Load the reference model for reward computation"""
        ref_model_name = self.reference_config.get('model_name', self.model_kwargs.model_id)

        logger.info(f"Loading reference model: {ref_model_name}")

        try:
            # Load reference model (keep on CPU or separate GPU to save memory)
            device_map = self.reference_config.get('device_map', 'cpu')

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

            # Set to eval mode
            self.reference_model.eval()

            logger.info(f"Reference model loaded successfully on {device_map}")

        except Exception as e:
            logger.error(f"Failed to load reference model: {e}")
            self.reference_model = None
            self.reference_tokenizer = None

    def get_group_rewards_and_acc(  # VR-CLI-REWARD-MOD: Override reward computation
        self, pred_answers: List[str], gt_answer: str, n_latent_forward: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards using reference model likelihood

        Args:
            pred_answers: List of generated answers
            gt_answer: Ground truth answer (used for accuracy monitoring)
            n_latent_forward: Number of latent reasoning steps

        Returns:
            rewards: Reference model based rewards
            accuracies: Binary accuracies for monitoring
        """
        group_size = len(pred_answers)
        rewards = torch.zeros(size=(group_size, 1), device=self.device, dtype=torch.float32)
        accuracies = torch.zeros(size=(group_size, 1), device=self.device, dtype=torch.float32)

        # VR-CLI-REWARD-MOD: Use reference model if available, fallback to binary accuracy
        if self.reference_model is not None and self.reference_config.get('enabled', False):
            for i, pred_answer in enumerate(pred_answers):
                try:
                    # Compute reference model reward
                    ref_reward = self._compute_reference_reward(pred_answer, gt_answer)
                    rewards[i] = ref_reward

                    # Still compute binary accuracy for monitoring
                    pred_a = self.extract_answer_from_output(pred_answer)
                    accuracies[i] = self.verify_answer(gt_answer=gt_answer, pred_answer=pred_a)

                except Exception as e:
                    logger.warning(f"Reference reward computation failed for sample {i}: {e}")
                    # Fallback to binary accuracy
                    pred_a = self.extract_answer_from_output(pred_answer)
                    acc = self.verify_answer(gt_answer=gt_answer, pred_answer=pred_a)
                    rewards[i] = acc
                    accuracies[i] = acc
        else:
            # VR-CLI-REWARD-MOD: Fallback to original binary accuracy if no reference model
            logger.warning("Reference model not available, falling back to binary accuracy")
            for i, pred_answer in enumerate(pred_answers):
                pred_a = self.extract_answer_from_output(pred_answer)
                accuracy = self.verify_answer(gt_answer=gt_answer, pred_answer=pred_a)
                rewards[i] = accuracy
                accuracies[i] = accuracy

        # VR-CLI-REWARD-MOD: Apply length penalty if configured
        if self.model_kwargs.rl_config.get('punish_latent_length', False):
            rewards = rewards / (n_latent_forward.unsqueeze(1) + 1.0)

        return rewards, accuracies

    def _compute_reference_reward(self, pred_answer: str, gt_answer: str) -> float:  # VR-CLI-REWARD-MOD: Core reference reward computation
        """
        Compute reward using reference model likelihood

        This implements: reward = log P_ref(target | question, answer) - log P_ref(target | question)

        Args:
            pred_answer: Generated answer from the model
            gt_answer: Ground truth answer (used to determine target)

        Returns:
            Reward based on reference model likelihood improvement
        """
        reward_type = self.reference_config.get('reward_type', 'target_likelihood')

        if reward_type == 'target_likelihood':
            return self._compute_target_likelihood_reward(pred_answer, gt_answer)
        elif reward_type == 'answer_quality':
            return self._compute_answer_quality_reward(pred_answer, gt_answer)
        elif reward_type == 'reasoning_quality':
            return self._compute_reasoning_quality_reward(pred_answer, gt_answer)
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")

    def _compute_target_likelihood_reward(self, pred_answer: str, gt_answer: str) -> float:  # VR-CLI-REWARD-MOD: Target likelihood reward
        """
        Compute reward based on how much the answer increases likelihood of target

        reward = log P_ref(target | question, answer) - log P_ref(target | question)
        """
        try:
            # VR-CLI-REWARD-MOD: Get current question from batch context
            current_question = self._get_current_question()

            # VR-CLI-REWARD-MOD: Define target based on configuration
            target = self._get_target_for_question(current_question, gt_answer)

            # VR-CLI-REWARD-MOD: Compute baseline likelihood P(target | question)
            baseline_logprob = self._compute_conditional_logprob(
                context=current_question,
                target=target
            )

            # VR-CLI-REWARD-MOD: Compute likelihood with answer P(target | question, answer)
            context_with_answer = self._format_context_with_answer(current_question, pred_answer)
            answer_logprob = self._compute_conditional_logprob(
                context=context_with_answer,
                target=target
            )

            # VR-CLI-REWARD-MOD: Reward is the improvement in likelihood
            reward = answer_logprob - baseline_logprob

            # VR-CLI-REWARD-MOD: Apply scaling and clipping
            reward_scale = self.reference_config.get('reward_scale', 1.0)
            reward_clip = self.reference_config.get('reward_clip', 5.0)

            reward = reward * reward_scale
            reward = torch.clamp(torch.tensor(reward), -reward_clip, reward_clip).item()

            # VR-CLI-REWARD-MOD: Convert to [0, 1] range using sigmoid
            if self.reference_config.get('use_sigmoid', True):
                reward = torch.sigmoid(torch.tensor(reward)).item()

            return reward

        except Exception as e:
            logger.error(f"Error computing target likelihood reward: {e}")
            return 0.0

    def _compute_answer_quality_reward(self, pred_answer: str, gt_answer: str) -> float:  # VR-CLI-REWARD-MOD: Answer quality reward
        """
        Compute reward based on reference model's assessment of answer quality

        This asks the reference model to directly evaluate the answer quality
        """
        try:
            current_question = self._get_current_question()

            # VR-CLI-REWARD-MOD: Create evaluation prompt
            eval_prompt = self._create_evaluation_prompt(current_question, pred_answer, gt_answer)

            # VR-CLI-REWARD-MOD: Get reference model's quality assessment
            quality_score = self._get_quality_assessment(eval_prompt)

            return quality_score

        except Exception as e:
            logger.error(f"Error computing answer quality reward: {e}")
            return 0.0

    def _compute_reasoning_quality_reward(self, pred_answer: str, gt_answer: str) -> float:  # VR-CLI-REWARD-MOD: Reasoning quality reward
        """
        Compute reward based on quality of reasoning steps
        """
        try:
            current_question = self._get_current_question()

            # VR-CLI-REWARD-MOD: Extract reasoning steps from answer
            reasoning_steps = self._extract_reasoning_steps(pred_answer)

            # VR-CLI-REWARD-MOD: Evaluate each step
            step_scores = []
            for step in reasoning_steps:
                step_quality = self._evaluate_reasoning_step(current_question, step, gt_answer)
                step_scores.append(step_quality)

            # VR-CLI-REWARD-MOD: Aggregate step scores
            if step_scores:
                return sum(step_scores) / len(step_scores)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error computing reasoning quality reward: {e}")
            return 0.0

    def _get_current_question(self) -> str:  # VR-CLI-REWARD-MOD: Helper to get current question
        """Get the current question being processed"""
        # VR-CLI-REWARD-MOD: This would need to be set during rollout
        # For now, store it as an instance variable
        return getattr(self, '_current_question', "")

    def _get_target_for_question(self, question: str, gt_answer: str) -> str:  # VR-CLI-REWARD-MOD: Define target for reward
        """
        Define what the target should be for reward computation

        This is highly task-specific and should be customized
        """
        target_type = self.reference_config.get('target_type', 'correct_answer')

        if target_type == 'correct_answer':
            # VR-CLI-REWARD-MOD: Target is the correct answer
            return f"The answer is {gt_answer}"

        elif target_type == 'answer_format':
            # VR-CLI-REWARD-MOD: Target is proper answer formatting
            return "Answer: [a clear, correct response]"

        elif target_type == 'reasoning_conclusion':
            # VR-CLI-REWARD-MOD: Target is reaching the right conclusion
            return f"Therefore, the conclusion is {gt_answer}"

        elif target_type == 'custom':
            # VR-CLI-REWARD-MOD: Custom target from config
            return self.reference_config.get('custom_target', gt_answer)

        else:
            return gt_answer

    def _format_context_with_answer(self, question: str, answer: str) -> str:  # VR-CLI-REWARD-MOD: Format context for likelihood computation
        """Format question and answer for reference model input"""
        format_type = self.reference_config.get('context_format', 'simple')

        if format_type == 'simple':
            return f"{question}\n{answer}"

        elif format_type == 'qa_format':
            return f"Question: {question}\nAnswer: {answer}"

        elif format_type == 'conversation':
            return f"Human: {question}\nAssistant: {answer}"

        else:
            return f"{question}\n{answer}"

    @torch.no_grad()
    def _compute_conditional_logprob(self, context: str, target: str) -> float:  # VR-CLI-REWARD-MOD: Compute conditional log probability
        """
        Compute log P(target | context) using reference model
        """
        try:
            # VR-CLI-REWARD-MOD: Tokenize context and target
            context_tokens = self.reference_tokenizer.encode(context, return_tensors='pt')
            full_text = context + " " + target
            full_tokens = self.reference_tokenizer.encode(full_text, return_tensors='pt')

            # VR-CLI-REWARD-MOD: Move to reference model device
            ref_device = next(self.reference_model.parameters()).device
            context_tokens = context_tokens.to(ref_device)
            full_tokens = full_tokens.to(ref_device)

            # VR-CLI-REWARD-MOD: Get target token positions
            context_len = context_tokens.shape[1]
            target_tokens = full_tokens[:, context_len:]

            # VR-CLI-REWARD-MOD: Forward pass
            with torch.no_grad():
                outputs = self.reference_model(full_tokens)
                logits = outputs.logits

            # VR-CLI-REWARD-MOD: Compute log probabilities for target tokens
            target_logits = logits[:, context_len-1:-1, :]  # Shift for next-token prediction
            target_logprobs = F.log_softmax(target_logits, dim=-1)

            # VR-CLI-REWARD-MOD: Gather log probabilities for actual target tokens
            target_logprobs_gathered = target_logprobs.gather(
                dim=-1,
                index=target_tokens.unsqueeze(-1)
            ).squeeze(-1)

            # VR-CLI-REWARD-MOD: Average log probability
            avg_logprob = target_logprobs_gathered.mean().item()

            return avg_logprob

        except Exception as e:
            logger.error(f"Error computing conditional log probability: {e}")
            return 0.0

    def _create_evaluation_prompt(self, question: str, pred_answer: str, gt_answer: str) -> str:  # VR-CLI-REWARD-MOD: Create evaluation prompt
        """Create prompt for reference model to evaluate answer quality"""
        prompt_template = self.reference_config.get('eval_prompt_template',
            "Question: {question}\nProposed Answer: {pred_answer}\nCorrect Answer: {gt_answer}\n"
            "Rate the quality of the proposed answer on a scale of 0-1, where 1 is perfect: ")

        return prompt_template.format(
            question=question,
            pred_answer=pred_answer,
            gt_answer=gt_answer
        )

    def _get_quality_assessment(self, eval_prompt: str) -> float:  # VR-CLI-REWARD-MOD: Get quality assessment from reference model
        """Get quality score from reference model"""
        try:
            # VR-CLI-REWARD-MOD: Generate assessment
            inputs = self.reference_tokenizer.encode(eval_prompt, return_tensors='pt')
            ref_device = next(self.reference_model.parameters()).device
            inputs = inputs.to(ref_device)

            with torch.no_grad():
                outputs = self.reference_model.generate(
                    inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                )

            # VR-CLI-REWARD-MOD: Decode and parse score
            response = self.reference_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

            # VR-CLI-REWARD-MOD: Extract numeric score
            import re
            score_match = re.search(r'(\d*\.?\d+)', response.strip())
            if score_match:
                score = float(score_match.group(1))
                # Normalize to [0, 1] if needed
                if score > 1.0:
                    score = score / 10.0  # Assume 0-10 scale
                return min(max(score, 0.0), 1.0)

            return 0.0

        except Exception as e:
            logger.error(f"Error getting quality assessment: {e}")
            return 0.0

    def _extract_reasoning_steps(self, answer: str) -> List[str]:  # VR-CLI-REWARD-MOD: Extract reasoning steps
        """Extract individual reasoning steps from answer"""
        # VR-CLI-REWARD-MOD: Simple implementation - split by sentences or reasoning markers
        steps = []

        # Split by common reasoning markers
        for delimiter in ['\n', '. ', 'First,', 'Next,', 'Then,', 'Finally,']:
            if delimiter in answer:
                parts = answer.split(delimiter)
                steps.extend([part.strip() for part in parts if part.strip()])
                break

        if not steps:
            steps = [answer]  # Fallback to full answer

        return steps

    def _evaluate_reasoning_step(self, question: str, step: str, gt_answer: str) -> float:  # VR-CLI-REWARD-MOD: Evaluate single reasoning step
        """Evaluate quality of a single reasoning step"""
        try:
            # VR-CLI-REWARD-MOD: Create step evaluation prompt
            step_prompt = f"Question: {question}\nReasoning step: {step}\nIs this step helpful and correct? (0-1): "

            return self._get_quality_assessment(step_prompt)

        except Exception as e:
            logger.error(f"Error evaluating reasoning step: {e}")
            return 0.0

    def rollout(self, questions: List[str], gt_answers) -> 'grpo.Experience':  # VR-CLI-REWARD-MOD: Override rollout to set context
        """
        Override rollout to set current question context for reward computation
        """
        # VR-CLI-REWARD-MOD: Store questions for reward computation
        self._current_questions = questions
        self._current_gt_answers = gt_answers

        # VR-CLI-REWARD-MOD: Call parent rollout
        experience = super().rollout(questions, gt_answers)

        return experience

    def get_group_rewards_and_acc_with_context(  # VR-CLI-REWARD-MOD: Extended reward method with context
        self,
        pred_answers: List[str],
        gt_answer: str,
        n_latent_forward: torch.Tensor,
        question: str,
        question_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards with explicit question context

        This version receives the question directly for cleaner reward computation
        """
        # VR-CLI-REWARD-MOD: Set current question for reward computation
        self._current_question = question

        # VR-CLI-REWARD-MOD: Call standard reward computation
        return self.get_group_rewards_and_acc(pred_answers, gt_answer, n_latent_forward)


# VR-CLI-REWARD-MOD: Configuration helper functions
def create_reference_reward_config(
    enabled: bool = True,
    model_name: str = None,
    reward_type: str = 'target_likelihood',
    target_type: str = 'correct_answer',
    context_format: str = 'simple',
    reward_scale: float = 1.0,
    reward_clip: float = 5.0,
    use_sigmoid: bool = True,
    device_map: str = 'cpu',
    **kwargs
) -> Dict[str, Any]:
    """
    Create configuration for reference model rewards

    Args:
        enabled: Whether to use reference model rewards
        model_name: Name of reference model (defaults to base model)
        reward_type: Type of reward ('target_likelihood', 'answer_quality', 'reasoning_quality')
        target_type: What to use as target ('correct_answer', 'answer_format', etc.)
        context_format: How to format context ('simple', 'qa_format', 'conversation')
        reward_scale: Scale factor for rewards
        reward_clip: Clipping value for rewards
        use_sigmoid: Whether to apply sigmoid to final reward
        device_map: Device placement for reference model

    Returns:
        Configuration dictionary
    """
    config = {
        'enabled': enabled,
        'model_name': model_name,
        'reward_type': reward_type,
        'target_type': target_type,
        'context_format': context_format,
        'reward_scale': reward_scale,
        'reward_clip': reward_clip,
        'use_sigmoid': use_sigmoid,
        'device_map': device_map,
    }
    config.update(kwargs)
    return config