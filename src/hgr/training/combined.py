"""
Combined DPO + HGR Training — Algorithm 3 from the paper.

Key references:
  - Algorithm 3 (Section III-D)
  - Equation 5 — Combined loss: L_final = α · L_DPO + γ · L_HGR
  - Equation 6 — Parameter update: θ ← θ - η · ∇θ L_final
  - Exponential Gradient Clipping (EGC) — Section IV
"""
import copy

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from hgr.config import ModelConfig, TrainingConfig, RewardConfig
from hgr.rewards.factory import create_reward_function

from hgr.training.hgr import (
    compute_seq_log_probs,
    compute_hgr_loss,
)


def compute_combined_loss(dpo_loss_val, hgr_loss_val, alpha=0.5, gamma=0.5):
    """Combined loss — Equation 5: L_final = α · L_DPO + γ · L_HGR"""
    return alpha * dpo_loss_val + gamma * hgr_loss_val


def exponential_gradient_clipping(model, max_norm=1.0):
    """
    Exponential Gradient Clipping (EGC) — Section IV.

    Dynamically scales gradients when they exceed a threshold,
    preventing gradient explosion that caused instability after epoch 10.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]))

    if total_norm > max_norm:
        clip_factor = max_norm / total_norm
        scale = torch.exp(clip_factor - 1)
        for g in grads:
            g.data.mul_(scale)

    return total_norm


def compute_dpo_loss(
    model, ref_model, tokenizer, sources, preferred, unpreferred, beta, device, max_length
):
    """
    Compute DPO loss — Equation 1.

    L_DPO = -log σ( β · [log(pθ(y_pref|x)/p_ref(y_pref|x))
                        - log(pθ(y_unpref|x)/p_ref(y_unpref|x))] )
    """
    # Tokenize source
    src_inputs = tokenizer(
        sources, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
    ).to(device)

    # Tokenize preferred and unpreferred targets
    pref_labels = tokenizer(
        preferred, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
    ).input_ids.to(device)
    unpref_labels = tokenizer(
        unpreferred, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
    ).input_ids.to(device)

    pref_labels[pref_labels == tokenizer.pad_token_id] = -100
    unpref_labels[unpref_labels == tokenizer.pad_token_id] = -100

    # Policy model log-probs
    policy_pref_logprobs = compute_seq_log_probs(
        model, src_inputs.input_ids, src_inputs.attention_mask, pref_labels
    )
    policy_unpref_logprobs = compute_seq_log_probs(
        model, src_inputs.input_ids, src_inputs.attention_mask, unpref_labels
    )

    # Reference model log-probs (frozen, no grad)
    with torch.no_grad():
        ref_pref_logprobs = compute_seq_log_probs(
            ref_model, src_inputs.input_ids, src_inputs.attention_mask, pref_labels
        )
        ref_unpref_logprobs = compute_seq_log_probs(
            ref_model, src_inputs.input_ids, src_inputs.attention_mask, unpref_labels
        )

    # DPO loss (Eq. 1)
    pref_ratio = policy_pref_logprobs - ref_pref_logprobs
    unpref_ratio = policy_unpref_logprobs - ref_unpref_logprobs
    loss = -F.logsigmoid(beta * (pref_ratio - unpref_ratio)).mean()

    return loss


class CombinedTrainer:
    """
    Algorithm 3 — Combined DPO + HGR.

    Both DPO and HGR share the same policy model.
    DPO additionally uses a frozen reference copy.
    """

    def __init__(self, model_name, reward_config: RewardConfig, training_config: TrainingConfig):
        self.config = training_config
        self.reward_config = reward_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        # Frozen reference model for DPO (deep copy, no gradients)
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Initialize reward function based on config
        self.reward_function = create_reward_function(
            reward_type=self.reward_config.reward_function,
            sbert_model_name=self.reward_config.sbert_model,
            phi=self.reward_config.phi,
            bleurt_model_name=self.reward_config.bleurt_model,
            comet_model_name=self.reward_config.comet_model,
            device=self.device
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=training_config.learning_rate
        )

    def train(self, dpo_dataset, num_epochs=10, batch_size=8):
        """
        Combined training loop (Algorithm 3).

        Args:
            dpo_dataset: Must have columns [source, preferred, unpreferred].
                         'preferred' doubles as the HGR reference target.
        """
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dpo_dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        max_len = self.config.max_source_length

        for epoch in range(num_epochs):
            epoch_dpo = 0.0
            epoch_hgr = 0.0
            epoch_total = 0.0
            num_batches = 0

            self.optimizer.zero_grad()
            for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):

                sources = batch["source"]
                preferred = batch["preferred"]
                unpreferred = batch["unpreferred"]



                # --- DPO loss (Eq. 1) ---
                dpo_loss = compute_dpo_loss(
                    self.model, self.ref_model, self.tokenizer,
                    sources, preferred, unpreferred,
                    beta=self.config.dpo_beta,
                    device=self.device,
                    max_length=max_len,
                )

                with torch.no_grad():
                    gen_ids = self.model.generate(**src_inputs, max_length=max_len)
                gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                # COMPUTE REWARDS USING PLUGGABLE REWARD FUNCTION
                rewards = self.reward_function.compute_rewards(
                    sources=sources,
                    references=preferred,  # 'preferred' serves as reference translation
                    hypotheses=gen_texts
                )

                # COMPUTE LOG PROBS FOR HYPOTHESES (needed for loss calculation)
                gen_labels = self.tokenizer(
                    gen_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len,
                ).to(self.device)
                gen_labels[gen_labels == self.tokenizer.pad_token_id] = -100

                log_probs = compute_seq_log_probs(
                    self.model, src_inputs.input_ids, src_inputs.attention_mask, gen_labels
                )

                # COMPUTE HGR-STYLE LOSS USING THE REWARDS
                # Note: compute_hgr_loss expects rewards and log_probs tensors
                hgr_loss = compute_hgr_loss(log_probs, rewards)

                # --- Combined loss (Eq. 5) ---
                total_loss = compute_combined_loss(
                    dpo_loss, hgr_loss, alpha=self.config.alpha, gamma=self.config.gamma
                )

                # --- Backprop with Gradient Accumulation (Eq. 6) ---
                grad_accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)
                loss_to_backprop = total_loss / grad_accum_steps
                loss_to_backprop.backward()
                
                grad_norm = 0.0
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
                    grad_norm = exponential_gradient_clipping(self.model, self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_dpo += dpo_loss.item()
                epoch_hgr += hgr_loss.item()
                epoch_total += total_loss.item()
                num_batches += 1

            n = max(num_batches, 1)
            print(
                f"Epoch {epoch+1}: "
                f"total={epoch_total/n:.4f}  "
                f"dpo={epoch_dpo/n:.4f}  "
                f"hgr={epoch_hgr/n:.4f}  "
                f"grad_norm={grad_norm:.4f}"
            )
