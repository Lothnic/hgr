"""
Hypergeometric Gamma Reward (HGR) — Algorithm 2 from the paper.

Key references:
  - Algorithm 2 (Section III-C3)
  - Equation 2 — Cosine similarity via Sentence-BERT
  - Equation 3 — HGR reward: r_i = ρ_i · exp(-φ · ρ_i)
  - Equation 4 — Reward-weighted loss: L_HGR = -(1/B) Σ r_i · log pθ(y_i^gen | x_i)
"""
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm


def compute_sbert_similarity(generated_texts, reference_texts, sbert_model):
    """
    Cosine similarity between generated and reference translations
    using Sentence-BERT embeddings — Equation 2.

    Returns tensor of shape (batch_size,) with values in [-1, 1].
    """
    gen_embedding = sbert_model.encode(generated_texts, convert_to_tensor=True)
    ref_embedding = sbert_model.encode(reference_texts, convert_to_tensor=True)

    similarity_scores = F.cosine_similarity(gen_embedding, ref_embedding, dim=1)

    return similarity_scores


def hypergeometric_gamma_reward(similarity_scores, phi=1.0):
    """
    HGR reward — Equation 3.

    r_i = ρ_i · exp(-φ · ρ_i)
    """
    reward = similarity_scores * torch.exp(-phi * similarity_scores)

    return reward


def compute_seq_log_probs(model, input_ids, attention_mask, labels):
    """
    Compute per-sequence log-probabilities: log p(target | source).

    Returns tensor of shape (batch_size,).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits  # (batch, seq_len, vocab_size)

    # Per-token log-probs
    log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab)

    # Gather log-probs at the label positions
    # Shift labels to ignore padding (-100)
    label_mask = labels != -100
    # Clamp labels to valid range for gather (replace -100 with 0, will be masked)
    safe_labels = labels.clamp(min=0).unsqueeze(-1)  # (batch, seq_len, 1)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels).squeeze(-1)  # (batch, seq_len)

    # Mask out padding positions and sum to get sequence-level log-prob
    token_log_probs = token_log_probs * label_mask.float()
    seq_log_probs = token_log_probs.sum(dim=-1)  # (batch,)

    # Length Normalization:
    # Without this, the model learns to "reward hack" by generating extremely short
    # sequences (e.g., just an EOS token) because shorter sequences naturally have
    # higher absolute log-probabilities (closer to 0), artificially inflating HGR.
    seq_lengths = label_mask.sum(dim=-1).float().clamp(min=1.0)
    normalized_seq_log_probs = seq_log_probs / seq_lengths

    return normalized_seq_log_probs


def compute_hgr_loss(log_probs, rewards):
    """
    Reward-weighted loss — Equation 4.

    L_HGR = -(1/B) Σ r_i · log pθ(y_i^gen | x_i)
    """
    # REINFORCE baseline: subtract the mean reward across the batch.
    # Without this, all rewards are positive (SBERT cosine sim is > 0),
    # meaning the model merely tries to maximize the probability of whatever
    # it generated (usually degenerating to just the [EOS] token).
    # By mean-centering, below-average generations receive a NEGATIVE reward,
    # penalizing bad outputs and pushing the model to generate better text.
    baseline_rewards = rewards - rewards.mean()
    return -torch.mean(baseline_rewards.detach() * log_probs)


class HGRTrainer:
    """
    Implements Algorithm 2 from the paper.

    Steps:
      1. Initialize mT5-Large with pretrained weights
      2. Initialize Sentence-BERT model
      3. For each training step:
         a. Generate translations y_gen ~ pθ(y|x)
         b. Compute SBERT cosine similarity ρ_i (Eq. 2)
         c. Compute HGR reward r_i (Eq. 3)
         d. Compute log-probs of generated text under the model
         e. Compute weighted loss L_HGR (Eq. 4)
         f. Backpropagate and update θ
    """

    def __init__(self, model_name, sbert_model_name, training_config):
        self.config = training_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.sbert_model = SentenceTransformer(sbert_model_name, device=self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=training_config.learning_rate
        )

    def train(self, dataset, num_epochs=10, batch_size=8):
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                self.optimizer.zero_grad()

                sources = batch["source"]
                references = batch["target"]

                # 1. Tokenize source sentences
                src_inputs = self.tokenizer(
                    sources, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.config.max_source_length,
                ).to(self.device)

                # 2. Generate translations (detached — no grad through generation)
                with torch.no_grad():
                    gen_ids = self.model.generate(
                        **src_inputs, max_length=self.config.max_target_length,
                    )
                gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                # 3. Compute SBERT similarity reward (Eq. 2 & 3)
                similarity = compute_sbert_similarity(gen_texts, references, self.sbert_model)
                rewards = hypergeometric_gamma_reward(similarity, phi=self.config.hgr_phi)

                # 4. Compute log-probs of generated tokens under current model
                #    (this IS differentiable — this is where gradients flow)
                gen_labels = self.tokenizer(
                    gen_texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.config.max_target_length,
                ).input_ids.to(self.device)
                # Replace pad token ids with -100 so they're ignored in loss
                gen_labels[gen_labels == self.tokenizer.pad_token_id] = -100

                log_probs = compute_seq_log_probs(
                    self.model, src_inputs.input_ids, src_inputs.attention_mask, gen_labels
                )

                # 5. Reward-weighted loss (Eq. 4)
                loss = compute_hgr_loss(log_probs, rewards.to(self.device))

                # 6. Backprop + update
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
