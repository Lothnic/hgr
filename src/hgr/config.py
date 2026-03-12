"""
Hyperparameters and configuration from the paper (Table 4).
Adjust these for your own dataset and hardware.
"""
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    reward_function: str = "hgr"  # Options: "hgr", "bleurt", "comet"
    
    # HGR-specific parameters
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    phi: float = 1.0
    
    # BLEURT-specific parameters
    bleurt_model: str = "Elron/bleurt-base-128"
    
    # COMET-specific parameters
    comet_model: str = "Unbabel/wmt22-comet-da"
    
    # Device specification
    device: str = "cuda"


@dataclass
class ModelConfig:
    """Model configuration."""
    # Base seq2seq model to fine-tune
    base_model: str = "google/mt5-large"
    # Model for generating unpreferred translations (for DPO pairs)
    unpreferred_model: str = "facebook/m2m100_418M"
    # Sentence-BERT model for HGR semantic similarity rewards
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Quantization — set True if GPU VRAM < 16GB
    load_in_4bit: bool = False
    # Sequence lengths
    max_source_length: int = 256
    max_target_length: int = 256
    # Reward function configuration
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class TrainingConfig:
    """Training hyperparameters from the paper."""
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    # DPO temperature β  (controls preference sharpness)
    dpo_beta: float = 0.1
    # HGR decay φ  (controls reward sensitivity)
    hgr_phi: float = 1.0
    # Combined loss weights (α + γ = 1)
    alpha: float = 0.5   # DPO weight
    gamma: float = 0.5   # HGR weight
    # Exponential Gradient Clipping threshold
    max_grad_norm: float = 1.0
    # Output
    output_dir: str = "outputs"
    fp16: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    metrics: list = field(default_factory=lambda: ["bleu", "chrf", "meteor", "bertscore"])
    bertscore_model: str = "bert-base-uncased"
    # Approximate Randomization Test trials
    art_num_trials: int = 10_000
