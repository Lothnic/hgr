"""
Stage 2: DPO + HGR Training on Modal
====================================
Runs Algorithm 3 (CombinedTrainer) using the generated
DPO triplets on a Modal A10G GPU.

Run: modal run --detach modal_stage2_train.py
Fetch: modal volume get hgr-stage2 / ./stage2_output/
"""
import modal
import os

app = modal.App("hgr-stage2-training")

vol_stage1 = modal.Volume.from_name("hgr-stage1")
vol_stage2 = modal.Volume.from_name("hgr-stage2", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.2",
        "transformers>=4.39.3",
        "peft>=0.10.0",
        "trl>=0.8.0",
        "datasets>=2.18.0",
        "sacrebleu>=2.4.0",
        "sentencepiece>=0.2.0",
        "accelerate>=0.28.0",
        "sentence-transformers>=2.6.0",
        "protobuf>=4.25.0",
        "pandas",
        "numpy",
        "scipy",
    )
    .add_local_file("src/hgr/config.py", remote_path="/src/hgr/config.py")
    .add_local_file("src/hgr/training/hgr.py", remote_path="/src/hgr/training/hgr.py")
    .add_local_file("src/hgr/training/combined.py", remote_path="/src/hgr/training/combined.py")
)

@app.function(
    image=image,
    gpu="H100",
    timeout=7200, # 2 hours
    volumes={
        "/stage1_output": vol_stage1,
        "/stage2_output": vol_stage2,
    },
)
def train():
    import json
    import logging
    import random
    import sys

    # Add /src to pythonpath so imports work
    sys.path.append("/src")

    # Prevent CUDA memory fragmentation on H100
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    from datasets import Dataset as HFDataset

    from hgr.config import TrainingConfig, RewardConfig, ModelConfig
    from hgr.training.combined import CombinedTrainer
    from hgr.rewards.factory import create_reward_function

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    DATA_PATH = "/stage2_output/dpo_dataset_30k_sampled.json"
    MODEL_NAME = "google/mt5-large"
    SBERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    STAGE1_LORA = "/stage1_output"

    # 1. Load Generated DPO Dataset
    logger.info(f"Loading dataset from {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run modal_stage2_data.py first.")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # Convert to HFDataset so CombinedTrainer's DataLoader works easily
    # We can split Train/Val here:
    random.seed(42)
    random.shuffle(pairs)
    
    # 90 train / 10 val
    split_idx = int(len(pairs) * 0.9)
    train_data = pairs[:split_idx]
    val_data = pairs[split_idx:]

    logger.info(f"Loaded {len(train_data)} train samples, {len(val_data)} val samples")
    hf_dataset = HFDataset.from_list(train_data)

    # 2. Config setup
    # From Table 4 in paper: DPO batch 8, LR 5e-5, beta 0.1, phi 1.0, alpha 0.5, gamma 0.5
    # We'll adapt for H100 mt5-large 1.2B (DPO requires 4x forward passes, so we scale down physical batch)
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    train_cfg.batch_size = 128
    train_cfg.gradient_accumulation_steps = 2
    
    # CRITICAL: We've dropped the learning rate back to a highly conservative 1e-5 despite
    # the large 1024 effective batch size. The HGR REINFORCE objective generates high-variance
    # policy gradients, and mT5-small is highly brittle. The previous 5.6e-4 LR caused a 
    # permanent local-minimum representation collapse (model exclusively output token 259).
    train_cfg.learning_rate = 1e-5
    train_cfg.dpo_beta = 0.05
    # Note: hgr_phi is in RewardConfig, not TrainingConfig
    train_cfg.alpha = 0.2
    train_cfg.gamma = 0.8
    
    # We lowered the max lengths heavily to fit 1024 batch sizes onto memory
    train_cfg.num_epochs = 5 # 10 might be too long, try 5
    # Override the model config lengths to fit in memory
    model_cfg.max_source_length = 48
    model_cfg.max_target_length = 48
    
    # Update reward config to use BLEURT
    model_cfg.reward_config.reward_function = "bleurt"
    model_cfg.reward_config.bleurt_model = "Elron/bleurt-base-128"

    # 3. Initialize Trainer
    logger.info("Initializing Combined DPO+HGR Trainer...")
    
    # Instead of raw CombinedTrainer doing from_pretrained internally,
    # we need the Trainer to load the PEFT adapter from Stage 1.
    # The current CombinedTrainer expects model_name to instantiate AutoModel.
    # We'll subclass/monkeypatch CombinedTrainer here to load PEFT.
    
    class PEFTCombinedTrainer(CombinedTrainer):
        def __init__(self, base_model_name, lora_path, reward_config: RewardConfig, training_config: TrainingConfig):
            self.config = training_config
            self.reward_config = reward_config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from peft import PeftModel
            import copy
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            # Load base with device_map={"": 0} to keep it strictly on a single GPU
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": 0}
            )
            
            # Policy model is PEFT (trainable)
            self.model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
            self.model.print_trainable_parameters()
            
            # Reference model is frozen PEFT
            # Reload base to avoid weight sharing issues
            ref_base = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": 0}
            )
            self.ref_model = PeftModel.from_pretrained(ref_base, lora_path)
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
    
    # Setup reward config - using BLEURT reward function
    reward_cfg = RewardConfig(
        reward_function="bleurt",
        bleurt_model="Elron/bleurt-base-128",
        phi=1.0  # This parameter is ignored for BLEURT but kept for compatibility
    )
    
    trainer = PEFTCombinedTrainer(MODEL_NAME, STAGE1_LORA, reward_cfg, train_cfg)

    # 4. Train
    logger.info("Starting training loop...")
    trainer.train(hf_dataset, num_epochs=train_cfg.num_epochs, batch_size=train_cfg.batch_size)

    # 5. Save model
    logger.info("Saving Stage 2 model...")
    self_model = trainer.model
    self_model.save_pretrained("/stage2_output/final_model")
    trainer.tokenizer.save_pretrained("/stage2_output/final_model")
    
    vol_stage2.commit()
    logger.info("Stage 2 Training Complete!")


@app.local_entrypoint()
def main():
    train.remote()
