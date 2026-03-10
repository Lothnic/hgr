"""
Direct Preference Optimization (DPO) — Using TRL's DPOTrainer.

The paper explicitly uses TRL's DPO implementation (Section IV):
  "Fine-tuning was performed using the Transformer Reinforcement Learning
   library [72], which includes the DPO trainer."

This module wraps TRL's DPOTrainer with the paper's configuration.
The novel work you implement is in hgr.py and combined.py.

Refs:
  - Equation 1 — DPO loss (handled by TRL internally)
  - Algorithm 1 — DPO procedure
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import Dataset

from hgr.config import ModelConfig, TrainingConfig


def prepare_dpo_dataset(dataset: Dataset) -> Dataset:
    """
    Format dataset for TRL's DPOTrainer.

    TRL expects columns: "prompt", "chosen", "rejected"
    Our data has: "source", "preferred", "unpreferred"

    Args:
        dataset: Dataset with [source, preferred, unpreferred] columns
                 (output of data.prepare.generate_unpreferred)
    Returns:
        Dataset formatted for DPOTrainer.
    """
    return dataset.rename_columns({
        "source": "prompt",
        "preferred": "chosen",
        "unpreferred": "rejected",
    })


def build_dpo_trainer(
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    train_dataset: Dataset | None = None,
) -> DPOTrainer:
    """
    Build a TRL DPOTrainer with the paper's hyperparameters.

    Args:
        model_config: Model configuration (default: paper settings).
        training_config: Training configuration (default: paper settings).
        train_dataset: DPO-formatted dataset with [prompt, chosen, rejected].

    Returns:
        A configured DPOTrainer ready to call .train()
    """
    if model_config is None:
        model_config = ModelConfig()
    if training_config is None:
        training_config = TrainingConfig()

    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model)

    # DPO training config from Table 4
    dpo_config = DPOConfig(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        gradient_checkpointing=training_config.gradient_checkpointing,
        learning_rate=training_config.learning_rate,
        beta=training_config.dpo_beta,  # Temperature β from Eq. 1
        max_length=model_config.max_source_length + model_config.max_target_length,
        max_prompt_length=model_config.max_source_length,
        fp16=training_config.fp16,
        logging_steps=50,
        save_strategy="epoch",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    return trainer
