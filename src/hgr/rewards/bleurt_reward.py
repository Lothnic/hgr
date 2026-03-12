import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base import BaseRewardFunction

class BLEURTReward(BaseRewardFunction):
    """BLEURT-based reward function."""
    
    def __init__(self, model_name: str = "Elron/bleurt-base-128", device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
    
    def compute_rewards(
        self, 
        sources: List[str], 
        references: List[str], 
        hypotheses: List[str]
    ) -> torch.Tensor:
        rewards = []
        for src, ref, hyp in zip(sources, references, hypotheses):
            # Standard BLEURT input format: "source reference candidate"
            text = f"{src} {ref} {hyp}"
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                reward = logits.item()  # BLEURT outputs single score
                rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards, device=self.device)
        
        # Optional normalization if needed (check your BLEURT version's range)
        # Example for [0,1] normalization if scores are in [min_score, max_score]:
        # rewards_tensor = (rewards_tensor - min_score) / (max_score - min_score)
        
        return rewards_tensor
    
    @property
    def name(self) -> str:
        return "BLEURT"