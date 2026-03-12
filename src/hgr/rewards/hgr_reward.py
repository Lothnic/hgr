import torch
import torch.nn.functional as F
from typing import List
from sentence_transformers import SentenceTransformer
from .base import BaseRewardFunction

class HGRReward(BaseRewardFunction):
    """Hypergeometric-Gamma Reward implementation."""
    
    def __init__(self, sbert_model_name: str, phi: float = 1.0, device: str = "cuda"):
        self.sbert_model = SentenceTransformer(sbert_model_name, device=device)
        self.phi = phi
        self.device = device
    
    def compute_rewards(
        self, 
        sources: List[str], 
        references: List[str], 
        hypotheses: List[str]
    ) -> torch.Tensor:
        # Compute SBERT similarity between references and hypotheses
        ref_embeddings = self.sbert_model.encode(
            references, convert_to_tensor=True, device=self.device
        )
        hyp_embeddings = self.sbert_model.encode(
            hypotheses, convert_to_tensor=True, device=self.device
        )
        
        # Cosine similarity [0,1]
        similarity = F.cosine_similarity(ref_embeddings, hyp_embeddings)
        
        # Apply HGR transformation: r = rho * exp(-phi * rho)
        rewards = similarity * torch.exp(-self.phi * similarity)
        return rewards
    
    @property
    def name(self) -> str:
        return "HGR"