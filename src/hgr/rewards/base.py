from abc import ABC, abstractmethod
from typing import List
import torch

class BaseRewardFunction(ABC):
    """Abstract base class for reward functions in HGR framework."""
    
    @abstractmethod
    def compute_rewards(
        self, 
        sources: List[str], 
        references: List[str], 
        hypotheses: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards for translation hypotheses.
        
        Args:
            sources: List of source sentences
            references: List of reference translations
            hypotheses: List of generated translations (hypotheses)
            
        Returns:
            Tensor of rewards (higher = better)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the reward function."""
        pass