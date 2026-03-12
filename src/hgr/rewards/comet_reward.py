import torch
from typing import List
from .base import BaseRewardFunction

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    download_model = load_from_checkpoint = None


class CometReward(BaseRewardFunction):
    """COMET-based reward function."""
    
    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da", device: str = "cuda"):
        if not COMET_AVAILABLE:
            raise ImportError(
                "COMET package is not installed. Please install it with: pip install comet-ml"
                " or install from source: https://github.com/Unbabel/COMET"
            )
        model_path = download_model(model_name)
        self.model = load_from_checkpoint(model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device
    
    def compute_rewards(
        self, 
        sources: List[str], 
        references: List[str], 
        hypotheses: List[str]
    ) -> torch.Tensor:
        # COMET expects list of dicts: {"src": source, "mt": hypothesis, "ref": reference}
        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, ref, hyp in zip(sources, references, hypotheses)
        ]
        
        # Get predictions (returns dict with 'scores' key)
        model_output = self.model.predict(data, batch_size=8, gpus=1 if self.device == "cuda" else 0)
        scores = model_output.scores  # List of float scores
        
        return torch.tensor(scores, device=self.device)
    
    @property
    def name(self) -> str:
        return "COMET"