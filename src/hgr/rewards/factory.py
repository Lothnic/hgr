from .base import BaseRewardFunction
from .hgr_reward import HGRReward
from .bleurt_reward import BLEURTReward
from .comet_reward import CometReward

def create_reward_function(reward_type: str, **kwargs) -> BaseRewardFunction:
    """
    Factory function to create reward instances.
    
    Args:
        reward_type: Type of reward ("hgr", "bleurt", "comet")
        **kwargs: Arguments passed to reward constructor
                  Expected keys:
                  - For HGR: sbert_model_name, phi, device
                  - For BLEURT: model_name, device
                  - For COMET: model_name, device
        
    Returns:
        Configured reward function instance
    """
    reward_type = reward_type.lower()
    
    if reward_type == "hgr":
        # Extract HGR-specific arguments
        hgr_kwargs = {
            'sbert_model_name': kwargs.get('sbert_model_name'),
            'phi': kwargs.get('phi', 1.0),
            'device': kwargs.get('device', 'cuda')
        }
        return HGRReward(**hgr_kwargs)
    elif reward_type == "bleurt":
        # Extract BLEURT-specific arguments
        bleurt_kwargs = {
            'model_name': kwargs.get('model_name'),
            'device': kwargs.get('device', 'cuda')
        }
        return BLEURTReward(**bleurt_kwargs)
    elif reward_type == "comet":
        # Extract COMET-specific arguments
        comet_kwargs = {
            'model_name': kwargs.get('model_name'),
            'device': kwargs.get('device', 'cuda')
        }
        return CometReward(**comet_kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}. "
                         f"Supported types: hgr, bleurt, comet")