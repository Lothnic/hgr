import torch
from sentence_transformers import SentenceTransformer
from src.hgr.training.hgr import compute_sbert_similarity, hypergeometric_gamma_reward

def test():
    sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    preferred = ["this is a good translation", "perfect match"]
    unpreferred = ["this is a bad translation", "not a match at all"]
    
    similarity = compute_sbert_similarity(preferred, unpreferred, sbert_model)
    print("Similarity scores:", similarity)
    
    for phi in [0.1, 0.5, 1.0, 2.0]:
        rewards = hypergeometric_gamma_reward(similarity, phi=phi)
        print(f"Phi={phi}, Rewards={rewards}, Mean={rewards.mean().item()} -> HGR Weight: {rewards.mean().item() + 0.1:.4f}")

if __name__ == "__main__":
    test()
