"""
Evaluation metrics for translation quality assessment (Table 3 from paper).

These are standard MT metrics — provided as ready-to-use utilities.
  - BLEU (n-gram precision + brevity penalty)
  - chrF++ (character-level F-score)
  - METEOR (stemming + synonym matching)
  - BERTScore (contextual embedding similarity)

Also includes statistical significance tests (Section V-B):
  - Approximate Randomization Test (ART)
  - Repeated Measures ANOVA with post-hoc paired t-tests
"""
import numpy as np
import sacrebleu
from nltk.translate.meteor_score import meteor_score as nltk_meteor
import nltk


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """BLEU score using SacreBLEU (0-100 scale as in the paper)."""
    result = sacrebleu.corpus_bleu(predictions, [references])
    return result.score


def compute_chrf(predictions: list[str], references: list[str]) -> float:
    """chrF++ score (0-100 scale)."""
    result = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    return result.score


def compute_meteor(predictions: list[str], references: list[str]) -> float:
    """METEOR score (0-100 scale). Requires nltk wordnet data."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    scores = [
        nltk_meteor([ref.split()], pred.split())
        for pred, ref in zip(predictions, references)
    ]
    return np.mean(scores) * 100


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "bert-base-uncased",
) -> float:
    """BERTScore F1 (0-100 scale)."""
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(predictions, references, model_type=model_type, verbose=False)
    return f1.mean().item() * 100


def evaluate_all(
    predictions: list[str],
    references: list[str],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """
    Run all translation metrics and return a results dict.

    Args:
        predictions: List of model-generated translations.
        references: List of human reference translations.
        metrics: Which metrics to compute. Default: all four.

    Returns:
        Dict like {"bleu": 49.81, "chrf": 69.02, "meteor": 45.62, "bertscore": 91.33}
    """
    if metrics is None:
        metrics = ["bleu", "chrf", "meteor", "bertscore"]

    results = {}
    if "bleu" in metrics:
        results["bleu"] = compute_bleu(predictions, references)
    if "chrf" in metrics:
        results["chrf"] = compute_chrf(predictions, references)
    if "meteor" in metrics:
        results["meteor"] = compute_meteor(predictions, references)
    if "bertscore" in metrics:
        results["bertscore"] = compute_bertscore(predictions, references)

    return results


# ---------------------------------------------------------------------------
# Statistical significance tests (Section V-B)
# ---------------------------------------------------------------------------

def approximate_randomization_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    num_trials: int = 10_000,
) -> float:
    """
    Approximate Randomization Test — Equation 7-8 from the paper.

    Tests whether the difference between two models' sentence-level scores
    is statistically significant.

    Args:
        scores_a: Sentence-level scores from model A, shape (N,).
        scores_b: Sentence-level scores from model B, shape (N,).
        num_trials: Number of random shuffle trials (paper uses 10,000).

    Returns:
        p-value. Small p (<0.05) means the difference is significant.
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    observed_diff = np.abs(scores_a.mean() - scores_b.mean())

    count = 0
    for _ in range(num_trials):
        # Randomly swap scores between models with 50% probability
        swap_mask = np.random.randint(0, 2, size=len(scores_a)).astype(bool)
        shuffled_a = np.where(swap_mask, scores_b, scores_a)
        shuffled_b = np.where(swap_mask, scores_a, scores_b)
        rand_diff = np.abs(shuffled_a.mean() - shuffled_b.mean())
        if rand_diff >= observed_diff:
            count += 1

    return count / num_trials


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Cohen's d effect size — Equation 10 from the paper.

    d = (mean_A - mean_B) / std(A - B)
    """
    diff = np.asarray(scores_a) - np.asarray(scores_b)
    return diff.mean() / (diff.std(ddof=1) + 1e-10)
