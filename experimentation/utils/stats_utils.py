import numpy as np
from scipy import special


def log_probs_from_logits(logits: np.ndarray) -> np.ndarray:
    """Compute log probabilities using the log-sum-exp trick."""
    log_probs = logits - special.logsumexp(logits, axis=-1, keepdims=True)
    return


def batch_kl_divergence(
    log_probs_1: np.ndarray, log_probs_2: np.ndarray
) -> np.ndarray:
    """Compute the KL divergence between two batches of log probabilities."""
    return special.stats.entropy(
        np.exp(log_probs_1), np.exp(log_probs_2), axis=-1
    )
