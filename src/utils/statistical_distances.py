import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import jensenshannon
from collections import Counter
from typing import List, Dict, Union, Tuple, Optional


def calculate_label_distributions(
    label_arrays: List[np.ndarray],
    num_classes: int = 2
) -> np.ndarray:
    """
    Convert list of label arrays to probability distributions.
    
    Args:
        label_arrays: List of arrays, each containing class labels for a client
        num_classes: Total number of classes (default 2 for polypgen)
    
    Returns:
        Array of shape (num_clients, num_classes) with probability distributions
    """
    distributions = np.zeros((len(label_arrays), num_classes))
    for i, labels in enumerate(label_arrays):
        counts = Counter(labels)
        total = max(1, len(labels))
        for cls, count in counts.items():
            if cls < num_classes:
                distributions[i, cls] = count / total
    return distributions


def jensen_shannon_divergence(
    label_arrays: List[np.ndarray],
    num_classes: int = 2
) -> float:
    """
    Calculate average pairwise Jensen-Shannon Divergence between label distributions of clients.
    
    Args:
        label_arrays: List of label arrays for each client
        num_classes: Total number of classes
    
    Returns:
        Average pairwise JS divergence score in [0, 1]
    """
    if len(label_arrays) < 2:
        return 0.0
    
    distributions = calculate_label_distributions(label_arrays, num_classes)
    n = len(label_arrays)
    
    js_divergences = []
    for i in range(n):
        for j in range(i + 1, n):
            # Base 2 JSD squared gives standard metric in [0, 1]
            jsd = jensenshannon(distributions[i], distributions[j], base=2) ** 2
            if not np.isnan(jsd):
                js_divergences.append(jsd)
    
    return float(np.mean(js_divergences)) if js_divergences else 0.0


def calculate_bn_distance(
    client_models_or_stats: List[Union[nn.Module, Dict[str, torch.Tensor]]]
) -> float:
    """
    Calculate average pairwise distance between the sites' batch-normalisation statistics
    (running_mean and running_var) to quantify scanner/feature shift (H2).
    
    Formula based on 2-Wasserstein distance between 1D Gaussians per channel:
    W_2^2(N(u_1, s_1^2), N(u_2, s_2^2)) = (u_1 - u_2)^2 + (s_1 - s_2)^2
    
    Args:
        client_models_or_stats: List of models or state_dicts from each client
    
    Returns:
        Average pairwise BN distance across all clients and all BN layers
    """
    if len(client_models_or_stats) < 2:
        return 0.0

    # Extract BN running stats for each client
    extracted_stats = []
    for item in client_models_or_stats:
        state_dict = item.state_dict() if isinstance(item, nn.Module) else item
        means = []
        vars_ = []
        for key, val in state_dict.items():
            if "running_mean" in key:
                means.append(val.detach().cpu().numpy().flatten())
            elif "running_var" in key:
                vars_.append(np.maximum(1e-6, val.detach().cpu().numpy().flatten()))
        if means and vars_:
            all_means = np.concatenate(means)
            all_stds = np.sqrt(np.concatenate(vars_))
            extracted_stats.append((all_means, all_stds))

    if len(extracted_stats) < 2:
        return 0.0

    n = len(extracted_stats)
    pairwise_distances = []
    for i in range(n):
        u1, s1 = extracted_stats[i]
        for j in range(i + 1, n):
            u2, s2 = extracted_stats[j]
            # Ensure matching shape
            min_len = min(len(u1), len(u2))
            mean_diff_sq = np.mean((u1[:min_len] - u2[:min_len]) ** 2)
            std_diff_sq = np.mean((s1[:min_len] - s2[:min_len]) ** 2)
            dist = np.sqrt(mean_diff_sq + std_diff_sq)
            pairwise_distances.append(dist)

    return float(np.mean(pairwise_distances)) if pairwise_distances else 0.0
