import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import jensenshannon
from collections import Counter
from typing import List, Dict, Union, Tuple


def calculate_label_distributions(
    label_arrays: List[np.ndarray],
    num_classes: int
) -> np.ndarray:
    """
    Convert list of label arrays to probability distributions.
    
    Args:
        label_arrays: List of 6 arrays, each containing class labels for a client
        num_classes: Total number of classes
    
    Returns:
        Array of shape (6, num_classes) with probability distributions
    """
    distributions = np.zeros((len(label_arrays), num_classes))
    for i, labels in enumerate(label_arrays):
        counts = Counter(labels)
        total = len(labels)
        for cls, count in counts.items():
            distributions[i, cls] = count / total
    return distributions


def jensen_shannon_divergence(
    label_arrays: List[np.ndarray],
    num_classes: int = None
) -> float:
    """
    Calculate average Jensen-Shannon Divergence between label distributions of 6 clients.
    
    Args:
        label_arrays: List of 6 arrays, each containing class labels for a client
        num_classes: Total number of classes (auto-detected if None)
    
    Returns:
        Average pairwise JS divergence score
    """
    if len(label_arrays) != 6:
        raise ValueError(f"Expected 6 clients, got {len(label_arrays)}")
    
    if num_classes is None:
        num_classes = max(max(arr) for arr in label_arrays) + 1
    
    distributions = calculate_label_distributions(label_arrays, num_classes)
    
    js_divergences = []
    for i in range(6):
        for j in range(i + 1, 6):
            jsd = jensenshannon(distributions[i], distributions[j], base=2) ** 2
            js_divergences.append(jsd)
    
    return float(np.mean(js_divergences))


def jensen_shannon_divergence_pairwise(
    label_arrays: List[np.ndarray],
    num_classes: int = None
) -> np.ndarray:
    """
    Calculate full pairwise Jensen-Shannon Divergence matrix.
    
    Returns:
        6x6 symmetric matrix of JS divergences
    """
    if num_classes is None:
        num_classes = max(max(arr) for arr in label_arrays) + 1
    
    distributions = calculate_label_distributions(label_arrays, num_classes)
    js_matrix = np.zeros((6, 6))
    
    for i in range(6):
        for j in range(6):
            if i != j:
                js_matrix[i, j] = jensenshannon(distributions[i], distributions[j], base=2) ** 2
    
    return js_matrix


def extract_bn_statistics(model: nn.Module) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract running_mean and running_var from all BatchNorm layers in a model.
    
    Args:
        model: PyTorch model with BatchNorm layers
    
    Returns:
        Dict mapping layer_name -> (running_mean, running_var)
    """
    bn_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_stats[name] = (
                module.running_mean.detach().clone(),
                module.running_var.detach().clone()
            )
    return bn_stats


def batch_norm_distance(
    client_models: List[nn.Module],
    layer_names: List[str] = None
) -> Dict[str, float]:
    """
    Calculate Batch-Normalization Distance (Feature Shift severity) between 6 client models.
    
    Measures L2 distance between running_mean and running_var of BN layers across clients.
    
    Args:
        client_models: List of 6 PyTorch models
        layer_names: Specific BN layer names to compare (None = all BN layers)
    
    Returns:
        Dict with:
            - 'mean_l2_dist': Average L2 distance of running_mean across all pairs
            - 'var_l2_dist': Average L2 distance of running_var across all pairs
            - 'combined_l2_dist': Average combined L2 distance
            - 'per_layer': Dict of per-layer distances
    """
    if len(client_models) != 6:
        raise ValueError(f"Expected 6 client models, got {len(client_models)}")
    
    all_bn_stats = [extract_bn_statistics(model) for model in client_models]
    
    if layer_names is None:
        layer_names = list(all_bn_stats[0].keys())
    
    per_layer = {}
    mean_distances = []
    var_distances = []
    combined_distances = []
    
    for layer_name in layer_names:
        if layer_name not in all_bn_stats[0]:
            continue
        
        layer_mean_distances = []
        layer_var_distances = []
        
        for i in range(6):
            for j in range(i + 1, 6):
                mean_i, var_i = all_bn_stats[i][layer_name]
                mean_j, var_j = all_bn_stats[j][layer_name]
                
                mean_l2 = torch.norm(mean_i - mean_j, p=2).item()
                var_l2 = torch.norm(var_i - var_j, p=2).item()
                
                layer_mean_distances.append(mean_l2)
                layer_var_distances.append(var_l2)
                mean_distances.append(mean_l2)
                var_distances.append(var_l2)
                combined_distances.append(mean_l2 + var_l2)
        
        per_layer[layer_name] = {
            'mean_l2': float(np.mean(layer_mean_distances)) if layer_mean_distances else 0.0,
            'var_l2': float(np.mean(layer_var_distances)) if layer_var_distances else 0.0,
            'combined_l2': float(np.mean(layer_mean_distances) + np.mean(layer_var_distances)) if layer_mean_distances else 0.0,
        }
    
    return {
        'mean_l2_dist': float(np.mean(mean_distances)) if mean_distances else 0.0,
        'var_l2_dist': float(np.mean(var_distances)) if var_distances else 0.0,
        'combined_l2_dist': float(np.mean(combined_distances)) if combined_distances else 0.0,
        'per_layer': per_layer
    }


def batch_norm_distance_matrix(
    client_models: List[nn.Module],
    layer_names: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate pairwise BN distance matrices for running_mean and running_var.
    
    Returns:
        Dict with 'mean_matrix' and 'var_matrix' as 6x6 numpy arrays
    """
    if len(client_models) != 6:
        raise ValueError(f"Expected 6 client models, got {len(client_models)}")
    
    all_bn_stats = [extract_bn_statistics(model) for model in client_models]
    
    if layer_names is None:
        layer_names = list(all_bn_stats[0].keys())
    
    mean_matrices = {}
    var_matrices = {}
    
    for layer_name in layer_names:
        if layer_name not in all_bn_stats[0]:
            continue
        
        mean_matrix = np.zeros((6, 6))
        var_matrix = np.zeros((6, 6))
        
        for i in range(6):
            for j in range(6):
                if i != j:
                    mean_i, var_i = all_bn_stats[i][layer_name]
                    mean_j, var_j = all_bn_stats[j][layer_name]
                    mean_matrix[i, j] = torch.norm(mean_i - mean_j, p=2).item()
                    var_matrix[i, j] = torch.norm(var_i - var_j, p=2).item()
        
        mean_matrices[layer_name] = mean_matrix
        var_matrices[layer_name] = var_matrix
    
    return {'mean_matrix': mean_matrices, 'var_matrix': var_matrices}