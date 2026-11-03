import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter
import torch


def dirichlet_partition(
    dataset,
    num_clients: int = 6,
    beta: float = 0.5,
    min_samples_per_client: int = 10,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Partition dataset indices across clients using Dirichlet distribution for label skew (H1).
    
    Args:
        dataset: PyTorch dataset with .targets attribute, or the label array itself
        num_clients: Number of clients (default 6)
        beta: Dirichlet concentration parameter (lower = more heterogeneous, alpha=0.1 vs alpha=100)
        min_samples_per_client: Minimum samples per client
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping client_id -> list of sample indices
    """
    np.random.seed(seed)
    
    targets = getattr(dataset, "targets", dataset)
    if hasattr(targets, 'numpy'):
        targets = targets.numpy()
    targets = np.array(targets, dtype=np.int64)
    
    label_set = set(np.unique(targets))
    
    indices_per_label = {label: np.where(targets == label)[0] for label in label_set}
    
    partition = {client_id: [] for client_id in range(num_clients)}
    min_size = 0
    
    max_retries = 20
    retry = 0
    while min_size < min_samples_per_client and retry < max_retries:
        retry += 1
        for client_id in range(num_clients):
            partition[client_id] = []
        
        for label in label_set:
            np.random.shuffle(indices_per_label[label])
            distribution = np.random.dirichlet(np.repeat(beta, num_clients))
            cumulative = np.cumsum(distribution) * len(indices_per_label[label])
            split_positions = cumulative.astype(int)[:-1]
            split_indices = np.split(indices_per_label[label], split_positions)
            
            for client_id in range(num_clients):
                partition[client_id].extend(split_indices[client_id].tolist())
        
        min_size = min(len(indices) for indices in partition.values())
    
    # Fallback if any client has too few
    if min_size == 0:
        for client_id in range(num_clients):
            if len(partition[client_id]) == 0:
                partition[client_id] = list(range(len(targets)))[:min_samples_per_client]
                
    return partition


def center_label_skew_partition(
    center_labels: List[np.ndarray],
    beta: float = 0.1,
    min_samples_per_client: int = 10,
    max_retries: int = 20,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Apply label skew (H1) while keeping every client on its own center, so feature shift (H2) survives.
    Each center draws a target label mix from Dirichlet(beta) and is subsampled to match it,
    which means images are dropped rather than moved between centers.

    Args:
        center_labels: Label array of each center, one entry per client
        beta: Dirichlet concentration parameter (lower = more heterogeneous)
        min_samples_per_client: Redraw a center's label mix if it would keep fewer samples
        max_retries: Maximum draws per center; the draw keeping the most samples is used if none qualify
        seed: Random seed for reproducibility

    Returns:
        Dict mapping client_id -> list of indices into that client's own center
    """
    rng = np.random.RandomState(seed)
    partition = {}

    for client_id, labels in enumerate(center_labels):
        present, counts = np.unique(labels, return_counts=True)
        best_keep = counts if len(present) == 1 else None

        retry = 0
        while best_keep is None or (best_keep.sum() < min_samples_per_client and retry < max_retries):
            retry += 1
            target = rng.dirichlet(np.repeat(beta, len(present)))
            # Largest subset with this label mix, limited by the scarcest label
            size = min(count / share for count, share in zip(counts, target) if share > 0)
            keep = np.minimum(counts, np.round(size * target).astype(int))
            if best_keep is None or keep.sum() > best_keep.sum():
                best_keep = keep

        partition[client_id] = []
        for label, keep_count in zip(present, best_keep):
            label_indices = np.where(labels == label)[0]
            rng.shuffle(label_indices)
            partition[client_id].extend(label_indices[:keep_count].tolist())

    return partition


def stratified_train_test_split(
    indices: List[int],
    labels: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Split a client's indices per label, so train and test keep the same label mix.
    A label with a single sample goes to train.

    Args:
        indices: Sample indices held by the client
        labels: Labels addressable by those indices
        train_ratio: Fraction of each label used for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train indices, test indices)
    """
    rng = np.random.RandomState(seed)
    indices = np.array(indices, dtype=np.int64)
    client_labels = labels[indices]
    train, test = [], []

    for label in np.unique(client_labels):
        label_indices = indices[client_labels == label]
        rng.shuffle(label_indices)
        split = max(1, int(train_ratio * len(label_indices)))
        train.extend(label_indices[:split].tolist())
        test.extend(label_indices[split:].tolist())

    return train, test


def apply_system_heterogeneity(
    client_id: int,
    base_local_epochs: int,
    multipliers: List[int] = [1, 2, 4],
    seed: int = 42
) -> int:
    """
    Apply system heterogeneity (H3) by assigning a computational multiplier to local_epochs.
    
    Args:
        client_id: Client identifier
        base_local_epochs: Base number of local epochs
        multipliers: List of possible multipliers [1, 2, 4]
        seed: Base seed for reproducible assignment
    
    Returns:
        Modified local_epochs = base_local_epochs * assigned_multiplier
    """
    rng = np.random.RandomState(seed + client_id)
    multiplier = rng.choice(multipliers)
    return base_local_epochs * multiplier


class HeterogeneousClientTrainer:
    """
    Wrapper for federated client training loop that applies system heterogeneity (H3).
    Tracks steps_completed dynamically for FedNova (P3).
    """
    def __init__(self, base_local_epochs: int, seed: int = 42, enabled: bool = True):
        self.base_local_epochs = base_local_epochs
        self.seed = seed
        self.enabled = enabled
        self.client_multipliers = {}
    
    def get_local_epochs(self, client_id: int) -> int:
        """Get the effective local epochs for a client."""
        if not self.enabled:
            return self.base_local_epochs
            
        if client_id not in self.client_multipliers:
            rng = np.random.RandomState(self.seed + client_id)
            multiplier = rng.choice([1, 2, 4])
            self.client_multipliers[client_id] = multiplier
        return self.base_local_epochs * self.client_multipliers[client_id]
    
    def train(self, client_id: int, model, train_loader, optimizer, criterion, device):
        """
        Train loop with system heterogeneity tracking.
        
        Returns:
            Dict with 'steps_completed' (exact number of optimizer steps),
                      'effective_epochs' (epochs actually run),
                      'multiplier' (speed multiplier)
        """
        local_epochs = self.get_local_epochs(client_id)
        model.train()
        
        steps_completed = 0
        total_loss = 0.0
        sample_count = 0
        
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                steps_completed += 1
                total_loss += loss.item() * len(target)
                sample_count += len(target)
        
        return {
            'steps_completed': steps_completed,
            'effective_epochs': local_epochs,
            'multiplier': self.client_multipliers.get(client_id, 1),
            'avg_loss': total_loss / max(1, sample_count)
        }
