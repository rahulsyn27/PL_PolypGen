import numpy as np
from typing import Dict, List, Set
from collections import Counter


def dirichlet_partition(
    dataset,
    num_clients: int = 6,
    beta: float = 0.5,
    min_samples_per_client: int = 10,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Partition dataset indices across clients using Dirichlet distribution for label skew.
    
    Args:
        dataset: PyTorch dataset with .targets attribute (or .targets.numpy() for tensors)
        num_clients: Number of clients (default 6)
        beta: Dirichlet concentration parameter (lower = more heterogeneous)
        min_samples_per_client: Minimum samples per client
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping client_id -> list of sample indices
    """
    np.random.seed(seed)
    
    targets = dataset.targets
    if hasattr(targets, 'numpy'):
        targets = targets.numpy()
    targets = np.array(targets, dtype=np.int64)
    
    label_set = set(np.unique(targets))
    target_indices = np.arange(len(targets))
    
    indices_per_label = {label: np.where(targets == label)[0] for label in label_set}
    
    partition = {client_id: [] for client_id in range(num_clients)}
    min_size = 0
    
    while min_size < min_samples_per_client:
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
    
    return partition


def apply_system_heterogeneity(
    client_id: int,
    base_local_epochs: int,
    multipliers: List[int] = [1, 2, 4],
    seed: int = 42
) -> int:
    """
    Apply system heterogeneity by assigning a computational multiplier to local_epochs.
    
    Args:
        client_id: Client identifier (used for deterministic assignment)
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
    Wrapper for a federated client train() loop that applies system heterogeneity.
    """
    def __init__(self, base_local_epochs: int, seed: int = 42):
        self.base_local_epochs = base_local_epochs
        self.seed = seed
        self.client_multipliers = {}
    
    def get_local_epochs(self, client_id: int) -> int:
        """Get the effective local epochs for a client."""
        if client_id not in self.client_multipliers:
            rng = np.random.RandomState(self.seed + client_id)
            multiplier = rng.choice([1, 2, 4])
            self.client_multipliers[client_id] = multiplier
        return self.base_local_epochs * self.client_multipliers[client_id]
    
    def train(self, client_id: int, model, train_loader, optimizer, criterion, device):
        """
        Modified train loop with system heterogeneity.
        
        Returns:
            Dict with 'steps_completed' (exact number of optimizer steps),
                     'effective_epochs' (epochs actually run)
        """
        local_epochs = self.get_local_epochs(client_id)
        model.train()
        
        steps_completed = 0
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                steps_completed += 1
        
        return {
            'steps_completed': steps_completed,
            'effective_epochs': local_epochs,
            'multiplier': self.client_multipliers[client_id]
        }