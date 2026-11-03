"""
Dataset registry: every dataset is loaded as a list of per-site datasets, one per client.
To add a dataset, write a loader that returns that list and register it in DATASETS
(and its default path in DEFAULT_DATA_DIRS, plus NUM_CLASSES / INPUT_CHANNELS in constants.py).
"""
from typing import Callable, Dict, List
import numpy as np
from torch.utils.data import Dataset

from src.utils.polypgen import PolypGenDataset


def load_polypgen(data_dir: str) -> List[Dataset]:
    """PolypGen single frames: client i holds center C{i+1}."""
    return [PolypGenDataset(root_dir=data_dir, center_id=cid) for cid in range(1, 7)]


DATASETS: Dict[str, Callable[[str], List[Dataset]]] = {
    "polypgen": load_polypgen,
}

DEFAULT_DATA_DIRS: Dict[str, str] = {
    "polypgen": "data/PolypGen",
}


def load_center_datasets(name: str, data_dir: str) -> List[Dataset]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Registered: {sorted(DATASETS)}")
    return DATASETS[name](data_dir)


def get_labels(dataset: Dataset) -> np.ndarray:
    """Label of every sample, read from .targets when available, else by indexing the dataset."""
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return np.asarray(targets.numpy() if hasattr(targets, "numpy") else targets, dtype=np.int64)
    return np.array([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)
