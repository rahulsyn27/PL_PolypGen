import importlib
import json
import os
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from data.utils.datasets import PolypGenDataset

DATA_ROOT = Path(__file__).parent.parent.absolute()


def prune_args(args: Namespace) -> dict:
    args_dict = {}
    # general settings
    args_dict["client_num"] = args.client_num
    args_dict["test_ratio"] = args.test_ratio
    args_dict["val_ratio"] = args.val_ratio
    args_dict["seed"] = args.seed
    args_dict["split"] = args.split
    args_dict["IID_ratio"] = args.iid
    args_dict["monitor_window_name_suffix"] = (
        f"{args.dataset}-{args.client_num}clients-{int(args.iid * 100)}%IID"
    )

    if args.dataset == "cifar100":
        args_dict["super_class"] = bool(args.super_class)
        args_dict["monitor_window_name_suffix"] += "-use20superclasses"

    if args.ood_domains is not None:
        args_dict["ood_domains"] = args.ood_domains
        args_dict["monitor_window_name_suffix"] += f"-{args.ood_domains}OODdomains"
    else:
        # Dirchlet
        if args.alpha > 0:
            args_dict["alpha"] = args.alpha
            args_dict["min_samples_per_client"] = args.min_samples_per_client
            args_dict["monitor_window_name_suffix"] += f"-Dir({args.alpha})"
        # randomly assign classes
        elif args.classes > 0:
            args_dict["classes_per_client"] = args.classes
            args_dict["monitor_window_name_suffix"] += f"-{args.classes}classes"
        # allocate shards
        elif args.shards > 0:
            args_dict["shards_per_client"] = args.shards
            args_dict["monitor_window_name_suffix"] += f"-{args.shards}shards"
        elif args.semantic:
            args_dict["pca_components"] = args.pca_components
            args_dict["efficient_net_type"] = args.efficient_net_type
            args_dict["monitor_window_name_suffix"] += "-semantic"

    args_dict["monitor_window_name_suffix"] += f"-seed{args.seed}"
    return args_dict


def process_polypgen(args, partition: dict, stats: dict):
    """Process PolypGen dataset - each center (C1-C6) becomes a client.
    
    Creates 6 client datasets (one per center) and uses center 6 as global test set.
    """
    dataset_root = DATA_ROOT / "polypgen"
    
    # Create 6 client datasets, one for each center C1-C6
    client_datasets = []
    for center_id in range(1, 7):
        client_datasets.append(
            PolypGenDataset(
                root_dir=str(dataset_root),
                center_id=center_id,
                transform=None,  # transforms applied later in BaseDataset
            )
        )
    
    # Concatenate all data for the global dataset (used for partitioning indices)
    all_data = []
    all_targets = []
    data_indices = {}
    data_cnt = 0
    
    for center_id in range(1, 7):
        dataset = client_datasets[center_id - 1]
        center_data = []
        center_targets = []
        
        for i in range(len(dataset)):
            img, label = dataset[i]
            center_data.append(img.numpy() if isinstance(img, torch.Tensor) else np.array(img))
            center_targets.append(label.item() if isinstance(label, torch.Tensor) else label)
        
        center_data = np.array(center_data)
        center_targets = np.array(center_targets)
        
        all_data.append(center_data)
        all_targets.append(center_targets)
        
        # Each center is a client with its own train/val/test split
        n_samples = len(center_data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        test_size = int(n_samples * args.test_ratio)
        val_size = int(n_samples * args.val_ratio)
        train_size = n_samples - test_size - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        data_indices[center_id - 1] = {
            "train": list(data_cnt + train_indices),
            "val": list(data_cnt + val_indices),
            "test": list(data_cnt + test_indices),
        }
        
        stats[center_id - 1] = {
            "x": n_samples,
            "y": Counter(center_targets.tolist()),
        }
        
        data_cnt += n_samples
    
    # All 6 centers are training clients
    clients_4_train = list(range(6))
    # Use center 6 (index 5) as test client as well for global evaluation
    clients_4_test = [5]
    clients_4_val = []
    
    partition["separation"] = {
        "train": clients_4_train,
        "val": clients_4_val,
        "test": clients_4_test,
        "total": 6,
    }
    partition["data_indices"] = [data_indices[i] for i in range(6)]
    args.client_num = 6
    
    # Save combined data
    all_data = np.concatenate(all_data)
    all_targets = np.concatenate(all_targets)
    np.save(dataset_root / "data.npy", all_data)
    np.save(dataset_root / "targets.npy", all_targets)
    
    # Return a combined dataset for the framework
    return PolypGenDataset(
        root_dir=str(dataset_root),
        center_id=None,
        transform=None,
    )


def plot_distribution(client_num: int, label_counts: np.ndarray, save_path: str):
    plt.figure()
    ax = plt.gca()
    left = np.zeros(client_num)
    client_ids = np.arange(client_num)
    for y, cnts in enumerate(label_counts):
        ax.barh(client_ids, width=cnts, label=y, left=left)
        left += cnts
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(bbox_to_anchor=(1.2, 1))
    plt.savefig(save_path, bbox_inches="tight")


def class_from_string(class_string: str) -> type:
    """Dynamically loads a class from a string representation.

    Args:
        class_string (str): The string representation of the class, including the module path.

    Returns:
        type: The loaded class.

    Example:
        class_from_string('path.to.module.ClassName') returns the class 'ClassName' from the module 'path.to.module'.
    """
    module = importlib.import_module(".".join(class_string.split(".")[:-1]))
    class_ = getattr(module, class_string.split(".")[-1])
    return class_


def partitioner_class_from_flwr_datasets(flower_partitioner_class: str):
    """Dynamically loads a partitioner class from a string representation of a
    flwr.datasets.Dataset.

    Args:
        flower_partitioner_class (str): The string representation of the partitioner class, including the module path.

    Returns:
        type: The loaded partitioner class.
    """
    return class_from_string(f"flwr_datasets.partitioner.{flower_partitioner_class}")