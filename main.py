import argparse
import sys
import os
import random
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure current directory is in sys.path
CURRENT_DIR = Path(__file__).parent.absolute()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.utils.constants import NUM_CLASSES, INPUT_CHANNELS, DATA_SHAPE
from src.utils.polypgen import PolypGenDataset
from src.utils.models import create_model
from src.utils.heterogeneity import dirichlet_partition, HeterogeneousClientTrainer
from src.utils.weight_packaging import package_client_weights, unpack_server_weights
from src.utils.statistical_distances import jensen_shannon_divergence, calculate_bn_distance
from src.server.fedavg import FedAvgServer
from src.server.fednova import FedNovaServer


def set_seed(seed: int, use_cuda: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Personalized Federated Learning on PolypGen")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--global_epoch", type=int, default=20, help="Number of communication rounds")
    parser.add_argument("--local_epoch", type=int, default=5, help="Base local training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--data_dir", type=str, default="data/polypgen", help="Path to polypgen data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")

    # Heterogeneity switches (The Problems)
    parser.add_argument("--h1_label_skew", action="store_true", default=False, help="H1: Dirichlet label skew")
    parser.add_argument("--h2_feature_shift", action="store_true", default=True, help="H2: Feature shift (distinct centers C1-C6 vs pooled IID)")
    parser.add_argument("--h3_system_heterogeneity", action="store_true", default=False, help="H3: System heterogeneity (1x, 2x, 4x local epochs)")

    # PFL fixes (The Solutions)
    parser.add_argument("--p1_fedrep", action="store_true", default=False, help="P1: FedRep (keep classification head local)")
    parser.add_argument("--p2_fedbn", action="store_true", default=False, help="P2: FedBN (keep batchnorm parameters local)")
    parser.add_argument("--p3_fednova", action="store_true", default=False, help="P3: FedNova (normalize aggregation by steps_completed)")

    # Alternative argument syntax support for flexible CLI
    parser.add_argument("--H1", type=lambda x: str(x).lower() in ['true', '1', 't'], default=None)
    parser.add_argument("--H2", type=lambda x: str(x).lower() in ['true', '1', 't'], default=None)
    parser.add_argument("--H3", type=lambda x: str(x).lower() in ['true', '1', 't'], default=None)
    parser.add_argument("--P1", type=lambda x: str(x).lower() in ['true', '1', 't'], default=None)
    parser.add_argument("--P2", type=lambda x: str(x).lower() in ['true', '1', 't'], default=None)
    parser.add_argument("--P3", type=lambda x: str(x).lower() in ['true', '1', 't'], default=None)

    args = parser.parse_args()

    # Consolidate alternative flags if provided
    if args.H1 is not None: args.h1_label_skew = args.H1
    if args.H2 is not None: args.h2_feature_shift = args.H2
    if args.H3 is not None: args.h3_system_heterogeneity = args.H3
    if args.P1 is not None: args.p1_fedrep = args.P1
    if args.P2 is not None: args.p2_fedbn = args.P2
    if args.P3 is not None: args.p3_fednova = args.P3

    return args


def prepare_datasets(args):
    """
    Prepare datasets for 6 clients according to H1 (Label Skew) and H2 (Feature Shift).
    If H2=True: map clients strictly to Centers C1-C6.
    If H2=False: pool all center data and distribute IID.
    If H1=True: apply Dirichlet partitioning (alpha=0.1) for severe label skew.
    """
    num_clients = 6
    client_train_datasets = []
    client_test_datasets = []
    client_label_arrays = []

    if args.h2_feature_shift:
        # Native multi-center feature shift: Client i gets Center i+1
        for cid in range(1, num_clients + 1):
            center_ds = PolypGenDataset(root_dir=args.data_dir, center_id=cid)
            n_samples = len(center_ds)
            indices = list(range(n_samples))
            random.shuffle(indices)

            # 80/20 train/test split per center
            split_point = max(1, int(0.8 * n_samples))
            train_idx = indices[:split_point]
            test_idx = indices[split_point:]

            train_subset = Subset(center_ds, train_idx)
            test_subset = Subset(center_ds, test_idx)

            client_train_datasets.append(train_subset)
            client_test_datasets.append(test_subset)

            # Extract labels for JSD
            labels = [center_ds[i][1].item() if isinstance(center_ds[i][1], torch.Tensor) else center_ds[i][1] for i in train_idx]
            client_label_arrays.append(np.array(labels))

        if args.h1_label_skew:
            # Further skew the labels across clients using Dirichlet
            all_train_samples = []
            for ds in client_train_datasets:
                for idx in range(len(ds)):
                    all_train_samples.append(ds[idx])
            
            # Simple wrapper to run dirichlet_partition
            class CombinedDataset:
                def __init__(self, samples):
                    self.samples = samples
                    self.targets = torch.tensor([s[1] for s in samples])
                def __len__(self): return len(self.samples)
                def __getitem__(self, idx): return self.samples[idx]

            comb = CombinedDataset(all_train_samples)
            part = dirichlet_partition(comb, num_clients=num_clients, beta=0.1, min_samples_per_client=5, seed=args.seed)
            client_train_datasets = [Subset(comb, part[i]) for i in range(num_clients)]
            client_label_arrays = [comb.targets[part[i]].numpy() for i in range(num_clients)]

    else:
        # Pooled IID across all centers (H2=False)
        combined_ds = PolypGenDataset(root_dir=args.data_dir, center_id=None)
        n_total = len(combined_ds)
        indices = list(range(n_total))
        random.shuffle(indices)

        if args.h1_label_skew:
            # Dirichlet label skew on pooled data
            part = dirichlet_partition(combined_ds, num_clients=num_clients, beta=0.1, min_samples_per_client=5, seed=args.seed)
            for cid in range(num_clients):
                c_idx = part[cid]
                split = int(0.8 * len(c_idx))
                client_train_datasets.append(Subset(combined_ds, c_idx[:split]))
                client_test_datasets.append(Subset(combined_ds, c_idx[split:]))
                labels = [combined_ds[i][1].item() if isinstance(combined_ds[i][1], torch.Tensor) else combined_ds[i][1] for i in c_idx[:split]]
                client_label_arrays.append(np.array(labels))
        else:
            # Standard uniform IID partition
            chunk = n_total // num_clients
            for cid in range(num_clients):
                c_idx = indices[cid * chunk : (cid + 1) * chunk]
                split = int(0.8 * len(c_idx))
                client_train_datasets.append(Subset(combined_ds, c_idx[:split]))
                client_test_datasets.append(Subset(combined_ds, c_idx[split:]))
                labels = [combined_ds[i][1].item() if isinstance(combined_ds[i][1], torch.Tensor) else combined_ds[i][1] for i in c_idx[:split]]
                client_label_arrays.append(np.array(labels))

    # Calculate actual JS Divergence between client label distributions
    js_divergence = jensen_shannon_divergence(client_label_arrays, num_classes=2)
    return client_train_datasets, client_test_datasets, js_divergence


def evaluate_client(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return (correct / max(1, total)) * 100.0


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed, use_cuda=(device.type == "cuda"))

    print(f"=== Starting FL Run: Seed={args.seed} | H1={args.h1_label_skew} H2={args.h2_feature_shift} H3={args.h3_system_heterogeneity} | P1={args.p1_fedrep} P2={args.p2_fedbn} P3={args.p3_fednova} ===")

    # 1. Dataset Preparation
    train_datasets, test_datasets, js_div = prepare_datasets(args)
    num_clients = len(train_datasets)

    train_loaders = [DataLoader(ds, batch_size=min(args.batch_size, max(1, len(ds))), shuffle=True) for ds in train_datasets]
    test_loaders = [DataLoader(ds, batch_size=min(args.batch_size, max(1, len(ds))), shuffle=False) for ds in test_datasets]

    # 2. Model & Server Initialization
    global_model = create_model(model_name="resnet18", dataset_name="polypgen", pretrained=False).to(device)
    
    if args.p3_fednova:
        server = FedNovaServer(global_model=global_model)
    else:
        server = FedAvgServer(global_model=global_model)

    # Initialize client models with synchronized global weights
    client_models = [create_model(model_name="resnet18", dataset_name="polypgen", pretrained=False).to(device) for _ in range(num_clients)]
    for m in client_models:
        m.load_state_dict(global_model.state_dict())

    # H3 System Heterogeneity Trainer
    trainer = HeterogeneousClientTrainer(
        base_local_epochs=args.local_epoch,
        seed=args.seed,
        enabled=args.h3_system_heterogeneity
    )
    criterion = nn.CrossEntropyLoss()

    # 3. Federated Training Loop
    for round_idx in range(1, args.global_epoch + 1):
        client_packages = []

        for cid in range(num_clients):
            model = client_models[cid]
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

            # Local training with system heterogeneity tracking
            train_res = trainer.train(
                client_id=cid,
                model=model,
                train_loader=train_loaders[cid],
                optimizer=optimizer,
                criterion=criterion,
                device=device
            )

            # Package weights with P1 (FedRep) and P2 (FedBN) filtering
            pkg = package_client_weights(
                model=model,
                client_id=cid,
                sample_count=len(train_datasets[cid]),
                steps_completed=train_res['steps_completed'],
                p1_fedrep=args.p1_fedrep,
                p2_fedbn=args.p2_fedbn
            )
            client_packages.append(pkg)

        # Server Aggregation (FedAvg or FedNova P3)
        if args.p3_fednova:
            server.aggregate(client_packages, p1_fedrep=args.p1_fedrep, p2_fedbn=args.p2_fedbn)
        else:
            server.aggregate(client_packages, use_fednova=False)

        # Broadcast aggregated weights back to clients
        for cid in range(num_clients):
            unpack_server_weights(
                model=client_models[cid],
                server_params=server.global_model.state_dict(),
                p1_fedrep=args.p1_fedrep,
                p2_fedbn=args.p2_fedbn
            )

    # 4. Final Evaluation & Distance Measurements
    client_accuracies = [evaluate_client(client_models[i], test_loaders[i], device) for i in range(num_clients)]
    final_test_accuracy = float(np.mean(client_accuracies))

    # Calculate BN Distance (distance between sites' batch-normalization statistics)
    bn_distance = calculate_bn_distance(client_models)

    # Output machine-parseable lines for run_matrix.py interceptor
    print(f"\n--- EXPERIMENT RESULTS ---")
    print(f"FINAL_TEST_ACCURACY: {final_test_accuracy:.4f}")
    print(f"FINAL_JS_DIVERGENCE: {js_div:.6f}")
    print(f"FINAL_BN_DISTANCE: {bn_distance:.6f}")
    for cid, acc in enumerate(client_accuracies):
        print(f"CLIENT_{cid+1}_ACCURACY: {acc:.2f}")
    print(f"---------------------------\n")

    return final_test_accuracy, js_div, bn_distance


if __name__ == "__main__":
    main()
