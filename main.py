#!/usr/bin/env python3
import argparse
import sys
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from sklearn.metrics import f1_score

# Ensure current directory is in sys.path
CURRENT_DIR = Path(__file__).parent.absolute()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.utils.constants import NUM_CLASSES, INPUT_CHANNELS, DATA_SHAPE
from src.utils.datasets import DATASETS, DEFAULT_DATA_DIRS, load_center_datasets, get_labels
from src.utils.models import create_model
from src.utils.heterogeneity import (
    center_label_skew_partition,
    dirichlet_partition,
    stratified_train_test_split,
    HeterogeneousClientTrainer,
)
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
        # Deterministic cuDNN kernels, so a rerun with the same seed gives the same result
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Personalized Federated Learning on multi-site data")
    parser.add_argument("--dataset", type=str, default="polypgen", choices=sorted(DATASETS), help="Dataset to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--global_epoch", type=int, default=20, help="Number of communication rounds")
    parser.add_argument("--local_epoch", type=int, default=5, help="Base local training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the dataset root (default depends on --dataset)")
    parser.add_argument("--check_data", action="store_true", default=False, help="Load the data, print each client's split and exit")
    # Default to CUDA when available; resolved safely at runtime below (not at import time)
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda / cpu). Defaults to cuda if available.")

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
    if args.data_dir is None: args.data_dir = DEFAULT_DATA_DIRS[args.dataset]

    return args


def prepare_datasets(args):
    """
    Prepare each client's train/test split according to H1 (Label Skew) and H2 (Feature Shift).
    If H2=True: client i keeps its own center/site.
    If H2=False: pool all center data and distribute IID.
    If H1=True and H2=True: subsample each center to a Dirichlet (alpha=0.1) label mix, keeping centers separate.
    If H1=True and H2=False: apply Dirichlet partitioning (alpha=0.1) to the pooled data.
    Every split is stratified per label, so a client's train and test sets keep the same label mix.
    """
    centers = load_center_datasets(args.dataset, args.data_dir)
    center_labels = [get_labels(ds) for ds in centers]
    num_clients = len(centers)

    if args.h2_feature_shift:
        # Native multi-center feature shift: client i draws only from center i
        sources = centers
        source_labels = center_labels
        if args.h1_label_skew:
            # Skew labels within each center, so H1 does not erase the feature shift from H2
            part = center_label_skew_partition(center_labels, beta=0.1, min_samples_per_client=10, seed=args.seed)
        else:
            part = {cid: list(range(len(labels))) for cid, labels in enumerate(center_labels)}
    else:
        pooled = ConcatDataset(centers)
        pooled_labels = np.concatenate(center_labels)
        sources = [pooled] * num_clients
        source_labels = [pooled_labels] * num_clients
        if args.h1_label_skew:
            part = dirichlet_partition(pooled_labels, num_clients=num_clients, beta=0.1, min_samples_per_client=5, seed=args.seed)
        else:
            order = np.random.RandomState(args.seed).permutation(len(pooled_labels))
            part = {cid: chunk.tolist() for cid, chunk in enumerate(np.array_split(order, num_clients))}

    client_train_datasets = []
    client_test_datasets = []
    client_label_arrays = []
    for cid in range(num_clients):
        train_idx, test_idx = stratified_train_test_split(part[cid], source_labels[cid], seed=args.seed)
        client_train_datasets.append(Subset(sources[cid], train_idx))
        client_test_datasets.append(Subset(sources[cid], test_idx))
        client_label_arrays.append(source_labels[cid][train_idx])
        test_counts = np.bincount(source_labels[cid][test_idx], minlength=NUM_CLASSES.get(args.dataset, 2))
        print(f"Client {cid+1}: train={len(train_idx)} test={len(test_idx)} "
              f"train labels={np.bincount(client_label_arrays[-1], minlength=len(test_counts)).tolist()} "
              f"test labels={test_counts.tolist()}")

    # Calculate actual JS Divergence between client label distributions
    js_divergence = jensen_shannon_divergence(client_label_arrays, num_classes=NUM_CLASSES.get(args.dataset, 2))
    return client_train_datasets, client_test_datasets, js_divergence


def evaluate_client(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Accuracy, balanced accuracy (mean recall over the labels in the test set), macro F1, and the
    accuracy of always predicting the most common test label. With imbalanced labels, plain
    accuracy can look high while the model only predicts the majority label.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            preds.append(output.argmax(dim=1).cpu())
            targets.append(target)
    if not targets:
        return {"acc": 0.0, "bacc": 0.0, "f1": 0.0, "majority": 0.0}
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    labels = np.unique(y_true)
    recalls = [np.mean(y_pred[y_true == label] == label) for label in labels]
    return {
        "acc": float(np.mean(y_pred == y_true)) * 100.0,
        "bacc": float(np.mean(recalls)) * 100.0,
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) * 100.0,
        "majority": float(np.bincount(y_true).max() / len(y_true)) * 100.0,
    }


def main():
    args = parse_args()

    # Resolve device at runtime so the default is evaluated after CUDA_VISIBLE_DEVICES is set
    # by run_matrix.py (or the scheduler), not at argument-parse time.
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)

    # Guard: if user asked for CUDA but it's not available, fall back gracefully
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available on this machine. Falling back to CPU.")
        device = torch.device("cpu")

    set_seed(args.seed, use_cuda=(device.type == "cuda"))

    print(f"=== Starting FL Run: Dataset={args.dataset} Seed={args.seed} | H1={args.h1_label_skew} H2={args.h2_feature_shift} H3={args.h3_system_heterogeneity} | P1={args.p1_fedrep} P2={args.p2_fedbn} P3={args.p3_fednova} ===")

    # 1. Dataset Preparation
    train_datasets, test_datasets, js_div = prepare_datasets(args)
    num_clients = len(train_datasets)
    if args.check_data:
        print(f"Data check passed: {num_clients} clients, JSD={js_div:.6f}")
        return None

    train_loaders = [DataLoader(ds, batch_size=min(args.batch_size, max(1, len(ds))), shuffle=True) for ds in train_datasets]
    test_loaders = [DataLoader(ds, batch_size=min(args.batch_size, max(1, len(ds))), shuffle=False) for ds in test_datasets]

    # 2. Model & Server Initialization
    global_model = create_model(model_name="resnet18", dataset_name=args.dataset, pretrained=False).to(device)
    
    if args.p3_fednova:
        server = FedNovaServer(global_model=global_model)
    else:
        server = FedAvgServer(global_model=global_model)

    # Initialize client models with synchronized global weights
    client_models = [create_model(model_name="resnet18", dataset_name=args.dataset, pretrained=False).to(device) for _ in range(num_clients)]
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
        round_start = time.time()

        for cid in range(num_clients):
            client_start = time.time()
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
            print(f"  [Round {round_idx}] Client {cid+1}: epochs={train_res['effective_epochs']} "
                  f"steps={train_res['steps_completed']} loss={train_res['avg_loss']:.4f} "
                  f"time={time.time() - client_start:.1f}s", flush=True)

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

        print(f"=== Round {round_idx}/{args.global_epoch} done in {time.time() - round_start:.1f}s ===", flush=True)

    # 4. Final Evaluation & Distance Measurements
    client_results = [evaluate_client(client_models[i], test_loaders[i], device) for i in range(num_clients)]
    final_test_accuracy = float(np.mean([r["acc"] for r in client_results]))

    # Calculate BN Distance (distance between sites' batch-normalization statistics)
    bn_distance = calculate_bn_distance(client_models)

    # Output machine-parseable lines for run_matrix.py interceptor
    print(f"\n--- EXPERIMENT RESULTS ---")
    print(f"FINAL_TEST_ACCURACY: {final_test_accuracy:.4f}")
    print(f"FINAL_BALANCED_ACCURACY: {np.mean([r['bacc'] for r in client_results]):.4f}")
    print(f"FINAL_MACRO_F1: {np.mean([r['f1'] for r in client_results]):.4f}")
    print(f"FINAL_JS_DIVERGENCE: {js_div:.6f}")
    print(f"FINAL_BN_DISTANCE: {bn_distance:.6f}")
    for cid, r in enumerate(client_results):
        print(f"CLIENT_{cid+1}_ACCURACY: {r['acc']:.2f}")
        print(f"CLIENT_{cid+1}_BALANCED_ACCURACY: {r['bacc']:.2f}")
        print(f"CLIENT_{cid+1}_MACRO_F1: {r['f1']:.2f}")
        print(f"CLIENT_{cid+1}_MAJORITY: {r['majority']:.2f}")
    print(f"---------------------------\n")

    return final_test_accuracy, js_div, bn_distance


if __name__ == "__main__":
    main()
