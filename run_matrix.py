#!/usr/bin/env python3
"""
Master Orchestrator: run_matrix.py
Runs the full 96-run combinatorial matrix (4 environmental states × 8 algorithmic states × 3 seeds):
- Heterogeneity Switches:
  - H1: Label Skew (Dirichlet alpha=0.1, applied within each center)
  - H2: Feature Shift, fixed on: every client keeps its own center (not a switch in this matrix)
  - H3: System Heterogeneity (1x, 2x, 4x local epoch multipliers)
- PFL Fixes:
  - P1: FedRep (Classification head kept private)
  - P2: FedBN (Normalization layers kept private)
  - P3: FedNova (Averaging adjusted for unequal training effort using steps_completed)
- Random Seeds: 42, 100, 999
Total Runs: 2 * 2 * 2 * 2 * 2 * 3 = 96 runs.

Results are intercepted and saved row-by-row into results_<dataset>.csv, and each run's full
output is kept in logs/<dataset>/.
"""

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

CLIENT_METRICS = ["acc", "bacc", "f1", "majority"]
CLIENT_PREFIXES = {
    "acc": "ACCURACY",
    "bacc": "BALANCED_ACCURACY",
    "f1": "MACRO_F1",
    "majority": "MAJORITY",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Master Orchestrator for 96-Run PFL Matrix")
    parser.add_argument("--dataset", type=str, default="polypgen", help="Dataset name passed to main.py")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset root passed to main.py (default: main.py's default for the dataset)")
    parser.add_argument("--num_clients", type=int, default=6, help="Number of clients, used for the per-client CSV columns")
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="CUDA device index or SLURM GPU ID (e.g., '0' or '1' or '0,1')",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: results_<dataset>.csv)",
    )
    parser.add_argument(
        "--global_epoch",
        type=int,
        default=20,
        help="Global communication rounds per experiment",
    )
    parser.add_argument(
        "--local_epoch",
        type=int,
        default=5,
        help="Base local training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for client training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume execution: skip runs that already succeeded in the output CSV",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Dry run: print commands without running full training",
    )
    return parser.parse_args()


def load_existing_run_keys(csv_path: str) -> set:
    """Read successful runs to support seamless SLURM job preemption and resuming.
    Failed runs are left out, so a resumed job retries them."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed

    try:
        with open(csv_path, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] != "SUCCESS":
                    continue
                key = (
                    int(row["seed"]),
                    row["H1_label_skew"].lower() == "true",
                    row["H2_feature_shift"].lower() == "true",
                    row["H3_system_heterogeneity"].lower() == "true",
                    row["P1_fedrep"].lower() == "true",
                    row["P2_fedbn"].lower() == "true",
                    row["P3_fednova"].lower() == "true",
                )
                completed.add(key)
    except Exception as e:
        print(f"[WARN] Error reading existing CSV: {e}")
    return completed


def parse_execution_output(stdout: str) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[int, float]]]:
    """
    Parse stdout from main.py to intercept the overall metrics (Test Accuracy, Balanced Accuracy,
    Macro F1, JS Divergence, BN Distance) and each client's metrics.
    """
    overall = {"test_accuracy": None, "balanced_accuracy": None, "macro_f1": None, "js_divergence": None, "bn_distance": None}
    overall_prefixes = {
        "FINAL_TEST_ACCURACY:": "test_accuracy",
        "FINAL_BALANCED_ACCURACY:": "balanced_accuracy",
        "FINAL_MACRO_F1:": "macro_f1",
        "FINAL_JS_DIVERGENCE:": "js_divergence",
        "FINAL_BN_DISTANCE:": "bn_distance",
    }
    client = {metric: {} for metric in CLIENT_METRICS}
    client_pattern = re.compile(r"CLIENT_(\d+)_(" + "|".join(CLIENT_PREFIXES.values()) + r"):\s*([0-9.]+)")
    metric_by_prefix = {prefix: metric for metric, prefix in CLIENT_PREFIXES.items()}

    for line in stdout.splitlines():
        line = line.strip()
        for prefix, name in overall_prefixes.items():
            if line.startswith(prefix):
                try:
                    overall[name] = float(line.split(":")[1].strip())
                except ValueError:
                    pass
        # Example: CLIENT_1_BALANCED_ACCURACY: 89.50
        match = client_pattern.fullmatch(line)
        if match:
            client[metric_by_prefix[match.group(2)]][int(match.group(1))] = float(match.group(3))

    return overall, client


def build_fieldnames(num_clients: int) -> List[str]:
    fieldnames = [
        "run_id",
        "seed",
        "H1_label_skew",
        "H2_feature_shift",
        "H3_system_heterogeneity",
        "P1_fedrep",
        "P2_fedbn",
        "P3_fednova",
        "test_accuracy",
        "balanced_accuracy",
        "macro_f1",
        "js_divergence",
        "bn_distance",
    ]
    for metric in CLIENT_METRICS:
        fieldnames += [f"client_{cid}_{metric}" for cid in range(1, num_clients + 1)]
    return fieldnames + ["status", "duration_sec"]


def main():
    args = parse_args()
    project_root = Path(__file__).parent.absolute()
    main_script = project_root / "main.py"

    if not main_script.exists():
        print(f"[ERROR] main.py not found at {main_script}")
        sys.exit(1)

    # Configure CUDA device for SLURM node
    env = os.environ.copy()
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"[INFO] Routing executions to CUDA_VISIBLE_DEVICES={args.gpu_id}")

    data_args = [f"--dataset={args.dataset}"]
    if args.data_dir is not None:
        data_args.append(f"--data_dir={args.data_dir}")

    # Load the data once before starting, so a wrong path fails here and not in every run
    check = subprocess.run([sys.executable, str(main_script), *data_args, "--check_data"], capture_output=True, text=True, env=env)
    print(check.stdout)
    if check.returncode != 0:
        print(f"[ERROR] Data check failed, nothing was run.\n{check.stderr[-1500:]}")
        sys.exit(1)

    log_dir = project_root / "logs" / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)

    # CSV Header setup
    csv_file = Path(args.output_csv or f"results_{args.dataset}.csv")
    file_exists = csv_file.exists()

    fieldnames = build_fieldnames(args.num_clients)
    if file_exists:
        with open(csv_file, mode="r", newline="") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != fieldnames:
            print(f"[ERROR] {csv_file} has different columns (an older results file?). Use --output_csv to write a new file.")
            sys.exit(1)

    if not file_exists:
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print(f"[INFO] Created new results CSV: {csv_file}")

    completed_runs = load_existing_run_keys(str(csv_file)) if args.resume else set()
    if completed_runs:
        print(f"[INFO] Resuming: found {len(completed_runs)} already completed runs in {csv_file}")

    # Combinatorial Definition:
    # 4 Environmental States (H1, H3; H2 fixed on) x 8 Algorithmic States (P1, P2, P3) x 3 Seeds = 96 Runs
    SEEDS = [42, 100, 999]
    H1_OPTS = [True, False]
    H2_OPTS = [True]
    H3_OPTS = [True, False]
    P1_OPTS = [True, False]
    P2_OPTS = [True, False]
    P3_OPTS = [True, False]

    all_combinations = list(itertools.product(SEEDS, H1_OPTS, H2_OPTS, H3_OPTS, P1_OPTS, P2_OPTS, P3_OPTS))
    total_runs = len(all_combinations)
    assert total_runs == 96, f"Expected 96 runs, but got {total_runs}"

    print(f"\n=======================================================")
    print(f" PFL Master Orchestrator: 96 Combinatorial Matrix Runs")
    print(f" Dataset: {args.dataset}")
    print(f" Seeds: {SEEDS}")
    print(f" Heterogeneity: H1(Label), H3(System); H2(Feature) fixed on")
    print(f" Fixes: P1(FedRep), P2(FedBN), P3(FedNova)")
    print(f" Output File: {csv_file}")
    print(f"=======================================================\n")

    run_idx = 0
    success_count = 0

    for seed, h1, h2, h3, p1, p2, p3 in all_combinations:
        run_idx += 1
        run_key = (seed, h1, h2, h3, p1, p2, p3)

        if args.resume and run_key in completed_runs:
            print(f"[{run_idx:03d}/{total_runs}] SKIP (Already in CSV): Seed={seed} H=({int(h1)},{int(h2)},{int(h3)}) P=({int(p1)},{int(p2)},{int(p3)})")
            continue

        cmd = [
            sys.executable,
            str(main_script),
            *data_args,
            f"--seed={seed}",
            f"--global_epoch={args.global_epoch}",
            f"--local_epoch={args.local_epoch}",
            f"--batch_size={args.batch_size}",
            f"--lr={args.lr}",
        ]

        if h1: cmd.append("--h1_label_skew")
        # H2 defaults to on in main.py, so it must be set explicitly for H2=False runs
        cmd.append(f"--H2={h2}")
        if h3: cmd.append("--h3_system_heterogeneity")
        if p1: cmd.append("--p1_fedrep")
        if p2: cmd.append("--p2_fedbn")
        if p3: cmd.append("--p3_fednova")

        desc = f"Run {run_idx:03d}/{total_runs} | Seed={seed} | H1={h1} H2={h2} H3={h3} | P1={p1} P2={p2} P3={p3}"
        print(f"\n>>> Starting {desc}...")

        if args.dry_run:
            print(f"[DRY-RUN CMD] {' '.join(cmd)}")
            continue

        start_time = time.time()
        status = "SUCCESS"
        overall = {}
        client = {metric: {} for metric in CLIENT_METRICS}
        log_file = log_dir / f"seed{seed}_H{int(h1)}{int(h2)}{int(h3)}_P{int(p1)}{int(p2)}{int(p3)}.log"

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

            duration = round(time.time() - start_time, 2)
            log_file.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)

            if result.returncode != 0:
                status = f"FAILED_EXIT_{result.returncode}"
                print(f"[ERROR] Process exited with code {result.returncode}")
                if result.stderr:
                    print(f"[STDERR]\n{result.stderr[-600:]}")
            else:
                overall, client = parse_execution_output(result.stdout)
                if None in overall.values():
                    status = "PARSE_ERROR"
                    print(f"[ERROR] Run finished but its results could not be read, see {log_file}")
                else:
                    success_count += 1
                    print(f"[{run_idx:03d}/{total_runs} COMPLETE in {duration}s] Test Acc: {overall['test_accuracy']}% | "
                          f"Balanced Acc: {overall['balanced_accuracy']}% | JSD: {overall['js_divergence']} | BN Dist: {overall['bn_distance']}")

        except Exception as e:
            duration = round(time.time() - start_time, 2)
            status = f"ERROR_{str(e)[:30]}"
            print(f"[EXCEPTION] {e}")

        # Write result row-by-row into CSV with immediate flush
        row_dict = {
            "run_id": run_idx,
            "seed": seed,
            "H1_label_skew": h1,
            "H2_feature_shift": h2,
            "H3_system_heterogeneity": h3,
            "P1_fedrep": p1,
            "P2_fedbn": p2,
            "P3_fednova": p3,
            "status": status,
            "duration_sec": duration,
        }
        for name in ["test_accuracy", "balanced_accuracy", "macro_f1", "js_divergence", "bn_distance"]:
            value = overall.get(name)
            row_dict[name] = value if value is not None else -1.0
        for metric in CLIENT_METRICS:
            for cid in range(1, args.num_clients + 1):
                row_dict[f"client_{cid}_{metric}"] = client[metric].get(cid, -1.0)

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_dict)
            f.flush()

        completed_runs.add(run_key)

    print(f"\n[DONE] Finished processing all {total_runs} runs ({success_count} succeeded this session). Results recorded in: {csv_file}")


if __name__ == "__main__":
    main()
