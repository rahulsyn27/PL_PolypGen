#!/usr/bin/env python3
"""
Master Orchestrator: run_matrix.py
Runs the full 192-run combinatorial matrix (8 environmental states × 8 algorithmic states × 3 seeds):
- Heterogeneity Switches:
  - H1: Label Skew (Dirichlet alpha=0.1)
  - H2: Feature Shift (True = Map to Centers C1-C6, False = Pool all center data IID)
  - H3: System Heterogeneity (1x, 2x, 4x local epoch multipliers)
- PFL Fixes:
  - P1: FedRep (Classification head kept private)
  - P2: FedBN (Normalization layers kept private)
  - P3: FedNova (Averaging adjusted for unequal training effort using steps_completed)
- Random Seeds: 42, 100, 999
Total Runs: 2 * 2 * 2 * 2 * 2 * 2 * 3 = 192 runs.

Results are intercepted and saved row-by-row into results_matrix.csv.
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
from typing import Dict, Any, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Master Orchestrator for 192-Run PFL Matrix")
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="CUDA device index or SLURM GPU ID (e.g., '0' or '1' or '0,1')",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results_matrix.csv",
        help="Path to output CSV file",
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
        help="Resume execution: skip runs that already exist in results_matrix.csv",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Dry run: print commands without running full training",
    )
    return parser.parse_args()


def load_existing_run_keys(csv_path: str) -> set:
    """Read existing runs to support seamless SLURM job preemption and resuming."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed

    try:
        with open(csv_path, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
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


def parse_execution_output(stdout: str) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[int, float]]:
    """
    Parse stdout from main.py to intercept Test Accuracy, JS Divergence, BN Distance,
    and individual client accuracies.
    """
    test_acc = None
    js_div = None
    bn_dist = None
    client_accs = {}

    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("FINAL_TEST_ACCURACY:"):
            try:
                test_acc = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("FINAL_JS_DIVERGENCE:"):
            try:
                js_div = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("FINAL_BN_DISTANCE:"):
            try:
                bn_dist = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("CLIENT_") and "_ACCURACY:" in line:
            # Example: CLIENT_1_ACCURACY: 89.50
            match = re.match(r"CLIENT_(\d+)_ACCURACY:\s*([0-9.]+)", line)
            if match:
                cid = int(match.group(1))
                c_acc = float(match.group(2))
                client_accs[cid] = c_acc

    return test_acc, js_div, bn_dist, client_accs


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

    # CSV Header setup
    csv_file = Path(args.output_csv)
    file_exists = csv_file.exists()

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
        "js_divergence",
        "bn_distance",
        "client_1_acc",
        "client_2_acc",
        "client_3_acc",
        "client_4_acc",
        "client_5_acc",
        "client_6_acc",
        "status",
        "duration_sec",
    ]

    if not file_exists:
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print(f"[INFO] Created new results CSV: {csv_file}")

    completed_runs = load_existing_run_keys(str(csv_file)) if args.resume else set()
    if completed_runs:
        print(f"[INFO] Resuming: found {len(completed_runs)} already completed runs in {csv_file}")

    # Combinatorial Definition:
    # 8 Environmental States (H1, H2, H3) x 8 Algorithmic States (P1, P2, P3) x 3 Seeds = 192 Runs
    SEEDS = [42, 100, 999]
    H1_OPTS = [True, False]
    H2_OPTS = [True, False]
    H3_OPTS = [True, False]
    P1_OPTS = [True, False]
    P2_OPTS = [True, False]
    P3_OPTS = [True, False]

    all_combinations = list(itertools.product(SEEDS, H1_OPTS, H2_OPTS, H3_OPTS, P1_OPTS, P2_OPTS, P3_OPTS))
    total_runs = len(all_combinations)
    assert total_runs == 192, f"Expected 192 runs, but got {total_runs}"

    print(f"\n=======================================================")
    print(f" PFL Master Orchestrator: 192 Combinatorial Matrix Runs")
    print(f" Seeds: {SEEDS}")
    print(f" Heterogeneity: H1(Label), H2(Feature), H3(System)")
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
            f"--seed={seed}",
            f"--global_epoch={args.global_epoch}",
            f"--local_epoch={args.local_epoch}",
            f"--batch_size={args.batch_size}",
            f"--lr={args.lr}",
        ]

        if h1: cmd.append("--h1_label_skew")
        if h2: cmd.append("--h2_feature_shift")
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
        test_acc = 0.0
        js_div = 0.0
        bn_dist = 0.0
        client_accs = {}

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

            duration = round(time.time() - start_time, 2)

            if result.returncode != 0:
                status = f"FAILED_EXIT_{result.returncode}"
                print(f"[ERROR] Process exited with code {result.returncode}")
                if result.stderr:
                    print(f"[STDERR]\n{result.stderr[-600:]}")
            else:
                test_acc, js_div, bn_dist, client_accs = parse_execution_output(result.stdout)
                success_count += 1
                print(f"[{run_idx:03d}/{total_runs} COMPLETE in {duration}s] Test Acc: {test_acc}% | JSD: {js_div} | BN Dist: {bn_dist}")

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
            "test_accuracy": test_acc if test_acc is not None else -1.0,
            "js_divergence": js_div if js_div is not None else -1.0,
            "bn_distance": bn_dist if bn_dist is not None else -1.0,
            "client_1_acc": client_accs.get(1, -1.0),
            "client_2_acc": client_accs.get(2, -1.0),
            "client_3_acc": client_accs.get(3, -1.0),
            "client_4_acc": client_accs.get(4, -1.0),
            "client_5_acc": client_accs.get(5, -1.0),
            "client_6_acc": client_accs.get(6, -1.0),
            "status": status,
            "duration_sec": duration,
        }

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_dict)
            f.flush()

        completed_runs.add(run_key)

    print(f"\n[DONE] Finished processing all 192 runs. Results recorded in: {csv_file}")


if __name__ == "__main__":
    main()
