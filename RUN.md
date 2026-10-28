# PL_PolypGen — How to Run

All commands are meant to be run from the **repo root** (`PL_PolypGen/`).

---

## 0 · One-time Setup

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## 1 · Dataset

Place the PolypGen dataset (Synapse: syn26376615) at:

```
data/polypgen/
├── C1/
│   ├── images/   ← colonoscopy frames (.jpg / .png)
│   └── masks/    ← matching binary segmentation masks
├── C2/ … C5/     ← same structure
└── C6/           ← held-out OOD test center
```

If your data lives somewhere else, pass `--data_dir /path/to/polypgen` to any command below.

---

## 2 · Single Experiment

### Baseline — FedAvg, natural center split (default)
```bash
python main.py \
    --seed 42 \
    --global_epoch 20 \
    --local_epoch 5 \
    --batch_size 32 \
    --lr 0.01 \
    --data_dir data/polypgen
```

### FedBN — keep Batch Norm layers local (best for sensor heterogeneity)
```bash
python main.py \
    --seed 42 \
    --global_epoch 20 \
    --h2_feature_shift \
    --p2_fedbn \
    --data_dir data/polypgen
```

### FedRep — keep classification head local
```bash
python main.py \
    --seed 42 \
    --global_epoch 20 \
    --h2_feature_shift \
    --p1_fedrep \
    --data_dir data/polypgen
```

### FedNova — normalised aggregation for system heterogeneity
```bash
python main.py \
    --seed 42 \
    --global_epoch 20 \
    --h2_feature_shift \
    --h3_system_heterogeneity \
    --p3_fednova \
    --data_dir data/polypgen
```

### Maximum heterogeneity — all problems + FedBN fix
```bash
python main.py \
    --seed 42 \
    --global_epoch 20 \
    --h1_label_skew \
    --h2_feature_shift \
    --h3_system_heterogeneity \
    --p2_fedbn \
    --data_dir data/polypgen
```

### Full combination — all three fixes enabled
```bash
python main.py \
    --seed 42 \
    --global_epoch 20 \
    --h1_label_skew \
    --h2_feature_shift \
    --h3_system_heterogeneity \
    --p1_fedrep \
    --p2_fedbn \
    --p3_fednova \
    --data_dir data/polypgen
```

---

## 3 · Full 192-Run Matrix (no SLURM)

Runs all 192 combinations of H1/H2/H3 × P1/P2/P3 × seeds {42, 100, 999}.
Results are written row-by-row to `results_matrix.csv` and resumable on crash.

```bash
python run_matrix.py \
    --gpu_id 0 \
    --global_epoch 20 \
    --local_epoch 5 \
    --batch_size 32 \
    --lr 0.01 \
    --output_csv results_matrix.csv \
    --resume
```

Dry-run (prints commands without training):
```bash
python run_matrix.py --dry_run
```

---

## 4 · Full 192-Run Matrix via SLURM

```bash
# Make sure logs/ exists (submit_matrix.sh does this automatically)
sbatch submit_matrix.sh
```

Monitor job:
```bash
squeue -u $USER
tail -f logs/matrix_<JOBID>.out
```

---

## 5 · React Front-end (Interactive Simulation UI)

```bash
npm install
npm run dev       # dev server on http://0.0.0.0:3000
npm run build     # production static build → dist/
```

---

## 6 · Argument Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--global_epoch` | 20 | FL communication rounds |
| `--local_epoch` | 5 | Base local training epochs per client |
| `--batch_size` | 32 | Mini-batch size |
| `--lr` | 0.01 | SGD learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--data_dir` | `data/polypgen` | Path to dataset root |
| `--device` | auto | `cuda` or `cpu` (auto-detected) |
| `--h1_label_skew` | off | Enable Dirichlet Dir(0.1) label skew |
| `--h2_feature_shift` | **on** | Map each client to its own hospital center |
| `--h3_system_heterogeneity` | off | Enable 1×/2×/4× local epoch multipliers |
| `--p1_fedrep` | off | FedRep: keep classification head private |
| `--p2_fedbn` | off | FedBN: keep batch norm layers private |
| `--p3_fednova` | off | FedNova: τ-normalised aggregation |

---

## 7 · Output

```
FINAL_TEST_ACCURACY: 88.4200      ← mean % across 6 clients
FINAL_JS_DIVERGENCE: 0.041823     ← label distribution heterogeneity [0,1]
FINAL_BN_DISTANCE:   0.218741     ← feature/sensor shift between hospitals
CLIENT_1_ACCURACY:   91.35
CLIENT_2_ACCURACY:   89.02
CLIENT_3_ACCURACY:   90.62
CLIENT_4_ACCURACY:   84.10        ← C4 straggler (legacy SD camera)
CLIENT_5_ACCURACY:   91.15
CLIENT_6_ACCURACY:   86.73        ← C6 OOD held-out generalization
```

Full matrix results → `results_matrix.csv`  
SLURM logs → `logs/matrix_<JOBID>.out` / `.err`


cd /home/rahulr/PL_PolypGen/
source PLenv/bin/activate
rm results_matrix.csv
srun --partition=h100 --gres=gpu:1 --time=24:00:00 --pty bash -i
python3 run_matrix.py --gpu_id=0 --global_epoch=20 --local_epoch=5 --batch_size=32


tmux ls
tmux attach -t 0
tmux kill-session -t 0