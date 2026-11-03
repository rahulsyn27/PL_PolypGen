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

Unzip the PolypGen release (Synapse: syn26376615) so the official folders sit at `data/PolypGen/`:

```
data/PolypGen/
├── data_C1/
│   ├── images_C1/   ← single frames (.jpg)
│   └── masks_C1/    ← matching masks (<name>_mask.jpg)
├── data_C2/ … data_C5/
└── data_C6/
```

Linux paths are case-sensitive: `data/PolypGen` and `data/polypgen` are different folders.
If your data lives somewhere else, pass `--data_dir /path/to/PolypGen` to any command below.

Check the data before a long run (prints each client's train/test sizes and label counts, then exits):
```bash
python main.py --check_data
```

A frame's label is 1 if its mask contains a polyp, else 0. The single frames are mostly polyp frames
(1,412 of 1,537), so read balanced accuracy and the majority baseline alongside plain accuracy.

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
    --data_dir data/PolypGen
```

### FedBN — keep Batch Norm layers local (best for sensor heterogeneity)
```bash
python main.py \
    --seed 42 \
    --global_epoch 20 \
    --p2_fedbn \
    --data_dir data/PolypGen
```

### FedRep — keep classification head local
```bash
python main.py \
    --seed 42 \
    --global_epoch 20 \
    --p1_fedrep \
    --data_dir data/PolypGen
```

### FedNova — normalised aggregation for system heterogeneity
```bash
python main.py \
    --seed 42 \
    --global_epoch 20 \
    --h3_system_heterogeneity \
    --p3_fednova \
    --data_dir data/PolypGen
```

### Maximum heterogeneity — all problems + FedBN fix
```bash
python main.py \
    --seed 42 \
    --global_epoch 20 \
    --h1_label_skew \
    --h3_system_heterogeneity \
    --p2_fedbn \
    --data_dir data/PolypGen
```

### Full combination — all three fixes enabled
```bash
python main.py \
    --seed 42 \
    --global_epoch 20 \
    --h1_label_skew \
    --h3_system_heterogeneity \
    --p1_fedrep \
    --p2_fedbn \
    --p3_fednova \
    --data_dir data/PolypGen
```

---

## 3 · Full 96-Run Matrix (no SLURM)

Runs all 96 combinations of H1/H3 × P1/P2/P3 × seeds {42, 100, 999}, with H2 fixed on
(every client keeps its own center). It checks the data first and stops if it cannot load it.
Results are written row-by-row to `results_<dataset>.csv` (here `results_polypgen.csv`) and each
run's full output to `logs/<dataset>/`. Re-running the same command resumes: successful runs are
skipped and failed ones are retried.

```bash
python run_matrix.py \
    --gpu_id 0 \
    --dataset polypgen \
    --data_dir data/PolypGen \
    --global_epoch 20 \
    --local_epoch 5 \
    --batch_size 32 \
    --lr 0.01
```

Dry-run (checks the data, then prints commands without training):
```bash
python run_matrix.py --dry_run
```

---

## 4 · Full 96-Run Matrix via SLURM

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
| `--dataset` | `polypgen` | Dataset to run on (registered in `src/utils/datasets.py`) |
| `--data_dir` | `data/PolypGen` | Path to dataset root (default depends on `--dataset`) |
| `--check_data` | off | Load the data, print each client's split and exit |
| `--device` | auto | `cuda` or `cpu` (auto-detected) |
| `--h1_label_skew` | off | Enable Dirichlet Dir(0.1) label skew |
| `--h2_feature_shift` | **on** | Map each client to its own hospital center (use `--H2=false` to pool centers) |
| `--h3_system_heterogeneity` | off | Enable 1×/2×/4× local epoch multipliers |
| `--p1_fedrep` | off | FedRep: keep classification head private |
| `--p2_fedbn` | off | FedBN: keep batch norm layers private |
| `--p3_fednova` | off | FedNova: τ-normalised aggregation |

---

## 7 · Output

```
FINAL_TEST_ACCURACY: 88.4200      ← mean % across clients
FINAL_BALANCED_ACCURACY: 71.2000  ← mean of each client's per-label recall
FINAL_MACRO_F1: 69.8000
FINAL_JS_DIVERGENCE: 0.041823     ← label distribution heterogeneity [0,1]
FINAL_BN_DISTANCE:   0.218741     ← distance between clients' BN statistics
CLIENT_1_ACCURACY: 91.35
CLIENT_1_BALANCED_ACCURACY: 75.00
CLIENT_1_MACRO_F1: 73.10
CLIENT_1_MAJORITY: 98.04          ← accuracy of always predicting the most common test label
…
```

Full matrix results → `results_<dataset>.csv`
Per-run output → `logs/<dataset>/seed<seed>_H<h1h2h3>_P<p1p2p3>.log`
SLURM logs → `logs/matrix_<JOBID>.out` / `.err`

---

## 8 · Adding Another Dataset

1. Write a loader in `src/utils/datasets.py` that returns one `Dataset` per client (site), each
   yielding `(image_tensor, label)`, and register it in `DATASETS` and `DEFAULT_DATA_DIRS`.
2. Add its `NUM_CLASSES` and `INPUT_CHANNELS` entries in `src/utils/constants.py`.
3. Run `python main.py --dataset <name> --check_data`, then
   `python run_matrix.py --dataset <name> --num_clients <clients>`.
