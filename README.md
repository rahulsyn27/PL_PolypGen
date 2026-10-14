# PL_PolypGen - Personalized Federated Learning Benchmark (React App)

A web-based benchmarking and simulation suite for **Personalized Federated Learning** on medical polyp detection datasets (**PolypGen**), rewritten from the original Python/PyTorch FL-bench repository into a modern, responsive React + TypeScript application.

## 🧬 Overview

In clinical multi-center endoscopy, each hospital operates unique endoscopic systems (e.g., Olympus EVIS Lucera NBI, Fujifilm ELUXEO BLI/LCI, Pentax i-scan) with differing illumination, dye staining, color profiles, and sensor noise.

Standard Federated Averaging (**FedAvg**) suffers significant performance degradation under this domain shift. This application benchmarks personalized federated algorithms designed to overcome data and system heterogeneity:

- **FedBN (Local Batch Normalization)**: Decouples batch normalization parameters (mean, variance, scale, shift), keeping them private to each clinical center while sharing the convolutional representation globally.
- **FedRep (Shared Representation, Local Heads)**: Learns a global polyp feature extractor trunk across all hospitals while personalizing classification heads locally.
- **FedNova (Normalized τ Aggregation)**: Eliminates objective inconsistency caused by computational speed differences and stragglers across medical centers.
- **DP-FedAvg (Opacus Differential Privacy)**: Calibrated Gaussian noise addition with Rényi Differential Privacy (RDP) accounting to protect patient endoscopic privacy.
- **Local-Only Baseline**: Isolated training without federated collaboration, revealing the critical need for federated cross-center transfer.

## 🏥 PolypGen Clinical Endoscopy Centers

| Center | Hospital / Clinical Site | Endoscope System | Samples | Pos Rate | System Speed | Role |
|--------|--------------------------|------------------|---------|----------|--------------|------|
| **C1** | Tokyo Medical University | Olympus EVIS LUCERA ELITE (NBI) | 520 | 62% | 1x (5 epochs) | Federated Train |
| **C2** | Charité University Clinic Berlin | Fujifilm ELUXEO 7000 (BLI/LCI) | 410 | 48% | 2x (10 epochs) | Federated Train |
| **C3** | Mayo Clinic Endoscopy Unit | Pentax Medical OPTIVISTA EPK-i7010 | 480 | 55% | 1x (5 epochs) | Federated Train |
| **C4** | St. Thomas NHS Foundation Trust | Olympus Exera II 180 (Legacy SD) | 390 | 41% | 4x (Straggler) | Federated Train |
| **C5** | Seoul National University Hospital | Fujifilm VP-4450HD (FICE) | 460 | 58% | 1x (5 epochs) | Federated Train |
| **C6** | Milan Digestive Disease Institute | Olympus EVIS X1 (EDOF / RDI) | 550 | 51% | 1x | **Held-out OOD Test** |

## ✨ Core Features

1. **Interactive Round-by-Round Federated Simulator**:
   - Step through or auto-play communication rounds ($T=1 \dots 50$)
   - Live convergence curves for Personalized Accuracy, Loss, and Out-of-Domain (OOD Center 6) Generalization
   - Per-hospital metrics matrix and real-time FL-bench server log output
   - Parameter transmission inspector showing local vs shared layers

2. **Multi-Center Endoscopy Explorer**:
   - Inspect clinical endoscopy frames from each hospital center
   - Interactive ground truth segmentation bounding box / mask overlay toggle
   - Optical sensor domain shift profiles (wavelengths, lighting glare, sensor resolution)

3. **Data Partitioning & Statistical Heterogeneity ($JSD$)**:
   - Continuous Dirichlet distribution ($\text{Dir}(\alpha)$) slider with live sample allocation
   - Real-time pairwise Jensen-Shannon Divergence ($JSD$) computation
   - System heterogeneity computational multipliers and straggler simulation

4. **Empirical Leaderboard & Analysis**:
   - Side-by-side leaderboard ranking FedBN, FedRep, FedNova, FedAvg, DP-FedAvg, and Local-Only
   - Export benchmark results directly to CSV or JSON

5. **Hydra Configuration Generator**:
   - Live preview and download of Hydra `defaults.yaml` matching `config/defaults.yaml`
   - Copy-paste ready Python CLI execution commands for exact reproduction

## 🚀 Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```
