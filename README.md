# Adversarial Robustness of Tree-Based Intrusion Detection Systems

> **Paper Title:** *Adversarial Robustness of Tree-Based Intrusion Detection Systems Under Feature-Constrained Evasion Attacks*  
> **Authors:** Dhananjay Kadge, Rupali Deshmukh  
> **Affiliation:** Department of Computer Science and Engineering, School of Engineering and Technology, MGM University  
> **Target Journal:** Elsevier *Computers & Security*  
> **Dataset:** CIC-IoT2023 (701,978 flows, 39 features, 31 attack classes)

---

## Project Overview

This repository contains the complete research pipeline, datasets, and reproducibility artifacts for our study on the adversarial vulnerabilities of tree-based and neural network intrusion detection systems under semantic protocol constraints.

### Core Empirical Findings
- **Phase Transition:** Tree ensembles exhibit an abrupt non-linear phase transition at $\varepsilon \approx 4.69$, where XGBoost Attack Success Rate (ASR) jumps from 2.40% to 63.20%.
- **Architectural Brittleness:** Continuous neural decision surfaces (Deep MLP) collapse to 100.00% ASR (0.00% residual recall), whereas XGBoost preserves 44.00% detection recall under identical constraints.
- **Gaussian Subspace Smoothing (GSS) Defense:** Adversarially trained XGBoost with GSS suppresses ASR from **41.67% down to 1.33%** while preserving baseline clean recall at **99.33%** (zero utility loss).
- **Random Forest Transfer Disparity:** Random Forest demonstrates high black-box transfer resistance (2.00% ASR) while failing under direct white-box query access (99.00% ASR).

---

## Repository Structure

```
adversarial-iot-ids-robustness/
├── src/                          # Python research pipeline
│   ├── config.py                 # System paths and global parameters
│   ├── data_loader.py            # Phase 1: CIC-IoT2023 ingestion & preprocessing
│   ├── train_baseline.py         # Phase 1: RF, XGBoost, LightGBM training
│   ├── shap_analysis.py          # Phase 1: SHAP feature importance attribution
│   ├── genetic_evasion.py        # Phase 2: Feature-constrained multi-objective GA
│   ├── phase2_extended_analysis.py  # Phase 2: Epsilon sweeps and phase transitions
│   ├── recall_drop_analysis.py   # Phase 2: Recall and F1 degradation metrics
│   ├── train_mlp.py              # Phase 3: Deep MLP architecture (512-256-128-64)
│   ├── phase3_cross_architecture.py # Phase 3: Cross-architecture comparison
│   ├── adversarial_training.py   # Phase 4: AT defense + Gaussian Subspace Smoothing
│   ├── phase4_defense_evaluation.py # Phase 4: Adaptive white-box defense benchmark
│   ├── phase5_visualizations.py  # Phase 5: t-SNE manifold, Pareto frontier, drift
│   └── generate_final_figures.py # Master script generating publication figures
├── paper/
│   ├── main.tex                  # Official Elsevier elsarticle manuscript
│   └── references.bib            # Peer-reviewed BibTeX bibliography
├── results/
│   ├── figures/                  # Publication-ready figures (600 DPI)
│   └── models/                   # Serialized model weights (.joblib)
├── data/
│   └── processed/                # Preprocessed Parquet partitions & scalers
├── requirements.txt              # Environment dependencies
└── pyproject.toml
```

---

## Baseline Performance & Defense Summary

### Clean Baseline Comparison

| Architecture | Clean Accuracy | Macro F1 | ROC-AUC | Training Time |
|---|:---:|:---:|:---:|:---:|
| Random Forest | 99.22% | 99.60% | 0.9981 | 10.18 s |
| **XGBoost** | **99.25%** | **99.62%** | **0.9981** | **3.41 s** |
| LightGBM | 99.24% | 99.61% | 0.9980 | **1.99 s** |
| Deep MLP (Neural) | 99.02% | 89.38% | 0.9969 | 16.42 s |

### Defense Evaluation Under Adaptive Attack ($\varepsilon = 5.0$)

| Model Configuration | Defense Mechanism | Clean Recall | Adversarial Recall | Evasion ASR |
|---|---|:---:|:---:|:---:|
| XGBoost (Baseline) | None | 99.33% | 58.33% | 41.67% |
| **XGBoost (AT + GSS)** | **Gaussian Subspace Smoothing** | **99.33%** | **98.67%** | **1.33%** |
| Deep MLP (Baseline) | None | 98.00% | 0.00% | 100.00% |
| Deep MLP (AT + GSS) | Gaussian Subspace Smoothing | 98.00% | 55.00% | 45.00% |

---

## Reproduction Instructions

```bash
# 1. Clone the repository
git clone https://github.com/DHANANJAY-2006/adversarial-iot-ids-robustness.git
cd adversarial-iot-ids-robustness

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute experimental pipeline
python src/data_loader.py
python src/train_baseline.py
python src/shap_analysis.py
python src/genetic_evasion.py
python src/train_mlp.py
python src/adversarial_training.py
python src/generate_final_figures.py
```

---

## Dataset Reference
The CIC-IoT2023 dataset is publicly hosted by the Canadian Institute for Cybersecurity:  
[https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

---

## Authors & Contact
- **Dhananjay Kadge** (Corresponding Author) — `dhananjay.kadge@mgmu.ac.in`
- **Rupali Deshmukh** (Co-Author) — `rupali.deshmukh@mgmu.ac.in`  
Department of Computer Science and Engineering, School of Engineering and Technology, MGM University, Chhatrapati Sambhajinagar 431003, Maharashtra, India.
