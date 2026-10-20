# Adversarial Robustness of Tree-Based Intrusion Detection Systems

> **Paper Title:** *Adversarial Robustness of Tree-Based Intrusion Detection Systems Under Feature-Constrained Evasion Attacks*  
> **Target Journal:** Elsevier *Computers & Security*  
> **Dataset:** CIC-IoT2023 (701,978 flows, 40 features, 34 attack classes)

---

## Project Overview

This repository contains the complete research pipeline for studying adversarial attacks and defenses on machine learning-based Intrusion Detection Systems (IDS), using the CIC-IoT2023 benchmark dataset.

### Key Findings
- **Phase Transition:** XGBoost ASR jumps from 2.4% → 63.2% at ε ≈ 4.69 (sharp robustness cliff)
- **Architectural Brittleness:** MLP achieves 100% ASR vs XGBoost's 57% under identical attack budget
- **AT Defense:** Adversarially-trained XGBoost drops ASR from **41.67% → 1.33%** with zero accuracy penalty
- **RF Security Paradox:** RF resists black-box transfer (2% ASR) but collapses under direct white-box attack (99% ASR)

---

## Repository Structure

```
Research_Paper_Stuff/
├── src/                          # All Python source scripts
│   ├── config.py                 # Paths configuration
│   ├── data_loader.py            # Phase 1: Data pipeline
│   ├── train_baseline.py         # Phase 1: RF, XGBoost, LightGBM training
│   ├── shap_analysis.py          # Phase 1: SHAP feature importance
│   ├── genetic_evasion.py        # Phase 2: Multi-objective GA attack
│   ├── phase2_extended_analysis.py  # Phase 2: 4 experiments
│   ├── recall_drop_analysis.py   # Phase 2: Recall/F1 drop analysis
│   ├── train_mlp.py              # Phase 3: MLP (512→256→128→64)
│   ├── phase3_cross_architecture.py # Phase 3: XGBoost vs MLP
│   ├── adversarial_training.py   # Phase 4: AT defense + Gaussian Subspace Smoothing
│   ├── phase4_defense_evaluation.py # Phase 4: 6 defense experiments
│   ├── phase5_visualizations.py  # Phase 5: t-SNE, Pareto, drift
│   └── generate_final_figures.py # Master figure generator (all 21 figs)
├── paper/
│   ├── main.tex                  # Complete Elsevier elsarticle manuscript
│   └── references.bib            # BibTeX references
├── results/
│   ├── figures/                  # All 22 publication figures (600 DPI)
│   └── models/                   # Trained model files (.joblib)
├── data/
│   └── processed/                # Processed parquet files + scaler
├── requirements.txt
└── pyproject.toml
```

---

## Experimental Results Summary

### Phase 1 — Baseline Models

| Model | Accuracy | Macro F1 | Train Time |
|-------|----------|----------|------------|
| Random Forest | 99.22% | 99.60% | 10.18s |
| **XGBoost** | **99.25%** | **99.62%** | **3.41s** |
| LightGBM | 99.24% | 99.61% | 1.99s |

### Phase 2 — Adversarial Attack (Multi-Objective GA, ε=5.0)

| Model | Clean Recall | Adv. Recall | ASR |
|-------|-------------|-------------|-----|
| Random Forest | 99.4% | 1.0% | 99.0% |
| XGBoost | 99.4% | 44.0% | 56.0% |
| LightGBM | 99.2% | 31.6% | 68.4% |

### Phase 4 — Defense (Adversarial Training)

| Model | Clean Recall | Adv. Recall | ASR |
|-------|-------------|-------------|-----|
| XGBoost (baseline) | 99.33% | 58.33% | 41.67% |
| **XGBoost (AT)** | **99.33%** | **98.67%** | **1.33%** |
| MLP (baseline) | 98.00% | 0.00% | 100.00% |
| MLP (AT) | 98.00% | 55.00% | 45.00% |

---

## Setup & Reproduction

### Requirements
- Python 3.10+ (tested on CPython 3.14 Windows)
- Install dependencies: `pip install -r requirements.txt`

### Run Order
```bash
# Phase 1 — Load data & train baselines
python src/data_loader.py
python src/train_baseline.py
python src/shap_analysis.py

# Phase 2 — Adversarial attack
python src/genetic_evasion.py
python src/phase2_extended_analysis.py
python src/recall_drop_analysis.py

# Phase 3 — Cross-architecture
python src/train_mlp.py
python src/phase3_cross_architecture.py

# Phase 4 — Defense
python src/adversarial_training.py
python src/phase4_defense_evaluation.py

# Phase 5 — Publication visuals
python src/phase5_visualizations.py

# Regenerate all 21 figures cleanly
python src/generate_final_figures.py
```

### Dataset
The raw CIC-IoT2023 dataset (138MB) is **not included** due to GitHub file size limits.  
Download it from: [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html)  
Place the merged CSV at: `data/raw/Merged19.csv`

> **Note:** The processed parquet files (`data/processed/`) ARE included, so you can skip data loading and run from Phase 2 onwards directly.

---

## Figures

All 22 publication-ready figures are in `results/figures/` at 600 DPI.

| Figure | Description |
|--------|-------------|
| fig01 | Baseline IDS model performance |
| fig02 | Cross-model adversarial transferability |
| fig03 | Epsilon sensitivity — phase transition at ε≈4.69 |
| fig04 | Per-attack-class evasion vulnerability (31 classes) |
| fig05 | Multi-objective GA vs random noise baseline |
| fig06 | Recall, F1, ASR drop across all models |
| fig07 | Master experimental overview (2×2) |
| fig08 | MLP training convergence curve |
| fig09 | XGBoost vs MLP clean baseline + ROC |
| fig10 | Cross-architecture attack comparison |
| fig11 | Epsilon sensitivity: tree vs. neural |
| fig12 | Model confidence degradation ± std |
| fig13 | Per-class ASR heatmap (XGBoost vs MLP) |
| fig14 | Defense core comparison (3-panel) |
| fig15 | Security-utility tradeoff curve |
| fig16 | Per-class defense effectiveness heatmap |
| fig17 | Cross-architecture transfer matrix |
| fig18 | Radar/spider chart — all 4 models |
| fig19 | t-SNE feature space + KDE overlay |
| fig20 | Pareto frontier: security vs. perturbation cost |
| fig21 | Feature importance drift after adversarial training |
| shap_beeswarm | SHAP beeswarm — feature attribution |

---

## Citation

If you use this code or findings, please cite:

```
[Paper under review — Elsevier Computers & Security]
```

---

## Author

**Dhananjay** — [@DHANANJAY-2006](https://github.com/DHANANJAY-2006)
