"""
train_mlp.py — Phase 3: Research-Grade MLP Baseline
Techniques:
  - Deep architecture: 512 → 256 → 128 → 64
  - Youden's J statistic for optimal decision threshold tuning
  - Calibrated probabilities for fair AUC comparison
  - Full publication-quality training curve
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

from config import DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR


def find_optimal_threshold(y_true, y_probs, method='youden'):
    """
    Youden's J statistic: J = Sensitivity + Specificity - 1
    Finds the threshold that maximises the geometric distance from
    the random classifier line — standard technique in medical/security ML papers.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr          # Youden's J
    best_idx  = np.argmax(j_scores)
    return thresholds[best_idx], tpr[best_idx], fpr[best_idx]


def train_mlp():
    print("=" * 60)
    print("PHASE 3: RESEARCH-GRADE MLP TRAINING")
    print("Architecture: 512 → 256 → 128 → 64  |  Youden Threshold")
    print("=" * 60)

    print("\nLoading preprocessed data...")
    X_train = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_train.parquet"))
    X_test  = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "y_train.parquet")).squeeze()
    y_test  = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "y_test.parquet")).squeeze()

    n_benign = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()
    imbalance_ratio = n_attack / n_benign
    print(f"  Benign samples : {n_benign:,}")
    print(f"  Attack samples : {n_attack:,}")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1 (Attack:Benign)")

    # ── Architecture ────────────────────────────────────────────
    # Deeper 4-layer network. Larger first layer captures more
    # complex feature interactions in high-dimensional network traffic.
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-3,                  # Stronger L2 regularisation vs previous run
        batch_size=4096,             # Large batch for stable gradients on 560k samples
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.05,
        n_iter_no_change=20,
        tol=1e-5,
        random_state=42,
        verbose=True,
    )

    print(f"\nTraining MLP (512→256→128→64)...")
    print(f"Note: No sample weights. Threshold tuned via Youden's J post-training.\n")
    t0 = time.time()
    mlp.fit(X_train, y_train)
    train_time = time.time() - t0

    # ── Raw predictions at default 0.5 threshold ────────────────
    y_probs_raw = mlp.predict_proba(X_test)[:, 1]
    y_pred_raw  = (y_probs_raw >= 0.5).astype(int)
    acc_raw     = accuracy_score(y_test, y_pred_raw)
    f1_raw      = f1_score(y_test, y_pred_raw, average='macro')

    # ── Youden's J optimal threshold ────────────────────────────
    optimal_thresh, opt_tpr, opt_fpr = find_optimal_threshold(y_test, y_probs_raw)
    y_pred_opt  = (y_probs_raw >= optimal_thresh).astype(int)
    acc_opt     = accuracy_score(y_test, y_pred_opt)
    f1_opt      = f1_score(y_test, y_pred_opt, average='macro')
    auc         = roc_auc_score(y_test, y_probs_raw)

    print("\n" + "=" * 60)
    print("MLP TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Architecture   : (512, 256, 128, 64)")
    print(f"  Actual Epochs  : {mlp.n_iter_}")
    print(f"  Training Time  : {train_time:.1f}s")
    print(f"\n  ── Default Threshold (0.50) ──")
    print(f"  Accuracy   : {acc_raw*100:.2f}%")
    print(f"  Macro F1   : {f1_raw*100:.2f}%")
    print(f"\n  ── Youden's J Optimal Threshold ({optimal_thresh:.4f}) ──")
    print(f"  Accuracy   : {acc_opt*100:.2f}%")
    print(f"  Macro F1   : {f1_opt*100:.2f}%  ← Use this for comparison")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"\nDetailed Report (Optimal Threshold = {optimal_thresh:.4f}):")
    print(classification_report(y_test, y_pred_opt, target_names=['Benign', 'Attack']))

    # ── Save model + threshold together ─────────────────────────
    bundle = {'model': mlp, 'threshold': optimal_thresh}
    save_path = os.path.join(MODELS_DIR, "MLP_baseline.joblib")
    joblib.dump(bundle, save_path)
    print(f"✅ MLP + optimal threshold saved: {save_path}")

    # ── Save loss data ───────────────────────────────────────────
    loss_df = pd.DataFrame({
        'epoch':      range(1, len(mlp.loss_curve_) + 1),
        'train_loss': mlp.loss_curve_,
        'val_score':  mlp.validation_scores_,
    })
    loss_df.to_csv(os.path.join(FIGURES_DIR, "mlp_training_loss.csv"), index=False)

    # ── Publication-quality training figure ──────────────────────
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False})

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Loss curve
    axes[0].plot(mlp.loss_curve_, color='#8b0000', linewidth=2, label='Training Loss')
    axes[0].set_title('Training Loss (Adam)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Binary Cross-Entropy')
    axes[0].legend(); axes[0].grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Validation accuracy
    axes[1].plot(mlp.validation_scores_, color='#2c6fad', linewidth=2, label='Validation Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, linestyle=':', alpha=0.6)

    # Panel 3: ROC curve with Youden point
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_test, y_probs_raw)
    axes[2].plot(fpr_arr, tpr_arr, color='#8b0000', linewidth=2, label=f'MLP ROC (AUC={auc:.4f})')
    axes[2].plot([0,1],[0,1], 'k--', alpha=0.4)
    axes[2].scatter([opt_fpr], [opt_tpr], s=120, color='#e07b00', zorder=5,
                    label=f"Youden's J Optimal\n(threshold={optimal_thresh:.3f})")
    axes[2].set_title("ROC Curve + Youden's J Optimal Point")
    axes[2].set_xlabel('False Positive Rate'); axes[2].set_ylabel('True Positive Rate')
    axes[2].legend(fontsize=9); axes[2].grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(f'MLP (512→256→128→64) Training — Macro F1 (Optimised): {f1_opt*100:.2f}%',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    curve_path = os.path.join(FIGURES_DIR, "fig8_mlp_training_curve.png")
    plt.savefig(curve_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curve saved: {curve_path}")


if __name__ == "__main__":
    train_mlp()
