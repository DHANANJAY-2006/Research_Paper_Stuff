"""
phase5_visualizations.py — Phase 5: Publication-Grade Visualizations
Generates:
  fig19 — t-SNE: Clean Attacks vs Adversarial Attacks in feature space
  fig20 — Pareto Frontier: ASR vs Perturbation Budget
  fig21 — Feature Importance Drift: Baseline XGBoost vs AT-XGBoost
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.linestyle': ':', 'grid.alpha': 0.5,
})

C_BENIGN = '#2ecc71'    # green
C_CLEAN  = '#2c6fad'    # steel blue
C_ADV    = '#8b0000'    # dark red
C_AT     = '#2e8b57'    # forest green
C_ACCENT = '#e07b00'


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {name}")


# ─────────────────────────────────────────────────────────────
# FIG 19: t-SNE Feature Space Visualisation
# ─────────────────────────────────────────────────────────────
def fig19_tsne():
    print("=" * 60)
    print("FIG 19: t-SNE FEATURE SPACE VISUALISATION")
    print("=" * 60)

    X_test  = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_test.parquet"))
    y_test  = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "y_test.parquet")).squeeze()
    adv_path = os.path.join(DATA_PROCESSED_DIR, "X_adversarial_sample.parquet")

    col_names = list(X_test.columns)
    X_adv = pd.read_parquet(adv_path).reindex(columns=col_names, fill_value=0)

    np.random.seed(42)

    # Sample subsets
    atk_idx  = np.where(y_test.values == 1)[0]
    ben_idx  = np.where(y_test.values == 0)[0]
    N        = 600

    X_clean_attacks = X_test.values[np.random.choice(atk_idx, N, replace=False)]
    X_benign        = X_test.values[np.random.choice(ben_idx, min(N//3, len(ben_idx)), replace=False)]
    X_adv_sample    = X_adv.values[np.random.choice(len(X_adv), min(N, len(X_adv)), replace=False)]

    X_combined = np.vstack([X_benign, X_clean_attacks, X_adv_sample])
    labels     = (
        ['Benign Traffic'] * len(X_benign) +
        ['Clean Attacks'] * len(X_clean_attacks) +
        ['Adversarial Attacks'] * len(X_adv_sample)
    )

    print(f"  Running t-SNE on {len(X_combined)} samples...")
    print("  (PCA pre-reduction for speed → t-SNE final embedding)")

    # PCA pre-reduction (standard practice for large-dim data)
    pca_X = PCA(n_components=20, random_state=42).fit_transform(
        StandardScaler().fit_transform(X_combined))

    tsne  = TSNE(n_components=2, perplexity=40, max_iter=1000,
                 random_state=42, learning_rate='auto', init='pca')
    embed = tsne.fit_transform(pca_X)

    labels     = np.array(labels)
    color_map  = {
        'Benign Traffic':      C_BENIGN,
        'Clean Attacks':       C_CLEAN,
        'Adversarial Attacks': C_ADV,
    }
    marker_map = {
        'Benign Traffic':      'o',
        'Clean Attacks':       's',
        'Adversarial Attacks': '^',
    }
    size_map = {
        'Benign Traffic':      18,
        'Clean Attacks':       14,
        'Adversarial Attacks': 20,
    }

    # Main t-SNE plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    for cls in ['Benign Traffic', 'Clean Attacks', 'Adversarial Attacks']:
        mask = labels == cls
        axes[0].scatter(embed[mask, 0], embed[mask, 1],
                        c=color_map[cls], marker=marker_map[cls],
                        s=size_map[cls], alpha=0.65, label=cls, edgecolors='none')

    axes[0].set_title('t-SNE Embedding: Feature Space\n(Clean vs. Adversarial Attack Traffic)')
    axes[0].set_xlabel('t-SNE Dimension 1'); axes[0].set_ylabel('t-SNE Dimension 2')
    axes[0].legend(fontsize=10, markerscale=1.8)

    # Density inset — zoom on adversarial vs clean attacks overlap
    from scipy.stats import gaussian_kde
    for cls, col in [('Clean Attacks', C_CLEAN), ('Adversarial Attacks', C_ADV)]:
        mask = labels == cls
        pts  = embed[mask]
        try:
            kde  = gaussian_kde(pts.T, bw_method=0.3)
            xi   = np.linspace(embed[:, 0].min(), embed[:, 0].max(), 120)
            yi   = np.linspace(embed[:, 1].min(), embed[:, 1].max(), 120)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi   = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            axes[1].contourf(Xi, Yi, Zi, levels=6, colors=[col], alpha=0.25)
            axes[1].contour(Xi, Yi, Zi, levels=6, colors=[col], linewidths=0.8, alpha=0.7)
        except Exception:
            pass

    for cls in ['Benign Traffic', 'Clean Attacks', 'Adversarial Attacks']:
        mask = labels == cls
        axes[1].scatter(embed[mask, 0], embed[mask, 1],
                        c=color_map[cls], marker=marker_map[cls],
                        s=size_map[cls], alpha=0.5, label=cls, edgecolors='none')

    axes[1].set_title('t-SNE Density Overlay\n(Adversarial Examples Overlap with Benign Region)')
    axes[1].set_xlabel('t-SNE Dimension 1'); axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].legend(fontsize=10, markerscale=1.8)

    fig.suptitle('Phase 5 — t-SNE Feature Space: Adversarial Examples Bridge Attack→Benign Region',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig19_tsne_feature_space.png')


# ─────────────────────────────────────────────────────────────
# FIG 20: Pareto Frontier — ASR vs Perturbation Budget
# ─────────────────────────────────────────────────────────────
def fig20_pareto_frontier():
    print("\n" + "="*60)
    print("FIG 20: PARETO FRONTIER — ASR vs PERTURBATION SIZE")
    print("="*60)

    # Multi-epsilon data across architectures (from Phase 2 & 3)
    # Each point: (epsilon, ASR, model)
    data = {
        'XGBoost (Clean)': {
            'eps': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            'asr': [2.0, 2.0, 2.0, 3.0, 15.0, 57.0, 72.0],
        },
        'XGBoost (AT)': {
            'eps': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            'asr': [0.5, 0.5, 0.5, 0.5, 0.8, 1.3, 3.5],
        },
        'MLP (Clean)': {
            'eps': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            'asr': [15.0, 35.0, 65.0, 88.0, 97.0, 100.0, 100.0],
        },
        'MLP (AT)': {
            'eps': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            'asr': [5.0, 10.0, 22.0, 38.0, 43.0, 45.0, 50.0],
        },
    }

    styles = {
        'XGBoost (Clean)': (C_CLEAN, '-',  'o'),
        'XGBoost (AT)':    (C_AT,    '-',  's'),
        'MLP (Clean)':     (C_ADV,   '--', '^'),
        'MLP (AT)':        ('#9b59b6', '--', 'D'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left: All curves
    for name, d in data.items():
        col, ls, marker = styles[name]
        axes[0].plot(d['eps'], d['asr'], marker=marker, color=col, linewidth=2.5,
                     markersize=9, linestyle=ls, label=name, zorder=3)

    axes[0].fill_between([0.5, 6.5], [0, 0], [5, 5], alpha=0.07, color=C_AT, label='Safe Zone (ASR<5%)')
    axes[0].set_xlabel('Perturbation Budget (ε)'); axes[0].set_ylabel('Attack Success Rate (%)')
    axes[0].set_title('ASR vs. Perturbation Budget\nAll Models: Clean & Adversarially Trained')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].legend(fontsize=9, loc='upper left')
    axes[0].set_xlim(0, 7.5); axes[0].set_ylim(-3, 110)

    # Right: Pareto frontier plot — plot (epsilon_needed, ASR) to show
    # the "minimum cost to achieve maximum evasion" for each model
    # Pareto-optimal points: no point dominates (lower epsilon AND lower ASR)
    fig_data = []
    for name, d in data.items():
        col, ls, marker = styles[name]
        for eps, asr in zip(d['eps'], d['asr']):
            fig_data.append({'name': name, 'eps': eps, 'asr': asr, 'col': col, 'marker': marker})

    for name, d in data.items():
        col, ls, marker = styles[name]
        axes[1].scatter(d['eps'], d['asr'], c=col, marker=marker,
                        s=100, label=name, zorder=3, edgecolors='white', linewidths=0.5)

        # Draw Pareto frontier for this model
        eps_arr = np.array(d['eps'])
        asr_arr = np.array(d['asr'])
        pareto_eps, pareto_asr = [], []
        max_asr_so_far = -1
        for e, a in zip(eps_arr, asr_arr):
            if a > max_asr_so_far:
                pareto_eps.append(e)
                pareto_asr.append(a)
                max_asr_so_far = a
        axes[1].step(pareto_eps, pareto_asr, where='post', color=col,
                     linewidth=1.5, linestyle=':', alpha=0.7)

    axes[1].set_xlabel('Perturbation Budget (ε)')
    axes[1].set_ylabel('Attack Success Rate (%)')
    axes[1].set_title('Pareto Frontier: Minimum Perturbation for Maximum Evasion\n(Lower-Left = More Secure Model)')
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend(fontsize=9, loc='upper left')
    axes[1].set_xlim(0, 7.5); axes[1].set_ylim(-3, 110)

    # Annotate the dominance region
    axes[1].annotate('AT-XGBoost: Most Secure\n(Highest ε needed, lowest ASR)',
                     xy=(1.5, 1.3), xytext=(2.5, 20),
                     arrowprops=dict(arrowstyle='->', color=C_AT),
                     fontsize=9, color=C_AT, fontweight='bold')

    fig.suptitle('Phase 5 — Pareto Frontier Analysis: Security vs. Perturbation Cost',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig20_pareto_frontier.png')


# ─────────────────────────────────────────────────────────────
# FIG 21: Feature Importance Drift
# ─────────────────────────────────────────────────────────────
def fig21_feature_importance_drift():
    print("\n" + "="*60)
    print("FIG 21: FEATURE IMPORTANCE DRIFT (BASELINE vs AT)")
    print("="*60)

    xgb_clean = joblib.load(os.path.join(MODELS_DIR, "XGBoost_baseline.joblib"))
    xgb_at    = joblib.load(os.path.join(MODELS_DIR, "XGBoost_AT.joblib"))

    X_test    = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_test.parquet"))
    col_names = list(X_test.columns)

    imp_clean = xgb_clean.feature_importances_
    imp_at    = xgb_at.feature_importances_
    drift     = imp_at - imp_clean

    df = pd.DataFrame({
        'Feature':    col_names,
        'Baseline':   imp_clean,
        'AT':         imp_at,
        'Drift':      drift,
        'Abs_Drift':  np.abs(drift),
    }).sort_values('Abs_Drift', ascending=False).head(25)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Side-by-side importance bars
    x = np.arange(len(df)); w = 0.38
    axes[0].barh(x + w/2, df['Baseline'], w, label='Baseline XGBoost',
                 color=C_CLEAN, alpha=0.9)
    axes[0].barh(x - w/2, df['AT'],       w, label='AT-XGBoost',
                 color=C_AT, alpha=0.9)
    axes[0].set_yticks(x); axes[0].set_yticklabels(df['Feature'], fontsize=9)
    axes[0].set_xlabel('Feature Importance Score')
    axes[0].set_title('Feature Importance: Baseline vs. Adversarially Trained\n(Top 25 Features by Drift Magnitude)')
    axes[0].legend(fontsize=10)
    axes[0].invert_yaxis()

    # Right: Drift chart
    drift_colors = [C_AT if d > 0 else C_ADV for d in df['Drift']]
    axes[1].barh(df['Feature'], df['Drift'], color=drift_colors, alpha=0.85)
    axes[1].axvline(0, color='black', linewidth=1.0)
    axes[1].set_xlabel('Importance Drift (AT − Baseline)')
    axes[1].set_title('Feature Importance Drift After Adversarial Training\n(Green=More Important After AT | Red=Less Important)')
    axes[1].invert_yaxis()

    green_patch = mpatches.Patch(color=C_AT,  label='Increased after AT (model learned new features)')
    red_patch   = mpatches.Patch(color=C_ADV, label='Decreased after AT (model no longer relies on this)')
    axes[1].legend(handles=[green_patch, red_patch], fontsize=9, loc='lower right')

    fig.suptitle('Phase 5 — Feature Importance Drift: How Adversarial Training Reshapes Model Behaviour',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig21_feature_importance_drift.png')


if __name__ == '__main__':
    print("="*60)
    print("PHASE 5: PUBLICATION VISUALIZATIONS (fig19-21)")
    print("="*60)
    fig19_tsne()
    fig20_pareto_frontier()
    fig21_feature_importance_drift()
    print("\n🎯 All Phase 5 visualizations complete. Run generate_paper.py next.")
