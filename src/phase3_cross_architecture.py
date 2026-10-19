"""
phase3_cross_architecture.py — Phase 3: Cross-Architecture Robustness Evaluation
Compares adversarial vulnerability of XGBoost (Tree) vs MLP (Deep Learning).

Experiments:
  1. Clean baseline comparison (Tree vs Neural)
  2. GA Attack ASR: XGBoost vs MLP — same attack budget
  3. Epsilon Sensitivity: Both architectures on one graph
  4. Confidence Degradation Curve under attack
  5. Per-Class ASR Heatmap: Tree vs Neural side-by-side
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
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

from config import DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR

# ── Global style ────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.linestyle': ':', 'grid.alpha': 0.6,
})
C_TREE   = '#8b0000'    # dark red   — tree-based
C_NEURAL = '#2c6fad'    # steel blue — neural network
C_ACCENT = '#e07b00'
C_GRAY   = '#555555'

# ── Attack Parameters ───────────────────────────────────────
POPULATION_SIZE = 150
GENERATIONS     = 60
EPSILON_DEFAULT = 5.0
N_PACKETS       = 500
RANDOM_SEED     = 42


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {name}")


def get_mutable_indices(X_test, model):
    """Uses XGBoost's feature importance to select mutable features."""
    importances = model.feature_importances_
    continuous = [(i, col, importances[i]) for i, col in enumerate(X_test.columns)
                  if len(X_test[col].unique()) > 15]
    continuous.sort(key=lambda x: x[2], reverse=True)
    return [x[0] for x in continuous[:15]]


def run_ga(original_packet, mutable_indices, model, col_names,
           epsilon=EPSILON_DEFAULT, population=POPULATION_SIZE, generations=GENERATIONS):
    """Single-packet Multi-Objective GA. Works for any sklearn-compatible model."""
    pop = np.zeros((population, len(original_packet)))
    for p in range(population):
        pop[p, mutable_indices] = np.random.uniform(-epsilon, epsilon, len(mutable_indices))

    best_adv = original_packet.copy()
    evaded   = False
    best_conf = 1.0

    for gen in range(generations):
        candidates = original_packet + pop
        cand_df    = pd.DataFrame(candidates, columns=col_names)
        probs      = model.predict_proba(cand_df)[:, 1]
        l2         = np.linalg.norm(pop[:, mutable_indices], axis=1)
        fitness    = probs + 0.05 * l2

        if np.any(probs < 0.5):
            ok = np.where(probs < 0.5)[0]
            idx = ok[np.argmin(l2[ok])]
            best_adv  = candidates[idx]
            best_conf = probs[idx]
            evaded    = True
            break

        idx_s = np.argsort(fitness)[:population // 2]
        surv  = pop[idx_s]
        nxt   = np.zeros_like(pop)
        nxt[:len(surv)] = surv
        for j in range(len(surv), population):
            p1    = surv[np.random.randint(len(surv))]
            p2    = surv[np.random.randint(len(surv))]
            child = np.where(np.random.rand(len(original_packet)) > 0.5, p1, p2)
            mask  = np.random.rand(len(mutable_indices)) > 0.4
            child[mutable_indices] = np.clip(
                child[mutable_indices] + np.random.normal(0, epsilon/3, len(mutable_indices)) * mask,
                -epsilon, epsilon)
            nxt[j] = child
        pop = nxt

        if np.min(probs) < best_conf:
            best_conf = np.min(probs)
            best_adv  = candidates[np.argmin(probs)]

    return evaded, best_adv, best_conf


def attack_model(model, X_sample, mutable_indices, col_names, epsilon=EPSILON_DEFAULT, label="Model"):
    """Run GA on all packets for a given model. Returns ASR, adv_packets, confidences."""
    evaded_count = 0
    adv_packets  = []
    confidences  = []

    for i, pkt in enumerate(X_sample):
        evaded, adv, conf = run_ga(pkt, mutable_indices, model, col_names, epsilon=epsilon)
        adv_packets.append(adv)
        confidences.append(conf)
        if evaded:
            evaded_count += 1
        if (i + 1) % 100 == 0:
            print(f"    [{label}] {i+1}/{len(X_sample)} | ASR: {evaded_count/(i+1)*100:.1f}%")

    asr = evaded_count / len(X_sample) * 100
    return asr, np.array(adv_packets), np.array(confidences)


# ─────────────────────────────────────────────────────────────
# EXP 1: Clean Baseline Comparison
# ─────────────────────────────────────────────────────────────
def exp1_baseline_comparison(X_test, y_test, models):
    print("\n" + "="*55)
    print("EXP 1: CLEAN BASELINE COMPARISON")
    print("="*55)

    results = {}
    for name, model in models.items():
        pred   = model.predict(X_test)
        acc    = accuracy_score(y_test, pred) * 100
        f1     = f1_score(y_test, pred, average='macro') * 100
        # ROC-AUC
        probs  = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, probs)
        results[name] = {'Accuracy': acc, 'F1': f1, 'AUC': auc, 'probs': probs}
        print(f"  {name}: Acc={acc:.2f}% | F1={f1:.2f}% | AUC={auc:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    arch_names = list(results.keys())
    colors     = [C_TREE if 'XGBoost' in n else C_NEURAL for n in arch_names]

    # Bar chart: Acc + F1
    x = np.arange(len(arch_names)); w = 0.35
    b1 = axes[0].bar(x-w/2, [results[n]['Accuracy'] for n in arch_names], w,
                     label='Accuracy', color=colors, alpha=0.9)
    b2 = axes[0].bar(x+w/2, [results[n]['F1'] for n in arch_names], w,
                     label='F1-Score', color=colors, alpha=0.55, hatch='//')
    axes[0].set_title('Clean Data: Accuracy & F1-Score')
    axes[0].set_xticks(x); axes[0].set_xticklabels(arch_names)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_ylim(95, 102); axes[0].legend()
    for bar in list(b1)+list(b2):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=8)

    # ROC curves
    for name, col in zip(arch_names, colors):
        fpr, tpr, _ = roc_curve(y_test, results[name]['probs'])
        axes[1].plot(fpr, tpr, color=col, linewidth=2,
                     label=f"{name} (AUC={results[name]['AUC']:.4f})")
    axes[1].plot([0,1],[0,1],'k--', linewidth=1, alpha=0.5)
    axes[1].set_title('ROC Curve — Clean Data')
    axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(fontsize=10); axes[1].grid(True, linestyle=':', alpha=0.6)

    fig.suptitle('Phase 3 — Exp.1: XGBoost vs MLP — Clean Baseline Performance',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig9_phase3_baseline_comparison.png')
    return results


# ─────────────────────────────────────────────────────────────
# EXP 2: GA Attack — XGBoost vs MLP
# ─────────────────────────────────────────────────────────────
def exp2_attack_comparison(X_test, y_test, models, mutable_indices, col_names):
    print("\n" + "="*55)
    print("EXP 2: GA ATTACK — XGBOOST vs MLP")
    print("="*55)

    np.random.seed(RANDOM_SEED)
    attack_mask = y_test.values == 1
    benign_mask = y_test.values == 0
    X_attacks = X_test.values[attack_mask][:N_PACKETS]
    X_benign  = X_test.values[benign_mask][:N_PACKETS]
    y_full    = np.concatenate([np.ones(len(X_attacks)), np.zeros(len(X_benign))])

    attack_results = {}
    for name, model in models.items():
        print(f"\n  Attacking {name}...")
        asr, X_adv, confs = attack_model(model, X_attacks, mutable_indices, col_names, label=name)

        # Evaluate metrics
        X_adv_df   = pd.DataFrame(np.concatenate([X_adv, X_benign]), columns=col_names)
        X_clean_df = pd.DataFrame(np.concatenate([X_attacks, X_benign]), columns=col_names)
        clean_pred = model.predict(X_clean_df)
        adv_pred   = model.predict(X_adv_df)

        from sklearn.metrics import recall_score
        clean_recall = recall_score(y_full, clean_pred) * 100
        adv_recall   = recall_score(y_full, adv_pred) * 100

        attack_results[name] = {
            'ASR': asr, 'Clean Recall': clean_recall, 'Adv Recall': adv_recall,
            'Recall Drop': clean_recall - adv_recall, 'confidences': confs,
            'X_adv': X_adv,
        }
        print(f"  [{name}] ASR={asr:.1f}% | Recall: {clean_recall:.1f}%→{adv_recall:.1f}% (Drop: {clean_recall-adv_recall:.1f}%)")

    # Plot
    arch_names = list(attack_results.keys())
    colors     = [C_TREE if 'XGBoost' in n else C_NEURAL for n in arch_names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(arch_names)); w = 0.35

    # ASR
    asrs = [attack_results[n]['ASR'] for n in arch_names]
    bars = axes[0].bar(arch_names, asrs, color=colors, width=0.4, alpha=0.9, zorder=3)
    axes[0].set_title('Attack Success Rate (ASR)')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_ylim(0, 110)
    for bar, v in zip(bars, asrs):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Recall comparison
    clean_recalls = [attack_results[n]['Clean Recall'] for n in arch_names]
    adv_recalls   = [attack_results[n]['Adv Recall']   for n in arch_names]
    axes[1].bar(x-w/2, clean_recalls, w, label='Clean',       color=colors, alpha=0.9)
    axes[1].bar(x+w/2, adv_recalls,   w, label='Adversarial', color=colors, alpha=0.4)
    axes[1].set_title('Recall: Clean vs Adversarial')
    axes[1].set_xticks(x); axes[1].set_xticklabels(arch_names)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_ylim(0, 120); axes[1].legend()

    # Confidence distributions (after attack)
    for name, col in zip(arch_names, colors):
        axes[2].hist(attack_results[name]['confidences'], bins=30, alpha=0.55, color=col, label=name, edgecolor='white')
    axes[2].axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Evasion threshold')
    axes[2].set_title('Post-Attack Model Confidence Distribution')
    axes[2].set_xlabel('Model Confidence (Attack Probability)'); axes[2].set_ylabel('Count')
    axes[2].legend()

    fig.suptitle('Phase 3 — Exp.2: XGBoost (Tree) vs MLP (Neural) Under Identical GA Attack',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig10_phase3_attack_comparison.png')
    return attack_results


# ─────────────────────────────────────────────────────────────
# EXP 3: Epsilon Sensitivity — Both Architectures
# ─────────────────────────────────────────────────────────────
def exp3_epsilon_sensitivity(X_test, y_test, models, mutable_indices, col_names):
    print("\n" + "="*55)
    print("EXP 3: EPSILON SENSITIVITY — BOTH ARCHITECTURES")
    print("="*55)

    np.random.seed(RANDOM_SEED)
    attack_mask = y_test.values == 1
    X_sample    = X_test.values[attack_mask][:200]   # Smaller subset for speed

    epsilons = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    asr_by_model = {name: [] for name in models}

    for eps in epsilons:
        print(f"  ε={eps:.1f}:", end="", flush=True)
        for name, model in models.items():
            evaded = 0
            for pkt in X_sample:
                ev, _, _ = run_ga(pkt, mutable_indices, model, col_names, epsilon=eps,
                                  population=80, generations=40)
                if ev: evaded += 1
            asr = evaded / len(X_sample) * 100
            asr_by_model[name].append(asr)
            print(f"  {name}={asr:.1f}%", end="", flush=True)
        print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    style = {name: (C_TREE, 'o', '-') if 'XGBoost' in name else (C_NEURAL, 's', '--')
             for name in models}

    for name in models:
        col, marker, ls = style[name]
        ax.plot(epsilons, asr_by_model[name], marker=marker, color=col,
                linewidth=2.5, markersize=9, linestyle=ls, label=name, zorder=3)

    # Shade the gap between architectures
    xgb_asr = asr_by_model['XGBoost']
    mlp_asr = asr_by_model['MLP']
    ax.fill_between(epsilons, xgb_asr, mlp_asr,
                    where=[x > m for x, m in zip(xgb_asr, mlp_asr)],
                    alpha=0.12, color=C_TREE, label='Tree Vulnerability Surplus')
    ax.fill_between(epsilons, xgb_asr, mlp_asr,
                    where=[m >= x for x, m in zip(xgb_asr, mlp_asr)],
                    alpha=0.12, color=C_NEURAL, label='Neural Vulnerability Surplus')

    ax.set_xlabel('Perturbation Budget (ε)')
    ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title('Phase 3 — Exp.3: Epsilon Sensitivity — Tree-Based vs Neural Network Architecture')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=10)
    save_fig(fig, 'fig11_phase3_epsilon_sensitivity.png')

    return asr_by_model, epsilons


# ─────────────────────────────────────────────────────────────
# EXP 4: Confidence Degradation Curve
# ─────────────────────────────────────────────────────────────
def exp4_confidence_degradation(X_test, y_test, models, mutable_indices, col_names):
    print("\n" + "="*55)
    print("EXP 4: CONFIDENCE DEGRADATION CURVE")
    print("="*55)

    np.random.seed(RANDOM_SEED)
    attack_mask = y_test.values == 1
    X_sample    = X_test.values[attack_mask][:150]
    epsilons    = np.linspace(0.1, 6.0, 10)

    mean_conf_by_model = {name: [] for name in models}
    std_conf_by_model  = {name: [] for name in models}

    for eps in epsilons:
        print(f"  ε={eps:.2f}:", end="", flush=True)
        for name, model in models.items():
            confs = []
            for pkt in X_sample:
                _, _, conf = run_ga(pkt, mutable_indices, model, col_names, epsilon=eps,
                                    population=60, generations=30)
                confs.append(conf)
            mean_conf_by_model[name].append(np.mean(confs))
            std_conf_by_model[name].append(np.std(confs))
            print(f"  {name}={np.mean(confs):.3f}", end="", flush=True)
        print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    style = {name: (C_TREE, '-') if 'XGBoost' in name else (C_NEURAL, '--') for name in models}

    for name in models:
        col, ls = style[name]
        means = np.array(mean_conf_by_model[name])
        stds  = np.array(std_conf_by_model[name])
        ax.plot(epsilons, means, color=col, linewidth=2.5, linestyle=ls, label=name)
        ax.fill_between(epsilons, means - stds, means + stds, alpha=0.15, color=col)

    ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5, label='Evasion Threshold (0.5)')
    ax.set_xlabel('Perturbation Budget (ε)')
    ax.set_ylabel('Mean Model Confidence (Attack Prob.)')
    ax.set_title('Phase 3 — Exp.4: Confidence Degradation Curve\n(Mean ± 1 Std Dev across 150 packets)')
    ax.legend(fontsize=10)
    ax.text(0.02, 0.08, 'Below this line → evasion achieved',
            transform=ax.transAxes, fontsize=8.5, style='italic', color=C_GRAY)
    save_fig(fig, 'fig12_phase3_confidence_degradation.png')

    return mean_conf_by_model


# ─────────────────────────────────────────────────────────────
# EXP 5: Per-Class ASR Heatmap — XGB vs MLP
# ─────────────────────────────────────────────────────────────
def exp5_per_class_heatmap(X_test, y_test, models, mutable_indices, col_names):
    print("\n" + "="*55)
    print("EXP 5: PER-CLASS ASR HEATMAP — XGB vs MLP")
    print("="*55)

    orig_label_path = os.path.join(DATA_PROCESSED_DIR, "y_test_original_labels.parquet")
    try:
        y_original = pd.read_parquet(orig_label_path)['Original_Label']
    except FileNotFoundError:
        print("  [SKIP] y_test_original_labels.parquet not found.")
        return {}

    per_class_data = {}
    classes = y_original.unique()

    for cls in classes:
        class_mask   = (y_original == cls).values
        true_pos_mask = class_mask & (y_test.values == 1)
        X_class = X_test.values[true_pos_mask]
        if len(X_class) < 10:
            continue

        sample = X_class[:min(80, len(X_class))]
        row    = {'Class': cls}

        for name, model in models.items():
            evaded = sum(
                run_ga(pkt, mutable_indices, model, col_names,
                       population=60, generations=30)[0]
                for pkt in sample
            )
            asr = evaded / len(sample) * 100
            row[name] = round(asr, 1)
            print(f"  [{cls}] {name}: {asr:.1f}%")

        per_class_data[cls] = row

    if not per_class_data:
        return {}

    df_heat = pd.DataFrame(per_class_data.values()).set_index('Class')
    df_heat = df_heat.sort_values(by='XGBoost', ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_heat) * 0.45)))
    sns.heatmap(df_heat, annot=True, fmt='.1f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'Attack Success Rate (%)', 'format': '%.0f%%'})
    ax.set_title('Phase 3 — Exp.5: Per-Class ASR — XGBoost vs MLP\n(Sorted by XGBoost ASR)',
                 fontweight='bold')
    ax.set_xlabel('Architecture'); ax.set_ylabel('Attack Class')
    plt.tight_layout()
    save_fig(fig, 'fig13_phase3_per_class_heatmap.png')

    return df_heat


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("PHASE 3: CROSS-ARCHITECTURE ROBUSTNESS EVALUATION")
    print("XGBoost (Tree-Based) vs MLP (Deep Learning)")
    print("=" * 55)

    # Load data
    X_test  = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_test.parquet"))
    y_test  = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "y_test.parquet")).squeeze()
    col_names = list(X_test.columns)

    # Load models — MLP is saved as a bundle {model, threshold}
    models = {}
    thresholds = {}
    for name in ['XGBoost', 'MLP']:
        path = os.path.join(MODELS_DIR, f"{name}_baseline.joblib")
        raw  = joblib.load(path)
        if isinstance(raw, dict):
            m   = raw['model']
            thresholds[name] = raw['threshold']
            print(f"  Loaded {name} (Youden threshold={raw['threshold']:.4f})")
        else:
            m   = raw
            thresholds[name] = 0.5
            print(f"  Loaded {name} (default threshold=0.50)")
        if hasattr(m, 'n_jobs'):
            m.n_jobs = 1
        models[name] = m


    # Get mutable features from XGBoost importance
    xgb_model = models['XGBoost']
    mutable_indices = get_mutable_indices(X_test, xgb_model)
    print(f"\n  Mutable features: {[X_test.columns[i] for i in mutable_indices[:5]]}...")

    # Run all 5 experiments
    baseline_results = exp1_baseline_comparison(X_test, y_test, models)
    attack_results   = exp2_attack_comparison(X_test, y_test, models, mutable_indices, col_names)
    epsilon_results, epsilons = exp3_epsilon_sensitivity(X_test, y_test, models, mutable_indices, col_names)
    conf_results     = exp4_confidence_degradation(X_test, y_test, models, mutable_indices, col_names)
    class_heatmap    = exp5_per_class_heatmap(X_test, y_test, models, mutable_indices, col_names)

    # Print final summary
    print("\n" + "=" * 55)
    print("PHASE 3 — COMPLETE SUMMARY")
    print("=" * 55)
    for name in models:
        print(f"\n  [{name}]")
        print(f"    Clean Accuracy : {baseline_results[name]['Accuracy']:.2f}%")
        print(f"    Clean AUC-ROC  : {baseline_results[name]['AUC']:.4f}")
        if name in attack_results:
            r = attack_results[name]
            print(f"    ASR (ε=5.0)    : {r['ASR']:.2f}%")
            print(f"    Recall Drop    : {r['Recall Drop']:.2f}%")

    print("\n  Figures saved:")
    for i in range(9, 14):
        tag = {9:'baseline', 10:'attack_comparison', 11:'epsilon_sensitivity',
               12:'confidence_degradation', 13:'per_class_heatmap'}.get(i,'')
        print(f"    fig{i}_phase3_{tag}.png")
    print("\n🎯 Phase 3 Complete. Ready for Phase 4 (Adversarial Training / Defense).")
