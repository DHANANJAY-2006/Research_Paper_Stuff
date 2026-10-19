"""
phase4_defense_evaluation.py — Phase 4: Defense Effectiveness Analysis
Compares Clean Models vs Adversarially-Trained (AT) Models across 5 experiments.

Experiments:
  1. Clean Accuracy Preservation — Does AT preserve detection on normal traffic?
  2. ASR Before vs After Defense — Core robustness improvement metric
  3. Security-Utility Tradeoff Curve — The canonical defense evaluation plot
  4. Per-Class Defense Effectiveness — Which attack types benefit most from AT?
  5. Cross-Architecture Defense Transfer — Does XGB-AT defense generalise to MLP?
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
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              roc_curve, recall_score, precision_score)
import warnings
warnings.filterwarnings('ignore')

from config import DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR

# ── Global Style ────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.linestyle': ':', 'grid.alpha': 0.6,
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})
C_CLEAN  = '#2c6fad'    # steel blue — original clean model
C_AT     = '#2e8b57'    # forest green — adversarially trained model
C_ADV    = '#8b0000'    # dark red — adversarial attack
C_ACCENT = '#e07b00'    # amber — highlights
C_GRAY   = '#555555'

POPULATION = 100
GENERATIONS = 50
EPSILON     = 5.0
N_PACKETS   = 300
RANDOM_SEED = 42


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved: {name}")


def load_model(name):
    """Load model. Handles both plain models and MLP bundles {model, threshold}."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    raw  = joblib.load(path)
    if isinstance(raw, dict):
        m, thr = raw['model'], raw['threshold']
    else:
        m, thr = raw, 0.5
    if hasattr(m, 'n_jobs'):
        m.n_jobs = 1
    return m, thr


def predict_with_threshold(model, X_df, threshold):
    probs = model.predict_proba(X_df)[:, 1]
    return (probs >= threshold).astype(int), probs


def get_mutable_indices(X_test, xgb_model):
    importances = xgb_model.feature_importances_
    cont = [(i, col, importances[i]) for i, col in enumerate(X_test.columns)
            if len(X_test[col].unique()) > 15]
    cont.sort(key=lambda x: x[2], reverse=True)
    return [x[0] for x in cont[:15]]


def run_ga(pkt, mutable_idx, model, col_names, threshold=0.5,
           pop=POPULATION, gens=GENERATIONS, eps=EPSILON):
    """Single-packet GA returning (evaded, adv_packet, final_confidence)."""
    population = np.zeros((pop, len(pkt)))
    for p in range(pop):
        population[p, mutable_idx] = np.random.uniform(-eps, eps, len(mutable_idx))

    best_adv  = pkt.copy()
    best_conf = 1.0
    evaded    = False

    for _ in range(gens):
        cands  = pkt + population
        df_c   = pd.DataFrame(cands, columns=col_names)
        probs  = model.predict_proba(df_c)[:, 1]
        l2     = np.linalg.norm(population[:, mutable_idx], axis=1)
        fit    = probs + 0.05 * l2

        below = np.where(probs < threshold)[0]
        if len(below) > 0:
            idx       = below[np.argmin(l2[below])]
            best_adv  = cands[idx]
            best_conf = probs[idx]
            evaded    = True
            break

        s_idx = np.argsort(fit)[:pop // 2]
        surv  = population[s_idx]
        nxt   = np.zeros_like(population)
        nxt[:len(surv)] = surv
        for j in range(len(surv), pop):
            p1    = surv[np.random.randint(len(surv))]
            p2    = surv[np.random.randint(len(surv))]
            child = np.where(np.random.rand(len(pkt)) > 0.5, p1, p2)
            mask  = np.random.rand(len(mutable_idx)) > 0.4
            child[mutable_idx] = np.clip(
                child[mutable_idx] + np.random.normal(0, eps/3, len(mutable_idx))*mask,
                -eps, eps)
            nxt[j] = child
        population = nxt
        best_conf = min(best_conf, np.min(probs))

    return evaded, best_adv, best_conf


def measure_asr(model, X_attacks, mutable_idx, col_names, threshold=0.5, label=''):
    evaded = 0
    adv_packets = []
    for i, pkt in enumerate(X_attacks):
        ev, adv, _ = run_ga(pkt, mutable_idx, model, col_names, threshold)
        adv_packets.append(adv)
        if ev: evaded += 1
        if (i+1) % 100 == 0:
            print(f"    [{label}] {i+1}/{len(X_attacks)} | ASR: {evaded/(i+1)*100:.1f}%")
    asr = evaded / len(X_attacks) * 100
    return asr, np.array(adv_packets)


# ─────────────────────────────────────────────────────────────
# EXP 1 & 2: Clean Accuracy + ASR Before vs After AT
# ─────────────────────────────────────────────────────────────
def exp1_2_core_comparison(X_test, y_test, models_dict, mutable_idx, col_names):
    print("\n" + "="*60)
    print("EXP 1+2: CLEAN ACCURACY & ASR — CLEAN vs AT MODELS")
    print("="*60)

    np.random.seed(RANDOM_SEED)
    atk_mask   = y_test.values == 1
    ben_mask   = y_test.values == 0
    X_atk      = X_test.values[atk_mask][:N_PACKETS]
    X_ben      = X_test.values[ben_mask][:N_PACKETS]
    y_full     = np.concatenate([np.ones(len(X_atk)), np.zeros(len(X_ben))])
    X_clean_df = pd.DataFrame(np.concatenate([X_atk, X_ben]), columns=col_names)

    results = {}
    for mname, (model, thr) in models_dict.items():
        print(f"\n  Evaluating [{mname}]...")
        # Clean metrics
        preds_clean, probs_clean = predict_with_threshold(model, X_clean_df, thr)
        clean_recall = recall_score(y_full, preds_clean) * 100
        clean_f1     = f1_score(y_full, preds_clean, average='macro') * 100
        clean_auc    = roc_auc_score(y_full, probs_clean)

        # ASR
        asr, X_adv = measure_asr(model, X_atk, mutable_idx, col_names, thr, label=mname)
        X_adv_df   = pd.DataFrame(np.concatenate([X_adv, X_ben]), columns=col_names)
        preds_adv, _ = predict_with_threshold(model, X_adv_df, thr)
        adv_recall   = recall_score(y_full, preds_adv) * 100

        results[mname] = {
            'Clean Recall': clean_recall, 'Adv Recall': adv_recall,
            'Recall Drop':  clean_recall - adv_recall,
            'Clean F1':     clean_f1, 'Clean AUC': clean_auc,
            'ASR': asr,
        }
        print(f"    Clean Recall: {clean_recall:.2f}% | Adv Recall: {adv_recall:.2f}% | ASR: {asr:.2f}%")

    # ── Figure: 3-panel comparison ───────────────────────────
    labels      = list(results.keys())
    is_at       = ['AT' in l for l in labels]
    bar_colors  = [C_AT if a else C_CLEAN for a in is_at]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    x = np.arange(len(labels)); w = 0.4

    # Panel 1: Clean Recall
    bars = axes[0].bar(labels, [results[l]['Clean Recall'] for l in labels],
                       color=bar_colors, width=w, alpha=0.9, zorder=3)
    axes[0].set_title('Clean Data Recall\n(Must Stay High)')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_ylim(85, 103)
    for bar in bars:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                     f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=10)

    # Panel 2: ASR Before vs After
    asrs = [results[l]['ASR'] for l in labels]
    bars = axes[1].bar(labels, asrs, color=bar_colors, width=w, alpha=0.9, zorder=3)
    axes[1].set_title('Attack Success Rate (ASR)\n(Must Decrease After AT)')
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_ylim(0, 120)
    for bar, val in zip(bars, asrs):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=10)

    # Panel 3: Adv Recall
    adv_recalls = [results[l]['Adv Recall'] for l in labels]
    bars = axes[2].bar(labels, adv_recalls, color=bar_colors, width=w, alpha=0.9, zorder=3)
    axes[2].set_title('Recall Under Adversarial Attack\n(Must Increase After AT)')
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[2].set_ylim(0, 120)
    for bar, val in zip(bars, adv_recalls):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=10)

    legend_handles = [
        mpatches.Patch(color=C_CLEAN, label='Original Model'),
        mpatches.Patch(color=C_AT,    label='Adversarially-Trained (AT)'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle('Phase 4 — Exp.1&2: Defense Effectiveness — Clean Model vs Adversarially-Trained',
                 fontsize=13, fontweight='bold', y=1.08)
    plt.tight_layout()
    save_fig(fig, 'fig14_phase4_core_comparison.png')
    return results


# ─────────────────────────────────────────────────────────────
# EXP 3: Security-Utility Tradeoff Curve
# ─────────────────────────────────────────────────────────────
def exp3_security_utility_tradeoff(X_train, X_test, y_train, y_test,
                                    X_adv_train, y_adv_train, mutable_idx, col_names):
    print("\n" + "="*60)
    print("EXP 3: SECURITY-UTILITY TRADEOFF CURVE")
    print("="*60)

    from xgboost import XGBClassifier
    from adversarial_training import build_augmented_dataset, get_xgb_params

    np.random.seed(RANDOM_SEED)
    atk_mask = y_test.values == 1
    X_atk    = X_test.values[atk_mask][:200]

    rates        = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    clean_accs   = []
    asrs         = []

    for rate in rates:
        print(f"  Rate={rate*100:.0f}%:", end='', flush=True)
        if rate == 0.0:
            X_tr, y_tr = X_train, y_train
        else:
            X_tr, y_tr = build_augmented_dataset(
                X_train, y_train, X_adv_train, y_adv_train, rate)

        model = XGBClassifier(**get_xgb_params())
        model.fit(X_tr, y_tr)
        model.n_jobs = 1

        # Clean accuracy
        preds_clean = model.predict(X_test)
        clean_accs.append(accuracy_score(y_test, preds_clean) * 100)

        # ASR (small sample for speed)
        evaded = sum(run_ga(pkt, mutable_idx, model, col_names)[0] for pkt in X_atk)
        asr    = evaded / len(X_atk) * 100
        asrs.append(asr)
        print(f" Clean={clean_accs[-1]:.2f}% | ASR={asr:.1f}%")

    # Save data
    df = pd.DataFrame({'aug_rate': rates, 'clean_accuracy': clean_accs, 'asr': asrs})
    df.to_csv(os.path.join(FIGURES_DIR, "security_utility_tradeoff.csv"), index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    l1 = ax1.plot([r*100 for r in rates], clean_accs, marker='o', color=C_CLEAN,
                  linewidth=2.5, markersize=9, label='Clean Accuracy (Utility)')
    l2 = ax2.plot([r*100 for r in rates], asrs, marker='s', color=C_ADV,
                  linewidth=2.5, markersize=9, linestyle='--', label='ASR (Security)')

    # Shade optimal zone
    opt_rate_idx = np.argmin(np.array(asrs) + (100 - np.array(clean_accs)))
    ax1.axvline(x=rates[opt_rate_idx]*100, color=C_ACCENT, linestyle=':', linewidth=2,
                label=f'Optimal Rate ({rates[opt_rate_idx]*100:.0f}%)')

    ax1.set_xlabel('Adversarial Augmentation Rate (%)')
    ax1.set_ylabel('Clean Accuracy (%)', color=C_CLEAN)
    ax2.set_ylabel('Attack Success Rate (%)', color=C_ADV)
    ax1.tick_params(axis='y', labelcolor=C_CLEAN)
    ax2.tick_params(axis='y', labelcolor=C_ADV)
    ax1.set_ylim(95, 101)
    ax2.set_ylim(0, 110)

    lines = l1 + l2
    labs  = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='center right', fontsize=10)
    ax1.set_title('Phase 4 — Exp.3: Security-Utility Tradeoff\n'
                  'Adversarial Augmentation Rate vs. Clean Accuracy & ASR')
    ax1.grid(True, linestyle=':', alpha=0.6)
    save_fig(fig, 'fig15_phase4_security_utility_tradeoff.png')

    return rates, clean_accs, asrs


# ─────────────────────────────────────────────────────────────
# EXP 4: Per-Class Defense Effectiveness Heatmap
# ─────────────────────────────────────────────────────────────
def exp4_per_class_defense(X_test, y_test, models_dict, mutable_idx, col_names):
    print("\n" + "="*60)
    print("EXP 4: PER-CLASS DEFENSE EFFECTIVENESS")
    print("="*60)

    orig_path = os.path.join(DATA_PROCESSED_DIR, "y_test_original_labels.parquet")
    try:
        y_orig = pd.read_parquet(orig_path)['Original_Label']
    except FileNotFoundError:
        print("  [SKIP] y_test_original_labels.parquet not found.")
        return

    target_models = {k: v for k, v in models_dict.items() if 'XGBoost' in k}
    per_class_rows = []

    for cls in y_orig.unique():
        cls_mask = (y_orig == cls).values & (y_test.values == 1)
        X_cls    = X_test.values[cls_mask]
        if len(X_cls) < 10:
            continue
        X_sample = X_cls[:min(60, len(X_cls))]

        row = {'Class': cls}
        for mname, (model, thr) in target_models.items():
            evaded = sum(run_ga(pkt, mutable_idx, model, col_names, thr,
                                pop=60, gens=30)[0] for pkt in X_sample)
            row[mname] = round(evaded / len(X_sample) * 100, 1)
        per_class_rows.append(row)
        print(f"  [{cls}] " + " | ".join([f"{k}={v['val']:.1f}%" for k, v in
                                           {m: {'val': row[m]} for m in target_models}.items()]))

    if not per_class_rows:
        return

    df_heat = pd.DataFrame(per_class_rows).set_index('Class')
    df_heat = df_heat.rename(columns={'XGBoost': 'Clean XGBoost', 'XGBoost_AT': 'AT-XGBoost'})
    df_heat['ASR Reduction'] = df_heat['Clean XGBoost'] - df_heat.get('AT-XGBoost', 0)
    df_heat = df_heat.sort_values('ASR Reduction', ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_heat)*0.42)))
    plot_cols = [c for c in df_heat.columns if 'Reduction' not in c]
    sns.heatmap(df_heat[plot_cols], annot=True, fmt='.1f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'ASR (%)', 'format': '%.0f%%'})
    ax.set_title('Phase 4 — Exp.4: Per-Class ASR — Clean vs AT Model\n(Sorted by ASR Reduction)',
                 fontweight='bold')
    ax.set_xlabel('Model'); ax.set_ylabel('Attack Class')
    plt.tight_layout()
    save_fig(fig, 'fig16_phase4_per_class_defense.png')


# ─────────────────────────────────────────────────────────────
# EXP 5: Cross-Architecture Defense Transfer
# ─────────────────────────────────────────────────────────────
def exp5_cross_architecture_transfer(X_test, y_test, models_dict, mutable_idx, col_names):
    print("\n" + "="*60)
    print("EXP 5: CROSS-ARCHITECTURE DEFENSE TRANSFERABILITY")
    print("="*60)
    """
    Tests: Does adversarial training on XGBoost make XGBoost-crafted
    adversarial examples less effective against MLP too?
    And vice versa.
    """
    np.random.seed(RANDOM_SEED)
    atk_mask = y_test.values == 1
    X_atk    = X_test.values[atk_mask][:200]

    # Generate adversarial examples from each surrogate model
    surrogate_names = [n for n in models_dict if '_AT' not in n]
    target_names    = list(models_dict.keys())

    transfer_matrix = {}

    for surrogate in surrogate_names:
        s_model, s_thr = models_dict[surrogate]
        print(f"\n  Generating adversarial examples using [{surrogate}] as surrogate...")
        _, X_adv_s = measure_asr(s_model, X_atk, mutable_idx, col_names, s_thr, surrogate)

        transfer_matrix[surrogate] = {}
        X_adv_df = pd.DataFrame(X_adv_s, columns=col_names)

        for target in target_names:
            t_model, t_thr = models_dict[target]
            preds = predict_with_threshold(t_model, X_adv_df, t_thr)[0]
            # ASR = fraction predicted as benign (0) on adversarial attack traffic
            asr   = (preds == 0).sum() / len(preds) * 100
            transfer_matrix[surrogate][target] = round(asr, 1)
            print(f"    [{surrogate}→{target}] Transfer ASR: {asr:.1f}%")

    df_transfer = pd.DataFrame(transfer_matrix).T
    df_transfer.to_csv(os.path.join(FIGURES_DIR, "cross_architecture_transfer.csv"))

    fig, ax = plt.subplots(figsize=(max(8, len(df_transfer.columns)*1.2),
                                     max(5, len(df_transfer)*1.0)))
    sns.heatmap(df_transfer, annot=True, fmt='.1f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'Transfer ASR (%)', 'format': '%.0f%%'})
    ax.set_title('Phase 4 — Exp.5: Cross-Architecture Defense Transferability\n'
                 'Row=Surrogate (attacker\'s model) | Col=Target (defender\'s model)',
                 fontweight='bold')
    ax.set_xlabel('Target Model'); ax.set_ylabel('Surrogate Model')
    plt.tight_layout()
    save_fig(fig, 'fig17_phase4_cross_architecture_transfer.png')
    return df_transfer


# ─────────────────────────────────────────────────────────────
# EXP 6: Radar / Spider Chart — Full Defense Profile
# ─────────────────────────────────────────────────────────────
def exp6_radar_chart(results_dict):
    print("\n" + "="*60)
    print("EXP 6: DEFENSE RADAR CHART")
    print("="*60)

    # Metrics (all higher = better, so invert ASR and Recall Drop)
    categories = ['Clean Recall', 'Adv Recall', 'Robustness\n(100-ASR)', 'F1-Score', 'AUC×100']
    model_data = {}

    for mname, r in results_dict.items():
        model_data[mname] = [
            r['Clean Recall'],
            r['Adv Recall'],
            100 - r['ASR'],
            r['Clean F1'],
            r['Clean AUC'] * 100,
        ]

    N     = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors  = {
        'XGBoost':    (C_CLEAN, '-',  'XGBoost (Clean)'),
        'XGBoost_AT': (C_AT,    '-',  'XGBoost (AT)'),
        'MLP':        (C_ADV,   '--', 'MLP (Clean)'),
        'MLP_AT':     ('#9b59b6', '--', 'MLP (AT)'),
    }

    for mname, vals in model_data.items():
        col, ls, label = colors.get(mname, ('#aaaaaa', '-', mname))
        v = vals + vals[:1]
        ax.plot(angles, v, color=col, linewidth=2, linestyle=ls, label=label)
        ax.fill(angles, v, color=col, alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
    ax.set_title('Phase 4 — Defense Profile Radar Chart\nAll Models: Clean vs Adversarially Trained',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
    save_fig(fig, 'fig18_phase4_radar_defense_profile.png')


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 4: DEFENSE EVALUATION — 6 EXPERIMENTS")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # Load data
    X_test  = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_test.parquet"))
    y_test  = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "y_test.parquet")).squeeze()
    X_train = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "y_train.parquet")).squeeze()
    col_names = list(X_test.columns)

    # Load adversarial training data
    adv_path = os.path.join(DATA_PROCESSED_DIR, "X_adversarial_sample.parquet")
    X_adv_train = pd.read_parquet(adv_path).reindex(columns=col_names, fill_value=0)
    y_adv_train = pd.Series(np.ones(len(X_adv_train), dtype=int))

    # Load all 4 models
    models_dict = {}
    for mname in ['XGBoost_baseline', 'XGBoost_AT', 'MLP_baseline', 'MLP_AT']:
        try:
            m, thr = load_model(mname)
            key    = mname.replace('_baseline', '')
            models_dict[key] = (m, thr)
            print(f"  Loaded {key} (threshold={thr:.4f})")
        except FileNotFoundError:
            print(f"  [SKIP] {mname}.joblib not found — run adversarial_training.py first")

    if len(models_dict) < 2:
        print("ERROR: Not enough models loaded. Run adversarial_training.py first.")
        exit(1)

    # Get mutable indices from XGBoost
    xgb_model = models_dict['XGBoost'][0]
    mutable_idx = get_mutable_indices(X_test, xgb_model)
    print(f"\n  Top mutable features: {[col_names[i] for i in mutable_idx[:5]]}...")

    # Run experiments
    core_results = exp1_2_core_comparison(X_test, y_test, models_dict, mutable_idx, col_names)

    exp3_security_utility_tradeoff(
        X_train, X_test, y_train, y_test, X_adv_train, y_adv_train, mutable_idx, col_names)

    exp4_per_class_defense(X_test, y_test, models_dict, mutable_idx, col_names)

    exp5_cross_architecture_transfer(X_test, y_test, models_dict, mutable_idx, col_names)

    exp6_radar_chart(core_results)

    # Final summary table
    print("\n" + "="*60)
    print("PHASE 4 — FINAL DEFENSE SUMMARY")
    print("="*60)
    rows = []
    for mname, r in core_results.items():
        rows.append({
            'Model':         mname,
            'Clean Recall':  f"{r['Clean Recall']:.2f}%",
            'Adv Recall':    f"{r['Adv Recall']:.2f}%",
            'Recall Drop':   f"{r['Recall Drop']:.2f}%",
            'ASR':           f"{r['ASR']:.2f}%",
        })
    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(os.path.join(FIGURES_DIR, "phase4_defense_summary.csv"), index=False)
    print(f"\n🎯 Phase 4 Complete. Figures fig14–fig18 saved.")
    print(f"   Ready for Phase 5: Paper Writing & LaTeX Report.")
