"""
phase2_extended_analysis.py — Phase 2 Extended: Full Academic Adversarial Analysis
Implements:
  1. Cross-Model Transferability Matrix
  2. Epsilon Sensitivity Ablation Study
  3. Per-Attack-Class Vulnerability Heatmap
  4. GA vs. Random Baseline Comparison
"""

import os
import joblib
import pandas as pd
import numpy as np
from time import time
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

from config import DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR

# ==========================================
# UNIVERSAL SETTINGS
# ==========================================
POPULATION_SIZE = 200
GENERATIONS = 80
EPSILON_DEFAULT = 5.0
N_PACKETS_PER_EXPERIMENT = 500  # Per experiment slice — keeps each ablation fast
RANDOM_SEED = 42

# ==========================================
# CORE GENETIC EVASION ENGINE (Importable)
# ==========================================
def _run_single_ga(original_packet, mutable_indices, model, epsilon):
    population = np.zeros((POPULATION_SIZE, len(original_packet)))
    for p in range(POPULATION_SIZE):
        noise = np.random.uniform(-epsilon, epsilon, size=len(mutable_indices))
        population[p, mutable_indices] = noise

    best_adv_packet = original_packet.copy()
    evaded = False
    best_confidence = 1.0
    best_l2 = float('inf')

    for gen in range(GENERATIONS):
        candidates = original_packet + population
        probs = model.predict_proba(candidates)[:, 1]
        l2_dists = np.linalg.norm(population[:, mutable_indices], axis=1)

        # Multi-Objective Fitness: Evasion + L2 Stealth
        fitness = probs + 0.05 * l2_dists

        if np.any(probs < 0.5):
            success_idx = np.where(probs < 0.5)[0]
            stealthiest = success_idx[np.argmin(l2_dists[success_idx])]
            best_adv_packet = candidates[stealthiest]
            evaded = True
            best_confidence = probs[stealthiest]
            best_l2 = l2_dists[stealthiest]
            break

        survivors_idx = np.argsort(fitness)[:POPULATION_SIZE // 2]
        survivors = population[survivors_idx]
        next_gen = np.zeros_like(population)
        next_gen[:len(survivors)] = survivors

        for j in range(len(survivors), POPULATION_SIZE):
            p1 = survivors[np.random.randint(len(survivors))]
            p2 = survivors[np.random.randint(len(survivors))]
            child = np.where(np.random.rand(len(original_packet)) > 0.5, p1, p2)
            mask = np.random.rand(len(mutable_indices)) > 0.3
            child[mutable_indices] = np.clip(
                child[mutable_indices] + np.random.normal(0, epsilon/3, len(mutable_indices)) * mask,
                -epsilon, epsilon
            )
            next_gen[j] = child

        population = next_gen
        if np.min(probs) < best_confidence:
            best_confidence = np.min(probs)
            best_adv_packet = candidates[np.argmin(probs)]
            best_l2 = l2_dists[np.argmin(probs)]

    return evaded, best_adv_packet, best_confidence, best_l2


def run_ga_batch(X_sample, mutable_indices, model, epsilon=EPSILON_DEFAULT):
    results = Parallel(n_jobs=-1)(
        delayed(_run_single_ga)(pkt, mutable_indices, model, epsilon) for pkt in X_sample
    )
    asr = sum(r[0] for r in results) / len(results) * 100
    return asr, results


def get_mutable_indices(X_test, model):
    importances = model.feature_importances_
    continuous = [(i, col, importances[i]) for i, col in enumerate(X_test.columns) if len(X_test[col].unique()) > 15]
    continuous.sort(key=lambda x: x[2], reverse=True)
    return [x[0] for x in continuous[:15]], [x[1] for x in continuous[:15]]


# ==========================================
# EXPERIMENT 1: CROSS-MODEL TRANSFERABILITY MATRIX
# ==========================================
def experiment_transferability(X_test, y_test, models_dict):
    print("\n" + "="*60)
    print("EXPERIMENT 1: CROSS-MODEL TRANSFERABILITY MATRIX")
    print("="*60)

    y_test_arr = y_test.values
    y_pred_initial = models_dict["XGBoost"].predict(X_test)
    true_positive_mask = (y_test_arr == 1) & (y_pred_initial == 1)
    X_attacks = X_test[true_positive_mask].values[:N_PACKETS_PER_EXPERIMENT]

    mutable_indices, mutable_names = get_mutable_indices(X_test, models_dict["XGBoost"])

    # Generate adversarial examples using XGBoost as surrogate
    print(f"  Generating adversarial examples using XGBoost as surrogate model...")
    _, ga_results = run_ga_batch(X_attacks, mutable_indices, models_dict["XGBoost"])
    
    adv_packets = np.array([r[1] for r in ga_results])

    # Test transferability to all models
    transfer_matrix = {}
    clean_preds = {}
    adv_preds = {}
    
    for model_name, model in models_dict.items():
        clean_acc = np.mean(model.predict(X_attacks) == 1) * 100
        adv_acc = np.mean(model.predict(adv_packets) == 1) * 100
        transfer_drop = clean_acc - adv_acc
        transfer_matrix[model_name] = transfer_drop
        clean_preds[model_name] = clean_acc
        adv_preds[model_name] = adv_acc
        print(f"  [{model_name}] Clean Acc: {clean_acc:.1f}% → Adversarial Acc: {adv_acc:.1f}% (Drop: {transfer_drop:.1f}%)")

    # Plot Transferability Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    model_names = list(models_dict.keys())
    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, [clean_preds[m] for m in model_names], width, label='Clean Data', color='steelblue')
    bars2 = ax.bar(x + width/2, [adv_preds[m] for m in model_names], width, label='Adversarial Data', color='#8b0000', alpha=0.85)

    ax.set_title("Cross-Model Transferability: Detection Rate Under Attack", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Detection Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.rcParams.update({'font.family': 'serif'})
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "transferability_matrix.png")
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Transferability graph saved: {path}")
    return transfer_matrix


# ==========================================
# EXPERIMENT 2: EPSILON SENSITIVITY ABLATION
# ==========================================
def experiment_epsilon_sensitivity(X_test, y_test, model):
    print("\n" + "="*60)
    print("EXPERIMENT 2: EPSILON SENSITIVITY ABLATION STUDY")
    print("="*60)

    y_pred_initial = model.predict(X_test)
    true_positive_mask = (y_test.values == 1) & (y_pred_initial == 1)
    X_attacks = X_test[true_positive_mask].values[:N_PACKETS_PER_EXPERIMENT]
    mutable_indices, _ = get_mutable_indices(X_test, model)
    
    epsilons = np.round(np.linspace(0.1, 6.0, 10), 2)
    asr_results = []

    for eps in epsilons:
        print(f"  Testing ε={eps:.2f}...", end="", flush=True)
        asr, _ = run_ga_batch(X_attacks, mutable_indices, model, epsilon=eps)
        asr_results.append(asr)
        print(f" ASR={asr:.1f}%")

    # Plot Epsilon Sensitivity Curve
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, asr_results, marker='o', color='#8b0000', linewidth=2, markersize=8)
    plt.fill_between(epsilons, 0, asr_results, alpha=0.15, color='#8b0000')
    plt.xlabel("Perturbation Budget (ε)")
    plt.ylabel("Attack Success Rate (%)")
    plt.title("Ablation Study: ASR vs. Perturbation Budget (ε)", fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    path = os.path.join(FIGURES_DIR, "epsilon_sensitivity.png")
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Epsilon Sensitivity graph saved: {path}")
    return dict(zip(epsilons.tolist(), asr_results))


# ==========================================
# EXPERIMENT 3: PER-CLASS VULNERABILITY HEATMAP
# ==========================================
def experiment_per_class_vulnerability(X_test, y_test, model, data_processed_dir):
    print("\n" + "="*60)
    print("EXPERIMENT 3: PER-ATTACK-CLASS VULNERABILITY ANALYSIS")
    print("="*60)

    # Load the dedicated original labels file saved by updated data_loader.py
    orig_label_path = os.path.join(data_processed_dir, "y_test_original_labels.parquet")
    try:
        y_original = pd.read_parquet(orig_label_path)
    except FileNotFoundError:
        print("  [SKIP] y_test_original_labels.parquet not found.")
        print("  Please re-run: python src/data_loader.py and python src/train_baseline.py")
        return {}

    mutable_indices, _ = get_mutable_indices(X_test, model)
    y_pred = model.predict(X_test)

    per_class_asr = {}
    
    for col in y_original.columns:
        if col.lower() in ['binary_label', 'label']:
            continue
        label_col = col
        break
    else:
        print("  [SKIP] No multiclass label column found in y_test parquet.")
        return {}

    classes = y_original[label_col].unique()

    for cls in classes:
        class_mask = (y_original[label_col] == cls).values
        true_pos_mask = class_mask & (y_pred == 1)
        X_class = X_test.values[true_pos_mask]

        if len(X_class) < 10:
            continue

        sample = X_class[:min(100, len(X_class))]
        asr, _ = run_ga_batch(sample, mutable_indices, model)
        per_class_asr[cls] = asr
        print(f"  [{cls}] ASR: {asr:.1f}%")

    if per_class_asr:
        sorted_classes = dict(sorted(per_class_asr.items(), key=lambda x: x[1], reverse=True))
        plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
        fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_classes) * 0.5)))
        
        bars = ax.barh(list(sorted_classes.keys()), list(sorted_classes.values()), color='#8b0000', alpha=0.8)
        ax.set_xlabel("Attack Success Rate (%)")
        ax.set_title("Per-Attack-Class Evasion Vulnerability", fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.grid(axis='x', linestyle=':', alpha=0.7)
        
        for bar, val in zip(bars, sorted_classes.values()):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "per_class_vulnerability.png")
        plt.savefig(path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Per-Class Vulnerability chart saved: {path}")

    return per_class_asr


# ==========================================
# EXPERIMENT 4: GA vs. RANDOM BASELINE
# ==========================================
def experiment_ga_vs_random(X_test, y_test, model):
    print("\n" + "="*60)
    print("EXPERIMENT 4: GA vs. RANDOM NOISE BASELINE")
    print("="*60)

    y_pred_initial = model.predict(X_test)
    true_positive_mask = (y_test.values == 1) & (y_pred_initial == 1)
    X_attacks = X_test[true_positive_mask].values[:N_PACKETS_PER_EXPERIMENT]
    mutable_indices, _ = get_mutable_indices(X_test, model)

    epsilons = np.round(np.linspace(0.5, 5.0, 6), 1)
    ga_asr = []
    random_asr = []

    for eps in epsilons:
        # GA Attack
        asr_ga, _ = run_ga_batch(X_attacks, mutable_indices, model, epsilon=eps)
        ga_asr.append(asr_ga)

        # Random Noise Baseline (just add random uniform noise — no evolution)
        random_evaded = 0
        for pkt in X_attacks:
            for _ in range(50):  # 50 random attempts per packet
                candidate = pkt.copy()
                noise = np.random.uniform(-eps, eps, size=len(mutable_indices))
                candidate[mutable_indices] += noise
                prob = model.predict_proba([candidate])[0, 1]
                if prob < 0.5:
                    random_evaded += 1
                    break
        random_asr.append(random_evaded / len(X_attacks) * 100)
        print(f"  ε={eps:.1f} | GA ASR: {asr_ga:.1f}% | Random ASR: {random_evaded/len(X_attacks)*100:.1f}%")

    # Plot Comparison
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, ga_asr, marker='o', color='#8b0000', linewidth=2, label='Multi-Objective GA (Ours)')
    plt.plot(epsilons, random_asr, marker='s', color='steelblue', linewidth=2, linestyle='--', label='Random Noise Baseline')
    plt.xlabel("Perturbation Budget (ε)")
    plt.ylabel("Attack Success Rate (%)")
    plt.title("Multi-Objective GA vs. Random Baseline Evasion", fontsize=14, fontweight='bold', pad=15)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "ga_vs_random.png")
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  ✅ GA vs. Random Baseline graph saved: {path}")
    return ga_asr, random_asr


# ==========================================
# MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("PHASE 2 EXTENDED ANALYSIS: FULL ACADEMIC SUITE")
    print("="*60)
    t_total = time()

    # Load everything
    X_test = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "y_test.parquet")).squeeze()

    models_dict = {}
    for name in ["RandomForest", "XGBoost", "LightGBM"]:
        path = os.path.join(MODELS_DIR, f"{name}_baseline.joblib")
        if os.path.exists(path):
            models_dict[name] = joblib.load(path)

    xgb_model = models_dict["XGBoost"]

    # Run all 4 experiments
    transfer_results = experiment_transferability(X_test, y_test, models_dict)
    epsilon_results  = experiment_epsilon_sensitivity(X_test, y_test, xgb_model)
    class_results    = experiment_per_class_vulnerability(X_test, y_test, xgb_model, DATA_PROCESSED_DIR)
    ga_asr, rand_asr = experiment_ga_vs_random(X_test, y_test, xgb_model)

    print("\n" + "="*60)
    print(f"✅ FULL ANALYSIS COMPLETE in {(time()-t_total)/60:.1f} minutes")
    print("Graphs saved to:", FIGURES_DIR)
    print("="*60)
