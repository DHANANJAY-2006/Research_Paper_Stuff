"""
genetic_evasion.py — Phase 2: Multi-Objective Adversarial Evasion (PhD Level)
Implements a massive-scale, parallelized Multi-Objective Genetic Algorithm 
targeting both Evasion Success and Perturbation Stealth (L2-Norm).
"""

import os
import joblib
import pandas as pd
import numpy as np
from time import time
import warnings
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

from config import DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR

# ==========================================
# HYPERPARAMETERS FOR RIGOROUS RESEARCH
# ==========================================
NUM_ATTACKS_TO_SIMULATE = 2000  # Large sample size for statistical significance
POPULATION_SIZE = 300           # Massive evolutionary pool
GENERATIONS = 100               # Deep evolutionary depth
EPSILON = 5.0                   # Max absolute feature change

def evaluate_packet(original_packet, mutable_indices, model):
    """
    Worker function to process a single network packet through the GA.
    Designed for parallel execution across CPU cores.
    """
    population = np.zeros((POPULATION_SIZE, len(original_packet)))
    for p in range(POPULATION_SIZE):
        noise = np.random.uniform(-EPSILON, EPSILON, size=len(mutable_indices))
        population[p, mutable_indices] = noise
        
    best_adv_packet = None
    evaded = False
    best_confidence = 1.0
    best_l2_dist = float('inf')
    
    for gen in range(GENERATIONS):
        candidates = original_packet + population
        probs = model.predict_proba(candidates)[:, 1]
        
        # Multi-Objective Fitness: Combine Confidence (Lower is better) + L2 Distance (Lower is better)
        l2_distances = np.linalg.norm(population[:, mutable_indices], axis=1)
        # We heavily weight evasion (probs), but use L2 distance as a tie-breaker/penalty
        fitness = probs + (0.05 * l2_distances)
        
        # Check for evasion
        if np.any(probs < 0.5):
            evasion_indices = np.where(probs < 0.5)[0]
            # Out of all successful evasions, pick the one with the smallest L2 distance (Stealthiest)
            stealthiest_idx = evasion_indices[np.argmin(l2_distances[evasion_indices])]
            best_adv_packet = candidates[stealthiest_idx]
            evaded = True
            best_confidence = probs[stealthiest_idx]
            best_l2_dist = l2_distances[stealthiest_idx]
            break # Evasion successful, stop evolving this packet
            
        # Evolution: Select best 50% based on fitness
        best_indices = np.argsort(fitness)[:POPULATION_SIZE // 2]
        survivors = population[best_indices]
        
        next_gen = np.zeros_like(population)
        next_gen[:len(survivors)] = survivors
        
        # Breed & Mutate
        for j in range(len(survivors), POPULATION_SIZE):
            parent1 = survivors[np.random.randint(len(survivors))]
            parent2 = survivors[np.random.randint(len(survivors))]
            
            child = np.where(np.random.rand(len(original_packet)) > 0.5, parent1, parent2)
            
            mutation_mask = np.random.rand(len(mutable_indices)) > 0.3 # Higher mutation chance
            noise = np.random.normal(0, EPSILON/3, size=len(mutable_indices)) * mutation_mask
            child[mutable_indices] = np.clip(child[mutable_indices] + noise, -EPSILON, EPSILON)
            
            next_gen[j] = child
            
        population = next_gen
        
        # Track the best non-evading packet just in case
        if np.min(probs) < best_confidence:
            best_confidence = np.min(probs)
            best_idx = np.argmin(probs)
            best_adv_packet = candidates[best_idx]
            best_l2_dist = l2_distances[best_idx]

    return evaded, best_adv_packet, best_confidence, best_l2_dist

def run_evasion_attack():
    print("="*60)
    print("PHASE 2: MULTI-OBJECTIVE ADVERSARIAL EVASION (PhD LEVEL)")
    print("="*60)

    try:
        X_test = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_test.parquet"))
        y_test = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "y_test.parquet")).squeeze()
        model = joblib.load(os.path.join(MODELS_DIR, "XGBoost_baseline.joblib"))
    except FileNotFoundError:
        print("[ERROR] Missing models or data.")
        return

    # WHITE-BOX FEATURE TARGETING
    importances = model.feature_importances_
    continuous_features = []
    for i, col in enumerate(X_test.columns):
        if len(X_test[col].unique()) > 15:
            continuous_features.append((i, col, importances[i]))
            
    continuous_features.sort(key=lambda x: x[2], reverse=True)
    mutable_indices = [x[0] for x in continuous_features[:15]]
    mutable_names = [x[1] for x in continuous_features[:15]]
    
    print(f"\n[INFO] Targeting top {len(mutable_indices)} continuous features:")
    for name, idx in zip(mutable_names, mutable_indices):
        print(f"  -> {name} (Importance: {importances[idx]:.4f})")

    # ISOLATE TARGETS
    y_pred_initial = model.predict(X_test)
    true_positive_mask = (y_test.values == 1) & (y_pred_initial == 1)
    X_attacks = X_test[true_positive_mask].values
    
    attacks_to_run = min(NUM_ATTACKS_TO_SIMULATE, len(X_attacks))
    np.random.seed(42)
    X_sample = X_attacks[np.random.choice(len(X_attacks), attacks_to_run, replace=False)]

    print(f"\n[INFO] Launching Parallel Multi-Objective GA on {attacks_to_run} packets...")
    print(f"[INFO] Evaluating {POPULATION_SIZE * GENERATIONS} mutations per packet.")
    print(f"[INFO] This will use 100% of your CPU cores. Please wait...\n")

    t0 = time()
    
    # PARALLEL EXECUTION: Utilize all CPU cores
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_packet)(packet, mutable_indices, model) for packet in X_sample
    )

    # Aggregate Results
    successful_evasions = 0
    adversarial_examples = []
    plot_confidences = []
    plot_distances = []

    for evaded, adv_packet, conf, dist in results:
        adversarial_examples.append(adv_packet)
        plot_confidences.append(conf)
        if evaded:
            successful_evasions += 1
            plot_distances.append(dist)

    attack_time = time() - t0
    asr = (successful_evasions / attacks_to_run) * 100

    # ==========================================
    # ACADEMIC PLOTTING: PARETO FRONTIER & DISTRIBUTIONS
    # ==========================================
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'}) 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Confidence Distribution
    ax1.hist(plot_confidences, bins=30, color='#8b0000', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0.5, color='black', linestyle='--', label='Evasion Threshold (0.5)')
    ax1.set_title("Post-Attack Model Confidence Distribution", fontweight='bold')
    ax1.set_xlabel("IDS Confidence (Lower = Better for Hacker)")
    ax1.set_ylabel("Number of Packets")
    ax1.legend()

    # Plot 2: L2 Distance (Stealth) vs Confidence
    # Only plot if we have successful evasions to avoid errors
    if plot_distances:
        evasion_confs = [c for c, e in zip(plot_confidences, [r[0] for r in results]) if e]
        sc = ax2.scatter(plot_distances, evasion_confs, alpha=0.6, c=plot_distances, cmap='viridis')
        ax2.set_title("Evasion Stealth Analysis (L2 Norm)", fontweight='bold')
        ax2.set_xlabel("Perturbation Distance (L2 Norm)")
        ax2.set_ylabel("Final Model Confidence")
        fig.colorbar(sc, ax=ax2, label="Perturbation Magnitude")

    plt.tight_layout()
    graph_path = os.path.join(FIGURES_DIR, "adversarial_stealth_analysis.png")
    plt.savefig(graph_path, dpi=600, bbox_inches='tight')
    plt.close()

    # SAVE DATA
    X_adv_df = pd.DataFrame(adversarial_examples, columns=X_test.columns)
    adv_path = os.path.join(DATA_PROCESSED_DIR, "X_adversarial_sample.parquet")
    X_adv_df.to_parquet(adv_path)

    # FINAL REPORT
    print("\n" + "="*60)
    print("RESULTS: MULTI-OBJECTIVE EVALUATION COMPLETE")
    print("="*60)
    print(f"Packets Attacked                         : {attacks_to_run:,}")
    print(f"Total AI Mutations Evaluated             : {attacks_to_run * POPULATION_SIZE * GENERATIONS:,}")
    print(f"Original Model Accuracy on these packets : 100.0%")
    print(f"Model Accuracy after Genetic Attack      : {100.0 - asr:.2f}%")
    print(f"Attack Success Rate (ASR)                : {asr:.2f}%")
    print(f"Compute Time                             : {attack_time/60:.1f} minutes")
    print(f"\n✅ High-Res Evasion Analysis Graph saved: {graph_path}")
    print(f"✅ Adversarial dataset saved to {adv_path} for Phase 4 (Adversarial Training).")

if __name__ == "__main__":
    run_evasion_attack()
