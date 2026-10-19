"""
recall_drop_analysis.py — Recall/Precision/F1 Drop Under Adversarial Attack
Fixed for Python 3.14 on Windows: forces n_jobs=1 on all loaded models.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

from config import DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR

POPULATION_SIZE = 150
GENERATIONS     = 60
EPSILON         = 5.0
N_PACKETS       = 500   # Full 500 — no compromise
RANDOM_SEED     = 42


def get_mutable_indices(X_test, model):
    importances = model.feature_importances_
    continuous = [
        (i, col, importances[i])
        for i, col in enumerate(X_test.columns)
        if len(X_test[col].unique()) > 15
    ]
    continuous.sort(key=lambda x: x[2], reverse=True)
    return [x[0] for x in continuous[:15]]


def predict_safe(model, X_np, col_names):
    """Always predicts using a DataFrame and with n_jobs=1 to avoid threading crashes."""
    df = pd.DataFrame(X_np, columns=col_names)
    return model.predict_proba(df)[:, 1]


def run_ga_single(original_packet, mutable_indices, model, col_names):
    """Single-packet Multi-Objective GA. No external parallelism."""
    population = np.zeros((POPULATION_SIZE, len(original_packet)))
    for p in range(POPULATION_SIZE):
        population[p, mutable_indices] = np.random.uniform(-EPSILON, EPSILON, len(mutable_indices))

    best_adv = original_packet.copy()
    evaded   = False

    for gen in range(GENERATIONS):
        candidates = original_packet + population
        probs = predict_safe(model, candidates, col_names)
        l2    = np.linalg.norm(population[:, mutable_indices], axis=1)
        fitness = probs + 0.05 * l2

        if np.any(probs < 0.5):
            success_idx = np.where(probs < 0.5)[0]
            stealthiest = success_idx[np.argmin(l2[success_idx])]
            best_adv    = candidates[stealthiest]
            evaded      = True
            break

        survivors_idx = np.argsort(fitness)[:POPULATION_SIZE // 2]
        survivors     = population[survivors_idx]
        next_gen      = np.zeros_like(population)
        next_gen[:len(survivors)] = survivors

        for j in range(len(survivors), POPULATION_SIZE):
            p1    = survivors[np.random.randint(len(survivors))]
            p2    = survivors[np.random.randint(len(survivors))]
            child = np.where(np.random.rand(len(original_packet)) > 0.5, p1, p2)
            mask  = np.random.rand(len(mutable_indices)) > 0.4
            child[mutable_indices] = np.clip(
                child[mutable_indices] + np.random.normal(0, EPSILON / 3, len(mutable_indices)) * mask,
                -EPSILON, EPSILON
            )
            next_gen[j] = child
        population = next_gen

    return evaded, best_adv


def compute_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
    fpr       = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    return {"Recall": recall, "Precision": precision, "F1": f1, "FPR": fpr}


def run_recall_drop_analysis():
    print("=" * 60)
    print("PHASE 2: RECALL/F1 DROP ANALYSIS (500 PACKETS, STABLE)")
    print("=" * 60)

    X_test = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "y_test.parquet")).squeeze()
    col_names = list(X_test.columns)

    # Load models and FORCE n_jobs=1 to prevent Windows threading crash
    models_dict = {}
    for name in ["RandomForest", "XGBoost", "LightGBM"]:
        path = os.path.join(MODELS_DIR, f"{name}_baseline.joblib")
        if os.path.exists(path):
            m = joblib.load(path)
            if hasattr(m, 'n_jobs'):
                m.n_jobs = 1   # ← KEY FIX: disable internal threading
            models_dict[name] = m
            print(f"  Loaded {name} (n_jobs forced to 1)")

    mutable_indices = get_mutable_indices(X_test, models_dict["XGBoost"])

    np.random.seed(RANDOM_SEED)
    attack_mask = y_test.values == 1
    benign_mask = y_test.values == 0
    X_attacks   = X_test.values[attack_mask][:N_PACKETS]
    y_attacks   = np.ones(len(X_attacks), dtype=int)
    X_benign    = X_test.values[benign_mask][:N_PACKETS]
    y_benign    = np.zeros(len(X_benign), dtype=int)
    y_full      = np.concatenate([y_attacks, y_benign])

    results_table = []

    for model_name, model in models_dict.items():
        print(f"\n  [{model_name}] Evaluating {N_PACKETS} packets...")

        adv_packets   = []
        evaded_count  = 0

        for i, pkt in enumerate(X_attacks):
            evaded, adv = run_ga_single(pkt, mutable_indices, model, col_names)
            adv_packets.append(adv)
            if evaded:
                evaded_count += 1
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{N_PACKETS} done | ASR so far: {evaded_count/(i+1)*100:.1f}%")

        asr    = evaded_count / N_PACKETS * 100
        X_adv  = np.array(adv_packets)

        # Evaluate on clean vs adversarial
        X_clean_df = pd.DataFrame(np.concatenate([X_attacks, X_benign]), columns=col_names)
        X_adv_df   = pd.DataFrame(np.concatenate([X_adv, X_benign]), columns=col_names)

        clean_pred = model.predict(X_clean_df)
        adv_pred   = model.predict(X_adv_df)

        cm = compute_metrics(y_full, clean_pred)
        am = compute_metrics(y_full, adv_pred)

        print(f"  [{model_name}] Final ASR: {asr:.2f}%")
        print(f"    Recall   : {cm['Recall']:.2f}% → {am['Recall']:.2f}%  (Drop: {cm['Recall']-am['Recall']:.2f}%)")
        print(f"    Precision: {cm['Precision']:.2f}% → {am['Precision']:.2f}%")
        print(f"    F1-Score : {cm['F1']:.2f}% → {am['F1']:.2f}%")
        print(f"    FPR      : {cm['FPR']:.2f}% → {am['FPR']:.2f}%")

        results_table.append({
            "Model":        model_name,
            "Clean Recall": round(cm["Recall"], 2),
            "Adv Recall":   round(am["Recall"], 2),
            "Recall Drop":  round(cm["Recall"] - am["Recall"], 2),
            "Clean F1":     round(cm["F1"], 2),
            "Adv F1":       round(am["F1"], 2),
            "ASR":          round(asr, 2)
        })

    df = pd.DataFrame(results_table)
    print("\n" + "=" * 60)
    print("FINAL SUMMARY TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(FIGURES_DIR, "recall_drop_table.csv"), index=False)

    # Publication-Quality 3-Panel Figure
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(df))
    w = 0.35

    # Panel 1: Recall
    axes[0].bar(x - w/2, df["Clean Recall"], w, label="Clean",       color="steelblue")
    axes[0].bar(x + w/2, df["Adv Recall"],   w, label="Adversarial", color="#8b0000", alpha=0.85)
    axes[0].set_title("Detection Recall Drop", fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(df["Model"])
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].legend(); axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    axes[0].set_ylim(0, 115)

    # Panel 2: F1
    axes[1].bar(x - w/2, df["Clean F1"], w, label="Clean",       color="steelblue")
    axes[1].bar(x + w/2, df["Adv F1"],   w, label="Adversarial", color="#8b0000", alpha=0.85)
    axes[1].set_title("F1-Score Drop", fontweight='bold')
    axes[1].set_xticks(x); axes[1].set_xticklabels(df["Model"])
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend(); axes[1].grid(axis='y', linestyle=':', alpha=0.6)
    axes[1].set_ylim(0, 115)

    # Panel 3: ASR per model
    axes[2].bar(df["Model"], df["ASR"], color="#8b0000", alpha=0.85, width=0.4)
    axes[2].set_title("Attack Success Rate (ASR)", fontweight='bold')
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[2].grid(axis='y', linestyle=':', alpha=0.6)
    axes[2].set_ylim(0, 115)

    plt.suptitle("Adversarial Evasion Impact on IDS Detection Metrics",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "recall_drop_analysis.png")
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 600 DPI Graph saved: {path}")
    print(f"✅ CSV saved: {os.path.join(FIGURES_DIR, 'recall_drop_table.csv')}")
    print("\n🎯 Phase 2 is 100% COMPLETE.")


if __name__ == "__main__":
    run_recall_drop_analysis()
