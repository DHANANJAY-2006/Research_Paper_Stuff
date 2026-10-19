"""
shap_analysis.py — Generates a publication-quality SHAP beeswarm graph.
"""

import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from config import DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR

def run_shap_analysis():
    print("="*50)
    print("GENERATING ACADEMIC SHAP BEESWARM PLOT")
    print("="*50)
    
    # 1. Load the Test Data and the XGBoost Model
    try:
        X_test = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_test.parquet"))
        model_path = os.path.join(MODELS_DIR, "XGBoost_baseline.joblib")
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("[ERROR] Could not find test data or trained model.")
        return

    # Sample data to make graph generation fast
    print("Calculating SHAP values...")
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    
    # 2. Create the SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # 3. Professional Academic Styling
    print("Plotting publication-quality graph...")
    # Journals (Elsevier, Springer) prefer Serif fonts and high resolution
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'}) 
    plt.figure(figsize=(10, 7))
    
    # The Beeswarm plot is the "gold standard" for SHAP in research papers
    # It shows not just importance, but exactly HOW a feature drives the model
    shap.summary_plot(shap_values, X_sample, show=False, max_display=12)
    
    # Tweak the layout so it looks like a textbook/journal figure
    plt.title("SHAP Value Distribution: Feature Impact on Evasion Detection", 
              fontsize=16, fontweight='bold', pad=20)
    
    # 4. Save the Graph (600 DPI is required by most academic journals)
    output_path = os.path.join(FIGURES_DIR, "shap_beeswarm_publication.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ High-Resolution Academic Graph successfully saved!")
    print(f"👉 Go open the image here: {output_path}")

if __name__ == "__main__":
    run_shap_analysis()
