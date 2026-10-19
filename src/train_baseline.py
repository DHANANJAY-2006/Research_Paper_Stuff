import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from time import time

from config import DATA_PROCESSED_DIR, MODELS_DIR, RANDOM_SEED

def load_processed_data():
    try:
        X_train = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_train.parquet"))
        X_test = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "X_test.parquet"))
        y_train = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "y_train.parquet")).squeeze()
        y_test = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "y_test.parquet")).squeeze()
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("Processed data not found. Please run data_loader.py first.")
        return None, None, None, None

def train_models():
    X_train, X_test, y_train, y_test = load_processed_data()
    if X_train is None:
        return

    print("="*50)
    print("PHASE 1: TRAINING BASELINE MODELS")
    print("="*50)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=RANDOM_SEED),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, n_jobs=-1, random_state=RANDOM_SEED),
        "LightGBM": LGBMClassifier(n_estimators=100, max_depth=15, learning_rate=0.1, n_jobs=-1, random_state=RANDOM_SEED)
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time()
        
        model.fit(X_train, y_train)
        train_time = time() - t0
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"  Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | Time: {train_time:.2f}s")
        
        model_path = os.path.join(MODELS_DIR, f"{name}_baseline.joblib")
        joblib.dump(model, model_path)

        results.append({"Model": name, "Accuracy": acc, "F1-Score": f1, "Train_Time_s": train_time})

    print("\n" + "="*50)
    print("f BASELINE METRICS SUMMARY")
    print(pd.DataFrame(results).to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    train_models()
