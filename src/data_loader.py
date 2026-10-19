"""
data_loader.py — Loads, cleans, and encodes the dataset.
Saves both binary AND original multiclass labels for per-class analysis.
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR,
    BENIGN_LABEL, TEST_SIZE, RANDOM_SEED
)

def load_and_preprocess():
    csv_files = glob.glob(os.path.join(DATA_RAW_DIR, "*.csv"))

    if not csv_files:
        print(f"\n[ERROR] No CSV files found in {DATA_RAW_DIR}")
        return None

    print(f"Found {len(csv_files)} CSV files. Loading all data...")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df.columns = df.columns.str.strip()
    print(f"Actually loaded {len(df)} rows and {df.shape[1]} columns.")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Find target column
    target_col = None
    for col in df.columns:
        if col.lower() in ['label', 'attack', 'class']:
            target_col = col
            break

    if not target_col:
        print(f"\n[ERROR] Target column not found.")
        return None

    print(f"Original Label Distribution:\n{df[target_col].value_counts().head(10)}")

    # Identify benign string dynamically
    benign_str = BENIGN_LABEL
    for u in df[target_col].unique():
        if 'benign' in str(u).lower() or 'normal' in str(u).lower():
            benign_str = u
            break

    # Binary label
    df['Binary_Label'] = df[target_col].apply(lambda x: 0 if x == benign_str else 1)
    # Keep original multiclass label for per-class analysis in Phase 2
    df['Original_Label'] = df[target_col].astype(str)

    X = df.drop(columns=[target_col, 'Binary_Label', 'Original_Label'])
    y_binary = df['Binary_Label']
    y_original = df['Original_Label']

    X = X.select_dtypes(include=[np.number])

    print("Splitting into Train and Test sets...")
    X_train, X_test, y_train, y_test, y_orig_train, y_orig_test = train_test_split(
        X, y_binary, y_original,
        test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_binary
    )

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    joblib.dump(scaler, os.path.join(DATA_PROCESSED_DIR, "scaler.joblib"))

    print("Saving processed datasets...")
    X_train_scaled.to_parquet(os.path.join(DATA_PROCESSED_DIR, "X_train.parquet"))
    X_test_scaled.to_parquet(os.path.join(DATA_PROCESSED_DIR, "X_test.parquet"))
    y_train.to_frame().to_parquet(os.path.join(DATA_PROCESSED_DIR, "y_train.parquet"))
    y_test.to_frame().to_parquet(os.path.join(DATA_PROCESSED_DIR, "y_test.parquet"))

    # Save original labels so Phase 2 Experiment 3 can run
    y_orig_test.to_frame(name="Original_Label").to_parquet(
        os.path.join(DATA_PROCESSED_DIR, "y_test_original_labels.parquet")
    )

    print("\n✅ Data preprocessing complete!")
    print(f"Final Train size: {X_train_scaled.shape}")
    print(f"Attack class distribution in test:\n{y_orig_test.value_counts().head(10)}")

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess()
