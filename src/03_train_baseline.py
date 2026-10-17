"""
Step 3 — Train your first classical ML baselines: SVM + Random Forest
(Week 3-4 of your roadmap).

Loads the feature CSV built by 02_build_feature_dataset.py, splits train/test
BY MACHINE ID where possible (to avoid leakage / test true generalization),
trains SVM and Random Forest, and reports Accuracy, Precision, Recall, F1,
and ROC-AUC for both.

Run: python src/03_train_baseline.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

FEATURES_PATH = "data/features/fan_mfcc_features.csv"


def load_features():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run 'python src/02_build_feature_dataset.py' first."
        )
    df = pd.read_csv(FEATURES_PATH)
    return df


def split_train_test(df):
    """
    Prefer splitting BY MACHINE ID (train on some IDs, test on others) — this
    is the stronger, leakage-safe evaluation recommended in the project guide.
    Falls back to a random stratified split if there's only one machine ID
    (e.g. when running on the synthetic demo data).
    """
    ids = df["machine_id"].unique()
    if len(ids) >= 2:
        train_ids, test_ids = train_test_split(ids, test_size=0.3, random_state=42)
        train_df = df[df["machine_id"].isin(train_ids)]
        test_df = df[df["machine_id"].isin(test_ids)]
        print(f"Splitting BY MACHINE ID -> train IDs: {list(train_ids)}, test IDs: {list(test_ids)}")
    else:
        print("Only one machine ID found (likely demo data) -> falling back to a random "
              "stratified split. Once you have real multi-ID MIMII data, this will "
              "automatically switch to the safer by-ID split.")
        train_df, test_df = train_test_split(
            df, test_size=0.3, random_state=42, stratify=df["label"]
        )
    return train_df, test_df


def to_xy(df):
    feature_cols = [c for c in df.columns if c.startswith("mfcc_")]
    X = df[feature_cols].values
    y = (df["label"] == "abnormal").astype(int).values  # 1 = abnormal, 0 = normal
    return X, y


def evaluate(name, y_true, y_pred, y_score):
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.3f}")
    try:
        auc = roc_auc_score(y_true, y_score)
        print(f"ROC-AUC:   {auc:.3f}")
    except ValueError:
        auc = float("nan")
        print("ROC-AUC:   undefined (only one class present in test set)")
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}")
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def main():
    df = load_features()
    print(f"Loaded {len(df)} clips with {df['label'].value_counts().to_dict()} label counts.\n")

    train_df, test_df = split_train_test(df)
    X_train, y_train = to_xy(train_df)
    X_test, y_test = to_xy(test_df)

    print(f"\nTrain set: {X_train.shape[0]} clips | Test set: {X_test.shape[0]} clips")

    # SVM needs scaled features; Random Forest does not
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # ---- SVM ----
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_score = svm.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate("SVM (MFCC)", y_test, svm_pred, svm_score))

    # ---- Random Forest ----
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    rf_pred = rf.predict(X_test)
    rf_score = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest (MFCC)", y_test, rf_pred, rf_score))

    # ---- Save results ----
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    out_path = "results/baseline_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results table to: {out_path}")
    print(results_df.to_string(index=False))

    print("\nNext steps:")
    print("  - If ROC-AUC is far below 0.5-0.6 on real data, check your labels/split first.")
    print("  - Once this works on the fan subset, add more machine IDs and re-run.")
    print("  - Week 5+: build the autoencoder (true unsupervised anomaly detection).")


if __name__ == "__main__":
    main()
