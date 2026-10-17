"""
Step 4 — SNR Robustness Study

Research question:
"How much does anomaly-detection performance degrade under noise?"

Experiment:
    Train:
        0dB
        Machine IDs: id_00, id_02, id_04

    Test:
        Machine ID: id_06
        SNR levels: -6dB, 0dB, 6dB

Models:
    1. SVM
    2. Random Forest

Metrics:
    Accuracy
    Precision
    Recall
    F1-score
    ROC-AUC

Outputs:
    results/snr_robustness_results.csv
    visualizations/04_snr_robustness.png

Run:
    python src/04_snr_robustness.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "data/features/fan_mfcc_features_all_snr.csv"

TRAIN_SNR = "0dB"

TRAIN_MACHINE_IDS = [
    "id_00",
    "id_02",
    "id_04",
]

TEST_MACHINE_IDS = [
    "id_06",
]

SNR_ORDER = [
    "-6dB",
    "0dB",
    "6dB",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("MIMII FAN — SNR ROBUSTNESS STUDY")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"\nFeature CSV not found:\n{CSV_PATH}\n"
            "Run the feature extraction script first."
        )

    df = pd.read_csv(CSV_PATH)

    print(f"\nLoaded dataset:")
    print(f"  Rows    : {df.shape[0]}")
    print(f"  Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # 2. Detect MFCC feature columns
    # --------------------------------------------------------

    feature_cols = [
        column
        for column in df.columns
        if column.lower().startswith("mfcc")
    ]

    if not feature_cols:
        raise ValueError(
            "No MFCC feature columns found in the CSV."
        )

    print(f"\nMFCC features found: {len(feature_cols)}")

    if len(feature_cols) != 26:
        print(
            f"WARNING: Expected 26 MFCC features, "
            f"but found {len(feature_cols)}."
        )

    # --------------------------------------------------------
    # 3. Check required columns
    # --------------------------------------------------------

    required_columns = [
        "snr",
        "machine_id",
        "label",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' not found."
            )

    print("\nRequired columns found:")
    print("  snr")
    print("  machine_id")
    print("  label")

    # --------------------------------------------------------
    # 4. Show dataset information
    # --------------------------------------------------------

    print("\nSNR distribution:")
    print(df["snr"].value_counts())

    print("\nMachine distribution:")
    print(df["machine_id"].value_counts())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    # --------------------------------------------------------
    # 5. Verify machine split
    # --------------------------------------------------------

    all_machine_ids = set(df["machine_id"].unique())

    if not set(TRAIN_MACHINE_IDS).issubset(all_machine_ids):
        raise ValueError(
            "One or more training machine IDs are not present."
        )

    if not set(TEST_MACHINE_IDS).issubset(all_machine_ids):
        raise ValueError(
            "One or more testing machine IDs are not present."
        )

    overlap = set(TRAIN_MACHINE_IDS) & set(TEST_MACHINE_IDS)

    if overlap:
        raise ValueError(
            f"Machine leakage detected! "
            f"Machines appear in both train and test: {overlap}"
        )

    print("\nMachine-wise split:")
    print(f"  Training machines: {TRAIN_MACHINE_IDS}")
    print(f"  Testing machines : {TEST_MACHINE_IDS}")

    # --------------------------------------------------------
    # 6. Create training dataset
    # --------------------------------------------------------

    train_df = df[
        (df["snr"] == TRAIN_SNR)
        & (df["machine_id"].isin(TRAIN_MACHINE_IDS))
    ].copy()

    if len(train_df) == 0:
        raise ValueError(
            "No training data found."
        )

    print("\n" + "-" * 65)
    print("TRAINING DATA")
    print("-" * 65)

    print(f"SNR: {TRAIN_SNR}")
    print(f"Machines: {TRAIN_MACHINE_IDS}")
    print(f"Number of clips: {len(train_df)}")

    print("\nTraining labels:")
    print(train_df["label"].value_counts())

    # --------------------------------------------------------
    # 7. Prepare training X and y
    # --------------------------------------------------------

    X_train_raw = train_df[feature_cols].values

    y_train = (
        train_df["label"]
        .astype(str)
        .str.lower()
        .eq("abnormal")
        .astype(int)
        .values
    )

    print(f"\nX_train shape: {X_train_raw.shape}")
    print(f"y_train shape: {y_train.shape}")

    # --------------------------------------------------------
    # 8. Scale features for SVM
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_raw
    )

    # --------------------------------------------------------
    # 9. Create models
    # --------------------------------------------------------

    models = {

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    # --------------------------------------------------------
    # 10. Train models
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("TRAINING MODELS")
    print("-" * 65)

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        if name == "SVM":

            model.fit(
                X_train_scaled,
                y_train,
            )

        else:

            model.fit(
                X_train_raw,
                y_train,
            )

        trained_models[name] = model

        print(f"{name} training complete.")

    # --------------------------------------------------------
    # 11. Test across all SNR levels
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SNR ROBUSTNESS EVALUATION")
    print("=" * 65)

    results = []

    for snr in SNR_ORDER:

        test_df = df[
            (df["snr"] == snr)
            & (df["machine_id"].isin(TEST_MACHINE_IDS))
        ].copy()

        if len(test_df) == 0:

            print(
                f"\nWARNING: No test data found for {snr}."
            )

            continue

        X_test_raw = test_df[feature_cols].values

        y_test = (
            test_df["label"]
            .astype(str)
            .str.lower()
            .eq("abnormal")
            .astype(int)
            .values
        )

        print("\n" + "-" * 65)
        print(f"TEST SNR: {snr}")
        print("-" * 65)

        print(f"Test machines: {TEST_MACHINE_IDS}")
        print(f"Test clips: {len(test_df)}")

        print(
            f"Normal: {(y_test == 0).sum()} | "
            f"Abnormal: {(y_test == 1).sum()}"
        )

        for name, model in trained_models.items():

            # SVM needs scaling.
            if name == "SVM":

                X_test = scaler.transform(
                    X_test_raw
                )

            else:

                # Random Forest does not need scaling.
                X_test = X_test_raw

            # Predictions
            y_pred = model.predict(X_test)

            # Probability / score
            y_score = model.predict_proba(X_test)[:, 1]

            # Metrics
            accuracy = accuracy_score(
                y_test,
                y_pred,
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0,
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0,
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0,
            )

            try:

                roc_auc = roc_auc_score(
                    y_test,
                    y_score,
                )

            except ValueError:

                roc_auc = np.nan

            # Save result
            results.append(
                {
                    "model": name,
                    "snr": snr,
                    "train_snr": TRAIN_SNR,
                    "train_machines": ",".join(
                        TRAIN_MACHINE_IDS
                    ),
                    "test_machines": ",".join(
                        TEST_MACHINE_IDS
                    ),
                    "n_test_clips": len(test_df),
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc,
                }
            )

            # Print result
            print(f"\n{name}")

            print(
                f"  Accuracy : {accuracy:.4f}"
            )

            print(
                f"  Precision: {precision:.4f}"
            )

            print(
                f"  Recall   : {recall:.4f}"
            )

            print(
                f"  F1       : {f1:.4f}"
            )

            print(
                f"  ROC-AUC  : {roc_auc:.4f}"
            )

    # --------------------------------------------------------
    # 12. Create results DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise ValueError(
            "No results were generated."
        )

    # --------------------------------------------------------
    # 13. Save results
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True,
    )

    results_path = (
        "results/snr_robustness_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(
        results_df[
            [
                "model",
                "snr",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nResults saved to:\n{results_path}"
    )

    # --------------------------------------------------------
    # 14. Create robustness plot
    # --------------------------------------------------------

    os.makedirs(
        "visualizations",
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    # --------------------------------------------------------
    # AUC plot
    # --------------------------------------------------------

    for name in trained_models.keys():

        model_results = (
            results_df[
                results_df["model"] == name
            ]
            .set_index("snr")
            .reindex(SNR_ORDER)
        )

        axes[0].plot(
            SNR_ORDER,
            model_results["roc_auc"],
            marker="o",
            linewidth=2,
            label=name,
        )

    axes[0].set_xlabel(
        "SNR level"
    )

    axes[0].set_ylabel(
        "ROC-AUC"
    )

    axes[0].set_title(
        "ROC-AUC vs. Noise Level"
    )

    axes[0].set_ylim(
        0,
        1.05,
    )

    axes[0].grid(
        alpha=0.3
    )

    axes[0].legend()

    # --------------------------------------------------------
    # F1 plot
    # --------------------------------------------------------

    for name in trained_models.keys():

        model_results = (
            results_df[
                results_df["model"] == name
            ]
            .set_index("snr")
            .reindex(SNR_ORDER)
        )

        axes[1].plot(
            SNR_ORDER,
            model_results["f1"],
            marker="o",
            linewidth=2,
            label=name,
        )

    axes[1].set_xlabel(
        "SNR level"
    )

    axes[1].set_ylabel(
        "F1-score"
    )

    axes[1].set_title(
        "F1-score vs. Noise Level"
    )

    axes[1].set_ylim(
        0,
        1.05,
    )

    axes[1].grid(
        alpha=0.3
    )

    axes[1].legend()

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        "SNR Robustness: "
        "Trained on 0dB, Tested Across All SNR Levels",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()

    plot_path = (
        "visualizations/04_snr_robustness.png"
    )

    plt.savefig(
        plot_path,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nHeadline plot saved to:\n{plot_path}"
    )

    # --------------------------------------------------------
    # 15. Research interpretation reminder
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("HOW TO INTERPRET THE RESULT")
    print("=" * 65)

    print(
        """
The key question is:

How much does performance change as SNR decreases?

A large drop from 6dB/0dB to -6dB
suggests reduced robustness to noise.

A smaller drop suggests better robustness.

Do NOT decide which model is better until
we inspect the actual results.
"""
    )

    print("=" * 65)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()