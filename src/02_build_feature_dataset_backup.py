"""
Step 2 — Build a feature dataset (Week 3 of your roadmap).

Walks through data/raw/fan/<machine_id>/<normal|abnormal>/*.wav, extracts MFCC
features (mean + std per coefficient) for each clip, and saves the result as
a CSV in data/features/ — one row per clip, ready for SVM/Random Forest.

If no real MIMII data is found, runs on the synthetic demo clips instead so
you can confirm the pipeline works end-to-end.

Run: python src/02_build_feature_dataset.py
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

SR = 16000
N_MFCC = 13


def extract_mfcc_features(y, sr=SR, n_mfcc=N_MFCC):
    """
    Convert a variable-length audio clip into a FIXED-length feature vector:
    the mean and standard deviation of each MFCC coefficient over time.
    This gives every clip the same-size feature vector regardless of duration.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])


def gather_clips(root_dir):
    """
    Expects structure: root_dir/<machine_id>/<normal|abnormal>/*.wav
    Returns a list of (filepath, machine_id, label) tuples.
    """
    rows = []
    machine_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
    for machine_dir in machine_dirs:
        if not os.path.isdir(machine_dir):
            continue
        machine_id = os.path.basename(machine_dir)
        for label in ["normal", "abnormal"]:
            files = sorted(glob.glob(os.path.join(machine_dir, label, "*.wav")))
            for f in files:
                rows.append((f, machine_id, label))
    return rows


def main():
    real_root = os.path.join("data", "raw", "fan")
    clips = gather_clips(real_root)

    if not clips:
        print(f"No data found under {real_root}/<id>/<normal|abnormal>/*.wav")
        print("Falling back to synthetic demo clips for a pipeline test.")
        demo_root = os.path.join("data", "raw", "synthetic_demo")
        clips = [(f, "synthetic_demo", "normal")
                 for f in glob.glob(os.path.join(demo_root, "normal", "*.wav"))]
        clips += [(f, "synthetic_demo", "abnormal")
                  for f in glob.glob(os.path.join(demo_root, "abnormal", "*.wav"))]

    if not clips:
        raise FileNotFoundError(
            "No audio found at all. Run 'python src/00_check_setup.py' first, "
            "or download MIMII per the README."
        )

    print(f"Found {len(clips)} clips. Extracting MFCC features...\n")

    feature_rows = []
    for filepath, machine_id, label in tqdm(clips):
        y, _ = librosa.load(filepath, sr=SR)
        feats = extract_mfcc_features(y)
        row = {
            "filepath": filepath,
            "machine_id": machine_id,
            "label": label,  # "normal" or "abnormal"
        }
        for i, v in enumerate(feats):
            row[f"mfcc_{i}"] = v
        feature_rows.append(row)

    df = pd.DataFrame(feature_rows)
    os.makedirs("data/features", exist_ok=True)
    out_path = "data/features/fan_mfcc_features.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved feature dataset to: {out_path}")
    print(f"Shape: {df.shape[0]} clips x {df.shape[1]} columns")
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("\nMachine ID distribution:")
    print(df["machine_id"].value_counts())


if __name__ == "__main__":
    main()
