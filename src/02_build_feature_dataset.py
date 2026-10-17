"""
Step 2 — Build the MFCC feature dataset for the MIMII fan data.

Dataset structure expected:

data/raw/fan/
├── -6dB/
│   ├── id_00/
│   ├── id_02/
│   ├── id_04/
│   └── id_06/
├── 0dB/
│   ├── id_00/
│   ├── id_02/
│   ├── id_04/
│   └── id_06/
└── 6dB/
    ├── id_00/
    ├── id_02/
    ├── id_04/
    └── id_06/

For every WAV file, this script extracts:
- 13 MFCC means
- 13 MFCC standard deviations

It also stores:
- SNR
- machine ID
- normal/abnormal label
- original filepath

Run:
    python src/02_build_feature_dataset.py
"""

import os
import glob

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

SR = 16000
N_MFCC = 13

DATA_ROOT = os.path.join("data", "raw", "fan")

SNR_FOLDERS = ["-6dB", "0dB", "6dB"]

OUTPUT_PATH = os.path.join(
    "data",
    "features",
    "fan_mfcc_features_all_snr.csv"
)


# ---------------------------------------------------------
# MFCC extraction
# ---------------------------------------------------------

def extract_mfcc_features(y, sr=SR, n_mfcc=N_MFCC):
    """
    Convert one audio clip into a fixed-length feature vector.

    For each MFCC coefficient:
        - calculate mean over time
        - calculate standard deviation over time

    13 MFCCs x 2 statistics = 26 features.
    """

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    return np.concatenate([mfcc_mean, mfcc_std])


# ---------------------------------------------------------
# Find all audio clips
# ---------------------------------------------------------

def gather_clips(root_dir):
    """
    Find all WAV files under:

        root_dir/<snr>/<machine_id>/<normal|abnormal>/*.wav

    Returns:
        (filepath, snr, machine_id, label)
    """

    rows = []

    for snr in SNR_FOLDERS:

        snr_dir = os.path.join(root_dir, snr)

        if not os.path.isdir(snr_dir):
            print(f"WARNING: SNR folder not found: {snr_dir}")
            continue

        machine_dirs = sorted(
            glob.glob(os.path.join(snr_dir, "*"))
        )

        for machine_dir in machine_dirs:

            if not os.path.isdir(machine_dir):
                continue

            machine_id = os.path.basename(machine_dir)

            for label in ["normal", "abnormal"]:

                pattern = os.path.join(
                    machine_dir,
                    label,
                    "*.wav"
                )

                files = sorted(glob.glob(pattern))

                for filepath in files:

                    rows.append(
                        (
                            filepath,
                            snr,
                            machine_id,
                            label
                        )
                    )

    return rows


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("MIMII FAN — MFCC FEATURE EXTRACTION")
    print("=" * 60)

    print(f"\nData root: {DATA_ROOT}")
    print(f"SNR folders: {SNR_FOLDERS}")

    # -----------------------------------------------------
    # Find clips
    # -----------------------------------------------------

    clips = gather_clips(DATA_ROOT)

    if not clips:
        raise FileNotFoundError(
            "\nNo MIMII audio files were found.\n"
            "Check that your data structure is:\n\n"
            "data/raw/fan/-6dB/id_00/normal/*.wav\n"
            "data/raw/fan/0dB/id_00/normal/*.wav\n"
            "data/raw/fan/6dB/id_00/normal/*.wav\n"
        )

    print(f"\nFound {len(clips)} audio clips.")

    # -----------------------------------------------------
    # Count clips by SNR
    # -----------------------------------------------------

    clip_df = pd.DataFrame(
        clips,
        columns=[
            "filepath",
            "snr",
            "machine_id",
            "label"
        ]
    )

    print("\nClips by SNR:")
    print(clip_df["snr"].value_counts().sort_index())

    print("\nClips by machine:")
    print(clip_df["machine_id"].value_counts().sort_index())

    print("\nClips by label:")
    print(clip_df["label"].value_counts())

    # -----------------------------------------------------
    # Extract MFCC features
    # -----------------------------------------------------

    feature_rows = []

    print("\nExtracting MFCC features...\n")

    for filepath, snr, machine_id, label in tqdm(
        clips,
        desc="Processing audio"
    ):

        try:

            # Load audio
            y, _ = librosa.load(
                filepath,
                sr=SR
            )

            # Extract 26 MFCC features
            features = extract_mfcc_features(y)

            # Basic information for this clip
            row = {
                "filepath": filepath,
                "snr": snr,
                "machine_id": machine_id,
                "label": label
            }

            # Add MFCC features
            for i, value in enumerate(features):

                row[f"mfcc_{i}"] = value

            feature_rows.append(row)

        except Exception as e:

            print(
                f"\nWARNING: Could not process {filepath}"
            )

            print(f"Reason: {e}")

    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    df = pd.DataFrame(feature_rows)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"\nSaved to:")
    print(OUTPUT_PATH)

    print(f"\nDataset shape:")
    print(
        f"{df.shape[0]} clips x {df.shape[1]} columns"
    )

    print("\nSNR distribution:")
    print(
        df["snr"].value_counts().sort_index()
    )

    print("\nMachine ID distribution:")
    print(
        df["machine_id"].value_counts().sort_index()
    )

    print("\nLabel distribution:")
    print(
        df["label"].value_counts()
    )

    print("\nExpected feature columns:")
    print("26 MFCC features = 13 means + 13 standard deviations")

    print("\nFirst few rows:")
    print(
        df[
            [
                "snr",
                "machine_id",
                "label"
            ]
        ].head()
    )


if __name__ == "__main__":
    main()