"""
Step 1 — Load and visualize audio (Week 1-2 of your roadmap).

Loads one normal and one abnormal clip, and plots:
  - waveform
  - spectrogram (STFT)
  - Mel-spectrogram
  - MFCC

Uses your real MIMII fan data if found at data/raw/fan/id_00/, otherwise
falls back to the synthetic demo clips from 00_check_setup.py.

Run: python src/01_visualize_audio.py
"""

import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000  # MIMII's sampling rate


def find_clip(machine_dir, condition):
    """Return the first .wav file found under machine_dir/condition/, or None."""
    pattern = os.path.join(machine_dir, condition, "*.wav")
    files = sorted(glob.glob(pattern))
    return files[0] if files else None


def locate_sample_clips():
    """
    Look for real MIMII fan data first (data/raw/fan/id_00/normal|abnormal),
    fall back to the synthetic demo clips if not found.
    """
    real_dir = os.path.join("data", "raw", "fan", "id_00")
    normal = find_clip(real_dir, "normal")
    abnormal = find_clip(real_dir, "abnormal")

    if normal and abnormal:
        print(f"Using real MIMII data:\n  normal:   {normal}\n  abnormal: {abnormal}")
        return normal, abnormal

    print("Real MIMII fan data not found at data/raw/fan/id_00/.")
    print("Falling back to synthetic demo clips (run 00_check_setup.py first if missing).")
    demo_dir = os.path.join("data", "raw", "synthetic_demo")
    normal = find_clip(demo_dir, "normal")
    abnormal = find_clip(demo_dir, "abnormal")
    if not (normal and abnormal):
        raise FileNotFoundError(
            "No audio found at all. Run 'python src/00_check_setup.py' first, "
            "or download MIMII per the README."
        )
    return normal, abnormal


def load_clip(path, sr=SR):
    y, _ = librosa.load(path, sr=sr)
    return y


def plot_comparison(y_normal, y_abnormal, sr, out_path):
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    fig.suptitle("Normal vs. Abnormal Machine Sound", fontsize=16, fontweight="bold")

    for col, (y, label) in enumerate([(y_normal, "Normal"), (y_abnormal, "Abnormal")]):
        # 1. Waveform
        ax = axes[0, col]
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title(f"{label} — Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

        # 2. Spectrogram (STFT)
        ax = axes[1, col]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title(f"{label} — Spectrogram (STFT)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        # 3. Mel-spectrogram
        ax = axes[2, col]
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(f"{label} — Mel-Spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        # 4. MFCC
        ax = axes[3, col]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax)
        ax.set_title(f"{label} — MFCC (13 coefficients)")
        fig.colorbar(img, ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved comparison plot to: {out_path}")


def main():
    normal_path, abnormal_path = locate_sample_clips()
    y_normal = load_clip(normal_path)
    y_abnormal = load_clip(abnormal_path)

    print(f"\nNormal clip:   {len(y_normal)/SR:.2f}s, {len(y_normal)} samples")
    print(f"Abnormal clip: {len(y_abnormal)/SR:.2f}s, {len(y_abnormal)} samples")

    plot_comparison(y_normal, y_abnormal, SR, "visualizations/01_waveform_spectrogram_mfcc.png")

    print("\nOpen the saved image and look for:")
    print("  - Does the abnormal waveform show extra spikes/bursts?")
    print("  - Does the abnormal spectrogram show new energy at higher frequencies?")
    print("  - Do the MFCC patterns visibly differ between normal and abnormal?")


if __name__ == "__main__":
    main()
