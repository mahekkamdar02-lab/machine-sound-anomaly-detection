"""
Step 0 — Sanity check.

Confirms your environment is set up correctly, and generates a tiny synthetic
"fake machine sound" so you can test the rest of the pipeline even before your
real MIMII download finishes.

Run: python src/00_check_setup.py
"""

import os
import numpy as np

print("Checking imports...")
import librosa               # noqa: E402
import matplotlib            # noqa: E402
matplotlib.use("Agg")        # safe for headless/script use
import matplotlib.pyplot as plt   # noqa: E402
import sklearn                # noqa: E402
import pandas as pd           # noqa: E402

print(f"  librosa    {librosa.__version__}")
print(f"  matplotlib {matplotlib.__version__}")
print(f"  sklearn    {sklearn.__version__}")
print(f"  pandas     {pd.__version__}")
print("All core libraries imported successfully.\n")

# ---- Generate a tiny synthetic "fake machine hum" for testing ----
# A real fan hum is roughly a steady tone + harmonics + a bit of noise.
# An "abnormal" version adds a rattle (extra high-frequency bursts).

sr = 16000          # 16kHz, matching MIMII's sampling rate
duration = 3.0       # seconds
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

def make_normal_hum():
    base_freq = 120.0  # Hz, a plausible fan/motor hum frequency
    signal = 0.5 * np.sin(2 * np.pi * base_freq * t)
    signal += 0.2 * np.sin(2 * np.pi * base_freq * 2 * t)   # harmonic
    signal += 0.02 * np.random.randn(len(t))                # background noise
    return signal.astype(np.float32)

def make_abnormal_hum():
    signal = make_normal_hum()
    # add an intermittent rattle: short bursts of high-frequency noise
    for start_sec in np.arange(0.3, duration, 0.5):
        start = int(start_sec * sr)
        end = min(start + int(0.03 * sr), len(signal))
        signal[start:end] += 0.6 * np.random.randn(end - start)
    return signal.astype(np.float32)

os.makedirs("data/raw/synthetic_demo/normal", exist_ok=True)
os.makedirs("data/raw/synthetic_demo/abnormal", exist_ok=True)

import soundfile as sf  # noqa: E402

sf.write("data/raw/synthetic_demo/normal/demo_normal.wav", make_normal_hum(), sr)
sf.write("data/raw/synthetic_demo/abnormal/demo_abnormal.wav", make_abnormal_hum(), sr)

print("Synthetic demo clips written to data/raw/synthetic_demo/")
print("You can now run 01_visualize_audio.py — it will use these demo files")
print("automatically if your real MIMII data isn't downloaded yet.")
