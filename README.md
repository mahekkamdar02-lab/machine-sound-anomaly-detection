# Industrial Machine Sound Anomaly Detection

An individual CSE (AI & ML) mini-project focused on classifying industrial machine sounds (normal vs. abnormal) using audio features and classical machine learning models.

## Project Overview
This project uses the **MIMII Fan dataset** to detect anomalies in industrial machine sounds under varying signal-to-noise ratio (SNR) conditions. 

- **Audio Processing:** Librosa for feature extraction (Short-Time Fourier Transform, Mel-spectrograms, and Mel-Frequency Cepstral Coefficients).
- **Features Extracted:** 26 MFCC-derived numerical features per clip (13 MFCC means and 13 standard deviations).
- **Models Trained:** Support Vector Machine (SVM) and Random Forest.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## Key Experiment: Unseen-Machine Noise Robustness
To test real-world generalization, an advanced experiment was performed:
- **Training Data:** Machine IDs `id_00`, `id_02`, and `id_04` at 0 dB SNR.
- **Test Data:** Completely unseen machine ID `id_06`.
- **Testing Conditions:** Evaluated across $-6\text{ dB}$, $0\text{ dB}$, and $+6\text{ dB}$ SNR noise levels.

### 1. Support Vector Machine (SVM) Results
| Test SNR | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **-6 dB** | 26.96% | 25.90% | 95.84% | 40.78% | 0.470 |
| **0 dB** | 59.16% | 34.98% | 64.82% | 45.44% | 0.653 |
| **+6 dB** | 81.61% | 89.13% | 34.07% | 49.30% | 0.795 |

### 2. Random Forest Results
| Test SNR | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **-6 dB** | 30.16% | 25.37% | 85.60% | 39.14% | 0.446 |
| **0 dB** | 44.40% | 29.47% | 80.33% | 43.12% | 0.632 |
| **+6 dB** | 59.01% | 36.70% | 77.56% | 49.82% | 0.795 |

### Key Findings & Limitations
- **Impact of Noise:** Performance drops significantly under heavy noise conditions ($-6\text{ dB}$ SNR), illustrating the core challenge of cross-machine acoustic anomaly detection. For instance, SVM's ROC-AUC drops from $0.795$ at $+6\text{ dB}$ to $0.470$ at $-6\text{ dB}$.
- **Baseline Scope:** These results represent initial baseline experiments using classical machine learning on handcrafted features, serving as a foundation for more advanced deep-learning approaches.

---

## Project Structure
- `src/` — Python scripts for environment checks, audio processing, feature building, and model training/evaluation.
- `data/features/` — Extracted MFCC feature CSV datasets (`fan_mfcc_features_all_snr.csv`).
- `results/` — Saved model evaluation results and CSV metrics logs.
- `visualizations/` — Generated evaluation plots (`04_snr_robustness.png`) and audio spectrograms.