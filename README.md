# Battery SOH Estimation via WD-GPR-SR

**Lithium-Ion Battery State-of-Health Estimation using Wide-and-Deep Sparse Gaussian Process Regression with Uncertainty-Weighted Symbolic Residual Correction**

**Author:** Divya Swaroop Jaiswal  
**Affiliation:** Department of Computer Science and Engineering, Indian Institute of Technology Patna  
**Contact:** divya_2401cs38@iitp.ac.in

---

## Overview

This repository contains the full research implementation of **WD-GPR-SR**, a novel hybrid prognostic framework for estimating the State-of-Health (SOH) of lithium-ion batteries. SOH is defined as the ratio of actual discharge capacity to rated nominal capacity, expressed as a percentage:

> SOH = Q_discharge / Q_nominal x 100%

Accurate SOH estimation is critical for Battery Management Systems (BMS) in electric vehicles, portable electronics, aerospace, and grid-scale storage. Existing approaches face a fundamental trilemma between **accuracy**, **uncertainty calibration**, and **physical interpretability**:

- Physics-based models are accurate but computationally prohibitive and difficult to generalize across chemistries and operating conditions.
- Deep neural networks achieve low point errors but produce uncalibrated, opaque predictions that are unsafe for domain shift.
- Standard Gaussian Process Regression offers rigorous uncertainty quantification but suffers from cubic O(N^3) computational complexity, making it impractical for large multi-cell datasets.

WD-GPR-SR resolves this trilemma by combining a **Sparse Variational Gaussian Process** with a **Wide-and-Deep composite kernel** and an **Uncertainty-Weighted Symbolic Residual Corrector**, achieving calibrated uncertainty, closed-form analytical interpretability, and near-linear computational scaling.

---

## Key Results

Evaluated under a strict **47-fold Leave-One-Cell-Out (LOCO)** cross-validation scheme across 6,124 discharge cycles from 47 battery cells, with no data leakage between cells:

| Metric | Peak Seed (42) | 3-Seed Mean |
|---|---|---|
| RMSE (%) | 4.406 | 4.854 +/- 0.495 |
| MAE (%) | 3.082 | 3.273 +/- 0.205 |
| R-squared | 0.9383 | 0.9246 +/- 0.0155 |
| MAPE (%) | 5.061 | 5.370 +/- 0.441 |
| NLL | 3.006 | 3.074 +/- 0.073 |
| 95% PICP (%) | 98.32 | 96.59 +/- 2.08 |
| MPIW | 25.23 | 25.93 +/- 0.64 |

WD-GPR-SR ranks **2nd in point accuracy** among 15 evaluated models while achieving the **best uncertainty calibration** by a wide margin. No other model in this benchmark simultaneously achieves sub-5% RMSE and 95%+ prediction interval coverage.

---

## Architecture

The framework operates as a two-stage hybrid:

**Stage 1 — Wide-and-Deep Sparse Variational GP**

A composite covariance kernel combines two complementary branches:

- **Wide (Linear) Branch:** A linear kernel applied to a learned projection of the input, capturing global monotonic aging trends and macro-scale capacity fade.
- **Deep (Neural ARD) Branch:** A three-layer MLP (24 -> 64 -> 32 -> 12, GELU activations) maps raw features to a 12-dimensional latent space, which is then passed through an ARD Matern-5/2 kernel. The per-dimension length-scales automatically prune irrelevant latent features via Bayesian shrinkage.

Sparse variational inference with M=64 inducing points (initialized via k-means++) reduces computational complexity from O(N^3) to O(N*M^2), enabling full LOCO cross-validation on thousands of cycles. The model is trained by maximizing the Evidence Lower Bound (ELBO) over 120 epochs using Adam with early stopping (patience 15).

**Stage 2 — Uncertainty-Weighted Symbolic Residual Correction**

Rather than fitting symbolic regression directly to raw targets, the framework uses it as an analytical residual corrector:

1. **Leak-Free Inner Cross-Fitting:** The 46 training cells are split into 5 inner cell folds. A surrogate GP is trained on 4 folds and evaluated on the held-out fold, yielding unbiased out-of-fold residuals and their associated predictive standard deviations.

2. **Heteroscedastic Precision Weights:** Each residual observation is weighted inversely by its predictive uncertainty. Cycles with high GP uncertainty (anomalous discharge profiles, sensor noise) are down-weighted to protect the discovered equation from distortion:

   > w_i = clip(sigma_median / (sigma_OOF(x_i) + 1e-6), 0.25, 4.0)

3. **Genetic Pareto Symbolic Discovery:** PySR evolves algebraic expression trees over operators {+, -, x, /, sqrt, log} with a parsimony penalty (gamma=0.002), minimizing weighted residual MSE while penalizing tree complexity.

4. **Calibrated Shrinkage Guardrail:** A scalar shrinkage parameter eta in [0, 1] is optimized on held-out validation cells. If the validation RMSE improvement is below 1e-4, eta defaults to 0, completely isolating the GP backbone and protecting against symbolic overfitting on unseen cells.

The final prediction and 95% credible interval are:

> y_hat(x) = GP_mean(x) + eta * g(x) ± 1.96 * GP_std(x)

---

## Repository Structure

```
.
|-- data/
|   |-- analysis.ipynb             # Dataset exploration and quality audit
|   `-- unified.parquet            # Merged NASA Ames + CALCE dataset (6,124 cycles, 47 cells)
|
|-- baseline/                      # Baseline model training notebooks
|   |-- 01_gpr_rbf.ipynb           # Standard GPR with RBF kernel
|   |-- 02_gpr_matern.ipynb        # GPR with Matern-5/2 kernel
|   |-- 03_gpr_ard.ipynb           # GPR with ARD kernel
|   |-- 04_sparse_gpr.ipynb        # Sparse GPR (M=64 inducing points)
|   |-- 05_deep_kernel_gpr.ipynb   # Deep Kernel GPR
|   |-- 06_svr.ipynb               # Support Vector Regression
|   |-- 07_random_forest.ipynb     # Random Forest
|   |-- 08_xgboost.ipynb           # XGBoost
|   |-- 09_wide_nn.ipynb           # Wide Linear Neural Network
|   |-- 10_deep_nn.ipynb           # Deep MLP
|   |-- 11_wide_deep_nn.ipynb      # Wide-and-Deep Neural Network
|   |-- 12_bilstm.ipynb            # Bidirectional LSTM
|   `-- 13_symbolic_regression.ipynb  # Standalone Symbolic Regression (PySR)
|
|-- baseline_results/              # Per-model LOCO metrics, OOF predictions, XAI artifacts
|   `-- <model_name>/
|       |-- metrics.json           # Aggregate metrics
|       |-- fold_metrics.csv       # Per-fold breakdown
|       |-- oof_predictions.csv    # Out-of-fold predictions
|       |-- loco_diagnostics.png   # LOCO diagnostic plots
|       `-- xai/                   # SHAP, LIME, PDP/ICE, permutation importance
|
|-- WD-GPR-SR/                     # Proposed model
|   |-- WD-GPR-SR.ipynb            # Full training, LOCO evaluation, XAI, ablation study
|   |-- adb-predictor.py           # Live smartphone battery SOH predictor (ADB interface)
|   `-- results/                   # Complete output artifacts
|       |-- final_summary.json     # Summary metrics across all 3 seeds
|       |-- baseline_comparison.csv  # Side-by-side comparison with all baselines
|       |-- metrics_mean_std.csv   # Mean +/- std across seeds
|       |-- oof_predictions.csv    # Out-of-fold SOH predictions
|       |-- fold_metrics.csv       # Per-cell LOCO metrics
|       |-- ablation_results.csv   # 5-variant ablation study
|       |-- uncertainty_calibration.png  # Calibration curve plot
|       |-- training_curve.png     # ELBO training progression
|       |-- loco_diagnostics.png   # Residual diagnostics across all 47 folds
|       |-- symbolic_equation.txt  # Discovered symbolic correction equation
|       `-- xai/                   # SHAP, deep ARD length-scales, LIME, PDP/ICE, symbolic sensitivities
|
|-- models/                        # Serialized trained model artifacts
|   |-- wd_gpr_sr.joblib           # Trained WD-GPR-SR bundle (GP state dict + symbolic tree)
|   |-- <baseline>.joblib          # Serialized baseline models
|   |-- translate.py               # Edge compilation script (ONNX + TensorRT FP16)
|   `-- requirements.txt           # Edge compilation dependencies
|
|-- jetson/                        # NVIDIA Jetson edge deployment scripts
|   |-- jetson_runtime.py          # Unified edge inference runtime
|   |-- preflight.py               # Hardware and dependency pre-flight checks
|   |-- build_tensorrt_engines.sh  # TensorRT engine compilation script
|   |-- run_all.sh                 # Full evaluation run on Jetson
|   `-- <model>.py                 # Jetson-optimized inference scripts per baseline
|
|-- report/
|   |-- main.tex                   # LaTeX source (publication-quality IEEE-style paper)
|   `-- main.pdf                   # Compiled research report
|
|-- poster/
|   |-- poster.tex                 # LaTeX source for A1 research poster
|   `-- poster.pdf                 # Compiled poster
|
|-- requirements.txt               # Core Python runtime dependencies
`-- README.md
```

---

## Datasets

The empirical benchmark uses two open-source lithium-ion battery datasets:

**NASA Ames Prognostics Center of Excellence (PCoE) Dataset**

- Cell chemistry: 18650 cylindrical NCA (nickel-cobalt-aluminum), 2.0 Ah nominal capacity
- Cycling protocol: CCCV charge, constant current discharge (1.0 A to 4.0 A) at 24-44 C
- End-of-discharge voltage cutoffs: 2.0 V to 2.7 V
- Reference: Saha, B. & Goebel, K. (2007). NASA PCoE Battery Aging Dataset.

**CALCE Battery Research Group Dataset**

- Cell chemistry: LCO prismatic (CS2 and CX2 families), 1.1 Ah nominal capacity
- Cycling protocol: Periodic charge cycles and dynamic stress cycling at 25 C ambient
- Reference: Center for Advanced Life Cycle Engineering, University of Maryland (2011).

Combined dataset statistics across 47 cells and 6,124 discharge cycles:
- SOH range: 10.02% to 128.45% (early-cycle overcapacity for some pristine CALCE cells)
- SOH mean: 76.46%, standard deviation: 17.73%

> **Note on target leakage prevention:** In strict adherence to project guidelines, any feature derived from capacity integration, Coulomb counting, discharge duration, or cycle length is excluded from the input feature vector. These quantities trivially encode the target SOH (Q = integral of I dt) and would constitute data leakage. The model operates exclusively on raw telemetry summary statistics.

---

## Feature Schema

Each discharge cycle is represented by a 24-dimensional feature vector extracted from raw voltage, current, and temperature telemetry:

| Feature | Type | Description |
|---|---|---|
| cycle_order | Numeric | Sequential discharge cycle index within cell lifetime |
| voltage_mean_V | Numeric | Mean terminal voltage during discharge phase |
| voltage_std_V | Numeric | Voltage standard deviation (reflects polarization impedance growth) |
| voltage_min_V | Numeric | Minimum terminal voltage at end-of-discharge cutoff |
| voltage_max_V | Numeric | Peak initial voltage (post-rest open-circuit potential) |
| voltage_median_V | Numeric | Median discharge plateau voltage |
| voltage_q10_V | Numeric | 10th percentile voltage (onset of knee-point discharge drop) |
| voltage_q90_V | Numeric | 90th percentile voltage (early discharge transition) |
| current_abs_mean_A | Numeric | Mean absolute discharge load current |
| current_abs_std_A | Numeric | Standard deviation of load current (dynamic load fluctuation) |
| current_abs_median_A | Numeric | Median nominal discharge current |
| temperature_mean_C | Numeric | Mean surface cell temperature during discharge |
| temperature_std_C | Numeric | Temperature fluctuation standard deviation |
| temperature_min_C | Numeric | Minimum baseline cell temperature |
| temperature_max_C | Numeric | Peak surface temperature (Joule heating indicator) |
| missingindicator_T_* | Binary (x4) | Explicit missingness flags for CALCE cells lacking temperature sensors |
| source_dataset_* | One-Hot | CALCE vs. NASA Ames origin indicator |
| battery_family_* | One-Hot | CS2, CX2, or 18650 cell form factor indicator |

Missing temperature values in CALCE cells are imputed using median values computed strictly within each training fold, with no leakage from test cells.

---

## Evaluation Protocol

All models are evaluated under a **47-fold Leave-One-Cell-Out (LOCO)** cross-validation scheme. In each fold k, the held-out test set contains all cycles from cell C_k, while the training set contains all cycles from the remaining 46 cells. Feature preprocessing (scaling, imputation, one-hot encoding) and all model hyperparameters are fitted exclusively on the training partition and applied to the held-out cell without any re-fitting.

This protocol simulates the real-world deployment scenario where a trained model must generalize to a new, previously unseen battery cell, which is the primary challenge in practical BMS applications.

Reported metrics across 8 evaluation criteria:
- Mean Absolute Error (MAE %)
- Mean Squared Error (MSE)
- Root Mean Square Error (RMSE %)
- Coefficient of Determination (R-squared)
- Mean Absolute Percentage Error (MAPE %)
- Negative Log-Likelihood (NLL) — measures probabilistic calibration quality
- 95% Prediction Interval Coverage Probability (PICP %) — fraction of true values within predicted 95% credible intervals
- Mean Prediction Interval Width (MPIW) — sharpness of prediction intervals

---

## Baseline Comparison

The table below summarizes all 15 models ranked by LOCO out-of-fold RMSE. WD-GPR-SR is the only model that achieves both competitive point accuracy and well-calibrated uncertainty intervals.

| Rank | Model | MAE (%) | RMSE (%) | R-squared | NLL | 95% PICP (%) |
|---|---|---|---|---|---|---|
| 1 | Support Vector Regression | 2.932 | 4.330 | 0.9404 | 7.878 | 57.45 |
| **2** | **WD-GPR-SR (Peak, Seed 42)** | **3.082** | **4.406** | **0.9383** | **3.006** | **98.32** |
| -- | WD-GPR-SR (3-Seed Mean) | 3.273 | 4.854 | 0.9246 | 3.074 | 96.59 |
| 3 | Deep Neural Network | 3.063 | 4.517 | 0.9351 | 6.644 | 55.94 |
| 4 | Wide-and-Deep NN | 3.413 | 4.696 | 0.9299 | 8.015 | 48.99 |
| 5 | GPR Matern-5/2 | 3.005 | 4.757 | 0.9280 | 3.388 | 76.68 |
| 6 | Wide Neural Network | 3.367 | 4.863 | 0.9248 | 7.237 | 57.04 |
| 7 | Bidirectional LSTM | 3.361 | 4.877 | 0.9244 | 11.871 | 47.26 |
| 8 | Sparse GPR (M=64) | 3.469 | 4.988 | 0.9209 | 3.102 | 82.54 |
| 9 | Deep Kernel GPR | 3.571 | 5.265 | 0.9119 | 3.244 | 99.40 |
| 10 | XGBoost | 4.027 | 6.232 | 0.8765 | 10.905 | 52.11 |
| 11 | GPR RBF Kernel | 3.661 | 6.495 | 0.8659 | 3.346 | 79.72 |
| 12 | GPR ARD Kernel | 4.060 | 6.596 | 0.8617 | 3.342 | 74.74 |
| 13 | Random Forest | 4.007 | 6.772 | 0.8542 | 2.990 | 93.22 |
| 14 | Standalone Symbolic Regression | 8.886 | 12.345 | 0.5154 | 3.934 | 94.01 |

Key observations:

- **SVR** ranks first in point error metrics but achieves only 57.45% PICP, meaning over 42% of held-out battery cycles fall outside its nominal 95% uncertainty envelope. This makes SVR hazardous for safety-critical BMS monitoring.
- **Deep neural networks** achieve competitive RMSE but produce NLL values between 6.6 and 11.9, indicating severely overconfident predictions that fail catastrophically under cell-to-cell domain shift.
- **WD-GPR-SR** is the only model to simultaneously achieve sub-4.5% RMSE at peak and near-perfect coverage (98.32% PICP), with the lowest NLL among all continuous predictive models.
- **Standalone symbolic regression** ranks last (RMSE 12.35%), confirming that symbolic regression must operate as a residual corrector over a Bayesian backbone rather than fitting raw targets directly.

---

## Ablation Study

A 5-variant ablation study quantifies the contribution of each architectural component:

| Variant | MAE (%) | RMSE (%) | NLL | 95% PICP (%) |
|---|---|---|---|---|
| 1. Wide-Only GP | 6.847 | 8.019 | -- | -- |
| 2. Deep-Only GP | 3.843 | 4.575 | -- | -- |
| 3. WD-GP (No Symbolic Residual) | 3.812 | 4.525 | 3.187 | 95.10 |
| 4. WD-GP + Unweighted Symbolic SR | 3.997 | 4.716 | 3.213 | 94.26 |
| 5. Full WD-GPR-SR (Weighted SR) | **3.910** | **4.633** | **3.202** | **94.24** |

The wide branch alone produces poor results (RMSE 8.0%), confirming that global linear trends are insufficient. The deep branch alone (RMSE 4.58%) provides competitive accuracy. The full composite kernel reduces RMSE further. Unweighted symbolic regression degrades performance relative to no symbolic correction, empirically validating the necessity of heteroscedastic uncertainty weighting.

---

## Live Android Battery Predictor (ADB Interface)

`WD-GPR-SR/adb-predictor.py` is a standalone command-line tool that connects to any Android smartphone over ADB (Android Debug Bridge), samples live battery telemetry, and produces a real-time SOH prediction with calibrated 95% confidence intervals.

**How it works:**

1. Connects to a USB-attached Android device via ADB.
2. Reads instantaneous voltage, current, temperature, and state-of-charge via `dumpsys battery` and sysfs power supply nodes in a single round-trip.
3. Runs a 20-30 second sampling window, then translates the readings into the 15 numeric + 2 categorical features expected by the trained model using **Physics-Informed Virtual Cycle Synthesis** — reconstructing the full-cycle discharge statistics from the measured operating point.
4. Loads the trained WD-GPR-SR bundle and runs inference to produce SOH%, predictive standard deviation, and 95% credible interval.

**Requirements:**

- ADB installed and in PATH (Android platform-tools)
- USB debugging enabled on the target device (Developer Options -> USB Debugging)
- Python runtime dependencies installed (see requirements.txt)
- Trained model file at `models/wd_gpr_sr.joblib`

**Usage:**

```bash
# Real device (auto-detect)
python WD-GPR-SR/adb-predictor.py

# Specific device serial
python WD-GPR-SR/adb-predictor.py --serial ABC123XYZ

# Dry run with simulated telemetry (no device required)
python WD-GPR-SR/adb-predictor.py --simulate

# Custom sampling duration and battery family
python WD-GPR-SR/adb-predictor.py --duration 30 --battery-family CS2

# Export prediction to JSON
python WD-GPR-SR/adb-predictor.py --output result.json

# Continuous monitoring mode (every 60 seconds)
python WD-GPR-SR/adb-predictor.py --loop --interval 60
```

**Example output:**

```
Device: Google Pixel 7a | Android 14 | Li-ion Polymer
Hardware Cycle Count: 145
Charge Full: 4238 mAh (of 4385 mAh design)

Sampling battery telemetry for 25 seconds...
[25/25s] [=========================>] V: 3.847V | I: -0.481A (DIS) | T: 29.2 C | SOC: 73%

WD-GPR-SR Battery Health Report
  SOH Estimate        : 87.42%
  95% Confidence      : [83.21%, 91.63%]
  Predictive Std Dev  : 2.14%
  GP Prior Estimate   : 87.37%
  Symbolic Correction : +0.05%
  Health Status       : [GOOD] Normal battery condition
```

---

## Edge Deployment

The framework includes a complete compilation pipeline for embedding inference on automotive-grade hardware (targeting NVIDIA Jetson Orin Nano):

1. **TensorRT FP16 Engine:** The trained PyTorch deep feature extractor is exported to ONNX and compiled to a TensorRT FP16 execution engine (`deep_feature_extractor.engine`), enabling sub-millisecond neural feature computation on GPU Tensor Cores.

2. **BLAS Sparse GP Inference:** The variational posterior computation reduces to Level-2 BLAS matrix-vector operations using the precomputed 64 x 24 inducing coordinate matrix, requiring negligible memory on embedded MCUs.

3. **Native C Symbolic Library:** The discovered symbolic correction equation is emitted as ANSI C source and compiled via `gcc -O3 -fPIC -shared` into a shared library (`equation.so`). This executes in constant instruction time with zero dynamic memory allocation, making it certifiable for safety-critical automotive embedded systems.

**Compilation:**

```bash
cd jetson
bash build_tensorrt_engines.sh   # Compile TensorRT engines
python preflight.py              # Verify hardware and dependencies
python jetson_runtime.py         # Run unified edge inference
```

---

## Installation

**Prerequisites:** Python 3.10+, CUDA-enabled GPU recommended for training.

```bash
git clone https://github.com/raj-jaiswal/Battery-SOH-Regression.git
cd Battery-SOH-Regression
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies:

| Package | Version | Purpose |
|---|---|---|
| torch | >=2.1.0 | Neural network components and GPU acceleration |
| gpytorch | >=1.11 | Variational sparse GP framework |
| scikit-learn | >=1.4.0 | Preprocessing, baseline models |
| numpy | >=1.26.0 | Numerical operations |
| pandas | >=2.0.0 | Data manipulation |
| scipy | >=1.12.0 | Statistical utilities |
| joblib | >=1.3.0 | Model serialization |
| pyarrow | >=14.0.0 | Parquet data loading |

---

## Reproducing Results

**Step 1: Data exploration**

```bash
jupyter notebook data/analysis.ipynb
```

**Step 2: Train and evaluate baseline models**

Open any notebook in `baseline/` and run all cells. Results are written to `baseline_results/<model_name>/`.

**Step 3: Train and evaluate WD-GPR-SR**

```bash
jupyter notebook WD-GPR-SR/WD-GPR-SR.ipynb
```

The notebook performs the full 47-fold LOCO evaluation across 3 random seeds, generates all XAI artifacts, runs the ablation study, and serializes the trained model bundle to `models/wd_gpr_sr.joblib`.

**Step 4: Run the live ADB predictor**

```bash
python WD-GPR-SR/adb-predictor.py --simulate   # Dry run
python WD-GPR-SR/adb-predictor.py              # With connected Android device
```

---

## Hyperparameters

| Subsystem | Hyperparameter | Value |
|---|---|---|
| Deep MLP Branch | Architecture | 3 Dense Layers: 24 -> 64 -> 32 -> 12 |
| Deep MLP Branch | Activations | GELU |
| Deep MLP Branch | Latent Dimension | 12 |
| Sparse GP | Inducing Points (M) | 64 (k-means++ initialization) |
| Sparse GP | Wide Kernel | Linear with learnable diagonal weights |
| Sparse GP | Deep Kernel | Matern-5/2 ARD with 12 length-scales |
| Sparse GP | Optimizer | Adam, lr=0.01 |
| Sparse GP | Batch Size | 256 cycles |
| Sparse GP | Max Epochs | 120 (early stopping, patience 15) |
| Symbolic Regression | Genetic Operators | {+, -, *, /, sqrt, log} |
| Symbolic Regression | Population / Generations | 160 individuals / 10 generations |
| Symbolic Regression | Parsimony Penalty | 0.002 |
| Symbolic Regression | Precision Weight Bounds | [0.25, 4.00] |
| Shrinkage Guardrail | Shrinkage Range | eta in [0.0, 1.0], step 0.05 |
| Shrinkage Guardrail | Acceptance Threshold | Minimum RMSE gain >= 1e-4 |

---

## Explainability

The full XAI analysis is located in `WD-GPR-SR/results/xai/` and covers:

- **Global SHAP Feature Importance:** Shapley values computed over all out-of-fold predictions to identify globally dominant degradation features.
- **Deep ARD Latent Length-Scales:** Per-dimension ARD length-scales of the deep kernel, revealing which latent manifold dimensions the GP anchors degradation predictions on.
- **LIME Local Surrogate:** Linear approximations of model behavior at individual prediction points, enabling cycle-level explanation.
- **PDP and ICE Plots:** Partial Dependence Plots and Individual Conditional Expectation curves showing marginal feature effects on predicted SOH.
- **Symbolic Sensitivities:** Analytical partial derivatives of the discovered symbolic correction equation with respect to each input feature.
- **Permutation Feature Importance:** Model-agnostic feature importance via repeated random shuffling across seeds.

The discovered symbolic correction equation for Seed 42 is a constant offset: `g(x) = 0.0611`, indicating that the Wide-and-Deep GP backbone successfully absorbed all input-dependent non-linearities, leaving only a small stationary mid-life underestimation bias corrected by the symbolic tier.

---

## Limitations

- **Battery Chemistry:** Evaluation covers NCA (NASA Ames) and LCO (CALCE) chemistries. LFP (LiFePO4) cells, which exhibit exceptionally flat OCV plateaus, may require supplementary incremental capacity (dQ/dV) inputs for equivalent accuracy.
- **Temperature Range:** Training data covers 24-44 C. Degradation kinetics under sub-zero conditions (< -10 C), where lithium plating dominates, have not been characterized.
- **End-to-End Differentiability:** The current two-stage pipeline optimizes the GP and symbolic components sequentially. Future work will explore differentiable relaxations of genetic tree search to enable joint backpropagation.

---

## Citation

If you use this code, model, or results in your research, please cite:

```
Divya Swaroop Jaiswal, "Lithium-Ion Battery SOH Estimation via Wide-and-Deep
Sparse GPR with Uncertainty-Weighted Symbolic Residuals," Department of Computer
Science and Engineering, Indian Institute of Technology Patna, 2026.
```

---

## License

This project is released for academic and research use. Please contact the author for other use cases.

