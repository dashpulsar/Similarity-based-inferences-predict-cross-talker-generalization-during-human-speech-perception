---

# Similarity-Based Inferences Predict Cross-Talker Generalization During Human Speech Perception

## Technical Documentation

---

## 1. Project Overview

This project investigates whether **similarity-based inferences** — as computed from deep speech representations — can predict how well human listeners generalize their speech perception abilities across different talkers. The core hypothesis is that listeners who are exposed to speech from certain talkers during a training phase will better understand a novel test talker if that test talker sounds *more similar* to the training talkers.

The pipeline consists of the following stages:

1. **Human behavioral data**: Collect and preprocess experimental data from three published speech perception studies.
2. **Feature extraction**: Use the HuBERT model (a self-supervised speech representation model) to simulate auditory processing and extract layer-wise representations from speech audio.
3. **Dimensionality reduction**: Apply t-SNE to project high-dimensional representations into a 3-dimensional space.
4. **Similarity computation**: Calculate pairwise talker similarity using Dynamic Time Warping (DTW) with Minkowski distance, then transform distances into similarity scores via an exponential decay function.
5. **Statistical modeling**: Fit Generalized Linear Mixed Models (GLMMs) to test whether the computed similarity predicts human behavioral accuracy.
6. **Variability analysis**: Compute various variability measures across linguistic units to characterize the acoustic diversity of the exposure signal.
7. **Baseline comparisons**: Compare HuBERT-based representations against traditional acoustic features (MFCC and STRF).

---

## 2. Human Behavioral Data

### 2.1 Datasets

The project uses behavioral data from three published speech perception experiments. Each experiment exposes human listeners to accented speech during a training phase and then tests their comprehension of (potentially novel) talkers during a test phase.

| Dataset | Reference | Experiment ID | Item Granularity | Accents | Talkers |
|---------|-----------|---------------|-----------------|---------|--------|
| **Alexander & Nygaard (2019, Exp. 2)** | AN19 | `AN19` | **Word** only | L1-English, L1-Korean, L1-Spanish | Multiple per accent group |
| **Xie, Liu & Jaeger (2021, Exp. 1a)** | X21 | `X21` | **Sentence**, **Word**, **Phoneme** | L1-English (ENG), L1-Mandarin (CMN) | 11 talkers (6 CMN, 5 ENG) |
| **Bradlow, Bassard & Paller (2023)** | B23 | `B23` | **Sentence** only | Portuguese (PBR), Farsi (FAR), Turkish (TUR), Spanish (SPA) | 4 talkers (1 per accent) |

#### 2.1.1 Alexander & Nygaard 2019 (AN19)

- **Granularity**: Word-level only.
- **Structure**: Each observation corresponds to one isolated word transcription. Each item ID has the format `AN19.{accent_id}.{talker_id}.W{number}`.
- **Conditions**: Listeners experience either single-accent or multi-accent exposure, and are tested on within-accent or cross-accent generalization.
- **Sound stimuli**: Organized by accent group into four directories: `L1-English`, `L1-Korean L2-English`, `L1-Spanish L2-English`, `L1-mixed L2-English`.

#### 2.1.2 Xie, Liu & Jaeger 2021 (X21)

- **Granularity**: The dataset supports **sentence**, **word**, and **phoneme** levels.
  - **Sentence**: Full utterance-level annotations from Praat TextGrid Tier 0.
  - **Word**: Keyword-level annotations from TextGrid Tier 1.
  - **Phoneme**: Fine-grained phoneme-level annotations from TextGrid Tier 2 (excluding silence marks `sp`).
- **Talkers**: 11 talkers — 6 Mandarin-accented (CMN) and 5 native English (ENG).
- **Sentence subset**: Only specific sentences are used: indices `[0–10, 12–16]` (Set 1) and `[17–22, 24–31, 37, 40]` (Set 2), yielding 32 sentences total.

#### 2.1.3 Bradlow, Bassard & Paller 2023 (B23)

- **Granularity**: Sentence-level only.
- **Talkers**: 4 talkers from different L1 backgrounds (Portuguese, Farsi, Turkish, Spanish). Each talker has two recording halves (HT1, HT2).

### 2.2 Preprocessed Data Format

All three datasets are standardized into a common CSV format (`*-behavioral-data.csv`) with key columns including `experiment_id`, `participant_id`, `exposure_condition.accent`, `exposure_condition.talker`, `test_condition.accent_generalization`, `test_condition.talker_generalization`, `phase`, `item_id`, `response_correct`, and `response_incorrect`.

---

## 3. Feature Extraction

### 3.1 HuBERT Model

Two model variants are used:

| Variant | Hugging Face ID | Description |
|---------|----------------|-------------|
| **Base (pre-trained)** | `facebook/hubert-large-ll60k` | Pre-trained on 60k hours of Libri-Light |
| **Fine-tuned** | `facebook/hubert-large-ls960-ft` | Fine-tuned on 960h LibriSpeech for ASR |

### 3.2 Layer Selection

- **CNN layers**: `[2, 3, 4, 5, 6]`
- **Transformer layers**: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]`

This yields **18 layers** total.

### 3.3 CNN Layer Temporal Alignment (Adaptive Average Pooling)

Different CNN layers produce outputs with different temporal resolutions. To align all CNN layers to a common temporal dimension, we apply **1D Adaptive Average Pooling**:

1. Identify the target temporal dimension `T_target` from the **last CNN layer** (layer 6).
2. For each CNN layer $i$ with temporal output $T_i$: if $T_i \neq T_{\text{target}}$, apply `torch.nn.functional.adaptive_avg_pool1d(feat, T_target)`.
3. All CNN outputs are transposed from `(B, C, T)` to `(T, C)`.

Implementation: `feature_utils.py :: _extract_selected_layers()`

### 3.4 CNN Hook Registration

CNN layer activations are captured using **PyTorch forward hooks** registered on `model.feature_extractor.conv_layers`.

Implementation: `feature_utils.py :: _register_cnn_hooks()`

### 3.5 Audio Preprocessing

All audio is resampled to **16 kHz**.

### 3.6 Feature Storage (HDF5)

Extracted features are stored in HDF5 with two layouts:
1. **Raw features**: `[speaker_id / item_id / layer_key]` → `(T, D)` array
2. **t-SNE reduced**: `[layer_key / speaker_id / item_id]` → `(T, 3)` array

---

## 4. Dimensionality Reduction (t-SNE)

### 4.1 Configuration

| Parameter | Value |
|-----------|-------|
| `n_components` | **3** |
| `random_state` | **42** |
| `n_jobs` | **1** (per worker; parallelized across layers) |

### 4.2 Procedure

1. For each layer, all frames across all speakers and items are vertically stacked.
2. t-SNE is fit on this global matrix, projecting to `(N_total_frames, 3)`.
3. Reduced frames are split back per-speaker, per-item using stored metadata.
4. Layer-wise t-SNE is parallelized across up to **30 CPU cores** using `joblib`.

### 4.3 Alternative: Full-Dimensional Computation

We also tested computing similarity directly on the **original high-dimensional features** (without t-SNE reduction).

---

## 5. Standardization and Normalization

### 5.1 Global Z-Score Standardization

$$x_{\text{std}} = \frac{x - \mu}{\sigma + \epsilon}$$

where $\epsilon = 10^{-8}$, applied per feature dimension.

Implementation: `project_utils.py :: standardization()` and `feature_utils.py :: standardize_features()`

### 5.2 Instance Normalization

An alternative that applies standardization **per word instance**:

$$x_{\text{inst}} = \frac{x - \mu_{\text{word}}}{\sigma_{\text{word}} + \epsilon}$$

### 5.3 Similarity Predictor Scaling in GLMM (Nygaard)

Gelman's (2008) standardization:

$$\text{similarity\_scaled} = \frac{\text{similarity} - \mu_{\text{sim}}}{2 \times \sigma_{\text{sim}}}$$

Computed on the training split and applied consistently to both train and test.

---

## 6. Similarity Computation

### 6.1 Word-Level Feature Alignment

Features are segmented to word level using Praat TextGrid annotations:

$$\text{frame\_start} = \text{round}\left(\frac{T_{\text{total}} \times (t_{\text{word\_start}} - t_{\text{sent\_start}})}{t_{\text{sent\_end}} - t_{\text{sent\_start}}}\right)$$

### 6.2 Dynamic Time Warping (DTW)

$$D[i, j] = \text{cost}(i, j) + \min(D[i-1, j],\; D[i, j-1],\; D[i-1, j-1])$$

Normalized by average length:

$$d_{\text{DTW}}(\mathbf{X}, \mathbf{Y}) = \frac{D[n, m]}{(n + m) / 2}$$

### 6.3 Frame-Level Cost Functions

**Weighted Minkowski Distance (Default, $\tau = 2$):**

$$\text{cost}(x_t, y_t) = \left(\sum_{m=1}^{D} w \cdot |x_{t,m} - y_{t,m}|^{\tau}\right)^{1/\tau}$$

**Cosine Distance:**

$$\text{cost}(x_t, y_t) = 1 - \frac{x_t \cdot y_t}{\|x_t\| \cdot \|y_t\|}$$

Both compiled with Numba `@njit`.

### 6.4 Exponential Transformation (Distance → Similarity)

$$\text{similarity} = \exp(-d_{\text{raw}} \times k)$$

where $k > 0$ is optimized through cross-validation.

### 6.5 Method Variants Tested

| Condition | Feature Space | Distance Metric | t-SNE |
|-----------|--------------|-----------------|-------|
| HuBERT + t-SNE + Minkowski (default) | 3D t-SNE | DTW + Minkowski | Yes |
| HuBERT + t-SNE + Minkowski (fine-tuned) | 3D t-SNE | DTW + Minkowski | Yes |
| HuBERT + Full-dim + Minkowski | Original 1024D | DTW + Minkowski | No |
| HuBERT + Full-dim + Cosine | Original 1024D | DTW + Cosine | No |
| HuBERT + Full-dim + Cosine (fine-tuned) | Original 1024D | DTW + Cosine | No |
| HuBERT + t-SNE + Instance Norm | 3D t-SNE (inst norm) | DTW + Minkowski | Yes |
| MFCC Baseline | 39D | DTW + Minkowski | No |
| STRF Baseline | 24D | DTW + Minkowski | No |

---

## 7. Statistical Modeling: GLMMs

### 7.1 GLMM for Xie (X21)

```r
glmer(cbind(numCorrect, numIncorrect) ~ 1 + similarity + (1 | SentenceID / Keyword) + (1 | TestTalkerID),
      data = data, family = binomial(link = "logit"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10000)))
```

### 7.2 GLMM for Nygaard (AN19)

```r
glmer(cbind(numCorrect, numIncorrect) ~ similarity_scaled + (1 + similarity_scaled | SubjectID),
      data = data, family = binomial(link = "logit"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5)))
```

### 7.3 K-Fold Cross-Validation

**Xie**: 3-fold stratified, $k$ optimized via `scipy.optimize.minimize_scalar` (range $[0.001, 5.0]$). Mean $k$ used for corrected evaluation.

**Nygaard**: 3-fold, $k$ optimized via `optuna` with L2-regularized objective:

$$\mathcal{L} = -z_{\text{target}} + \alpha \cdot k^2 \quad (\alpha = 0.1)$$

### 7.4 Behavioral Noise Ceiling (Jaeger Self-Predictability Method)

The behavioral noise ceiling quantifies the **upper bound** of predictive performance — the maximum achievable z-statistic using observed human behavior as the predictor.

We use **Florian Jaeger's Self-Predictability Method**, which dynamically computes the ceiling per cross-validation fold without data leakage:

**Procedure (per CV fold):**

1. **Compute per-item log-odds from TRAINING data only**: For each unique item (word × talker), compute proportion correct across all training subjects, then convert to log-odds:

$$\text{logodds} = \log\frac{p}{1-p}$$

with $p$ clipped to $[0.01, 0.99]$ to avoid infinity.

2. **Map training-derived log-odds to test data**: Items in the test fold receive the log-odds computed from the training fold only.

3. **Scale using TRAINING statistics** (Gelman, 2008):

$$\text{logodds\_scaled} = \frac{\text{logodds} - \mu_{\text{train}}}{2 \times \sigma_{\text{train}}}$$

4. **Fit GLMM on test data** with training-derived predictor:

**Nygaard ceiling GLMM:**
```r
glmer(cbind(numCorrect, numIncorrect) ~ logodds_scaled + (1 + logodds_scaled | SubjectID),
      data = r_test, family = binomial(link = "logit"))
```

**Xie ceiling GLMM:**
```r
glmer(cbind(numCorrect, numIncorrect) ~ 1 + logodds_scaled + (1 | SentenceID / Keyword) + (1 | TestTalkerID),
      data = r_test, family = binomial(link = "logit"))
```

5. **Extract Wald z-statistic** per fold. The ceiling is the **mean z** across all folds.

**Key design principles:**
- **No cross-fold leakage**: predictor computed from training data only
- **Log-odds scale**: natural parameter space for logistic GLMM
- **Consistent evaluation**: same GLMM fitting + cross-validation pipeline as model comparisons

Implementation: `project_utils.py :: compute_jaeger_ceiling_nygaard()` and `compute_jaeger_ceiling_xie()`

---

## 8. Baseline Features

### 8.1 MFCC (39D: 13 MFCC + Δ + ΔΔ)

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| `n_mfcc` | 13 |
| `n_fft` | 400 (25ms) |
| `hop_length` | 160 (10ms) |
| `n_mels` | 23 |
| `center` | False |

### 8.2 STRF (24D Gabor Kernels)

$$\text{Envelope} = \exp\left(-0.5 \left[(T \cdot r \cdot 1.5)^2 + (F \cdot s \cdot 1.5)^2\right]\right)$$

$$K_{\text{real}} = \text{Env} \cdot \cos(2\pi(rT + dsF)), \quad K_{\text{imag}} = \text{Env} \cdot \sin(2\pi(rT + dsF))$$

Magnitude: $\text{STRF} = \sqrt{K_r^2 + K_i^2}$, averaged across frequency.

Rates: `[2, 4, 8, 16]` Hz, Scales: `[0.25, 0.5, 1.0]`, Directions: `[+1, −1]` → 24 kernels.

---

## 9. Variability Analysis

### 9.1 Linguistic Unit Segmentation

| Level | Source | Description |
|-------|--------|-------------|
| Sentence | TextGrid Tier 0 | Full utterance boundaries |
| Word | TextGrid Tier 1 | Keyword boundaries |
| Phoneme | TextGrid Tier 2 | Phone boundaries (excl. silence) |

### 9.2 Variability Measure Categories

#### 9.2.1 Generalized Order-Insensitive Variance

$$V_{\text{OI}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{d=1}^{D} |x_{t,d} - \mu_d|^{\tau}$$

Scopes: WithinToken, WithinType, BetweenType.

#### 9.2.2 Generalized Order-Sensitive Variance

$$V_{\text{OS}} = \frac{1}{T-1} \sum_{t=2}^{T} \sum_{d=1}^{D} |x_{t,d} - x_{t-1,d}|^{\tau}$$

Scopes: OrderSentence, OrderWord, OrderPhoneme.

#### 9.2.3 Mean Order-Sensitive Dissimilarity

$$V_{\text{MOS}} = \frac{2}{N(N-1)} \sum_{i < j} d_{\text{DTW}}(S_i, S_j)$$

Scopes: BetweenSentence, BetweenTypeWord, BetweenTypePhoneme.

---

## 10. Software Dependencies

| Package | Purpose |
|---------|--------|
| `torch`, `torchaudio` | HuBERT model, MFCC extraction, audio processing |
| `transformers` | Hugging Face model interface |
| `numpy`, `pandas` | Data manipulation |
| `h5py` | HDF5 feature storage |
| `sklearn` | t-SNE, K-fold cross-validation |
| `scipy` | Scalar optimization |
| `optuna` | Bayesian hyperparameter optimization |
| `numba` | JIT compilation for DTW |
| `rpy2` | Python-R interface for GLMM |
| `lme4` (R) | Generalized Linear Mixed Models |
| `textgrid` | Praat TextGrid parsing |
| `librosa` | Audio resampling |
| `joblib` | Parallel execution |
| `seaborn`, `matplotlib` | Visualization |

---
