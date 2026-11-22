# Data Pipeline (`data/`)

This directory contains the dataset preparation pipeline for the **Federated Human Activity Recognition (HAR)** project.

---

## 1. Supported Dataset
Currently configured for the **MotionSense** dataset:
- **Sensor:** Raw 3-axis Accelerometer ($x, y, z$) in $g$ units ($1.0g \approx$ resting gravity).
- **Sampling Rate:** $50\text{ Hz}$.
- **Subjects:** 24 participants across 15 trial scenarios.
- **Classes (6):**
  - `0`: Downstairs (`dws`)
  - `1`: Upstairs (`ups`)
  - `2`: Walking (`wlk`)
  - `3`: Jogging (`jog`)
  - `4`: Sitting (`sit`)
  - `5`: Standing (`std`)

---

## 2. What the Pipeline Does
`prepare_dataset.py` processes raw trial CSVs into ML-ready inputs:
- **Windowing:** Sliding window of **128 samples (2.56s)** with a **stride of 64 samples (50% overlap)**.
- **Feature Extraction:** Calculates orientation-invariant acceleration magnitude:
  $$a_{mag} = \sqrt{x^2 + y^2 + z^2}$$
  yielding an input shape of `(128,)` (`float32`).
- **Subject-Wise Holdout Split:**
  - **Train:** Subjects `1..20` (18,296 windows).
  - **Global Test:** Subjects `21..24` (3,569 windows).
- **Federated Partitions:**
  - **IID:** 10 clients with randomly shuffled, evenly split shards of the centralized training set (~1,830 windows/client).
  - **Non-IID:** 10 clients partitioned strictly by subject pairs (`client_0` = Subjects 1–2, `client_1` = Subjects 3–4, ..., `client_9` = Subjects 19–20) reflecting realistic individual gait variations.
- **Mobile Offline Asset:** Generates `mobile_demo_bundle.json` with 180 balanced test windows (30 per class) with raw 3-axis coordinates rounded to 4 decimals for mobile preprocessing/inference validation.

---

## 3. How to Prepare the Dataset

### Option A: Fully Automated (Recommended)
Run the script directly from the repository root. If raw data is missing, the script will automatically clone the repository and extract the accelerometer archive:

```bash
python data/prepare_dataset.py
```

### Option B: Manual Setup
If you prefer to download or manage raw archives manually:

1. Create directory `data/raw/` and clone the MotionSense repository:
   ```bash
   git clone --depth 1 https://github.com/mmalekzadeh/motion-sense.git data/raw/motion-sense
   ```
2. Extract `B_Accelerometer_data.zip`:
   - Archive location: `data/raw/motion-sense/data/B_Accelerometer_data.zip`
   - Destination directory: `data/raw/extracted/`
   - Ensure the path resolves to: `data/raw/extracted/B_Accelerometer_data/`
3. Run the processing script:
   ```bash
   python data/prepare_dataset.py
   ```

---

## 4. Output Structure (`data/processed/`)
All generated artifacts are ignored by Git via `.gitignore`:

```text
data/processed/
├── metadata.json               # Schema, sample counts, and class balance statistics
├── centralized_train.npz       # Centralized baseline train set (X: [18296, 128], y: [18296])
├── global_test.npz             # Unseen holdout test set (X: [3569, 128], y: [3569])
├── iid/
│   ├── client_0.npz ... client_9.npz
├── non_iid/
│   ├── client_0.npz ... client_9.npz
└── mobile_demo_bundle.json     # 180 balanced raw 3-axis windows for Flutter test harness
```
