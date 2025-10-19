# ECG Dataset Processing Guide

This guide describes how to build and preprocess two ECG datasets for atrial fibrillation (AF) forecasting experiments: **ICENTIA11K** and **IRIDIA-AF**.

---

## ICENTIA11K

### 1. Download Dataset
Download the ICENTIA11K dataset from https://physionet.org/content/icentia11k-continuous-ecg/1.0/

Then, modify the path variables in `icentia11k/cfgs/dataset_cfg.yaml`

### 2. Calculate AF Burden per Sample - `select_af_samples.py`
```bash
python icentia11k/select_af_samples.py icentia11k/cfgs/dataset_cfg.yaml
```

**Functionality:**
- Calculates AF burden for each segment of each patient
- Designates patients with any detected AF burden as experimental subjects
- Outputs: `af_burden_list.csv`

### 3. Build Sampling Spans - `build_sampling_spans.py`
```bash
python icentia11k/build_sampling_spans.py icentia11k/cfgs/dataset_cfg.yaml
```

**Functionality:**
- Identifies 5-minute spans corresponding to label 1 (pre-AF) and spans available for sampling label 0 (non-AF) samples
- Calculates AF burden per patient and skips patients whose AF burden exceeds a specified threshold
- Creates rows even for cases where spans cannot be extracted (e.g., only sustained AF episodes from the beginning, intermittent AF throughout the entire recording)
- Outputs: `span_list.csv`

### 4. Split Patients into Training/Validation/Test - `split_patients.py`
```bash
python icentia11k/split_patients.py icentia11k/cfgs/dataset_cfg.yaml
```

**Functionality:**
- Divides patients into training, validation, and test sets
- Excludes patients with no label 0 spans or no label 1 spans
- Uses configuration options: `PERSONALIZED_MIN_NUM_SEGMENTS`, `PERSONALIZED_TRAINING_SEGMENTS`, `PERSONALIZED_MIN_EPISODES`
- Outputs: `patient_split.csv`

### 5. Build Final Dataset - `build_dataset.py`
```bash
python icentia11k/build_dataset.py icentia11k/cfgs/dataset_cfg.yaml
```

**Functionality:**
- Constructs the final dataset by dividing data into windows and computing features
- Balances the number of samples per label for each patient
- Outputs:
  - `training.pickle` (training set)
  - `validation.pickle` (validation set)
  - `test_p_tr.pickle` (patient-specific training set)
  - `test_p_te.pickle` (patient-specific test set)

---

## IRIDIA-AF

### 1. Download Dataset
Download the IRIDIA-AF dataset from https://zenodo.org/records/8405941

Then, modify the path variables in `iridia-af/cfgs/dataset_cfg.yaml`

### 2. Preprocess Data - `preprocess.py`
Modify the path variables in `iridia-af/preprocess.py` before running.

```bash
python iridia-af/preprocess.py
```

**Functionality:**
- Resamples lead I from 200Hz to 250Hz (for ICENTIA11K external validation)
- Outputs:
  - `ecgSignal1.npy` (resampled ECG signal)
  - `sample_length1.txt` (recording length)
  - `rhythmClassP1.csv` (AF episodes recorded in point-based format)

### 3. Calculate AF Burden per Sample - `select_af_samples.py`
```bash
python iridia-af/select_af_samples.py iridia-af/cfgs/dataset_cfg.yaml
```

**Functionality:**
- Calculates AF burden for each patient
- Designates patients with any detected AF burden as experimental subjects
- Outputs: `af_burden_list.csv`

### 4. Build Sampling Spans - `build_sampling_spans.py`
```bash
python iridia-af/build_sampling_spans.py iridia-af/cfgs/dataset_cfg.yaml
```

**Functionality:**
- Identifies 5-minute spans corresponding to label 1 (pre-AF) and spans available for sampling label 0 (non-AF) samples
- Calculates AF burden per patient and skips patients whose AF burden exceeds a specified threshold
- Creates rows even for cases where spans cannot be extracted (e.g., only sustained AF episodes from the beginning, intermittent AF throughout the entire recording)
- Outputs: `span_list.csv`

### 5. Split Patients into Training/Validation/Test - `split_patients.py`
```bash
python iridia-af/split_patients.py iridia-af/cfgs/dataset_cfg.yaml
```

**Functionality:**
- Divides patients into training, validation, and test sets
- Excludes patients with no label 0 spans or no label 1 spans
- Uses configuration options: `PERSONALIZED_MIN_LENGTH_POINTS`, `PERSONALIZED_TRAINING_DAYS`, `PERSONALIZED_MIN_EPISODES`
- Outputs: `patient_split.csv`

### 6. Build Final Dataset - `build_dataset.py`
```bash
python iridia-af/build_dataset.py iridia-af/cfgs/dataset_cfg.yaml
```

**Functionality:**
- Constructs the final dataset by dividing data into windows and computing features
- Balances the number of samples per label for each patient
- Outputs:
  - `test_p_tr.pickle` (patient-specific training set)
  - `test_p_te.pickle` (patient-specific test set)

---

## Label Definitions

- **Label 0**: Non-AF segments (no AF within specified distance)
- **Label 1**: Pre-AF segments (segments immediately before AF onset)
- **Label 2**: AF segments (segments during AF episodes)
