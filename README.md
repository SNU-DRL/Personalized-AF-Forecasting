# Personalized Deep Learning Models for Forecasting Impending Atrial Fibrillation Episodes Using Wearable Electrocardiograms

This repository contains the code for the experiments conducted in our research paper titled _Personalized Deep Learning Models for Forecasting Impending Atrial Fibrillation Episodes Using Wearable Electrocardiograms_.
The paper is under review; a detailed explanation will be added after publication.

## Requirements
This experiment was tested using the following environments:
- Python 3.10
- PyTorch 1.12.1
- Please refer to `requirements.txt` for other libraries.

```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Building Datasets
Please refer to the [Dataset README](dataset/README.md) for detailed instructions on how to build and preprocess the ICENTIA11K and IRIDIA-AF datasets.

The dataset building process will generate the following files:
- **ICENTIA11K**: `training.pickle`, `validation.pickle`, `test_p_tr.pickle`, `test_p_te.pickle`
- **IRIDIA-AF**: `test_p_tr.pickle`, `test_p_te.pickle`

### 2. Running Experiments

Our experiments follow a four-stage pipeline:

#### Stage 1: Train a Global Model
Train a global AF forecasting model using the ICENTIA11K training and validation sets.

```bash
bash scripts/train_global_icentia11k.sh
```

---

#### Stage 2: Evaluate Global Model on Personalization Sets
Evaluate the pre-trained global model on patient-specific test sets from both ICENTIA11K and IRIDIA-AF datasets (without personalization).

**For ICENTIA11K:**
```bash
bash scripts/eval_global_icentia11k.sh
```

**For IRIDIA-AF (External Validation):**
```bash
bash scripts/eval_global_iridia-af.sh
```

---

#### Stage 3: Personalize the Global Model
Fine-tune the global model using patient-specific training sets to create personalized models for each patient.

**For ICENTIA11K:**
```bash
bash scripts/train_personalized_icentia11k.sh
```

**For IRIDIA-AF:**
```bash
bash scripts/train_personalized_iridia-af.sh
```

---

#### Stage 4: Visualize Feature Attribution
Analyze which ECG features are most important for AF forecasting using attribution methods on the personalized models.

```bash
bash scripts/run_attribution_icentia11k.sh
```

## Notes

- All scripts run experiments with 5 different random seeds (0-4) to ensure reproducibility and statistical robustness
- Modify the `GPU_NUM` variable in each script to specify which GPU to use
- Modify data paths in the scripts if your datasets are stored in different locations
- The IRIDIA-AF dataset is used for external validation to test generalizability