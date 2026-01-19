# TPRS: Transformer-based Prognostic Risk Stratification for Gastric Cancer

TPRS is a transformer-based survival prediction framework for whole-slide images (WSI).  
It supports an end-to-end workflow from **WSI preprocessing → patch feature extraction (UNI/CONCH) → patient-level survival modeling**.

![TPRS Framework](https://github.com/user-attachments/assets/8c49d095-a3c8-4ced-b95a-10ea481aaa5f)

---

## 🧭 Overview

This repository includes:

- **WSI preprocessing (CLAM pipeline)**: tissue segmentation, patching, optional stitching visualization
- **Interactive region selection (optional)**: polygon-based inclusion/exclusion and coordinate export
- **Patch feature extraction**: CLAM `extract_features_fp.py` with **UNI / CONCH / ResNet**
- **Survival modeling**: Transformer aggregation + combined survival loss (NLL + ranking)
- **Training & evaluation**: cross-validation + test + external test, C-index reporting, W&B logging

---

## ⚙️ Installation

### Step 1: Clone repository

    git clone https://github.com/TPRS-GC/TPRS-GC.git
    cd TPRS-GC

### Step 2: Create & activate a Python environment (recommended)

Conda:

    conda create -n tprs-gc python=3.9 -y
    conda activate tprs-gc

or venv:

    python -m venv .venv
    # Linux / macOS
    source .venv/bin/activate
    # Windows (PowerShell)
    .\.venv\Scripts\Activate.ps1

### Step 3: Install dependencies

    pip install -r requirements.txt

Notes:
- For GPU training, install a CUDA-compatible PyTorch build.
- Training uses `wandb`. Login once:

      wandb login

---

## 📁 Project Inputs / Outputs

This repo typically interacts with three types of files:

1. **Raw WSI slides**: e.g. `.svs`
2. **Patch coordinates**: `.h5` produced by CLAM patching or interactive grid tool
3. **Patch features**:
   - `.pt` (tensor only; recommended for training)
   - `.h5` (features + coords; optional)

A suggested structure:

    data/
      ├── raw_slides/                 # *.svs
      ├── patches/                    # created by Step 1 (CLAM patching)
      │   ├── patches/                # <SLIDE_ID>.h5  (coords + metadata)
      │   ├── masks/                  # <SLIDE_ID>.jpg
      │   └── stitches/               # <SLIDE_ID>.jpg (optional)
      └── features/
          ├── pt_files/               # <SLIDE_ID>.pt  (tensor, shape: N x D)
          └── h5_files/               # <SLIDE_ID>.h5  (datasets: features, coords)

---

## 🧪 Data Preprocessing Pipeline (WSI → Patches → Features)

This section reproduces the feature extraction pipeline used before survival training.

---

### Step 1: WSI Preprocessing (Segmentation + Patching)

We use CLAM’s preprocessing pipeline for **tissue segmentation**, **patch coordinate generation**, and optional **stitching**.

Run (seg + patch + stitch):

    python create_patches_fp.py \
        --source "data/raw_slides/" \
        --save_dir "data/patches/" \
        --patch_size 256 \
        --step_size 256 \
        --seg \
        --patch \
        --stitch

Outputs under `data/patches/`:
- `patches/`  → `<SLIDE_ID>.h5`
- `masks/`    → tissue mask preview images
- `stitches/` → stitched grid preview images (if enabled)

Notes:
- The script writes/updates:

      data/patches/process_list_autogen.csv

- Auto-skip is enabled by default: if `data/patches/patches/<SLIDE_ID>.h5` exists, the slide is skipped.
  Disable skipping with:

      --no_auto_skip

---

### Step 2 (Optional): Interactive Grid Selection & Coordinate Export

If you want interactive ROI selection, use:

    python wsi_ink_removal_grid_tool.py

You can draw:
- **GREEN inclusion polygons**: regions to sample patches
- **BLUE exclusion polygons**: regions to remove (artifacts, pen marks, etc.)

The tool exports Trident-compatible coordinate `.h5` and QC images.

---

### Step 3: Patch Feature Extraction (UNI / CONCH via CLAM)

We use CLAM’s `extract_features_fp.py` to extract patch-level features from coordinate `.h5`.

References:
- UNI repo: https://github.com/mahmoodlab/UNI
- UNI weights (HF): https://huggingface.co/MahmoodLab/UNI
- CONCH weights (HF): https://huggingface.co/MahmoodLab/CONCH

#### 3.1 Prepare model weights

If using UNI / CONCH, download `pytorch_model.bin` and set env vars:

    export UNI_CKPT_PATH=checkpoints/uni/pytorch_model.bin
    export CONCH_CKPT_PATH=checkpoints/conch/pytorch_model.bin

#### 3.2 Run extraction (UNI example)

Important path rule (per code):
- `extract_features_fp.py` looks for patch H5 at:

  `<data_h5_dir>/patches/<SLIDE_ID>.h5`

So `--data_h5_dir` should be the *parent directory created in Step 1* (e.g. `data/patches/`), not `data/patches/patches/`.

Example:

    python extract_features_fp.py \
        --data_h5_dir "data/patches/" \
        --data_slide_dir "data/raw_slides/" \
        --csv_path "slide_list.csv" \
        --feat_dir "data/features/" \
        --batch_size 256 \
        --slide_ext .svs \
        --model_name "uni_v1" \
        --target_patch_size 224

Outputs:

    data/features/
      ├── h5_files/<SLIDE_ID>.h5     # datasets: features, coords
      └── pt_files/<SLIDE_ID>.pt     # tensor of features only

---

## 🧾 Survival Training Data Format (CSV)

Training uses patient-level CSV files.  
`SwinPrognosisDataset` expects the following columns:

- `case_id`
- `gender` (string: `"male"` / `"female"`)
- `age_at_index` (numeric)
- `label` (discrete interval label used by survival loss; integer-like)
- `survival_months` (float)
- `censor` (0 = event, 1 = censored)
- `slide_id` (**python list string**), e.g.:

  `["TCGA-XXX.pt", "TCGA-YYY.pt"]`

Implementation note:
- `slide_id` is parsed using `ast.literal_eval()`, so it must be a valid python list literal.

---

## 🧠 Model & Loss (Implementation)

### Dataset
File: `dataset_position.py`

Modes:
- `load_mode="pt"`: loads `torch.load(<pt_dir>/<slide_id>)` and concatenates slides per patient.
- `load_mode="h5"`: loads `<h5_dir>/<slide_id>.h5` and reads:
  - `features`
  - `coords`

### Collate (padding + attention mask)
File: `model_utils.py`

`custom_collate_fn`:
- pads per-patient patch features to batch max length
- returns `mask` of shape `(B, max_patches)` with `1` valid / `0` padded

### Transformer with context (age, gender)
File: `transformer_context.py`

- patch feature projection + positional embedding
- transformer blocks with attention masking
- pooling: `cls` or `mean`
- concatenates `(age, gender)` before final MLP head

### Survival loss
File: `loss_func.py`

`CombinedSurvLoss`:
- discrete-time hazard NLL loss
- pairwise ranking loss
- combined objective: `loss_nll + lambda_rank * loss_rank`

---

## 🏋️ Training & Evaluation

Main training code:
- `train_fold.py` (Optuna CV driver)
- `train_utils_new.py` (training loop, evaluation, checkpointing)

Typical workflow:
- Split train/val/test (and load an external test set)
- Run 10-fold CV for Optuna trial evaluation
- Report C-index for val / test / external

### Run (cross-validation + Optuna)

Example:

    python train_fold.py \
        --data_dir "/path/to/pt_files_dir" \
        --train_csv "/path/to/train.csv" \
        --external_csv "/path/to/external.csv" \
        --external_data_dir "/path/to/external_pt_dir" \
        --save_dir "/path/to/save_outputs" \
        --mode cross_validation \
        --max_epochs 100 \
        --seed 42

Outputs are saved under `--save_dir` (trial/fold subfolders), including:
- best checkpoints: `best_model.pth`
- training logs + metrics CSV
- test & external predictions CSV

---

## ⚠️ Important Implementation Note (Age/Gender Order)

In `transformer_context.py`, the forward signature is:

    forward(self, x, age, gender, mask=None)

Make sure training calls the model in the same order:

    outputs = model(feature, age, gender, mask)

If your training script currently calls `(feature, gender, age, mask)`, age and gender will be swapped.

---

## 📓 Notebook Demo

We provide an example notebook:

- `examples/demo_pt_loading_and_forward.ipynb`

It demonstrates:
- creating a minimal CSV
- loading `.pt` patch features
- DataLoader + `custom_collate_fn`
- running a forward pass and computing `CombinedSurvLoss`

---

