 # TPRS: Transformer-based Prognostic Risk Stratification for Gastric Cancer

A streamlined deep learning framework for survival prediction from whole-slide images using transformer architecture.

## 🎯 Overview

TPRS is a comprehensive pathological image analysis pipeline that combines:

- **Interactive WSI Annotation**: Semi-automated region selection with polygon-based annotation
- **Transformer Architecture**: Multi-head attention for histopathological patch aggregation  
- **Survival Analysis**: Combined negative log-likelihood and ranking loss
- **Interpretability**: CLAM-based heatmap generation with cellular feature extraction

![TPRS Framework](https://github.com/user-attachments/assets/8c49d095-a3c8-4ced-b95a-10ea481aaa5f)

## 🚀 Quick Start

### Installation

**Step 1: Clone repository**
git clone https://github.com/TPRS-GC/TPRS-GC.git
cd TPRS-GC


## 📊 Data Preprocessing Pipeline

## Step 1: WSI Preprocessing (Segmentation + Patching)

We use the preprocessing pipeline from the CLAM repository for **tissue segmentation**, **patch coordinate generation**, and optional **stitching visualization**.

This step is implemented by `create_patches_fp.py` (see source code). It will create three subfolders under `--save_dir`:
- `patches/`  → patch coordinate `.h5` files (one per slide)
- `masks/`    → tissue segmentation mask previews (`.jpg`)
- `stitches/` → stitched patch-grid previews (`.jpg`, if enabled)

### Run

Example (seg + patch + stitch):

    python create_patches_fp.py \
        --source "raw_slides/" \
        --save_dir "patches/" \
        --patch_size 256 \
        --step_size 256 \
        --seg \
        --patch \
        --stitch

### Notes (based on the script behavior)
- The script writes/updates:

      patches/process_list_autogen.csv

  during processing (used for tracking slide-level parameters/status).
- Auto-skip is enabled by default: if `patches/patches/<SLIDE_ID>.h5` already exists, that slide is skipped.
  To disable skipping, use:

      --no_auto_skip
- `--patch_level` defaults to `0` (level-0). You can change it if needed.

---

## Step 2: Grid Selection & Patch Coordinate Export (Interactive)

This step provides an interactive workflow to select regions on a WSI thumbnail and generate a **grid of patch coordinates** on the **level-0 coordinate system**. Draw **GREEN inclusion polygons** to indicate where to sample patches, and optionally draw **BLUE exclusion polygons** to remove unwanted areas (e.g., artifacts) from the sampled grid. The tool exports **Trident-compatible** patch coordinates and QC visualizations.

### Run

    python wsi_ink_removal_grid_tool.py

After launching, choose one of the modes:
- **[1] Batch Processing**: process all WSI files in a directory (recommended)
- **[2] Single File Processing**: process one WSI file
- **[3] View Results**: browse and open saved QC images, extract coordinates
- **[4] Exit**

### Interactive Controls (OpenCV window)

- **Left click**: add a vertex to the current polygon
- **Right click**: close and finalize the polygon (requires ≥ 3 points)
- **`g`**: switch to **GREEN** mode (Inclusion)
- **`b`**: switch to **BLUE** mode (Exclusion)
- **`SPACE`**: generate / regenerate the grid (requires at least one completed GREEN polygon)
- **`z`**: undo last point (current mode)
- **`r`**: reset the current (in-progress) polygon
- **`c`**: clear all polygons and grids
- **`q`**: save results and move to the next file (batch mode)
- **`s`**: skip this file (do not save)
- **`ESC`**: exit the entire program

---

## Step 3: Feature Extraction (UNI via CLAM `extract_features_fp.py`)

This step extracts **patch-level deep features** using the **MahmoodLab UNI** encoder, through CLAM’s feature extraction script `extract_features_fp.py`.

References:
- UNI repo: https://github.com/mahmoodlab/UNI
- UNI weights (HF): https://huggingface.co/MahmoodLab/UNI

---

### 3.1 Download UNI Weights (Required)

`extract_features_fp.py` uses `--model_name uni_v1`.  
You must download the UNI checkpoint (`pytorch_model.bin`) from Hugging Face and set:

    export UNI_CKPT_PATH=checkpoints/uni/pytorch_model.bin

---

### 3.2 Run Feature Extraction (UNI)

Important: the script constructs the patch H5 path as:

    <data_h5_dir>/patches/<SLIDE_ID>.h5

So `--data_h5_dir` should be the **same `--save_dir` used in Step 1** (the parent folder that contains the `patches/` subfolder), not the `patches/` subfolder itself.

Example:

    python extract_features_fp.py \
        --data_h5_dir "patches/" \
        --data_slide_dir "raw_slides/" \
        --csv_path "slide_list.csv" \
        --feat_dir "features/" \
        --batch_size 256 \
        --slide_ext .svs \
        --model_name "uni_v1" \
        --target_patch_size 224

What it produces under `--feat_dir` (created automatically):

    features/
      ├── h5_files/
      │   └── <SLIDE_ID>.h5        # datasets: 'features', 'coords'
      └── pt_files/
          └── <SLIDE_ID>.pt        # torch tensor of features only

Auto-skip behavior:
- By default, if `features/pt_files/<SLIDE_ID>.pt` already exists, that slide is skipped.
- To recompute everything, add:

      --no_auto_skip

---

### 3.3 UNI Output + Model Details

- **Encoder**: `uni_v1` (loaded by `get_encoder()`)
- **Target patch size**: `--target_patch_size` (default `224`)
- **Feature dimension**: **1024-d**
- Saved feature file contents:
  - `h5_files/<SLIDE_ID>.h5` contains:
    - `features`: shape `(N, 1024)`
    - `coords`: shape `(N, 2)` (int32), matching the patch coordinates

### 3.4 GPU / Batch Size Notes

UNI (ViT-L) is GPU-memory intensive. If you encounter CUDA OOM, reduce `--batch_size` (e.g., 128 / 64 / 32).
