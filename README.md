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

**Step 2: Create environment**
conda create --name TPRS python=3.9
conda activate TPRS

**Step 3: Install dependencies**
pip install -r requirements.txt

## 📊 Data Preprocessing Pipeline

### Step 1: WSI Preprocessing
For WSI segmentation and patching, we employed the preprocessing pipeline from the [CLAM repository](https://github.com/mahmoodlab/CLAM).

To reproduce our patching step, use the `create_patches_fp.py` script with the following arguments:

```bash
python create_patches_fp.py --source "raw_slides/" --save_dir "patches/" --patch_size 256 --step_size 256 --seg --patch --stitch


### Step 2: Ink Removal (Optional - for marker-annotated slides)
**Semi-automated ink removal using interactive tool:**
python wsi_ink_removal_grid_tool.py \
    --slide_dir "raw_slides/" \
    --output_dir "cleaned_slides/" \
    --interactive_mode True

**Features:**
- Interactive polygon-based annotation for ink regions
- Grid-based tissue region extraction
- Semi-automated marker detection and removal
- Quality control visualization

### Step 3: Feature Extraction (UNI Network)
**Extract features using UNI foundation model:**
python extract_features_fp.py \
    --data_h5_dir "patches/" \
    --data_slide_dir "raw_slides/" \
    --csv_path "slide_list.csv" \
    --feat_dir "features/" \
    --batch_size 512 \
    --slide_ext .svs \
    --model_name "UNI"

**UNI Model Configuration:**
- **Architecture**: Vision Transformer (ViT-Large)
- **Input Size**: 224×224 patches
- **Feature Dimension**: 1024
- **Pre-training**: 100M+ histopathology images
- **Performance**: State-of-the-art on multiple histopathology tasks
