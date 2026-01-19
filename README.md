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

Step 2: Grid Selection & Patch Coordinate Export (Interactive)
This step provides an interactive workflow to select regions on a WSI thumbnail and generate a grid of patch coordinates on the level-0 coordinate system. You can draw GREEN inclusion polygons to indicate where to sample patches, and optionally draw BLUE exclusion polygons to remove unwanted areas (e.g., artifacts) from the sampled grid. The tool exports Trident-compatible patch coordinates and QC visualizations.

Run
复制
python wsi_ink_removal_grid_tool.py
After launching, choose one of the modes:

[1] Batch Processing: process all WSI files in a directory (recommended)
[2] Single File Processing: process one WSI file
[3] View Results: browse and open saved QC images, extract coordinates
[4] Exit
Key Features
Interactive polygon annotation
GREEN (Include) polygons define regions where patches will be generated
BLUE (Exclude) polygons carve out areas to be excluded from the GREEN regions
Grid-based patch coordinate generation (level-0)
Iterates a regular grid at rect_size stride on the original slide
A patch is kept if its center point falls inside GREEN and not inside BLUE
Automatic patch/grid sizing by objective power
If slide is 40x → rect_size = 1024
If slide is 20x → rect_size = 512
Batch mode with progress tracking
Writes processing_progress.json in the output directory to track completed and skipped
QC outputs + exports
Thumbnail image
Annotated preview image (polygons + generated grid)
Trident-compatible .h5 coordinates + metadata JSON
Also saves a legacy coordinate file for convenience
Interactive Controls (in the OpenCV window)
Left click: add a vertex to the current polygon
Right click: close and finalize the polygon (requires ≥ 3 points)
g: switch to GREEN mode (Inclusion)
b: switch to BLUE mode (Exclusion)
SPACE: generate / regenerate the grid (requires at least one completed GREEN polygon)
z: undo last point (current mode)
r: reset the current (in-progress) polygon
c: clear all polygons and grids
q: save results and move to the next file (batch mode)
s: skip this file (do not save)
ESC: exit the entire program
Output Structure
Each slide creates its own subfolder under the chosen output directory:

复制
output_dir/
  ├── processing_progress.json
  ├── SLIDE_NAME/
  │   ├── SLIDE_NAME_thumbnail.png
  │   ├── SLIDE_NAME_annotated.png
  │   ├── SLIDE_NAME_patches.h5
  │   ├── SLIDE_NAME_coordinates_legacy.h5
  │   └── SLIDE_NAME_metadata.json
Trident-Compatible H5 Format
File: *_patches.h5

Dataset: coords with shape (N, 2) and dtype int64

Each row is the top-left patch coordinate on level-0: [[x, y], ...]
Important attributes written to coords.attrs include (among others):

patch_size, patch_size_level0
target_magnification, level0_magnification
level0_width, level0_height
overlap (currently 0)
name, savetodir
total_grids, green_polygons, blue_polygons
Magnification logic (as implemented):

40x slides: cut 1024 at level-0 (patch_size_level0=1024) and set target_magnification=20, patch_size=512
20x slides: cut 512 at level-0 (patch_size_level0=512) and set target_magnification=20, patch_size=512


**Step 3: Install dependencies**
pip install -r requirements.txt

## 📊 Data Preprocessing Pipeline

### Step 1: WSI Preprocessing


For WSI segmentation and patching, we employed the preprocessing pipeline from the [CLAM repository](https://github.com/mahmoodlab/CLAM).

To reproduce our patching step, use the `create_patches_fp.py` script with the following arguments:

```bash
python create_patches_fp.py \
    --source "raw_slides/" \
    --save_dir "patches/" \
    --patch_size 256 \
    --step_size 256 \
    --seg \
    --patch \
    --stitch
```



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
