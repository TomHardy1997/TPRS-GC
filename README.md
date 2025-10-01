# TPRS: Transformer-based Prognostic Risk Stratification for Gastric Cancer

A streamlined deep learning framework for survival prediction from whole-slide images using transformer architecture.

## 🎯 Overview

TPRS is a comprehensive pathological image analysis pipeline that combines:

- **Interactive WSI Annotation**: Semi-automated region selection with polygon-based annotation
- **Transformer Architecture**: Multi-head attention for histopathological patch aggregation  
- **Survival Analysis**: Combined negative log-likelihood and ranking loss
- **Interpretability**: CLAM-based heatmap generation with cellular feature extraction

<img width="4000" height="945" alt="TPRS Framework" src="https://github.com/user-attachments/assets/8c49d095-a3c8-4ced-b95a-10ea481aaa5f" />

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/TPRS-GC/TPRS-GC.git
cd TPRS-GC

# Create environment
conda create --name TPRS python=3.9
conda activate TPRS

# Install dependencies
pip install -r requirements.txt
Basic Usage
复制
# 1. Interactive WSI annotation
python wsi_ink_removal_grid_tool.py

# 2. Train model with cross-validation
python main.py \
    --train_csv "data/train.csv" \
    --data_dir "features/" \
    --save_dir "results/" \
    --mode "cross_validation"

# 3. Generate interpretability heatmaps
python generate_heatmaps.py \
    --model_path "results/best_model.pth" \
    --wsi_dir "slides/" \
    --output_dir "heatmaps/"
📊 Data Format
CSV Structure
复制
case_id,slide_id,gender,age_at_index,survival_months,censor,label
TCGA-XX-XXXX,"['slide1.pt']",male,65,24.5,0,2
TCGA-YY-YYYY,"['slide2.pt']",female,58,18.2,1,1
Feature Files
PyTorch format: slide_id.pt containing [N_patches, feature_dim] tensor
HDF5 format: slide_id.h5 with features and coords datasets
🏗️ Architecture
Transformer Model
复制
Transformer(
    num_classes=4,      # Survival intervals
    input_dim=1024,     # Patch feature dimension
    dim=512,           # Hidden dimension
    depth=2,           # Transformer layers
    heads=8,           # Attention heads
    pool='cls'         # Pooling strategy
)
Loss Function
复制
L_total = L_NLL + λ_rank × L_rank
L_NLL: Negative log-likelihood survival loss
L_rank: Pairwise ranking loss
λ_rank: Ranking weight (default: 0.5)
🔍 Interpretability Pipeline
1. Heatmap Generation (CLAM-based)
