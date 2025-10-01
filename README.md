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
‘’‘bash
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
Column Descriptions:

case_id: Unique patient identifier
slide_id: List of slide files (as string representation of Python list)
gender: Patient gender ("male"/"female")
age_at_index: Age at diagnosis
survival_months: Survival time in months
censor: Censorship indicator (0=event occurred, 1=censored)
label: Survival interval label for discrete survival analysis
Feature Files
PyTorch format: slide_id.pt containing [N_patches, feature_dim] tensor
HDF5 format: slide_id.h5 with features and coords datasets
复制
# PyTorch format loading
features = torch.load('slide_id.pt')  # Shape: [N, 1024] or [N, 2048]

# HDF5 format loading
with h5py.File('slide_id.h5', 'r') as f:
    features = f['features'][:]  # Shape: [N, feature_dim]
    coords = f['coords'][:]      # Shape: [N, 2] - patch coordinates
🏗️ Architecture
Transformer Model
复制
Transformer(
    num_classes=4,      # Survival intervals
    input_dim=1024,     # Patch feature dimension
    dim=512,           # Hidden dimension
    depth=2,           # Transformer layers
    heads=8,           # Attention heads
    pool='cls',        # Pooling strategy ('cls' or 'mean')
    dropout=0.1,       # Dropout rate
    emb_dropout=0.1    # Embedding dropout rate
)
Loss Function
复制
L_total = L_NLL + λ_rank × L_rank

where:
- L_NLL: Negative log-likelihood survival loss
- L_rank: Pairwise ranking loss
- λ_rank: Ranking weight (default: 0.5)
🔍 Interpretability Pipeline
1. Heatmap Generation (CLAM-based)
复制
python create_heatmaps.py \
    --config_file "heatmap_config.yaml" \
    --checkpoint_path "models/best_fold.pth" \
    --data_root_dir "slides/" \
    --results_dir "heatmaps/"
2. Top-K Patch Extraction
复制
# Extract top-10 high-attention patches
top_patches = extract_top_patches(
    heatmap_path="heatmaps/slide_id.h5",
    wsi_path="slides/slide_id.svs",
    top_k=10,
    patch_size=256
)
3. Cellular Feature Analysis (TIAToolbox)
复制
from tiatoolbox.models import get_pretrained_model

# Load cellular feature extractor
model = get_pretrained_model("resnet18-kather100k")

# Extract cellular features from top patches
cellular_features = model.infer_batch(top_patches)
📁 Project Structure
复制
TPRS-GC/
├── main.py                          # Main training script
├── wsi_ink_removal_grid_tool.py     # Interactive annotation tool
├── transformer_context.py           # Transformer model
├── dataset_position.py              # Dataset loader
├── create_heatmaps.py               # Heatmap generation
├── extract_cellular_features.py     # Cellular analysis
├── requirements.txt                 # Dependencies
├── environment.txt                  # Conda environment
├── environment.yml                  # Conda environment YAML
├── configs/
│   ├── heatmap_config.yaml          # Heatmap configuration
│   └── model_config.yaml            # Model parameters
└── utils/
    ├── survival_utils.py            # Survival analysis utilities
    └── visualization.py             # Plotting functions
⚙️ Configuration
Training Parameters
复制
# model_config.yaml
model:
  input_dim: 1024
  dim: 512
  depth: 2
  heads: 8
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 1e-5
  max_epochs: 100
  ranking_loss_weight: 0.5
Heatmap Generation
复制
# heatmap_config.yaml
heatmap:
  patch_size: 256
  overlap: 0.5
  top_k: 10
  cmap: 'coolwarm'
  alpha: 0.6
🔧 Advanced Features
Hyperparameter Optimization
复制
# Enable Optuna optimization
python main.py \
    --use_optuna True \
    --n_trials 100 \
    --optimization_direction "maximize"

# Training without hyperparameter optimization
python main.py \
    --use_optuna False \
    --learning_rate 1e-5 \
    --batch_size 16
Multi-GPU Training
复制
# Automatic DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
Custom Dataset Integration
复制
from dataset_position import SwinPrognosisDataset

# Custom dataset with H5 format
dataset = SwinPrognosisDataset(
    df='your_data.csv',
    pt_dir=None,
    h5_dir='/path/to/h5/files',
    load_mode='h5'
)
Model Customization
复制
from transformer_context import Transformer

# Custom model configuration
model_params = {
    'num_classes': 4,
    'input_dim': 2048,      # Adjust based on your features
    'dim': 1024,            # Larger model capacity
    'depth': 4,             # Deeper architecture
    'heads': 16,            # More attention heads
    'mlp_dim': 2048,
    'pool': 'mean',         # Alternative pooling
    'dropout': 0.2
}
📈 Results Visualization
Survival Curves
复制
from utils.visualization import plot_survival_curves

plot_survival_curves(
    predictions=model_outputs,
    ground_truth=survival_data,
    save_path="results/survival_curves.png"
)
Attention Heatmaps
复制
# Generate attention visualization
attention_maps = model.get_attention_weights(features)
visualize_attention(attention_maps, patch_coords)
🛠️ Troubleshooting
Common Issues
CUDA Memory Error

复制
# Reduce batch size
--batch_size 8

# Enable gradient checkpointing
--gradient_checkpointing True

# Monitor GPU memory
gpustat -i 1
OpenSlide Installation

复制
# Ubuntu/Debian
sudo apt-get install openslide-tools python3-openslide

# macOS
brew install openslide

# Test installation
python -c "import openslide; print('OpenSlide installed successfully')"
Feature Extraction Issues

复制
# Verify feature file format
import torch
features = torch.load('slide.pt')
print(f"Shape: {features.shape}")  # Should be [N, feature_dim]
Package Conflicts

复制
# Clean conda cache
conda clean --all

# Update conda
conda update conda

# Reinstall problematic packages
conda install --force-reinstall package_name
Performance Optimization
Multi-GPU Training

复制
# Automatic DataParallel usage
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
Memory Optimization

复制
# Enable memory efficient attention
torch.backends.cuda.enable_flash_sdp(True)

# Use gradient checkpointing
model.gradient_checkpointing = True
Data Loading

复制
# Optimize DataLoader
DataLoader(
    dataset, 
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
