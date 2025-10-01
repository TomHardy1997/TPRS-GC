# WSI-Transformer: Whole Slide Image Analysis for Survival Prediction

A comprehensive framework for survival analysis on Whole Slide Images (WSI) using Transformer architecture with interactive region selection and automated hyperparameter optimization.

## Overview

This project implements a deep learning pipeline for pathological image analysis, specifically designed for survival prediction tasks. The framework combines:

- **Interactive WSI Grid Selection**: Semi-automated tool for region-of-interest selection with polygon-based annotation
- **Transformer-based Architecture**: Advanced attention mechanism for feature aggregation from histopathological patches
- **Combined Survival Loss**: Integration of negative log-likelihood and ranking loss for robust survival analysis
- **Automated Hyperparameter Optimization**: Optuna-based parameter tuning with cross-validation

## Key Features

### 🔍 Interactive WSI Processing
- **Polygon-based Region Selection**: Interactive tool for manual annotation of tissue regions
- **Automated Grid Generation**: Systematic patch extraction within selected regions
- **Multi-resolution Support**: Adaptive grid sizing based on objective magnification (20x/40x)
- **Ink Artifact Removal**: Semi-automated preprocessing for marker pen removal

### 🧠 Advanced Deep Learning Architecture
- **Transformer Encoder**: Multi-head attention mechanism for patch-level feature aggregation
- **Positional Encoding**: Learned positional embeddings for spatial relationship modeling
- **Clinical Data Integration**: Incorporation of age and gender as contextual features
- **Flexible Pooling**: Support for both CLS token and mean pooling strategies

### 📊 Robust Survival Analysis
- **Combined Loss Function**: NLL survival loss + pairwise ranking loss
- **C-index Optimization**: Concordance index as primary evaluation metric
- **Censorship Handling**: Proper treatment of censored survival data
- **Cross-validation**: 10-fold stratified cross-validation for model validation

### ⚙️ Automated Optimization
- **Hyperparameter Tuning**: Optuna-based Bayesian optimization
- **Multi-objective Optimization**: Balance between model complexity and performance
- **Experiment Tracking**: Integration with Weights & Biases (wandb)
- **Reproducible Results**: Comprehensive seed setting and logging

## Installation

### Prerequisites
- Python 3.9
- CUDA 11.3+ compatible GPU (recommended)
- OpenSlide library for WSI processing
- Conda package manager

### Environment Setup

#### Option 1: Using the provided environment file (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/WSI-Transformer.git
cd WSI-Transformer

# Create environment from the provided file
conda create --name clam --file environment.txt

# Activate the environment
conda activate clam
Option 2: Manual installation
复制
# Create new conda environment
conda create -n wsi-transformer python=3.9
conda activate wsi-transformer

# Install CUDA toolkit
conda install cudatoolkit=11.3.1 -c conda-forge

# Install PyTorch with CUDA support
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==0.10.0

# Install core dependencies
pip install -r requirements.txt
Requirements.txt
复制
# Core ML/DL frameworks
torch==2.5.1
torchvision==0.20.1
torchaudio==0.10.0
numpy==1.26.4
einops==0.8.0

# Data handling and analysis
pandas==2.2.3
h5py==3.10.0
scikit-learn==1.4.2
scikit-survival==0.22.2
lifelines==0.30.0
scipy==1.13.1

# Computer Vision & WSI Processing
opencv-python-headless==4.11.0.86
Pillow==11.0.0
histolab==0.7.0
openslide-python==1.4.1
tiatoolbox==1.6.0
staintools==2.1.2
scikit-image==0.24.0
imageio==2.35.1
tifffile==2024.8.30

# Medical imaging
pydicom==2.4.4
simpleitk==2.4.0
dicomweb-client==0.59.3
wsidicom==0.20.6

# Hyperparameter optimization & Experiment tracking
optuna==4.0.0
wandb

# Visualization
matplotlib==3.9.3
seaborn==0.13.2
bokeh==3.4.3

# Jupyter ecosystem
jupyter==1.1.1
jupyterlab==4.3.2
ipykernel==6.29.5
notebook==7.4.5

# Utilities
tqdm==4.67.1
pyyaml==6.0.2
requests==2.32.3
flask==3.1.0
flask-cors==5.0.0

# Image augmentation
albumentations==1.4.21

# Additional scientific computing
numba==0.60.0
sympy==1.13.1
networkx==3.2.1
zarr==2.18.2
numcodecs==0.12.1

# Dimensionality reduction
umap-learn==0.5.7
pynndescent==0.5.13

# Optimization
osqp==0.6.5
ecos==2.0.13

# GPU monitoring
gpustat==1.1.1
nvidia-ml-py==12.535.133

# Development tools
ipdb==0.13.9
System Dependencies (Linux)
复制
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    openslide-tools \
    libopencv-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev

# CentOS/RHEL
sudo yum install -y \
    openslide-devel \
    opencv-devel \
    hdf5-devel \
    libjpeg-turbo-devel \
    libpng-devel \
    libtiff-devel
Quick Start
1. Environment Verification
复制
# Activate environment
conda activate clam

# Verify key installations
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import openslide; print('OpenSlide: OK')"
python -c "import optuna; print(f'Optuna: {optuna.__version__}')"
2. Interactive WSI Grid Selection
复制
# Configure your WSI file path in the script
python wsi_ink_removal_grid_tool.py

# Interactive workflow:
# 1. Left click: Add polygon vertices
# 2. Right click: Complete polygon and generate grid
# 3. Press any key: Save results and exit
3. Training with Hyperparameter Optimization
复制
python main.py \
    --train_csv "path/to/training_data.csv" \
    --external_csv "path/to/external_test.csv" \
    --data_dir "/path/to/feature/files" \
    --external_data_dir "/path/to/external/features" \
    --save_dir "./results" \
    --mode "cross_validation" \
    --max_epochs 100 \
    --seed 42
4. Training without Hyperparameter Optimization
复制
python main.py \
    --train_csv "path/to/training_data.csv" \
    --external_csv "path/to/external_test.csv" \
    --data_dir "/path/to/feature/files" \
    --external_data_dir "/path/to/external/features" \
    --save_dir "./results" \
    --mode "cross_validation" \
    --use_optuna False \
    --learning_rate 1e-5 \
    --batch_size 16
Data Format
CSV File Structure
Your training CSV should contain the following columns:

复制
case_id,slide_id,gender,age_at_index,survival_months,censor,label
TCGA-XX-XXXX,"['slide1.pt', 'slide2.pt']",male,65,24.5,0,2
TCGA-YY-YYYY,"['slide3.pt']",female,58,18.2,1,1
Column Descriptions:

case_id: Unique patient identifier
slide_id: List of slide files (as string representation of Python list)
gender: Patient gender ("male"/"female")
age_at_index: Age at diagnosis
survival_months: Survival time in months
censor: Censorship indicator (0=event occurred, 1=censored)
label: Survival interval label for discrete survival analysis
Feature File Formats
PyTorch Format (.pt)
复制
# Each .pt file contains a tensor of shape [N_patches, feature_dim]
features = torch.load('slide_id.pt')  # Shape: [N, 1024] or [N, 2048]
HDF5 Format (.h5)
复制
# Each .h5 file contains:
with h5py.File('slide_id.h5', 'r') as f:
    features = f['features'][:]  # Shape: [N, feature_dim]
    coords = f['coords'][:]      # Shape: [N, 2] - patch coordinates
Model Architecture
Transformer Components
复制
# Core architecture
Transformer(
    num_classes=4,          # Number of survival intervals
    input_dim=1024,         # Feature dimension from patch encoder
    dim=512,                # Transformer hidden dimension
    depth=2,                # Number of transformer layers
    heads=8,                # Number of attention heads
    mlp_dim=512,           # MLP hidden dimension
    pool='cls',            # Pooling strategy ('cls' or 'mean')
    dropout=0.1,           # Dropout rate
    emb_dropout=0.1        # Embedding dropout rate
)
Loss Function
The model uses a combined loss function:

复制
L_total = L_NLL + λ_rank × L_rank

where:
- L_NLL: Negative log-likelihood survival loss
- L_rank: Pairwise ranking loss
- λ_rank: Ranking loss weight (default: 0.5)
Advanced Usage
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
Environment Management
Export Current Environment
复制
# Export complete environment
conda env export > environment.yml

# Export package list
conda list --export > environment.txt

# Export pip requirements only
pip freeze > requirements_pip.txt
Recreate Environment
复制
# From conda environment file
conda env create -f environment.yml

# From package list
conda create --name new-env --file environment.txt

# From pip requirements
pip install -r requirements_pip.txt
Troubleshooting
Common Issues
CUDA Compatibility

复制
# Check CUDA version
nvidia-smi
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
OpenSlide Installation

复制
# Ubuntu/Debian
sudo apt-get install openslide-tools python3-openslide

# Test installation
python -c "import openslide; print('OpenSlide installed successfully')"
Memory Issues

复制
# Monitor GPU memory
gpustat -i 1

# Reduce batch size if needed
--batch_size 8
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
Hardware Requirements
Minimum Requirements
CPU: 8 cores, 16GB RAM
GPU: 8GB VRAM (GTX 1080 or better)
Storage: 100GB free space for datasets
Recommended Requirements
CPU: 16+ cores, 32GB+ RAM
GPU: 16GB+ VRAM (RTX 3080/4080 or better)
Storage: 500GB+ SSD for fast I/O
Citation
If you use this code in your research, please cite:

复制
@article{your_paper_2024,
  title={WSI-Transformer: Deep Learning Framework for Survival Analysis on Whole Slide Images},
  author={Your Name and Co-authors},
  journal={Your Journal},
  year={2024},
  volume={XX},
  pages={XXX-XXX}
}
Acknowledgments
CLAM for inspiration on WSI analysis frameworks
Histolab for WSI processing utilities
TIAToolbox for advanced WSI analysis tools
Optuna for hyperparameter optimization
scikit-survival for survival analysis metrics
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contributing
We welcome contributions! Please see CONTRIBUTING.md for guidelines.

Contact
For questions and support:

Email: your.email@institution.edu
Issues: GitHub Issues
Discussions: GitHub Discussions
Note: This framework is designed for research purposes. For clinical applications, please ensure proper validation and regulatory compliance.
