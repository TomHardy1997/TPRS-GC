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
- Python 3.8+
- CUDA-capable GPU (recommended)
- OpenSlide library for WSI processing

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/WSI-Transformer.git
cd WSI-Transformer

# Create conda environment
conda create -n wsi-transformer python=3.8
conda activate wsi-transformer

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install OpenSlide (Linux/macOS)
sudo apt-get install openslide-tools  # Ubuntu/Debian
# or
brew install openslide  # macOS

# For Windows, download from: https://openslide.org/download/
