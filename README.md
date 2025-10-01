# TPRS: Transformer-based biomarker from whole-slide images for prognostication in gastric cancer

A Comprehensive Framework for Survival Analysis on Whole Slide Images (WSI) Using Transformer Architecture

## Overview

This project implements a deep learning pipeline for pathological image analysis, specifically designed for survival prediction tasks. The framework combines:

- **Interactive WSI Grid Selection**: Semi-automated tool for region-of-interest selection with polygon-based annotation targeting marker pen annotated areas
- **Transformer-based Architecture**: Advanced attention mechanism for feature aggregation from histopathological patches
- **Multi-Patient Batch Processing**: Single batch can handle multiple patients with variable numbers of slides per patient
- **Combined Survival Loss**: Integration of negative log-likelihood and ranking loss for robust survival analysis
- **Automated Hyperparameter Optimization**: Optuna-based parameter tuning with cross-validation

## Key Features

### 🔍 Interactive WSI Processing
- **Polygon-based Region Selection**: Interactive tool for manual annotation of tissue regions
- **Automated Grid Generation**: Systematic patch extraction within selected regions
- **Multi-resolution Support**: Adaptive grid sizing based on objective magnification (20x/40x)
- **Ink Artifact Removal**: Semi-automated preprocessing for marker pen removal
<img width="416" height="99" alt="image" src="https://github.com/user-attachments/assets/f14d424d-6295-48b4-ab85-e9b98139df9d" />

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

    # Clone the repository
    git clone https://github.com/yourusername/WSI-Transformer.git
    cd WSI-Transformer

    # Create environment from the provided file
    conda create --name clam --file environment.txt

    # Activate the environment
    conda activate clam

#### Option 2: Manual installation

    # Create new conda environment
    conda create -n wsi-transformer python=3.9
    conda activate wsi-transformer

    # Install CUDA toolkit
    conda install cudatoolkit=11.3.1 -c conda-forge

    # Install PyTorch with CUDA support
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==0.10.0

    # Install core dependencies
    pip install -r requirements.txt

### System Dependencies (Linux)

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

## Quick Start

### 1. Environment Verification

    # Activate environment
    conda activate clam

    # Verify key installations
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
    python -c "import openslide; print('OpenSlide: OK')"
    python -c "import optuna; print(f'Optuna: {optuna.__version__}')"

### 2. Interactive WSI Grid Selection

    # Configure your WSI file path in the script
    python wsi_ink_removal_grid_tool.py

    # Interactive workflow:
    # 1. Left click: Add polygon vertices
    # 2. Right click: Complete polygon and generate grid
    # 3. Press any key: Save results and exit

### 3. Training with Hyperparameter Optimization

    python main.py \
        --train_csv "path/to/training_data.csv" \
        --external_csv "path/to/external_test.csv" \
        --data_dir "/path/to/feature/files" \
        --external_data_dir "/path/to/external/features" \
        --save_dir "./results" \
        --mode "cross_validation" \
        --max_epochs 100 \
        --seed 42

### 4. Training without Hyperparameter Optimization

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

## Data Format

### CSV File Structure

Your training CSV should contain the following columns:

    case_id,slide_id,gender,age_at_index,survival_months,censor,label
    TCGA-XX-XXXX,"['slide1.pt', 'slide2.pt']",male,65,24.5,0,2
    TCGA-YY-YYYY,"['slide3.pt']",female,58,18.2,1,1

**Column Descriptions:**
- `case_id`: Unique patient identifier
- `slide_id`: List of slide files (as string representation of Python list)
- `gender`: Patient gender ("male"/"female")
- `age_at_index`: Age at diagnosis
- `survival_months`: Survival time in months
- `censor`: Censorship indicator (0=event occurred, 1=censored)
- `label`: Survival interval label for discrete survival analysis

### Feature File Formats

#### PyTorch Format (.pt)

    # Each .pt file contains a tensor of shape [N_patches, feature_dim]
    features = torch.load('slide_id.pt')  # Shape: [N, 1024] or [N, 2048]

#### HDF5 Format (.h5)

    # Each .h5 file contains:
    with h5py.File('slide_id.h5', 'r') as f:
        features = f['features'][:]  # Shape: [N, feature_dim]
        coords = f['coords'][:]      # Shape: [N, 2] - patch coordinates

## Model Architecture

### Transformer Components

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

### Loss Function

The model uses a combined loss function:

    L_total = L_NLL + λ_rank × L_rank

    where:
    - L_NLL: Negative log-likelihood survival loss
    - L_rank: Pairwise ranking loss
    - λ_rank: Ranking loss weight (default: 0.5)

## Advanced Usage

### Custom Dataset Integration

    from dataset_position import SwinPrognosisDataset

    # Custom dataset with H5 format
    dataset = SwinPrognosisDataset(
        df='your_data.csv',
        pt_dir=None,
        h5_dir='/path/to/h5/files',
        load_mode='h5'
    )

### Model Customization

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

## Environment Management

### Export Current Environment

    # Export complete environment
    conda env export > environment.yml

    # Export package list
    conda list --export > environment.txt

    # Export pip requirements only
    pip freeze > requirements_pip.txt

### Recreate Environment

    # From conda environment file
    conda env create -f environment.yml

    # From package list
    conda create --name new-env --file environment.txt

    # From pip requirements
    pip install -r requirements_pip.txt

## Troubleshooting

### Common Issues

1. **CUDA Compatibility**

       # Check CUDA version
       nvidia-smi
       nvcc --version
       
       # Verify PyTorch CUDA
       python -c "import torch; print(torch.cuda.is_available())"

2. **OpenSlide Installation**

       # Ubuntu/Debian
       sudo apt-get install openslide-tools python3-openslide
       
       # Test installation
       python -c "import openslide; print('OpenSlide installed successfully')"

3. **Memory Issues**

       # Monitor GPU memory
       gpustat -i 1
       
       # Reduce batch size if needed
       --batch_size 8

4. **Package Conflicts**

       # Clean conda cache
       conda clean --all
       
       # Update conda
       conda update conda
       
       # Reinstall problematic packages
       conda install --force-reinstall package_name

### Performance Optimization

1. **Multi-GPU Training**

       # Automatic DataParallel usage
       if torch.cuda.device_count() > 1:
           model = nn.DataParallel(model)

2. **Memory Optimization**

       # Enable memory efficient attention
       torch.backends.cuda.enable_flash_sdp(True)
       
       # Use gradient checkpointing
       model.gradient_checkpointing = True

3. **Data Loading**

       # Optimize DataLoader
       DataLoader(
           dataset, 
           batch_size=16,
           num_workers=8,
           pin_memory=True,
           persistent_workers=True
       )

## Hardware Requirements

### Minimum Requirements
- **CPU**: 8 cores, 16GB RAM
- **GPU**: 8GB VRAM (GTX 1080 or better)
- **Storage**: 100GB free space for datasets

### Recommended Requirements
- **CPU**: 16+ cores, 32GB+ RAM
- **GPU**: 16GB+ VRAM (RTX 3080/4080 or better)
- **Storage**: 500GB+ SSD for fast I/O

## Citation

If you use this code in your research, please cite:

    @article{your_paper_2024,
      title={WSI-Transformer: Deep Learning Framework for Survival Analysis on Whole Slide Images},
      author={Your Name and Co-authors},
      journal={Your Journal},
      year={2024},
      volume={XX},
      pages={XXX-XXX}
    }

## Acknowledgments

- [CLAM](https://github.com/mahmoodlab/CLAM) for inspiration on WSI analysis frameworks
- [Histolab](https://github.com/histolab/histolab) for WSI processing utilities
- [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox) for advanced WSI analysis tools
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [scikit-survival](https://scikit-survival.readthedocs.io/) for survival analysis metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

For questions and support:
- Email: your.email@institution.edu
- Issues: [GitHub Issues](https://github.com/yourusername/WSI-Transformer/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/WSI-Transformer/discussions)

---

**Note**: This framework is designed for research purposes. For clinical applications, please ensure proper validation and regulatory compliance.

---

## Configuration Files

### requirements.txt

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

### environment.txt

    # This file may be used to create an environment using:
    # $ conda create --name <env> --file <this file>
    # platform: linux-64
    _libgcc_mutex=0.1=main
    _openmp_mutex=5.1=1_gnu
    blas=1.0=mkl
    ca-certificates=2024.7.2=h06a4308_0
    certifi=2024.8.30=py39h06a4308_0
    cudatoolkit=11.3.1=h2bc3f7f_2
    intel-openmp=2023.1.0=hdb19cb5_46306
    ld_impl_linux-x86_64=2.38=h1181459_1
    libffi=3.4.4=h6a678d5_1
    libgcc-ng=11.2.0=h1234567_1
    libgomp=11.2.0=h1234567_1
    libstdcxx-ng=11.2.0=h1234567_1
    mkl=2023.1.0=h213fc3f_46344
    mkl-service=2.4.0=py39h5eee18b_1
    mkl_fft=1.3.8=py39h5eee18b_0
    mkl_random=1.2.4=py39hdb19cb5_0
    ncurses=6.4=h6a678d5_0
    numpy=1.26.4=py39h5f9d8c6_0
    numpy-base=1.26.4=py39hb5e798b_0
    openssl=3.0.15=h5eee18b_0
    pip=24.2=py39h06a4308_0
    python=3.9.20=h5148396_1
    readline=8.2=h5eee18b_0
    setuptools=75.1.0=py39h06a4308_0
    sqlite=3.45.3=h5eee18b_0
    tk=8.6.14=h39e8969_0
    tzdata=2024a=h04d1e81_0
    wheel=0.44.0=py39h06a4308_0
    xz=5.4.6=h5eee18b_1
    zlib=1.2.13=h5eee18b_1

### environment.yml

    name: clam
    channels:
      - pytorch
      - conda-forge
      - defaults
    dependencies:
      - python=3.9
      - cudatoolkit=11.3.1
      - pytorch=2.5.1
      - torchvision=0.20.1
      - torchaudio=0.10.0
      - numpy=1.26.4
      - scipy=1.13.1
      - scikit-learn=1.4.2
      - pandas=2.2.3
      - matplotlib=3.9.3
      - seaborn=0.13.2
      - jupyter=1.1.1
      - jupyterlab=4.3.2
      - h5py=3.10.0
      - opencv=4.11.0
      - pillow=11.0.0
      - tqdm=4.67.1
      - pyyaml=6.0.2
      - requests=2.32.3
      - pip=24.2
      - pip:
        - openslide-python==1.4.1
        - scikit-survival==0.22.2
        - lifelines==0.30.0
        - optuna==4.0.0
        - wandb
        - einops==0.8.0
        - histolab==0.7.0
        - tiatoolbox==1.6.0
        - staintools==2.1.2
        - scikit-image==0.24.0
        - imageio==2.35.1
        - tifffile==2024.8.30
        - pydicom==2.4.4
        - simpleitk==2.4.0
        - dicomweb-client==0.59.3
        - wsidicom==0.20.6
        - bokeh==3.4.3
        - ipykernel==6.29.5
        - notebook==7.4.5
        - flask==3.1.0
        - flask-cors==5.0.0
        - albumentations==1.4.21
        - numba==0.60.0
        - sympy==1.13.1
        - networkx==3.2.1
        - zarr==2.18.2
        - numcodecs==0.12.1
        - umap-learn==0.5.7
        - pynndescent==0.5.13
        - osqp==0.6.5
        - ecos==2.0.13
        - gpustat==1.1.1
        - nvidia-ml-py==12.535.133
```

