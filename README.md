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

```markdown
Step 1: Clone repository
git clone https://github.com/TPRS-GC/TPRS-GC.git
cd TPRS-GC

Step 2: Create environment
conda create --name TPRS python=3.9
conda activate TPRS

Step 3: Install dependencies
pip install -r requirements.txt
