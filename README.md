# DFAB-GCN

**DFAB-GCN: A graph convolutional network framework for ASD classification and model-level analysis of candidate ASD-related regions**

![Overview of the DFAB-GCN architecture](figure1.PNG)

## Overview

DFAB-GCN is a graph neural network framework for classifying Autism Spectrum Disorder (ASD) from functional magnetic resonance imaging (fMRI) data. The model is designed to learn discriminative representations from brain connectivity graphs while integrating information across the two cerebral hemispheres.

In addition to ASD classification, DFAB-GCN supports model-level interpretability analysis. It can be used to identify candidate ASD-related brain regions and investigate the functional associations captured by the model.

## Highlights

- Graph-based representation learning for fMRI-derived brain connectivity data
- Transhemispheric feature fusion for modeling interactions between the cerebral hemispheres
- End-to-end ASD classification
- Model-level analysis of candidate ASD-related brain regions
- Support for biomarker-oriented interpretability studies

## Publication

For the methodology and experimental results, see the accompanying article:

[A graph convolutional network framework for ASD classification and model-level analysis of candidate ASD-related regions](https://www.sciencedirect.com/science/article/abs/pii/S1746809426011341)

If you use this repository in your research, please cite the article. A complete BibTeX entry should be added here once the final bibliographic metadata is available.

## Requirements

The reference environment is:

- Python 3.10
- CUDA 12.1
- PyTorch 2.5.1
- PyTorch Geometric 2.6.1

The following PyTorch Geometric extension packages were used:

- `torch_cluster==1.6.3+pt25cu121`
- `torch_scatter==2.1.2+pt25cu121`
- `torch_sparse==0.6.18+pt25cu121`
- `torch_spline_conv==1.2.2+pt25cu121`

> CUDA support is recommended for training but is not required for basic code inspection or CPU execution. If you use another PyTorch or CUDA version, install the matching PyTorch Geometric wheels.

## Installation

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd DFAB-GCN
```

Replace `<REPOSITORY_URL>` with the URL of this repository.

### 2. Create an isolated environment

Using Conda:

```bash
conda create -n dfab-gcn python=3.10 -y
conda activate dfab-gcn
python -m pip install --upgrade pip
```

### 3. Install PyTorch

For PyTorch 2.5.1 with CUDA 12.1:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only execution:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install PyTorch Geometric and its extensions

For PyTorch 2.5.x with CUDA 12.1:

```bash
pip install \
  torch_cluster==1.6.3 \
  torch_scatter==2.1.2 \
  torch_sparse==0.6.18 \
  torch_spline_conv==1.2.2 \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

pip install torch_geometric==2.6.1
```

For a different PyTorch or CUDA version, select the corresponding wheel index from the [official PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html).

### 5. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

To keep the environment reproducible, `requirements.txt` should contain only project-specific dependencies and should not override the PyTorch or PyTorch Geometric versions installed above.

### 6. Verify the environment

```bash
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

For the recommended GPU environment, the output should report PyTorch 2.5.1, PyTorch Geometric 2.6.1, and CUDA 12.1.

## Data Preparation

Prepare the fMRI-derived graph data before training and update the dataset paths and experiment settings required by the repository.

To make an experiment reproducible, record at least the following information:

- Dataset name and preprocessing pipeline
- Node definitions and graph construction method
- Feature and adjacency-matrix formats
- Training, validation, and test splits
- Random seed
- Evaluation metrics

Do not commit private, restricted, or personally identifiable neuroimaging data to the repository.

## Training and Prediction

Run the main entry point to train DFAB-GCN and generate predictions:

```bash
python main.py
```

If validation data are supported by the current implementation, enable them through the corresponding runtime arguments or configuration file. Before training on a new dataset, review and tune the relevant hyperparameters, including the learning rate, hidden dimensions, dropout rate, batch size, and graph-construction settings.

## Reproducibility

For reliable comparisons, we recommend reporting:

- Exact dataset split or cross-validation protocol
- Random seeds and number of repeated runs
- Mean and standard deviation across runs
- Accuracy, sensitivity, specificity, F1 score, and ROC-AUC
- Hardware and software versions
- Hyperparameters used for the reported results

Where possible, save model checkpoints, prediction files, and experiment logs under clearly named output directories.

## Troubleshooting

### PyTorch Geometric reports an undefined symbol

This usually indicates a mismatch between PyTorch, CUDA, and one or more PyTorch Geometric extension wheels. Check the installed versions:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
pip show torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
```

Reinstall the extension packages using the wheel index that matches both the installed PyTorch version and its CUDA build.

### CUDA is not available

Check the environment:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

The CUDA version bundled with PyTorch must be compatible with the installed NVIDIA driver. A full local CUDA toolkit is not required for most pre-built PyTorch wheels, but a compatible NVIDIA driver is required.

## Contributing

Issues and pull requests are welcome. When reporting a problem, please include:

- Operating system
- Python version
- PyTorch and PyTorch Geometric versions
- CUDA version, if applicable
- Full error message
- Minimal steps needed to reproduce the issue

If this project supports your research, consider starring the repository and citing the associated publication.

