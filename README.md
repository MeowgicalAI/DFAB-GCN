DFAB-GCN

A graph convolutional network framework for ASD classification and model-level analysis of candidate ASD-related regions

![alt text](figure1-1.PNG)

Overview

DFAB-GCN is an end-to-end graph neural network framework for Autism Spectrum Disorder (ASD) classification using resting-state functional magnetic resonance imaging (rs-fMRI). It models brain connectivity at both the individual and population levels while preserving subject-specific differences.

Beyond classification, DFAB-GCN integrates data-driven modeling with interpretability analysis. The framework identifies candidate ASD-related brain regions and provides model-level evidence for associations between regional importance and ASD-related functional differences.

Experiments reported in the accompanying paper demonstrate competitive classification performance on the ABIDE dataset and evaluate the model's generalization potential through supplementary cross-dataset analysis.

Research-use notice: This repository is intended for academic research and is not a clinical diagnostic system.

Model Architecture

DFAB-GCN consists of three main components:

1. Transhemispheric Fusion Graph Convolutional Network (THF-GCN)

THF-GCN models information exchange between the left and right cerebral hemispheres. It combines Self-Attention Graph Pooling and Dense Differentiable Pooling to capture complementary local and global representations while reducing redundant brain-region information.

2. Graph Embedding Graph Attention Network (GEmb-GAT)

GEmb-GAT constructs a population graph from individual-level representations using cosine similarity. A Graph Transformer then learns latent interaction patterns across subjects while retaining subject-specific characteristics.

3. Probability-Based Functional Association (PBFA)

PBFA combines brain-region importance scores with within-group probability statistics. It supports the identification of candidate ASD-related regions and the model-level analysis of between-group differences in regional importance.

Key Features

End-to-end ASD classification from rs-fMRI-derived brain graphs

Joint modeling of individual brain graphs and population-level relationships

Explicit fusion of information across the two cerebral hemispheres

Dual-channel graph pooling for complementary local and global representations

Candidate brain-region identification through model-level interpretability analysis

Optional validation support during training

Requirements

The reference environment is:

Python 3.10

CUDA 12.1

PyTorch 2.5.1

PyTorch Geometric 2.6.1

The following PyTorch Geometric extension packages were used:

torch_cluster==1.6.3+pt25cu121

torch_scatter==2.1.2+pt25cu121

torch_sparse==0.6.18+pt25cu121

torch_spline_conv==1.2.2+pt25cu121

CUDA support is recommended for training but is not required for basic code inspection or CPU execution. If you use another PyTorch or CUDA version, install the matching PyTorch Geometric wheels.

Installation

1. Clone the repository

git clone https://github.com/MeowgicalAI/DFAB-GCN.git
cd DFAB-GCN

2. Create an isolated environment

conda create -n dfab-gcn python=3.10 -y
conda activate dfab-gcn
python -m pip install --upgrade pip

3. Install PyTorch

For PyTorch 2.5.1 with CUDA 12.1:

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

For CPU-only execution:

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

4. Install PyTorch Geometric

For PyTorch 2.5.x with CUDA 12.1:

pip install torch_cluster==1.6.3 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

pip install torch_geometric==2.6.1

For a different environment, select the matching wheel index from the official PyTorch Geometric installation guide.

5. Install the remaining dependencies

pip install -r requirements.txt

For reproducibility, requirements.txt should contain project-specific dependencies without overriding the PyTorch or PyTorch Geometric versions installed above.

6. Verify the installation

python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

For the recommended GPU environment, the output should report:

PyTorch: 2.5.1+cu121
PyG: 2.6.1
CUDA available: True
CUDA version: 12.1

Data Preparation

Prepare the rs-fMRI-derived graph data before training and update the dataset paths and experiment settings required by the code.

For reproducible experiments, document the following:

Dataset name and preprocessing pipeline

Brain atlas and region-of-interest definitions

Node features and graph-construction method

Feature-matrix and adjacency-matrix formats

Training, validation, and test splits

Random seeds

Evaluation metrics

Do not commit private, restricted, or personally identifiable neuroimaging data to the repository.

Training and Prediction

Run the main entry point to train the model and generate predictions:

python main.py

If supported by the current code version, provide a validation set through the relevant runtime arguments or configuration fields.

For a new dataset, consider tuning:

Learning rate and weight decay

Hidden dimensions

Dropout rate

Batch size

Graph-construction threshold

Population-graph similarity settings

Pooling ratio

Reproducibility

For reliable comparisons, report:

Dataset split or cross-validation protocol

Random seeds and number of repeated runs

Mean and standard deviation across runs

Accuracy, sensitivity, specificity, F1 score, and ROC-AUC

Hardware and software versions

Hyperparameters used for the reported results

Where possible, retain model checkpoints, prediction files, and experiment logs under clearly named output directories.

Troubleshooting

Undefined symbols when importing PyTorch Geometric

This usually indicates a mismatch among PyTorch, CUDA, and the PyTorch Geometric extension wheels.

Check the installed versions:

python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
pip show torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

Reinstall the extensions using a wheel index that matches both the installed PyTorch version and its CUDA build.

CUDA is not available

Check the NVIDIA driver and PyTorch environment:

nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

The CUDA runtime bundled with PyTorch must be compatible with the installed NVIDIA driver. Most pre-built PyTorch wheels do not require a complete local CUDA toolkit, but they do require a compatible NVIDIA driver.

Out-of-memory errors

Try one or more of the following:

Reduce the batch size

Reduce the hidden dimensions or pooling ratio

Disable unnecessary cached tensors

Use mixed-precision training if supported by the implementation

Close other GPU processes

Citation

If you use DFAB-GCN in your research, please cite:

@article{meng2026dfabgcn,
  title   = {A graph convolutional network framework for ASD classification and model-level analysis of candidate ASD-related regions},
  author  = {Meng, Lu and Zhu, Weitao and Gao, Shuoqian and Xie, Keli and Li, Zheng and Wu, Rina and Lin, Xuejie},
  journal = {Biomedical Signal Processing and Control},
  volume  = {123},
  pages   = {110580},
  year    = {2026},
  doi     = {10.1016/j.bspc.2026.110580}
}

Contributing

Issues and pull requests are welcome. When reporting a problem, include:

Operating system

Python version

PyTorch and PyTorch Geometric versions

CUDA version, if applicable

Full error message

Minimal steps required to reproduce the issue

If this project supports your research, consider starring the repository and citing the associated publication.