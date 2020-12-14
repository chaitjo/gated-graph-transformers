# Gated Graph Transformer for OGB Graph Property Prediction

This repository implements Gated Graph Transformers for `ogbg` datasets using DGL.

## Installation

```
conda create -n ogb python=3.7
conda activate ogb

# Install PyTorch 1.6 for CUDA 10.x
conda install pytorch=1.6 cudatoolkit=10.x -c pytorch

# Install DGL for CUDA 10.x
conda install -c dglteam dgl-cuda10.x

# Install other dependencies
conda install tqdm scikit-learn pandas urllib3 tensorboard

# Install OGB
pip install -U ogb

# Optional
conda install jupyterlab -c conda-forge
```
