#!/bin/bash

eval "$(conda shell.bash hook)"

# Create a new environment
conda create -n facetalk python=3.9

# Activate virtual environment if needed
conda activate facetalk

# Install packages
pip3 install torch==2.2.0+cu118 torchaudio==2.2.0+cu118 torchvision==0.17.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip3 install tqdm==4.66.2 librosa==0.10.1 omegaconf==2.3.0 einops==0.7.0 moviepy==1.0.3 transformers==4.38.2
pip3 install git+https://github.com/facebookresearch/pytorch3d.git@7566530669203769783c94024c25a39e1744e4ed
pip3 install torch_geometric==2.5.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip3 install open3d==0.18.0 opencv-python==4.9.0.80 point-cloud-utils==0.30.4 pytorch-lightning==1.8.5 vtk==9.3.0 trimesh==4.1.8 PyMCubes==0.1.4 pyvista==0.43.3 PyYAML==6.0.1
pip3 install pydub==0.25.1 wandb==0.16.4 hydra-core==1.3.2







