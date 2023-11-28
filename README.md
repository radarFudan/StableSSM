<div align="center">

# StableSSM

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2311.14495-B31B1B.svg)](https://arxiv.org/abs/2311.14495)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Use stable reparameterizations to improve the long-term memory learning and optimization stability.

## Installation

#### Pip

```bash
# clone project
git clone git@github.com:radarFudan/StableSSM.git
cd StableSSM

# [OPTIONAL] create conda environment
conda create -n StableSSM python=3.11
conda activate StableSSM

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip --default-timeout=1000 install -U -r requirements.txt # CUDA 12
pip --default-timeout=1000 install -U -r requirements_11.txt --index-url https://download.pytorch.org/whl/cu117 # CUDA11
```

#### Conda

```bash
# clone project
git clone git@github.com:radarFudan/StableSSM.git
cd StableSSM

# create conda environment and install dependencies
conda env create -f environment.yaml -n StableSSM

# activate conda environment
conda activate StableSSM
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
