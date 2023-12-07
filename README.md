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

### SSMs

The state-space models we are talking about refer to the linear RNNs with layer-wise nonlinear activations.

Discrete-time case:
$$h_{k+1} = \Lambda h_k+Ux_k+b$$

$$y_k = c^\top \sigma(h_k)$$

Continuous-time case:
$$\frac{dh_{t}}{dt} = \Lambda h_t+Ux_t+b$$

$$y_t = c^\top \sigma(h_t)$$

### Stable reparameterization

Let $W$ be the trainable parameters. 
No reparameterization is unstable parameterization
$$\Lambda = W.$$
Stable reparameterization:
$$\Lambda = -e^W, -\log(1+e^W).$$

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

## Refs

### Curse of memory phenomneon / definition of memory functions / concept of stable approximation

```bibtex
@inproceedings{
    wang2023statespace,
    title={State-space models with layer-wise nonlinearity are universal approximators with exponential decaying memory},
    author={Shida Wang and Beichen Xue},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=i0OmcF14Kf}
}
@misc{wang2023stablessm,
    title={StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization},
    author={Shida Wang and Qianxiao Li},
    year={2023},
    eprint={2311.14495},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

### Survey on sequence modelling from approximation perspective

```bibtex
@Article{JML-2-1,
    author = {Haotian Jiang and Qianxiao Li and Zhong Li and Shida Wang},
    title = {A Brief Survey on the Approximation Theory for Sequence Modelling},
    journal = {Journal of Machine Learning},
    year = {2023},
    volume = {2},
    number = {1},
    pages = {1--30},
    abstract = {We survey current developments in the approximation theory of sequence modelling in machine learning. Particular emphasis is placed on classifying existing results for various model architectures through the lens of classical approximation paradigms, and the insights one can gain from these results. We also outline some future research directions towards building a theory of sequence modelling.},
    issn = {2790-2048},
    doi = {https://doi.org/10.4208/jml.221221},
    url = {http://global-sci.org/intro/article_detail/jml/21511.html} }
```
