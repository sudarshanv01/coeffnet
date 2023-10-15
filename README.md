`coeffnet`
---------------

![example workflow](https://github.com/sudarshanv01/minimal-basis/actions/workflows/main.yml/badge.svg?event=push)

![network](images/network.pdf)

Equivariant Neural Network to predict transition state properties with knowledge of only the reactant and product graphs and coefficient matrices. 

## Installation

Install non-pytorch dependencies with conda:
```
conda env create -f environment.yml
```

Install pytorch dependencies with mamba[^1]:
```
mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch -c conda-forge
mamba install pyg -c pyg
mamba install pytorch-scatter -c pyg
mamba install pytorch-cluster -c pyg
```

Install `pip`-only dependencies:
```
pip install -r requirements.txt
```

Install `coeffnet`:
```
pip install -e .
```

(Optional) Install requirements for testing

```
pip install -r requirements-test.txt
```

(Optional) Install requirements for docs

```
pip install -r requirements-docs.txt
```

[^1]: Note that M1 macs do not have `mamba` support for `pyg`. Instead, follow `pip` instructions [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-source).