`minimal-basis`
---------------

![example workflow](https://github.com/sudarshanv01/minimal-basis/actions/workflows/main.yml/badge.svg?event=push)


Predict transition state properties with knowledge of only the reactant and product graphs and coefficient matrices.  

## Installation


- Install non-pytorch dependencies with conda:
```
conda env create -f environment.yml
```

- Install pytorch dependencies with mamba:
```
mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch -c conda-forge
mamba install pyg -c pyg
mamba install pytorch-scatter -c pyg
mamba install pytorch-cluster -c pyg
```

- Install `pip`-only dependencies:
```
pip install -r requirements.txt
```

- Install `minimal-basis`:
```
pip install -e .
```

- (Optional) Install requirements for testing

```
pip install -r requirements-test.txt
```