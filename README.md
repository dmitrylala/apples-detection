<div align="center">

# Apples detection

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![tests](https://github.com/dmitrylala/apples-detection/actions/workflows/test.yml/badge.svg)](https://github.com/dmitrylala/apples-detection/actions/workflows/test.yml)
[![code-quality](https://github.com/dmitrylala/apples-detection/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/dmitrylala/apples-detection/actions/workflows/code-quality-main.yaml)
[![codecov](https://codecov.io/gh/dmitrylala/apples-detection/branch/main/graph/badge.svg)](https://codecov.io/gh/dmitrylala/apples-detection) <br>

</div>

## Description

What it does

## How to run

Install dependencies

```bash
some install commands
```

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
