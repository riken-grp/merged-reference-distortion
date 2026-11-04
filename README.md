# merged-reference-distortion

Code for experiments on merged-reference hidden-state distortion after model merging.

## Installation

This project uses `pyenv` and `poetry`.

### 1. Set Python version

We use Python 3.12.3.

```bash
pyenv install 3.12.3
pyenv local 3.12.3
```

Check the Python version:

```bash
python --version
```

Expected output:

```text
Python 3.12.3
```

### 2. Install dependencies

Install the project and its dependencies with Poetry:

```bash
poetry install
```

[Optional] Install pre-commit for development:

```bash
poetry run pre-commit install
```

### 3. Check installation

Check that the local package is importable:

```bash
poetry run python -c "import mrd; print(mrd.__name__)"
```

Expected output:

```text
mrd
```

Check that PyTorch can access CUDA:

```bash
poetry run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The second value should be `True` when running the GPU experiments.

## Experiments

Experiment scripts for T5-large and Phi-3.5 are provided under:

```text
experiments/t5-large/
experiments/phi35/
```

Checkpoints and experimental outputs are available at:

https://huggingface.co/datasets/canasai/merged-reference-distortion-artifacts

## Citation

What Changes in Hidden States When Model Merging Hurts Task Performance?
Canasai Kruengkrai and Koichiro Yoshino
To appear in AACL-IJCNLP 2026.

The official BibTeX entry and paper link will be added when they become available.
