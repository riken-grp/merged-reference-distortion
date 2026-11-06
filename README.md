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

If you use this code or models, please cite:

```bibtex
@inproceedings{kruengkrai-yoshino-2026,
    title = "What Changes in Hidden States When Model Merging Hurts Task Performance?",
    author = "Kruengkrai, Canasai  and Yoshino, Koichiro",
    booktitle = "Proceedings of the 15th International Joint Conference on Natural Language Processing and the 5th Conference of the {A}sia-Pacific Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    year = "2026",
    url = "https://aclanthology.org/2026.aacl-long.180/",
    pages = "4021--4034",
}
```


## Acknowledgments

This work was supported by JSPS KAKENHI Grant Number JP26K14954.
Part of this research was conducted by AIST and its subcontractor, The University of Osaka, under FRONTia, a Japanese national program led by the Ministry of Economy, Trade and Industry (METI) and the New Energy and Industrial Technology Development Organization (NEDO), aimed at developing a domestic multimodal foundation model for AI robots and physical AI.
