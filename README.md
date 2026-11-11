# FedNiST

## Table of Contents

- `config_file`: training and model configurations for each dataset.
- `lib`: utilities for data loading, preprocessing, normalization, and evaluation metrics.
- `model`: implementation of FedNiST.
- `data`: directory for the datasets.
- `train.py`: command-line entry point.
- `model/BasicTrainer.py`: federated training, validation, checkpointing, and final evaluation.

## Requirements

Python 3.9 or later, PyTorch 2.0 or later, NumPy, pandas, SciPy, and PyTables. `argparse` and `configparser` are included in the Python standard library.

Install a PyTorch build compatible with your CUDA environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

# Data Preparation

For convenience, we package these datasets used in our model in [Google Drive](https://drive.google.com/file/d/1IQ3DX9GwLr5L-6nHY5AYFT-0z5JIuzdi/view?usp=sharing).

Unzip the downloaded dataset files into the `data` folder.

Dataset files are not included in this repository. Directory and file names are case-sensitive on Linux.

## Model Training

Run the following command from the project root:

```bash
python train.py --dataset DATASET_NAME --lazy-windows --output outputs/RUN_NAME
```

Replace `DATASET_NAME` with one of the following identifiers:

| Dataset | Identifier |
|---|---|
| METR-LA | METR-LA |
| PEMS-BAY | PEMS-BAY |
| PEMS03 | PEMSD3 |
| PEMS04 | PEMSD4 |
| PEMS07 | PEMSD7 |
| PEMS08 | PEMSD8 |
| PEMS07(M) | PEMSD7(M) |
| PEMS07(L) | PEMSD7(L) |

For example, train FedNiST on PEMS08:

```bash
python train.py --dataset PEMSD8 --num-clients 4 --embedding-dim 32 --num-experts 4 --seed 0 --lazy-windows --output outputs/PEMS08-seed0
```

Quote dataset identifiers containing parentheses:

```bash
python train.py --dataset "PEMSD7(M)" --lazy-windows --output outputs/PEMS07M-seed0
```

The corresponding configuration is loaded from `config_file` automatically. To specify another configuration file, use `--config-file`:

```bash
python train.py --dataset PEMSD8 --config-file config_file/PEMSD8_FedNiST.conf --lazy-windows --output outputs/PEMS08-custom
```

Training saves the configuration, checkpoint, and evaluation results in the specified output directory. Final test evaluation runs automatically after training. Use a new output directory for each run.

GPU execution uses `cuda:0` by default. Add `--device cpu` to run on CPU. Use `python train.py --help` to view all available options.
