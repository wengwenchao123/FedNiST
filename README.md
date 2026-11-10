# FedNiST

This directory contains the complete code, configuration, example dataset, and
existing pretrained weights required by `run.py`.

## Dataset Notice

Due to the AAAI upload file-size limit, only the PEMS08 dataset is included as
an example. Please download the other datasets from their public sources and
place them in the `data` directory before running the corresponding experiments.

## Environment

Python 3.9 or later is recommended. Install PyTorch for the CUDA version used by
the machine first, then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

`pymetis` is required only when using `--client_partition_method metis_dtw`.

## Run

Run commands from this directory because configuration, data, cache, and
pretrained model paths are relative to the working directory.

```powershell
cd FedNiST
python run.py --dataset PEMSD4 --mode train --device cuda:0
```

Use ratio-based random node partitioning:

```powershell
python run.py --dataset PEMSD4 --num_clients 4 --client_node_ratio 0.4,0.3,0.2,0.1
```

Use DTW and METIS node partitioning:

```powershell
python run.py --dataset PEMSD4 --client_partition_method metis_dtw
```

Supported configuration names are `PEMSD3`, `PEMSD4`, `PEMSD7`, `PEMSD7(L)`,
`PEMSD7(M)`, `PEMSD8`, `PEMS-BAY`, and `METR-LA`.

Training outputs are created under `experiments/<dataset>/<timestamp>` inside
this directory.
