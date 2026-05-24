# ATAuth

Reference implementation of **ATAuth: Adaptive Transformer-Based Continuous Authentication via Neural Architecture Search**.

This repository implements the full pipeline described in the paper:
1. Multi-view tokenization (GSFF / GTFF / LSFF / LTFF) realized as internal reshape operations inside each operator.
2. Adaptive Transformer Architecture (ATA) with two operator slots per layer.
3. DC-sMoE: depthwise 1D convolution-integrated sparse Mixture-of-Experts, replacing the standard FFN.
4. Personalized CNN extractor (4-stage funnel structure with NAS-selected kernel sizes and strides).
5. Factorization Machine classifier in O(kn) linear time.
6. Single-path one-shot supernet training over the search space (12,544 candidates per layer).
7. Regularized evolution with FLOPs and latency budgets, population 500, tournament 100, 1000 cycles.
8. From-scratch retraining of the selected architecture.
9. Mimic-attack security analysis under additive Gaussian sensor noise.

## Project layout

```
ATAuth/
├── configs/               # Hydra configs (dataset / search / train)
├── scripts/               # Entry-point pipeline + dataset preprocessing
├── src/
│   ├── data/              # UCI_HAR / WISDM_HARB / self-collected dataloaders + time-based split
│   ├── models/atauth/     # tokenizer, attention, dc_smoe, adaptive_layer, cnn_extractor, fm, supernet
│   ├── nas/               # supernet training, regularized evolution
│   ├── training/          # retraining and evaluation
│   ├── evaluation/        # metrics (Acc/EER/F1/FAR/FRR), mimic attack
│   └── utils/             # logger, seeding
├── run.sh                 # one-click pipeline launcher
└── requirements.txt
```

## Installation

```
pip install -r requirements.txt
```

## Datasets

The repository does **not** ship raw datasets. Place them under `./data/` as follows:

- `./data/UCI_HAR/`   — official UCI HAR layout with `train/` and `test/` directories containing `Inertial Signals/` and `subject_*.txt`.
- `./data/WISDM_HARB/` — official WISDM-HARB layout with `raw/phone/accel/` and `raw/phone/gyro/`.
- `./data/self_collected/` — a directory containing `user_<id>.npy` files. Each file is `[T, 9]` (accel xyz + gyro xyz + mag xyz). Use `scripts/preprocess_self_collected.py` to convert raw CSV streams.

Time-based 80% / 20% split is performed automatically inside the dataloader to eliminate temporal leakage between training and testing windows. Random sampling is restricted to the training segment; testing uses non-overlapping segmentation.

## Quick start

End-to-end pipeline on the self-collected dataset:

```
DATASET=ours LEGIT_USER=1 NUM_UNSEEN=70 ./run.sh
```

End-to-end pipeline on UCI_HAR:

```
DATASET=uci_har LEGIT_USER=1 NUM_UNSEEN=10 ./run.sh
```

End-to-end pipeline on WISDM_HARB:

```
DATASET=wisdm_harb LEGIT_USER=1600 NUM_UNSEEN=20 ./run.sh \
  sensor_channels=6 t1=5 t2=40
```
