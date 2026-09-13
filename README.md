# SaE — Similarity-as-Evidence (CVPR 2026 Highlight)

Official **conference-version** release of our CVPR 2026 paper:

> **Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning**

This repository implements the core **SaE active-learning pipeline**:

- frozen **BiomedCLIP** AL-VLM backbone;
- **scalar** MEH evidence head (`VLM_EH_V2`) with Dirichlet **vacuity** and **dissonance**;
- round-scheduled acquisition score **`wv · vac + wd · dis`** (`ACQ_SCORE_MODE: legacy_wv_wd`);
- **class-balanced acquisition** via pseudo-label bucketing (`METHOD: sae_ca`).

## Repository Layout

```text
SaE/
├── train.py
├── configs/
│   ├── datasets/          # 10 medical ID datasets + CuPL ASPATH
│   ├── methods/sae.yaml   # conference default
│   └── trainers/ALVLM/vit_b16.yaml
├── datasets/
├── trainers/
├── cupl/descriptors_*.json
├── examples/
└── DATA.md
```

## Supported Datasets (10)

`BTMRI`, `BUSI`, `CHMNIST`, `COVID_19`, `DermaMNIST`, `KneeXray`, `Kvasir`, `LungColon`, `OCTMNIST`, `RETINA`

Each dataset config sets `TRAINER.COOPAL.ASPATH` to the matching CuPL descriptor file under `cupl/`.

## Installation

```bash
pip install -r requirements.txt
```

Optional BiomedCLIP weight download:

```bash
bash scripts/download_biomedclip_weights.sh
```

## Data

No medical images are included. See `DATA.md` for split JSON format and directory layout.

## Quick Start

Single dataset (BUSI):

```bash
DATA_ROOT=/path/to/data OUTPUT_DIR=output/sae_busi_seed1 GPU=0 \
  bash examples/run_sae_busi.sh
```

All ten datasets sequentially:

```bash
DATA_ROOT=/path/to/data SEED=1 GPU=0 bash examples/run_sae_ten_datasets.sh
```

## Paper-aligned Active-learning Budget

The default configs follow the 20% annotation-budget protocol used in the
paper:

- 5 acquisition rounds with 4% of the original training pool queried per
  round;
- 100 training epochs per round and a training batch size of 32;
- `METHOD: "sae_ca"` for class-balanced acquisition;
- `pcb_flag: "false"`, because `sae_ca` performs class balancing internally.

The number queried per round is computed as
`int(len(training_pool) * 0.04)`. Consequently, datasets whose pool size is not
divisible by 25 can be slightly below 20% after integer rounding. For example,
BUSI selects 15 samples per round and 75/389 (19.28%) over five rounds, while
Kvasir selects 80 per round and 400/2000 (20%). The optimizer and learning-rate
scheduler are carried across acquisition rounds rather than reset each round.

## Method Config (`configs/methods/sae.yaml`)

```yaml
TRAINER:
  COOPAL:
    query: 0.04
    Totalrounds: 5
    pcb_flag: "false"
    METHOD: "sae_ca"
    ACQ_SCORE_MODE: "legacy_wv_wd"   # wv*vacuity + wd*dissonance
    ACQ_NORM_MODE: "batch_minmax"
    SAE_CA:
      ENABLE: true
    MEH_VERSION: "v2"
```

Acquisition weights `wv`, `wd` are scheduled by AL round inside `MEH_Selector` (early rounds emphasize vacuity, later rounds emphasize dissonance).

## Notes

- BiomedCLIP and medical datasets have separate licenses; check upstream terms before redistribution.

## Acknowledgements

Built on PCB/ALVLM-style active learning, Dassl, OpenCLIP, and BiomedCLIP.

## Citation

```bibtex
@InProceedings{Xie_2026_CVPR,
    author    = {Xie, Zhuofan and Lin, Zishan and Lin, Jinliang and Qi, Jie and Hong, Shaohua and Li, Shuo},
    title     = {Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {20973-20984}
}
```
