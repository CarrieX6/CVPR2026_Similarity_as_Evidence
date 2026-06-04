# Data Preparation

This is the **conference SaE** release. It does not ship medical images. Prepare datasets locally and pass their parent directory with `--root`.

## Supported Datasets (10)

| Dataset | Config | CuPL prompts |
|---------|--------|--------------|
| BTMRI | `configs/datasets/BTMRI.yaml` | `cupl/descriptors_btmri.json` |
| BUSI | `configs/datasets/BUSI.yaml` | `cupl/descriptors_busi.json` |
| CHMNIST | `configs/datasets/CHMNIST.yaml` | `cupl/descriptors_chmnist.json` |
| COVID_19 | `configs/datasets/COVID_19.yaml` | `cupl/descriptors_covid_19.json` |
| DermaMNIST | `configs/datasets/DermaMNIST.yaml` | `cupl/descriptors_dermamnist.json` |
| KneeXray | `configs/datasets/KneeXray.yaml` | `cupl/descriptors_kneexray.json` |
| Kvasir | `configs/datasets/Kvasir.yaml` | `cupl/descriptors_kvasir.json` |
| LungColon | `configs/datasets/LungColon.yaml` | `cupl/descriptors_lungcolon.json` |
| OCTMNIST | `configs/datasets/OCTMNIST.yaml` | `cupl/descriptors_octmnist.json` |
| RETINA | `configs/datasets/RETINA.yaml` | `cupl/descriptors_retina.json` |

Each dataset is expected under:

```text
<DATA_ROOT>/<DatasetName>/
```

The loader looks for:

1. `split_<DatasetName>_adapted.json`
2. `split_<DatasetName>.json`

The adapted split is preferred when both exist.

## Split JSON Format

Use three top-level keys: `train`, `val`, and `test`. Each item should contain an image path relative to the dataset directory, an integer label, and a class name.

```json
{
  "train": [
    ["benign tumor/example_001.png", 0, "benign tumor"]
  ],
  "val": [
    ["normal scan/example_010.png", 2, "normal scan"]
  ],
  "test": [
    ["malignant tumor/example_020.png", 1, "malignant tumor"]
  ]
}
```

## Example Layout (BUSI)

```text
data/BUSI/
├── split_BUSI.json
├── benign tumor/
├── malignant tumor/
└── normal scan/
```

## Dataset Licenses

Download images from the official dataset sources and follow their licenses. Do not redistribute medical images unless the corresponding dataset license allows it.
