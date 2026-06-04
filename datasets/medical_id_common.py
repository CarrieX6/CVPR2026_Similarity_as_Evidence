"""十个医学 ID 数据集共用：从 cfg.DATASET.ROOT 解析 split 与图像根目录。"""
import os


def resolve_medical_id_paths(cfg, dataset_name):
    """
    在 ROOT/<DatasetName>/ 下查找：
    - split：优先 split_<Name>_adapted.json，否则 split_<Name>.json
    - 图像根：若存在 jpg/ 且 split 中首条路径不含子目录（如 BTMRI），则用 jpg/；
      否则用数据集根（类文件夹结构，如 BUSI/xxx/yyy.png）
    """
    root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
    if not root:
        raise ValueError(
            f"{dataset_name}: cfg.DATASET.ROOT 为空，请在 train.py 中传入 --root "
            "(例如仓库下 data 目录的绝对路径)。"
        )
    dataset_dir = os.path.join(root, dataset_name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"{dataset_name}: 数据目录不存在: {dataset_dir}")

    split_adapted = os.path.join(dataset_dir, f"split_{dataset_name}_adapted.json")
    split_plain = os.path.join(dataset_dir, f"split_{dataset_name}.json")
    if os.path.isfile(split_adapted):
        split_path = split_adapted
    elif os.path.isfile(split_plain):
        split_path = split_plain
    else:
        raise FileNotFoundError(
            f"{dataset_name}: 在 {dataset_dir} 下未找到 "
            f"split_{dataset_name}_adapted.json 或 split_{dataset_name}.json"
        )

    jpg_dir = os.path.join(dataset_dir, "jpg")
    # BTMRI：扁平文件名在 jpg/ 下；其余数据集为「类名/文件」相对路径，根目录为数据集文件夹。
    # 不在此整文件 read_json（BTMRI 的 split 很大，避免重复 IO）。
    if dataset_name == "BTMRI" and os.path.isdir(jpg_dir):
        image_dir = jpg_dir
    else:
        image_dir = dataset_dir

    return split_path, image_dir, dataset_dir
