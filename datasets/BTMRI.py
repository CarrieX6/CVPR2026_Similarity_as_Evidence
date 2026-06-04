# 文件路径: dassl/data/datasets/BUSI.py

import os
import pickle
from collections import defaultdict

from .oxford_pets import OxfordPets # 我们会复用它的一些通用函数
from .medical_id_common import resolve_medical_id_paths
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class BTMRI(DatasetBase):

    # 定义数据集的文件夹名称
    def __init__(self, cfg):
        self.dataset_name = "BTMRI"  # 数据集的名称
        self.split_path, self.image_dir, self.dataset_dir = resolve_medical_id_paths(
            cfg, self.dataset_name
        )
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # BUSI 数据集直接从我们创建的 JSON 文件读取，不需要像 OxfordFlowers 那样从 .mat 文件生成
        # read_split 是一个通用函数，可以直接读取我们生成的 JSON 格式
        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)

        # --- 小样本 (Few-Shot) 数据集生成的逻辑 ---
        # 这部分逻辑和 OxfordFlowers 完全一样，可以直接复用
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                # 如果已经生成过同样配置的小样本数据，直接加载
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                # 否则，从完整数据集中生成小样本数据
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                # 验证集通常也做小样本处理，以加快验证速度
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # --- 子类采样逻辑 (如果只想用一部分类别训练) ---
        # 这部分逻辑也和 OxfordFlowers 完全一样
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        # 调用父类初始化，完成数据集对象的创建
        super().__init__(train_x=train, val=val, test=test)