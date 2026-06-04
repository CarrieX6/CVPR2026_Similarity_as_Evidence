# Provenance: RA-SaE active-learning base class.
import torch

class AL(object):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, **kwargs):
        self.unlabeled_dst = unlabeled_dst     # 1. 原始的、完整的未标注数据集
        self.U_index = U_index                  # 2. 原始数据集中，所有未标注样本的“全局索引”列表
        self.unlabeled_set = torch.utils.data.Subset(unlabeled_dst, U_index)    # 3. 根据全局索引，创建一个 PyTorch 的 Subset 对象，方便后续处理
        self.n_unlabeled = len(self.unlabeled_set) # 4. 未标注样本的总数
        self.n_class = n_class         # 5. 数据集的总类别数
        self.model = model              # 6. 当前的模型
        self.index = []
        self.cfg = cfg            # 7. 配置信息

    def select(self, **kwargs):
        return
    

