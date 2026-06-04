import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix



@EVALUATOR_REGISTRY.register()
class ClassificationPerClass(object): # 确保类名和您的代码一致
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        self.cfg = cfg
        self._lab2cname = lab2cname
        # ==================== 修改区域 1: __init__ ====================
        # 我们在这里直接初始化所有变量，确保它们不是None
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        # 直接初始化，不再需要配置文件来激活
        self._per_class_res = defaultdict(list)
        # ==================== 修改结束 ====================

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        # 重置时也确保它是一个新的空字典
        self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        # ==================== 修改区域 2: process ====================
        # 去掉了 "if self._per_class_res is not None:" 的判断
        # 确保每次都收集每个类别的匹配结果
        for i, label in enumerate(gt):
            label = label.item()
            matches_i = int(matches[i].item())
            self._per_class_res[label].append(matches_i)
        # ==================== 修改结束 ====================

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total if self._total > 0 else 0
        err = 100.0 - acc
        
        unique_labels = np.unique(self._y_true)
        macro_f1 = 100.0 * f1_score(self._y_true, self._y_pred, average="macro", labels=unique_labels) if len(unique_labels) > 0 else 0

        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._lab2cname is not None:
            labels = sorted(self._per_class_res.keys())

            print("\n=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname.get(label, "Unknown")
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc_class = 100.0 * correct / total if total > 0 else 0
                accs.append(acc_class)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc_class:.1f}%"
                )
            
            if accs:
                mean_acc = np.mean(accs)
                print(f"* per-class average: {mean_acc:.1f}%")
                results["perclass_accuracy"] = mean_acc

        return results