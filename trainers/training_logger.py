import os
import json
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def _tail_recall_mean_from_counts(
    class_sample_counts: dict,
    per_class_recall: dict,
    num_classes: int,
) -> float:
    """尾类 = 当前训练集类频不高于中位数的类；返回这些类在测试集上的 recall 均值。"""
    if not class_sample_counts or not per_class_recall or num_classes <= 0:
        return float("nan")
    counts = [int(class_sample_counts.get(i, 0)) for i in range(num_classes)]
    if max(counts, default=0) <= 0:
        return float("nan")
    med = float(np.median(counts))
    tail_idx = [i for i in range(num_classes) if counts[i] <= med]
    vals = [float(per_class_recall.get(i, 0.0)) for i in tail_idx]
    return float(np.mean(vals)) if vals else float("nan")


def _head_recall_mean_from_counts(
    class_sample_counts: dict,
    per_class_recall: dict,
    num_classes: int,
) -> float:
    """头类 = 当前训练集类频严格大于中位数的类；返回这些类在测试集上的 recall 均值。"""
    if not class_sample_counts or not per_class_recall or num_classes <= 0:
        return float("nan")
    counts = [int(class_sample_counts.get(i, 0)) for i in range(num_classes)]
    if max(counts, default=0) <= 0:
        return float("nan")
    med = float(np.median(counts))
    head_idx = [i for i in range(num_classes) if counts[i] > med]
    vals = [float(per_class_recall.get(i, 0.0)) for i in head_idx]
    return float(np.mean(vals)) if vals else float("nan")


class TrainingLogger:
    """主动学习训练日志记录器 - 增强版，包含样本统计"""
    
    def __init__(self, output_dir, dataset_name, num_classes, class_names, total_samples=None, cfg=None):
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.class_names = class_names
        self.total_samples = total_samples  # 总样本数
        self.cfg = cfg  # 可选：保存完整配置快照，便于复现与指标脚本读取超参
        
        # 创建保存目录
        self.log_dir = os.path.join(output_dir, "training_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化数据存储
        self.loss_data = {
            'round': [],
            'epoch': [],
            'batch': [],
            'loss_vlm': [],
            'loss_meh': [],
            'total_loss': [],
            'acc': []
        }
        
        # 扩展round_accuracy，增加样本统计
        self.round_accuracy = {
            'round': [],
            'overall_accuracy': [],
            'macro_f1': [],
            'balanced_accuracy': [],
            'worst_class_acc': [],
            'tail_recall_mean': [],
            'head_recall_mean': [],
            'num_samples_used': [],  # 当前轮使用的样本数
            'total_samples': [],  # 总样本数
            'sample_usage_ratio': [],  # 使用比例
            'ece': [],        # 新增
            'avg_nll': [],    # 新增
            **{f'class_{i}_{name}': [] for i, name in enumerate(class_names)},
            **{f'class_{i}_{name}_count': [] for i, name in enumerate(class_names)},  # 每类的样本数
            **{f'f1_{i}_{name}': [] for i, name in enumerate(class_names)},
            **{f'recall_{i}_{name}': [] for i, name in enumerate(class_names)},
        }
        
        # 存储每轮的样本统计
        self.sample_statistics = {}
        self.calibration_data = {}

        self.best_results = None
        self.best_round = -1
        self.best_accuracy = -1  # 保持兼容：代表 best_metric_value（不一定是 overall acc）
        self.best_metric_name = "overall_accuracy"

    def import_prior_rounds_from_csv(self, before_round: int) -> int:
        """续跑时从已有 CSV 恢复已完成轮次的指标，保证 run_summary 完整。"""
        csv_path = os.path.join(self.log_dir, "round_accuracy_with_statistics.csv")
        if not os.path.isfile(csv_path) or before_round <= 0:
            return 0
        df = pd.read_csv(csv_path)
        if "Round" not in df.columns:
            return 0
        imported = 0
        for _, row in df.iterrows():
            r = int(row["Round"])
            if r >= before_round:
                continue
            if r in self.round_accuracy["round"]:
                continue
            usage = row.get("Usage_Ratio", "0%")
            if isinstance(usage, str) and usage.endswith("%"):
                usage_ratio = float(usage.strip("%")) / 100.0
            else:
                usage_ratio = float(usage) if usage == usage else 0.0
            self.round_accuracy["round"].append(r)
            self.round_accuracy["overall_accuracy"].append(float(row.get("Overall_Accuracy", 0.0)))
            self.round_accuracy["macro_f1"].append(float(row.get("Macro_F1", float("nan"))))
            self.round_accuracy["balanced_accuracy"].append(float(row.get("Balanced_Accuracy", float("nan"))))
            self.round_accuracy["worst_class_acc"].append(float(row.get("Worst_Class_Acc", float("nan"))))
            self.round_accuracy["tail_recall_mean"].append(float(row.get("Tail_Recall_Mean", float("nan"))))
            self.round_accuracy["head_recall_mean"].append(float(row.get("Head_Recall_Mean", float("nan"))))
            self.round_accuracy["ece"].append(float(row.get("ECE", float("nan"))))
            self.round_accuracy["avg_nll"].append(float(row.get("Avg_NLL", float("nan"))))
            self.round_accuracy["num_samples_used"].append(int(row.get("Samples_Used", 0)))
            self.round_accuracy["total_samples"].append(int(row.get("Total_Samples", 0)))
            self.round_accuracy["sample_usage_ratio"].append(usage_ratio)
            for i, name in enumerate(self.class_names):
                acc_key = f"class_{i}_{name}"
                count_key = f"class_{i}_{name}_count"
                f1_key = f"f1_{i}_{name}"
                rec_key = f"recall_{i}_{name}"
                col_acc = f"{name}_Accuracy"
                col_cnt = f"{name}_Count"
                self.round_accuracy[acc_key].append(float(row.get(col_acc, 0.0)))
                self.round_accuracy[count_key].append(int(row.get(col_cnt, 0)))
                if col_acc.replace("_Accuracy", "_F1") in row or f"{name}_F1" in row:
                    self.round_accuracy[f1_key].append(float(row.get(f"{name}_F1", float("nan"))))
                if f"{name}_Recall" in row:
                    self.round_accuracy[rec_key].append(float(row.get(f"{name}_Recall", float("nan"))))
            imported += 1
        if imported:
            print(f"[al_resume] imported {imported} prior round(s) from {csv_path}")
        return imported
    
    def log_batch_loss(self, round_idx, epoch, batch_idx, loss_summary):
        """记录每个batch的loss"""
        self.loss_data['round'].append(round_idx)
        self.loss_data['epoch'].append(epoch)
        self.loss_data['batch'].append(batch_idx)
        self.loss_data['loss_vlm'].append(loss_summary.get('loss_vlm', 0.0))
        self.loss_data['loss_meh'].append(loss_summary.get('loss_meh', 0.0))
        
        # 计算总loss
        total = loss_summary.get('loss_vlm', 0.0) + loss_summary.get('loss_meh', 0.0)
        self.loss_data['total_loss'].append(total)
        self.loss_data['acc'].append(loss_summary.get('acc', 0.0))
    
    def update_sample_statistics(self, round_idx, class_counts, total_used):
        """更新样本统计信息
        
        Args:
            round_idx: 当前轮次
            class_counts: dict，每个类的样本数 {class_idx: count}
            total_used: 当前使用的总样本数
        """
        self.sample_statistics[round_idx] = {
            'class_counts': class_counts.copy(),
            'total_used': total_used,
            'total_available': self.total_samples
        }
    
    def log_round_accuracy(
        self,
        round_idx,
        overall_acc,
        per_class_acc,
        class_sample_counts=None,
        total_samples_used=None,
        ece=None,
        avg_nll=None,
        macro_f1=None,
        per_class_f1=None,
        per_class_recall=None,
        worst_class_acc=None,
        tail_recall_mean=None,
    ):
        """记录每一轮的准确率和样本统计
        
        Args:
            round_idx: 当前轮次
            overall_acc: 总体准确率
            per_class_acc: 每个类别的准确率字典 {class_idx: accuracy}
            class_sample_counts: 每个类别的样本数 {class_idx: count}
            total_samples_used: 当前轮使用的总样本数
        """
        self.round_accuracy['round'].append(round_idx)
        self.round_accuracy['overall_accuracy'].append(overall_acc)
        self.round_accuracy['macro_f1'].append(float(macro_f1) if macro_f1 is not None else float('nan'))
        self.round_accuracy['ece'].append(float(ece) if ece is not None else float('nan'))
        self.round_accuracy['avg_nll'].append(float(avg_nll) if avg_nll is not None else float('nan'))
        # balanced accuracy: mean(per-class acc)
        bal_acc = float(np.mean([per_class_acc.get(i, 0.0) for i in range(self.num_classes)])) if self.num_classes > 0 else float('nan')
        self.round_accuracy['balanced_accuracy'].append(bal_acc)

        if per_class_f1 is None:
            per_class_f1 = {i: float('nan') for i in range(self.num_classes)}
        if per_class_recall is None:
            per_class_recall = {i: float('nan') for i in range(self.num_classes)}
        if worst_class_acc is None:
            worst_class_acc = (
                min(per_class_acc.values()) if per_class_acc else float('nan')
            )
        if tail_recall_mean is None:
            tail_recall_mean = _tail_recall_mean_from_counts(
                class_sample_counts or {}, per_class_recall, self.num_classes
            )
        head_recall_mean = _head_recall_mean_from_counts(
            class_sample_counts or {}, per_class_recall, self.num_classes
        )
        self.round_accuracy['worst_class_acc'].append(float(worst_class_acc))
        self.round_accuracy['tail_recall_mean'].append(float(tail_recall_mean))
        self.round_accuracy['head_recall_mean'].append(float(head_recall_mean))
        # 添加样本统计
        if total_samples_used is not None:
            self.round_accuracy['num_samples_used'].append(total_samples_used)
            self.round_accuracy['total_samples'].append(self.total_samples or 0)
            if self.total_samples:
                ratio = total_samples_used / self.total_samples
                self.round_accuracy['sample_usage_ratio'].append(ratio)
            else:
                self.round_accuracy['sample_usage_ratio'].append(0)
        else:
            self.round_accuracy['num_samples_used'].append(0)
            self.round_accuracy['total_samples'].append(self.total_samples or 0)
            self.round_accuracy['sample_usage_ratio'].append(0)
        
        # 记录每个类的准确率和样本数
        for i, name in enumerate(self.class_names):
            acc_key = f'class_{i}_{name}'
            count_key = f'class_{i}_{name}_count'
            
            self.round_accuracy[acc_key].append(per_class_acc.get(i, 0.0))
            
            if class_sample_counts:
                self.round_accuracy[count_key].append(class_sample_counts.get(i, 0))
            else:
                self.round_accuracy[count_key].append(0)
            f1_key = f'f1_{i}_{name}'
            rec_key = f'recall_{i}_{name}'
            self.round_accuracy[f1_key].append(float(per_class_f1.get(i, float('nan'))))
            self.round_accuracy[rec_key].append(float(per_class_recall.get(i, float('nan'))))
        
        # 更新样本统计
        if class_sample_counts and total_samples_used:
            self.update_sample_statistics(round_idx, class_sample_counts, total_samples_used)
        
        # 更新最佳结果（默认用 overall_accuracy；可通过 cfg.TRAINER.COOPAL.BEST_ROUND_METRIC 切换）
        metric = "overall_accuracy"
        if self.cfg is not None:
            try:
                metric = str(getattr(getattr(self.cfg.TRAINER, "COOPAL", object()), "BEST_ROUND_METRIC", "overall_accuracy"))
            except Exception:
                metric = "overall_accuracy"
        metric = metric.lower().strip()
        self.best_metric_name = metric

        if metric == "macro_f1":
            metric_value = float(macro_f1) if macro_f1 is not None else float("-inf")
        elif metric in (
            "balanced_accuracy",
            "balanced_acc",
            "bal_acc",
            "macro_acc",
            "macro_accuracy",
            "macro accuracy",
        ):
            metric_value = float(bal_acc)
        else:
            metric_value = float(overall_acc)

        if metric_value > self.best_accuracy:
            self.best_accuracy = metric_value
            self.best_round = round_idx
            self.best_results = {
                'round': round_idx,
                'overall_accuracy': overall_acc,
                'macro_f1': float(macro_f1) if macro_f1 is not None else float('nan'),
                'balanced_accuracy': float(bal_acc),
                'worst_class_acc': float(worst_class_acc),
                'tail_recall_mean': float(tail_recall_mean),
                'head_recall_mean': float(head_recall_mean),
                'per_class_f1': {int(k): float(v) for k, v in per_class_f1.items()},
                'per_class_recall': {int(k): float(v) for k, v in per_class_recall.items()},
                'best_metric': metric,
                'best_metric_value': float(metric_value),
                'per_class_accuracy': per_class_acc.copy(),
                'class_sample_counts': class_sample_counts.copy() if class_sample_counts else {},
                'total_samples_used': total_samples_used or 0,
                'total_samples': self.total_samples or 0,
                'ece': float(ece) if ece is not None else float('nan'),
                'avg_nll': float(avg_nll) if avg_nll is not None else float('nan'),
            }
    def log_calibration_data(self, round_idx, probs, labels, ece, avg_nll):
        """
        保存某一轮的校准数据，用于之后画 reliability diagram。
        probs: numpy array, shape (N, C), softmax 后的概率
        labels: numpy array, shape (N,)
        """
        self.calibration_data[round_idx] = {
            'probs': np.asarray(probs, dtype=np.float32),
            'labels': np.asarray(labels, dtype=np.int64),
            'ece': float(ece),
            'avg_nll': float(avg_nll),
    }

    def save_round_accuracy(self):
        """保存每轮准确率和样本统计到CSV"""
        # 准备数据，重新组织列的顺序使其更清晰
        ordered_data = {
            'Round': self.round_accuracy['round'],
            'Overall_Accuracy': self.round_accuracy['overall_accuracy'],
            'Macro_F1': self.round_accuracy['macro_f1'],
            'Balanced_Accuracy': self.round_accuracy['balanced_accuracy'],
            'Worst_Class_Acc': self.round_accuracy.get('worst_class_acc', []),
            'Tail_Recall_Mean': self.round_accuracy.get('tail_recall_mean', []),
            'Head_Recall_Mean': self.round_accuracy.get('head_recall_mean', []),
            'ECE': self.round_accuracy['ece'],          # 新增
            'Avg_NLL': self.round_accuracy['avg_nll'],  # 新增
            'Samples_Used': self.round_accuracy['num_samples_used'],
            'Total_Samples': self.round_accuracy['total_samples'],
            'Usage_Ratio': [f"{r:.2%}" for r in self.round_accuracy['sample_usage_ratio']],

        }
        
        # 添加每个类的准确率和数量
        for i, name in enumerate(self.class_names):
            acc_key = f'class_{i}_{name}'
            count_key = f'class_{i}_{name}_count'
            
            ordered_data[f'{name}_Accuracy'] = self.round_accuracy[acc_key]
            ordered_data[f'{name}_Count'] = self.round_accuracy[count_key]
            f1_key = f'f1_{i}_{name}'
            rec_key = f'recall_{i}_{name}'
            if f1_key in self.round_accuracy:
                ordered_data[f'{name}_F1'] = self.round_accuracy[f1_key]
            if rec_key in self.round_accuracy:
                ordered_data[f'{name}_Recall'] = self.round_accuracy[rec_key]
        
        df = pd.DataFrame(ordered_data)
        csv_path = os.path.join(self.log_dir, "round_accuracy_with_statistics.csv")
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Round accuracy with statistics saved to: {csv_path}")
        return csv_path
    
    def save_best_results(self):
        """保存最佳结果到CSV（包含详细的样本统计）- 修正版"""
        if self.best_results is None:
            print("No best results to save!")
            return None
        
        csv_path = os.path.join(self.log_dir, "best_results_with_statistics.csv")

        # --- 第一部分：概览信息 ---
        # 这部分数据结构是 key-value 对，适合做成一个2列的DataFrame
        overview_data = [
            ['Dataset', self.dataset_name],
            ['Best Round', str(self.best_round)],
            ['Overall Accuracy', f"{self.best_results['overall_accuracy']:.4f}"],
            ['Macro F1', f"{self.best_results.get('macro_f1', float('nan')):.4f}"],
            ['Balanced Accuracy', f"{self.best_results.get('balanced_accuracy', float('nan')):.4f}"],
            ['Worst-Class Acc', f"{self.best_results.get('worst_class_acc', float('nan')):.4f}"],
            ['Tail Recall Mean', f"{self.best_results.get('tail_recall_mean', float('nan')):.4f}"],
            ['Head Recall Mean', f"{self.best_results.get('head_recall_mean', float('nan')):.4f}"],
            ['Best Metric', str(self.best_results.get('best_metric', 'overall_accuracy'))],
            ['Best Metric Value', f"{self.best_results.get('best_metric_value', float('nan')):.4f}"],
            ['ECE', f"{self.best_results.get('ece', float('nan')):.4f}"],
            ['Average NLL', f"{self.best_results.get('avg_nll', float('nan')):.4f}"],
            ['Total Samples Used', str(self.best_results['total_samples_used'])],
            ['Total Available Samples', str(self.best_results['total_samples'])]
        ]
        
        usage_ratio = 0
        if self.best_results['total_samples'] > 0:
            usage_ratio = self.best_results['total_samples_used'] / self.best_results['total_samples']
        overview_data.append(['Sample Usage Ratio', f"{usage_ratio:.2%}"])
        
        # 为概览数据创建独立的DataFrame
        overview_df = pd.DataFrame(overview_data, columns=['Metric', 'Value'])
        
        # --- 第二部分：每类的详细统计 ---
        # 这部分数据是一个表格，有自己的5个列
        class_header = ['Class', 'Class Name', 'Accuracy', 'Sample Count', 'Sample Proportion']
        class_rows = []
        
        total_class_samples = sum(self.best_results.get('class_sample_counts', {}).values())
        
        for i, name in enumerate(self.class_names):
            accuracy = self.best_results['per_class_accuracy'].get(i, 0.0)
            count = self.best_results.get('class_sample_counts', {}).get(i, 0)
            
            proportion = 0
            if total_class_samples > 0:
                proportion = count / total_class_samples
            
            class_rows.append([
                f'Class {i}',
                name,
                f"{accuracy:.4f}",
                str(count),
                f"{proportion:.2%}"
            ])
        
        # 添加汇总行
        class_rows.append([
            'Total',
            '',
            f"{self.best_results['overall_accuracy']:.4f}",
            str(total_class_samples),
            '100.00%'
        ])
        
        # 为分类统计数据创建独立的DataFrame
        class_df = pd.DataFrame(class_rows, columns=class_header)
        
        # --- 将两个DataFrame写入同一个CSV文件 ---
        # 1. 首先以写入模式(w)保存第一个DataFrame，这会创建或覆盖文件
        overview_df.to_csv(csv_path, index=False)
        
        # 2. 然后以追加模式(a)打开文件，写入一个空行和第二个DataFrame
        with open(csv_path, 'a', newline='') as f:
            f.write('\n') # 写入一个空行用于分隔
            class_df.to_csv(f, index=False)
        
        print(f"\nBest results with statistics saved to: {csv_path}")
        print(f"Best Round: {self.best_round}")
        print(f"Best Metric: {self.best_results.get('best_metric', 'overall_accuracy')}")
        print(f"Best Metric Value: {self.best_results.get('best_metric_value', self.best_accuracy):.4f}")
        print(f"Samples Used: {self.best_results['total_samples_used']}/{self.best_results['total_samples']} ({usage_ratio:.2%})")
        
        return csv_path
    
    def plot_sample_usage(self):
        """绘制样本使用情况图表"""
        df = pd.DataFrame(self.round_accuracy)
        
        if len(df) == 0:
            print("No data to plot sample usage!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 样本使用量增长曲线
        ax1 = axes[0, 0]
        ax1.plot(df['round'], df['num_samples_used'], 
                marker='o', linewidth=2, markersize=8, 
                color='#2E86AB', label='Samples Used')
        ax1.axhline(y=self.total_samples, color='red', linestyle='--', 
                   label=f'Total Available ({self.total_samples})')
        ax1.fill_between(df['round'], df['num_samples_used'], alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_title('Sample Usage Growth', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率 vs 样本使用比例
        ax2 = axes[0, 1]
        ax2.scatter(df['sample_usage_ratio'] * 100, df['overall_accuracy'],
                   s=100, c=df['round'], cmap='viridis', marker='o')
        ax2.set_xlabel('Sample Usage Ratio (%)', fontsize=12)
        ax2.set_ylabel('Overall Accuracy', fontsize=12)
        ax2.set_title('Accuracy vs Sample Usage', fontsize=14, fontweight='bold')
        
        # 添加颜色条显示轮次
        cbar = plt.colorbar(ax2.scatter(df['sample_usage_ratio'] * 100, 
                                        df['overall_accuracy'],
                                        s=100, c=df['round'], cmap='viridis'), ax=ax2)
        cbar.set_label('Round', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 每类样本数分布（最后一轮）
        ax3 = axes[1, 0]
        if self.best_results and 'class_sample_counts' in self.best_results:
            class_counts = self.best_results['class_sample_counts']
            class_names_short = [name[:15] + '...' if len(name) > 15 else name 
                                for name in self.class_names]
            counts = [class_counts.get(i, 0) for i in range(len(self.class_names))]
            
            bars = ax3.bar(range(len(counts)), counts, color='#A8DADC')
            ax3.set_xticks(range(len(class_names_short)))
            ax3.set_xticklabels(class_names_short, rotation=45, ha='right')
            ax3.set_xlabel('Class', fontsize=12)
            ax3.set_ylabel('Sample Count', fontsize=12)
            ax3.set_title(f'Class Distribution at Best Round (Round {self.best_round})', 
                         fontsize=14, fontweight='bold')
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count)}', ha='center', va='bottom', fontsize=8)
        
        # 4. 效率曲线：准确率增益 vs 样本增量
        ax4 = axes[1, 1]
        if len(df) > 1:
            acc_gains = df['overall_accuracy'].diff()[1:]
            sample_increments = df['num_samples_used'].diff()[1:]
            efficiency = acc_gains / (sample_increments + 1e-6)  # 避免除零
            
            ax4.plot(df['round'][1:], efficiency, 
                    marker='s', linewidth=2, markersize=8, 
                    color='#F1935C', label='Efficiency')
            ax4.set_xlabel('Round', fontsize=12)
            ax4.set_ylabel('Accuracy Gain per Sample', fontsize=12)
            ax4.set_title('Learning Efficiency', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.log_dir, "sample_usage_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Sample usage analysis saved to: {plot_path}")
        plt.close()
    
    def save_loss_data(self):
        """保存loss数据到CSV"""
        df = pd.DataFrame(self.loss_data)
        csv_path = os.path.join(self.log_dir, "training_losses.csv")
        df.to_csv(csv_path, index=False)
        print(f"Loss data saved to: {csv_path}")
        return csv_path
    
    def plot_losses(self):
        """
        简化版平滑曲线绘制 - 直接替换原有的plot_losses方法
        使用滚动平均进行平滑，无需额外依赖
        """
        df = pd.DataFrame(self.loss_data)
        
        if len(df) == 0:
            print("No data to plot!")
            return
        
        # 设置样式
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        rounds = df['round'].unique()
        colors = plt.cm.tab10(range(len(rounds)))
        
        # ============ 平滑函数（使用pandas滚动平均）============
        def smooth_data(data, window=10):
            """使用滚动平均平滑数据"""
            if len(data) < window:
                window = max(3, len(data) // 3)
            return data.rolling(window=window, center=True, min_periods=1).mean()
        
        # ============ 1. VLM Loss ============
        ax1 = axes[0, 0]
        for i, round_idx in enumerate(rounds):
            round_data = df[df['round'] == round_idx].copy()
            round_data = round_data.reset_index(drop=True)
            
            # 原始数据（淡色散点）
            ax1.scatter(round_data.index, round_data['loss_vlm'], 
                    color=colors[i], alpha=0.1, s=3, zorder=1)
            
            # 平滑数据（主曲线）
            smoothed = smooth_data(round_data['loss_vlm'], window=15)
            ax1.plot(round_data.index, smoothed, 
                    label=f'Round {round_idx}', 
                    color=colors[i], 
                    linewidth=2.5, 
                    alpha=0.85,
                    zorder=2)
        
        ax1.set_xlabel('Batch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('VLM Loss', fontsize=12, fontweight='bold')
        ax1.set_title('VLM Loss Over Training (Smoothed)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # ============ 2. MEH Loss ============
        ax2 = axes[0, 1]
        for i, round_idx in enumerate(rounds):
            round_data = df[df['round'] == round_idx].copy()
            
            # 只显示有效的MEH训练数据
            active_data = round_data[round_data['loss_meh'] > 0.01].copy()
            
            if len(active_data) > 0:
                active_data = active_data.reset_index(drop=True)
                
                # 原始数据
                ax2.scatter(active_data.index, active_data['loss_meh'],
                        color=colors[i], alpha=0.1, s=3, zorder=1)
                
                # Round 1使用更强的平滑（因为冷启动噪声大）
                window = 25 if round_idx == 1 else 15
                smoothed = smooth_data(active_data['loss_meh'], window=window)
                
                line_label = f'Round {round_idx}' + (' (cold start)' if round_idx == 1 else '')
                ax2.plot(active_data.index, smoothed,
                        label=line_label, 
                        color=colors[i], 
                        linewidth=2.5, 
                        alpha=0.85,
                        zorder=2)
        
        ax2.set_xlabel('Batch (Active Training Only)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MEH Loss', fontsize=12, fontweight='bold')
        ax2.set_title('MEH Loss Over Training (Smoothed)', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # ============ 3. Total Loss (按epoch聚合) ============
        ax3 = axes[1, 0]
        for i, round_idx in enumerate(rounds):
            round_data = df[df['round'] == round_idx].copy()
            
            # 按epoch聚合，减少噪声
            epoch_data = round_data.groupby('epoch').agg({
                'loss_vlm': 'mean',
                'loss_meh': 'mean'
            }).reset_index()
            
            epoch_data['total'] = epoch_data['loss_vlm'] + epoch_data['loss_meh']
            
            # Epoch级别的数据通常已经比较平滑，只需轻度平滑
            if len(epoch_data) > 3:
                smoothed = smooth_data(epoch_data['total'], window=3)
            else:
                smoothed = epoch_data['total']
            
            # 绘制
            ax3.plot(epoch_data['epoch'], smoothed,
                    label=f'Round {round_idx}', 
                    color=colors[i], 
                    linewidth=2.5, 
                    alpha=0.85,
                    marker='o',
                    markersize=5,
                    markeredgecolor='white',
                    markeredgewidth=1,
                    zorder=2)
            
            # 原始点（小点）
            ax3.scatter(epoch_data['epoch'], epoch_data['total'],
                    color=colors[i], s=20, alpha=0.3, zorder=1)
        
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Total Loss', fontsize=12, fontweight='bold')
        ax3.set_title('Total Loss (VLM + MEH) per Epoch', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # ============ 4. Training Accuracy (按epoch聚合) ============
        ax4 = axes[1, 1]
        for i, round_idx in enumerate(rounds):
            round_data = df[df['round'] == round_idx].copy()
            
            # 按epoch聚合
            epoch_acc = round_data.groupby('epoch')['acc'].mean().reset_index()
            
            # 轻度平滑
            if len(epoch_acc) > 3:
                smoothed = smooth_data(epoch_acc['acc'], window=3)
            else:
                smoothed = epoch_acc['acc']
            
            # 主曲线
            ax4.plot(epoch_acc['epoch'], smoothed,
                    label=f'Round {round_idx}', 
                    color=colors[i], 
                    linewidth=2.5, 
                    alpha=0.85,
                    marker='s',
                    markersize=5,
                    markeredgecolor='white',
                    markeredgewidth=1,
                    zorder=2)
            
            # 原始曲线（虚线，淡色）
            ax4.plot(epoch_acc['epoch'], epoch_acc['acc'],
                    color=colors[i], linewidth=1, alpha=0.2, 
                    linestyle='--', zorder=1)
        
        ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Training Accuracy per Epoch', fontsize=14, fontweight='bold')
        ax4.legend(loc='lower right', fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # 调整y轴范围
        all_acc = df['acc'].values
        if len(all_acc) > 0:
            ax4.set_ylim([max(0, all_acc.min() - 5), min(100, all_acc.max() + 5)])
        
        plt.tight_layout()
        
        # 保存
        plot_path = os.path.join(self.log_dir, "training_losses.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Smoothed loss plots saved to: {plot_path}")
        plt.close()
    def plot_losses_global_view(self):
        """
        新增方法：按全局batch索引绘制loss曲线（跨所有rounds连续）
        这样可以看到整个训练过程的连续趋势
        """
        df = pd.DataFrame(self.loss_data)
        
        if len(df) == 0:
            print("No data to plot!")
            return
        
        # ============ 关键：创建全局batch索引 ============
        # 为每个batch分配一个全局的连续索引
        df['global_batch'] = range(len(df))
        
        # 设置样式
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        rounds = df['round'].unique()
        colors = plt.cm.tab10(range(len(rounds)))
        
        # 平滑函数
        def smooth_data(data, window=20):
            """使用滚动平均平滑数据 - 全局视图用更大的window"""
            if len(data) < window:
                window = max(3, len(data) // 5)
            return data.rolling(window=window, center=True, min_periods=1).mean()
        
        # ============ 1. VLM Loss - 全局连续视图 ============
        ax1 = axes[0, 0]
        
        # 绘制所有数据的连续曲线
        x_all = df['global_batch'].values
        y_all = df['loss_vlm'].values
        
        # 原始数据（淡色）
        ax1.scatter(x_all, y_all, color='lightgray', s=2, alpha=0.3, zorder=1, label='Raw data')
        
        # 平滑后的全局曲线（主曲线）
        y_smooth = smooth_data(df['loss_vlm'], window=30)
        ax1.plot(x_all, y_smooth, 
                color='#2E86AB', 
                linewidth=3, 
                alpha=0.9,
                zorder=3,
                label='Smoothed (global)')
        
        # 添加round分隔线和标注
        round_boundaries = []
        for i, round_idx in enumerate(rounds):
            round_data = df[df['round'] == round_idx]
            start_batch = round_data['global_batch'].iloc[0]
            end_batch = round_data['global_batch'].iloc[-1]
            
            # 记录边界
            if i > 0:  # 不在第一个round前画线
                round_boundaries.append(start_batch)
                ax1.axvline(x=start_batch, color='red', linestyle='--', 
                        alpha=0.5, linewidth=1.5, zorder=2)
            
            # 在每个round区域中间标注
            mid_point = (start_batch + end_batch) / 2
            ax1.text(mid_point, ax1.get_ylim()[1] * 0.95, 
                    f'R{round_idx}',
                    ha='center', va='top',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor=colors[i], alpha=0.3))
        
        ax1.set_xlabel('Global Batch Index (Across All Rounds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('VLM Loss', fontsize=12, fontweight='bold')
        ax1.set_title('VLM Loss - Continuous Training View', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # ============ 2. MEH Loss - 全局连续视图（特殊处理） ============
        ax2 = axes[0, 1]
        
        # 只取有效的MEH训练数据
        meh_active = df[df['loss_meh'] > 0.01].copy()
        
        if len(meh_active) > 0:
            # 为active数据也创建连续索引
            meh_active = meh_active.reset_index(drop=True)
            meh_active['meh_continuous_idx'] = range(len(meh_active))
            
            x_meh = meh_active['meh_continuous_idx'].values
            y_meh = meh_active['loss_meh'].values
            
            # 原始数据
            ax2.scatter(x_meh, y_meh, color='lightcoral', s=2, alpha=0.3, zorder=1, label='Raw data')
            
            # 检测是否有Round 1的冷启动阶段
            round1_mask = meh_active['round'] == 1
            if round1_mask.any():
                # Round 1数据单独平滑（用更强的平滑）
                round1_data = meh_active[round1_mask].copy()
                x_r1 = round1_data['meh_continuous_idx'].values
                y_r1 = round1_data['loss_meh'].values
                y_r1_smooth = smooth_data(pd.Series(y_r1), window=40)
                
                ax2.plot(x_r1, y_r1_smooth,
                        color='#FF6B6B',
                        linewidth=3,
                        alpha=0.9,
                        zorder=3,
                        label='Round 1 (cold start)')
                
                # 其他rounds
                other_rounds = meh_active[~round1_mask].copy()
                if len(other_rounds) > 0:
                    x_other = other_rounds['meh_continuous_idx'].values
                    y_other = other_rounds['loss_meh'].values
                    y_other_smooth = smooth_data(pd.Series(y_other), window=20)
                    
                    ax2.plot(x_other, y_other_smooth,
                            color='#4ECDC4',
                            linewidth=3,
                            alpha=0.9,
                            zorder=3,
                            label='Round 2+ (stable)')
            else:
                # 全部数据一起平滑
                y_smooth = smooth_data(meh_active['loss_meh'], window=30)
                ax2.plot(x_meh, y_smooth,
                        color='#F1935C',
                        linewidth=3,
                        alpha=0.9,
                        zorder=3,
                        label='Smoothed (global)')
            
            # 标注round边界
            for round_idx in meh_active['round'].unique()[1:]:  # 跳过第一个
                round_start = meh_active[meh_active['round'] == round_idx]['meh_continuous_idx'].iloc[0]
                ax2.axvline(x=round_start, color='red', linestyle='--', 
                        alpha=0.5, linewidth=1.5, zorder=2)
        
        ax2.set_xlabel('MEH Training Step (Active Batches Only)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MEH Loss', fontsize=12, fontweight='bold')
        ax2.set_title('MEH Loss - Continuous Training View', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # ============ 3. Total Loss - 全局视图 ============
        ax3 = axes[1, 0]
        
        df['total_loss_calc'] = df['loss_vlm'] + df['loss_meh']
        
        x_all = df['global_batch'].values
        y_total = df['total_loss_calc'].values
        
        # 原始数据
        ax3.scatter(x_all, y_total, color='lightgray', s=2, alpha=0.3, zorder=1)
        
        # 平滑曲线
        y_total_smooth = smooth_data(df['total_loss_calc'], window=30)
        ax3.plot(x_all, y_total_smooth,
                color='#95E1D3',
                linewidth=3,
                alpha=0.9,
                zorder=3,
                label='Total Loss (smoothed)')
        
        # 可选：分别显示VLM和MEH的贡献（堆叠面积图）
        y_vlm_smooth = smooth_data(df['loss_vlm'], window=30)
        y_meh_smooth = smooth_data(df['loss_meh'], window=30)
        
        ax3.fill_between(x_all, 0, y_vlm_smooth, 
                        alpha=0.3, color='#2E86AB', label='VLM component')
        ax3.fill_between(x_all, y_vlm_smooth, y_vlm_smooth + y_meh_smooth,
                        alpha=0.3, color='#F38181', label='MEH component')
        
        # Round分隔线
        for boundary in round_boundaries:
            ax3.axvline(x=boundary, color='red', linestyle='--', 
                    alpha=0.5, linewidth=1.5, zorder=2)
        
        # Round标注
        for i, round_idx in enumerate(rounds):
            round_data = df[df['round'] == round_idx]
            mid_point = (round_data['global_batch'].iloc[0] + round_data['global_batch'].iloc[-1]) / 2
            ax3.text(mid_point, ax3.get_ylim()[1] * 0.95,
                    f'R{round_idx}',
                    ha='center', va='top',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3',
                            facecolor=colors[i], alpha=0.3))
        
        ax3.set_xlabel('Global Batch Index', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Total Loss', fontsize=12, fontweight='bold')
        ax3.set_title('Total Loss - Continuous Training View', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # ============ 4. Training Accuracy - 全局视图 ============
        ax4 = axes[1, 1]
        
        x_all = df['global_batch'].values
        y_acc = df['acc'].values
        
        # 原始数据（淡色）
        ax4.scatter(x_all, y_acc, color='lightgray', s=2, alpha=0.3, zorder=1)
        
        # 平滑曲线
        y_acc_smooth = smooth_data(df['acc'], window=30)
        ax4.plot(x_all, y_acc_smooth,
                color='#38A3A5',
                linewidth=3,
                alpha=0.9,
                zorder=3,
                label='Accuracy (smoothed)')
        
        # Round分隔线和标注
        for i, round_idx in enumerate(rounds):
            round_data = df[df['round'] == round_idx]
            start_batch = round_data['global_batch'].iloc[0]
            
            if i > 0:
                ax4.axvline(x=start_batch, color='red', linestyle='--',
                        alpha=0.5, linewidth=1.5, zorder=2)
            
            mid_point = (round_data['global_batch'].iloc[0] + round_data['global_batch'].iloc[-1]) / 2
            
            # 计算该round的平均accuracy用于标注
            round_acc = round_data['acc'].mean()
            ax4.text(mid_point, ax4.get_ylim()[0] + (ax4.get_ylim()[1] - ax4.get_ylim()[0]) * 0.05,
                    f'R{round_idx}\n{round_acc:.1f}%',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3',
                            facecolor=colors[i], alpha=0.3))
        
        ax4.set_xlabel('Global Batch Index', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Training Accuracy - Continuous View', fontsize=14, fontweight='bold')
        ax4.legend(loc='lower right', fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # 调整y轴
        if len(y_acc) > 0:
            ax4.set_ylim([max(0, y_acc.min() - 5), min(100, y_acc.max() + 5)])
        
        plt.tight_layout()
        
        # 保存
        plot_path = os.path.join(self.log_dir, "training_losses_global_view.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Global view plots saved to: {plot_path}")
        plt.close()
        
        # ============ 额外：生成统计信息 ============
        self._print_global_training_stats(df, rounds)


    def _print_global_training_stats(self, df, rounds):
        """打印全局训练统计信息"""
        print("\n" + "="*70)
        print("GLOBAL TRAINING STATISTICS")
        print("="*70)
        
        total_batches = len(df)
        print(f"Total training batches: {total_batches}")
        
        for round_idx in rounds:
            round_data = df[df['round'] == round_idx]
            n_batches = len(round_data)
            n_epochs = round_data['epoch'].nunique()
            
            vlm_loss_start = round_data['loss_vlm'].iloc[0]
            vlm_loss_end = round_data['loss_vlm'].iloc[-1]
            vlm_improvement = ((vlm_loss_start - vlm_loss_end) / vlm_loss_start * 100)
            
            meh_active = round_data[round_data['loss_meh'] > 0.01]
            meh_status = f"{len(meh_active)} active batches" if len(meh_active) > 0 else "inactive"
            
            acc_start = round_data['acc'].iloc[0]
            acc_end = round_data['acc'].iloc[-1]
            acc_gain = acc_end - acc_start
            
            print(f"\nRound {round_idx}:")
            print(f"  Batches: {n_batches} ({n_batches/total_batches*100:.1f}% of total)")
            print(f"  Epochs: {n_epochs}")
            print(f"  VLM Loss: {vlm_loss_start:.4f} → {vlm_loss_end:.4f} ({vlm_improvement:+.1f}%)")
            print(f"  MEH Status: {meh_status}")
            print(f"  Accuracy: {acc_start:.2f}% → {acc_end:.2f}% ({acc_gain:+.2f}%)")
        
        print("\n" + "="*70 + "\n")
    def plot_round_accuracy(self):
        """绘制每轮准确率对比图（保持原有功能）"""
        df = pd.DataFrame(self.round_accuracy)
        
        if len(df) == 0:
            print("No accuracy data to plot!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Overall Accuracy趋势
        ax1 = axes[0]
        ax1.plot(df['round'], df['overall_accuracy'], 
                marker='o', linewidth=2, markersize=8, 
                color='#2E86AB', label='Overall Accuracy')
        ax1.fill_between(df['round'], df['overall_accuracy'], 
                         alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('Active Learning Round', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Overall Accuracy Across Rounds', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 标注最佳点
        best_idx = df['overall_accuracy'].idxmax()
        ax1.scatter(df.loc[best_idx, 'round'], 
                   df.loc[best_idx, 'overall_accuracy'],
                   color='red', s=200, marker='*', 
                   zorder=5, label='Best')
        ax1.annotate(f"Best: {df.loc[best_idx, 'overall_accuracy']:.4f}",
                    xy=(df.loc[best_idx, 'round'], 
                        df.loc[best_idx, 'overall_accuracy']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, color='red',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor='yellow', alpha=0.7))
        
        # 2. Per-class Accuracy热力图
        ax2 = axes[1]
        class_columns = [col for col in df.columns if col.startswith('class_') and not col.endswith('_count')]
        class_acc_data = df[class_columns].T
        class_labels = [col.split('_', 2)[2] for col in class_columns]
        
        im = ax2.imshow(class_acc_data, aspect='auto', cmap='YlOrRd')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['round'])
        ax2.set_yticks(range(len(class_labels)))
        ax2.set_yticklabels(class_labels, fontsize=9)
        ax2.set_xlabel('Active Learning Round', fontsize=12)
        ax2.set_ylabel('Class', fontsize=12)
        ax2.set_title('Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Accuracy', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.log_dir, "round_accuracy.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy plots saved to: {plot_path}")
        plt.close()
    def plot_calibration_diagrams(self, n_bins: int = 15):
        """
        根据 self.calibration_data 画校准图。
        每个 round 一个子图，横轴是 confidence，纵轴是 accuracy，
        背景蓝色柱子是样本占比，右上角显示 ECE / NLL。
        """
        if not self.calibration_data:
            print("No calibration data to plot calibration diagrams!")
            return

        rounds = sorted(self.calibration_data.keys())
        n = len(rounds)

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, round_idx in zip(axes, rounds):
            data = self.calibration_data[round_idx]
            probs = data["probs"]     # (N, C)
            labels = data["labels"]   # (N,)

            # 置信度和预测
            confidences = probs.max(axis=1)          # (N,)
            preds = probs.argmax(axis=1)            # (N,)
            accuracies = (preds == labels).astype(np.float32)

            bins = np.linspace(0.0, 1.0, n_bins + 1)
            bin_centers = []
            bin_acc = []
            bin_conf = []
            bin_frac = []

            for i in range(n_bins):
                if i == 0:
                    mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
                else:
                    mask = (confidences >  bins[i]) & (confidences <= bins[i + 1])

                center = 0.5 * (bins[i] + bins[i + 1])

                if mask.any():
                    conf_bin = confidences[mask]
                    acc_bin = accuracies[mask]
                    bin_conf.append(conf_bin.mean())
                    bin_acc.append(acc_bin.mean())
                    bin_frac.append(mask.mean())   # 占比
                else:
                    bin_conf.append(center)
                    bin_acc.append(0.0)
                    bin_frac.append(0.0)

                bin_centers.append(center)

            bin_centers = np.array(bin_centers)
            bin_acc = np.array(bin_acc)
            bin_conf = np.array(bin_conf)
            bin_frac = np.array(bin_frac)

            # 左轴：样本占比直方图
            width = 1.0 / n_bins
            ax.bar(
                bin_centers,
                bin_frac,
                width=width,
                alpha=0.4,
                color="#4C72B0",
                edgecolor="k",
                label="Data fraction",
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence")
            if ax is axes[0]:
                ax.set_ylabel("Fraction of samples")

            # 右轴：accuracy 曲线 + 理想对角线
            ax2 = ax.twinx()
            ax2.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            ax2.plot(bin_conf, bin_acc, marker="o", color="#4C72B0", linewidth=2)
            ax2.set_ylim(0, 1.0)
            if ax is axes[0]:
                ax2.set_ylabel("Accuracy")

            ax.set_title(f"Round {round_idx}")

            # 右上角文本框：ECE / NLL
            text = f"ECE={data['ece']:.3f}\nNLL={data['avg_nll']:.3f}"
            ax2.text(
                0.05,
                0.95,
                text,
                transform=ax2.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        save_path = os.path.join(self.log_dir, "calibration_diagrams.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Calibration diagrams saved to: {save_path}")
        plt.close()

    def save_config_snapshot(self):
        """将 yacs 配置写入 YAML，便于事后对齐实验与计算指标。"""
        if self.cfg is None:
            return None
        path = os.path.join(self.log_dir, "config_snapshot.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.cfg.dump())
        print(f"Config snapshot saved to: {path}")
        return path

    def save_run_summary_json(self):
        """
        单一 JSON：每轮 accuracy / ECE / NLL / 各类准确率与标注计数 + best + 产物路径，
        便于外部脚本（Python/R）聚合论文表格与画图。
        """
        n_r = len(self.round_accuracy.get("round", []))
        rounds_out = []
        for i in range(n_r):
            mf1_list = self.round_accuracy.get("macro_f1", [])
            mf1_val = float(mf1_list[i]) if i < len(mf1_list) else float('nan')
            bal_list = self.round_accuracy.get("balanced_accuracy", [])
            bal_i = float(bal_list[i]) if i < len(bal_list) else float("nan")
            wc_list = self.round_accuracy.get("worst_class_acc", [])
            wc_i = float(wc_list[i]) if i < len(wc_list) else float("nan")
            tr_list = self.round_accuracy.get("tail_recall_mean", [])
            tr_i = float(tr_list[i]) if i < len(tr_list) else float("nan")
            hr_list = self.round_accuracy.get("head_recall_mean", [])
            hr_i = float(hr_list[i]) if i < len(hr_list) else float("nan")
            rec = {
                "round": int(self.round_accuracy["round"][i]),
                "overall_accuracy": float(self.round_accuracy["overall_accuracy"][i]),
                "macro_f1": mf1_val,
                "balanced_accuracy": bal_i,
                "worst_class_acc": wc_i,
                "tail_recall_mean": tr_i,
                "head_recall_mean": hr_i,
                "ece": float(self.round_accuracy["ece"][i]),
                "avg_nll": float(self.round_accuracy["avg_nll"][i]),
                "num_samples_used": int(self.round_accuracy["num_samples_used"][i]),
                "total_samples": int(self.round_accuracy["total_samples"][i]),
                "sample_usage_ratio": float(self.round_accuracy["sample_usage_ratio"][i]),
                "per_class_accuracy": {},
                "per_class_f1": {},
                "per_class_recall": {},
                "per_class_labeled_count": {},
            }
            for j, name in enumerate(self.class_names):
                acc_key = f"class_{j}_{name}"
                cnt_key = f"{acc_key}_count"
                rec["per_class_accuracy"][name] = float(self.round_accuracy[acc_key][i])
                rec["per_class_labeled_count"][name] = int(self.round_accuracy[cnt_key][i])
                f1_key = f"f1_{j}_{name}"
                r_key = f"recall_{j}_{name}"
                if f1_key in self.round_accuracy and i < len(self.round_accuracy[f1_key]):
                    rec["per_class_f1"][name] = float(self.round_accuracy[f1_key][i])
                if r_key in self.round_accuracy and i < len(self.round_accuracy[r_key]):
                    rec["per_class_recall"][name] = float(self.round_accuracy[r_key][i])
            rounds_out.append(rec)

        best_out = None
        if self.best_results is not None:
            br = self.best_results.copy()
            br["per_class_accuracy"] = {
                str(k): float(v) for k, v in br["per_class_accuracy"].items()
            }
            br["class_sample_counts"] = {
                str(k): int(v) for k, v in br.get("class_sample_counts", {}).items()
            }
            if br.get("per_class_f1"):
                br["per_class_f1"] = {
                    str(k): float(v) for k, v in br["per_class_f1"].items()
                }
            if br.get("per_class_recall"):
                br["per_class_recall"] = {
                    str(k): float(v) for k, v in br["per_class_recall"].items()
                }
            br["overall_accuracy"] = float(br["overall_accuracy"])
            br["macro_f1"] = float(br.get("macro_f1", float("nan")))
            br["balanced_accuracy"] = float(br.get("balanced_accuracy", float("nan")))
            br["worst_class_acc"] = float(br.get("worst_class_acc", float("nan")))
            br["tail_recall_mean"] = float(br.get("tail_recall_mean", float("nan")))
            if "head_recall_mean" not in br or br.get("head_recall_mean") != br.get("head_recall_mean"):
                br["head_recall_mean"] = _head_recall_mean_from_counts(
                    {int(k): int(v) for k, v in br.get("class_sample_counts", {}).items()},
                    {int(k): float(v) for k, v in br.get("per_class_recall", {}).items()},
                    self.num_classes,
                )
            else:
                br["head_recall_mean"] = float(br.get("head_recall_mean", float("nan")))
            br["ece"] = float(br.get("ece", float("nan")))
            br["avg_nll"] = float(br.get("avg_nll", float("nan")))
            br["total_samples_used"] = int(br.get("total_samples_used", 0))
            br["total_samples"] = int(br.get("total_samples", 0))
            best_out = br

        summary = {
            "dataset": self.dataset_name,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "total_unlabeled_pool_samples": self.total_samples,
            "best_metric": str(self.best_metric_name),
            "best_round": int(self.best_round),
            "best_metric_value": float(self.best_accuracy) if self.best_accuracy >= 0 else None,
            "rounds": rounds_out,
            "best": best_out,
            "artifacts": {
                "training_logs_dir": "training_logs",
                "round_accuracy_csv": os.path.join("training_logs", "round_accuracy_with_statistics.csv"),
                "best_results_csv": os.path.join("training_logs", "best_results_with_statistics.csv"),
                "training_losses_csv": os.path.join("training_logs", "training_losses.csv"),
                "run_summary_json": os.path.join("training_logs", "run_summary.json"),
                "config_snapshot_yaml": os.path.join("training_logs", "config_snapshot.yaml"),
                "final_round_predictions_csv": "final_round_predictions.csv",
                "calibration_npz_pattern": os.path.join("training_logs", "calibration_round_{round}.npz"),
            },
        }

        path = os.path.join(self.log_dir, "run_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Run summary (JSON) saved to: {path}")
        return path

    def maybe_append_experiment_registry(self):
        """
        训练结束后把本次 run 追加到统一台账（CSV + Markdown）。

        默认开启：cfg.TRAINER.COOPAL.EXPERIMENT_REGISTRY.AUTO_APPEND=True
        可用环境变量关闭：EXPERIMENT_REGISTRY_AUTO_APPEND=0/false/no
        """
        try:
            if self.cfg is None:
                return
            if not hasattr(self.cfg, "TRAINER"):
                return
            if not hasattr(self.cfg.TRAINER, "COOPAL"):
                return
            coop = self.cfg.TRAINER.COOPAL
            if not hasattr(coop, "EXPERIMENT_REGISTRY"):
                return

            env_off = os.environ.get("EXPERIMENT_REGISTRY_AUTO_APPEND", "").strip().lower()
            if env_off in {"0", "false", "no", "off"}:
                return

            reg = coop.EXPERIMENT_REGISTRY
            if not bool(getattr(reg, "AUTO_APPEND", False)):
                return

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            tool = os.path.join(project_root, "tools", "log_experiment_run.py")
            if not os.path.exists(tool):
                print(f"[WARN] experiment registry tool missing: {tool}")
                return

            csv_path = str(getattr(reg, "CSV_PATH", "") or "").strip()
            md_path = str(getattr(reg, "MD_PATH", "") or "").strip()
            if csv_path and not os.path.isabs(csv_path):
                csv_path = os.path.join(project_root, csv_path)
            if md_path and not os.path.isabs(md_path):
                md_path = os.path.join(project_root, md_path)

            gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

            name = str(getattr(reg, "NAME", "") or "").strip()
            notes = str(getattr(reg, "NOTES", "") or "").strip()

            cmd = [sys.executable, tool, "--run-dir", self.output_dir, "--gpu", gpu]
            if name:
                cmd += ["--experiment-name", name]
            if notes:
                cmd += ["--notes", notes]
            if csv_path:
                cmd += ["--registry-csv", csv_path]
            if md_path:
                cmd += ["--registry-md", md_path]

            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"[WARN] failed to append experiment registry: {e}")

    def save_calibration_npz(self):
        """导出每轮测试集 softmax 概率与标签，便于离线重算 ECE / Brier / AUROC 等。"""
        if not self.calibration_data:
            return []
        paths = []
        for round_idx in sorted(self.calibration_data.keys()):
            data = self.calibration_data[round_idx]
            path = os.path.join(self.log_dir, f"calibration_round_{int(round_idx)}.npz")
            np.savez_compressed(
                path,
                probs=data["probs"],
                labels=data["labels"],
                ece=np.float32(data["ece"]),
                avg_nll=np.float32(data["avg_nll"]),
            )
            paths.append(path)
        if paths:
            print(f"Calibration arrays saved: {len(paths)} file(s) under {self.log_dir}")
        return paths

    def finalize(self):
        """训练结束时调用，保存所有数据和图表"""
        print("\n" + "="*60)
        print("Finalizing Training Logs...")
        print("="*60)

        self.save_config_snapshot()
        
        # 保存所有CSV文件
        self.save_loss_data()
        self.save_round_accuracy()
        self.save_best_results()
        self.save_calibration_npz()
        self.save_run_summary_json()
        self.maybe_append_experiment_registry()
        
        # 生成所有图表
        self.plot_losses()
        self.plot_round_accuracy()
        self.plot_sample_usage()  # 新增的样本使用情况图表
        self.plot_losses_global_view()    # 全局视图，看整体趋势
        self.plot_calibration_diagrams()   # ✅ 新增：校准图

        print("\nAll logs and visualizations saved successfully!")
        print(f"Location: {self.log_dir}")
        print("="*60 + "\n")