import copy
from pathlib import Path
from random import sample
import time 
import os
import os.path as osp
import numpy as np
import json 
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score, confusion_matrix

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.prompt_templates import biomedcoop_template_at
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from ..active_learning.pcb import PCB
from ..active_learning.badge import BADGE
from ..active_learning.coreset import Coreset
from ..active_learning.entropy import Entropy

from ..active_learning.MEH import MEH_Selector


from ..active_learning.MEH import meh_loss_v2_alternative
from ..active_learning.MEH import meh_loss_v2_log



from ..active_learning.MEH import VLM_EH_V2


from ..training_logger import TrainingLogger, _tail_recall_mean_from_counts, _head_recall_mean_from_counts
from .. import al_resume
from ..checkpoint_prune import (
    list_epoch_checkpoints,
    prune_output_dir,
    purge_all_checkpoints,
    _sync_checkpoint_pointer,
)
from dassl.evaluation.evaluator_per_class import ClassificationPerClass
from ..utils.tsne_visualizer import TSNEVisualizer

 
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from dassl.evaluation.evaluator_per_class import ClassificationPerClass


# ====== Metrics Utils: ECE & NLL ======
import torch


#要记得看probs的输入是什么
@torch.no_grad()
def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    probs: (N, C) 经过 softmax 的概率
    labels: (N,) 真实类别 (long)
    n_bins: 分箱个数
    return: 标量 ECE（float）
    """
    device = probs.device
    confidences, predictions = probs.max(dim=1)                  # (N,), (N,)
    accuracies = predictions.eq(labels).float()                  # (N,)
    bins = torch.linspace(0, 1, n_bins + 1, device=device)       # [0, 1] 线性分箱

    ece = torch.zeros((), device=device)
    N = probs.size(0)
    for i in range(n_bins):
        # 第一个箱子左闭右闭，后续左开右闭，避免边界重复计数
        if i == 0:
            in_bin = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            in_bin = (confidences >  bins[i]) & (confidences <= bins[i + 1])

        if in_bin.any():
            prop_in_bin = in_bin.float().mean()                  # 该箱样本占比
            avg_conf = confidences[in_bin].mean()                # 该箱平均置信度
            avg_acc  = accuracies[in_bin].mean()                 # 该箱平均准确率
            ece += torch.abs(avg_conf - avg_acc) * prop_in_bin

    return float(ece.item())

#############这个输入的是温度缩放的logit，全部叠加的
@torch.no_grad()
def compute_avg_nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    logits: (N, C) 未经 softmax
    labels: (N,)
    return: 全量样本的平均 NLL
    """
    log_probs = torch.log_softmax(logits, dim=1)
    # sum 再除以 N，避免多 batch 计算时的均值偏差
    nll_sum = torch.nn.functional.nll_loss(log_probs, labels, reduction='sum')
    return float((nll_sum / logits.size(0)).item())


##########建议使用ae 和amp，可以适当增加batch_size

class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts,tokenized_prompts):

        x = self.model.encode_text(prompts,True,tokenized_prompts)

        return x
################
# ncls有点不一样，一个是读取classnames的长度，一个是embedding.size(0)


##########
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDCOOP.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDCOOP.CTX_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

   
       

        classnames = [name.replace("_", " ") for name in classnames]
        prompt_prefix = " ".join(["X"] * n_ctx)
        ############################################## 这里出现一个问题，分词器不一样
        
        if cfg.TRAINER.COOPAL.ASPATH:
            with open(f"cupl/descriptors_{cfg.TRAINER.COOPAL.ASPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
                
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(self.tokenizer(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
                    
        elif cfg.TRAINER.COOPAL.AEPATH:
            with open(f"cupl/descriptors_{cfg.TRAINER.COOPAL.AEPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())

            
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(self.tokenizer(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
        #name_lens 的核心作用是为了在构建最终的 prompt embedding 时，能够精确定位和切分不同的部分。
        #在end的时候不用，在middle和front的时候要用到           
        else:
            name_lens = [len(self.tokenizer(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        #######################################拼接成功
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])  # (n_cls, n_tkn)
        ###########################################  创建embedding
        # Also create frozen CLIP  
        biomedclip_model_temp,_ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        biomedclip_model_temp = biomedclip_model_temp.float().eval().cuda()
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = biomedclip_model_temp.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []

            for i in range(cfg.TRAINER.BIOMEDCOOP.N_PROMPTS):########调试时候看看什么参数
                x_tokenized = torch.cat([self.tokenizer(biomedcoop_template_at(classname, i)) for classname in classnames])
                text_features = biomedclip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)
        
        
        
        

        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")


        


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = embedding.size(0)
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION


             ############################################# 初始化文本上下文
        if ctx_init and n_ctx==4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.BIOMEDCOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)#####维度不对
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            
            
        #########现在放在这里是正好
        self.ctx = nn.Parameter(ctx_vectors)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def forward(self):

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model,desc_file=None):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = biomedclip_model.visual
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH

        self.n_class_desc = []
        self.n_cls = len(classnames)
        ################################################################################################
        if desc_file is not None:
            with open(f"cupl/descriptors_{desc_file}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
            classnames = [name.replace("_", " ") for name in classnames]
            for name in classnames:
                name = name.lower()
                self.n_class_desc.append(len(desc_dict[name]))
        
        ###################################################################################################
        if hasattr(self.image_encoder, 'head'):
            # 这是我们之前处理的 open_clip/timm 模型的逻辑
            self._fdim = self.image_encoder.head[-1].out_features
        elif hasattr(self.image_encoder, 'output_dim'):
            # 原始 OpenAI CLIP 的 VisionTransformer 有一个 output_dim 属性
            self._fdim = self.image_encoder.output_dim
        else:
            # 如果以上都不匹配，抛出错误
            raise AttributeError("无法从 image_encoder 中自动确定特征维度 fdim。")
        print(f"成功识别图像编码器，特征维度 fdim 设置为: {self._fdim}")   

        # Optional output-end adapter to fine-tune without unfreezing CLIP
        self.use_output_adapter = getattr(cfg.TRAINER.COOPAL, "OUTPUT_ADAPTER", False)
        self.output_adapter = None
        if self.use_output_adapter:
            self.output_adapter = nn.Linear(self._fdim, self._fdim, bias=False)
            with torch.no_grad():
                self.output_adapter.weight.copy_(torch.eye(self._fdim))


    @property
    def fdim(self):
        """
        这个 @property 装饰器让我们可以像访问普通属性一样调用这个方法，
        即通过 self.model.fdim 就能得到返回值。
        这是 dassl 框架的标准做法。
        """
        return self._fdim

    def _encode_text_features_norm(self):
        """(n_cls, dim)，L2 按行归一化。"""
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_batch_size = 128
        text_features_list = []
        num_prompts = prompts.size(0)
        for i in range(0, num_prompts, text_batch_size):
            p_batch = prompts[i : i + text_batch_size]
            tp_batch = tokenized_prompts[i : i + text_batch_size]
            features_batch = self.text_encoder(p_batch, tp_batch)
            text_features_list.append(features_batch)
        text_features = torch.cat(text_features_list, dim=0)
        if self.cfg.TRAINER.COOPAL.AEPATH:
            tmp = []
            start = 0
            for n in self.n_class_desc:
                tmp.append(text_features[start : start + n].mean(dim=0))
                start += n
            text_features = torch.stack(tmp)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return text_features

    def forward(self, image, get_feature=False, return_branch_logits=False):
        z = self.image_encoder(image.type(self.dtype))
        if self.use_output_adapter and self.output_adapter is not None:
            z = z + self.output_adapter(z)

        text_features = self._encode_text_features_norm()
        logit_scale = self.logit_scale.exp()
        image_features = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        original_logits = image_features @ text_features.t()
        logits = logit_scale * original_logits
        if self.cfg.TRAINER.COOPAL.ASPATH:
            tmp = []
            tmp_original = []
            start = 0
            for n in self.n_class_desc:
                tmp.append(torch.sum(logits[:, start : start + n], dim=1) / n)
                tmp_original.append(torch.sum(original_logits[:, start : start + n], dim=1) / n)
                start += n
            logits = torch.stack(tmp, dim=1)
            original_logits = torch.stack(tmp_original, dim=1)
        self.original_logits = original_logits
        if get_feature:
            return logits, image_features
        return logits



@TRAINER_REGISTRY.register()
class BiomedCLIP(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.acc = []
        print("Initializing with Per-Class Accuracy Evaluator.")
        self.evaluator = ClassificationPerClass(cfg, lab2cname=self.dm.lab2cname)

        total_samples = len(self.dm.dataset.train_x) if hasattr(self.dm.dataset, 'train_x') else None
        
        self.logger = TrainingLogger(
            output_dir=cfg.OUTPUT_DIR,
            dataset_name=cfg.DATASET.NAME,
            num_classes=self.dm.num_classes,
            class_names=list(self.dm.lab2cname.values()),
            total_samples=total_samples,
            cfg=cfg
        )

    def test(self, split=None):
        """
        重写测试流程，包含样本统计功能
        """
        self.set_model_mode("eval")
        self.evaluator.reset()
        
        if split is None:
            split = self.cfg.TEST.SPLIT
        
        data_loader = self.test_loader
        
        current_round = getattr(self, 'current_round', 'N/A')
        total_rounds = getattr(self, 'total_rounds', 'N/A')

        print(f"\n--- Testing for Round {current_round} ---")

        # 用于存储CSV文件所需的数据列
        image_paths = []
        true_labels = []
        predicted_labels = []
        all_probabilities = []
        all_logits = []

        # 遍历测试数据集
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            with torch.no_grad():
                output = self.model(input)
            
            self.evaluator.process(output, label)

            # 收集详细数据
            if 'impath' in batch:
                image_paths.extend(batch['impath'])
            
            pred = output.max(1)[1]
            true_labels.extend(label.cpu().numpy())
            predicted_labels.extend(pred.cpu().numpy())
            
            probabilities = F.softmax(output, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())
            all_logits.append(output.cpu())

        # 获取评估结果
        results = self.evaluator.evaluate()
        
        #计算ece 和 nll
                # ====== 新增：计算 ECE 与 平均 NLL ======
        if len(all_probabilities) > 0:
            probs_tensor = torch.from_numpy(np.stack(all_probabilities, axis=0))   # (N, C)
            labels_tensor = torch.from_numpy(np.array(true_labels, dtype=np.int64))# (N,)
            ece_value = compute_ece(probs_tensor, labels_tensor, n_bins=15)
        else:
            ece_value = float('nan')

        if len(all_logits) > 0:
            logits_tensor = torch.cat(all_logits, dim=0)                            # (N, C)
            labels_tensor = torch.from_numpy(np.array(true_labels, dtype=np.int64)) # (N,)
            avg_nll = compute_avg_nll_from_logits(logits_tensor, labels_tensor)
        else:
            avg_nll = float('nan')

        print(f"Calibration (ECE): {ece_value:.6f}")
        print(f"Average NLL:       {avg_nll:.6f}")




        print("\n[DEBUG] Available keys in results:", list(results.keys()))
        
        if hasattr(self, 'logger') and current_round != 'N/A':
            overall_acc = results.get("accuracy", 0.0)
            
            # 获取每个类别的准确率
            per_class_acc = {}
            
            # 手动计算每个类的准确率和样本数（从预测结果计算）
            from collections import defaultdict
            correct_per_class = defaultdict(int)
            total_per_class = defaultdict(int)
            
            for true_label, pred_label in zip(true_labels, predicted_labels):
                total_per_class[true_label] += 1
                if true_label == pred_label:
                    correct_per_class[true_label] += 1
            
            # 计算每类准确率
            for class_idx in range(self.dm.num_classes):
                if total_per_class[class_idx] > 0:
                    per_class_acc[class_idx] = correct_per_class[class_idx] / total_per_class[class_idx]
                else:
                    per_class_acc[class_idx] = 0.0
            
            # 获取当前训练集的样本统计
            class_sample_counts = {}
            total_samples_used = 0
            
            if hasattr(self, 'dm') and hasattr(self.dm.dataset, 'train_x'):
                # 统计训练集中每个类的样本数
                for item in self.dm.dataset.train_x:
                    class_idx = item.label
                    class_sample_counts[class_idx] = class_sample_counts.get(class_idx, 0) + 1
                    total_samples_used += 1
            
            # 计算 macro_f1
            if len(true_labels) > 0:
                macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
                lab_ids = list(range(self.dm.num_classes))
                f1_arr = f1_score(
                    true_labels, predicted_labels, labels=lab_ids, average=None, zero_division=0
                )
                per_class_f1 = {i: float(f1_arr[i]) for i in lab_ids}
                rec_arr = recall_score(
                    true_labels, predicted_labels, labels=lab_ids, average=None, zero_division=0
                )
                per_class_recall = {i: float(rec_arr[i]) for i in lab_ids}
                worst_class_acc = float(min(per_class_acc.values())) if per_class_acc else float("nan")
                tail_recall_mean = _tail_recall_mean_from_counts(
                    class_sample_counts or {}, per_class_recall, self.dm.num_classes
                )
            else:
                macro_f1 = 0.0
                per_class_f1 = {i: float("nan") for i in range(self.dm.num_classes)}
                per_class_recall = {i: float("nan") for i in range(self.dm.num_classes)}
                worst_class_acc = float("nan")
                tail_recall_mean = float("nan")
            print(f"Macro-F1: {macro_f1:.4f}")

            # 记录到logger
            self.logger.log_round_accuracy(
                round_idx=current_round,
                overall_acc=overall_acc,
                per_class_acc=per_class_acc,
                class_sample_counts=class_sample_counts,
                total_samples_used=total_samples_used,
                ece=ece_value,
                avg_nll=avg_nll,
                macro_f1=macro_f1,
                per_class_f1=per_class_f1,
                per_class_recall=per_class_recall,
                worst_class_acc=worst_class_acc,
                tail_recall_mean=tail_recall_mean,
            )
            if len(all_probabilities) > 0:
                probs_np = np.stack(all_probabilities, axis=0)           # (N, C)
                labels_np = np.array(true_labels, dtype=np.int64)        # (N,)
                self.logger.log_calibration_data(
                    round_idx=current_round,
                    probs=probs_np,
                    labels=labels_np,
                    ece=ece_value,
                    avg_nll=avg_nll
                )

            
            # 打印统计信息
            print(f"\n[Round {current_round} Statistics]")
            print(f"Total samples used: {total_samples_used}")
            print(f"Overall accuracy: {overall_acc:.4f}")
            print("\nPer-class statistics:")
            for class_idx in range(self.dm.num_classes):
                class_name = self.dm.lab2cname[class_idx]
                count = class_sample_counts.get(class_idx, 0)
                acc = per_class_acc.get(class_idx, 0.0)
                print(f"  {class_name}: {count} samples, {acc:.4f} accuracy")

        # 最后一轮生成CSV
        is_final_round = (current_round != 'N/A' and 
                        total_rounds != 'N/A' and 
                        current_round == total_rounds - 1)

        if is_final_round:
                if image_paths:
                    print("\nThis is the final round. Generating CSV file with prediction results...")
                    
                    df_data = {
                        'ImagePath': image_paths,
                        'TrueLabel': true_labels,
                        'PredictedLabel': predicted_labels
                    }
                    
                    classnames = self.dm.dataset.classnames
                    prob_array = np.array(all_probabilities)
                    for i, classname in enumerate(classnames):
                        df_data[f'Prob_{classname}'] = prob_array[:, i]
                    
                    df = pd.DataFrame(df_data)

                    save_path = os.path.join(self.output_dir, "final_round_predictions.csv")
                    df.to_csv(save_path, index=False)
                    print(f"Prediction results have been saved to: {save_path}")
                else:
                    print("\nThis is the final round, but no image paths were found to generate a CSV.")
        else:
                print(f"\n--- End of Test for Round {current_round} ---")

        return results["accuracy"]
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDCOOP.PREC in ["fp16", "fp32", "amp"]



    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # --- 开始替换 ---
        print(f"Loading BiomedCLIP (backbone: hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)")
        # 1. 使用 BiomedCoOp 的方式加载模型
        biomedclip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        # 2. 设置模型精度
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            biomedclip_model.float()

        print("Building custom CLIP with BiomedCoOp structure")
        # 3. 实例化我们刚刚复制过来的 CustomCLIP 类

        if cfg.TRAINER.COOPAL.ASPATH:
            self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval(), desc_file=cfg.TRAINER.COOPAL.ASPATH)
        elif cfg.TRAINER.COOPAL.AEPATH:
            self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval(), desc_file=cfg.TRAINER.COOPAL.AEPATH)
        else:
            self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())
        
       
        #self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())
        #print(self.model)

        print("Turning off gradients in both the image and the text encoder")

        # 4. 使用 BiomedCoOp 的方式冻结参数
        names_to_update = ["prompt_learner.ctx"]
        if getattr(cfg.TRAINER.COOPAL, "OUTPUT_ADAPTER", False):
            names_to_update.append("output_adapter.weight")
        for name, param in self.model.named_parameters():
            param.requires_grad_(name in names_to_update)

        # 打印可训练参数以供检查
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"CLIP Parameters to be updated: {enabled}")

        self.model.to(self.device)
        
        # 5. 使用 BiomedCoOp 的方式设置优化器
        # 注意：BiomedCoOp 的优化器是 self.model，而不是 self.model.prompt_learner
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        # 6. 使用 BiomedCoOp 的方式注册模型
        # 注意：注册的是 self.model，而不是 self.model.prompt_learner
        self.register_model("prompt_learner", self.model, self.optim, self.sched)


        # ================= 新增: 构建 MEH 网络 =================
        print("Building the Model Evidence Head (MEH) network")
        # 从主模型获取特征维度，同时处理 DataParallel 包装器的情况
        fdim = self.model.module.fdim if isinstance(self.model, nn.DataParallel) else self.model.fdim
        
   
        num_classes = self.num_classes
        if cfg.TRAINER.COOPAL.MEH_VERSION != 'v2':
            raise ValueError(f"[biomedclip] Unsupported MEH_VERSION: {cfg.TRAINER.COOPAL.MEH_VERSION}")
        print(f"[INFO] Detected {num_classes} classes for MEH_V2 initialization")
        self.meh_net = VLM_EH_V2(fdim, num_classes).to(self.device)
        self._meh_mode = 'v2'

        print("MEH Parameters to be updated:")
        for name, param in self.meh_net.named_parameters():
            print(name)
        
        # 为 MEH 网络创建独立的优化器和学习率调度器
        self.optim_meh = build_optimizer(self.meh_net, cfg.OPTIM)
        self.sched_meh = build_lr_scheduler(self.optim_meh, cfg.OPTIM)
        self.register_model("meh_net", self.meh_net, self.optim_meh, self.sched_meh)
        # =============================================================

        # 7. 初始化 GradScaler，用于混合精度训练
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None
        self.scaler_meh = GradScaler() if self.cfg.TRAINER.COOP.PREC == "amp" else None

        # --- 结束替换 ---

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)        

        # ================= 新增: 计算并显示模型参数量 =================
        # self._print_model_statistics()
        # ==============================================================

 
    def forward_backward_vlm(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        loss_summary = {
            "loss_vlm": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        return loss_summary



    def forward_backward_MEH(self, batch):
        """处理MEH网络的前向和后向传播，支持AMP"""
        image, label = self.parse_batch_train(batch)
        
        # 标志：记录优化器是否执行了step
        self.meh_optimizer_stepped = False
        
        # 调试模式开关
        debug_meh = False  # 设置为 True 启用详细的MEH调试信息
        
        prec = self.cfg.TRAINER.COOP.PREC

        # 1. 从 VLM 获取特征和 logits，不计算梯度
        with torch.no_grad():
            logits, features = self.model(image, get_feature=True)
            
            if isinstance(self.model, nn.DataParallel):
                logits_original = self.model.module.original_logits
            else:
                logits_original = self.model.original_logits

            classification_loss = F.cross_entropy(logits_original, label, reduction='none')
            # Use original (unscaled) logits to avoid overconfidence
            similarity_scores = logits_original
            
        features = features.detach().requires_grad_()
        similarity_scores = logits_original.detach().requires_grad_()
        logits_original = logits_original.detach().requires_grad_()

        # =================================================================
        # [DEBUG] 打印 MEH 输入张量的平均值
        # =================================================================
        # print("\n" + "="*60)
        # print(f"[MEH 均值调试] Epoch {self.epoch}, Batch {self.batch_idx}")
        
        # # 使用 .detach() 来计算均值，这样不会影响梯度计算（即使 requires_grad=True）
        # # 使用 .float() 来确保均值计算的精度，防止溢出
        
        # # --- 1. 'features' 平均值 ---
        # mean_features = torch.mean(features.detach().float())
        # print(f"  - 1. features:             Mean = {mean_features:.6f}") 

        # # --- 2. 'similarity_scores' 平均值 ---
        # mean_sim_scores = torch.mean(similarity_scores.detach().float())
        # print(f"  - 2. similarity_scores:    Mean = {mean_sim_scores:.6f}")

        # # --- 3. 'logits_original' 平均值 ---
        # mean_logits_orig = torch.mean(logits_original.detach().float())
        # print(f"  - 3. logits_original:      Mean = {mean_logits_orig:.6f}")
        
        # print("="*60 + "\n")
        # =================================================================
        # [DEBUG] 打印结束
        # =================================================================

        feat_mean = None
        sim_mean = None
        lambda_fused_mean = None
        if prec == "amp":
        # MEH 的前向传播和损失计算在 autocast 上下文中
            with autocast():
                if self.cfg.TRAINER.COOPAL.MEH_VERSION == 'v2':
                    feat_encoding, sim_encoding = self.meh_net(features, similarity_scores)
                    if self.cfg.TRAINER.COOPAL.MEH_LOSS_VERSION == 'meh_loss_v2_alternative':
                        loss_meh = meh_loss_v2_alternative(self, classification_loss, feat_encoding, sim_encoding, similarity_scores)
                    elif self.cfg.TRAINER.COOPAL.MEH_LOSS_VERSION == 'meh_loss_v2_log':
                        loss_meh = meh_loss_v2_log(self, classification_loss, feat_encoding, sim_encoding, similarity_scores)
                    else:
                        raise ValueError(f"❌ 无效的 MEH_LOSS_VERSION: {self.cfg.TRAINER.COOPAL.MEH_LOSS_VERSION}")
                else:
                    raise ValueError(f"❌ 无效的 MEH_VERSION: {self.cfg.TRAINER.COOPAL.MEH_VERSION}")
            
            if feat_encoding is not None:
                feat_mean = feat_encoding.detach().mean().item()
                sim_mean = sim_encoding.detach().mean().item()
                lambda_fused_mean = (feat_encoding + sim_encoding).detach().mean().item()
            
            self.optim_meh.zero_grad()
            self.scaler_meh.scale(loss_meh).backward()
            
            # ✅ 需要先unscale梯度才能进行裁剪
            self.scaler_meh.unscale_(self.optim_meh)
            
            # 在裁剪前检查梯度（用于显示原始梯度大小）
            grad_norms_before = []
            for param in self.meh_net.parameters():
                if param.grad is not None:
                    grad_norms_before.append(param.grad.norm().item())
            
            if len(grad_norms_before) > 0:
                avg_grad_before = sum(grad_norms_before) / len(grad_norms_before)
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.meh_net.parameters(), max_norm=1.0)
                
                # 始终显示梯度异常情况
                if avg_grad_before > 1000 or not torch.isfinite(torch.tensor(avg_grad_before)):
                    print(f"[MEH-AMP] WARNING: Epoch {self.epoch}, Batch {self.batch_idx}: "
                            f"Abnormal gradient! loss={loss_meh.item():.4f}, avg_grad_before={avg_grad_before:.2f} "
                            f"(clipped to max_norm=1.0)")
                
                self.scaler_meh.step(self.optim_meh)
                self.scaler_meh.update()
                self.meh_optimizer_stepped = True
                
                if debug_meh:
                    # 计算裁剪后的梯度（可选）
                    grad_norms_after = []
                    for param in self.meh_net.parameters():
                        if param.grad is not None:
                            grad_norms_after.append(param.grad.norm().item())
                    avg_grad_after = sum(grad_norms_after) / len(grad_norms_after) if grad_norms_after else 0
                    print(f"[MEH-AMP] Epoch {self.epoch}, Batch {self.batch_idx}: "
                            f"loss={loss_meh.item():.4f}, grad_before={avg_grad_before:.2f}, "
                            f"grad_after={avg_grad_after:.2f}, optimizer.step() ✅")
            else:
                    # 这种情况不应该发生，始终打印警告
                    print(f"[MEH-AMP] WARNING: Epoch {self.epoch}, Batch {self.batch_idx}: "
                          f"NO GRADIENTS! loss={loss_meh.item():.4f}, skipping step ❌")

        else:
                # -------------------
                # 标准精度 (FP32) 分支
                # -------------------
                
                # 2. MEH 前向传播

                if self.cfg.TRAINER.COOPAL.MEH_VERSION == 'v2':
                    feat_encoding, sim_encoding = self.meh_net(features, similarity_scores)
                    if self.cfg.TRAINER.COOPAL.MEH_LOSS_VERSION == 'meh_loss_v2_alternative':
                        loss_meh = meh_loss_v2_alternative(self, classification_loss, feat_encoding, sim_encoding, similarity_scores)
                    elif self.cfg.TRAINER.COOPAL.MEH_LOSS_VERSION == 'meh_loss_v2_log':
                        loss_meh = meh_loss_v2_log(self, classification_loss, feat_encoding, sim_encoding, similarity_scores)
                    else:
                        raise ValueError(f"❌ 无效的 MEH_LOSS_VERSION: {self.cfg.TRAINER.COOPAL.MEH_LOSS_VERSION}")
                else:
                    raise ValueError(f"❌ 无效的 MEH_VERSION: {self.cfg.TRAINER.COOPAL.MEH_VERSION}")

                # 4. MEH 标准反向传播和优化
                if loss_meh.requires_grad:
                    self.optim_meh.zero_grad()
                    loss_meh.backward()
                    
                    # 在裁剪前检查梯度
                    grad_norms_before = []
                    for param in self.meh_net.parameters():
                        if param.grad is not None:
                            grad_norms_before.append(param.grad.norm().item())
                    
                    if len(grad_norms_before) > 0:
                        avg_grad_before = sum(grad_norms_before) / len(grad_norms_before)
                        
                        # ✅ 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.meh_net.parameters(), max_norm=1.0)
                        
                        # 始终显示梯度异常情况
                        if avg_grad_before > 1000 or not torch.isfinite(torch.tensor(avg_grad_before)):
                            print(f"[MEH-FP32] WARNING: Epoch {self.epoch}, Batch {self.batch_idx}: "
                                  f"Abnormal gradient! loss={loss_meh.item():.4f}, avg_grad_before={avg_grad_before:.2f} "
                                  f"(clipped to max_norm=1.0)")
                        
                        self.optim_meh.step()
                        self.meh_optimizer_stepped = True
                        
                        if debug_meh:
                            grad_norms_after = []
                            for param in self.meh_net.parameters():
                                if param.grad is not None:
                                    grad_norms_after.append(param.grad.norm().item())
                            avg_grad_after = sum(grad_norms_after) / len(grad_norms_after) if grad_norms_after else 0
                            print(f"[MEH-FP32] Epoch {self.epoch}, Batch {self.batch_idx}: "
                                  f"loss={loss_meh.item():.4f}, grad_before={avg_grad_before:.2f}, "
                                  f"grad_after={avg_grad_after:.2f}, optimizer.step() ✅")
                    else:
                        print(f"[MEH-FP32] WARNING: Epoch {self.epoch}, Batch {self.batch_idx}: "
                              f"NO GRADIENTS! loss={loss_meh.item():.4f}, skipping step ❌")
                else:
                    print(f"[MEH-FP32] WARNING: Epoch {self.epoch}, Batch {self.batch_idx}: "
                          f"loss.requires_grad=False, skipping backward ❌")
    
        loss_summary = {
            "loss_meh": loss_meh.item()
        }
        if feat_mean is not None:
            loss_summary["feat_enc"] = feat_mean
        if sim_mean is not None:
            loss_summary["sim_enc"] = sim_mean
        if lambda_fused_mean is not None:
            loss_summary["lambda_fused"] = lambda_fused_mean

        return loss_summary


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def _load_checkpoint_on_cpu(self, fpath):
        """Load checkpoint on CPU to avoid GPU OOM during resume on busy cards."""
        try:
            return torch.load(fpath, map_location="cpu")
        except UnicodeDecodeError:
            import pickle
            from functools import partial

            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            return torch.load(fpath, pickle_module=pickle, map_location="cpu")

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = self._load_checkpoint_on_cpu(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def before_train(self):
        print("INITIALIZE the prompts weights")
        self.build_model()

    def _al_checkpoint_policy(self):
        coop = getattr(self.cfg.TRAINER, "COOPAL", object())
        keep_last = int(getattr(coop, "AL_CHECKPOINT_KEEP_LAST", 2))
        interval = max(int(getattr(coop, "AL_CHECKPOINT_INTERVAL", 1)), 1)
        return keep_last, interval

    def after_epoch(self):
        """AL 训练期间按策略落盘 checkpoint，并裁剪旧权重避免占满磁盘。"""
        if getattr(self, "_al_resume_active", False):
            keep_last, interval = self._al_checkpoint_policy()
            epoch_no = self.epoch + 1
            is_last = epoch_no >= self.max_epoch
            should_save = (epoch_no % interval == 0) or is_last
            if should_save:
                self.save_model(self.epoch, self.output_dir)
                if keep_last >= 0:
                    deleted, freed = prune_output_dir(self.output_dir, keep_last=keep_last)
                    if deleted:
                        print(
                            f"[al_checkpoint] pruned {deleted} old checkpoint(s), "
                            f"freed {freed / (1024 ** 2):.1f} MiB in {self.output_dir}"
                        )
            ctx = getattr(self, "_al_resume_ctx", None)
            if ctx is not None:
                al_resume.save_resume_state(
                    self.output_dir,
                    cfg=self.cfg,
                    total_n=ctx["total_n"],
                    n_query=ctx["n_query"],
                    total_rounds=ctx["total_rounds"],
                    labeled_global_idx=ctx["labeled_global_idx"],
                    u_index=ctx["u_index"],
                    next_round=ctx["current_round"],
                    resume_epoch=epoch_no,
                    completed_round=max(-1, ctx["current_round"] - 1),
                    note=f"mid-round epoch {epoch_no}/{self.max_epoch}",
                )
            return
        last_epoch = (self.epoch + 1) == self.max_epoch
        if last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def _load_al_checkpoint_weights(self):
        """续跑时加载上一轮权重（不继承 epoch 计数）。"""
        try:
            self.load_model(self.output_dir, epoch=self.max_epoch)
        except FileNotFoundError:
            self._resume_al_from_latest_checkpoint(fallback_epoch=self.max_epoch)
        self.start_epoch = 0

    def _resume_al_from_latest_checkpoint(self, fallback_epoch=None):
        """Load newest on-disk checkpoint; repair stale pointer after prune."""
        names = self.get_model_names()
        loaded_epoch = None
        for name in names:
            model_dir = Path(self.output_dir) / name
            candidates = sorted(
                list_epoch_checkpoints(model_dir),
                key=lambda item: item[1].stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                print(f"[al_checkpoint] no checkpoint under {model_dir}, train from scratch")
                return None
            checkpoint = None
            path = None
            for _ep, _path in candidates:
                try:
                    checkpoint = self._load_checkpoint_on_cpu(str(_path))
                    path = _path
                    break
                except Exception as exc:
                    print(f"[al_checkpoint] skip unreadable checkpoint {_path}: {exc}")
            if checkpoint is None or path is None:
                print(f"[al_checkpoint] no readable checkpoint under {model_dir}, train from scratch")
                return None
            state_dict = checkpoint["state_dict"]
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            print(
                f'Loading weights to {name} from "{path}" '
                f"(epoch = {checkpoint['epoch']})"
            )
            self._models[name].load_state_dict(state_dict, strict=False)
            loaded_epoch = int(checkpoint["epoch"]) if loaded_epoch is None else min(
                loaded_epoch, int(checkpoint["epoch"])
            )
            _sync_checkpoint_pointer(model_dir, path)
        return loaded_epoch

    def _resolve_al_resume_start_epoch(self, target_epoch: int) -> int:
        """Resume mid-round training, falling back when prune removed old ckpts."""
        names = self.get_model_names()
        ptr_ok = True
        for name in names:
            ptr = Path(self.output_dir) / name / "checkpoint"
            if not ptr.is_file():
                ptr_ok = False
                break
            ckpt_name = ptr.read_text(encoding="utf-8").strip()
            if not (Path(self.output_dir) / name / ckpt_name).is_file():
                ptr_ok = False
                break
        if ptr_ok:
            try:
                self.resume_model_if_exist(self.output_dir)
                return int(target_epoch)
            except Exception as exc:
                print(
                    f"[al_checkpoint] resume_model_if_exist failed ({exc}); "
                    f"falling back to latest readable checkpoint"
                )

        loaded_epoch = self._resume_al_from_latest_checkpoint()
        if loaded_epoch is None:
            return 0
        target_epoch = int(target_epoch)
        if loaded_epoch + 1 < target_epoch or loaded_epoch >= target_epoch:
            print(
                f"[al_checkpoint] stale/missing mid-round ckpt "
                f"(wanted epoch {target_epoch}, latest={loaded_epoch}); "
                f"retrain current round from epoch 0"
            )
            return 0
        return target_epoch

    def _al_resume_ctx_update(self, total_n, n_query, total_rounds, current_round, labeled, u_index):
        self._al_resume_ctx = {
            "total_n": total_n,
            "n_query": n_query,
            "total_rounds": total_rounds,
            "current_round": current_round,
            "labeled_global_idx": sorted(labeled),
            "u_index": list(u_index),
        }

    def _al_save_round_complete(self, total_n, n_query, total_rounds, round_i, labeled, u_index):
        if not getattr(self, "_al_resume_active", False):
            return
        al_resume.save_resume_state(
            self.output_dir,
            cfg=self.cfg,
            total_n=total_n,
            n_query=n_query,
            total_rounds=total_rounds,
            labeled_global_idx=sorted(labeled),
            u_index=list(u_index),
            next_round=round_i + 1,
            resume_epoch=0,
            completed_round=round_i,
            note=f"round {round_i} complete",
        )
        
    def after_train(self):
        print("Finish training")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.acc.append(self.test())
            
        # Close writer
        self.close_writer()
        
    # def _print_model_statistics(self):
    #     """
    #     计算并打印模型的参数量统计信息
    #     """
    #     def count_parameters(model):
    #         """统计模型的总参数量和可训练参数量"""
    #         total = sum(p.numel() for p in model.parameters())
    #         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #         return total, trainable
        
    #     print("\n" + "="*70)
    #     print("MODEL PARAMETER STATISTICS")
    #     print("="*70)
        
    #     # 处理DataParallel包装
    #     model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
    #     # 1. Prompt Learner参数
    #     prompt_total, prompt_trainable = count_parameters(model.prompt_learner)
    #     print(f"\n1. Prompt Learner:")
    #     print(f"   - Total params:      {prompt_total:,}")
    #     print(f"   - Trainable params:  {prompt_trainable:,}")
        
    #     # 2. Image Encoder参数
    #     image_total, image_trainable = count_parameters(model.image_encoder)
    #     print(f"\n2. Image Encoder (Vision Transformer):")
    #     print(f"   - Total params:      {image_total:,}")
    #     print(f"   - Trainable params:  {image_trainable:,}")
        
    #     # 3. Text Encoder参数
    #     text_total, text_trainable = count_parameters(model.text_encoder)
    #     print(f"\n3. Text Encoder:")
    #     print(f"   - Total params:      {text_total:,}")
    #     print(f"   - Trainable params:  {text_trainable:,}")
        
    #     # 4. MEH Network参数
    #     meh_total, meh_trainable = count_parameters(self.meh_net)
    #     print(f"\n4. MEH Network (Model Evidence Head):")
    #     print(f"   - Total params:      {meh_total:,}")
    #     print(f"   - Trainable params:  {meh_trainable:,}")
        
    #     # 5. 总计
    #     vlm_total = prompt_total + image_total + text_total
    #     vlm_trainable = prompt_trainable + image_trainable + text_trainable
    #     all_total = vlm_total + meh_total
    #     all_trainable = vlm_trainable + meh_trainable
        
    #     print(f"\n" + "-"*70)
    #     print(f"TOTAL (VLM Model):")
    #     print(f"   - Total params:      {vlm_total:,}")
    #     print(f"   - Trainable params:  {vlm_trainable:,}")
    #     print(f"   - Frozen params:     {vlm_total - vlm_trainable:,}")
        
    #     print(f"\nTOTAL (VLM + MEH):")
    #     print(f"   - Total params:      {all_total:,}")
    #     print(f"   - Trainable params:  {all_trainable:,}")
    #     print(f"   - Frozen params:     {all_total - all_trainable:,}")
    #     print(f"   - Trainable ratio:   {all_trainable/all_total*100:.2f}%")
        
    #     # 6. 按MB计算模型大小（假设float32）
    #     model_size_mb = all_total * 4 / (1024 ** 2)  # 4 bytes per float32
    #     trainable_size_mb = all_trainable * 4 / (1024 ** 2)
        
    #     print(f"\nModel Size (float32):")
    #     print(f"   - Total size:        {model_size_mb:.2f} MB")
    #     print(f"   - Trainable size:    {trainable_size_mb:.2f} MB")
        
    #     print("="*70 + "\n")
    
    def _save_config_snapshot(self):
        """保存当前 cfg 的完整快照到 OUTPUT_DIR/config_snapshot.yaml（每次训练开始时调用）。"""
        import yaml, os
        out_dir = self.cfg.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        snap_path = os.path.join(out_dir, "config_snapshot.yaml")

        def _cfg_to_dict(node):
            """将 YACS CfgNode 递归转为普通 dict（方便 yaml.dump）。"""
            try:
                from yacs.config import CfgNode
                if isinstance(node, CfgNode):
                    return {k: _cfg_to_dict(v) for k, v in node.items()}
            except ImportError:
                pass
            if isinstance(node, dict):
                return {k: _cfg_to_dict(v) for k, v in node.items()}
            if isinstance(node, (list, tuple)):
                return [_cfg_to_dict(v) for v in node]
            return node

        try:
            cfg_dict = _cfg_to_dict(self.cfg)
            with open(snap_path, "w") as f:
                yaml.dump(cfg_dict, f, default_flow_style=False, allow_unicode=True)
            print(f"[SaE] config_snapshot saved → {snap_path}")
        except Exception as e:
            print(f"[LT-SaE] WARNING: config_snapshot save failed: {e}")


    def train(self):
        """Generic training loops."""
        dataset = build_dataset(self.cfg)

        # ── 实验可复现性：所有 AL/2x2/adapter 运行都保存配置快照 ──
        self._save_config_snapshot()

        print(f"dataset length: {len(dataset.train_x)}")
        unlabeled_dst = dataset.train_x  #一开始全是未知的train_x
        U_index = list(range(len(unlabeled_dst))) #给所有的人标 标签
      
        self.num_classes = dataset.get_num_classes(unlabeled_dst)
        
        if self.cfg.TRAINER.COOP.CSC: #这里在设置 n_query，它代表在每一轮主动学习循环中，学生最终要挑选出来请教老师的题目数量。
            n_query = dataset.get_num_classes(unlabeled_dst)
        else:
            n_query = dataset.get_num_classes(unlabeled_dst)
        # Set n_query to 4% of the total number of unlabeled samples
        n_query = int(len(unlabeled_dst) * self.cfg.TRAINER.COOPAL.query)  # 4% of entire dataset
        # n_query = int(len(unlabeled_dst)) # for debug
       
        dataset._train_x = []
        self.total_rounds  =  self.cfg.TRAINER.COOPAL.Totalrounds #5

        self.GAMMA = round(1 / self.total_rounds, 2)
        n_cand = int(len(unlabeled_dst) * self.GAMMA) # 10% of entire dataset
        total_n = len(unlabeled_dst)

        resume_plan = al_resume.get_resume_plan(
            self.output_dir, self.cfg, total_n, n_query, self.total_rounds
        )
        start_round = 0
        resume_epoch_at_start = 0
        labeled_global_idx: list = []
        if resume_plan is not None:
            train_x, U_index = al_resume.apply_labeled_pool(
                unlabeled_dst,
                resume_plan["labeled_global_idx"],
                resume_plan["u_index"],
            )
            dataset._train_x = train_x
            start_round = int(resume_plan["next_round"])
            resume_epoch_at_start = int(resume_plan.get("resume_epoch", 0))
            labeled_global_idx = list(resume_plan["labeled_global_idx"])
            src = resume_plan.get("source", "checkpoint")
            print(
                f"[al_resume] resume from {src}: "
                f"round={start_round} epoch={resume_epoch_at_start} "
                f"labeled={len(labeled_global_idx)}/{total_n}"
            )

        print("\n" + "="*60+"\n")
        print(len(unlabeled_dst),n_query,n_cand)
        print("\n" + "="*60+"\n")
        pcb_flag = self.cfg.TRAINER.COOPAL.pcb_flag.strip().lower() == "true"  
        
        # Initialize t-SNE visualizer if enabled
        tsne_visualizer = None
        if self.cfg.TRAINER.COOPAL.TSNE_VIS:
            print("=" * 50)
            print("t-SNE Visualization Enabled")
            print("=" * 50)
            tsne_visualizer = TSNEVisualizer(output_dir=self.cfg.OUTPUT_DIR)
        
        self._al_resume_active = al_resume._enabled(self.cfg)
        self.before_train()
        if resume_plan is not None and (start_round > 0 or resume_epoch_at_start > 0):
            self._load_al_checkpoint_weights()
        if resume_plan is not None and start_round > 0 and hasattr(self, "logger"):
            self.logger.import_prior_rounds_from_csv(before_round=start_round)

        for i in range(start_round, self.total_rounds):
            self.current_round = i
            start = time.time()
            skip_acquisition = (i == start_round and resume_epoch_at_start > 0)
            selected_idx_this_round = []
            if not skip_acquisition:
                if i == 0:
                    idx = sample(U_index, n_query)
                else:
                    k = n_cand if pcb_flag else n_query  #现在是ture 表示开启
                    if self.cfg.TRAINER.COOPAL.METHOD == "random" :
                        idx = sample(U_index, n_query)
                    elif self.cfg.TRAINER.COOPAL.METHOD == "entropy":
                        selector = Entropy(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device)
                        idx = selector.select(k)
                    
                    elif self.cfg.TRAINER.COOPAL.METHOD == "badge":
                        selector = BADGE(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device)
                        idx = selector.select(k)
                        
                    elif self.cfg.TRAINER.COOPAL.METHOD in ("MEH", "sae_ca"):
                        selector = MEH_Selector(self.cfg, self.model, self.meh_net, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.current_round, self.total_rounds, self.device)
                        # B4g / B2：IR 与 logit-adjust 先验需要「全训练集」类频次 = 已标注 + 仍在池中的未标注
                        from collections import Counter as _Counter_lc
                        _lc = _Counter_lc(int(elem.label) for elem in dataset._train_x)
                        _labeled_counts = {c: int(_lc.get(c, 0)) for c in range(self.num_classes)}
                        idx = selector.select(k, labeled_counts=_labeled_counts)
                    elif self.cfg.TRAINER.COOPAL.METHOD == "coreset":
                        val_x = dataset._train_x.copy()
                        selector = Coreset(self.cfg, self.model, unlabeled_dst, U_index, val_x, dataset.get_num_classes(unlabeled_dst))
                        idx = selector.select(k)
                    else:
                        print("NotImplementedError")
                        idx = U_index
                    # sae_ca 方法内部 (MEH_Selector) 已经处理平衡采集，跳过 PCB 后处理
                    if pcb_flag and self.cfg.TRAINER.COOPAL.METHOD not in ("sae_ca",):
                        statistics = torch.zeros(self.num_classes)
                        for elem in dataset._train_x:
                            statistics[elem.label] += 1
                        pcb = PCB(self.cfg, self.model, unlabeled_dst, idx, self.num_classes, statistics, self.device, tsne_visualizer=tsne_visualizer)
                        idx = pcb.select(n_query)

                # Filtering       
                for k in idx:
                    dataset._train_x.append(unlabeled_dst[k]) #把这道题的完整内容拿出来，放入“已学知识
                    U_index.remove(k) #“待学习”编号列表(`U_index`)中划掉，避免重复。
                selected_idx_this_round = [int(k) for k in idx]
            else:
                print(f"[al_resume] skip acquisition for round {i}, resume epoch {resume_epoch_at_start}")

            labeled_global_idx = [j for j in range(total_n) if j not in set(U_index)]
            try:
                queried_class_counts = {
                    int(c): 0 for c in range(int(self.num_classes))
                }
                for gidx in selected_idx_this_round:
                    lab = int(unlabeled_dst[gidx].label)
                    queried_class_counts[lab] = queried_class_counts.get(lab, 0) + 1
                query_log_dir = os.path.join(self.output_dir, "training_logs")
                os.makedirs(query_log_dir, exist_ok=True)
                with open(os.path.join(query_log_dir, "query_rounds.jsonl"), "a", encoding="utf-8") as qf:
                    qf.write(
                        json.dumps(
                            {
                                "round": int(i),
                                "seed": int(getattr(self.cfg, "SEED", -1)),
                                "dataset": str(getattr(self.cfg.DATASET, "NAME", "")),
                                "n_query": int(n_query),
                                "selected_global_idx": selected_idx_this_round,
                                "queried_class_counts": queried_class_counts,
                                "labeled_global_idx_after_query": [int(x) for x in labeled_global_idx],
                                "unlabeled_pool_size_after_query": int(len(U_index)),
                                "skipped_for_resume": bool(skip_acquisition),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception as _e:
                print(f"[AL query log] WARNING: failed to write round {i}: {_e}")
            

            assert len(U_index) + len(dataset.train_x) == len(unlabeled_dst), f"u index: {len(U_index)}\t train set: {len(dataset.train_x)}\t unlabeled_dst: {len(unlabeled_dst)}"
            
            self.train_loader_x = build_data_loader(
                self.cfg,
                sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=dataset.train_x,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=True),
                is_train=True,
                dataset_wrapper=None,
            )   
            self._al_resume_ctx_update(
                total_n, n_query, self.total_rounds, i, labeled_global_idx, U_index
            )
            if skip_acquisition:
                self.start_epoch = self._resolve_al_resume_start_epoch(resume_epoch_at_start)
                resume_epoch_at_start = 0
            else:
                self.start_epoch = 0
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch()
                self.after_epoch()
            self.after_train()
            self._al_save_round_complete(
                total_n, n_query, self.total_rounds, i, labeled_global_idx, U_index
            )
            
            # Generate t-SNE visualization after training (for all rounds including Round 0)
            if tsne_visualizer is not None and len(dataset._train_x) > 0:
                print(f"\n[Round {i}] Generating t-SNE visualization after training...")
                
                # Extract labeled features
                labeled_features_list = []
                labeled_labels_list = []
                with torch.no_grad():
                    labeled_loader = build_data_loader(
                        self.cfg,
                        data_source=dataset._train_x,
                        batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
                        tfm=build_transform(self.cfg, is_train=False),
                        is_train=False,
                    )
                    for batch in labeled_loader:
                        inputs = batch["img"].to(self.device)
                        labels = batch["label"]
                        _, features = self.model(image=inputs, get_feature=True)
                        labeled_features_list.append(features.cpu())
                        labeled_labels_list.append(labels)
                
                labeled_features = torch.cat(labeled_features_list)
                labeled_labels = torch.cat(labeled_labels_list)
                
                # Extract unlabeled features
                unlabeled_features_list = []
                unlabeled_subset = torch.utils.data.Subset(unlabeled_dst, U_index)
                with torch.no_grad():
                    unlabeled_loader = build_data_loader(
                        self.cfg,
                        data_source=unlabeled_subset,
                        batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
                        tfm=build_transform(self.cfg, is_train=False),
                        is_train=False,
                    )
                    for batch in unlabeled_loader:
                        inputs = batch["img"].to(self.device)
                        _, features = self.model(image=inputs, get_feature=True)
                        unlabeled_features_list.append(features.cpu())
                
                unlabeled_features = torch.cat(unlabeled_features_list) if unlabeled_features_list else torch.tensor([])
                
                if len(unlabeled_features) > 0:
                    tsne_visualizer.visualize_round(
                        labeled_features=labeled_features,
                        unlabeled_features=unlabeled_features,
                        labeled_labels=labeled_labels,
                        round_idx=i,
                        n_labeled=len(dataset._train_x)
                    )
                    tsne_visualizer.visualize_round_umap(
                        labeled_features=labeled_features,
                        unlabeled_features=unlabeled_features,
                        labeled_labels=labeled_labels,
                        round_idx=i,
                        n_labeled=len(dataset._train_x)
                    )
            
            print("training time for {}-th round: {:.2f} seconds".format(i, time.time() - start))

        # Generate summary visualization after all rounds
        if tsne_visualizer is not None:
            print("\n" + "=" * 50)
            print("Generating summary t-SNE visualization for all rounds...")
            print("=" * 50)
            tsne_visualizer.create_summary_visualization()
            print("\n" + "=" * 50)
            print("Generating summary UMAP visualization for all rounds...")
            print("=" * 50)
            tsne_visualizer.create_summary_visualization_umap()

        if hasattr(self, 'logger'):
            al_resume.clear_resume_state(self.output_dir)
            self.logger.finalize()
            deleted, freed = purge_all_checkpoints(self.output_dir)
            if deleted:
                print(
                    f"[al_checkpoint] purged {deleted} checkpoint(s) after run complete, "
                    f"freed {freed / (1024 ** 2):.1f} MiB in {self.output_dir}"
                )

        print("=== Result Overview ===")
        for i in range(len(self.acc)):
            print(f"{i}: {self.acc[i]}")
        print("=======================")  

