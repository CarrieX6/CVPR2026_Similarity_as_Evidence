import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from tqdm import tqdm
from dassl.data.data_manager import build_data_loader
from dassl.data.transforms.transforms import build_transform

from .AL import AL


def _entropy(p: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """计算一批概率分布的香农熵。"""
    p_safe = torch.clamp(p, min=epsilon)
    return -torch.sum(p_safe * torch.log(p_safe), dim=-1)


# =============================================================================
# 方案1: λ基于图像特征预测 (原始方案改进版)
# =============================================================================

class VLM_EH_V1(nn.Module):
    """
    MEH网络 - 方案1：基于图像特征预测全局λ
    
    该方案将λ视为样本级别的全局置信度，仅从图像特征预测。
    适用于认为样本难度与类别无关的场景。
    
    Args:
        fdim (int): 输入图像特征的维度 (例如 CLIP ViT-B/16 为 512)
    """
    def __init__(self, fdim: int):
        super().__init__()
        # 更深的网络以更好地捕捉特征-难度映射

        self.evidence_head = nn.Sequential(
            nn.Linear(fdim, 256),
            nn.BatchNorm1d(256),  # 添加BN
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus(beta=0.5)
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: shape (batch_size, fdim)
        Returns:
            lambda_evidence: shape (batch_size, 1)
        """
        lambda_evidence = self.evidence_head(image_features)
        # 添加最小值约束，防止λ过小导致数值不稳定
        return torch.clamp(lambda_evidence, min=1e-3,max=3.0) + 1e-4



def meh_loss_v1(
    classification_loss: torch.Tensor, 
    lambda_evidence: torch.Tensor
) -> torch.Tensor:
    """
    MEH损失 - 方案1: L_MEH = MSE(1/λ, l_cls)
    
    这个损失函数训练MEH网络使得预测的困难度(1/λ)接近真实分类损失。
    
    Args:
        classification_loss: 分类损失 (已detach), shape (batch_size,)
        lambda_evidence: MEH预测的λ, shape (batch_size, 1) 或 (batch_size,)
    
    Returns:
        scalar loss
    """
    if lambda_evidence.ndim > 1 and lambda_evidence.shape[1] == 1:
        lambda_evidence = lambda_evidence.squeeze(1)
    
    epsilon = 1e-4  # 增大epsilon
    # 约束lambda范围
    lambda_evidence = torch.clamp(lambda_evidence, min=0.05, max=5.0)
    # 计算并裁剪困难度
    predicted_difficulty = 1.0 / (lambda_evidence + epsilon)
    predicted_difficulty = torch.clamp(predicted_difficulty, min=0.1, max=50.0)
    loss = F.mse_loss(predicted_difficulty, classification_loss)
    return loss


# =============================================================================
# 方案2: λ基于相似度分数预测 (类别感知方案)
# =============================================================================

class VLM_EH_V2(nn.Module):
    """
    专门为少样本场景设计的轻量级MEH网络
    - 更少的参数防止过拟合
    - 更强的正则化
    - dropout增加泛化能力
    """
    def __init__(self, fdim: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        
        # 简化的特征分支 - 减少参数
        self.feature_branch = nn.Sequential(
            nn.Linear(fdim, 128),  # 从256降到128
            nn.LayerNorm(128),     # LayerNorm对少样本更友好
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),    # 进一步降维
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 简化的相似度分支
        self.similarity_branch = nn.Sequential(
            nn.Linear(num_classes, 64),  # 从128降到64
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 融合层 - 也减少参数
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),  # 64+64=128输入
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Softplus(beta=1.0)  # 增大beta使输出更稳定
        )
        
        # 初始化权重 - 对少样本很重要
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)  # 减小初始权重
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, image_features, similarity_scores):
        feat_encoding = self.feature_branch(image_features)
        sim_encoding = self.similarity_branch(similarity_scores)
        
        combined = torch.cat([feat_encoding, sim_encoding], dim=1)
        lambda_evidence = self.fusion(combined)
        
        # 更保守的约束范围
        return torch.clamp(lambda_evidence, min=0.2, max=2.0)


def meh_loss_v2_alternative(
    classification_loss: torch.Tensor,
    lambda_evidence: torch.Tensor,
    similarity_scores: torch.Tensor,
    temperature: float = 0.5,
    l2_weight: float = 0.005
) -> torch.Tensor:
    """少样本专用MEH损失函数 - 修复梯度流"""
    
    # 确保输入形状正确
    if lambda_evidence.dim() == 1:
        lambda_evidence = lambda_evidence.unsqueeze(1)
    
    # ✅ 约束lambda范围（这个操作保持梯度）
    lambda_evidence = torch.clamp(lambda_evidence, min=0.1, max=3.0)
    
    # 项1: λ预测的困难度
    epsilon = 1e-3
    predicted_difficulty = 1.0 / (lambda_evidence.squeeze(1) + epsilon)
    predicted_difficulty = torch.clamp(predicted_difficulty, min=0.5, max=10.0)
    
    # ✅ classification_loss已经是detached的，这里不需要再detach
    classification_loss_clamped = torch.clamp(
        classification_loss,  # ✅ 输入已经detached
        min=0.1, max=10.0
    )
    
    # 主损失 - 这里lambda_evidence→predicted_difficulty有梯度
    loss_lambda = F.smooth_l1_loss(predicted_difficulty, classification_loss_clamped)
    
    # ✅ 确保loss_lambda有梯度
    if not loss_lambda.requires_grad:
        print("⚠️ loss_lambda does not require grad!")
    
    # 项2: 相似度一致性
    # ✅ similarity_scores已经detached，这里的计算只是监督信号
    probs = F.softmax(similarity_scores / temperature, dim=1)
    similarity_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    similarity_entropy = torch.clamp(similarity_entropy, min=0.1, max=5.0)
    
    target_lambda = 1.0 / (similarity_entropy + epsilon)
    target_lambda = torch.clamp(target_lambda, min=0.1, max=3.0)
    
    # 一致性损失 - lambda_evidence与target对比
    loss_consistency = F.smooth_l1_loss(
        lambda_evidence.squeeze(1), 
        target_lambda  # ✅ target_lambda是从detached的scores计算的，作为监督信号
    )
    
    # L2正则化 - 直接作用在lambda_evidence上
    l2_reg = torch.mean(lambda_evidence ** 2)
    
    # 组合损失 - 所有项都依赖lambda_evidence
    total_loss = loss_lambda + 0.3 * loss_consistency + l2_weight * l2_reg
    
    # ✅ 最终检查
    if not total_loss.requires_grad:
        print("❌ ERROR: total_loss does not require grad!")
        print(f"   loss_lambda.requires_grad: {loss_lambda.requires_grad}")
        print(f"   loss_consistency.requires_grad: {loss_consistency.requires_grad}")
        print(f"   l2_reg.requires_grad: {l2_reg.requires_grad}")
    
    return total_loss

def meh_loss_v2(
    classification_loss: torch.Tensor,
    similarity_scores: torch.Tensor,
    lambda_evidence: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    修正版 meh_loss_v2：保持以 classification_loss & similarity_scores 为监督，
    但使用 lambda_evidence（来自 self.meh_net）作为可导的校正项 (gamma, beta)，
    以便 loss 能向 meh_net 反传梯度。

    Args:
        classification_loss: 分类损失 (可能在 no_grad 环境下计算)，shape (B,)
        similarity_scores: 相似度分数 (来自 frozen CLIP)，shape (B, C)
        lambda_evidence: meh_net 的输出，shape (B, K) 可为 (B, 2*C) / (B, C) / (B,1) 等可广播形状
        temperature: softmax 温度
    """
    # 保证 shape 为 (B, *)
    if lambda_evidence is None:
        # 保守回退：如果没提供 lambda_evidence，则不产生可导项（极端情况）
        gamma = torch.zeros_like(similarity_scores[..., :1])
        beta = torch.zeros_like(similarity_scores[..., :1])
    else:
        if lambda_evidence.dim() == 1:
            lambda_evidence = lambda_evidence.unsqueeze(1)  # (B,1)

        # 若 lambda_evidence 列数可被2整除，则拆为 gamma, beta；否则把它作为 gamma，beta 为 0（可广播）
        if lambda_evidence.size(1) % 2 == 0:
            gamma, beta = torch.chunk(lambda_evidence, 2, dim=1)  # 可能为 (B, C) 或 (B,1) 等
        else:
            gamma = lambda_evidence
            beta = torch.zeros_like(gamma)

    # 让 gamma/beta 在列方向能够广播到 similarity_scores 的列数
    # 如果 gamma.shape==(B,1) 会自动广播到 (B, C)
    scaled_sim = (similarity_scores.detach() * (1.0 + gamma)) + beta
    scaled_sim = scaled_sim / temperature

    probs = F.softmax(scaled_sim, dim=1)
    max_confidence, _ = torch.max(probs, dim=1)
    max_confidence = torch.clamp(max_confidence, min=0.01, max=1.0)

    predicted_difficulty = 1.0 / (max_confidence + 1e-4)
    predicted_difficulty = torch.clamp(predicted_difficulty, min=0.1, max=50.0)

    classification_loss_clamped = torch.clamp(classification_loss.detach(), min=0.01, max=50.0)

    loss = F.mse_loss(predicted_difficulty, classification_loss_clamped)
    return loss


def calculate_vlm_uncertainties(
    vlm_similarities: torch.Tensor,
    lambda_evidence: torch.Tensor,
    num_samples_for_epistemic: int = 50
) -> dict:
    """
    根据VLM相似度和MEH模型证据，计算三种不确定性。
    
    实现公式：
        p_k^{AE} = exp(s_k/τ) / Σ_c exp(s_c/τ)
        e_k = λ * p_k^{AE}
        α_k = e_k + 1
        S = Σ_k α_k
    
    Args:
        vlm_similarities: VLM相似度分数, shape (batch_size, num_classes)
        lambda_evidence: MEH输出的λ, shape (batch_size,) 或 (batch_size, 1)
        temperature: softmax温度参数 τ
        num_samples_for_epistemic: 蒙特卡洛采样次数
    
    Returns:
        dict: 包含 'epistemic', 'vacuity', 'dissonance', 'alpha', 'S'
    """
    if vlm_similarities.ndim != 2:
        raise ValueError("vlm_similarities must be 2D (batch_size, num_classes)")
    
    if lambda_evidence.ndim > 1:
        lambda_evidence = lambda_evidence.squeeze(-1)
    
    if lambda_evidence.ndim != 1 or lambda_evidence.shape[0] != vlm_similarities.shape[0]:
        raise ValueError("lambda_evidence shape mismatch")

    batch_size, num_classes = vlm_similarities.shape
    device = vlm_similarities.device

    beta = vlm_similarities
    p_ae = F.softmax(beta, dim=1)

    # 使用MEH的λ进行缩放，得到alpha
    # unsqueeze将lambda从(B,)变为(B, 1)以进行广播乘法
    evidence = lambda_evidence.unsqueeze(1) * p_ae
    # evidence = 1*p_ae
    # # 3. 计算 Dirichlet 参数 α_k = e_k + 1
    alpha = evidence + 1.0

    # 4. 总强度 S = Σ_k α_k
    S = torch.sum(alpha, dim=1)
    
    # 5. 计算认知不确定性 (Epistemic Uncertainty)
    dirichlet_dist = Dirichlet(alpha)
    samples = dirichlet_dist.sample((num_samples_for_epistemic,))
    
    mean_probs = torch.mean(samples, dim=0)
    entropy_of_mean = _entropy(mean_probs)
    
    entropies_of_samples = _entropy(samples)
    mean_of_entropies = torch.mean(entropies_of_samples, dim=0)
    
    epistemic_uncertainty = entropy_of_mean - mean_of_entropies
    
    # 6. 计算空缺性 (Vacuity)
    vacuity = num_classes / S
    
    # 7. 计算冲突性 (Dissonance)
    belief_masses = evidence / S.unsqueeze(1)
    
    dissonance = torch.zeros(batch_size, device=device)
    for i in range(num_classes):
        b_i = belief_masses[:, i]
        
        # 其他类别信念
        mask = torch.ones(num_classes, dtype=torch.bool, device=device)
        mask[i] = False
        other_beliefs = belief_masses[:, mask]
        
        sum_other_beliefs = torch.sum(other_beliefs, dim=1)
        
        # Bal(b_j, b_i) = 1 - |b_j - b_i| / (b_j + b_i)
        bal = 1 - torch.abs(other_beliefs - b_i.unsqueeze(1)) / \
              (other_beliefs + b_i.unsqueeze(1) + 1e-12)
        
        inner_sum = torch.sum(other_beliefs * bal, dim=1)
        
        term = torch.where(
            sum_other_beliefs > 1e-6,
            b_i * inner_sum / sum_other_beliefs,
            torch.zeros_like(b_i)
        )
        dissonance += term
    
    return {
        'epistemic': epistemic_uncertainty,
        'vacuity': vacuity,
        'dissonance': dissonance,
        'alpha': alpha,
        'S': S
    }

   


# =============================================================================
# 主动学习选择器
# =============================================================================

class MEH_Selector(AL):
    """
    基于MEH的主动学习样本选择器
    
    支持两种方案：
    - V1: 基于图像特征预测λ
    - V2: 基于相似度和特征预测λ
    """
    def __init__(
        self, 
        cfg, 
        model, 
        meh_net, 
        unlabeled_dst, 
        U_index, 
        n_class, 
        device,
        meh_version='v2',  # 'v1' 或 'v2'
        temperature=1,
        **kwargs
    ):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device
        self.meh_net = meh_net
        self.meh_version = self.cfg.TRAINER.COOPAL.MEH_VERSION
        self.temperature = temperature

    def select(self, n_query, **kwargs):
        """基于组合不确定性选择样本"""
        self.model.eval()
        self.meh_net.eval()

        unlabeled_loader = build_data_loader(
            self.cfg,
            data_source=self.unlabeled_set,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            is_train=False,
            tfm=build_transform(self.cfg, is_train=False),
        )
       
        batch_epistemic = []
        batch_vacuity = []
        batch_dissonance = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(unlabeled_loader, desc="Computing uncertainties"):
                inputs = batch["img"].to(self.device)

                # 获取VLM相似度和特征
                vlm_similarities, features = self.model(inputs, get_feature=True)

                # 根据版本获取λ
                if self.cfg.TRAINER.COOPAL.MEH_VERSION == 'v1':
                    lambda_evidence = self.meh_net(features).squeeze(1)
                elif self.cfg.TRAINER.COOPAL.MEH_VERSION == 'v2':
                    lambda_evidence = self.meh_net(features, vlm_similarities).squeeze(1)
                else:
                    raise ValueError(f"Unknown MEH version: {self.cfg.TRAINER.COOPAL.MEH_LOSS_VERSION}")

                # 计算不确定性
                uncertainties_batch = calculate_vlm_uncertainties(
                    vlm_similarities, 
                    lambda_evidence,
                )
                
                batch_epistemic.append(uncertainties_batch['epistemic'])
                batch_vacuity.append(uncertainties_batch['vacuity'])
                batch_dissonance.append(uncertainties_batch['dissonance'])
                
                # 组合不确定性
                total_uncertainty = (
                    uncertainties_batch['epistemic'] +
                    uncertainties_batch['vacuity'] +
                    uncertainties_batch['dissonance']
                )
                all_uncertainties.append(total_uncertainty)

        # 拼接结果
        all_uncertainties = torch.cat(all_uncertainties)
        all_epistemic = torch.cat(batch_epistemic)
        all_vacuity = torch.cat(batch_vacuity)
        all_dissonance = torch.cat(batch_dissonance)

        total_uncertainty = all_epistemic + all_vacuity + all_dissonance

        # 统计信息
        print("\n" + "="*60)
        print(f"MEH Version: {self.meh_version.upper()}")
        print("="*60)
        print(f"Epistemic    : Mean={all_epistemic.mean():.4f}, Std={all_epistemic.std():.4f}, "
              f"Min={all_epistemic.min():.4f}, Max={all_epistemic.max():.4f}")
        print(f"Vacuity      : Mean={all_vacuity.mean():.4f}, Std={all_vacuity.std():.4f}, "
              f"Min={all_vacuity.min():.4f}, Max={all_vacuity.max():.4f}")
        print(f"Dissonance   : Mean={all_dissonance.mean():.6f}, Std={all_dissonance.std():.6f}, "
              f"Min={all_dissonance.min():.6f}, Max={all_dissonance.max():.6f}")
        print("-"*60)
        print(f"Total        : Mean={total_uncertainty.mean():.4f}, Std={total_uncertainty.std():.4f}, "
              f"Min={total_uncertainty.min():.4f}, Max={total_uncertainty.max():.4f}")
        print("="*60 + "\n")

        # 选择最高不确定性样本
        _, selected_indices = torch.topk(all_uncertainties, n_query)
        Q_index = [self.U_index[idx] for idx in selected_indices.cpu().numpy()]
        
        return Q_index


# =============================================================================
# 使用示例
# =============================================================================

"""
# 方案1: 基于图像特征
meh_net_v1 = VLM_EH_V1(fdim=512).to(device)
selector_v1 = MEH_Selector(
    cfg, model, meh_net_v1, unlabeled_dst, U_index, n_class, device,
    meh_version='v1'
)

# 训练时使用
loss_cls = criterion(logits, labels)  # 主任务损失
with torch.no_grad():
    lambda_pred = meh_net_v1(image_features)
loss_meh = meh_loss_v1(loss_cls.detach(), lambda_pred)

# 方案2: 基于相似度和特征
meh_net_v2 = VLM_EH_V2(fdim=512, num_classes=100).to(device)
selector_v2 = MEH_Selector(
    cfg, model, meh_net_v2, unlabeled_dst, U_index, n_class, device,
    meh_version='v2'
)

# 训练时使用（备选损失函数）
loss_meh = meh_loss_v2_alternative(
    loss_cls.detach(), 
    lambda_pred, 
    similarity_scores
)
"""
