"""
Title: MEH — Model Evidence Head + Acquisition Selector
Provenance: RA-SaE active-learning evidence-head selector.
Scope:
  - VLM_EH_V2 + scalar vacuity/dissonance (conference SaE)
  - MEH_Selector with legacy_wv_wd + balanced class-aware acquisition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from tqdm import tqdm
from dassl.data.data_manager import build_data_loader
from dassl.data.transforms.transforms import build_transform

from .AL import AL
import numpy as np
from typing import Dict, List, Optional
from collections import Counter


# =============================================================================
# 证据头网络
# =============================================================================

def _entropy(p: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Shannon entropy over last dimension."""
    p_safe = torch.clamp(p, min=epsilon)
    return -torch.sum(p_safe * torch.log(p_safe), dim=-1)


class VLM_EH_V2(nn.Module):
    """Evidence Head V2: dual-branch (feature + similarity) returning (feat_enc, sim_enc)."""

    def __init__(self, fdim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.feature_branch = nn.Sequential(
            nn.Linear(fdim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
        )
        self.similarity_branch = nn.Sequential(
            nn.Linear(num_classes, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
        )
        self.feature_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus(beta=0.5))
        self.similarity_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus(beta=0.5))

    def forward(self, image_features, similarity_scores):
        feat_enc = self.feature_head(self.feature_branch(image_features))
        sim_enc = self.similarity_head(self.similarity_branch(similarity_scores))
        return feat_enc, sim_enc


# =============================================================================
# MEH 损失函数
# =============================================================================

def meh_loss_v2_alternative(
    self,
    classification_loss: torch.Tensor,
    feat_encoding: torch.Tensor,
    sim_encoding: torch.Tensor,
    similarity_scores: torch.Tensor,
) -> torch.Tensor:
    """MEH loss (alternative): 1/lambda difficulty prediction + similarity consistency."""
    if getattr(self.cfg.TRAINER.COOPAL, 'Ablation', 'None') == 'None':
        if feat_encoding.ndim > 1:
            feat_encoding = feat_encoding.squeeze(1)
        if sim_encoding.ndim > 1:
            sim_encoding = sim_encoding.squeeze(1)

    eps = 1e-4
    pred_diff = torch.clamp(1.0 / (feat_encoding + eps), max=50.0)
    loss_cls_clamped = torch.clamp(classification_loss, max=50.0)
    loss_lambda = F.mse_loss(pred_diff, loss_cls_clamped)

    probs = F.softmax(similarity_scores, dim=1)
    sim_ent = torch.clamp(_entropy(probs), max=10.0)
    target_lam = torch.clamp(1.0 / (sim_ent + eps), max=5.0)
    loss_consistency = F.mse_loss(sim_encoding, target_lam)

    reg_w = 0.010
    lb, ub = 0.2, 10.0
    loss_reg = reg_w * (
        torch.relu(lb - feat_encoding).mean()
        + torch.relu(lb - sim_encoding).mean()
        + torch.relu(feat_encoding - ub).mean() * 0.1
        + torch.relu(sim_encoding - ub).mean() * 0.1
        + torch.abs(feat_encoding - sim_encoding).mean() * 0.05
    )
    return loss_lambda + 0.5 * loss_consistency + loss_reg


def meh_loss_v2_log(
    self,
    classification_loss: torch.Tensor,
    feat_encoding: torch.Tensor,
    sim_encoding: torch.Tensor,
    similarity_scores: torch.Tensor,
) -> torch.Tensor:
    """MEH loss (log form): -log(lambda) as difficulty proxy."""
    if getattr(self.cfg.TRAINER.COOPAL, 'Ablation', 'None') == 'None':
        if feat_encoding.ndim > 1:
            feat_encoding = feat_encoding.squeeze(1)
        if sim_encoding.ndim > 1:
            sim_encoding = sim_encoding.squeeze(1)

    eps = 1e-6
    loss_lambda = F.mse_loss(-torch.log(feat_encoding + eps), classification_loss)
    probs = F.softmax(similarity_scores, dim=1)
    sim_ent = _entropy(probs)
    loss_consistency = F.mse_loss(-torch.log(sim_encoding + eps), sim_ent)

    reg_w = 0.008
    lb, ub = 0.3, 6.0
    loss_reg = reg_w * (
        torch.relu(lb - feat_encoding).mean()
        + torch.relu(lb - sim_encoding).mean()
        + torch.relu(feat_encoding - ub).mean() * 0.1
        + torch.relu(sim_encoding - ub).mean() * 0.1
        + torch.abs(feat_encoding - sim_encoding).mean() * 0.05
    )
    return loss_lambda + 0.5 * loss_consistency + loss_reg


# =============================================================================
# 不确定性计算
# =============================================================================

def calculate_vlm_uncertainties(
    vlm_similarities: torch.Tensor,
    lambda_evidence: torch.Tensor,
    num_samples_for_epistemic: int = 50,
) -> dict:
    """Compute vacuity, dissonance, epistemic uncertainty from VLM logits + MEH lambda."""
    if vlm_similarities.ndim != 2:
        raise ValueError("vlm_similarities must be 2D (batch_size, num_classes)")
    if lambda_evidence.ndim > 1:
        lambda_evidence = lambda_evidence.squeeze(-1)
    if lambda_evidence.ndim != 1 or lambda_evidence.shape[0] != vlm_similarities.shape[0]:
        raise ValueError("lambda_evidence shape mismatch")

    batch_size, num_classes = vlm_similarities.shape
    device = vlm_similarities.device

    p_ae = F.softmax(vlm_similarities, dim=1)
    evidence = lambda_evidence.unsqueeze(1) * p_ae
    alpha = evidence + 1.0
    S = alpha.sum(dim=1)

    dirichlet_dist = Dirichlet(alpha)
    samples = dirichlet_dist.sample((num_samples_for_epistemic,))
    mean_probs = samples.mean(dim=0)
    entropy_of_mean = _entropy(mean_probs)
    mean_of_entropies = _entropy(samples).mean(dim=0)
    epistemic_uncertainty = entropy_of_mean - mean_of_entropies

    vacuity = num_classes / S

    belief_masses = evidence / S.unsqueeze(1)
    dissonance = torch.zeros(batch_size, device=device)
    for i in range(num_classes):
        b_i = belief_masses[:, i]
        mask = torch.ones(num_classes, dtype=torch.bool, device=device)
        mask[i] = False
        other_beliefs = belief_masses[:, mask]
        sum_other = other_beliefs.sum(dim=1)
        bal = 1 - torch.abs(other_beliefs - b_i.unsqueeze(1)) / (
            other_beliefs + b_i.unsqueeze(1) + 1e-12
        )
        inner_sum = (other_beliefs * bal).sum(dim=1)
        term = torch.where(
            sum_other > 1e-6,
            b_i * inner_sum / sum_other,
            torch.zeros_like(b_i),
        )
        dissonance += term

    return {
        'epistemic': epistemic_uncertainty,
        'vacuity': vacuity,
        'dissonance': dissonance,
        'alpha': alpha,
        'S': S,
    }


# =============================================================================
# 归一化工具
# =============================================================================

def min_max_normalize(scores: torch.Tensor) -> torch.Tensor:
    """In-tensor (batch-level) min-max normalization to [0, 1]."""
    if not torch.is_tensor(scores):
        scores = torch.tensor(scores, dtype=torch.float32)
    mn, mx = scores.min(), scores.max()
    if torch.isclose(mn, mx):
        return torch.zeros_like(scores, dtype=torch.float32)
    return (scores - mn) / (mx - mn)


def _coop_get(cfg, key: str, default):
    """Safely retrieve a nested cfg attribute; return default if attribute missing."""
    try:
        parts = key.split('.')
        node = cfg
        for p in parts:
            node = getattr(node, p)
        return node
    except AttributeError:
        return default


def global_min_max_1d(t: torch.Tensor) -> torch.Tensor:
    """Pool-level min-max normalization to [0, 1]."""
    mn, mx = t.min(), t.max()
    if torch.isclose(mn, mx):
        return torch.zeros_like(t)
    return (t - mn) / (mx - mn)


def global_zscore_1d(t: torch.Tensor) -> torch.Tensor:
    """Pool-level z-score normalization."""
    mu, sigma = t.mean(), t.std()
    if sigma < 1e-8:
        return torch.zeros_like(t)
    return (t - mu) / sigma


def _normalize_vac_dis_global(
    vac: torch.Tensor,
    dis: torch.Tensor,
    mode: str,
):
    """Normalize full-pool vac and dis tensors according to ACQ_NORM_MODE."""
    if mode == 'global_minmax':
        return global_min_max_1d(vac), global_min_max_1d(dis)
    elif mode == 'global_zscore':
        return global_zscore_1d(vac), global_zscore_1d(dis)
    elif mode == 'batch_minmax':
        # When called on the full pool this is equivalent to global minmax
        return global_min_max_1d(vac), global_min_max_1d(dis)
    else:
        raise ValueError(f"Unknown ACQ_NORM_MODE: {mode}")


# =============================================================================
# 打分函数
# =============================================================================

def build_acquisition_score(
    vac_norm: torch.Tensor,
    dis_norm: torch.Tensor,
    mode: str,
    current_round: int = 0,
    total_rounds: int = 1,
    cfg=None,
) -> torch.Tensor:
    """Combine normalized vac and dis into a scalar acquisition score per sample."""
    if mode == 'vac_plus_dis':
        return vac_norm + dis_norm
    elif mode == 'vac_minus_lambda_dis':
        lam = float(_coop_get(cfg, 'TRAINER.COOPAL.ACQ_LAMBDA_DIS', 0.5))
        return vac_norm - lam * dis_norm
    elif mode == 'legacy_wv_wd':
        if total_rounds <= 1:
            wv, wd = 1.0, 0.0
        else:
            t = max(1, current_round + 1)
            frac = (t - 1) / (total_rounds - 1)
            wv = 1.0 - frac
            wd = frac
        return wv * vac_norm + wd * dis_norm
    else:
        raise ValueError(f"Unknown ACQ_SCORE_MODE: {mode}")


# =============================================================================
# 均衡采集（BALANCED_QUERY，PCB 风格集成在 MEH_Selector 内）
# =============================================================================

def balanced_acquisition_from_scores(
    scores: torch.Tensor,
    n_query: int,
    pseudo_labels: torch.Tensor,
    num_classes: int,
    features: Optional[torch.Tensor] = None,
    use_kcenter: bool = False,
    topm_ratio: float = 3.0,
) -> List[int]:
    """
    Per-class balanced selection: each class gets floor(n_query/K) slots,
    remainder assigned to highest-score classes.
    Within each class, selects highest-scored candidates (optionally with K-Center).
    Returns local indices into scores / pseudo_labels.
    """
    base_per_class = n_query // num_classes
    remainder = n_query - base_per_class * num_classes

    class_max_score = {}
    for cls_id in range(num_classes):
        mask = (pseudo_labels == cls_id).nonzero(as_tuple=False).squeeze(1)
        class_max_score[cls_id] = scores[mask].max().item() if len(mask) > 0 else -float('inf')

    sorted_cls = sorted(class_max_score, key=class_max_score.get, reverse=True)
    extra_classes = set(sorted_cls[:remainder])

    selected: List[int] = []
    selected_set: set = set()
    for cls_id in range(num_classes):
        budget = base_per_class + (1 if cls_id in extra_classes else 0)
        if budget == 0:
            continue
        cls_mask = (pseudo_labels == cls_id).nonzero(as_tuple=False).squeeze(1)
        if len(cls_mask) == 0:
            continue
        cls_budget = min(budget, len(cls_mask))

        if use_kcenter and features is not None:
            topm_n = min(int(cls_budget * topm_ratio), len(cls_mask))
            _, top_m = torch.topk(scores[cls_mask], topm_n)
            cand_feat = features[cls_mask[top_m]]
            kc = greedy_kcenter_cosine(cand_feat, cls_budget)
            chosen = cls_mask[top_m[kc]].tolist()
        else:
            _, top_in_cls = torch.topk(scores[cls_mask], cls_budget)
            chosen = cls_mask[top_in_cls].tolist()

        for idx in chosen:
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)

    # Fill gaps if sparse classes led to fewer than n_query
    if len(selected) < n_query:
        for idx in torch.argsort(scores, descending=True).tolist():
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
            if len(selected) >= n_query:
                break

    return selected[:n_query]


def balanced_acquisition_inclass_kcenter(
    scores: torch.Tensor,
    n_query: int,
    pseudo_labels: torch.Tensor,
    num_classes: int,
    features: torch.Tensor,
    topm_ratio: float = 3.0,
) -> List[int]:
    """Per-class K-Center: balanced selection with intra-class diversity."""
    return balanced_acquisition_from_scores(
        scores=scores,
        n_query=n_query,
        pseudo_labels=pseudo_labels,
        num_classes=num_classes,
        features=features,
        use_kcenter=True,
        topm_ratio=topm_ratio,
    )


def greedy_kcenter_cosine(features: torch.Tensor, n_select: int) -> List[int]:
    """Greedy K-Center using cosine distance. Returns indices into features tensor."""
    n = features.shape[0]
    if n_select >= n:
        return list(range(n))
    n_select = max(1, n_select)

    normed = F.normalize(features.float(), dim=1)  # (N, D)
    selected = [0]
    min_dist = torch.full((n,), float('inf'), device=features.device)

    for _ in range(n_select - 1):
        last = normed[selected[-1]].unsqueeze(0)   # (1, D)
        cos_sim = (normed @ last.T).squeeze(1)      # (N,)
        dist = 1.0 - cos_sim                        # cosine distance
        min_dist = torch.minimum(min_dist, dist)
        for s in selected:
            min_dist[s] = -float('inf')
        selected.append(int(min_dist.argmax().item()))

    return selected


# =============================================================================
# MEH_Selector：主动学习样本选择器
# =============================================================================

class MEH_Selector(AL):
    """
    MEH-based active learning selector supporting:
      - ACQ_NORM_MODE: batch_minmax | global_minmax | global_zscore
      - ACQ_SCORE_MODE: legacy_wv_wd | vac_plus_dis | vac_minus_lambda_dis
      - BALANCED_QUERY: per-class balanced selection (PCB-style, integrated)
      - SAE_CA.USE_KCENTER: greedy K-Center diversity on feature space
    Falls back gracefully if config keys are absent (legacy_wv_wd + batch_minmax).
    """

    def __init__(
        self,
        cfg,
        model,
        meh_net,
        unlabeled_dst,
        U_index,
        n_class,
        current_round: int,
        total_rounds: int,
        device,
        meh_version: str = 'v2',
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device
        self.meh_net = meh_net
        self.meh_version = getattr(cfg.TRAINER.COOPAL, 'MEH_VERSION', meh_version)
        self.temperature = temperature
        self.current_round = current_round
        self.total_rounds = total_rounds

    def select(
        self,
        n_query: int,
        labeled_counts: Optional[Dict[int, int]] = None,
        labeled_feats_by_class: Optional[Dict[int, torch.Tensor]] = None,
        pi_pool_ema: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[int]:
        """
        Returns list of n_query global indices (from U_index).

        额外参数（LT-SaE B1-B3 路径需要，B0+ 路径忽略）
        -----------------------------------------------
        labeled_counts         : {class_id: count}，已标注集类别计数
        labeled_feats_by_class : {class_id: (M, D)}，已标注集 frozen visual features（B3 使用）
        pi_pool_ema            : (K,) 上一轮 pool pseudo prior EMA（B2 使用）
        """
        cfg = self.cfg
        acq_norm  = _coop_get(cfg, 'TRAINER.COOPAL.ACQ_NORM_MODE',  'batch_minmax')
        acq_score = _coop_get(cfg, 'TRAINER.COOPAL.ACQ_SCORE_MODE', 'legacy_wv_wd')
        balanced  = _coop_get(cfg, 'TRAINER.COOPAL.BALANCED_QUERY', False)
        use_kc    = _coop_get(cfg, 'TRAINER.COOPAL.SAE_CA.USE_KCENTER', False)
        topm      = float(_coop_get(cfg, 'TRAINER.COOPAL.SAE_CA.TOPM_RATIO', 10.0))

        self.model.eval()
        self.meh_net.eval()

        loader = build_data_loader(
            cfg,
            data_source=self.unlabeled_set,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            is_train=False,
            tfm=build_transform(cfg, is_train=False),
        )

        all_vac:      List[torch.Tensor] = []
        all_dis:      List[torch.Tensor] = []
        all_feats:    List[torch.Tensor] = []
        all_pseudo:   List[torch.Tensor] = []
        all_probs:    List[torch.Tensor] = []
        batch_scores: List[torch.Tensor] = []   # B0+ batch_minmax scores

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"MEH [{acq_norm}/{acq_score}]"):
                inputs = batch["img"].to(self.device)
                vlm_sim, features = self.model(inputs, get_feature=True)

                logits_orig = (
                    self.model.module.original_logits
                    if isinstance(self.model, nn.DataParallel)
                    else self.model.original_logits
                )

                if self.meh_version != 'v2':
                    raise ValueError(f"Unsupported MEH_VERSION: {self.meh_version}")
                feat_enc, sim_enc = self.meh_net(features, vlm_sim)
                lam = torch.clamp(feat_enc + sim_enc, min=0.1, max=10.0)
                unc = calculate_vlm_uncertainties(logits_orig, lam)

                vac_b  = unc['vacuity'].cpu()
                dis_b  = unc['dissonance'].cpu()
                feat_b = features.cpu()
                pseu_b = vlm_sim.argmax(dim=1).cpu()
                prob_b = torch.softmax(logits_orig, dim=1).cpu()

                all_vac.append(vac_b)
                all_dis.append(dis_b)
                all_feats.append(feat_b)
                all_pseudo.append(pseu_b)
                all_probs.append(prob_b)
                if acq_norm == 'batch_minmax':
                    vn = min_max_normalize(vac_b)
                    dn = min_max_normalize(dis_b)
                    batch_scores.append(build_acquisition_score(
                        vn, dn, acq_score,
                        current_round=self.current_round,
                        total_rounds=self.total_rounds,
                        cfg=cfg,
                    ))

        all_vac_t     = torch.cat(all_vac)
        all_dis_t     = torch.cat(all_dis)
        all_feats_t   = torch.cat(all_feats)
        all_pseudo_t  = torch.cat(all_pseudo)
        all_probs_t   = torch.cat(all_probs)
        print("\n" + "=" * 60)
        print(f"[MEH_Selector] norm={acq_norm}, score={acq_score}, "
              f"balanced={balanced}, kcenter={use_kc}, "
              f"round={self.current_round}/{self.total_rounds}")
        print(f"  Vac   mean={all_vac_t.mean():.4f}  std={all_vac_t.std():.4f}  "
              f"min={all_vac_t.min():.4f}  max={all_vac_t.max():.4f}")
        print(f"  Dis   mean={all_dis_t.mean():.6f}  std={all_dis_t.std():.6f}")
        print("=" * 60 + "\n")

        # ── Conference SaE acquisition path ──
        if acq_norm == 'batch_minmax':
            scores = torch.cat(batch_scores)
        else:
            vac_n, dis_n = _normalize_vac_dis_global(all_vac_t, all_dis_t, acq_norm)
            scores = build_acquisition_score(
                vac_n, dis_n, acq_score,
                current_round=self.current_round,
                total_rounds=self.total_rounds,
                cfg=cfg,
            )

        if balanced:
            local_idx = balanced_acquisition_from_scores(
                scores=scores,
                n_query=n_query,
                pseudo_labels=all_pseudo_t,
                num_classes=self.n_class,
                features=all_feats_t if use_kc else None,
                use_kcenter=use_kc,
                topm_ratio=topm,
            )
        elif use_kc:
            topm_n = min(int(n_query * topm), len(self.U_index))
            _, top_m = torch.topk(scores, topm_n)
            kc = greedy_kcenter_cosine(all_feats_t[top_m], n_query)
            local_idx = top_m[kc].tolist()
        else:
            _, top_idx = torch.topk(scores, n_query)
            local_idx = top_idx.tolist()

        global_idx = [self.U_index[i] for i in local_idx]

        # Conference SaE: 与 B2/B4 对齐的每轮采集 JSON（Spearman、真标签分布等）
        try:
            import json as _json
            import os as _os
            from scipy.stats import spearmanr as _spearmanr

            _ent = -(all_probs_t.clamp(1e-8) * all_probs_t.clamp(1e-8).log()).sum(dim=1)
            try:
                rho_e, p_e = _spearmanr(scores.detach().float().numpy(), _ent.detach().numpy())
                _spe_ent = {"rho": float(rho_e), "p": float(p_e)}
            except Exception as _se:
                _spe_ent = {"rho": float("nan"), "p": float("nan"), "error": str(_se)}

            acq_b0: Dict = {
                "round_id": self.current_round,
                "num_classes": self.n_class,
                "acquisition_path": (
                    "b0plus_balanced"
                    if balanced
                    else ("b0plus_kcenter" if use_kc else "b0plus_topk")
                ),
                "spearman_score_vs_entropy": _spe_ent,
                "b4_alpha_mean_per_class": None,
                "b4_alpha_max": None,
                "selected_local_idx": local_idx,
                "selected_global_idx": global_idx,
            }
            _pseu_sel = all_pseudo_t[local_idx]
            acq_b0["selected_pseudo_dist"] = _pseu_sel.bincount(
                minlength=self.n_class
            ).tolist()

            try:
                true_labels = [self.unlabeled_set[i].label for i in local_idx]
                num_cls = self.n_class
                true_dist = [0] * num_cls
                for lbl in true_labels:
                    if 0 <= lbl < num_cls:
                        true_dist[lbl] += 1
                acq_b0["selected_true_dist"] = true_dist
                dataset_name = _coop_get(cfg, "DATASET.NAME", "")
                if "BUSI" in dataset_name:
                    acq_b0["special_class_true_selected"] = {
                        "malignant(label1)": true_dist[1] if num_cls > 1 else 0,
                        "normal(label2_tail)": true_dist[2] if num_cls > 2 else 0,
                        "benign(label0_head)": true_dist[0] if num_cls > 0 else 0,
                    }
                elif "RETINA" in dataset_name:
                    acq_b0["special_class_true_selected"] = {
                        "glaucoma(label1_tail)": true_dist[1] if num_cls > 1 else 0,
                        "diabetic_retinopathy(label0)": true_dist[0] if num_cls > 0 else 0,
                        "cataract(label2)": true_dist[2] if num_cls > 2 else 0,
                        "normal_retina(label3_head)": true_dist[3] if num_cls > 3 else 0,
                    }
                elif "ISIC" in dataset_name:
                    acq_b0["special_class_true_selected"] = {
                        "df(label5)": true_dist[5] if num_cls > 5 else 0,
                        "vasc(label6)": true_dist[6] if num_cls > 6 else 0,
                        "scc(label7)": true_dist[7] if num_cls > 7 else 0,
                        "akiec(label3)": true_dist[3] if num_cls > 3 else 0,
                    }
            except Exception as _te:
                acq_b0["true_label_error"] = str(_te)

            log_dir = _os.path.join(cfg.OUTPUT_DIR, "al_round_logs")
            _os.makedirs(log_dir, exist_ok=True)
            log_path = _os.path.join(log_dir, f"round_{self.current_round:02d}.json")
            with open(log_path, "w") as _f:
                _json.dump(
                    acq_b0,
                    _f,
                    indent=2,
                    default=lambda x: x
                    if isinstance(x, (int, float, str, list, dict, type(None), bool))
                    else str(x),
                )
            print(f"[B0+ ACQ] round log saved → {log_path}")
        except Exception as _log_e:
            print(f"[B0+ ACQ] WARNING: could not save round log: {_log_e}")

        return global_idx

