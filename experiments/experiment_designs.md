# 实验设计 (Dry Run)

---

## exp001: PAMS v9 Baseline (完整 120 epoch)

### 目的
确认 PAMS 的完整训练性能，作为后续所有改进的 baseline。

### 配置
使用已有的 `configs/occluded_duke/pams_tiny.yml`，无修改。

### 运行命令
```bash
python train.py --config_file configs/occluded_duke/pams_tiny.yml \
  DATASETS.ROOT_DIR '/path/to/occluded_duke' \
  OUTPUT_DIR './log/occluded_duke/exp001_pams_v9'
```

### 预期结果
- global mAP: 55-60%
- parts mAP: 50-58%
- 训练应完全稳定（v8 的 L2 norm + soft margin 已解决爆炸）

### 论文用途
- 主实验表格中的 "Ours (base)" 行
- 消融实验的基准线

---

## exp002: Swin-Tiny Baseline (无 pose, 完整 120 epoch)

### 目的
确认纯 Swin-Tiny 的性能，量化 PAMS 带来的提升。

### 配置
使用 `configs/occluded_duke/swin_tiny.yml`，无修改。

### 运行命令
```bash
python train.py --config_file configs/occluded_duke/swin_tiny.yml \
  DATASETS.ROOT_DIR '/path/to/occluded_duke' \
  OUTPUT_DIR './log/occluded_duke/exp002_baseline'
```

### 预期结果
- global mAP: ~55% (与之前 E0 一致)

### 论文用途
- 主实验表格中的 "Baseline" 行

---

## exp003: PAMS + Soft BPA

### 目的
验证创新点 1: 软标签监督是否优于硬标签。

### 代码修改

**文件: `model/backbones/pams.py`**

修改 `build_bpa_target()` 函数，添加软标签模式:

```python
def build_bpa_target(heatmaps, visibility, target_hw, vis_threshold=0.5,
                     soft=False, temperature=0.5):
    """Build BPA targets from pose heatmaps.

    When soft=True, returns soft probability targets [B, K+1, H, W]
    instead of hard argmax labels [B, H, W].
    """
    hm = F.interpolate(heatmaps, target_hw, mode='bilinear', align_corners=False)

    # Mask out invisible keypoints
    vis_mask = (visibility > vis_threshold).float()
    hm = hm * vis_mask.unsqueeze(-1).unsqueeze(-1)

    # Aggregate keypoints into body part groups
    part_maps = []
    for group in COCO_PART_GROUPS:
        part_hm = hm[:, group].max(dim=1)[0]
        part_maps.append(part_hm)
    part_maps = torch.stack(part_maps, dim=1)  # [B, K, H, W]

    # Background
    bg = 1.0 - part_maps.max(dim=1)[0]
    bg = bg.clamp(min=0.0)

    all_maps = torch.cat([bg.unsqueeze(1), part_maps], dim=1)  # [B, K+1, H, W]

    if soft:
        # Soft targets: temperature-scaled softmax
        return F.softmax(all_maps / temperature, dim=1)  # [B, K+1, H, W]
    else:
        return all_maps.argmax(dim=1)  # [B, H, W]
```

**文件: `loss/make_loss.py`**

修改 BPA loss 计算:

```python
# 在 PAMS loss function 中:
if extras and 'bpa_logits' in extras:
    if soft_bpa:
        # KL divergence for soft targets
        log_probs = F.log_softmax(extras['bpa_logits'].float(), dim=1)
        soft_targets = extras['bpa_targets']  # [B, K+1, H, W]
        bpa_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
    else:
        bpa_loss = F.cross_entropy(extras['bpa_logits'].float(), extras['bpa_targets'])
    total = total + pams_bpa_w * bpa_loss
```

**文件: `config/defaults.py`**

添加:
```python
_C.MODEL.PAMS.SOFT_BPA = False
_C.MODEL.PAMS.BPA_TEMPERATURE = 0.5
```

### 配置 (新 yml 文件)
```yaml
# configs/occluded_duke/exp003_soft_bpa.yml
# 基于 pams_tiny.yml, 修改:
MODEL:
  PAMS:
    SOFT_BPA: True
    BPA_TEMPERATURE: 0.5
OUTPUT_DIR: './log/occluded_duke/exp003_soft_bpa'
```

### 运行命令
```bash
python train.py --config_file configs/occluded_duke/exp003_soft_bpa.yml \
  DATASETS.ROOT_DIR '/path/to/occluded_duke'
```

### 预期结果
- mAP 比 exp001 提升 1-3%
- Part attention map 在遮挡边界应更平滑

### 消融变量
- temperature: 0.1, 0.3, 0.5, 1.0
- 可以做温度敏感性分析（论文图表）

### 论文用途
- 消融表格: "Soft BPA" vs "Hard BPA" 对比
- 可视化: Part attention map 对比

---

## exp004: PAMS + NFC 后处理

### 目的
验证 Pose2ID 的 NFC 方法在我们 PAMS 特征上的效果。

### 代码修改

**新文件: `utils/nfc.py`**

```python
import torch
import torch.nn.functional as F

def pairwise_distance(x, y=None):
    """Pairwise euclidean distance."""
    if y is None:
        y = x
    m, n = x.size(0), y.size(0)
    xx = x.pow(2).sum(1, keepdim=True).expand(m, n)
    yy = y.pow(2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * x.mm(y.t())
    return dist.clamp(min=1e-12).sqrt()

def nfc(feat, k1=2, k2=2):
    """Neighbor Feature Centralization (from Pose2ID).

    For each sample, finds mutual k-nearest neighbors and
    averages features with them.

    Args:
        feat: [N, D] feature matrix
        k1: number of nearest neighbors to consider
        k2: reciprocal condition threshold
    Returns:
        centralized features [N, D]
    """
    N = feat.size(0)
    dist = pairwise_distance(feat)

    # Top-k nearest neighbors (k1+1 because first is self)
    _, rank = dist.topk(k1 + 1, largest=False)
    rank = rank[:, 1:]  # remove self

    # Find mutual nearest neighbors
    feat_new = feat.clone()
    for i in range(N):
        neighbors = rank[i]
        mutual = []
        for j in neighbors:
            j_neighbors = rank[j.item()][:k2]
            if i in j_neighbors:
                mutual.append(j.item())
        if mutual:
            mutual_feats = feat[mutual]
            feat_new[i] = feat[i] + mutual_feats.sum(0)

    return F.normalize(feat_new, p=2, dim=1)

def part_nfc(part_feats, part_vis, k1=2, k2=2, vis_threshold=0.1):
    """Part-level NFC: apply NFC per body part with visibility filtering.

    Args:
        part_feats: [N, K, D]
        part_vis: [N, K]
        k1, k2: NFC hyperparameters
        vis_threshold: minimum visibility to include a sample
    Returns:
        centralized part features [N, K, D]
    """
    N, K, D = part_feats.shape
    result = part_feats.clone()

    for k in range(K):
        # Only use samples where part k is visible
        visible = part_vis[:, k] > vis_threshold
        if visible.sum() < 3:
            continue

        vis_indices = torch.where(visible)[0]
        vis_feats = part_feats[vis_indices, k]  # [M, D]

        centralized = nfc(vis_feats, k1, k2)
        result[vis_indices, k] = centralized

    return result
```

**修改: `utils/metrics.py`**

在 `R1_mAP_eval.compute()` 中添加 NFC 后处理选项:

```python
# 在计算距离之前:
if self.use_nfc and 'parts' in feats:
    from utils.nfc import part_nfc
    feats['parts'] = part_nfc(feats['parts'], feats['part_vis'])
```

### 配置
```yaml
# configs/occluded_duke/exp004_nfc.yml
# 基于 pams_tiny.yml, 修改:
TEST:
  NFC: True
  NFC_K1: 2
  NFC_K2: 2
OUTPUT_DIR: './log/occluded_duke/exp004_nfc'
```

### 论文用途
- 主实验表格: "+NFC" 行
- 可与任何方法组合，作为论文的额外贡献点

---

## exp005: OA-PAMS (全部组件)

### 目的
验证 OA-PAMS 的完整方法。

### 代码修改

**新模块: Visibility-Guided Feature Calibration (VGFC)**

在 `model/backbones/pams.py` 中添加:

```python
class VisibilityGuidedCalibration(nn.Module):
    """Uses visibility scores to adaptively fuse global and part features.

    When visibility is low (heavy occlusion), rely more on global feature.
    When visibility is high, leverage fine-grained part features.
    """
    def __init__(self, n_parts, feat_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_parts, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, global_feat, part_feats, part_vis):
        """
        Args:
            global_feat: [B, D]
            part_feats: [B, K, D]
            part_vis: [B, K] visibility scores
        Returns:
            calibrated_feat: [B, D]
        """
        # Fusion weight based on overall visibility
        alpha = self.mlp(part_vis)  # [B, 1] — higher alpha = more global

        # Visibility-weighted part feature aggregation
        vis_weights = F.softmax(part_vis, dim=1)  # [B, K]
        weighted_parts = (part_feats * vis_weights.unsqueeze(2)).sum(dim=1)  # [B, D]

        # Adaptive fusion
        calibrated = alpha * global_feat + (1 - alpha) * weighted_parts
        return calibrated
```

**修改推理距离计算**

```python
def continuous_vis_weighted_distance(q_parts, g_parts, q_vis, g_vis):
    """Continuous visibility-weighted part distance.

    Instead of binary masking, uses continuous visibility scores.
    """
    K = q_parts.shape[1]
    Nq, Ng = q_parts.shape[0], g_parts.shape[0]

    # Per-part distances: [K, Nq, Ng]
    part_dists = []
    for k in range(K):
        d = pairwise_distance(q_parts[:, k], g_parts[:, k])
        part_dists.append(d)
    part_dists = torch.stack(part_dists, dim=0)

    # Continuous weight: geometric mean of query and gallery visibility
    # [K, Nq, Ng]
    weights = q_vis.t().unsqueeze(2) * g_vis.t().unsqueeze(1)  # [K, Nq, Ng]

    # Weighted average
    total_weight = weights.sum(0).clamp(min=1e-6)
    dist = (part_dists * weights).sum(0) / total_weight

    return dist
```

### 配置
```yaml
# configs/occluded_duke/exp005_oapams.yml
MODEL:
  PAMS:
    ENABLE: True
    SOFT_BPA: True
    BPA_TEMPERATURE: 0.5
    VGFC: True
    VGFC_HIDDEN_DIM: 64
TEST:
  CONTINUOUS_VIS_DIST: True
OUTPUT_DIR: './log/occluded_duke/exp005_oapams'
```

### 预期结果
- mAP 比 PAMS baseline 提升 2-5%
- 在高遮挡场景提升更明显

### 论文用途
- 主实验表格中的 "OA-PAMS (Ours)" 行
- 消融实验逐一关闭 Soft BPA / VGFC / Continuous Vis Distance

---

## exp006-010: 消融实验

### exp006: 消融 Soft BPA
- 配置: OA-PAMS 但 `SOFT_BPA: False`

### exp007: 消融 VGFC
- 配置: OA-PAMS 但 `VGFC: False`

### exp008: 消融 Continuous Vis Distance
- 配置: OA-PAMS 但 `CONTINUOUS_VIS_DIST: False`（用 binary visibility）

### exp009: BPA Temperature 敏感性
- 温度: 0.1, 0.3, 0.5, 0.7, 1.0
- 结果用折线图展示

### exp010: N_PARTS 敏感性
- 部件数: 3, 5, 7, 9
- 结果用折线图展示

---

## 实验优先级排序

| 优先级 | 实验 | 预计时间 | 依赖 |
|--------|------|---------|------|
| P0 | exp001 (PAMS baseline) | ~2h on 4090 | 无 |
| P0 | exp002 (Swin baseline) | ~2h on 4090 | 无 |
| P1 | exp003 (Soft BPA) | ~2h | exp001 |
| P1 | exp005 (OA-PAMS full) | ~2h | exp003 的代码改动 |
| P2 | exp004 (NFC) | ~2h | exp001 |
| P3 | exp006-010 (消融) | 各~2h | exp005 |

**建议**: exp001 和 exp002 可以并行跑（如果有两张卡），或者先跑 exp001。
