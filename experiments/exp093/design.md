# 实验 exp093: Pose-Guided Token Merging (PGTM)

## 动机 — 真正的范式级创新

当前所有方法（包括 PSG、PAA、GCN）都把 backbone 当黑盒：
- 输入 48 个空间 token
- 经过 self-attention (spatial patch attention)
- 输出 48 个 token → GAP → global feature

**根本问题**: self-attention 在 12×4=48 个 spatial patch 上操作。
遮挡的 patch 和可见的 patch 一样参与 attention——遮挡信息污染了特征。

PSG 用乘法 gate 缓解了这个问题，但本质没变——48 个 token 仍然全部参与。

## 核心创新
**在 Stage 3 内部把空间 token 合并成语义 body-part token**

用 pose heatmap 将 48 个 spatial patch token 分配到 5 个身体部位：
- head (nose, eyes, ears)
- upper_body (shoulders, torso)
- arms (elbows, wrists)  
- hips
- legs (knees, ankles)

每个部位内的 token 做 weighted average（权重来自 heatmap 响应）→ 1 个 body-part token

然后 5 个 body-part token 做 self-attention → body-part 间的关系推理

**效果**:
1. 遮挡部位没有高响应 token → 不贡献 → 天然遮挡鲁棒
2. Self-attention 在 5 个语义 token 上做 → 直接建模身体部位关系
3. 减少 token 数量 (48→5) → 更高效

## 这是真正没人做过的
- Token merging 在 ViT 加速中有（ToMe），但没人用 POSE 来指导 merging
- Part-based methods 在 feature 后处理中做 part pooling，但没人在 BACKBONE INSIDE 做 part merging
- DPEFormer 做 dynamic patch selection，但只是选择/删除，不是 merge into semantic tokens

## 技术方案
修改 Stage 3 SwinBlock:

```
Original:
  x (48, 768) → WindowAttention → FFN → x (48, 768)

PGTM:
  x (48, 768) → PSG gate (existing)
              → PGTM merge: (48, 768) → (5, 768) body-part tokens
              → PartAttention: self-attn on 5 tokens
              → PGTM expand: (5, 768) → (48, 768) scatter back
              → PSG gate result (residual)
```

PGTM merge:
  - 对每个 body part，计算 heatmap 在 12×4 grid 上的权重
  - 用加权平均合并 token

PGTM expand:
  - 把 body-part token 散回原始 48 个位置
  - 每个位置加权所属 body part 的 token

## 改动范围
- 新模块: `model/modules/pose_token_merge.py`
- 修改: `model/pose_backbone_model.py` 的 `_run_stage_with_psg`
- 不改 SwinBlock 本身——在 block 之后做 merge-attend-expand

## 对照
- exp066 PAA = 61.6%/74.2%
- exp030a 3-seed = 60.73%/72.57%
