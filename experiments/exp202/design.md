# 实验 exp202: Swin-Small + SupCon + Full Architecture

## 动机
- Swin-Tiny (28M) 达到 ceiling ~64.9% mAP
- Swin-Small (50M) baseline 曾达 65.8% (PSG-only 67.8%)
- 用户明确要求尝试 Swin-Small
- 如果我们的 SupCon + full arch 方法在 Small 上有类似增益 (+6-8%)，可达 **72%+ mAP**

## 核心假设
SupCon + STD-PR + PSG + PLBOA + PAPE + multi-stage PSG 在 Swin-Small 上的增益至少与 Swin-Tiny 相当。

## 技术方案
- 配置基于 `swin_small.yml` + 所有 pose 组件
- **不用 3-view** (显存限制 24GB，Small 3-view 可能 OOM)
- **不用 OA-SD** (SupCon 路线，不加 distillation)
- BASE_LR=0.0004 (Small 标准 LR)
- CHECKPOINT_PERIOD=20

### 关键差异 vs Tiny
- `swin_small_patch4_window7_224` backbone
- `pretrained/swin_small.pth` 权重
- LR 0.0004 (半于 Tiny 的 0.0008)
- Stage 3 feature: 768ch (与 Tiny 相同 ch，但 18 blocks vs 6)

### 显存估计
- Swin-Small 1-view: ~12-14GB (Tiny 1-view ~8GB)
- +SupCon: +1-2GB
- +OA-SD: +6-8GB → 可能 OOM
- 结论：1-view + SupCon 应该 OK (~16GB)

## 预期结果
- exp176 (Tiny + SupCon 1-view): 64.1/75.5
- Small baseline: ~65.8 mAP
- **预期 Small + SupCon: ~70-72% mAP**

## 对照组
- exp187 (Tiny + SupCon + 3-view): 64.9/76.6
- Swin-Small baseline: ~65.8 mAP (之前数据)
