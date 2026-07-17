# 实验 exp207: Swin-Base + GCN+PAA+CE+OA-SD (Backbone Scaling)

## 动机
- 目标: 76% mAP / 85% R1 on Occluded-Duke, 无 NFC/reranking
- 当前最佳: Small GCN+PAA+OA-SD = 70.5/82.3
- Gap: +5.5/+2.7
- KPR (ECCV 2024) 用 Swin-Base (88M): 73.3/82.5 → Base 比 Small 高 ~3%
- 如果我们的方法在 Base 上也有类似 scaling: 70.5 + 3 = **73.5% mAP**
- 再加 OA-SD 等创新 → 可能 **75%+**

## 核心假设
Swin-Base (88M) 相比 Small (50M) 带来 +2-3% mAP 的 backbone capacity boost。
我们的 GCN+PAA+OA-SD 方法应在更大 backbone 上有类似或更大的增益。

## 技术方案
- `swin_base_patch4_window7_224` backbone (depths=[2,2,18,2], embed_dim=128)
- `pretrained/swin_base.pth` 权重
- 配置同 exp206: GCN+PAA+ROA+CE+OA-SD+PLBOA
- LR=0.0002 (Base 更大模型用更小 LR)
- 1-view (Base 在 24GB 上可能需要 CP)

### 显存估计
- Swin-Base 特征维度: 1024 (vs Small 768, Tiny 768)
- 估计 1-view: ~20-22GB → 3090 可能勉强 OK
- 如 OOM 则加 WITH_CP=True

## 预期结果
- 假设成立: mAP 73-75%, R1 83-85%
- 如果 OOM: 加 CP 重跑

## 对照组
- exp206 (Small + GCN+PAA+OA-SD): 70.5/82.3
- KPR (Base, different method): 73.3/82.5
