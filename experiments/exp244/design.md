# exp244: LGPA-D — CLIP Part Head on Detached Features

## 动机

exp243 LGPA 发现两个关键事实:
1. CLIP 语义锚定加速 part assignment (ep20-40 +3.5~+4.1 mAP, 超过所有 PPA)
2. Non-detached cross-attention 梯度后期严重干扰 backbone (ep80 -1.1/-1.9)

如果 CLIP 的价值在语义初始化而非梯度传播, 那么在 detached features 上用 LGPA
应该能得到:
- 比 GCN 更好的 part pooling (CLIP 语义 + pose-conditioned attention)
- 没有 backbone 干扰 (detached)
- 如果成功: 证明 "CLIP 语义 > skeleton graph" 做 part assignment

## 核心假设

CLIP cross-attention 在 detached features 上比 GCN skeleton graph 能提取更好的 part features。

## 技术方案

与 exp243 完全相同的 CLIPPartHead, 但:
- `featmaps[-1].detach()` 传入 (替代 non-detached)
- 等效于把 LGPA 当作 GCN 的替代方案
- 其他一切不变: OA-SD, PLBOA, PSG, 0.5x global loss

### 代码修改

仅修改 `model/pose_backbone_model.py` 的 LGPA training path:
```python
# 原: featmaps[-1]  (non-detached)
# 改: featmaps[-1].detach()  (detached, like GCN)
```

## 对照组
- exp243 (LGPA non-detached): ep80 60.9/72.5 (-1.1/-1.9)
- exp191 (GCN detached): 63.2/75.4
- exp237 (PPA non-detached): 63.7/75.0 (+0.5/-0.4)

## 预期结果
- 成功: LGPA-D ≈ exp191 or better (63.2+ mAP), 证明 CLIP 做 part pooling 有效
- 失败: LGPA-D < exp191, 说明 CLIP cross-attention 在 detached 上不如 GCN
