# exp012: Pose-Conditioned Self-Attention (PCA)

## 动机
PSG (Pose Spatial Gate) 在 Stage 3 blocks 之后作用，只能调制特征值（feature values），不能改变 tokens 之间的信息交互模式（attention pattern）。PSG 的性能上限已被确认为 mAP 58.3%（exp007/009/011 三次独立验证）。

要突破这个上限，需要让 pose 信息参与更底层的计算 — 直接影响 self-attention 权重。

## 核心假设
将 pose heatmap 编码为 attention bias，注入 Swin 的 window self-attention 中，可以让 pose-relevant 的 token 之间产生更强的注意力连接，从而生成更好的 pose-aware 特征。

## 技术方案

### Pose Attention Bias (PAB) 模块
- 输入: scene_heatmaps (B, 17, Hh, Hw)
- 输出: pose_importance (B, num_heads, H, W)
- 结构: Conv2d(17→32→num_heads, 1×1, zero-init)
- resize heatmap → sigmoid → project to num_heads channels

### 注入方式
在 Swin Window MSA 中：
```
attn = Q @ K^T / sqrt(d)
attn += relative_position_bias     # 原有
attn += pose_attention_bias         # 新增！
attn = softmax(attn)
```

pose_attention_bias 使用 additive decomposition:
```
bias(i,j) = pose_val[i] + pose_val[j]
```
- 含义：可见部位的 token 作为 query 时更积极地关注其他 token
- 同时：可见部位的 token 作为 key 时更容易被其他 token 关注
- 效果最强：两个可见部位 token 之间的注意力被双重加强

### 修改的文件
1. `model/backbones/swin_transformer.py`: WindowMSA, ShiftWindowMSA, SwinBlock 接受可选 attention bias
2. `model/modules/pose_attention_bias.py`: 新 PAB 模块
3. `model/pose_backbone_model.py`: 使用 PAB 替代 PSG

### 关键超参数
- hidden_dim: 32 (PAB 内部)
- num_heads: 24 (Stage 3 的 head 数)
- zero-init: 是（保持 identity start）

## 预期结果
- 最优: mAP 59-60%（注意力级别的干预比特征级别更有效）
- 中性: mAP 58-58.5%（与 PSG 类似，换个方式但效果相同）
- 失败: mAP < 58%（注意力偏置干扰了 Swin 的 relative position bias）

## 对照组
- Baseline: exp007 (PSG Stage 3, mAP 58.3%, R1 67.9%)
- 消融变量: pose 信息的注入位置（attention bias vs post-block gate）
