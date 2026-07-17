# 实验 exp151: PVAT（Pose-Visibility Adversarial Training）

## 动机

`exp109` 的核心发现：`single-image support incomplete` 是遮挡 ReID 的根本问题。

PCVT（exp148）从"数据增强"角度解决：创建互补视图训练多样性。
PVAT 从"表示学习"角度解决：让最终特征不携带可见性信息，从而对遮挡模式不变。

核心洞察：在 test-time，模型不知道 gallery 图的哪些部位可见。
如果 query 特征编码了"我的左臂不可见"这一信息，这对 gallery 匹配反而是噪声。
理想的特征应该只编码 identity 信息，不编码 visibility 信息。

## 核心假设

1. 当前特征隐式编码了 visibility 模式（哪些部位可见/遮挡）
2. 这些 visibility 信息对 test-time 检索是噪声
3. 通过梯度反转训练，强制特征不携带 visibility 信息，应提升跨遮挡模式的匹配稳定性
4. 这与 PSG 不矛盾：PSG 用 pose 引导特征提取（过程），PVAT 确保最终特征不泄露 visibility（结果）

## 技术方案

### 1. Gradient Reversal Layer

```python
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None
```

### 2. Visibility Predictor Head

```
global_feat (768-d) → GradientReversal → Linear(768, 17) → Sigmoid → visibility_pred (17-d)
```

- 仅 768×17 + 17 = 13,073 个参数
- 目标：预测 17 个 COCO keypoint 的可见性（二分类）
- 可见性 ground truth 来自 `pose_dict['scores']`，score > 0.5 → visible

### 3. 损失

```
L_total = L_reid + λ_vis * L_vis_adv
L_vis_adv = BCE(visibility_pred, visibility_gt)
```

- 通过 gradient reversal，这个 loss 的梯度对 backbone 是反向的
- backbone 被训练去"欺骗" visibility predictor
- predictor 被训练去"看穿" backbone 的特征
- 纳什均衡点：特征不携带 visibility 信息，predictor 无法预测

### 4. 训练细节

- `λ_vis = 0.1`（初始值，可能需要调整）
- Gradient reversal strength `α` 使用 warmup：
  - ep1-20: `α = 0`（不对抗，让 backbone 先正常训练）
  - ep20-120: `α` 线性从 0 升到 1.0
- 这避免了早期对抗训练干扰 backbone 收敛

### 5. 不改动的部分

- PSG + GCN + equal_concat 主架构不变
- ID + Triplet 主损失不变
- 0.5x global loss 不变
- 测试时完全不需要 visibility predictor

## 对照组

- 主基线：`exp030a-eq`
- 直接对照：`exp148 PCVT`（数据增强范式 vs 表示学习范式）
- 历史对照：
  - `exp062 LKU`：learned uncertainty，中性偏负
  - `exp047 CSGT`：common-support mining，失败

PVAT 与这些的区别：
- LKU：用 uncertainty 做加权（正向利用 visibility），PVAT 做对抗（移除 visibility）
- CSGT：用 overlap 做 mining filter，PVAT 改变特征表示本身

## 预期结果

### 场景 1: PVAT 有效（mAP > exp030a-eq 3-seed mean 60.73%）
- 说明 visibility-invariant 特征确实更好
- 论文可以写："adversarial visibility removal forces the model to encode only identity-discriminative information"

### 场景 2: PVAT 中性（与 exp030a-eq 持平）
- 说明 visibility 信息既不帮助也不妨碍
- 有消融价值

### 场景 3: PVAT 有害
- 说明 visibility 信息对当前模型是有用的（PSG 的 pose gating 使得 visibility 编码是有益的）
- 同样有消融价值——证明 PSG 的 pose-aware 特征是有意义的

## 关键日志

- `pvat_loss`: visibility adversarial loss
- `pvat_acc`: visibility 预测准确率
- `pvat_alpha`: 当前 gradient reversal 强度
- `pvat_vis_ratio`: batch 中平均可见关键点比例

如果 `pvat_acc` 在训练后期仍然很高 → gradient reversal 不够强
如果 `pvat_acc` 降到 ~0.5（随机猜） → 特征已成功隐藏 visibility

## 风险

1. 对抗训练不稳定：可能导致训练震荡
2. 移除有用信息：如果 visibility 对身份识别有价值，移除它会降低性能
3. α 调参敏感：gradient reversal 强度需要仔细调
