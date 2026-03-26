# 实验 exp197: Structural Token Mixup (STM)

## 动机
- 当前最强: exp187 (3v+SupCon) = 64.9/76.6, exp193 (3v+OA-SD+CE) = 64.4/76.5
- OA-SD 和 SupCon 互斥（exp196 负结果）
- 需要一个全新方向来突破当前天花板
- **核心观察**: STD-PR 产生 6 个 body-part structural tokens，但训练中每个 ID 的每张图只有一种 token 组合
- **创新**: 如果同 ID 的两张图可以交换部分 token（如图 A 的 head+torso + 图 B 的 legs），就能创造更多样的训练样本

## 核心假设
在 structural token 级别做 cross-instance mixup（同 ID 内）可以提升模型对部分可见的鲁棒性。这与 PLBOA（pixel-level occlusion）正交——PLBOA 创造"看不到某部分"的情况，STM 创造"看到不同人同一部分"的组合。

## 技术方案

### 核心机制
1. 每个 mini-batch 中，同 ID 的图像已经分组（sampler 保证每 ID 4 张）
2. 随机选择同 ID 的两张图的 structural tokens
3. 随机选择 K 个 body parts (K=1~3)，交换这些 parts 的 tokens
4. 用混合后的 token set 计算 loss（仍是同一个 ID，所以 label 不变）

### 数据流
```
Image A → Backbone+STD-PR → tokens_A = [global_A, head_A, torso_A, larm_A, rarm_A, lleg_A, rleg_A]
Image B → Backbone+STD-PR → tokens_B = [global_B, head_B, torso_B, larm_B, rarm_B, lleg_B, rleg_B]

Random swap legs: → mixed_A = [global_A, head_A, torso_A, larm_A, rarm_A, lleg_B, rleg_B]
                  → mixed_B = [global_B, head_B, torso_B, larm_B, rarm_B, lleg_A, rleg_A]

Loss: CE(mixed_A, ID) + triplet(mixed_A, ID) + CE(mixed_B, ID) + triplet(mixed_B, ID)
```

### 实现位置
- `processor/processor.py`: 在 loss 计算前，对同 ID 的样本做 token swap
- 新函数 `_structural_token_mixup(feat_list, labels, num_swap=2, prob=0.5)`
- 仅在训练时使用，测试时不变

### 关键超参
- `num_swap`: 每次交换的 body part 数量 (1~3)
- `mixup_prob`: 每个样本做 mixup 的概率 (0.5)
- 只在同 ID 内交换（保持 label 正确性）

## 预期结果
- 假设成立: mAP +0.5-1.0%，因为更多样的 token 组合提升泛化
- 如果失败: 可能因为交换 token 后的 feature 不一致（不同图的 backbone 特征空间不完全对齐）

## 对照组
- exp187 (3v+SupCon): 64.9/76.6 — 在此基础上加 STM
- exp193 (3v+OA-SD+CE): 64.4/76.5 — 在此基础上加 STM

## 创新门槛评估
1. ✅ 问题层面：重新定义训练样本——从"每图一种 token 组合"到"跨图 token 组合"
2. ✅ 机制层面：token-level mixup 在 ReID 中无先例（CutMix/Mixup 都是 pixel 级）
3. ✅ 证据层面：可设计消融（num_swap=0/1/2/3, prob, 同 ID vs 跨 ID）
