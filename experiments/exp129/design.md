# 实验 exp129: Residual-Correction SCRD

## 动机

- `exp119` 证明了 relational distillation 本身有效
- `exp120` 证明了 support-complete teacher 会显著改变 teacher 几何，但提升没有自动兑现
- `exp123/125` 又共同说明：teacher-change pairs 确实更重要，但当前收益仍然偏弱

这三步拼起来更像是在指向同一个问题：
**support-complete teacher 带来的增量信息太小，直接去拟合完整 teacher 几何时，这部分“修正量”会被 base teacher 的主体结构稀释。**

因此当前最合理的新假设不是继续改 sparse 强度，而是：
**只蒸馏 support-complete teacher 相对 base teacher 的“残差关系修正”。**

## 核心假设

1. `exp120` 之所以没把更强 teacher 兑现成更强结果，不是 teacher 无效，而是 distillation target 过于“完整”，把真正新增的 correction 淹没了
2. `exp125` 的 pair routing 能带来 late-stage 正向，说明 “changed pairs” 确实重要
3. 如果直接对齐 `dist_sc - dist_base` 这部分 residual relation，而不是继续对齐完整 `dist_sc`，那么 global embedding 更容易学到 support-complete 的真正新增信息

## 技术方案

相对 `exp125`，只改 `CSRD` 的 target 形式：

- `POSE_CSRD_TARGET_MODE: 'full' -> 'residual'`

其余保持与 `exp125` 完全一致：
- online support-complete teacher 不 freeze
- `PAIR_WEIGHT_MODE = delta_top`
- `PAIR_TOP_RATIO = 0.25`
- `PAIR_WEIGHT_ALPHA = 1.0`
- backbone / batch size / 主损失配比全部不变

### Residual-Correction CSRD

对每个 anchor 的正/负子集：

1. `dist_base`: 原始 keypoint teacher 几何
2. `dist_sc`: support-complete teacher 几何
3. `dist_s`: global student 几何
4. 构造 residual target:
   - teacher residual: `dist_sc - dist_base`
   - student residual: `dist_s - dist_base`
5. 只对 selected teacher-change pairs 做 residual regression

直觉上，它不再要求 global embedding 去复刻“整个 skeleton teacher”，而是只去学习 support completion 引入的那部分关系修正。

## 对照组

- 直接对照: `exp125 Sparse Pair-Delta SCRD`
- 机制对照: `exp123 Pair-Delta Focused SCRD`
- supporting 对照: `exp120 SCRD`
- 主基线: `exp030a`

## 预期结果

如果假设成立：

1. `ep30/40` 起应至少不弱于 `exp125`
2. `csrd_tr` 与 `csrd_gr` 会更清楚地区分 teacher correction 与 student correction
3. late-stage 应比 `exp125` 更容易兑现 mAP，而不是只表现为弱 trade-off

如果失败：

1. 说明当前瓶颈不是 target dilution
2. 更可能需要改的是 pair signal 本身，而不是 full-vs-residual target

## 风险与失败解释

1. 若 residual target 过小且噪声大，可能导致监督太弱
2. 若 `dist_base` 与 `dist_s` 的尺度差异过大，residual regression 可能不稳定
3. 若结果不如 `exp125`，说明完整 teacher 分布虽然“稀释”了 correction，但仍提供了必要的整体排序先验
