# 实验 exp076: Target-Distractor Pose Conditioning (TDPC)

## 动机
- exp066 PAA 证明加法 pose adapter 在 PSG 基础上进一步有效 (+0.87% mAP / +1.63% R1 vs 3-seed)
- 但 PAA 使用 scene-level max-merge 热图，把 target 和 distractor 混在一起
- exp070 证明 naive target-only PAA 不如 scene PAA（scene context 对 suppress 很重要）
- 但 exp070 **不能**否定 target-distractor 区分的价值——它只说明硬切换太粗暴
- KPR / TTPM 等近期工作已把问题推进到 target ambiguity / non-target pedestrian occlusion
- 当前主线缺少"显式区分 target vs distractor"的机制，这是支撑 B 类论文的关键 gap

## 创新点 / 核心想法
- **核心假设**: 在多人遮挡场景中，除了 scene-level suppress（PSG+PAA 已做），还需要 target-distractor differential conditioning 来帮助 backbone 区分被检索人和干扰人
- **与 exp066 的区别**: exp066 PAA 只看 scene-level 热图；exp076 额外引入 target-distractor 差异信号
- **与 exp070 的区别**: exp070 直接把 PAA 从 scene 切到 target-only（硬替换）；exp076 **保留 scene PAA** 并额外添加 differential adapter（增量添加）

## 技术方案

### 核心改动
在 PSG + PAA (scene) 之后，增加一个 **Target-Distractor Differential Adapter (TDDA)**:

```
Stage 3 block output
    ↓
x = x * (1 + PSG_gate(H_scene))         # PSG: 乘性门控 (已有)
    ↓
x = x + PAA(H_scene)                     # PAA: 加法注入 (已有)
    ↓
x = x + TDDA(H_diff)                     # TDDA: 差异信号注入 (新增)
```

### 差异热图计算
```python
H_target = heatmaps[:, 0]                       # (B, 17, H, W) person 0
H_distractor = max_merge(heatmaps[:, 1:])        # (B, 17, H, W) 非目标人最大值
H_diff = H_target - H_distractor                 # (B, 17, H, W) 差异信号
```

**H_diff 的语义**:
- H_diff > 0 的区域: target person 独有的身体部位 → 应被强化
- H_diff < 0 的区域: distractor person 独有的身体部位 → 应被抑制
- H_diff ≈ 0 的区域: target 和 distractor 重叠 → 歧义区域

**单人图退化行为**:
- 只有 1 人时: H_distractor = 0 → H_diff = H_target ≈ H_scene
- TDDA 退化为第二个 PAA (但 zero-init 所以开始为恒等)

### TDDA 模块 (与 PAA 同构)
```python
class TDDAdapter(nn.Module):
    # 17 → bottleneck(32) → 768, zero-init output
    # 与 PoseAdditiveAdapter 完全相同结构
    # 输入: H_diff (17 channels, 可以是负值，不过 sigmoid)
```

**关键设计选择**:
- TDDA **不对输入做 sigmoid**（与 PAA 不同）。H_diff 取值 [-1, 1]，sigmoid 会压缩负值信息。改用 tanh 或直接输入。
- 使用 `tanh` 替代 `sigmoid` 以保留正负差异语义。
- Zero-init 确保安全退化。

### 修改文件清单
1. `model/modules/pose_additive_adapter.py`: 新增 `TDDAdapter` 类
2. `model/pose_backbone_model.py`:
   - `_prepare_pose()`: 计算 `distractor_heatmaps` 和 `diff_heatmaps`
   - `__init__()`: 根据 config 创建 TDDA 模块
   - `_run_stage_with_psg()`: 在 PAA 之后应用 TDDA
   - `forward()`: 传递 diff_heatmaps
3. `config/defaults.py`: 新增 `POSE_TDPC` 开关
4. `configs/occluded_duke/pose_psg_gcn_paa_tdpc.yml`: 新配置文件

### 关键超参数
- TDDA bottleneck_dim = 32 (与 PAA 相同)
- 输入激活: tanh (不用 sigmoid，保留差异信号的正负)
- zero-init output layer

## 预期结果
- 如果成功: mAP 进一步提升 0.5-1.0%（特别是在多人高歧义图上），R1 也改善
  - 理由: differential signal 让 backbone 在歧义区域更关注 target
- 如果失败:
  - 最可能原因 1: H_diff 信号在 12×4 分辨率上太粗糙，难以区分
  - 最可能原因 2: 单人图占比较高时，TDDA 大部分时间退化为 noise
  - 最可能原因 3: 51.8K 额外参数在 120 epoch 内不足以学到有意义的差异表示

## 对照组
- **Baseline 对照**: exp066 PAA seed1234 = 61.6% / 74.2%
- **3-seed 对照**: exp030a-eq 3-seed mean = 60.73% / 72.57%
- **消融变量**: 相对 exp066，只新增 TDDA 模块和 diff_heatmap 输入。其他完全不变：
  - PSG: scene heatmap (不变)
  - PAA: scene heatmap (不变)
  - GCN: 不变
  - 0.5x global loss: 不变
  - batch size / lr / epochs: 不变

## 数据统计
- 训练集: 26.4% 多人图 (4127/15618)，73.6% 单人图
- Query 集: 49.3% 多人图 (1090/2210)
- Gallery 集: 25.7% 多人图
- **影响**: 训练中 ~74% 样本的 diff_heatmaps 退化为 target_heatmaps（无 distractor）
  - TDDA 在这些样本上等效于第二个 PAA
  - 真正的 differential conditioning 只在 ~26% 训练样本上生效
  - 但 query 侧有 ~49% 多人图，测试时收益空间更大

## 风险与止损
- 如果 ep60 eval 明显低于 exp066 同期 → 分析 TDDA 的梯度和输出幅值
- 如果最终 mAP ≤ exp066 → 记录负结论，转入 retrieval-time common-support recovery (Plan B)
- 不做 TDPC 小变体扩散（bottleneck 调参、gating、multi-stage 等）
