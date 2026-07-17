# exp076 TDPC 代码审查记录

## 第一轮审查
- **结果**: 未通过
- **发现问题**:
  - **C1 (Critical)**: `_prepare_pose()` 中 `distractor_mask` shape 不匹配。`person_mask[:, 1:]` 需要 3 次 unsqueeze 才能与 5D heatmap tensor 广播，但只做了 2 次。
  - **C2 (Critical)**: `_prepare_pose()` 返回值从 3 个改为 4 个，但 `PosePSGPartModel` 仍解包 3 个值，会崩溃。
  - **W1 (Medium)**: ReLU 在 tanh 之后可能丢弃有用的负通道信号（设计选择，不阻塞）。

## 第二轮审查
- **结果**: 未通过
- **C1/C2 修复确认**: 正确
- **新发现问题**:
  - **C3 (Critical)**: `self.use_pcl` 赋值行丢失，会导致所有 PSG 代码路径 AttributeError。
  - **L1 (Medium)**: `processor.py` PAMC 路径仍解包 3 值（虽然当前不触发）。

## 第三轮审查
- **结果**: ✅ 通过
- **所有 4 个 fix 已验证正确**:
  - C1: 3 unsqueezes → (B, P-1, 1, 1, 1)
  - C2: PosePSGPartModel 4-value unpack
  - C3: self.use_pcl 赋值恢复
  - L1: processor.py 4-value unpack
- **全局搜索确认**: 所有 `PoseBackboneModel._prepare_pose` 调用者都正确解包 4 值
- **结论**: 审查通过，可以开始训练
