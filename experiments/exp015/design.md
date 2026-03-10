# 实验 exp015: PSG with Spatial-Aware Gate (3×3 Depthwise Conv)

## 动机
- PSG Stage 3 是当前最佳方法 (mAP 58.3%, +1.7%)
- 当前 PSG gate 由纯 1×1 conv 构成: Conv1x1(17→64) → ReLU → Conv1x1(64→768) → sigmoid
- 每个空间位置独立计算 gate，不考虑邻域关系
- 但人体部件是连续的空间区域，相邻位置应该有相似的 gate 值
- 加 3×3 depthwise conv 让 gate 有空间感受野，产生更平滑、更一致的 gate 模式

## 核心假设
在 PSG gate 网络中加入 3×3 depthwise conv，使 gate 计算考虑空间邻域，产生更连贯的空间 gating pattern，提升特征质量。

## 技术方案
修改 `PoseSpatialGate`：
- 原始: Conv1x1(17→64) → ReLU → Conv1x1(64→768) → sigmoid
- 改进: Conv1x1(17→64) → ReLU → **DWConv3x3(64, padding=1)** → ReLU → Conv1x1(64→768) → sigmoid
- 额外参数: 64×3×3 = 576 参数（极少）
- 仍保持 zero-init（最后一层 Conv1x1 的 bias 初始化为 0）

### 修改文件
1. `model/modules/pose_spatial_gate.py` — 增加 depthwise conv 选项
2. `config/defaults.py` — 增加 `POSE_PSG_SPATIAL` 开关
3. `configs/occluded_duke/pose_psg_spatial.yml` — 新配置

### 关键超参数
- PSG hidden_dim: 64（与 exp007 相同）
- Depthwise conv kernel: 3×3, padding=1
- 其余与 exp007 完全一致

## 预期结果
- 如果假设成立：mAP 58.5-59.0%（+0.2-0.7% vs PSG-only）
- 如果失败：可能 12×4 分辨率下 3×3 邻域太大（覆盖 25% 宽度），空间平滑过度

## 对照组
- Baseline 对照: exp007 (PSG 1×1 only, mAP 58.3%)
- 消融变量: 仅增加 depthwise conv，其他不变
