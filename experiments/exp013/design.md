# exp013: PSG + PAB 双重姿态注入

## 动机
- PSG (feature gating) 和 PAB (attention bias) 分别在不同层面注入 pose 信息
  - PSG: SwinBlock 之后，调制 feature values（x * (1 + gate)）
  - PAB: WindowMSA 内部，调制 attention scores（attn += pose_bias）
- 单独使用时 PSG (+1.7%) > PAB (+0.8%)，但两者理论上互补
- 如果组合有效，构成"多层次姿态注入"的创新点

## 创新点 / 核心假设
PAB 改善 attention pattern（让模型关注正确的身体部位），PSG 增强这些位置的特征表达，两者叠加应该 > 单独使用。

## 技术方案
- 同时启用 PAB 和 PSG
- 在 `_run_stage_with_psg` 中：先对 block 传入 PAB 的 pose_bias_map，然后在 block 输出后应用 PSG 的 feature gate
- 修改 `pose_backbone_model.py`：`__init__` 中同时创建 pab_modules_dict 和 psg_modules_dict
- 新 config 开关：`POSE_PSG_PAB_COMBO: True`（或直接同时启用 POSE_BACKBONE_PSG + POSE_ATTN_BIAS 时触发组合模式）
- 关键超参数：hidden_dim 32 (PAB) + 64 (PSG)
- 总参数：~108K (102K PSG + 5.4K PAB)

## 预期结果
- 最优: mAP 59-60%（两者互补，突破 PSG 上限）
- 中性: mAP 58-58.5%（与 PSG 持平，PAB 没有增量价值）
- 失败: mAP < 58%（两种注入互相干扰）

## 对照组
- Baseline: exp007 PSG-only (mAP 58.3%, R1 67.9%)
- 消融变量: 在 PSG 基础上是否加入 PAB
