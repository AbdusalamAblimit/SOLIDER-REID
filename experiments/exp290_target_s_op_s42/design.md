# exp290_target_s_op_s42 — Target-only heatmap → Occ-PTrack SOTA

## 动机

Phase 1 OP 主数字 (Swin-Small 78.4/86.2, Swin-Base 78.5/86.2) 超过 KPR-without-prompt (75.4/85.3) 但远落后 **KPR-with-prompt 82.3/92.3** (KPR ECCV 2024 Table 1)。差距 3.8 mAP / 6.1 R1 完全来自 **prompt mechanism**, 即多人场景下**显式指明 target person**。

### 当前代码问题

`model/modules/pose_utils.py:21-39` `merge_person_heatmaps()`:
```python
scene_heatmap = (heatmaps * person_mask).max(dim=1)[0]  # max over persons
```
**OP dataset 图像多人时, scene_heatmap 混合 target + distractor 关键点** → PSG/LGPA gate 无法区分目标:

| Dataset | 多人场景比例 | 现行 scene_heatmap 合理性 |
|---------|-------------|------------------------|
| Occ-Duke (OD) | 低 (大部分 single-person) | scene ≈ target, 无歧义 |
| Market | 几乎全 single | scene ≈ target |
| **Occ-PTrack (OP)** | **高 (定义就是多人场景)** | **scene 包含 distractor**, 歧义严重 |

### 已有但闲置的目标信号

`datasets/pose_dataset.py:356-362`: target person 始终 reorder 到 index 0 (via `target_person_idx`)。`pose_backbone_model.py:912`:
```python
target_heatmaps = heatmaps[:, 0] * person_mask[:, 0].view(-1, 1, 1, 1)
```
**target_heatmaps 已计算但未用** — 仅 `scene_heatmaps` 被传下游 PSG/LGPA/VCSR/PPA/STR/FSDC 等。

## 核心假设

**通过训练+测试都用 target-only heatmap (person 0), 在 OP 上逼近甚至超过 KPR-with-prompt 82.3/92.3**。

Target annotation 在 OP 训练 & 测试数据都有 (KPR 依赖 test-time manual keypoint prompt, 我们用 annotation-provided target 更优雅)。

OD / Market 无多人歧义, 预期 target-only ≈ scene-only, 0 回归风险 (target=person 0 = scene 唯一 person)。

## 技术方案

### 修改 1: `config/defaults.py` (新增 1 flag)

```python
# Target-only heatmap (Occ-PTrack-style target disambiguation)
_C.MODEL.POSE_USE_TARGET_HEATMAP = False  # default off preserves existing behavior
```

### 修改 2: `model/pose_backbone_model.py` __init__ (~3 行)

```python
self.use_target_heatmap = getattr(cfg.MODEL, 'POSE_USE_TARGET_HEATMAP', False)
if self.use_target_heatmap:
    print('[POSE] POSE_USE_TARGET_HEATMAP=True: ...')
```

### 修改 3: `model/pose_backbone_model.py` forward() (~2 行, 位置: `_prepare_pose` 之后)

```python
scene_heatmaps, _, target_heatmaps, _ = self._prepare_pose(pose_dict)

# Target-only heatmap swap (NEW, only when flag is True)
if self.use_target_heatmap and target_heatmaps is not None:
    scene_heatmaps = target_heatmaps
```

### Backward compatibility 铁保证

- Flag 默认 `False` — 和现有配置完全等价
- 仅 `pose_backbone_model.py forward()` 的单一 if-分支切换 `scene_heatmaps` 指向, 不影响任何其他代码路径
- `_prepare_pose()` 返回签名不变 (已有返回 scene + target + diff)
- `merge_person_heatmaps()` 未修改
- 所有 PSG/LGPA/VCSR/PPA/STR/FSDC 等下游调用点不变, 仍收 `scene_heatmaps` 变量名, 只是内容在 flag 开时指向 target

数据流 when flag OFF (默认):
```
heatmaps (B, N, 17, H, W) → max over N → scene_heatmaps → PSG/LGPA/... [与过去完全相同]
```
数据流 when flag ON (本实验):
```
heatmaps (B, N, 17, H, W) → [:, 0] → target_heatmaps → (swap) → scene_heatmaps var → PSG/LGPA/... [下游无感知]
```

### 实验配置

- **Backbone**: Swin-Small (Phase 1 exp265 主力)
- **Dataset**: Occluded-PoseTrack-ReID
- **Config**: `configs/occluded_posetrack/prcv_best_small.yml` + `MODEL.POSE_USE_TARGET_HEATMAP True`
- **Scaffold**: Full Scaffold (LGPA + GCN512 + OA-SD + ParAug + LOWER_BODY_OCC) + 2-stage PSG `[-2,-1]` (同 exp265)
- **Seed**: 42 (对齐 exp265)
- **Epochs**: 120 (同 exp265)
- **机器**: lab4090 (mmpose-abu env, idle)
- **Speed**: ~45-50s/epoch × 120 ≈ 1.5-2h (Small, 4090, OP dataset 较小)
- **FINAL ETA**: 训练启动后 ~2h

## 预期结果

| 指标 | exp265 s42 (scene) | **exp290 (target)** | KPR-w/prompt |
|------|---------------------|---------------------|--------------|
| mAP | 78.4 | **80-82 预期** | 82.3 |
| R1 | 86.2 | **89-92 预期** | 92.3 |

**如果 ≥ 82/90**: 打平 KPR-w/prompt (他们需要 test-time manual prompt, 我们用 annotation-embedded target, 更优雅), **OP SOTA 达成**。

**如果仅 +1~2 mAP**: target-only 部分有效, 可能需要再加 target vs distractor contrastive (用 `diff_heatmaps` 分支)。

**如果持平 exp265 (无改善)**: 说明 PSG/LGPA gate 对 distractor 的受扰动不大, 本方向不是 OP 瓶颈。转而考虑 prompt-token 实现。

## 对照组

- **主对照**: exp265 s42 Small OP 78.4/86.2 (srvC, scene_heatmap 默认路径)
- **辅对照**: exp265b s41 78.5/85.9 (跨 seed/设备), exp266b_3090 s41 Base 78.5/86.2
- **Phase 1 OP 文献**: KPR-w/o-prompt 75.4/85.3, KPR-w/prompt 82.3/92.3, SOLIDER 76.1/84.4

## 风险评估

1. **Train-test mismatch**: 无。训练+测试都用 target_heatmaps (flag 控制全流程)。
2. **Target annotation 错误**: 数据集层面问题, 非本实验引入。若 `target_person_idx` 有噪声, 影响所有已有 OP 实验 (exp264/265/265b/266b_3090), 本实验不例外。
3. **OD/Market 回归**: 本实验只跑 OP。OD/Market 保持原 scene_heatmap 路径 (flag 默认 off)。OD 如需对照, 后续另起 exp291_target_s_od_s42。
4. **Full Scaffold 兼容性**: 所有下游 module (PSG/LGPA/VCSR/PPA/STR/FSDC) 收的仍是 `scene_heatmaps` 变量, 只是内容切到 target。它们的 forward 签名、shape 要求 (B, 17, H, W) 都满足。
5. **Pose dropout**: `self.training and scene_heatmaps is not None and pose_dropout_p > 0` 时 dropout。target_heatmaps 也是 (B, 17, H, W), dropout 正常。

## 代码审查重点

- [ ] flag 默认 False, 对现有所有 exp (尤其 exp266b 训练中) 无影响
- [ ] target_heatmaps 未正确计算时 (pose_dict=None 或 person_mask 全 0) 不 crash
- [ ] shape 和 dtype 和 scene_heatmaps 完全一致 (`_prepare_pose` 里已保证, 验证即可)
- [ ] 下游所有 scene_heatmaps 消费点无 assumption 关于 person-merge 语义 (应该只看 shape + 关键点响应)
- [ ] 训练/测试 forward 路径都 swap (都在同一 forward 中, 无分支差异)

## 论文定位

如果 OP SOTA 达成 (≥82/90):
- **新增 contribution**: "training-time target selection via annotation replaces test-time keypoint prompt"
- **narrative**: KPR 要求测试时额外给 keypoint prompt, 我们用 PoseTrack 已有 target_person_idx 在训练 & 测试都定位, 零额外标注 / 零 test-time 开销
- main_results Table 1 Small OP 主数字替换为 exp290, 写入 supplementary 讨论 "target disambiguation in PSG gate"
- ablation Table H 新增: scene vs target heatmap 对比 (OD baseline 持平, OP 大增益)
