# exp229 BT-PKD 审查报告

## a. design.md 审查

**合理性**: 设计合理。核心假设清晰且有理论依据：L2/cosine distillation 的梯度比 SupCon 的 hard-mining 梯度更平滑，因此即使流到 backbone 也不会灾难。与前序实验 (exp210, exp215, exp220) 的对比表格清晰展示了实验变量的差异。

**单变量原则**: 满足。在 OA-SD baseline (exp191) 基础上只增加了 BT-PKD 一个变量。

**假设清晰性**: 核心假设一句话总结清楚。预期结果有具体范围 (+0.5~1.5%)，失败原因分析合理。

**创新性评估**: BT-PKD 不是小调参。它解决的是一个根本性问题（detach 屏障阻止 part 梯度到达 backbone），用一种新的信号类型（cosine distillation vs SupCon）来绕过 BA-PKC 的失败模式。代码修改量不大（~50行），但核心机制有创新。不属于"只改几行配置"的小调参范畴。

**文档缺失**: design.md 没有提到 BT-PKD 隐式依赖 POSE_SKELETON_GCN=True（因为 bt_kp_feats 的采样代码在 `elif self.use_skeleton_gcn` 分支内，且 teacher 的 kp_feats 也来自 GCN head）。建议补充。

**评级**: Medium（文档不完整但不影响正确性）

## b. 代码逐行审查

### config/defaults.py（7行新增）

```python
_C.MODEL.POSE_BT_PKD = False              # Enable BT-PKD (requires OA-SD)
_C.MODEL.POSE_BT_PKD_WEIGHT = 0.01       # Loss weight
```

- 默认 False，不影响已有实验
- weight=0.01 保守合理（BA-PKC 用 0.1 灾难）
- 注释清晰说明了与 BA-PKC 的区别
- 位置在 BA-PKC 相关配置之后，逻辑相邻

### model/pose_backbone_model.py（12行修改）

**__init__（新增5行）**:
```python
self.bt_pkd = getattr(cfg.MODEL, 'POSE_BT_PKD', False)
if self.bt_pkd:
    print('[BT-PKD] Backbone-through per-keypoint distillation enabled')
```
- 正确使用 getattr 读取 config
- 位于 BA-PKC 代码块之后，位置合理

**forward（修改7行）**:
```python
if getattr(self, 'ba_pkc', False) or getattr(self, 'bt_pkd', False):
    raw_fm = featmaps[-1]  # NOT detached!
    ...
    if getattr(self, 'ba_pkc', False):
        kp_data['ba_kp_feats'] = ba_kp_feats
    if getattr(self, 'bt_pkd', False):
        kp_data['bt_kp_feats'] = ba_kp_feats  # non-detached
```

- `raw_fm = featmaps[-1]` 正确获取 NON-detached feature map
- grid_sample 的 gradient flow 到 backbone 是 PyTorch 原生支持的
- `ba_kp_feats` 形状 (B, 17, C) 正确
- BA-PKC 和 BT-PKD 可以同时启用（两个 if 不互斥），这不会导致问题但可能不是有意为之
- 此代码在 `elif self.use_skeleton_gcn and pose_dict is not None:` 分支内（line 452），所以 BT-PKD 隐式要求 POSE_SKELETON_GCN=True

### processor/processor.py（29行修改）

**teacher_kp_data 提取（2行修改）**:
```python
teacher_kp_data = None  # NEW: 初始化
if len(teacher_out) == 5:
    _, teacher_feat, _, _, teacher_kp_data = teacher_out  # CHANGED: 不再忽略第5项
```

- 之前 teacher_out 的第5个返回值被 `_` 丢弃了，现在正确捕获为 `teacher_kp_data`
- 初始化为 None，安全
- teacher_out 长度检查（5/4/3）逻辑正确

**BT-PKD loss 计算（26行新增）**:
```python
bt_pkd_enabled = getattr(cfg.MODEL, 'POSE_BT_PKD', False)
if bt_pkd_enabled and kp_data is not None and teacher_kp_data is not None:
    bt_kp_feats = kp_data.get('bt_kp_feats')        # (B, 17, C) non-detached
    t_kp_feats = teacher_kp_data.get('kp_feats')    # (B, 17, C) from GCN
    t_kp_weights = teacher_kp_data.get('kp_weights') # (B, 17)
    if bt_kp_feats is not None and t_kp_feats is not None:
        s_norm = F.normalize(bt_kp_feats, p=2, dim=2)     # (B, 17, C)
        t_norm = F.normalize(t_kp_feats.detach(), p=2, dim=2)  # (B, 17, C)
        per_kp_dist = 1.0 - (s_norm * t_norm).sum(dim=2)  # (B, 17)
        if t_kp_weights is not None:
            w = t_kp_weights.detach().clamp(min=0.0)
            bt_pkd_loss = (per_kp_dist * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-6)
        else:
            bt_pkd_loss = per_kp_dist.mean(dim=1)
        bt_pkd_loss = bt_pkd_loss.mean()
        loss = loss + bt_pkd_weight * bt_pkd_loss
        details['bt_pkd'] = bt_pkd_loss.item()
        loss._loss_details = details
```

**梯度流验证**:
- `bt_kp_feats`: NON-detached (来自 `featmaps[-1]` → `grid_sample` → `ba_kp_feats`)
- `s_norm = F.normalize(bt_kp_feats, ...)`: 保持梯度
- `t_kp_feats.detach()`: 正确 detach teacher 目标
- `t_kp_weights.detach()`: 正确 detach weights（不从 confidence weighting 反传）
- `per_kp_dist`: (B, 17)，通过 s_norm 保持对 backbone 的梯度
- `loss = loss + bt_pkd_weight * bt_pkd_loss`: 正确加入总 loss

**形状验证**:
- `bt_kp_feats`: (B, 17, C) — 来自 grid_sample 后 permute
- `t_kp_feats`: (B, 17, C) — 来自 skeleton_gcn 的 `kp_feats_enhanced`
- `F.normalize dim=2`: 在 channel 维度归一化，正确
- `(s_norm * t_norm).sum(dim=2)`: 点积在 channel 维度，结果 (B, 17)
- `w.sum(dim=1)`: 在 17 keypoints 维度求和，结果 (B,)
- `bt_pkd_loss.mean()`: 在 batch 维度求均值，标量

**数值稳定性**:
- `F.normalize` 内置 eps=1e-12，AMP 下安全（已有 OA-SD 使用相同模式）
- `w.sum(dim=1).clamp(min=1e-6)`: 防止除零
- cosine distance 范围 [0, 2]，不会溢出

**代码位置**: 在 OA-SD loss 计算之后（line 775 之后），仍在 `if oa_sd_enabled and ...` 块内。这意味着 BT-PKD 只在 OA-SD 激活时才运行，符合设计要求。

## c. 配置引用检查

- `POSE_BT_PKD`: defaults.py (line 219) → model (line 127 via getattr) → processor (line 780 via getattr) -- 一致
- `POSE_BT_PKD_WEIGHT`: defaults.py (line 220) → processor (line 786 via getattr) -- 一致
- 两处 getattr 都带了正确的默认值 (False / 0.01)

## d. defaults.py 安全性

- POSE_BT_PKD 默认 False：当 False 时，model 不会执行 grid_sample，processor 不会计算 loss
- POSE_BT_PKD_WEIGHT 默认 0.01：仅在 BT_PKD=True 时使用
- 两个新默认值不影响任何已有实验的可复现性

## e. Processor loss 计算详细审查

- BT-PKD loss 正确使用 cosine distance (1 - cos_sim)
- Teacher features 正确 detach
- Student features 保持梯度链到 backbone
- Confidence weighting 正确：高置信 keypoint 的 distillation 信号更强
- 当 t_kp_weights 为 None 时有合理 fallback (simple mean)
- loss 正确累加到 total loss
- logging 正确（`details['bt_pkd']`）
- 位于 AMP autocast 上下文内，与已有 OA-SD 代码一致

## f. 与前序实验对照

- **vs exp191 (OA-SD only)**: 仅新增 BT-PKD loss，单变量隔离
- **vs exp215 (BA-PKC)**: 相同的 non-detached 采样方式，但 loss 函数不同（cosine distillation vs SupCon）。这是核心消融变量。
- **vs exp220 (GSPB 5%)**: GSPB 缩放全部梯度，BT-PKD 仅在 17 个位置加 distillation 梯度。不同机制。

## 问题/风险汇总

| # | 级别 | 描述 | 状态 |
|---|------|------|------|
| 1 | Low | design.md 没有明确说明依赖 POSE_SKELETON_GCN=True | 无需代码修改，建议文档补充 |
| 2 | Low | teacher 的 EMA model 也会执行 bt_kp_feats 采样（deepcopy 继承了 bt_pkd=True），产生不必要的 grid_sample 计算 | 在 torch.no_grad() 下执行，无梯度开销，仅多一次 grid_sample（微量计算），不影响正确性 |
| 3 | Low | BA-PKC 和 BT-PKD 可以同时启用（两个 if 不互斥），不会 crash 但可能不是有意设计 | 功能正确，只是多产生两份 kp_feats，不影响本实验 |

## 结论

代码修改量精确、最小、安全。核心机制（non-detached backbone features → cosine distillation toward EMA teacher）实现正确。梯度流、形状、数值稳定性均已验证。默认值不影响已有实验。所有问题均为 Low 级别，不影响训练正确性。

**审查通过**
