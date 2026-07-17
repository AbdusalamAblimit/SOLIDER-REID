# exp207 审查: Swin-Base + GCN+PAA+CE+OA-SD

## 审查范围

a. design.md 合理性
b. Backbone 可用性 (swin_base_patch4_window7_224 + pretrained weights)
c. 模块兼容性 (PSG / PAA / GCN / OA-SD vs 1024-dim features)
d. 配置文件正确性
e. 显存/训练可行性
f. 与 exp206 对照

---

## 1. Backbone 可用性: PASS

- `swin_base_patch4_window7_224` 函数存在于 `model/backbones/swin_transformer.py:1426`
  - `embed_dims=128, depths=(2,2,18,2), num_heads=(4,8,16,32)`
- 已注册在 `model/make_model.py:448` 的 factory dict
- `pretrained/swin_base.pth` 存在 (1.77GB)
- `configs/occluded_duke/swin_base.yml` 已有正确的 `TRANSFORMER_TYPE: 'swin_base_patch4_window7_224'` 和 `PRETRAIN_PATH: 'pretrained/swin_base.pth'`

## 2. Feature Dimension 传播: PASS

Swin-Base Stage 3 feature dim = 128 * 2^3 = **1024** (vs Small/Tiny 768).

所有下游模块通过 `self.in_planes = self.base.num_features[-1]` 动态获取维度:

- **PSG**: `feat_ch = self.base.num_features[stage_idx]` (pose_backbone_model.py:55) -- 自动 1024
- **PAA**: 同上 (pose_backbone_model.py:79) -- 自动 1024
- **SkeletonGCN**: `feat_dim=self.in_planes` (pose_backbone_model.py:124) -- 自动 1024
- **SkeletonGCNHead**: `feat_dim=self.in_planes` (pose_backbone_model.py:124) -- 自动 1024
- **BN / Classifier**: `self.in_planes` (make_model.py:237-238) -- 自动 1024
- **OA-SD**: 使用 `F.normalize` + cosine distance -- 完全 dim-agnostic

结论: **所有模块正确处理可变维度，无需硬编码修改。**

## 3. OA-SD 兼容性: PASS

- EMA teacher 是 `copy.deepcopy(model)` (processor.py:401) -- 结构自动匹配
- EMA update 是逐参数 `t_param.data.mul_(decay).add_(s_param.data, alpha=1-decay)` -- 无维度依赖
- Distillation loss = cosine distance: `(1 - (sf_norm * tf_norm).sum(dim=1)).mean()` -- dim-agnostic
- exp206 已验证 OA-SD 在 GCN 模式 + Small backbone 下正常工作 (70.5/82.3)

## 4. 配置文件: 需要创建

**尚无 exp207 专用 config 文件。** 需基于 `pose_psg_gcn_paa_plboa_roa.yml` 创建，修改:

| 项目 | 原值 (Tiny) | exp207 (Base) |
|------|------------|---------------|
| TRANSFORMER_TYPE | swin_tiny_patch4_window7_224 | swin_base_patch4_window7_224 |
| PRETRAIN_PATH | pretrained/swin_tiny.pth | pretrained/swin_base.pth |
| WITH_CP | False | **True** (显存) |
| BASE_LR | 0.0008 | **0.0002** (design.md 指定) |
| POSE_OA_SD | (absent) | **True** |
| POSE_OA_SD_EMA_DECAY | (default 0.999) | 0.999 |
| CHECKPOINT_PERIOD | 20 | 20 |
| OUTPUT_DIR | ... | ./log/occluded_duke/exp207_base_gcn_paa_oasd |

**PLBOA 必须启用** (POSE_LOWER_BODY_OCC: True)。无 PLBOA 时 OA-SD 的 teacher/student 看到近乎相同的图像，exp206 monitor 中有此警告。

## 5. 显存估计: HIGH RISK

- Swin-Base 参数: ~88M (vs Small ~50M, Tiny ~28M)
- 1024-dim 特征 → 所有 linear 层/BN/classifier 按 1024/768 = 1.33x 增长
- GCN head 内部: feat_dim 768→1024, hidden_dim 不变 (256)
- PSG: Conv2d(17→64→**1024**) (vs 768)
- PAA: Conv2d(17→32→**1024**) (vs 768)
- OA-SD 需要额外 EMA teacher (完整模型副本) -- **这是关键约束**
  - Teacher ~88M params + 激活内存
  - Teacher forward 无需梯度 (no_grad), 但仍需激活内存

**估计 (1-view, WITH_CP=True):**
- Swin-Base 单独 + CP: ~12-14 GB (经验值)
- + GCN/PAA/PSG 模块: +1-2 GB
- + OA-SD teacher forward: +4-6 GB (no_grad 但需激活)
- **总计: 17-22 GB on 24GB 3090 → 可能勉强 OK**

**如果 OOM:**
- 降低 NUM_WORKERS (节省 CPU 内存)
- 降低 TEST.IMS_PER_BATCH 到 128
- 最坏情况: 只能在远程 16GB 5060 Ti 上跑不了 (除非减 batch)

## 6. LR 选择: REASONABLE BUT UNCERTAIN

- design.md 指定 BASE_LR=0.0002 (4x lower than Tiny 0.0008)
- 原 swin_base.yml 用 LR=0.0004
- exp206 (Small + OA-SD) 使用多少 LR? 需确认
- 经验上大模型 + distillation 确实需要更低 LR
- **建议: 0.0002 可以，但如果 10ep 收敛过慢可考虑提升到 0.0003**

## 7. Design.md 合理性: PASS

- 动机清晰: backbone scaling 是标准消融
- 假设合理: KPR Base vs Small 有 ~3% gap, 我们的方法类似
- 对照明确: exp206 (Small, same recipe)
- 不属于"小调参逃避创新" -- backbone scaling 是论文主表的必要行

## 8. 潜在问题

### Medium: Swin-Base pretrained weights 来源
- `pretrained/swin_base.pth` (1.77GB) 已存在
- 需确认是否为 SOLIDER pretrained (与 Tiny/Small 一致) 还是 ImageNet-22K
- 如果是 ImageNet-22K 而非 SOLIDER, 基线可能不同, 对比不完全公平

### Medium: 远程 5060 Ti 16GB 内存不足
- Base + OA-SD + CP 估计 17-22GB, **5060 Ti 16GB 必定 OOM**
- 此实验只能在本地 3090 24GB 上跑
- 远程需要跑别的实验

### Low: CHECKPOINT_PERIOD
- 确保设为 20 (不是 120), 以便中间测试

## 总结

| 检查项 | 状态 |
|--------|------|
| Backbone 存在 | PASS |
| Pretrained weights | PASS |
| PSG 维度兼容 | PASS |
| PAA 维度兼容 | PASS |
| GCN 维度兼容 | PASS |
| OA-SD 维度兼容 | PASS |
| Config 文件 | **需创建** |
| 显存 24GB | HIGH RISK (17-22GB est.) |
| 显存 16GB | FAIL (远程不可用) |
| LR 选择 | REASONABLE |

## 审查结论

**审查通过 (有条件)。**

条件:
1. 必须创建 exp207 专用 config 文件 (基于 pose_psg_gcn_paa_plboa_roa.yml + Base backbone + OA-SD + LR 0.0002 + WITH_CP=True)
2. 只能在本地 3090 上跑 (远程 16GB 不够)
3. 启动后第一时间检查 GPU 内存占用, 如 >22GB 立即准备 fallback plan
4. 确认 swin_base.pth 的预训练来源 (SOLIDER vs ImageNet-22K)
