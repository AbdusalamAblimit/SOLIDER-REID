# Claude Review -- exp254: 2-Stage PSG (Stage 2+3, no PAA) on Tiny LGPA-D+GCN

**Date**: 2026-04-08
**Review round**: 2 (expanded from prior abbreviated review)
**Reviewer**: Claude Opus 4.6

## 1. design.md 审查

design.md 存在且结构合理。实验目的清晰：填补 PSG 阶段消融表中缺失的一格。

**消融矩阵（完成后）**:

| 实验 | PSG Stages | PAA | mAP/R1 |
|------|-----------|-----|--------|
| exp246b | Stage 3 only ([-1]) | No | 65.5/77.2 |
| exp254 | Stage 2+3 ([-2,-1]) | No | ? |
| exp251 | Stage 2+3 ([-2,-1]) | Yes | 65.2/76.2 |
| exp253 | Stage 1+2+3 ([-3,-2,-1]) | No | 65.1/76.2 |

这是一个合理的消融实验。通过比较 exp254 vs exp251 可以隔离 PAA 的贡献；通过 exp254 vs exp246b 可以隔离 Stage 2 PSG 的贡献。

**注意**: 这是纯消融/config-only 实验，不涉及新代码或新创新。作为消融表格的填充是合理的，但不应视为主线研究方向。设计文档承认了这一点。

判定: **通过**

## 2. 代码审查: PSG stage 解析与 PAA 门控

### 2a. POSE_PSG_STAGES=[-2,-1] 解析 (pose_backbone_model.py L39-46)

```python
psg_stages = list(getattr(cfg.MODEL, 'POSE_PSG_STAGES', [-1]))
num_backbone_stages = len(self.base.stages)  # = 4 for Swin-Tiny
self.psg_stage_indices = set()
for s in psg_stages:
    idx = s if s >= 0 else num_backbone_stages + s
    self.psg_stage_indices.add(idx)
```

`[-2, -1]` -> `{4+(-2), 4+(-1)}` = `{2, 3}`. 正确解析为 Stage 2 和 Stage 3。

### 2b. PSG 模块创建 (L52-63)

对 stage 2 (384 dim, 6 blocks) 和 stage 3 (768 dim, 2 blocks) 分别创建 PSG:
- Stage 2: 6 个 PSG(17->64->384)，每个约 17*64+64 + 64*384+384 = 1152 + 24960 = 26112 params
- Stage 3: 2 个 PSG(17->64->768)，每个约 1152 + 64*768+768 = 1152 + 49920 = 51072 params
- 总计: 6*26112 + 2*51072 = 156672 + 102144 = ~259K params

vs exp246b (Stage 3 only): 2*51072 = ~102K params
vs exp251 (Stage 2+3 + PAA): ~259K PSG + PAA params

参数量合理，不会导致 VRAM 问题。

### 2c. PAA 门控验证 (L73-93) -- 关键检查项

```python
self.use_paa = getattr(cfg.MODEL, 'POSE_ADDITIVE_ADAPTER', False)
if self.use_paa:
    self.paa_modules_dict = nn.ModuleDict()
    ...
```

当 `POSE_ADDITIVE_ADAPTER=False`（默认值）:
- `self.use_paa = False`
- `self.paa_modules_dict` 不会被创建（整个 if 块被跳过）
- 不会有 PAA 参数打印 `[PAA] Pose Additive Adapter enabled:` -- 日志可作为验证

在 forward 路径 (`_run_stage_with_psg`, L395-397):
```python
if getattr(self, 'use_paa', False) and scene_heatmaps is not None and key in getattr(self, 'paa_modules_dict', {}):
    x = self.paa_modules_dict[key](x, hw_shape, scene_heatmaps)
```

`getattr(self, 'use_paa', False)` 返回 False，整行短路。即使 `paa_modules_dict` 属性不存在，`getattr` 的默认值 `{}` 也安全处理了这一情况。PAA forward 路径永远不会执行。

判定: **PAA 在 POSE_ADDITIVE_ADAPTER=False 时完全禁用。init 不创建模块，forward 不执行注入。逻辑正确。**

## 3. Config 默认值安全性

检查 `config/defaults.py`:
- `POSE_PSG_STAGES = [-1]` (默认只注入 Stage 3) -- 安全
- `POSE_ADDITIVE_ADAPTER = False` (默认禁用 PAA) -- 安全
- exp254 通过命令行覆盖 `POSE_PSG_STAGES=[-2,-1]` 且不设置 `POSE_ADDITIVE_ADAPTER`，使用默认 False
- 所有其他默认值不受影响

现有实验的 config 文件中无一被修改。exp254 是纯命令行覆盖实验。

判定: **通过。不影响已有实验的可复现性。**

## 4. VRAM / WITH_CP 检查 (5060 Ti 16GB)

exp254 运行在 5060 Ti (16GB)。基础配置 `pose_psg_lgpa.yml` 设置 `WITH_CP: False`。

对比参考:
- exp251 (2-stage PSG + PAA): 使用 5542 MiB，WITH_CP=False
- exp254 (2-stage PSG, 无 PAA): 少了 PAA 模块的参数和激活值内存

exp254 的峰值 VRAM 应 < exp251 的 5542 MiB。Swin-Tiny + 2-stage PSG 在 16GB 卡上完全安全，无需启用 WITH_CP。

判定: **通过。无 OOM 风险。**

## 5. 单变量原则检查

### vs exp251 (同 stage config，仅去掉 PAA)
- exp251: POSE_PSG_STAGES=[-2,-1], POSE_ADDITIVE_ADAPTER=True
- exp254: POSE_PSG_STAGES=[-2,-1], POSE_ADDITIVE_ADAPTER=False
- 差异: 仅 PAA 开关。**单变量满足。**

### vs exp246b (同 PAA=off，仅加 Stage 2 PSG)
- exp246b: POSE_PSG_STAGES=[-1] (Stage 3 only), PAA=off
- exp254: POSE_PSG_STAGES=[-2,-1] (Stage 2+3), PAA=off
- 差异: 仅 Stage 2 PSG 是否启用。**单变量满足。**

两个方向的对照均严格隔离单一变量。

判定: **通过。消融隔离性良好。**

## 6. Train/Test 对称性

### 训练路径:
1. `forward()` L420-421: `_run_backbone_with_psg(x, scene_heatmaps)` -- Stage 2+3 均有 PSG 注入
2. `_run_backbone_with_psg` 中 L350-357: 对每个 stage 检查 `i in self.psg_stage_indices`，stage 2 和 3 走 PSG 路径，stage 0 和 1 走普通路径
3. LGPA-D + GCN 双分支 (L521-540): 从 `featmaps[-1]` (PSG-gated Stage 3 输出) 提取
4. LGPA detach + GCN detach -- 不干扰 backbone 梯度

### 测试路径:
1. 同一 `_run_backbone_with_psg()` 函数 (L421)，训练/测试共享，无分支
2. `self.training = False` 进入 else 分支 (L698+)
3. LGPA test path (L716-728): 从同一 `featmaps[-1]` 提取，加 GCN dual
4. `pose_test_feat = 'equal_concat'` -- 全局 + LGPA parts + GCN pooled 拼接

**PSG 的核心路径 (`_run_backbone_with_psg`) 在训练和测试中完全共享，无条件分支差异。**
**LGPA+GCN 的训练/测试路径一致（feature map 来源、detach 设置均相同）。**

判定: **通过。Train/test 完全对称。**

## 7. 其他检查

### 7a. Backward compat: psg_modules 列表 (L65-71)
Stage 3 在 `psg_stage_indices` 中 (index 3)，所以 `self.psg_modules` 兼容列表仍会被创建。不影响任何依赖此列表的遗留代码路径。

### 7b. Stage 2 downsample 处理 (L399-404)
Stage 2 有 downsample (384->768)。在 `_run_stage_with_psg` 中，PSG 只在 blocks 循环内部注入，downsample 在所有 blocks 之后执行 (L400-404)。数据流: block -> PSG gate -> next block -> ... -> downsample。逻辑正确，downsample 不受 PSG 干扰。

### 7c. Semantic weight 应用 (L360-363)
语义权重（SOLIDER 预训练特有）在每个 stage 循环之后应用。PSG 注入发生在 block 内部（stage 循环内部）。两者层级不冲突。semantic weight 作用于 PSG-gated 后的特征。

### 7d. 优化器参数覆盖
PSG 模块通过 `nn.ModuleDict` 注册，自动被 PyTorch 优化器发现。无需手动添加参数组。Stage 2 PSG 的 6 个新模块会被正确加入优化器。

### 7e. AMP 安全
PSG 模块使用 Conv2d + sigmoid + 乘法，均为 AMP 安全操作。无自定义 autograd 函数。无 half precision 下的数值风险。

### 7f. 预期结果合理性
- exp246b (1-stage): 65.5
- exp251 (2-stage+PAA): 65.2
- exp253 (3-stage): 65.1
- 趋势: 增加 PSG stage 数量和/或 PAA 对 Tiny 模型似乎无帮助或略有害
- exp254 预期: 约 64.5-65.5 之间
- 如果 exp254 > exp251 (65.2)，则证明 PAA 在 multi-stage 场景下有害
- 如果 exp254 < exp246b (65.5)，则 Stage 2 PSG 在 Tiny 上无收益
- 任何结果都能提供有价值的消融证据

## 8. 日志充分性检查

模型初始化时会打印:
- PSG 模块创建信息 (通过 `__init__` 中的 ModuleDict 创建)
- **应该不会出现** `[PAA] Pose Additive Adapter enabled` 打印（因为 PAA=False）
- LGPA 和 GCN 的初始化信息照常打印

建议：训练开始后，在 log 中确认无 `[PAA]` 打印，作为 PAA 确实禁用的二次验证。

## 9. 综合评估

| 检查项 | 结果 |
|--------|------|
| design.md 完整性 | 通过 |
| 代码正确性 (PSG stage 解析) | 通过 |
| PAA 禁用逻辑 (init L73-93 + forward L395-397) | 通过 |
| Config 默认值安全 | 通过 |
| VRAM 安全 (5060 Ti 16GB, < 5542 MiB) | 通过 |
| 单变量原则 (vs exp251 + vs exp246b) | 通过 |
| Train/test 对称 | 通过 |
| 向后兼容 (psg_modules list) | 通过 |
| Stage 2 downsample 处理 | 通过 |
| Semantic weight 交互 | 通过 |
| 优化器参数注册 | 通过 |
| AMP 安全 | 通过 |

无 Critical / High / Medium / Low 级别问题。

## 结论

审查通过。exp254 是一个设计清晰的消融实验，填补 PSG 阶段表中 "2-stage 无 PAA" 的空格。代码逐行验证确认：(1) `[-2,-1]` 正确解析为 Stage 2+3；(2) PAA 在 `POSE_ADDITIVE_ADAPTER=False` 时从 init 到 forward 完全禁用，不存在遗漏调用；(3) Stage 2 downsample 在 PSG 注入之后正确执行；(4) 单变量隔离满足两个方向的对照需求。无 VRAM 风险，无 train/test 不对称，无 AMP 或优化器问题。

**PASS**
