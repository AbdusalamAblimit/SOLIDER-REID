# Codex Review — exp324d

**Verdict**: approve
**Date**: 2026-06-16
**Review round**: 第 1 轮（codex --search exec, xhigh, 141,797 tokens）

## Findings
- **Low**: 严格"单变量"措辞应注明 schedule 差异。exp324d 默认 `--epochs 35` / cosine `T_max=35`，exp324b 默认 60 / T_max=60。非阻断（exp324b 已 ceilinged），但别声称 epoch-matched ablation 除非记录/控制。
- **Low**: `assert bs == 64` 检查配置的 P*K，非 sampler 实际输出。全量 Occluded-Duke 必出 64；但极小 `--limit_train`（<16 identities）的 sanity run 可能 assert 过但 batch 更小。对全量run无影响。

## Verified（codex 独立确认）
- `bmm(pool_w, patch)` 与 `build_part_pose` cell 等价：同 part groups / visibility skip / (0,0) sentinel / rounded grid cell / 3×3 窗 / set-union / mean 权重。`r*GRID_W+c` 匹配 DINO patch row-major 顺序。
- Train path 无 detach / no_grad / numpy 转换；梯度流 `DINO LoRA → patch → bmm → PartHead → losses`。
- `micro_bs` 只切 DINO forward 再 concat；batch-hard triplet 见完整逻辑 batch；每逻辑 batch 恰一次 optimizer.step()。
- `use_reentrant=False` 是 frozen-base/LoRA/非梯度图像输入的正确选择（PyTorch 文档：reentrant 需 grad-requiring input/output，non-reentrant 不需）。
- 优化器含 LoRA + head params；BN/1-D head 排除 WD。LoRA q/v rank8 ×12 层 = 294,912 可训 adapter 参数（符合预期）。
- Eval 与 exp324b 对称：global cosine + 归一化投影 part-MaxSim、同可见部位 masking、同 heavy-occ mask、同 Market same-pid/same-cam 排除。
- Dtype 全 float32；无 autocast/混精不匹配。

## Novelty Check
LoRA / DINOv2 / pose-visible-part matching 各自有先例（PVPM/PFD），DINOv2-for-ReID 新兴，LoRA-DINOv2 在 ReID 外存在。**未找到 LoRA-finetuned DINOv2 + 可微姿态锚定部位池化 + mutually-visible part-MaxSim 用于遮挡单图行人 ReID 的直接先例** → 组合 plausibly new。

## 结论
codex 审查通过（approve），无 Critical/High/Medium 阻断项。两条 Low 均非 run-blocker，已知晓（epoch schedule 差异 + assert 仅检查配置）。
