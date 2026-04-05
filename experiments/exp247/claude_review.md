# exp247 Claude Review: VCSR (Visibility-Conditional Semantic Routing)

## 审查范围 (Agent 全范围审查)

1. design.md — 创新设计
2. model/modules/vcsr_head.py — 新文件 (~240 行)
3. model/pose_backbone_model.py — VCSR 集成
4. processor/processor.py — VCSR assign_loss + diagnostics
5. config/defaults.py — VCSR 配置
6. configs/occluded_duke/pose_psg_vcsr.yml — 实验配置

## 审查结论

### 代码正确性
- VCSRHead visibility 计算: max over keypoint group, mean over spatial ✅
- active_mask threshold logic: `> vis_threshold` binary mask ✅
- Cross-attention 与 LGPA 一致: 正确复制 ✅
- vis_threshold=0.3: 合理, 与 COCO keypoint confidence 惯例一致 ✅
- Detached features: `featmaps[-1].detach()` ✅
- Loss: list-loss (同 LGPA-D), VCSR assign_loss 正确 ✅
- Test: equal_concat (同 LGPA-D 初始对比) ✅
- OA-SD: teacher deepcopy + list distillation ✅
- AMP: 标准操作, heatmaps `.float()` ✅
- 单变量: 仅添加 visibility computation + weighted pooling ✅
- Optimizer: VCSR params via model.named_parameters() ✅

### Low 级别发现 (不影响训练)
- L1: vcsr_active_mask 和 active_mask 冗余存储 (harmless)
- L2: _compute_visibility 和 _compute_pose_bias 重复 F.interpolate (negligible)

### 创新性评估
这不是"加模块"的小改动. VCSR 重新定义了问题:
"遮挡 ReID 失败是因为固定 part 词汇表假设完整语义支持"
Novelty 7/10, Story 8/10 (GPT-5.4 评估)

## 结论

审查通过。代码干净, 机制正确, 创新性充分, 风险低。
