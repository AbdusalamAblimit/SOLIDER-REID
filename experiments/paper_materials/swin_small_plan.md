# Swin-Small 验证计划（4090）

## 需要验证的配置

### 1. GCN+PLBOA (当前 Tiny 最强 test-time)
- Config: `pose_psg_gcn_plboa.yml` 改 TRANSFORMER_TYPE 为 swin_small
- 预期 mAP: 67-70% eq, 70-73% maxsim

### 2. STD-PR+PLBOA (当前 Tiny 最强纯模型)
- Config: `pose_psg_stdpr_plboa.yml` 改 TRANSFORMER_TYPE 为 swin_small
- 预期 mAP: 68-71% eq

### 3. Full recipe: GCN+PLBOA+PAA
- Config: `pose_psg_gcn_paa_plboa.yml` 改 TRANSFORMER_TYPE
- 预期 mAP: 67-70%

## 注意事项
- Swin-Small pretrained weights 路径需要更新
- 384×128 input size 与 Tiny 相同
- Batch size 可能需要减半（Small 显存更大但参数更多）
- 3-seed 验证很重要

## Occluded-Duke SOTA 参考 (标准方法)
- OGFR (2025): 76.6 R1 / 64.7 mAP (ViT)
- PAB-ReID (2024): 72.6 R1 / 63.5 mAP (ViT)
- 我们 Tiny: 73.4 R1 / 63.4 mAP (STD-PR+PLBOA)
- 我们 Tiny+SGCFR: 75.3 R1 / 65.2 mAP
