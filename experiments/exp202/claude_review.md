# exp202 审查报告: Swin-Small + SupCon + Full Architecture

## 审查范围
- design.md
- 新配置文件: `configs/occluded_duke/pose_psg_stdpr_pertoken_plboa_pape_ms_supcon_small.yml`
- 模型代码: `model/make_model.py`, `model/pose_backbone_model.py`
- Backbone: `model/backbones/swin_transformer.py`
- PSG 模块: `model/modules/pose_spatial_gate.py`
- Loss: `loss/make_loss.py`, `loss/supcon_loss.py`
- 默认配置: `config/defaults.py`

## 1. Backbone 配置 -- 通过

- `TRANSFORMER_TYPE: swin_small_patch4_window7_224` 正确注册于 `__factory_T_type` (make_model.py:449)
- Swin-Small 定义: `depths=(2, 2, 18, 2), embed_dims=96` (swin_transformer.py:1431)
- 与 Tiny 共享 embed_dims=96, 因此 `num_features = [96, 192, 384, 768]`, `in_planes=768`
- 所有依赖 in_planes 的组件 (classifier, bottleneck, STD-PR, str_classifier) 自动获取正确维度

## 2. 预训练权重 -- 通过

- `pretrained/swin_small.pth` 存在 (1.15GB)
- `PRETRAIN_CHOICE: 'self'` -> `convert_weights=False`, 直接加载 SOLIDER checkpoint
- init_weights 支持 position bias 插值, 适配 384x128 输入的 PRETRAIN_HW_RATIO=2

## 3. 学习率 -- 通过

- BASE_LR=0.0004 (Tiny 用 0.0008), 符合 Swin-Small 在本项目的历史设定
- 与 diff 确认: 这是与 Tiny 配置唯一的训练超参变化 (除 SupCon 温度外)

## 4. PSG 多阶段注入与 18 blocks -- 通过

- `POSE_PSG_STAGES: [2, 3]` -> Stage 2 (2 blocks) + Stage 3 (18 blocks)
- `__init__` 中 `for block_idx in range(len(stage.blocks))` 动态创建 PSG (pose_backbone_model.py:56)
- 字典 key 格式 `s{stage}_b{block}` 无硬编码 block 数
- `_run_stage_with_psg` 遍历 `stage.blocks` (ModuleList), 通用于任意 block 数
- 预估 PSG 参数: Stage 2 (2 x 26K) + Stage 3 (18 x 51K) = ~972K (Tiny 为 ~359K), 增幅合理

## 5. 所有 Pose 组件启用 -- 通过

- PSG: `POSE_BACKBONE_PSG: True`, `POSE_PSG_STAGES: [2, 3]`
- PAPE: `POSE_PATCH_EMBED: True` -> Conv2d(17, 96, 1x1), embed_dims=96 对 Small 正确
- STD-PR: `POSE_STRUCTURAL_ROUTING: True`, 6 parts, 8 heads, 2 layers, per-token
- SupCon: `POSE_STR_SUPCON: True`, `POSE_STR_SUPCON_TEMP: 0.05`
- PLBOA: `POSE_LOWER_BODY_OCC: True`, `POSE_LOWER_BODY_OCC_PROB: 0.7`
- Multi-stage: `POSE_PSG_STAGES: [2, 3]`

## 6. SupCon 温度差异 -- 注意 (非 bug)

- Small 配置用 T=0.05, Tiny 模板 yml 用 T=0.07
- 但 exp176/exp187 的实际最佳配置是 T=0.05, design.md 也引用了 exp176 (T=0.05)
- 因此 T=0.05 是正确选择, 与参考实验一致

## 7. 显存估计 -- 通过 (需关注)

- WITH_CP=False, Swin-Small Stage 3 有 18 blocks, 比 Tiny 的 6 blocks 多 3 倍激活
- 设计估计 ~16GB (base 14GB + SupCon 2GB), 3090 24GB 有余量
- 风险: 18 blocks 的梯度激活可能超出预期, 但无 3-view augmentation, 应可控
- 如果 OOM, 可启用 WITH_CP=True 作为 fallback (无需改模型代码)

## 8. CHECKPOINT_PERIOD -- 通过

- `CHECKPOINT_PERIOD: 20`, 符合要求

## 9. 配置 Diff 确认 (vs Tiny SupCon config)

仅 5 处差异, 全部合理:
1. PRETRAIN_PATH: swin_tiny.pth -> swin_small.pth
2. TRANSFORMER_TYPE: swin_tiny -> swin_small
3. POSE_STR_SUPCON_TEMP: 0.07 -> 0.05 (匹配 exp176 最佳)
4. BASE_LR: 0.0008 -> 0.0004
5. OUTPUT_DIR: exp174 -> exp202

无意外改动, 单变量 (backbone 规模) 对照干净。

## 10. 不破坏现有实验 -- 通过

- 新配置文件, 不修改任何现有代码
- 默认值 defaults.py 无变动
- 输出到独立目录 `./log/occluded_duke/exp202_small_supcon`

## 结论

审查通过

所有检查项均无问题。配置干净, 仅 backbone 规模和对应 LR 变化, 与 Tiny SupCon 实验形成严格对照。PSG 代码对 18-block Stage 3 天然兼容, 无硬编码风险。唯一需关注的是显存: 如果训练 OOM, 设置 WITH_CP=True 即可解决。
