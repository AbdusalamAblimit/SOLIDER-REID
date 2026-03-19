# exp056 PGAM 多 Stage 注入审查报告

## 审查范围
- `experiments/exp056/design.md`
- `configs/occluded_duke/pose_psg_gcn_pgam_s23.yml`
- `config/defaults.py`
- `model/pose_backbone_model.py`
- `model/modules/pose_attn_mask.py`
- `log/occluded_duke/exp056_pgam_s23/train_log.txt`

## 发现的问题

| # | 严重程度 | 文件 | 描述 | 状态 |
|---|---------|------|------|------|
| 1 | HIGH | pose_attn_mask.py / config | 虽然 `POSE_ATTN_MASK_STAGES: [2, 3]` 配置正确，但真实 heatmap 进入 PGAM 后在 `24x8` 和 `12x4` 上 body mask 覆盖率都为 **100%**。Stage 2+3 扩层不会改变实际注意力计算 | 未修复 |
| 2 | LOW | design.md | 文档里保留了“可能需要改 `POSE_PSG_STAGES`”的中间表述；实际实现已经正确拆成独立的 `POSE_ATTN_MASK_STAGES` | 建议后续修文档 |

## 审查通过项

- `POSE_ATTN_MASK_STAGES: [2, 3]` 已被单独解析，不会误改 PSG 注入层
- `pose_backbone_model.py` 中 PGAM 和 PSG stage 配置完全分离，符合实验目标
- Stage 2 / Stage 3 的每个 block 都会各自实例化 PGAM 模块
- 不会影响默认 baseline，因为 PGAM 总开关默认为关闭
- 训练日志完整，无运行时错误

## 结论

❌ **不通过**

代码接线虽然正确，但 PGAM 在两个 stage 上都没有形成有效 mask，所以 `exp056` 没有真正测试“多 Stage PGAM”这个想法。
