# exp244 Claude Review: LGPA-Detach

## 审查范围
1. `experiments/exp244/design.md` — 设计文档
2. `config/defaults.py` — 新增 POSE_LGPA_DETACH (默认 False)
3. `model/pose_backbone_model.py` — detach flag init + forward
4. `configs/occluded_duke/pose_psg_lgpa_detach.yml` — 实验配置
5. `model/modules/clip_part_head.py` — 无修改 (与 exp243 v3 相同)

## 代码变更 (极小)

### config/defaults.py
- 新增 `_C.MODEL.POSE_LGPA_DETACH = False` — 默认不 detach, 不影响 exp243

### model/pose_backbone_model.py  
- init: `self._lgpa_detach = getattr(cfg.MODEL, 'POSE_LGPA_DETACH', False)`
- forward: `lgpa_input = featmaps[-1].detach() if self._lgpa_detach else featmaps[-1]`
- test path: 不变 (eval 时 detach 无意义, 无梯度)

### pose_psg_lgpa_detach.yml
- 与 pose_psg_lgpa.yml 完全相同, 仅增加 `POSE_LGPA_DETACH: True`

## 验证

| 检查项 | 状态 |
|--------|------|
| .detach() 正确应用 | PASS |
| 默认值安全 (False) | PASS |
| Test path 无变化 | PASS |
| 单变量 (仅 DETACH) | PASS |
| Loss 结构不变 | PASS |
| OA-SD 兼容 | PASS (teacher 也走 detach path) |
| AMP 安全 | PASS (.detach() 不影响 dtype) |

## 结论

审查通过。变更极小 (3行代码 + 1行配置), 风险极低。
