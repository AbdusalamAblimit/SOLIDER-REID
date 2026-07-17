# Claude Broad Review: exp181 SupCon T=0.05 without PLBOA (Opus 4.6)

## 审查范围

a. design.md — 合理性、单变量原则
b. 代码变更 — 无新增/修改代码（纯 CLI 覆盖）
c. 配置文件 — base config 中 PLBOA 和 SupCon 设置
d. CLI 覆盖机制 — `MODEL.POSE_LOWER_BODY_OCC False` 是否正确生效
e. 单变量隔离 vs exp176

---

## a. design.md

消融实验目的清晰：isolate SupCon 效果，测试无 PLBOA 时 SupCon 是否仍有效。对照组正确选择 exp166r (CE, no PLBOA) 和 exp176 (SupCon + PLBOA)。

**注意**: design.md 标题写 "SupCon T=0.05"，但 base config `pose_psg_stdpr_pertoken_plboa_pape_ms_supcon.yml` 内 `POSE_STR_SUPCON_TEMP: 0.07`。exp176 也是通过 CLI 覆盖到 0.05。因此 exp181 的 CLI 需要同时覆盖两个参数：
1. `MODEL.POSE_LOWER_BODY_OCC False`
2. `MODEL.POSE_STR_SUPCON_TEMP 0.05`

**Medium**: 请确认启动命令包含 `MODEL.POSE_STR_SUPCON_TEMP 0.05`，否则实际运行的是 T=0.07 而非 T=0.05，无法与 exp176 (T=0.05) 构成单变量消融。

## b. 代码变更

无新增或修改的 .py 文件（git status 确认 exp181/ 目录仅含 design.md）。纯 CLI 覆盖实验，无代码层面风险。

## c. 配置文件

Base config 关键设置：
- `POSE_STR_SUPCON: True` — SupCon 启用
- `POSE_STR_SUPCON_TEMP: 0.07` — 需 CLI 覆盖为 0.05
- `POSE_LOWER_BODY_OCC: True` — 需 CLI 覆盖为 False
- `POSE_LOWER_BODY_OCC_PROB: 0.7` — PLBOA 关闭后此项无效
- `OUTPUT_DIR: ./log/occluded_duke/exp174_triple_supcon` — 需 CLI 覆盖为 exp181

其他设置（triple injection, PAPE, STD-PR, PSG stages 等）与 exp176 一致。

## d. CLI 覆盖机制

`train.py` line 39: `cfg.merge_from_list(args.opts)` — yacs 标准机制。

`MODEL.POSE_LOWER_BODY_OCC False` 的处理：
- defaults.py 中 `_C.MODEL.POSE_LOWER_BODY_OCC = False`（布尔类型）
- yacs `merge_from_list` 对布尔类型：CLI 传入字符串 "False" 会被 yacs 正确解析为 Python `False`
- `make_dataloader.py:107`: `if getattr(cfg.MODEL, 'POSE_LOWER_BODY_OCC', False)` — False 时跳过 PLBOA 设置
- `make_dataloader.py:80`: `if ... or getattr(cfg.MODEL, 'POSE_LOWER_BODY_OCC', False)` — 如果 POSE_ROA 也为 False，则不加载 occluders（节省内存和启动时间）

PLBOA 完全依赖 `POSE_LOWER_BODY_OCC` 开关，关闭后下游无残留影响。正确。

## e. 单变量隔离 vs exp176

假设 CLI 命令正确包含 `MODEL.POSE_STR_SUPCON_TEMP 0.05`：

| 参数 | exp176 | exp181 |
|------|--------|--------|
| POSE_LOWER_BODY_OCC | True | **False** |
| POSE_STR_SUPCON | True | True |
| POSE_STR_SUPCON_TEMP | 0.05 (CLI) | 0.05 (CLI) |
| 其他 | 相同 | 相同 |

严格单变量：仅 PLBOA 开/关。

## 问题清单

| 级别 | 问题 | 状态 |
|------|------|------|
| Medium | CLI 必须包含 `MODEL.POSE_STR_SUPCON_TEMP 0.05`，否则实际 T=0.07 | 需确认 |

---

## 审查通过

前提条件：启动命令必须包含以下 CLI 覆盖（完整示例）：
```bash
MODEL.POSE_LOWER_BODY_OCC False MODEL.POSE_STR_SUPCON_TEMP 0.05 OUTPUT_DIR ./log/occluded_duke/exp181_supcon_no_plboa
```

如三个 CLI 覆盖均正确包含，则实验设计正确、单变量隔离、无代码风险。
