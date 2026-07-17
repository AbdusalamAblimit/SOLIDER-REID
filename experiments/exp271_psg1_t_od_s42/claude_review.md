# Claude Review — exp271_psg1_t_od_s42

**审查对象**: Phase 3-A 的第二个 run,在 exp270 baseline 基础上加 1-stage PSG

## 审查范围

1. `design.md` — 单变量增加 PSG stage 3,其他 pose 模块仍关
2. 代码改动: **无**(纯 CLI override,不新增/修改任何文件,沿用 commit `4aec3ee` 代码)
3. 配置: 相对 exp270 只增 `POSE_BACKBONE_PSG=True` + `POSE_PSG_STAGES=[-1]`
4. `config/defaults.py`: PSG 默认 stages list 取 `POSE_PSG_STAGES` 读取,本 run 覆盖为 `[-1]`
5. 代码路径 (不同于 exp270):
   - `make_model.py:467` 进 `POSE_BACKBONE_PSG=True` 分支 → `from .pose_backbone_model import PoseBackboneModel` (不触发 exp270 的死 import)
   - `PoseBackboneModel` 构造时带 PSG gate 模块,仅注入 stage 3 (最后一个 swin stage)
   - LGPA/GCN/OA-SD 都有 `if cfg.MODEL.POSE_LGPA: ...` 守卫,关闭时不构造
6. 与 exp270 对照: **单变量 = PSG 是否启用**(纯净对照)

## 单变量原则检查

与 exp270 的差异:
- `POSE_BACKBONE_PSG`: False → True
- `POSE_PSG_STAGES`: (n/a) → `[-1]`

这是 Phase 3-A 的 **核心科学问题**: PSG 在 pure backbone setting 下的独立贡献。exp270 vs exp271 单变量隔离完美。

## 代码安全性检查

1. `POSE_BACKBONE_PSG=True` 路径在 Phase 1 所有 9 个 run 都用过,代码经过充分验证
2. `POSE_PSG_STAGES=[-1]` 等于 exp007 的历史配置,无 regression 风险
3. 其他关的模块(LGPA/GCN/OA-SD/PLBOA/Parallel-Aug)都有对应 `if` 守卫,关闭时不构造
4. `POSE_ENABLED=True` 走 pose 分支,但 eval 时用 `POSE_TEST_FEAT='global'`,不用 branch fusion
5. OOM 风险: 低。Tiny + PSG only ~6-7GB 峰值,eval 加 flip ~9-10GB,远低于 16GB

## 结论

**审查通过**。exp270 → exp271 的单变量差异清楚,代码零改动,期望数字 ~59-60(对照 exp007 历史 58.3 + default flip +1 ≈ 59)。
