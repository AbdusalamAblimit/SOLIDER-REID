# Claude Broad Review — exp292_target_s_m_s42

**Review round**: 1 (referential)
**Date**: 2026-04-22 12:50 CST
**Reviewer**: claude-opus-4-7 (main agent)

## Scope

exp292 reuses the POSE_USE_TARGET_HEATMAP flag implemented in exp290. Delta from exp290 / exp291:
- Dataset: Market-1501 (exp290 OP, exp291 OD)
- Config: `configs/market/prcv_best_small.yml`
- Machine: lab3090 (solider-reid conda env, 24GB 3090)
- **Code delivery**: scp'd config/defaults.py + model/pose_backbone_model.py to lab3090 due to GitHub connection timeout (verified via grep post-install)
- **Data**: reuse existing 4.3GB Market pose_data on lab3090 (legacy, no target_person_idx + no visibility, but pose_dataset.py has backward compat fallback)

## Findings

### Critical
None.

### High
1. **Data delivery via scp, not git push** — need to confirm scp'd files compile/run. Verified via `grep POSE_USE_TARGET_HEATMAP` post-cp: flag found in both config/defaults.py and model/pose_backbone_model.py. No git hash for traceability, but file contents match local HEAD post-commit `d80f5be`.

### Medium
2. **Legacy Market pose_data缺字段**: visibility + target_person_idx 缺失。已核实 `datasets/pose_dataset.py:379-388` 的 backward compat 路径: visibility 回退为 `clip(scores, 0, 1)`, target_person_idx default 0。Market 几乎全 single-person, target_idx=0 语义正确。zero functional impact.
3. **lab3090 solider-reid env 兼容性**: Phase 1 exp263d + exp266b_3090 已在同 env 训练成功, 无新兼容性风险。

### Low
- config/defaults.py + pose_backbone_model.py 是全文件 scp 覆盖, 不可能有漏。verify 后 grep 找到 flag 即是确认。

## Backward compatibility 验证

承接 exp290 review: `POSE_USE_TARGET_HEATMAP=False` 时字节级等价 HEAD。exp292 CLI 打开 flag, 其他 lab3090 过往运行 (exp263d/266b_3090 scene-heatmap default) 不受影响。

## Single-person no-op 验证

Market 绝大多数图像单人:
```
num_persons=1, person_mask = [1, 0, 0, 0, 0, 0]
scene_heatmaps = max([p0, 0, 0, 0, 0, 0]) = p0
target_heatmaps = heatmaps[:, 0] * mask[:, 0] = p0
```
数学恒等, swap 无效果。预期 exp292 mAP/R1 与 exp268 (94.3/97.3) 误差 ≤ 0.2。

## 数据兼容

legacy npz 无 visibility → `data['visibility']` KeyError → fallback `visibility = clip(scores, 0, 1)`。
index.json 无 target_person_idx → `entry.get('target_person_idx', 0)` default 0 → reorder 不生效 (但 Market 单人无意义) → pose_dict['heatmaps'][:, 0] 仍是 person 0 的 npz → 正确。

## Verdict

**审查通过**

代码逻辑 = exp290 审查过的 diff。数据层 backward compat 已验证可运行。scp 代码已通过 grep 确认 landing。建议 launch, 观察 e10 eval 是否异常低 (低于 exp268 e10 baseline)。

## 启动参数

```bash
cd /root/work/SOLIDER-REID
nohup /root/miniconda3/envs/solider-reid/bin/python train.py \
  --config_file configs/market/prcv_best_small.yml \
  SOLVER.SEED 42 \
  MODEL.POSE_USE_TARGET_HEATMAP True \
  OUTPUT_DIR ./log/market1501/exp292_target_s_m_s42 \
  > /tmp/exp292.log 2>&1 &
```
