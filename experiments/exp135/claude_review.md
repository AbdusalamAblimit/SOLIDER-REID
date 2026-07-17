Good. Now I have a complete picture. Let me write the review report.

---

# exp135 代码审查报告：Corrected LPCS Clean Rerun

## 1. 审查结论：**允许启动**

---

## 2. Critical

**无。**

---

## 3. High

**无。**

---

## 4. Medium

### M1: processor.py 尚未提交

`git status` 显示 `processor/processor.py` 处于 `M`（modified）状态。如果在启动训练前不提交，exp135 的代码状态将无法被 git hash 精确回溯。

**建议**：启动训练前先 `git commit`，把 processor.py 的 bug fix 锁定到 exp135 的 commit。

### M2: `_fix` 配置文件与原配置文件内容冗余

`pose_psg_gcn_lpcs_fix.yml` 与 `pose_psg_gcn_lpcs.yml` 内容完全相同（仅 OUTPUT_DIR 不同）。两个几乎相同的配置文件容易造成后续维护混乱。

**建议**：可接受。但后续如需修改 LPCS 参数，需要同步两个文件或删除旧的。

---

## 5. Low

### L1: `lpcs_teacher_bank` 在 warmup 期间用早期特征更新

`processor.py:1073-1078` 中 bank 更新无 epoch 下限。epoch 1-20 期间 backbone 尚未收敛，bank 中积累的是低质量特征。虽然 EMA（momentum=0.9）会在后续指数衰减这些早期特征，但理论上在 epoch 21 激活时 bank 质量可能不够理想。

**影响**：设计本身如此（与 exp133 intended 设计一致），不构成阻塞。

### L2: warmup 期间 `lpcs_head` 仅受 weight decay 影响

epoch 1-20 期间 LPCS loss 不会被计算，但 `lpcs_head` 的参数（1313 个）仍然在 optimizer 中接受 weight decay（1e-4）。20 个 epoch 的微量 decay 对 tanh(0)=0 的 zero-init 权重影响极小。

**影响**：可忽略，与 exp133 intended 设计一致。

---

## 6. 逐项检查表

| 检查项 | 结论 | 详细说明 |
|--------|------|----------|
| **共享接线 bug 是否已真正修复** | **已修复** | `processor.py:602` 条件已包含 `ltcs_enabled or lpcs_enabled`。当 `POSE_LPCS=True` 且无其他 kp 辅助 flag 时，`kp_aux_data` 会被正确构建。exp133 失效的根因已消除。 |
| **单变量性** | **满足** | 相对 exp133 intended 设计，唯一变化是 `processor.py` 的 bug fix（line 602 新增条件）+ OUTPUT_DIR。配置文件参数完全一致。 |
| **默认行为是否被破坏** | **未破坏** | `lpcs_enabled` 默认为 `False`（`defaults.py:245`）。当 `POSE_LPCS=False` 时，在 OR 条件中不影响 `kp_aux_data` 构建。所有已有实验不受影响。 |
| **LPCS loss 是否真的会进入训练** | **会** | 完整数据流追踪：(1) `lpcs_enabled=True` → (2) `kp_aux_data` 在 line 605 被构建 → (3) epoch>20 时 `lpcs_teacher_bank.replace()` 被调用（line 639-646），`lpcs_teacher_feats` 写入 `kp_aux_data` → (4) `_compute_lpcs_loss()` 在 line 682-704 被调用 → (5) `loss += lpcs_weight * lpcs_loss` → (6) 梯度仅流经 `lpcs_head`（所有输入均 `.detach()`），不干扰 backbone。 |
| **日志是否会出现 `lpcs_*`** | **会** | `details` dict 在 line 695-704 被填充 `lpcs`, `lpcs_dm`, `lpcs_ds`, `lpcs_sm`, `lpcs_cm`, `lpcs_wm`, `lpcs_bg`, `lpcs_fg`, `lpcs_psr`, `lpcs_pf`。这些通过 `detail_meters`（line 1094-1098）进入日志输出（line 1105）。首次出现应在 epoch 21 第一个 log period。 |
| **bank 更新是否正常** | **正常** | `processor.py:1073-1078` 使用 `kp_data`（非 `kp_aux_data`），bank 更新路径从 epoch 1 开始工作，不受本次 bug fix 影响。exp133 的 bank 更新本身也是正常的，只是 replacement + loss 计算被跳过了。 |
| **evaluator 接线** | **正确** | line 1153/1181/1227 均正确设置 `evaluator.pair_residual_head`。`metrics.py:277-279` 在 mode=`cvk_residual` 时正确使用该 head。 |
| **optimizer 覆盖** | **正确** | `make_optimizer.py` 遍历 `model.named_parameters()` 所有 requires_grad 参数。`lpcs_head` 作为模型子模块自动包含。 |
| **本地启动风险** | **低** | 3090 24GB。LPCS head 仅 1313 参数，bank 内存为 `num_classes * 17 * 768 * float32 ≈ 30MB`。训练期间 LPCS loss 构建 `(B, B)` 距离矩阵若干个（B=64），每个约 16KB，总增加极小。无 OOM 风险。 |
| **配置参数与代码默认值一致** | **一致** | yml 中每个 `POSE_LPCS_*` 参数与 `defaults.py:245-256` 的默认值和 `processor.py:180-205` 的读取逻辑完全吻合。 |
| **与 exp133 log 目录隔离** | **已隔离** | exp133: `exp133_lpcs`，exp135: `exp135_lpcs_fix`。不会互相覆盖。 |

---

## 7. 最终结论

**允许启动。**

exp133 失效的根因（`kp_aux_data` 构建条件缺少 `lpcs_enabled`）已在当前 `processor.py:602` 中被修复。修复精确且最小——只在 OR 条件中加入了 `ltcs_enabled or lpcs_enabled`，不改变任何其他逻辑路径。配置文件与 exp133 intended 设计完全一致（仅 OUTPUT_DIR 不同），满足单变量原则。

**启动前必须完成的唯一动作**：将 `processor.py` 的改动 `git commit`，确保 exp135 的代码状态可追溯。
