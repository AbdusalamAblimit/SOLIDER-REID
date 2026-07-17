# Claude Review: exp200 — OA-RD + CE (远程对照)

**审查日期**: 2026-03-30
**审查范围**: config-only experiment, OA-RD code already reviewed in exp199/claude_review.md

---

## a. 实验合理性

- exp200 是 exp199 (OA-RD + SupCon + 3-view) 的 CE 路线对照
- 与 exp191 (OA-SD + CE, 63.2/75.4) 直接对比：relational vs feature distillation
- 与 exp166r (CE base, 60.3/72.8) 对比：OA-RD 的独立贡献
- 单变量原则满足：仅在 CE base 上加 OA-RD
- **不是小调参**：复用 exp199 已审查代码，探索 OA-RD 在不同 loss 路线的泛化性

**通过。**

---

## b. 数据管道：1-view + OA-RD 2-view tuple

### b.1 Dataset 端 (`datasets/pose_dataset.py`)

- L126 (`make_dataloader.py`): `POSE_OA_RD=True` 触发 `train_set._oa_sd_mode = True`
- L167: `self._oa_sd_mode = getattr(self, '_oa_sd_mode', False)` — 安全读取
- L171-172: 保存 `img_clean_for_oa_sd = img.copy()` BEFORE PLBOA
- `parallel_aug=False`（无 3-view），走标准单视图管道 (L228+)
- L264-269: `img_clean_for_oa_sd is not None` → 创建 teacher tensor → 返回 `(img_tensor, img_clean_tensor)` 2-element tuple

**正确：student 经过 PLBOA/RE，teacher 从 clean 副本独立生成。**

### b.2 Collate 端 (`pose_train_collate_fn`)

- L1057: `n_views = len(img_tuples[0])` = 2
- L1064: 走 `else` 分支 → 返回 `imgs = [tensor_view0, tensor_view1]` — **list of 2 tensors**

**正确：collate 将 tuple 转为 list，与 processor 期望的格式匹配。**

---

## c. Processor 端 (`processor/processor.py`)

### c.1 模式检测 (L441-442)

- `parallel_aug = isinstance(img, list) and len(img) >= 3` → False (len=2)
- `oa_sd_mode = isinstance(img, list) and len(img) == 2` → **True**

**正确：1-view + OA-RD 被识别为 oa_sd_mode。**

### c.2 数据解包 (L452-455)

- `img_student = img[0].to(device)` — occluded view
- `img_teacher = img[1].to(device)` — clean view
- `img = img_student` — 后续正常 forward 使用 student

**正确：`img_teacher` 变量在此处定义，后续 OA-RD 可用。**

### c.3 EMA Teacher 创建 (L394-409)

- L396: `oa_rd_enabled = getattr(cfg.MODEL, 'POSE_OA_RD', False)` → True
- L399: `if oa_sd_enabled or oa_rd_enabled` → True → 创建 `ema_teacher`
- EMA decay 使用 `POSE_OA_SD_EMA_DECAY`（复用，默认 0.999）

**正确：ema_teacher 在 OA-RD-only 模式下正确创建。**

### c.4 OA-RD 触发 (L736)

- `if oa_rd_enabled and (oa_sd_mode or parallel_oa_sd) and use_pose and ema_teacher is not None`
- `oa_rd_enabled=True`, `oa_sd_mode=True`, `use_pose=True`, `ema_teacher` 已创建
- **触发条件满足。**

### c.5 Teacher Forward — OA-RD only path (L741-755)

- L741: `if not oa_sd_enabled` → True（OA-SD 未启用）
- L743-748: 独立运行 teacher forward with `img_teacher`（已在 L454 定义）
- Teacher 输出解包处理 3/4/5 元素格式
- `teacher_feat` 在此处赋值

**正确：OA-RD-only 模式下 teacher forward 独立运行，无 NameError 风险。**

### c.6 OA-SD 块 (L685) 不触发

- `if oa_sd_enabled and ...` — `oa_sd_enabled=False` → 整个 OA-SD 块跳过

**正确：两个 distillation 块互不干扰。**

---

## d. Config 默认值安全性

- `POSE_OA_RD=False` 默认 → 不影响其他实验
- `POSE_OA_RD_TEMP=0.1`, `POSE_OA_RD_WEIGHT=1.0` — 与 exp199 相同
- 无新增代码修改，仅 config 参数

**安全。**

---

## e. 日志充分性

- OA-RD loss 通过 `details['oa_rd']` 记录，可在 monitor 中观察
- EMA teacher 创建时打印日志 `[OA-RD] EMA teacher created`

**充分。**

---

## 审查结论

| 级别 | 发现 | 状态 |
|------|------|------|
| — | 无新发现 | OA-RD 代码已在 exp199 审查中通过 |

1-view + OA-RD 模式下数据管道完整验证：
- Dataset 正确生成 2-view tuple (student + teacher)
- Collate 正确转为 list
- Processor 正确识别 `oa_sd_mode`，解包 `img_teacher`
- EMA teacher 在 OA-RD-only 下正确创建
- Teacher forward 独立运行（不依赖 OA-SD）
- OA-SD 块不触发，OA-RD 块正确触发

**审查通过**
