# exp214 审查: Small GCN+PAA + 3-view (无 OA-SD)

## 审查范围

a. design.md 合理性
b. 数据集 parallel_aug 路径 (pose_dataset.py, lines 197-229)
c. 处理器 3-view 训练路径 (processor.py, lines 446-690)
d. make_dataloader.py 标志设置 (lines 122-127)
e. config/defaults.py 默认值安全性
f. 与 exp206r 对照

## 审查结论

**审查通过**

## 详细审查

### a. design.md — Low Risk

- 动机清晰：3-view 在 Tiny 有效 (+1.4%)，尝试在 Small 上去掉 OA-SD 的纯 3-view
- 假设明确：3-view CE/triplet 平均 → +1-2% mAP
- 对照组正确：exp206r (1-view + OA-SD) 作为参照
- 单变量原则：相比 exp206r，仅 OA-SD → off，parallel_aug → on

### b. 数据集路径 — 正确

**关键流程 (POSE_OA_SD=False, POSE_PARALLEL_AUG=True):**

1. `make_dataloader.py` line 123-124: `parallel_aug = True` ✓
2. `make_dataloader.py` line 126-127: `_oa_sd_mode` 不被设置，保持 False ✓
3. `pose_dataset.py` line 170: `img_clean_for_oa_sd = None` (因为 `_oa_sd_mode=False`) ✓
4. `pose_dataset.py` line 197-229: 进入 parallel_aug 分支
   - view 1: 标准 RE (概率 0.5)
   - view 2: ROA (无概率，始终应用)
   - view 3: 强制 RE (概率 1.0)
5. `pose_dataset.py` line 228-229: `img_clean_for_oa_sd is None` → else 分支 → `img_tensor = (view1, view2, view3)` — 3-tuple ✓
6. `pose_dataset.py` line 274 不会执行 (在 parallel_aug 分支内已 return) ✓

**collate_fn (line 1089-1105):**
- `n_views = 3` → 返回 list of 3 stacked tensors ✓

### c. 处理器路径 — 正确

**关键变量:**
- `oa_sd_enabled = False` (line 403, POSE_OA_SD=False) ✓
- `ema_teacher = None` (line 405, 不进入 line 407 的 if block) ✓
- `parallel_aug = True` (line 449, img 是 3 元素 list) ✓
- `oa_sd_mode = False` (line 450, len(img)=3 ≠ 2) ✓
- `parallel_oa_sd = False` (line 452, oa_sd_enabled=False) ✓

**Forward 路径 (line 453-458):**
- `parallel_oa_sd=False` → `img_views = [v.to(device) for v in img]` — 全部 3 views ✓

**3-view forward loop (line 473-492):**
- 对每个 view 做 model forward，存储 score/feat/recon/kp_data ✓
- view 0 的输出用于主 loss + kp_aux_data ✓

**Loss averaging (line 681-690):**
- view 1, 2 的 loss 加入总 loss，然后除以 3 ✓
- 只有 view 0 参与 kp_aux_data / LTCS / LPCS 等辅助 loss ✓

**OA-SD block (line 693):**
- `oa_sd_enabled=False` → 整个 block 跳过 ✓

**OA-RD block (line 763):**
- `oa_rd_enabled=False` → 整个 block 跳过 ✓

**EMA update (line 943):**
- `ema_teacher is None` → 跳过 ✓

### d. defaults.py — 安全

- `POSE_PARALLEL_AUG` 默认 False (line 177) ✓
- `POSE_OA_SD` 默认 False ✓
- 不影响已有实验复现 ✓

### e. 与 exp206r 对照

| 设置 | exp206r | exp214 | 变量隔离 |
|------|---------|--------|----------|
| POSE_OA_SD | True | False | ✓ 变化 |
| POSE_PARALLEL_AUG | False | True | ✓ 变化 |
| Backbone | Swin-Small | Swin-Small | 相同 |
| BASE_LR | 0.0004 | 0.0004 | 相同 |
| PLBOA | True | True | 相同 |
| ROA | True | True | 相同 |
| WITH_CP | False | False | 相同 |

注意：严格来说改了两个变量（OA-SD off + parallel_aug on），不是纯单变量实验。但设计意图是用 3-view 替换 OA-SD 作为正则化手段，所以两个变量同时改是合理的。

### f. 潜在风险

**Medium — 内存压力:**
3 个 with-grad forward pass 同时保持计算图（直到 line 937 的 backward()），内存占用约为单 view 的 3 倍。Swin-Small + bs64 + 384x128 在 3090 (24GB) 上可能紧张。若 OOM，启用 `WITH_CP=True` 即可。

此前 exp206 3-view+OA-SD 在 WITH_CP=True 下成功运行（4 views），但出现学习停滞。exp214 去掉了 OA-SD（3 views）且不用 CP，内存应该是够的（3 with-grad < 3 with-grad + 1 no-grad + CP overhead），但建议监控前几个 iter 的 GPU 内存。

**Low — PLBOA 仍启用:**
没有 OA-SD 的 teacher-student 不对称，PLBOA 只作为数据增强。3-view 中所有 view 共享相同的 PLBOA 变换（在 branching 前应用），这是正确的。

## 结论

代码逻辑正确。3-view 无 OA-SD 路径经过完整追踪：
1. 数据集正确生成 3 个 view (不带 teacher)
2. collate 正确打包为 list of 3 tensors
3. 处理器正确识别 parallel_aug 模式
4. 3 次 forward + loss 平均 + backward 逻辑无误
5. OA-SD/OA-RD/EMA 全部正确跳过

**审查通过** — 可以启动训练。
