# exp086 PA-PAT 代码审查记录

## 第一轮审查 — 未通过 (2C + 4H)
- **C1**: `img.shape[0]` 对 list 崩溃 → 改用 `batch_size` 变量
- **C2**: `_loss_details` 在 loss 平均后丢失 → 保存并重新附加
- **H1**: `persons` 原地修改影响共享 pose_dict → 跳过修改
- **H2**: design doc vs 实现不匹配 → 更新 doc
- **H3**: PAMC 兼容性（当前不触发）
- **H4**: OOM 风险 → 用户说可以开 WITH_CP

## 第二轮审查 — 通过 ✅
- C1 fix: `batch_size` 在两处使用，正确
- C2 fix: `saved_details` 保存 → 平均 → 重新附加，正确
- H1 fix: parallel 路径不修改 persons，正确
- backward: 单次 backward 在 3 view loss 平均后，正确
