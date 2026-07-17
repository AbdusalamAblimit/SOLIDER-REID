# exp023 PDS + Stop Gradient 审查记录

## 审查轮次 1 — PASS

**审查日期**: 2026-03-11
**审查者**: Opus Agent

### 审查范围
- experiments/exp023/design.md
- model/pose_dual_stream_model.py
- configs/occluded_duke/pose_pds_stopgrad.yml
- config/defaults.py

### 审查结论: PASS

关键验证点：
1. **`.detach()` 位置正确** — `shared_x.detach()` 正确阻断 Part→共享层梯度
2. **`part_input` 在训练和测试路径都正确使用**
3. **Config 一致性** — POSE_PART_STOP_GRAD 正确添加并读取
4. **语义权重无梯度泄漏** — `_run_part_branch` 中的 semantic weight 是死代码（结果未使用），不影响梯度流
5. **单变量消融** — 相对 exp022 仅添加 stop_gradient

### 注意事项（非阻塞）
- `getattr(cfg.MODEL, 'POSE_PART_STOP_GRAD', False)` 使用 getattr 是冗余的（defaults.py 已定义），但无害
- Stage 3 语义权重在两个分支都是死代码，继承自原始 backbone 行为
