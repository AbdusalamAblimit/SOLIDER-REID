# Codex Review — exp358 cross-part (channel) shuffle

**Verdict**: approve
**Date**: 2026-06-21

## 结论
codex 审查通过(verdict: approve to run, 无 blocking 实现 bug)。验证(commit fe4e089):
- gather 无 runtime blocker: cperm per-image (B,K), idx (B,K,H,W), gather dim1; scene/target 同 idx 配对一致。
- 训练端正确: shuffle 由 self.training 守卫; eval 路径 _lgpa_heatmap 用真未 shuffle 通道。
- 单变量 clean: exp358 仅加 POSE_CHANNEL_SHUFFLE True + OUTPUT_DIR vs exp353; POSE_SHUFFLE 默认 False。
- 判读 caveat(同 Claude): (1)背景置换不变(body_max 全前景 PART_KPS 并集覆盖 17 通道 → fg/bg 分离保留, 只打乱 5 前景部位身份); (2)NO-DROP 被随机 shuffle-as-regularizer 混淆(每 forward 重采样)→ 不掉应表述为"特定解剖 assignment 在此随机前景分解下非必需", 非"任意固定随机分解都行"的干净证明。
- 非阻断: 未来若 config 同开 POSE_SHUFFLE + POSE_CHANNEL_SHUFFLE 会顺序都触发(exp358 非 bug, 加 mutual-exclusion assert 可防混合诊断)。

双审查(Claude PASS + Codex approve)全过, 可训练。
