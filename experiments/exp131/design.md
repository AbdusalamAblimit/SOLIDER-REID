# 实验 exp131: Cross-Batch Pair SCRD

## 动机

`exp125` 已经证明结构化 pair routing 有效，`exp130` 又进一步说明 `target form` 不是当前主瓶颈。当前更合理的缺口是：

1. teacher-change pairs 本身是稀疏的
2. batch-only `CSRD` 每次只能看到当前 64 张图里的 relations
3. 即使 routing 正确，informative changed pairs 的覆盖仍可能不够

因此本轮不改 teacher、不改 target、不改 routing，而是只扩大 **可蒸馏 relations 的覆盖范围**。

## 核心假设

如果当前瓶颈主要是 batch 内 changed-pair coverage 不足，那么在保持 `exp125` 的 online support teacher 与 `delta_top` routing 不变的前提下，引入 cross-batch relation queue 应当带来更强的 late-stage 收益。

## 技术方案

1. 保持 `exp125` 的：
   - `POSE_CSRD_SUPPORT_TEACHER = True`
   - `POSE_CSRD_PAIR_WEIGHT_MODE = delta_top`
   - `POSE_CSRD_PAIR_WEIGHT_ALPHA = 1.0`
   - `POSE_CSRD_PAIR_TOP_RATIO = 0.25`
   - `target_mode = full`
2. 新增 `POSE_CSRD_QUEUE_SIZE = 256`
3. `CSRD` 仍以当前 batch 样本为 anchor，但 candidate relations 从：
   - batch 内 positives / negatives
   扩展为：
   - batch 内 + cross-batch queue
4. queue 中存储：
   - global student feat
   - base kp feats / kp weights
   - support-complete teacher kp feats
   - labels
5. routing 仍使用 `delta_top`，只是 top-delta 的候选集合从 batch 内扩展到 batch+queue

## 对照组

- 直接对照: `exp125`
- 机制参照: `exp126`（它回答“真稀疏 routing 是否更优”，而 `exp131` 回答“relation coverage 是否不足”）

## 预期结果

1. 若假设成立：
   - `ep50/60+` 应比 `exp125` 更早或更强地转正
   - 日志中应看到 `csrd_qn > 0`
   - `csrd_qr` 应表明 queue relations 实际参与了 distillation
2. 若结果仍与 `exp125` 近乎等价：
   - 说明当前主瓶颈不在 coverage，而更可能在 student capacity / relation form

## 风险与失败解释

1. cross-batch queue 可能引入过时 relations，导致噪声增大
2. 若收益为负，可能说明：
   - current batch 的 on-the-fly relations 已足够
   - 或 stale queue 比新鲜 batch relations 更不可靠
3. 若 `csrd_qr` 很低，则说明 queue 虽接入但未真正改变 supervision
