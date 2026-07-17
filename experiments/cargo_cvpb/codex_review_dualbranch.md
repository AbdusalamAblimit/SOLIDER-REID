# Codex Review — AIRL 单模型双分支(--airl_dualbranch)

**Verdict**: **approve**(round 3; round 1 = needs-attention → framing/w-lock/wording 修复 → round 2 = needs-attention[1 新 Medium + 1 Low]→ 文档化/收紧 → round 3 = approve)
**Date**: 2026-06-23
**Review round**: 3(final）

---

## Round 1 — needs-attention(framing/protocol/wording, 非代码崩坏)

### High / novelty collision
"按 query 像素预算路由证据空间" 撞 CRReID 的 RAR(resolution-adaptive metric,用 query resolution 选共享子空间)+ MRJL(multi-resolution dual-branch fusion)。**且当前实现并非动态路由,是固定 w=0.25 全局软融合。** 建议 framing 避开 "query-budget routing" 主 claim,改成 "observation-limited ceiling 下 clean/recover evidence head 分化 + 固定先验融合"。Sources: RAR(arxiv 2207.13037)/MRJL(2105.12684)/DI-REID(2004.04933)。

### Medium / protocol
设计说 w=0.25 固定非 test 调,但代码暴露 `--airl_fuse_w` 任意可传(仅校验 [0,1])。建议 headline 路径锁 w=0.25,扫 w 放单独 ablation flag。

### Low / NaN-safe wording
consistency 确在 autocast disabled fp32 计算,但无 nan_to_num/finite guard。建议文档 "NaN-safe" 收紧为 "fp32 for numeric safety" 或加 finite guard。

### Checklist(codex round-1 全过)
①off 不建 2nd head ✓ ②两 head 都进 optimizer(Swin full-LR + assert)✓ ③f_full 零 consistency 梯度 / f_rec 有 CE+consistency ✓ ④eval 融合公式 `dm_fuse=2−2(w·s_rec+(1−w)·s_full)` 正确 ✓ ⑤AMP fp32 ✓(见 Low)⑥w 默认 0.25(见 Medium)⑦triplet 不重复 ✓ ⑧classifier_rec 注册 ✓

---

## 修复(只改注释/文档/measure,不改训练行为；lab-3090 正跑此代码,未 sync)

1. **High framing**: `afd_train.py` --airl_dualbranch 注释块(~1474)+ `new_angle_AIRL.md`(~160)framing 改为 **"observation-limited evidence ceiling 下 clean(f_full)/recover(f_rec)evidence head 分化 + 固定先验软融合(fixed-prior fusion)"**,明确写 **不是 query-budget routing / 动态 router**(kill-switch #3 已证硬路由失败 ≤+0.41,增益全来自固定 w 软混合),避 RAR/MRJL/cross-resolution 撞车。"dual-branch routes it to f_rec" 措辞改 "applies it to f_rec"。
2. **Medium w-lock(软)**: `--airl_fuse_w` help 标 **"ABLATION-ONLY; headline fixed at 0.25"**;parse 时若 `airl_dualbranch and w != 0.25` → print `[AIRL-DUAL][WARN]`(软保护,**不 hard assert**,扫 w 消融仍要用)。默认仍 0.25。
3. **Low wording**: consistency 注释 "NaN-safe" → "fp32 for numeric safety (finite inputs)"(docstring + 行内 + md review-point #4);`airl_consistency_loss` 两个 return 加 `torch.nan_to_num` 轻 finite guard(finite 输入下 value/gradient 恒等,no-op,不改训练行为)。

**修复后**: py_compile 过;`smoke_airl_dualbranch.py` 11/11、`smoke_airl.py` 21/21 全过(逻辑未改,D7/S9 extreme-logits 仍 finite)。

---

## Round 2 — needs-attention(1 新 Medium + 1 Low)

- **Medium(新)**: 退化 forward 是整模型 `model(deg_imgs)`(模型无 rec-only 路径),f_full 的 frozen-bias BNNeck running mean/var 会"看到"退化 ground 图(仅统计跟踪)。codex 指出这弱化"f_full stays clean"协议。
- **Low**: `new_angle_AIRL.md` 第157行 smoke 摘要里 `D7 ... NaN-safe` 措辞仍旧。
- 其余全 confirm 修好:framing 诚实(fixed-prior,非 routing,避 RAR/MRJL,web 查无 AG-ReID 精确先例);w-lock 正确(默认 0.25 + ablation-only + WARN,不挡 sweep);`nan_to_num` finite 输入下恒等;主路径 checklist 全过。

### Round-2 处理(仍是 framing/wording 范围,不改训练行为)
- **Medium → 文档化为已知接受项,刻意不改**:该 BN-stat 暴露是 `--airl` 单头路径(已接受、kill-switch #2 PASS 用的就是它)**完全相同的预存行为**(同一 degrade+forward 原语,afd_train.py ~1958),非本轮引入;任务限定 framing/wording 且禁止改训练行为;lab-3090 正跑此代码,改 rec-only forward 会改训练行为 + 与运行中训练 desync。→ 在退化 forward 注释(~1973-1987)写清 f_full 零 consistency **梯度**(smoke D4)、BN-stat 仅统计跟踪、与 --airl 对齐保消融诚实、由 kill-switch #4 实证裁决;`new_angle_AIRL.md` review-point #2 同步记录。
- **Low → 修**:md 第157行 D7 改为 "f_rec consistency 输出 finite(... fp32 + nan_to_num finite guard)"。
- 修复后 py_compile 过 + 11/11 + 21/21 不破。

---

## Round 3 — approve(final)

**Verdict: approve. Findings: none.**

codex 确认:Round-2 的 BN-stat 项在"纯 framing/protocol 轮 + 不改训练行为 + 与运行中 --airl 对齐"约束下,**文档化(而非改 forward)是可接受的解决**;若 kill-switch #4 后证明有实质影响,再作为单独的 behavior-changing 实验。三处原始修复(fixed-prior framing 避 RAR/MRJL、软 w-lock、fp32+nan_to_num wording)与主路径 checklist(off 字节级、两 head 入优化器 + f_rec Swin full-LR、f_full 零 consistency 梯度、eval `2−2(w·cos_rec+(1−w)·cos_full)`、fp32 consistency under autocast disabled、w 默认 0.25、triplet 不重复、classifier_rec 注册 + CE 接地)全 verified。

## 结论
codex 审查通过(approve)。双审 protocol 闭环:Claude broad review(此前) + Codex round-3 approve,双分支训练结果出来时 review 已 ready。修复仅改注释/文档/measure(framing 对齐 fixed-prior fusion、软 w-lock warning、wording 收紧 + 轻 finite guard),**未改训练行为,未 sync lab-3090**。
