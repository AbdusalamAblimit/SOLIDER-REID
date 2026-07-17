# Claude Broad Review — exp355 PGPD

## Round 1: NEEDS-FIX (Critical C1)
**C1 (Critical)**: `self.use_pgpd` init 被误放进 `if self.use_clip_id_noparam_pool:` 块(indent 16)。exp355 未开 noparam_pool(默认 False)→ `self.use_pgpd` 永不创建 → forward 守卫 `getattr(self,'use_pgpd',False)` 返回 False → `_pgpd_loss` 永不调用 → **PGPD 静默失效, exp355 == exp341**。单变量对照将变成空对照。
**修复**: de-indent PGPD init 从 16 → 12, 移出 noparam_pool if, 进 `if self.use_clip_id_prompt:`(与 use_clip_id_noparam_pool 同级)。+ 加首call诊断 print(teacher 覆盖率/mean_w/mean_dark, 早发现 w≈0 空转)。

## Round 2: PASS — 审查通过
全范围复审(非仅修复点)。6 个 focus area 全过, 无 Critical/High/Medium。

**(a) C1 修复确认**: `self.use_pgpd` 现在 line 250 indent 12, 在 `if self.use_clip_id_prompt:`(line 208)内、noparam_pool if 外。exp355 POSE_CLIP_ID_PROMPT True + NOPARAM_POOL 未设(False)→ use_pgpd=True, forward 守卫 line 700 为 True。激活确认。

**(b) _pgpd_loss math 正确**:
- unique-prototype scatter `uniq_protos[inv]=txt_proto`: pose_cond=False 故同 ID prototype 相同, scatter 写序无关一致。P<3 跳过(需 ≥2 硬负)。
- teacher 选择轴正确: row=student col=teacher, `comp[j]>comp[i]` 严格更完整, argmax(dim=1) 选最完整 teacher。
- 硬负屏蔽对 student+teacher 双方正确(teacher 同 label → inv 相同 → 同列是 teacher 真 ID)。
- KD 方向正确: student log_softmax, teacher softmax+detach, `-(teacher_p*student_logp)` = student 学 teacher。
- NaN 防护: 0*(-inf) 经 masked_fill(prod,0)+nan_to_num 双保险。
- w 归一化 `/w.sum().clamp(min=1e-6)` 全零安全。

**(c) 数据流单计数**: PGPD 加到 clip_id_loss(line 701)→ return line 978 {'clip_id_loss'} → processor 1297-1302 `clip_id_w * clip_id_loss` → backward。SupCon i2t/t2i 只算一次, PGPD 是独立加项, 无双计数。

**(d) 单变量**: 仅 4 个 POSE_PGPD* flag + OUTPUT_DIR 与 exp341 不同, 其余全同。POSE_PGPD False 完全复现 exp341。

**(e) AMP/边界**: fp32 softmax; P<3 跳过; 全异 ID batch(全 w=0 → loss 0); has_teacher=False 行被 w=0 nullify; 首call诊断 print 在 return 前, 变量都已定义, _pgpd_logged 守卫只打一次。

**(f) 修复无新问题**: 缩进正确(ast.parse OK), 诊断 print 安全。

## Low (非阻断)
- L1: `_pgpd_loss` 内局部 `import torch.nn.functional as F` 与模块级(line 11)冗余, 同绑定无害。

## 结论
**审查通过.** C1 已修, PGPD 激活, dark-KD math 健全, 数据流单计数, 单变量隔离成立, AMP/边界处理完备。可进 Codex 审查。
