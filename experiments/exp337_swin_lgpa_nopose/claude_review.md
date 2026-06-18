# Claude Broad Review — exp337 (Swin 纯 LGPA-D + no-pose ablation)

**审查范围**：design.md / exp337.yml vs exp336.yml / git diff(defaults.py + pose_backbone_model.py) / clip_part_head.py None 分支 / processor.py assign_loss 消费 / equal_concat 测试组装 / 单变量隔离。
**结论**：审查通过（PASS）。无 Critical/High/Medium。仅 2 条 Low（解释口径，不阻断）。

## 逐项核验

### Finding 1 — None heatmaps 是否安全产出"纯 CLIP-text parts"（无 NaN/crash）→ PASS
clip_part_head.py forward 的三条 None 分支全部干净：
- `_compute_pose_bias`：line 243-246 `if target_heatmaps is not None ... else pose_bias=None`。pose_bias=None 进 `_cross_attention_with_pose`，line 193 `if pose_bias is not None` 跳过 → attn_scores 就是标准 QK^T/sqrt(d)，再走 clamp[-50,50] + softmax。**不是退化的全等分数**（CLIP 文本原型 5+1 个 query 互不相同，K=spatial tokens 正常），是合法 cross-attn。
- visibility：clip_part_head.py:270-271 `else: visibility = torch.ones(B, NUM_PARTS)/NUM_PARTS` → 均匀 1/5。`pooled_feat = sum(visibility[:,k:k+1]*part_feats[k])`（line 274）= 5 个 part 的均匀平均。无除零（uniform 不经过 line 269 的归一化）。
- assign_loss：clip_part_head.py:277 `assign_loss = torch.tensor(0.0)`；line 278 `if self.training and target_heatmaps is not None and ...` → None 时整块跳过，assign_loss 恒为 0.0 tensor。`kp_data['assign_loss']` 仍存在（line 296），不是缺键。
- 无 NaN：clamp[-50,50]（line 199）+ log clamp[-30]（line 218）本就是 baseline 既有防护，None 路径反而更安全（无 pose_bias 注入，分数幅度更小）。

### Finding 2（CRITICAL CHECK）— LGPA 分支门控是否仍通过 → PASS（核心担忧已排除）
门控是 `scene_heatmaps is not None`（train line 602 / eval line 800），**不是 lgpa_hm**。
- POSE_LGPA_NO_POSE=True 时，scene_heatmaps 照常由 `_prepare_pose`(line 484-493) + `merge_person_heatmaps`(line 946) 正常构建，始终非 None。
- `lgpa_hm = None if self._lgpa_no_pose else scene_heatmaps`（line 605/802）只把传给 head 的热图置 None，**不动 scene_heatmaps 本身**。
- 关键旁证：pose_dropout_p 默认 0.0（line 129），exp336/exp337 都没设 POSE_DROPOUT_P → line 496 的热图清零路径永不触发 → scene_heatmaps 不会被意外置 None。
- 结论：整个 LGPA 分支照常运行，只是 head 内部 pose-bias/visibility/assign 关掉。flag **没有**误关整个 LGPA 分支。

### Finding 3 — 是否真单变量 vs exp336 → PASS
逐行 diff 两个 yml：唯一差异是 exp337 多一行 `POSE_LGPA_NO_POSE: True`（其余 backbone/LR/epoch/aug/PSG-stages=[]/DETACH=True/ASSIGN_WEIGHT=0.5/TEST_FEAT=equal_concat 完全一致）。代码侧 `_lgpa_no_pose` 仅在 line 605/802 两个 LGPA 调用点取值，default=False（defaults.py:223），不触碰任何其他实验路径。无交互。

### Finding 4 — eval 对比是否仍有效；equal_concat 是否正确组装 → PASS
- 与 exp336 同协议：同 ckpt，test.py 切 POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline)。两 config 均 POSE_LGPA_DETACH:True → LGPA 跑 detached 特征(line 604)，global 描述子==baseline backbone 全局特征，对照成立。
- `global` 路径：eval line 800-801 门控含 `pose_test_feat != 'global'`，=global 时 LGPA 测试分支跳过，gcn_feats 留 None，line 911 不进组装 → 只返回 backbone 全局特征 == baseline。✓
- equal_concat：line 914-917 `g_norm=normalize(global)`，`p_norm=[normalize(f) for f in gcn_feats]`，gcn_feats=lgpa_feats=[pooled, p1..p5]。no-pose 下 pooled=均匀 mean(p1..p5)，p1..p5 为各 part cross-attn 输出。各自 L2-norm 后 cat=[g, pooled, p1..p5]（7×C）。组装正确，pooled 即 uniform-mean。✓

### Finding 5 — lgpa_assign 应=0 且能正常 log → PASS
processor.py:1027-1033：`if lgpa_enabled and not ppa_enabled and kp_data is not None and 'assign_loss' in kp_data`。no-pose 下 assign_loss=tensor(0.0) → `loss = loss + 0.5*0.0`（loss 不变）→ `details['lgpa_assign']=0.0`，`.item()` 对 0.0 tensor 安全。对照 exp336 实测 `lgpa_assign: 6.977 @ e1`（monitor.md 已存），exp337 应稳定 0。sanity 成立且可观测。✓

### Finding 6 — 是否真能回答"CLIP-语义 vs pose-注入" → PASS（含口径说明，见 Low-1）
能。两端唯一变量是 pose 三注入(bias+assign+visibility)的开/关。equalcat−global ≈+1.7 → 增益来自 CLIP 文本语义本身；~0 → 靠 pose 注入。逻辑闭合。

## Low（不阻断）
- **Low-1（解释口径）**：no-pose 并非"无任何空间引导"，而是"CLIP 文本原型仍 cross-attend 全部 spatial tokens，只是无 pose 偏置"。即测的是 "CLIP-text parts **without pose guidance**"，不是 "CLIP-text 完全不看图"。design.md line 15-16 表述已基本准确（"纯 CLIP 文本部位原型 cross-attend tokens(无 pose)"），论文写作时建议明确这一点，避免读者误解为"零定位"。这是正确的消融语义，不是缺陷。
- **Low-2（次要）**：no-pose 下 `kp_weights=visibility.detach()` 为均匀 1/5（aux_data, line 299/298）。本实验 equal_concat 不消费该键（line 918 dict 分支不触发），MaxSim 也未开，故无影响。仅记录：若后续切 maxsim 系测试特征，需注意 no-pose 的 kp_weights 退化为均匀。

## 结论
审查通过。单变量隔离严格，None 分支数学/数值安全，门控核心担忧（flag 误关整支）已明确排除，eval 对照与 equal_concat 组装正确，lgpa_assign=0 sanity 成立且可观测。可进入 Codex 审查。
