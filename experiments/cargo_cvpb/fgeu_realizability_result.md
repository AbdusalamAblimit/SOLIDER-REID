# FGEU Realizability Kill-Switch — 结果 (DEAD 3/10, 撞 exp109 query-side 墙)

**日期**: 2026-06-25
**脚本**: `experiments/cargo_cvpb/cvpb_realizability_killswitch.py`
**机器**: lab-3090-d (frozen + numpy, 无 backward, 无训练)
**ckpt**: `log/occluded_posetrack/exp266b_best_b_op_s41_3090/transformer_120.pth` (vs-KPR posetrack 实验, in-domain mAP 78.5/R1 86.2)
**数据**: occluded_posetrack_reid (有 tracklet/帧信息), feature dim=8192 (equal_concat)
**log**: lab-3090 `/tmp/realiz_smoke.log`, feature cache `/tmp/realiz_posetrack_feats.npz`

---

## 一句话结论

**realizable 同 camera tracklet union 只拿到 16.3% 的 oracle 恢复 (远低于 40% 门槛), 且只比免费 k-reciprocal 强 ~2.3x。FGEU 撞 exp109 query-side 墙 — 大 headroom 只存在于部署不可得的 cross-camera 同 ID 证据里, realizable 的同机位多帧拿不到。判 DEAD 3/10。**

random 异 ID 控制完美砸毁 (-28.86), 证明对照设置可信。

---

## 关键事实: posetrack tracklet 结构 (生死前提)

- 命名 `pid_cVID_TIMESTAMP.jpg`, **同 pid + 同 cVID = 同 video/tracklet (一个机位连续帧) = realizable**。
- ⚠️ repo 的 posetrack loader 给每张图分配**逐图唯一 camid** (junk filter 是 no-op, KPR mot_inter_intra_video 协议)。所以缓存的 cam **不是** tracklet id, 必须从**文件名 c{VID}** 重建 tracklet。脚本已正确处理。
- query 1439 个 tracklet, 帧数分布 `{1帧: 297, 2帧: 1142}` — **每 tracklet 最多 2 帧** (realizable 预算 B=2)。
- cross-video oracle 稀薄: 仅 **147/1379** query pid 在 gallery 有不同 video_id 的同 ID 图 (posetrack gallery 大量是同 video 帧, 与 Occluded-Duke 的跨相机 gallery 性质不同)。

## SANITY

- frozen 全 query mAP = **78.43** (in-domain train log 报 78.5, 完全对上 → ckpt 加载正确, 用的是真训练 eval 特征)。
- 单帧 (每 tracklet 取第一帧) mAP = 78.04。
- 失败子集 = 单帧 AP bottom-50% = 720 tracklet (mean AP 56.6), 其中 568 个有 ≥1 realizable 同 video 额外帧。

---

## 核心生死对照 (n=568 realizable-failure tracklet; oracle 可用 90 个)

| arm | AP | dAP | 部署性 |
|-----|-----|-----|--------|
| baseline 单帧 | 57.73 | — | — |
| **A_realizable** 同 video union (MEAN) | 65.30 | **+7.57** | ✅ 部署可得 |
| A_realizable 同 video union (MAX) | 65.24 | +7.51 | ✅ |
| **B_oracle** cross-video gallery union | 76.58 | **+26.77** | ❌ 部署不可得 (exp109 上界) |
| **C_rand** 随机异 ID union | 28.87 | **-28.86** | 控制 — 必须砸毁 ✅ |
| C k-reciprocal (单帧免费 re-rank) | 61.00 | +3.27 | ✅ 免费, 无新证据 |

**apples-to-apples (90 个 oracle 可用 tracklet, 同一子集):**
- dAP realizable-MEAN = **+4.38** vs dAP oracle = **+26.77**
- **Recovery_realizable / Recovery_oracle = 16.3% (MEAN) / 15.6% (MAX)** ← 远低于 40% 门槛
- recovery-rate (dAP>+0.05): realizable **0.366**, oracle **0.911**, random **0.000**

**fragility gate (只融弱 support 失败 vs 全融):**
- fuse-ALL dAP = +7.57 (n=568)
- fuse-FRAGILE-only (bottom-50% support) dAP = +5.51 (n=45)
- non-fragile dAP = +3.24 (n=45)
- gate 方向对 (fragile > non-fragile), 但绝对量级小, 救不回 oracle gap。

---

## 诚实解读 (不粉饰)

1. **realizable 信号是真的, 但小**: 同机位第二帧 union 在宽失败集上 +7.57, 确实 >免费 k-reciprocal +3.27 (~2.3x)。所以"多帧 union 有效"这一点成立, 不是 0。
2. **但远够不到 headroom**: 在能直接和 oracle 比的 90 个 tracklet 上, realizable 只恢复 oracle 的 **16.3%**。大 headroom (oracle +26.77, 91% query 被救) 来自**跨 video / 跨机位的同 ID 证据**, 而那正是部署不可得的 (= exp109 墙)。
3. **为什么**: 同一机位连续两帧高度冗余 (同视角、同光照、相邻时刻, 遮挡常常一样), 提供的**新身份证据**很少; 真正能救遮挡 query 的是另一个机位/视角的同 ID 图, 但那需要身份标签才能取 (oracle)。
4. **posetrack 协议的额外不利**: 这个 benchmark 的 gallery 以同 video 帧为主, cross-video oracle 本身就稀薄 (147 pid), oracle 可用子集只有 90 — 比例估计有噪声, 但方向极其清楚 (16% vs 40% 门槛, recovery-rate 0.37 vs 0.91 差距巨大)。

---

## 鲁棒性复核 (bottom-30% 更严失败定义, reuse 缓存特征)

换更"真遮挡"的失败子集 (单帧 AP bottom-30%, n=336, base 仅 39.56) 重跑, 结论不变:

| arm | dAP | 比例 |
|-----|-----|------|
| A_realizable 同 video MEAN | +12.72 | — |
| B_oracle cross-video | +32.09 | — |
| C_rand 随机异 ID | -21.93 | 砸毁 ✅ |
| k-reciprocal | +1.97 | — |
| **Recovery_realizable/oracle (65 oracle 子集)** | — | **16.5%** |
| recovery-rate | realizable 0.467 / oracle 0.938 / random 0.000 | — |

ratio 从 bottom-50% 的 16.3% → bottom-30% 的 **16.5%**, 纹丝不动。注意越难的 query realizable 绝对增益越大 (+12.72, 比 k-recip 强 6.5x), 但 oracle 同步放大 (+32.09), **headroom-realizable 的差是结构性的, 不是阈值噪声**。比例 pin 在 ~16% = realizable 永远只能恢复约 1/6 的 cross-camera 证据 headroom。

---

## VERDICT

**DEAD 3/10 — exp109 query-side 墙。**

realizable 同 camera tracklet union 拿不到 ≥40% 的 oracle 恢复 (只有 16.3%)。FGEU 的大 headroom 是 cross-camera 同 ID 证据 (oracle, 部署不可得) 独有的, realizable 的同机位多帧复制不了。这正是 query-side oracle 变体, 撞 exp109 墙。

**不推进 FGEU 作为方法稿。** 若仍想要 multi-frame ReID 的角度, 残存的 +7.57 realizable 增益只能当一个 test-time trick / set-based ReID 的小补充, 撑不起 B 类方法稿 (且 set-based ReID + UFFM/AMC 已占满该位, 见 evidence_method_design.txt 撞车核查)。
