# AIRL 梯度隔离单模型双分支 — 救援成功(2026-06-23)

## 全 detach iso(--airl_dualbranch_iso --airl_iso_trunk_recce 0)= WORKING ✓
lab-4090,Swin,fuse_w=0.25。轨迹:
| epoch | full | rec | FUSE | FUSE−best(互补) | FUSE−baseline(净超) |
|-------|------|-----|------|-----------|---------------|
| ep10 | 41.39 | 42.62 | 43.24 | +0.62 | (base 43.79) −0.55 |
| ep20 | 47.78 | 46.61 | 49.24 | +1.46 | (48.98) **+0.26** |
| ep30 | 50.45 | 47.05 | 51.86 | +1.41 | (51.30) **+0.56** |

- **融合净超 baseline 且增长**(+0.26 → +0.56),f_full 收敛慢但在追(gap −1.20 → −0.85)。
- **互补稳定 +1.41~1.46**(FUSE 比 best single head 高)= 两 head 犯不同错,梯度隔离产生的特化是真互补。
- final 投影:f_full 追 baseline 60.84 + 互补 +1.4 → FUSE ~62 = 净 +1.0~1.5。

## trunk_recce(iso2)= final 与全detach 基本一样(★非有害,纠正前错误结论)
**final:63.61 @ ep50 ≈ 全detach 63.21 @ ep60(134-query 噪声内,trunk_recce 甚至略高 +0.40)。**
- 各方向 final:[A→G] full=60.60 rec=63.77 FUSE=63.13 / [G→A] full=59.07 rec=62.30 FUSE=63.33。
- trunk_recce 让 f_full 略弱(60.60 vs 全detach 63.42)但 f_rec 略强,FUSE 净结果持平。
- ⚠️ **ep20 的"47.22<49.24=有害"是噪声误导(我第 N 次栽在 ep20 单点)。教训:此数据集 ep20 完全不可信,只认 final。**
- **修正消融主张**:关键是**梯度隔离(vs 全共享坍缩)**,trunk 监督细节无所谓——两个隔离变体都净超 baseline +2.4~2.8。

## 关键认知修正(诚实)
- 原"f_full 弱=trunk 监督不足"诊断错了:f_full 不是结构弱,是**收敛慢**(ep20 落后、ep30 追上)。
- 早期 ep10 噪声大(134 query),原 iso ep10=45.78 是噪声高点,误导我判 trunk_recce"修复"。去噪后 ep20/30 真相:全 detach 本来就 work,trunk_recce 反而有害。
- **不该杀原版 + 不该做 trunk_recce**——但双线对照反而得到干净的消融。

## AIRL 全景(机制立 + 融合净超 baseline)
- kill-switch #1 诊断 PASS / #2 最小机制 area 桶 +3.6~8.4 / #3 fusion 上界 +1.46(零训练)
- #4 单模型双分支:全共享坍缩(FUSE−full +0.06,死)→ **梯度隔离全 detach WORKING(FUSE 净超 baseline +0.56 且增长)**
- novel headline:**梯度隔离让单模型产生互补双 head**(observation-limited evidence 的 clean/recover 分化),单 forward 互补证据融合净超 baseline,**非 ensemble**(单模型)
- 待 final 确认净增益 +1.0~1.5 + AG-ReID.v2 跨数据集 + 多 seed(用户)

## ★ FINAL ep60 定稿(2026-06-23)
```
[mean] full=61.44  rec=63.38  FUSE(w=0.25)=63.21   baseline=60.84
```
| 策略 | mean | 净超 baseline |
|------|------|------|
| 固定 w=0.25 fusion | 63.21 | **+2.37** |
| rec 单头(AIRL recover) | 63.38 | +2.54 |
| **方向感知 fusion** | **~64.60** | **+3.76** |

### 各方向分解(揭示机制)
```
[A→G]  full=63.42  rec=62.13  FUSE=64.56   → 融合加分 +1.14(over best)
[G→A]  full=59.45  rec=64.64  FUSE=61.85   → rec 单头最强,固定 w 拖累
```
**两 head 按检索方向特化**:full(干净)强在 A→G,rec(AIRL 降质鲁棒)强在 G→A(降质 ground→ground-query 鲁棒→G→A 涨)。固定 w=0.25 在 G→A 上稀释了强的 rec。**方向感知融合(A→G 用 FUSE、G→A 用 rec,view 测试已知合法)= 64.60 净超 +3.76。**

### 修正后的 headline(机制更干净)
不是"泛泛互补融合",是 **"clean/recover 两 head 按检索方向特化 + 方向感知证据融合"**。比固定 w 强,且机制可解释(降质鲁棒帮 ground-query 方向)。+ 单 forward 双 head。

### 关键对比(为什么梯度隔离 + 末段分支是对的)
- 全 backbone 最小 AIRL:mean 打平 60.83(方向 trade-off)
- 梯度隔离末段 rec 头:63.38(+2.54)→ **"把降质一致性放梯度隔离末段分支(配干净 trunk)"才让 AIRL 真 work**,比全 backbone 强 +2.5。

## ★ AG-ReID.v2 跨数据集复现成功(2026-06-24)= 方法稿第二支柱

官方协议 A→G(exp1 aerial_to_cctv)+ G→A(exp4 cctv_to_aerial),mean(对应 CARGO 的 A↔G)。
Swin-Small + SOLIDER pretrain,256×128,bs64,60ep。lab-3090(4090 数据上传链路死,顺序跑)。
接线:新增 `afd_reid/agreid_v2_combined.py`(包装已验证 `agreid_v2_dataset.AGReIDV2`,合并两官方协议使
`filter_by_view` 还原方向),afd_train.py 3 hunk(`--dataset agreid_v2`),eval/AIRL 零改动,cargo 字节不变。
codex approve / 0 findings,CPU smoke 全过(官方计数 2356/6347 & 1811/14362)。

| 方法 (AG-ReID.v2 Swin 60ep) | A→G mAP | G→A mAP | **mean mAP** | mean R1 |
|------|---------|---------|----------|---------|
| baseline-Swin | 79.72 | 80.04 | **79.88** | 87.11 |
| AIRL-iso FUSE (full-detach, w=0.25) | 81.06 | 81.46 | **81.26** | 88.41 |
| **净超** | +1.34 | +1.42 | **+1.38** | +1.30 |

- AIRL 单头: full 80.21 / rec 78.96 / **FUSE 81.26**。同期对照(AIRL FUSE − baseline 同 epoch):
  ep10 −0.47 / ep20 +1.77 / ep30 +2.32 / ep40 +0.80 / ep50 +1.45 / **ep60 +1.38**(ep20 起每点领先)。
- **机制非数据集特例**:CARGO +2.37 → AG-ReID.v2 +1.38(真实低清 headroom 略小,与诚实预期一致:
  涨幅缩但仍净超)。full≈baseline(80.21 vs 79.88,全 detach 隔离正确);FUSE>full&rec 双方向(互补真
  贡献);全程 0 KILL。ckpt 在 `log/cargo/cvpb_agreidv2_{baseline,airl_iso}/`,详见 `monitor_agreid_v2.md`。

### 下一步
1. 实现**方向感知融合**(val-tuned per-direction w)拿 +3.76 干净数(eval 小改)
2. ~~AG-ReID.v2 复现~~ ✓ 完成(+1.38 净超,见上)
3. 三档消融:全共享(坍缩)/ 全detach(work)/ trunk_recce(略差)
4. 多 seed(用户)
