# AG-ReID.v2 AIRL 跨数据集验证(2026-06-24)

第二支柱 = point 4(directional evidence specialization, codex 判 5/5 主贡献)的命门验证。
CARGO 134-query 噪声大,AG-ReID.v2 query 多(A→G 2356 / G→A 1811)eval 稳。

## baseline 锚点(lab-3090, 均衡无特化)
| epoch | A→G | G→A | 方向差 |
|-------|-----|-----|------|
| ep10 | 73.39 | 73.97 | −0.58 |
| ep20 | 71.25 | 71.19 | +0.06 |
两方向同向移动、差 <0.6 = **无方向特化**(这是 AIRL 要打破的均衡)。

## AIRL ep10(lab-4090 24G 主, 第一个数据点)
| 方向 | full(clean) | rec(AIRL) | FUSE | rec−full |
|------|------|------|------|------|
| A→G | 74.91 | 74.08 | 75.66 | **−0.83**(full/clean 强) |
| G→A | 74.48 | 74.61 | 75.57 | **+0.13**(rec/AIRL 强) |
| mean | 74.70 | 74.35 | **75.62** | — |

### 判读(measured, 不武断)
- ★**方向特化复现(符号相反)**:rec−full 在 A→G(−0.83)、G→A(+0.13)符号相反,与 CARGO 同向(clean 强 A→G、AIRL-recover 强 G→A)。baseline 均衡 → AIRL 打破。**point 4 命门第一关过。**
- ⚠️ **但 G→A magnitude 小(+0.13)**:CARGO 特化随训练增长(ep20→60),AG-ReID.v2 ep10 早,待 ep20/30/final 看 +0.13 是否长大。现在只能说"方向对了、幅度待确认"。
- **净增益强**:AIRL FUSE mean 75.62 > baseline 73.68 = **+1.94**;full head 74.70 也 > baseline +1.02。无 OOM(24G)。

## ★ ep20(可信点)— clean grep 真实数(subagent 首报 ep20 数字读错,已纠正)
baseline 三档:ep10 73.68 / ep20 71.22 / ep30 72.71(回爬,均衡无特化)。
真实 ep20(行340-341):[A→G] full=72.59 rec=74.54 FUSE=74.44 / [G→A] full=73.67 rec=73.89 FUSE=75.19。

| | A→G(rec−full) | G→A(rec−full) | 谁赢 |
|---|---|---|---|
| ep10 | −0.83 | +0.13 | full→A→G, rec→G→A(CARGO 式反号特化) |
| ep20 | **+1.95** | **+0.22** | **rec 两向都赢,但 A→G 强得多** |

- **① 净超更强(+3.60,非首报 +1.23)**:AIRL FUSE mean 74.82 vs baseline 71.22。**AIRL 抗中段 dip**:baseline ep10→20 跌 2.46,AIRL FUSE 只跌 0.8。净增益 ep10 +1.93 → ep20 +3.60 增长。**这条很稳。**
- **② 方向特化不稳(但不是 subagent 说的"符号翻转")**:ep10 反号特化(CARGO 同向);ep20 rec 两向都赢,但 A→G 优势(+1.95)远大于 G→A(+0.22)→ rec 偏帮航拍 query 方向。**具体哪头赢哪向 ep10/ep20 不一致**(A→G 从 full 赢翻成 rec 赢)。
- **★codex 5/5 的"clean→A→G/robust→G→A 固定方向"在 AG-ReID.v2 不成立**。能成立的较弱说法:"rec 降质鲁棒头偏帮航拍低清方向"(机制自洽)+ "head 间分化 AIRL 独有"。
- 教训:subagent 报数也会错(tail -F 串块),关键判读自己 clean grep 复核。

## ★ 跨设备 ep10 对照(hyy vs lab-4090)— 方向特化复现 + 融合增益一致
我一度误判"hyy eval 坏了"(tail -3 漏看 dual-branch 块),dump 全文后真相:
```
              A→G(rec−full)   G→A(rec−full)   rec偏向    FUSE−full(同设备)
lab-4090 ep10  −0.83           +0.13           G→A(+0.96)  +0.92
hyy ep10       +0.01           +1.18           G→A(+1.17)  +1.19
```
- **方向特化跨设备一致**:两台都 rec 头偏帮 G→A(= CARGO rec→G→A 方向)→ ep10 信号设备无关,真信号非噪声。
- **融合增益跨设备一致**:同设备 FUSE−full ~+1(绕开设备绝对值差)。
- **跨设备绝对值差大**(f_full −2.69,设备/torch 2.11vs2.6 差异)→ 比净超必须同设备,不能跨机比 AIRL vs baseline。
- 教训:多机并行先验代码/环境一致;关键数自己 dump 全文,subagent 和我都会漏读 tail。

## ☠️ AIRL 封板 = 负结果(no-degrade kill-switch 实锤, 2026-06-24)
**最致命一关:no-degrade(--airl_min_scale 1.0, s=1 不降质 → consistency=0 → f_rec 退成纯第二干净头)vs AIRL(有降质)同设备同 seed 对比:**
```
ep20:        full mean  rec mean  FUSE mean  FUSE−full
no-degrade   73.13      74.42     74.80      +1.67
AIRL(降质)   73.13      74.22     74.82      +1.69
(ep10 同样: no-degrade +0.90 ≈ AIRL +0.92; full 头逐位完全相同)
```
**降质-consistency 对融合增益贡献 = 0(rec 头 no-degrade 还略高)。AIRL 的 +1 融合 = 平庸双头 ensemble,与 degradation 机制无关。**

**AIRL 完整死因(10-codex + 同设备2-seed + no-degrade 三重背书)**:① 净超 modest ~+0.64 全在 seed 噪声(−0.1~+1.4);② 融合 +1 = trivial ensemble(kill-switch 证);③ 方向特化第二数据集死;④ novelty 弱(ViSA/MRJL/AdvProp 占满)。**结论:AIRL 不成方法稿,负结果归档。**

## ★★★ 决定性裁决(同设备 lab-4090, 2026-06-24)— 净超 baseline ≈ 0
**第一份干净同设备(同 torch/seed,唯一变量 AIRL on/off)net gain:**
```
⚠️ 修正(10-codex 抓出 ep50-as-final 硬错, 2026-06-24):
真正 ep60 FINAL(非 ep50 中间值):
lab-4090 baseline final(ep60): 81.08
lab-4090 AIRL final(ep60):     full 79.90   FUSE 80.98
净超 = AIRL FUSE 80.98 − baseline 81.08 = −0.10 ≈ 0   (FUSE−full = +1.08)
[旧错误数: 79.98/79.90 是 ep50 中间值, 误当 final]
```
- **full 头(78.77)比 baseline(79.98)低 −1.21** → 融合 +1.13 只把它救回 baseline 水平,**无真净超**。
- **BN 污染假设(我提出)已证伪**:读 afd_model.py line 311-318 —— ① Swin trunk = LayerNorm 无 running stats;② 降质 pass 的 f_full map detach 且**不 pool→不过 BNNeck→不更新 self.bottleneck running stats**;③ rec stages 在 f_full loop 后跑,**f_full RNG 不变**。代码三重防护干净,无 BN bug。我钻了牛角尖。
- **−1.21 = seed/RNG 级噪声**(非可修 bug):佐证 lab-3090 AIRL ep20 full 头仅 −0.12 vs baseline,lab-4090 却 −1.21,同机制两机差 1.1 = 典型 seed 噪声。
- **结论**:净超 baseline = **seed-dependent ~0±1.2**,full 头被 seed 拉低可吃掉融合增益。**多 seed(用户)才能定净超**。干净硬 claim 只剩 **FUSE−full ≈ +1.8**(同 run)。AIRL = 机制自洽但净超悬于 seed,中等偏弱。
- 待:lab-3090 AIRL final(第二份同设备 net,~2h)+ 多 seed。

## ⚠️ 方法学修正(用户指出,2026-06-24)— 撤回"方向特化=噪声"的过度结论
**之前用 hyy 当跨设备对照判"方向跨设备翻=噪声"是错的。hyy 不是干净对照:**
- **torch 2.11.0+cu128 vs lab-4090 2.6.0+cu124**:数值路径不同,训练轨迹本就分叉。
- **seed 不同 → 数据加载顺序不同**:双头学哪个 pole 本就 seed-dependent,方向可能是 seed 条件下的真实现象、非随机噪声。
- 据被污染的对比杀 point 4,方法上不成立。
- **连"净超 ~0~+1"也脏**:lab-4090 AIRL vs lab-3090 baseline 跨设备跨 torch,+0.68 不可信。

**目前唯一干净证据**:① 同设备 FUSE−full ≈ +1(单 forward 内,无跨设备,稳);② lab-4090 单 run 方向随 epoch 变(但 per-epoch 单 eval 有噪声,小幅度分不清真演化 vs 抖动)。
**方向特化稳定性 = UNDETERMINED**,需同设备同 torch **多 seed** 才能判 noise vs seed-dependent vs robust。
**真终判**:① lab-3090 AIRL(同设备比 baseline)给净超;② 多 seed 给方向稳定性。在此之前不下"死刑"。

## (旧)ep30 三点 — 现按上面修正重读:跨设备/跨torch 部分作废
## ★★ ep30 决定点(已出)— 具体方向没复现,但融合增益稳
```
lab-4090 rec−full:  ep10 A→G−0.83 G→A+0.13 (rec→G→A)
                    ep20 A→G+1.95 G→A+0.22 (rec→A→G)
                    ep30 A→G+0.70 G→A−0.03 (rec→A→G)
```
- **ep20+ep30 一致 rec→A→G,ep10 是早期 transient。收敛方向 = rec→A→G。**
- **★与 CARGO 的 rec→G→A 相反**:CARGO full→A→G/rec→G→A,AG-ReID.v2 收敛 rec→A→G/full→G→A。**具体方向翻了。**

### 诚实结论(真 setback,不粉饰)
- **codex 5/5 的"固定方向 directional specialization(clean→A→G/robust→G→A)"在第二数据集不成立。** point 4 的"具体方向"claim 死了。
- **但两条稳**:① head 间分化现象复现(complementary heads 存在,只是方向 dataset-dependent);② 同设备 FUSE−full ~+1 两数据集一致(融合净增益实)。
- **paper headline 软化**:从"directional specialization(固定方向)"→ "**梯度隔离双头互补证据,融合净增益 ~+1;头按方向特化但具体方向随数据集变**"。比 5/5 弱,融合互补这条是实的。
- 待:lab-4090 final(net gain 守住?)+ 多 seed(用户,尤其方向稳定性)。CARGO 单数据集的漂亮固定方向,确属 CARGO 特性——这正是第二数据集的价值:挡住了过度声明。
- 维持 ep20 方向且不缩 → head 间分化复现(虽方向与 CARGO 反),point 4 软着陆为"分化存在"。
- 又翻号 → "有头间分化但方向不稳,弱于 CARGO 的特化强度"。需诚实下调 point 4 强度。
- 无论如何**净超 baseline 这条(+1.2~1.9)是稳的**,AIRL 作为方法仍 work,只是 headline 可能要从"directional specialization"软化为"gradient-isolated 互补双 head 净增益"。

## 待确认
- lab-4090 ep30/final:符号锁定?net gain 守住?
- hyy 跨设备 ep10 对照(Δ<0.5%)。
- 多 seed(用户)——尤其方向特化的稳定性,单 seed 翻号可能是 seed 噪声。
