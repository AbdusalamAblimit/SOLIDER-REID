# AIRL 主结果表(2026-06-23,CARGO 定稿 + AG-ReID.v2 待补)

## 主表:CARGO Aerial↔Ground(mAP / R1)

| 方法 | backbone | mAP | R1 | 备注 |
|------|----------|-----|-----|------|
| VDT (CVPR24) | ViT | 42.76 | 48.12 | CARGO 原始 |
| SD-ReID (2025) | — | 46.44 | 53.12 | 生成派 |
| GSAlign (NeurIPS25) | — | 61.55 | 64.89 | 空间对齐 SOTA |
| **baseline-Swin**(我方,同 backbone) | SOLIDER-Swin | **60.84** | — | 强基线(团队资产) |
| **AIRL iso (fixed w=0.25)** | SOLIDER-Swin | **63.21** | — | **+2.37 over baseline** |
| **AIRL iso (rec 单头)** | SOLIDER-Swin | 63.38 | — | +2.54 |
| **AIRL iso (方向感知融合)** | SOLIDER-Swin | **~64.60** | — | **+3.76**,待 val-tuned w 实现 |

**关键点**:
- AIRL 主张 = **同 backbone 上的净增益**(+2.37~+3.76 over baseline-Swin),不是跨 backbone 刷 SOTA(那不干净)。
- 但顺带:AIRL mAP(63.21~64.60)**已超 GSAlign 的 61.55**(虽 backbone 不同,作上下文参考)。
- 强 backbone 上有净增益 = 机制内在价值(OVLI 正是栽在这关:强 backbone 上无增益)。

## 消融表:梯度隔离方式(CARGO,FUSE mean)

| 双分支变体 | FUSE−full(互补)| FUSE−baseline(净超)| 结论 |
|-----------|------|------|------|
| 全共享(无隔离) | +0.06 | ~0 | 两 head 坍缩,死 |
| **全 detach(正确)** | 收敛(final 方向特化)| **+2.37** | WORKING |
| trunk_recce(补 trunk 监督) | 方向特化 | **+2.77**(63.61) | ≈全detach(噪声内,非有害) |
| 全 backbone 最小 AIRL(无双分支) | — | 0(方向 trade-off) | 打平 baseline |

**消融主张(纠正后)**:① 必须**梯度隔离**(全共享坍缩,FUSE−full 仅 +0.06);② 必须放**梯度隔离末段分支**(全 backbone 只打平 baseline);③ trunk 监督 detach 与否无所谓(全detach 63.21 ≈ trunk_recce 63.61,134-query 噪声内)。核心设计 = "梯度隔离末段 recover 分支",细节鲁棒。

## 方向特化(机制可解释性,CARGO final)

| 方向 | full(干净) | rec(AIRL) | FUSE | 谁强 |
|------|-----------|----------|------|------|
| A→G | 63.42 | 62.13 | **64.56** | 融合(clean 主) |
| G→A | 59.45 | **64.64** | 61.85 | rec 单头(AIRL 降质鲁棒帮 ground-query) |

**机制故事**:降质一致性让 rec 头对低像素鲁棒 → 强在 ground-query 方向(G→A);clean 头强在 A→G。方向感知融合(view 测试已知)取两方向最优 → +3.76。

## 第二支柱(待补)

| benchmark | baseline | AIRL | 状态 |
|-----------|----------|------|------|
| AG-ReID.v2 A↔G | 待跑 | 待跑 | staged(lab-3090 顺序,iso2 完即启) |

SOTA 参考:AG-ReID.v2 上 GSAlign 81.38 / SD-ReID 81.01(headroom 小,关键看 AIRL 是否仍净超 baseline)。

## 诚实定位(codex 联网核实后更新,2026-06-24)
- 不再是"无惊艳单点":codex(gpt-5.5 联网 132k)查完 **方向特化双 head(directional evidence specialization)= 5/5 无先例**,最值得当主贡献。
- 四点评分:① observation-limited recoverability 4/5 / ② 非对称降质 3/5(别单独包装,DI-ReID 先例多)/ ③ 梯度隔离双 head 4/5(绑 ④ 讲)/ ④ **方向特化 5/5**。详见 codex_novelty_airl.md。
- 诚实补充:④ 是**经验发现**(非深机制),direction-aware fusion 本身简单;撑 B 类靠"新发现 + 框架 + 两数据集 + 干净消融"。
- 撑 B 类:**方向特化 5/5 + 强 backbone 净增益 +2.37~3.76 + 两数据集 + 干净消融 + 多 seed(用户)**。
- headline:**Observation-Limited Identity Recoverability for AG-ReID** + 副 **Directional Evidence Specialization via Asymmetric Degradation Learning**。
- ★命门验证结果(AG-ReID.v2, 2026-06-24, 详见 airl_agreidv2_result.md):
  - **directional specialization 的"固定方向"不 robust**:CARGO rec→G→A,AG-ReID.v2 收敛 rec→A→G(反向)。codex 5/5 的固定方向 claim 死了。
  - **真正站得住**:① 同设备 FUSE−full ≈ +1 两数据集一致(互补融合实);② head 间分化现象复现(AIRL 独有),但方向 dataset-dependent。
  - **净超数字注意**:跨设备 f_full 飘 ±1.5,不能拿 lab-4090 FUSE 比 lab-3090 baseline(灌水成 +1.9~3.6),干净指标是同设备 FUSE−full ≈ +1。lab-3090 同设备 AIRL 对照在跑。
- **修正后 headline**:从"directional specialization(固定方向)"软化为 **"梯度隔离互补双头 + 方向感知融合,同设备净增益 ~+1,两 aerial-ground 数据集复现;头按方向特化但具体方向随数据集 degradation 结构变"**。比 5/5 弱,互补融合这条实、可消融、二数据集。
