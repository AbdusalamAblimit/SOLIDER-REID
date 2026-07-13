# exp371 CASD：Gate C 后 matched RGB-only student 设计

日期：2026-07-14

状态：**只完成设计；Gate C formal frozen oracle 未裁决前，不实现、不训练。**

## 一、前置条件与目标边界

只有 Gate C formal frozen oracle 的 routing、transfer、strongest-control、bootstrap 与 sensitivity 门禁全部通过，才允许实现本设计。Gate C 失败不能靠 student、temperature、queue 或 loss 权重补救。

本阶段只验证：

> 训练期 strict-LOO support 的逐 slot routing 与 support-vs-self relation advantage，能否迁移到测试时仅输入单张 RGB 的 student，并保留旧 LGPA 已验证增益的至少 80%。

`MVCD` 与 `MHSF` 正文仍未合法取得。即使内部数值全部过门，也只能先称“机制成立、外部新颖性未决”。

## 二、唯一变量与 matched 原则

所有 arm 必须共享：

- Swin-Tiny；
- batch size `64`，普通 `P=16,K=4` 主 batch；
- 完全相同的 RGB-only student 结构、参数量和初始化；
- data order、augmentation、optimizer、schedule 与 epoch 数；
- `global + pooled + 5 slots = 7×768 = 5376-D` 描述子；
- 完全相同的 train-only frozen teacher cache 与 donor index；
- ID/triplet 主损失；
- 辅助 loss 的正负 mass、总 mass 与 denominator。

各 arm 只改变训练期 support routing 或 relation transfer。测试统一使用单张 RGB、标准 cosine；禁止 pose、support、PID/camera context、GCN、matching、MaxSim、NFC 或 re-ranking。

## 三、Teacher 与 Student

Teacher 完全冻结并 stop-gradient：

```text
T_i = [g_i, p_i, t_i1, ..., t_i5]
```

只使用 SHA-bound train cache 中的：七个 raw blocks、target-only raw pose response、PID/CAM/path/content SHA 与 checkpoint/cache provenance。

Student forward 只能接受 RGB：

```text
z_i = StudentRGB(x_i)
```

student 使用所有 arm 共有的 generic RGB local receiver，从 spatial tokens 读取五个 slots；它不接 pose bias、pose assignment、CLIP text 或 canonical heatmap。该 receiver 是 matched 容器，不是创新贡献。B0 与全部 KD arm 结构必须逐参数一致。

## 四、Frozen donor index 与 strict LOO

普通 `K=4` batch 内 strict-3 coverage 只有 `31.48%`，不能直接兑现 Gate C。正式方案固定为 train-only、离线 donor index：

```text
same PID
donor != anchor
donor != relation endpoint
path/content disjoint
cam(donor) != cam(anchor)
exactly 3 donors selected by preregistered hash
```

donor index 不读取 feature、pose response、loss 或验证集；所有 arm 共享 donor paths 和 mask。正式实现前必须验证 anchor/PID/正 relation/负 relation coverage 均 `>=70%`。不足的 relation 只允许 CASD loss=`0`，严禁任何 fallback。

### Routing

POSE-RESP：

```text
w_mk = r_mk / sum_n r_nk
c_ik = sum_m w_mk * t_mk
```

POSE-SCALAR：

```text
q_m = sum_k r_mk
w_m = q_m / sum_n q_n
c_ik = sum_m w_m * t_mk
```

POSE-SCALAR 保留人体大小、总热图能量和 donor-level 质量，只删除 part-specific allocation。若 CASD 不优于它，不能称 pose-response organization。

## 五、Support-vs-self relation transfer

标准 equal-block cosine 定义：

```text
d_self(i,j) = d(T_i, T_j)
d_Q(i,j)    = d(C_i, T_j)
d_R(i,j)    = d(T_i, C_j)
d_SS(i,j)   = d(C_i, C_j)
```

主 target 预注册为对称单侧 support：

```text
d_support(i,j) = 0.5 * [d_Q(i,j) + d_R(i,j)]
```

关系增量：

```text
s_ij = +1 if y_i == y_j else -1
gain_ij = s_ij * [d_self(i,j) - d_support(i,j)]
```

- `R/FULL`：所有 eligible relations 匹配 `d_support`；
- `A/ADV`：只迁移 `gain_ij>0` 的 relation，权重为 detached gain。

每 query 先归一，正负 relation 各占 `0.5` mass，每 batch 辅助总 mass 固定为 `1`，首验 `lambda=1.0` 且禁止扫描。必须落盘实际 mass、eligible coverage、effective sample size、max weight、per-query supervision mass 及 auxiliary/main gradient norm。

## 六、routing × transfer 2×2

| Routing | FULL relation | ADV increment |
|---|---|---|
| PART-EQUAL | `PE-FULL / E+R` | `PE-ADV / E+A` |
| POSE-RESP | `PR-FULL / P+R / R0` | `CASD / P+A` |

`R0` 与 `PR-FULL` 是同一 arm，不重复训练。

三 seed 交互量：

```text
I = CASD - PR-FULL - PE-ADV + PE-FULL
```

必须满足 `mean(I)>=+0.3 mAP` 且 PID-grouped paired bootstrap 95% CI lower `>0`。否则 routing 和 ADV 只能被视为两个可替换组件，联合机制 claim 失败。

## 七、必做 matched controls

| Arm | 单变量定义 |
|---|---|
| `B0` | 同一 RGB student，无 KD |
| `KD0` | same-image LGPA relation KD |
| `ID0-G/M` | strict-LOO identity-global / identity-bag support + ADV |
| `P0-S` | donor feature slots derangement + ADV |
| `P0-R` | response slots derangement + ADV |
| `PS0` | POSE-SCALAR + ADV |
| `AI-ADV` | 与 CASD 完全相同，只把 current slot 放回 support |
| `MV-INCL` | anchor-inclusive full-feature KD |
| `LR-INCL` | current+support full-feature + full-relation KD |
| `LR-LOO` | strict-LOO full-feature + full-relation KD |
| `EXP123` | code-faithful continuous pair-delta full relation control，不做正增益筛选 |
| `CASD` | strict-LOO POSE-RESP + ADV |

`AI-ADV` 必须复用 CASD 的外部 donors、routing、transfer、mask 与 loss mass，只允许 current 是否进入 support 一处变化。若 `AI-ADV>=CASD`，strict LOO 只能称协议卫生，不能列为有效机制。

## 八、Q/R sensitivity

同一冻结 teacher geometry 同时报告：

- `Q-support / R-self`；
- `Q-self / R-support`；
- `Q-support / R-support`。

主 student 使用预注册的两个单侧结果平均，不从三种结果中挑最好。至少两种单侧结果同向，双 support 不得方向反转。若只有 query 侧 support 有效，共享 RGB encoder 的可迁移解释不成立。

## 九、RGB-only 逐元素不变证明

必须同时通过结构与运行时审计：

1. 导出 student-only checkpoint，state dict 不含 teacher/cache/raw-response/support index；
2. `StudentRGB.forward` 只接收 image tensor，不存在可选 pose/support 参数；
3. 推理 config 禁用 pose，dataloader 不加载 pose；
4. 删除或重命名 pose_data、teacher cache 与 donor index 后，完整 query/gallery 推理仍成功；
5. 同一 RGB 搭配 correct/random/zero/shuffled heatmap 与不同 support/PID metadata 时 descriptor 必须 `torch.equal`；
6. normal、pose-shuffled、support-deleted 三次完整 descriptor 文件 SHA256 完全一致；
7. PCA-768 后重复同一 SHA 审计。

任一项失败，均不得称 RGB-only 或 pose-free inference。

## 十、数值门禁

旧增益按 seed 冻结：

```text
G_old,s = mAP(exp336 equal-concat,s) - mAP(exp336 global,s)
G_new,s = mAP(CASD_s) - mAP(B0_s)
```

单 seed kill-switch：

```text
CASD - B0 >= +0.8 mAP
CASD - strongest_control >= +0.5 mAP
CASD - PS0 >= +0.3 mAP
CASD - AI-ADV > 0
```

三 seed final gate：

```text
每 seed G_new,s > 0
mean(G_new) >= +0.8 mAP
每 seed G_new,s >= 0.8 * G_old,s
mean(G_new) >= 0.8 * mean(G_old)
CASD - strongest_control 每 seed同向且 mean >= +0.5 mAP
对应 paired 95% CI lower > 0
2×2 interaction mean >= +0.3 mAP 且 CI lower > 0
```

## 十一、5376-D、768-D 与多数据集顺序

首轮全部 matched 5376-D，避免把 CASD 与压缩混为多变量。通过后，再对每个 arm/seed 使用同一 train-only PCA-768；query/gallery 不参与 fit。packed CASD 仍需 `CASD-B0>=+0.72 mAP`、相对 strongest control `>=+0.5 mAP`，并重做 RGB-only SHA 审计。

最低多数据集证据：

- Occluded-Duke：三 seed；
- Market1501：预注册三 seed；
- MSMT17：至少一 seed，若声称通用方法则升为三 seed。

每个数据集都必须重跑 matched B0 和 strongest controls，不能搬用 PSG 原文 baseline。没有 paired LGPA/global reference 的数据集不能计算 80% retention，必须先补 reference。

## 十二、停止规则

出现任一项立即 NO-GO：

- Gate C formal 未通过；
- 离线 exact-3 relation coverage `<70%`；
- CASD 不优于 POSE-SCALAR；
- 2×2 interaction 不过；
- anchor-inclusive 持平或更好；
- 增益只能由 special sampler 解释；
- RGB descriptor invariance 失败；
- matched 768-D 不保留旧增益的 80%；
- 三 seed 由单 seed 驱动；
- 多数据集方向反转。

禁止失败后扫描 temperature、queue、slot 数、donor 数、threshold 或 loss weight。
