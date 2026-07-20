# exp405 独立审查记录

## 审查边界

用户明确禁止Claude，因此未调用Claude。本文件记录两路彼此独立的Codex审查；第三路近期论文/公开代码审计
完成后追加。当前审查只授权设计迭代，不授权训练。

## 代码路径审查

结论：未发现能把旧失败归因于xy/patch-layout/`ln_post/proj`的tensor bug；发现六项结构错配：

1. student evidence来自full-map GAP，不是slot-local pooling；
2. `source_feature.detach()`之外又有`hidden.detach()`，CLIP loss只能更新最后的evidence linear；
3. in-bounds pose valid被当成presence，导致五槽几乎全1；
4. PC-MBCLS只在最后4个block限制CLS-query，前20层global residual仍保留；
5. PCA code逐槽L2 normalization可能删除support/occlusion幅值；
6. SPK在执行前求slot mean，直接消除解剖通道身份。

额外边界：旧tight-crop part-name审计只用mask求bbox，没有抑制bbox内非目标像素；arms/upper-leg失败不能
单独证明CLIP连简单人体部位都不认识。

裁决：旧路线没有实现真正双编码、slot-local、可执行的CLIP–TAPF；继续新对象合理，但不得调SPK旧臂。

## 机制红队

初稿“same-ID donor + CLIP选槽 + TAPF搬运”分别接近MVI²P/LUPI、多视图KD、masked feature modeling、
SPT/TokenMix、KPR/ProFD与pose-aware part ReID，组合本身不够。

红队要求五项同时成立：

1. identity donor与anatomical slot的二维可辨识干预；
2. original/deleted/donor构成可观察counterfactual target；
3. donor-free student在同一TAPF路径预测state transition；
4. anchor field同时负责定位、抽取和回写，并最终descriptor可达；
5. teacher与student两级反事实顺序均成立。

Phase 0必须配齐same-ID/same-slot、same-ID/wrong-slot、wrong-ID/same-slot、generic、NULL、random-key，
并报告per-slot/PID-cluster结果。teacher端任何主偏序失败即NO-GO，不启动student或e120。

## 近期论文/官方代码审计

第三路审计核对了MVI²P、RegionCLIP、ProFD、KPR、MUVA，并把FLaN-Net、Composite-Attribute ReID、
DPM++、VLCDC和MVCD纳入危险近邻。结论是：局部CLIP soft teacher、pose part slot、同ID多视图teacher、
inference-free语言指导与patch transfer均已有直接先例；裸的CAVT组合不能作为机制创新。

第二轮机制红队只认可一个收窄对象：可观察original/deleted/donor target、identity×slot双干预、CLIP相对
pose-only的独立增量，以及同一operator上的donor-free transition。它要求增加MVI²P-full、pose-part、
attribute-relation、generic-transport和held-out PID可预测性门；宽claim判`1/3`，窄claim仅“问题+证据”
条件性`2/3`。

## 当前审查裁决

`ROOT-CAUSE SUPPORTED / BROAD MECHANISM NOVELTY FAIL / NARROW PHASE-0 CONDITIONAL GO /
STATIC CONTRACT AUTHORIZED / FORMAL CONFIG NO-CREATE / GPU NO-START`

## Phase 0 static v14最终代码盲审

审查材料严格限定为三份冻结源码、两份v14 canonical JSON和两份receipt，不读取设计辩护。v11--v13的
未授权结论保持历史有效；v14不覆盖这些产物，只作为修正后新执行。

最终计数：`BLOCKER=0 / HIGH=0 / MEDIUM=0 / LOW=0`。盲审独立复核：Torch RECORD `12,713`项、完整
site-packages `18,763`文件/`585,021,614` bytes及前后树摘要；两阶段`-I -S`与同字节执行；payload/receipt
异常清理；逐target-slot/PID latent与PID-disjoint probe；固定解剖列；极值/次正规数值；设备合同；联合
avalanche删除；camera-aware mAP/R1和稳定sample-key tie。两份payload均`56/56 PASS`且SHA256=
`6d073b72894c65236a53ee52d8e1d868e8492c60cc69ade139d57d0560130ee3`，两份receipt均可独立复算。

**裁决**：授权真实teacher measurement代码实现；尚不授权真实数据执行、GPU或训练。后续代码盲审必须聚焦
真实image/text encoder、pose同步、region isolation、donor配对、反事实臂和mAP/R1，不再把启动器外围强化扩展
成阻塞项。

## 真实teacher v10 Python3.8兼容修复三路盲审

审查材料固定为v10 contract、未改变的measurement/teacher/core及两份canonical结果。三路分别检查代码、
复现/once-only和最终统计/裁决，结论均为`BLOCKER=0 / HIGH=0`。唯一源码差分是把Python 3.9的
`ast.unparse()`换为Python 3.8可用的`ast.Attribute.attr / ast.Name.id`；对`backward / step / zero_grad`的
禁止调用检测语义不变。两份结果均`8/8 PASS`且byte-exact，SHA256=
`15ae43641d2e13afd487978033b61b8f83d1702fbfc74972d95a3f733230723c`，provenance与源码一致。

**裁决**：v10代码可用于用户指定的MMPOSE-ABU远端static复核；复核完成前不授权CUDA preflight或formal。
