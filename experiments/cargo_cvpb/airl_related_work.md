# AIRL Related Work 定位(撞车先例 + 区分,2026-06-23)

> 趁双分支训练时整理。codex 多轮 web search 挖出的相邻工作 + AIRL 如何切开。**核心:AIRL 不是又一个 resolution-invariant/dual-branch ReID,区别在「问题定义」「非对称退化」「固定先验融合」三轴。**

## 相邻先例(codex 找到的)

| 工作 | 做什么 | 与 AIRL 的关系 |
|------|--------|---------------|
| **VDT** (CVPR24, CARGO 原始) | aerial-ground = view discrepancy → view-decoupled transformer | 同 benchmark,但定义成**视角差异**;AIRL 定义成**像素预算可辨识性** |
| **GSAlign** (NeurIPS25) | geometric spatial alignment/warping | 空间对齐派;AIRL **拒绝对齐/hallucination**,只学可恢复证据 |
| **RAR** (arxiv 2207.13037) | resolution-adaptive metric,用 query resolution **动态选**子空间 | ★最近撞车点:RAR 是**动态路由**;AIRL 是**固定先验融合**(#3 已证动态硬路由失败 ≤+0.41,软固定融合 +1.46) |
| **MRJL** (2105.12684) | multi-resolution dual-branch fusion | dual-branch 撞车:MRJL 多分辨率分支;AIRL 是 clean/recover **证据分化**非多分辨率 |
| **DI-REID** (2004.04933) | degradation-invariance learning(学不变性) | ★关键区别:DI 追求**不变性**(抹掉退化);AIRL 追求**可恢复性上界**(承认退化、建模信息损失) |
| **CRReID** 系 | cross-resolution,LR↔HR 匹配 | 通用 LR/HR;AIRL 是 aerial-ground 特有的**非对称**(只 ground 高清需退化) |
| **AG-VPReID-Net** (2503.08121) | aerial-ground video,normalized appearance + multi-scale | video + 多尺度注意力;AIRL 是 image-level 证据建模 |

## AIRL 的三轴区分(钉死,避撞车)

### 轴 1:问题定义(最强区分)
现有 aerial-ground ReID 全在问"**如何跨视角对齐**"(VDT/GSAlign/ViSA)。AIRL 重新定义成"**航拍观测条件下身份信息何时物理上不可辨识**"——这是 observation-limited 视角,文献空白。kill-switch #1 证明了这是真实物理问题(强 Swin 上小桶仍塌 +13~19)。

### 轴 2:非对称 ground-degradation(机制区分)
DI-REID/CRReID 做**对称** resolution-invariance(LR/HR 都拉到不变空间)。AIRL **只退化 ground**(高清地面图 → aerial 像素预算),因为 aerial 已经是低预算、不该再退化。这个非对称性来自 aerial-ground 的物理结构,通用 cross-resolution 没有。

### 轴 3:固定先验融合 ≠ 动态路由(codex 关键修正)
RAR 用 query resolution **动态路由**选子空间。AIRL **不路由**——#3 oracle 实测硬路由(area/reliability 阈值)失败(≤+0.41),**软固定先验融合**(w=0.25,clean+recover 两证据头)反而 +1.46。所以 AIRL 的 claim 必须是 "fixed-prior fusion of clean/recover evidence heads under an observation-limited ceiling",**绝不吹成 query-budget routing**(那既撞 RAR 又与实现不符)。

## 一句话 positioning
> Unlike view-alignment (VDT/GSAlign) or resolution-invariance (DI-REID/CRReID) approaches, AIRL reframes aerial-ground ReID as **observation-limited identity recoverability**: it does not align views or erase degradation, but models the recoverable-evidence ceiling under an aerial pixel budget via an asymmetric ground-degradation consistency, and fuses a clean and a recover evidence head with a fixed prior — recovering the directional trade-off into a net gain without dynamic routing.

## 待补(双分支结果出来后)
- 主表:AIRL dualbranch vs baseline-Swin vs VDT/GSAlign/SeCap(强 backbone 同设置)。
- 消融:f_full only / f_rec only / fuse;kl vs feat(feat 变体在跑);w 敏感性(plateau)。
- 第二数据集:AG-ReID.v2(codex 强调跨数据集必需)。
