# 2026-04-15 PRCV 重审（PSG 主线版）

## 0. 当前结论

这轮重审后的主判断改为：

1. **PRCV 不必强行重开全新大路线**
2. **论文主创新点优先收敛到 PSG**
3. **2-stage PSG 可以作为当前最终版本，但不必在主叙事里与 1-stage 正面对打；只需在消融中交代选择依据**
4. `GCN` 虽然也属于 pose 信息利用，但应统一写成 **structural pose branch**；`LGPA-D / OA-SD / MaxSim / POT / flip-test` 仍作为 supporting assets，不再抢主创新位置

用户已明确说明：**所有实验都可以重跑**。  
因此当前文档不再把旧消融视为不可动的最终证据，而是把它们当作“提示我们下一轮该怎么重跑”的先验。

---

## 1. 只看当前实验，什么是最硬的事实

### 1.1 可以稳定 claim 的

1. `PSG` 本身是稳定有效的  
   - `exp007`: `58.3 / 67.9`
   - 3-seed mean: `57.83 / 67.13`
   - 早期 phase 已经清楚证明：`backbone injection > post-hoc pooling`

2. `PSG` 在不同 backbone / 数据集上有迁移性  
   - 文档里已有 Swin-Small / Market 的正信号
   - 这使它比“单一 recipe 改动”更像方法点

3. 当前最强系统 `exp255` 使用的是 **2-stage PSG**
   - `exp255`: Small + GCN512 + 2-stage PSG = `73.2 / 83.3`
   - 当前系统最优配置里，`2-stage PSG` 是 final recipe 的一部分，但不需要在题目和摘要里单独拔成主术语

4. `GCN` 应该明确写进最终方法，但不能与 `PSG` 并列成两个主创新
   - `GCN` 的职责是提供 explicit skeleton relational evidence
   - 它更适合作为 `PSG` 支撑下的结构分支，而不是单独的问题级创新

5. `exp255 vs exp255b` 给出最强的 multi-stage 证据
   - `exp255`: GCN512 + **2-stage PSG** = `73.2 / 83.3`
   - `exp255b`: GCN512 + **1-stage PSG** = `71.5 / 81.9`
   - 差异：`+1.7 / +1.4`
   - 这说明：**在高容量结构分支下，最终采用 `2-stage PSG` 是有依据的**

### 1.2 不能硬讲的

1. **不能说 multi-stage PSG 在所有设置下都优于 single-stage**
   - `exp009`: Stage2+3 vs Stage3-only 基本持平
   - `exp251/253`: Tiny `LGPA-D+GCN` scaffold 上，2-stage / 3-stage 最终未超过 1-stage `exp246b`

2. **不能说 LGPA-D 是更强主创新**
   - 它有结果，但更像 detached semantic branch
   - 可以当 strong module，不宜当主问题定义

3. **不能把 MaxSim / POT / flip 当训练端主贡献**
   - 这几个都偏 test-time 或 supporting evaluation

---

## 2. 现在最合理的 PRCV 写法

## 主创新点

### PSG（Pose Spatial Gate）

这是最稳的主创新点，理由：

1. 机制清楚  
   - 在 backbone 中间层注入 pose-conditioned spatial gate
   - 改的是特征形成过程，而不是事后 part pooling

2. 证据链完整  
   - baseline → PSG
   - 多种 PSG 变体对照
   - 跨数据集 / 跨 backbone 正信号

3. 创新表达自然  
   - 从 post-hoc pose usage 转向 in-backbone pose-conditioned representation learning

## 最终方法版本

### Two-Stage Instantiation of PSG

这里更稳的写法不是把它写成新的主术语，而是：

> **我们提出的是 PSG；最终实现采用 two-stage instantiation。**
> 消融再补充说明：该配置在 stronger semantic-structural scaffold 中更适合支撑高容量结构分支。

---

## 3. 推荐论文叙事

### 一句话版本

> 现有 pose-guided ReID 大多在 feature 形成后利用 pose，而我们提出在 backbone 内部进行 pose-guided spatial gating。  
> 最终系统采用 two-stage PSG，并结合 structural pose branch，在 stronger semantic-structural pipeline 中更稳定地释放结构证据。

### 叙事层级

1. **Level 1: PSG 基础机制**
   - 解决“pose 只被 post-hoc 使用”的问题
   - 证明 backbone-level injection 有效

2. **Level 2: PSG 的最终实例化**
   - 最终实现采用 `2-stage PSG`
   - 为什么不是 `1-stage`，放到消融里回答

3. **Level 3: 完整系统**
   - `GCN / LGPA-D / OA-SD / MaxSim`
   - 其中 `GCN` 是 structural pose branch，`LGPA-D` 是 semantic branch
   - 它们作为建立在 `PSG` 之上的完整系统资产

---

## 4. 为什么现在不应该再主打“新问题定义”

不是因为 `exp109` 不重要。  
相反，`exp109` 仍然是很强的问题证据：`single-image support incomplete` 没被推翻。

但如果目标是 **4 月 30 日先交一篇 PRCV**，当前更务实的选择是：

1. 先把 **PSG** 这条已经站稳的主线写出来
2. 把 `exp109` 留作 motivation / future direction / discussion
3. 不在最后两周里强行把论文重写成全新问题范式

也就是说：

- `exp109` 继续作为理论动机资产保留
- 但本轮 PRCV 主线优先回到 **PSG**

---

## 5. 因为实验可重跑，所以必须重新设计的 PSG 消融

当前最大问题不是“没有结果”，而是 **multi-stage PSG 的证据还不够干净**。

旧结果里存在三类变量混杂：

1. stage 数变化
2. branch 容量变化（如 GCN 256 → 512）
3. scaffold 变化（有无 LGPA-D / GCN / PAA）

所以必须重新做干净矩阵。

### 5.1 基础 PSG 消融

在尽量纯净的 PSG scaffold 上重跑：

1. no PSG
2. 1-stage PSG（Stage 3）
3. 2-stage PSG（Stage 2+3）
4. 3-stage PSG（Stage 1+2+3）

目标：回答 **hierarchical PSG 在纯 backbone setting 下到底是不是稳定正增益**。

### 5.2 结构分支依赖性消融

固定 branch 容量，单改 PSG stages：

1. GCN256 + 1-stage
2. GCN256 + 2-stage
3. GCN512 + 1-stage
4. GCN512 + 2-stage

目标：把 `exp255 vs exp255b` 的发现从“经验观察”升级成正式结论：

> **2-stage PSG 是否是高容量 GCN branch 的必要条件？**

### 5.3 Semantic branch 依赖性消融

如果时间允许，再补：

1. LGPA-only + 1-stage PSG
2. LGPA-only + 2-stage PSG
3. LGPA+GCN + 1-stage PSG
4. LGPA+GCN + 2-stage PSG

目标：回答 2-stage PSG 的收益是更偏 structural branch、还是对 semantic branch 也有帮助。

---

## 6. 当前最推荐的 PRCV 落地方案

### 论文主标题方向

- `Pose Spatial Gate for Occluded Person Re-Identification`
- `Hierarchical Pose Spatial Gating for Occluded Person Re-Identification`
- `Scaling Pose Spatial Gates for Semantic-Structural Occluded Re-Identification`

### 贡献点写法建议

1. 提出 `PSG`，在 backbone 内进行 pose-guided spatial gating，而非 post-hoc part filtering
2. 提出 `hierarchical / 2-stage PSG` 作为 scalable PSG 版本，并证明其对高容量结构分支是关键条件
3. 在 semantic-structural occluded ReID system 上给出完整验证，其中 `GCN` 提供 explicit structural pose evidence，最终 Small 结果达到当前项目最佳训练端配置之一

### 当前文档口径

从现在开始统一采用：

- `PSG` = 主创新
- `2-stage PSG` = 最终版本 / scalable extension
- `LGPA-D / GCN / OA-SD / MaxSim` = system assets / supporting modules

---

## 7. 最终结论

这轮 PRCV 重审后的方向不是：

- 再硬切到全新问题定义
- 再把 LGPA 或 test-time matching 写成主创新

而是：

> **回到 PSG 主线，把 multi-stage PSG 收紧成“可重跑、可证伪、可讲清”的扩展版本。**

在用户允许重跑所有实验的前提下，下一步最重要的不是继续想故事，
而是 **把 PSG / 2-stage PSG 的干净消融重新设计并补齐**。
