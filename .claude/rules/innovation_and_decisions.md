# 创新方向与决策记录

> **阶段（2026-06-14）**：PRCV 2026 已投（PSG 主线，快照在 `experiments/prcv_2026_psg/`）。现处于**探索新创新点**阶段——本文件「值得推进的方向」+「默认路线」即新阶段主指引。PSG 等已投资产当 baseline，**不再在其上堆小模块**。新方向先写设计、红蓝队辩论、再开实验。

## 决策记录格式

追加到 `experiments/decisions.md`：

```markdown
### [{日期} {时间}] 决策 #{N}

**上下文**: {什么情况下做的决策}
**选项**:
  A. {方案 A 及预期}
  B. {方案 B 及预期}
**选择**: {A/B}
**理由**: {为什么}
**执行结果**: {后续补填}
```

## 红蓝队辩论制度

**每次重大决策前**，启动两个 Opus 子代理并行辩论：

- **红队**：为方案 A 辩护（技术可行性、论文价值、风险控制）
- **蓝队**：为方案 B 辩护（同维度论证，攻击对方弱点）

子代理 prompt：
```
你是 {红/蓝}队辩手。上下文：{上下文}。为 {方案 X} 辩护。
1. 从技术可行性、创新性、论文价值、风险、成本论证
2. 攻击对方方案的弱点和风险
3. 提供具体技术论据（实验数据、代码结构、论文先例）
4. 给出信心分数 1-10
```

记录格式：
```markdown
**红蓝队辩论**:
- 红队（方案 A）: {核心论点}，信心: {N}/10
- 蓝队（方案 B）: {核心论点}，信心: {N}/10
- 综合判断: {最终选择及理由}
```

## 创新方向管控

### 已确认有效的基础资产（PRCV 已投，固化为 baseline）
- PSG（Pose Spatial Gate）— PRCV 2026 主创新，已投
- 0.5x global loss
- exp030a-eq（PRCV 主线 3-seed）

### 已证伪的方向（不再作为主线）
- Visibility 各种变体（加权、pooling、小 head）
- Branch 内小修补（keypoint weighting, attention, GCN 小变体）
- Test-time trick 当主创新（NFC, re-ranking）
- Retrieval-side scorer 微变体
- Feature-level completion 小残差/bank/gate
- Skeleton attention bias / SASA
- Symmetry aggregation / SCFA

### 值得推进的方向
- **Target ambiguity / 主要人物归属**
- **Common visible support / pair comparability**
- **Reliability / uncertainty-aware matching**
- **从相邻领域迁移新的问题定义或机制**

### 默认路线
连续负结果 → 止损记录 → 读论文 → gap analysis → 新机制设计。
不在旧 branch 上堆模块，不做组合实验逃避创新。
