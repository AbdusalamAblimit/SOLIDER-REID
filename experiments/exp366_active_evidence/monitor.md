# exp366 Active Evidence Acquisition — monitor

## cheap kill-switch（frozen SOLIDER, 零训练, 2026-06-28）

验 codex 范式 #1：query 主动获取第二证据（同 ID 不同 camera），policy（margin 选 hard query 花预算）能否接近 oracle 且 >> random。

| 数据集 | baseline | oracle-all (2nd) | policy (hard 20%) | random (20%) | frac | verdict |
|---|---|---|---|---|---|---|
| Market（exp260b 强）| 94.43 | +2.51 | **+0.31** | +0.48 | 0.12 | DEAD（policy<random）|
| Occluded-Duke（exp004_pfm⚠️）| 3.09⚠️ | +6.45 | **+1.29** | +1.31 | 0.20 | DEAD（policy≈random）|

⚠️ Occluded baseline 3.09 异常低（exp004_pfm ckpt 配 market config 的 FrozenExtractor 加载不匹配）；但 policy vs random 是相对比较、不依赖 baseline 绝对值，结论成立。

## ★VERDICT DEAD（两数据集坐实）

margin（top1-top2 检索不确定性）**不是好的"值得获取证据"预算信号**：Market policy+0.31<random+0.48；Occluded policy+1.29≈random+1.31。两数据集 policy 都 ≈/< random。

**★范式根本困难（诚实诊断）**：
1. 系统不知道哪个 query 的第二证据有用（要获取才知道=鸡生蛋）。
2. margin 小（难 query）给证据没用——第二证据也可能难（occluded query 另一张也 occluded）。
3. 任何"检索不确定性"policy（margin/entropy 同质）都救不了，因为不确定 ≠ 第二证据能救。

oracle headroom 真实（occluded +6.45 R1 大涨），但**没有 cheap policy 信号能逼近它**。主动获取证据要价值，需要预测"第二证据质量"，而那本身要先获取（鸡生蛋）。

## 决定

Active Evidence policy DEAD（cheap kill-switch 半小时验透，没浪费训练）。转 **Generative Index ReID（codex 范式 #2，6.5/10 真空白）**：gallery identity 离散 token，query 生成 code prefix，kill-switch=PQ code recall<95% 则杀。
