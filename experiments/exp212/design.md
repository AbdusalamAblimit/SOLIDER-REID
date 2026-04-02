# 实验 exp212: Small GCN+PAA+CE+OA-SD LR=0.0008 (Higher LR)

## 动机
- 当前所有 Small 实验使用 LR=0.0004 (vs Tiny 使用 0.0008)
- exp207 Base 表现不佳可能因 LR 太低
- 假设: Small 也可能受限于 LR=0.0004，更高的 LR 可能带来更好收敛

## 核心假设
LR=0.0008 (与 Tiny 相同) 在 Small 上可能带来更快/更好的收敛。

## 技术方案
- 与 exp206r 完全相同，仅 LR 从 0.0004 改为 0.0008

## 对照组
- exp206r (LR=0.0004): 70.6/82.6 (eq), 72.3/82.9 (maxsim)
