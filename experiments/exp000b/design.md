# 实验 exp000b: Baseline Seed 42 方差检查

## 动机
- 前序实验（exp007/007a/030a/030b）中观察到 2%+ mAP 波动，疑似训练方差而非方法贡献
- exp031（多种子验证）已安排在 4090 上运行，但结果需等待
- 在 3090 上先跑一个 baseline seed 42，快速获得 baseline 方差范围的初步估计
- 如果 baseline 本身在不同 seed 下就有 1-2% 波动，那 PSG 的 +1.7% 就不够显著

## 创新点 / 核心想法
- 本实验不验证任何创新点，是纯方差检测实验
- 核心假设：baseline 在不同随机种子下的 mAP 波动 < 1%（如果 > 1%，说明前序实验的增益不可信）

## 技术方案
- 与 exp000 完全相同的代码和配置
- 唯一区别：SOLVER.SEED 从 1234 改为 42
- 配置文件：`configs/occluded_duke/swin_tiny.yml`（原始 baseline config）
- 通过命令行 `SOLVER.SEED 42` 覆盖
- OUTPUT_DIR: `./log/occluded_duke/exp000b_baseline_seed42`

## 预期结果
- 如果 mAP 在 55.5-57.5% 范围内（±1%），说明 baseline 方差可控
- 如果 mAP 偏离 56.6% 超过 1.5%，说明训练方差确实很大，需要多种子实验才能得出可靠结论

## 对照组
- exp000: baseline seed 1234, mAP 56.6%, R1 66.5%
- 消融变量：仅 random seed（1234 → 42）
