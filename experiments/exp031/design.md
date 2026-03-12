# 实验 exp031: 多种子验证 (Multi-Seed Validation)

## 动机
- exp030b 消融发现 global mAP 60.6%，远超预期的 58.3% (exp007)，暴露了严重的训练方差问题
- 不同模型类初始化消耗不同随机状态 → DropPath/数据增强序列不同 → 2%+ mAP 方差
- 四个实验 (exp007: 58.3%, exp007a: 59.5%, exp030a: 59.8%, exp030b: 60.6%) global mAP 无规律分布
- **没有多种子数据，所有 single-seed 对比的结论都不可靠**

## 核心假设
- 通过 3 个种子 × 3 个配置 = 9 次训练，可以估算每个方法的 mAP 均值和标准差
- 预期：标准差约 1.0-1.5%，足以解释 exp007→exp030b 的 2.3% 波动
- 如果 PSG+GCN (equal_concat) 均值显著高于 PSG-only (p<0.05)，则 GCN 贡献确认

## 技术方案

### 配置
1. **exp007 (PSG, 1.0x loss)** — `pose_backbone_psg.yml` — PSG baseline
2. **exp007a (PSG, 0.5x loss)** — `pose_psg_half_loss.yml` — loss scaling 效果
3. **exp030a (PSG+GCN, equal_concat)** — `pose_psg_gcn.yml` — 完整方法

### 种子
- 1234 (原始)、42、2024

### 测试模式
- exp007/007a: 单模式 (global)
- exp030a: global + equal_concat (两种模式)

### 脚本
- `scripts/run_multiseed_3090.sh`
- 预计 ~18 小时 (9 × ~2h)

## 预期结果

### 乐观场景 (PSG+GCN 真的有效)
| 方法 | mAP (mean±std) |
|------|----------------|
| PSG (1.0x) | 58.0±0.8% |
| PSG (0.5x) | 59.2±0.8% |
| PSG+GCN (global) | 59.5±0.8% |
| PSG+GCN (equal_concat) | 60.8±0.8% |

### 悲观场景 (大部分是方差)
| 方法 | mAP (mean±std) |
|------|----------------|
| PSG (1.0x) | 58.5±1.5% |
| PSG (0.5x) | 58.8±1.5% |
| PSG+GCN (global) | 59.0±1.5% |
| PSG+GCN (equal_concat) | 59.5±1.5% |

## 对照组
- 所有 9 个实验使用完全相同的代码版本，仅 config 和 seed 不同
- 消融变量：(1) loss scaling (2) GCN 特征 (3) 随机种子

## 论文用途
- Table: 主实验表报告 mean±std
- 证明方法的统计显著性（如果结论成立）
- 或诚实报告方差大于预期（学术诚信）
