# 实验 exp095: DPF (Distributional Part Features)

## 动机
- **范式级创新，而非模块堆叠**
- 现有 ReID 方法（包括我们之前的 GCN 分支）在关键点位置做**单点采样**，得到一个确定性特征向量
- 但关键点代表的是一个**身体部位区域**，不是一个点
- 我们有原始 ViTPose 热图 (17, 64, 48)，它天然编码了每个身体部位的**空间概率分布**
- **范式转变**：将 body-part representation 从「特征点」升级为「特征分布」

## 创新点 / 核心想法
**重新定义 occluded ReID 的表示方式**：

- 现有：每个身体部位 = 一个特征向量（point estimate）
- 我们：每个身体部位 = 一个特征分布（mean μ + variance σ²）

**这给出三个范式级优势**：
1. **更好的特征提取**：热图加权空间池化 > 单点采样
   - 对关键点定位误差鲁棒
   - 低置信度关键点自动退化为全局平均（而非在错误位置采样）
2. **内建不确定性估计**：特征方差天然编码可靠性
   - 无遮挡部位 → 低方差（特征空间一致）
   - 遮挡部位 → 高方差（混合了人+遮挡物特征）
3. **概率匹配**：逆方差加权比较（precision-weighted matching）
   - 每对 (query, gallery) 有不同的比较权重
   - 高可靠区域主导匹配，不可靠区域自动降权

## 技术方案

### 数据流

```
Input: feat_map (B,768,12,4) + person0_heatmaps (B,17,96,32)
                                         ↓
                              Resize to (B,17,12,4)
                              ReLU + Normalize (sum=1 per channel)
                                         ↓
                          Heatmap-Weighted Spatial Pooling
                                    ↙          ↘
                    μ_k (B,17,768)              σ²_k (B,17,768)
                    body-part mean features     body-part feature variance
                          ↓                           ↓
                    GCN propagation             scalar precision per kp:
                          ↓                     p_k = 1/mean(σ²_k)
                    Enhanced μ_k                      ↓
                          ↓                     Precision-weighted
                    Precision-weighted pool      matching at test time
                          ↓
                    skeleton_feat (B,768) → BN → ID loss + triplet
```

### 修改文件
1. **`model/modules/skeleton_gcn.py`** — 核心改动
   - 新增 `_heatmap_pool_features()`: 热图加权空间池化 + 方差计算
   - `forward()` 中：当 DPF 启用时使用热图池化代替单点采样
   - 加权平均改为 precision-weighted (1/variance)
   - aux_data 中输出 kp_vars 供测试时使用

2. **`model/pose_backbone_model.py`** — 传递热图给 skeleton_head

3. **`config/defaults.py`** — 新增 POSE_DPF

4. **`scripts/eval_dpf.py`** — 新的测试评估脚本
   - 提取 per-keypoint μ 和 σ²
   - 逆方差加权 per-keypoint 余弦距离
   - 评估 mAP/R1

### 关键超参数
- 无新超参数！DPF 是一种计算范式改变，不需要调参
- variance_eps = 1e-6（数值稳定性）

## 预期结果
- **训练**：heatmap pooling 应该给出更好的 per-keypoint 特征 → GCN 分支更强
- **测试**：precision-weighted matching > confidence-weighted matching
- 预期 mAP 提升 +1~3%（如果假设成立）

## 对照组
- Baseline: exp030a (PSG+GCN+PAA, point sampling, confidence weighting)
- 消融变量：仅改变 keypoint feature extraction 方法（heatmap pooling + variance）
- 单变量对照：
  - DPF pooling + confidence weighting vs point sampling + confidence weighting
  - DPF pooling + precision weighting vs DPF pooling + confidence weighting

## 论文价值
这不是一个模块，而是一个**表示范式的改变**：
- 标题："Distributional Part Features: Probabilistic Body-Part Representation for Occluded Person Re-Identification"
- 贡献1：将 body-part representation 从 point estimate 升级为 distribution
- 贡献2：基于特征方差的内建不确定性估计
- 贡献3：Precision-weighted probabilistic matching
