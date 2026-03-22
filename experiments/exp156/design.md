# 实验 exp156: SPLADE — Learned Sparse Representation for Occluded ReID

## 动机

当前 equal_concat 产出 dense 1536-d vector。所有维度权重相同。
遮挡时，部分维度对应的 keypoint 不可靠，但 dense cosine 无法区分。

SPLADE (SIGIR 2021) 用 log(1+ReLU(x)) 产生 sparse representation：
- 大部分维度 = 0
- 少数维度有高激活
- 遮挡的 keypoint 天然激活更少维度
- matching 自动聚焦在双方都激活的维度上

**这从根本上改变了"特征长什么样"——从 dense 到 sparse。**

## 技术方案

在 GCN branch pooled feature (768-d) 之后加一个 sparse projection head：

```
gcn_feat (768-d) → Linear(768, 2048) → log(1 + ReLU(x)) → sparse_feat (2048-d)
```

- 大部分维度 = log(1+0) = 0
- 激活的维度 ∈ (0, ~10)
- test-time distance: dot product on sparse features (等价于只在双方都非零的维度上算)

训练时用 sparse feature 做 ID + triplet，加一个 sparsity regularizer：
```
L_sparse = mean(sparse_feat)  # 惩罚过多激活
```

## 与 MaxSim 的区别
- MaxSim: set-level matching on 17 keypoint tokens
- SPLADE: dimension-level sparsity on a single pooled vector
- 两者正交，可以组合

## 实现
- 新模块: model/modules/sparse_head.py (~30 行)
- 修改 skeleton_gcn.py forward: 返回 sparse feature
- 修改 test.py: sparse distance function
