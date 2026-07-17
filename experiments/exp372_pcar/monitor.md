# exp372 PCAR 监控记录

## 2026-07-15 00:27 — Goal 启动

- 目标：先查新，再决定是否在 official CLIP-ReID 上做 pose-conditioned attention residual；
- 预注册 stop rule：若只剩 ordinary additive pose bias / module transplant，直接 NO-GO；
- 保护边界：不碰用户 decisions #99/#100，不恢复 IPER/PBSR/CASD，不用 Claude，不启动重复训练。

## 2026-07-15 00:30 — 本地与旧实验审计

- exp012 已做 unary pose attention bias；
- exp052 已做 pairwise keypoint-RPE；
- exp143 已做 skeleton-aware self-attention；
- exp354 设计过 CLIP ViT ownership bias，但因路由前提失败未训练；
- exp371 Gate B 证明 correct≈shuffled/canonical，实例 pose residual 燃料很弱；
- 官方 CLIP-ReID 的最小插入点和 global descriptor接口已确认，工程可行。

## 2026-07-15 00:41 — 三路查新收敛

- PeVL 已覆盖 pose mask 调制 CLIP visual attention；
- PAAB 已覆盖 pose mask进入 ViT attention logits并残差写回；
- MUVA 已覆盖 ReID 中动态 body-part mask逐层注入 CLIP ViT self-attention；
- PAFormer/KPR/ProFD 分别覆盖 pose-supervised attention、pose-conditioned encoder和 CLIP part decoder；
- canonical subtraction 可代数归约为普通 additive pose bias 的中心化。

当前判断：**停止。** 失败是新颖性门禁失败，不是实现、数值或性能失败。

## 最终状态

- 新颖性 Gate：FAIL；
- code/config diff：无；
- checkpoint/download：无；
- GPU 训练：无；
- 六臂性能结果：未运行，不能报告；
- 下一步：PCAR 封板，不做小变体。历史 LGPA `+0.82～0.85 mAP` 局部结构资产继续保留，但不能由 PCAR 重新包装为自有创新。
