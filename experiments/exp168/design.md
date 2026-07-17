# 实验 exp168: 17-Token Per-Token + PLBOA

## 动机
- exp166 使用 6 body-group tokens 取得 63.1/73.9
- exp161c 使用 17 individual keypoint tokens (无 per-token, 无 PLBOA) 取得 58.2/67.3 — 与 6-part (58.7) 持平
- 但 17-token + per-token classification 是全新组合：每个关键点独立分类 + PLBOA
- 17 个 token 提供更细粒度的身体分解

## 核心假设
17 个独立的 keypoint-level tokens 比 6 个 body-group tokens 提供更精细的判别力

## 技术方案
- POSE_STR_NUM_PARTS: 17（每个 COCO 关键点一个 query）
- 每个 token 独立 CE + triplet（17 个 CE 损失 + 17 个 triplet 损失）
- Test: pooled feature（confidence-weighted mean of 17 tokens → 768-d + global 768-d = 1536-d）
- 注意：test 使用 simple mean pooling（17 parts 的 confidence-weighted pooling 代码路径需确认）

## 预期结果
- 如果更细粒度更好：mAP/R1 > exp166 (63.1/73.9)
- 如果 6 groups 是最优：mAP/R1 ≈ exp166（说明 6-group 分组是合适的抽象层次）
- 最可能失败原因：17 个冗余 tokens（左右眼、鼻子太相似）导致 gradient 稀释

## 对照组
- exp166 (6-part per-token + PLBOA): 63.1/73.9
- 消融变量：仅改 POSE_STR_NUM_PARTS: 6 → 17
