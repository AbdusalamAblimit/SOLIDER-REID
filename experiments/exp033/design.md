# 实验 exp033: Target Person Assignment

## 动机
- 当前 GCN/KPP 分支硬编码使用 person 0（按检测框面积降序排列的第一个人）
- 在多人图（占训练集 ~26.4%）中，person 0 不一定是 ReID 标注对应的目标人物
- Occluded-Duke 的每张图是围绕一个目标行人裁剪的，所以目标人应该大致在图片中心
- 如果 target assignment 错误，KPP/GCN 分支会从错误的人的关键点采样特征，引入噪声
- 在引入 visibility 之前必须先解决此问题，否则 visibility 语义会被污染

## 核心假设
- 目标人物应满足以下特征：
  1. 检测框中心靠近裁剪图中心
  2. 检测框面积较大（目标人通常是主体）
  3. 检测框大部分在裁剪图内（目标人不会大面积超出画面）
  4. 关键点置信度较高（目标人通常较完整可见）

## 技术方案
1. 对每个 person 计算 `targetness` 得分，加权组合四个因子：
   - `center_dist`: bbox 中心到 crop 中心的归一化距离（越小越好）
   - `area_ratio`: bbox 面积 / crop 面积（越大越好，但不应过大）
   - `containment`: bbox 在 crop 内的面积占比（越大越好）
   - `mean_score`: 17 个关键点的平均置信度（越大越好）
2. 选 targetness 最高的 person 作为 target
3. 计算 target_margin = best_score - second_best_score（衡量置信度）
4. 产出保存到 index.json（新增 target_person_idx, target_score, target_margin 字段）
5. 生成 200 张多人图可视化，标注所有 person 和 target

## 预期结果
- 绝大多数多人图（>=85%）的 target assignment 应该是正确的
- 如果 person 0 已经是正确的 target，则 exp034 可能不会改善性能，但代码更健壮

## 对照组
- 当前行为：始终使用 person 0（最大面积）
- 新行为：使用 targetness 得分最高的 person
