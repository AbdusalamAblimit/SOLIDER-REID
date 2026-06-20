# 实验 exp348: de-occluded 对齐 + occluder repulsion

## 动机
exp347(de-occluded 对齐)把可见特征拉向 ID,但 GAP global 仍平均了遮挡区(中性,稀释)。exp348 加**显式遮挡排斥**:把遮挡区特征推离 ID 原型 → backbone 学"遮挡=非ID"→ GAP global 里遮挡贡献被压成非ID方向 → 更干净。

## 核心假设
**可见特征 → ID 原型(拉,exp347)+ 遮挡特征 → 远离 ID 原型(推,exp348)→ backbone 显式分离 可见=ID/遮挡=非ID → raw GAP global 更判别。**

## 技术方案
- PoseWeightedPool 加 `invert` → 加权**低可见(遮挡)**区 → 遮挡特征(参数-free)。
- forward:`POSE_CLIP_ID_OCC_REPEL` 开 → `occ_feat = pose_weighted_pool(feat, pose, invert=True)`;`repel = (norm(proj(occ_feat))·norm(txt_proto)).clamp(min=0).mean()`(只罚正相似度→推成中性,不推成相反);`clip_id_loss += w·repel`。
- = exp347 + `POSE_CLIP_ID_OCC_REPEL True` + `W 0.5`。

## 预期
exp348 global > exp347 ≥ exp341(59.8)。失败可能:pose 不准时遮挡区含人体部位 → 推走人体特征(伤);clamp(min=0) 缓解。

## 对照
exp347(只 de-occluded 对齐)vs exp348(+ occluder repulsion)。单变量 = POSE_CLIP_ID_OCC_REPEL。

## 审查重点
invert pool 正确(-vis softmax 加权遮挡区);repulsion loss 符号对(minimize 正相似度=推离);clamp(min=0) 防过推;占 backbone 梯度(占 featmaps[-1]);w=0.5 不压垮主 loss;参数-free;单变量 vs exp347。
