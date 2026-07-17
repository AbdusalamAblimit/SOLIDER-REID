# Codex Review — exp348

**Verdict**: approve
**Date**: 2026-06-20

## 结论
codex 审查通过。invert sign 正确(occluder=低可见→-vis→softmax高权重);repel loss minimize 遮挡特征与ID原型余弦相似度=推离;clamp(min=0)只罚正相似度=推成中性非相反;梯度经无参数池化流进 backbone;w=0.5 不压垮主 supcon;参数-free;单变量 vs exp347。
findings: M1(姿态噪声→遮挡区混真人,被 clamp+w 限幅,实验固有非bug); M2(pose dropout 把 heatmap 置零→occ_feat 退化整图均值——但 **exp348 pose_dropout_p=0,M2 不触发,moot**)。Verdict: approve。
