# 实验 exp358: cross-PART (channel) shuffle kill-switch (LGPA-pose 地基二次确认)

## 动机
exp357(cross-image shuffle)= 59.8, vs exp353 真 pose 60.5 = **-0.7 弱掉点**。被 ReID 裁剪对齐混淆: 别人的 pose 在对齐裁剪里仍粗略定位头/躯干/腿, 所以乱 pose 还拿了大头(+2.2/+2.9)。correct pose 因果很弱(+0.7)。
exp358 disambiguate: **打乱 17 关键点通道(per-image)→ 破坏解剖部位身份, 但保留同图自己的空间 support**(关键点位置是本图的, 只是哪个点属于哪个部位被打乱)。无裁剪对齐 rescue。

## 方法
= exp353(un-detach LGPA no-CLIP 60.5)+ POSE_CHANNEL_SHUFFLE: 训练端, per-image 置换 scene/target heatmaps 的 17 通道(同一 per-image cperm 保持 scene/target 配对)。测试用真 pose。

## 判读(地基最终)
- exp358 ≈ 60.5(no drop)→ 解剖部位身份也不重要, 只是"某种空间池化结构 + 部位监督"在涨 → "Pose, Not Prompt"地基进一步塌(连部位结构都无所谓)。
- exp358 << 60.5(drop)→ 解剖部位结构重要(虽然 exp357 说精确图对应不重要)→ 故事可改 framing 为"部位结构(非精确 pose)是价值"。

## 对照
exp353 60.5 / exp357 59.8(cross-image)/ exp358(channel)。单变量 = POSE_CHANNEL_SHUFFLE。

## 实现
forward: POSE_SHUFFLE block 后加 per-image channel gather(argsort(rand(B,K))→ gather dim1)。defaults POSE_CHANNEL_SHUFFLE。config = exp353 + True。
