# 自己造创新的方法论 + 我们的 re-framing(2026-06-24,用户纠偏"学创新不是抄模块")

## 用户纠偏(关键)
不要抄模块,去学人家**怎么把观察构造成创新**。目标发 B 类。

## B 类创新构造套路(deep-read codex 拆 5 篇: Base-Detail/Beyond-geometry/Find-Hidden-Modality-Divergence/Mix-Modality/Towards-Anytime-Retrieval)
共同套路:**先抓旧范式解释不了的失败 → 给失败起可测的名字 → 让机制看起来像新问题的直接解法。**
- 重定义 = "大家以为 X,其实是 Y"(新视角,不是新模块),这步最值钱、抄不来
- 机制几乎从重定义"自然推出",逻辑绑定要紧
- 关键消融要证"重定义对",不只是"机制涨点"
- reviewer 买的是视角,机制服务于故事

## ★ 我们自己的 re-framing(从我们独有数据长出来,不是抄)
**观察(团队独有):** CARGO 上 avg-pool 52.37 > token-MaxSim 45.19,差 7 分。极端跨视角(航拍↔地面 90°)下 late-interaction 反而有害。

**7 步打磨清单:**
1. 观察变稳定事实:普通 vs 极端视角分桶,证 MaxSim 只在极端视角输 avg
2. 旧假设:"局部最高相似 token 最可靠"(ColBERT/AlignedReID 信仰)
3. 反命题:"极端跨视角下最高局部相似被视角偶然模式劫持,不代表身份对应"
4. 命名+可测指标:"跨视角局部匹配不适定 / 局部证据劫持" + 匹配 token 互一致率、匹配熵、前景一致率、和错误检索相关性
5. 机制只解决这失败:可靠性加权 / 互为最近邻约束 / 视角条件软化聚合 / 稳定身份证据 vs 视角特有证据分开 —— 不是"更强模块"
6. 证据证重定义:MaxSim 错例热图、极端视角分桶、不适定指标 vs AP 相关、只在高不适定样本收益更大、随机权重/普通 softmax 替代不了
7. **主句:"大家以为极端跨视角缺更强局部匹配,其实局部最优匹配本身不可靠"**

## 为什么不撞红海
反 late-interaction 主流(ColBERT/AlignedReID 说局部匹配好,我们说极端跨视角下有害)= 新视角;证据是别人没有的我们自己数据。不是可见性/几何那堆。

## kill-switch(零训练,验第①④步)
frozen Swin(swin_fix256,67.33)提 token 特征,CARGO A↔G:
- ① 按视角极端度(altitude/scale 差)分桶,证 avg>MaxSim 集中在极端视角
- ④ 定义"不适定指标"= MaxSim 选中的匹配 token 对的互为最近邻率 / 匹配熵;证它和 per-query AP 误差强相关
成立 → re-framing 有据,这是我们自己的 B 类创新;不成立 → 观察不稳,回头。
复用 error_analysis_geom.py 基建。GPU 两卡现忙(Swin baseline),脚本先写,等空跑。
