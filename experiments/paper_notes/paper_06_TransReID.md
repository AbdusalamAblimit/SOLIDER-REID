# Paper 6: TransReID
**来源**: ICCV 2021
**仓库**: https://github.com/damo-cv/TransReID
**核心**: Transformer ReID 标准框架, JPM + SIE

## 可拆解模块清单

### M1: JPM (Jigsaw Patch Module)
- 文件: `model/make_model.py` L215-372
- 功能: patch token 分成4个区域, 各自过 last block + norm → 4个局部特征
- 测试时: 全局+4局部拼接 = 5×768 = 3840维
- **移植可行性**: 高(需适配Swin多阶段) | **显存**: ~0.5G

### M2: SIE (Side Information Embedding)
- 文件: `model/backbones/vit_pytorch.py` L316-331
- 功能: 每个camera/view学习一个embedding, 加到位置编码上
- sie_xishu=1.5 缩放因子
- **移植可行性**: 极高 | **显存**: <0.1G | **预期收益**: +0.5-1%

## 关键洞察
1. JPM 的均匀分割不如姿态引导的分割有效(我们的VPReID更优)
2. SIE 低成本高收益, 可直接加到 Swin-Tiny
3. TransReID 是 Part-Based ReID 的标准参照
