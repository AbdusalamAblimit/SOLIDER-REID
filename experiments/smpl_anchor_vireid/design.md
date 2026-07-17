# 实验 SMPL-Anchor VI-ReID (v0 preliminary) — SMPL 几何当模态无关锚

> 状态: v0 草案(远程宕机期间写, 待 kill-switch + baseline 复现后refine)。文档先行铁律。

## 动机
- VI-ReID(可见光-红外行人ReID)核心 confound = **模态 gap**: RGB 有颜色/纹理, IR 有热强度, 外观差异巨大。
- 现有去模态-gap 手段: 共享-特有特征解耦、频率域、中间模态(灰图/X-modality/MTRL)、**轮廓/体形(MSO2021/GSMEN2023/Contour-MMN2024 已占满)**。
- ★洞察: **SMPL 3D 几何(姿态参数 + 投影关节 + mesh)是从人体结构恢复的, 与外观(颜色/热强度)无关 → 天然模态不变**。比 2D 轮廓更"光谱无关"(纯几何, RGB/IR 投影一致, 不受 IR 边缘检测退化影响)。
- ★空白: 文献里没人用 **SMPL 3D 几何**当 VI-ReID 模态锚(轮廓/体形有人做, 3D mesh 几何是空白)。团队有 SMPL 基建(ROMP/SMPLer-X)。
- 区别于团队 exp333(SMPL-β 随机判负): 这里用**几何(姿态/关节)做结构对齐锚**, 不是 β 当身份特征。绕开 exp333 死因。

## 核心假设
SMPL 3D 几何当**训练期特权(LUPI)模态共享锚**, 把同人的 RGB/IR 特征拉到同一几何上 → 缩小模态 gap → VI-ReID 涨点; 且增益来自几何对齐(非通用正则, 由 shuffle 对照证伪)。

## 技术方案
1. **离线**: 对 RGB 和 IR 行人 crop 各跑 SMPL fit(ROMP/SMPLer-X) → 几何(姿态参数 / 投影 2D 关节热图 / mesh 顶点)。缓存成 .pth。
2. **训练**: 共享 backbone(Swin-Small, 团队资产)双流(RGB/IR)→ baseline ID+triplet loss + **SMPL 几何对齐 loss**: 同人 RGB/IR 特征对齐到共享几何(privileged 监督)。
3. **测试**: 只用 RGB/IR encoder, **丢 SMPL** → 单 embedding 零外部(LUPI 干净口径)。
4. **baseline**: mangye16/Cross-Modal-Re-ID-baseline 的 CAJ(~67 mAP) → 换 Swin backbone。

## ★ kill-switch (step 0, 远程一回来先跑)
**低对比 IR 图上 SMPL 能不能 fit 准?** 取 RegDB / SYSU-MM01 的 IR crop 样本(~200), 跑 SMPL fit, 量: valid rate / 2D 关节 reprojection error / 人工视觉检查 N 张。
- fit 是 garbage(valid<60% 或 reproj 大) → **SMPL-anchor 死**, 转 Swin-VI 机制(领域无 Swin 赢家那条)。
- fit 尚可 → 进 design v1 + 训练。

## ★★ kill-switch 结果 (2026-06-22): SMPL 锚 **死**
torchvision keypointrcnn(COCO 2D pose)在 RegDB 上:
- RGB-visible: 检测率@0.5=85% / @0.7=69%, kp_conf=0.74(正常)
- **IR-thermal: 检测率@0.5=12% / @0.7=5%, kp_conf=-0.10(几乎全失效)**
- IR/RGB 检测率比 0.14, 置信比 -0.13。

**判定: 2D pose 在热成像上完全 OOD(所有 pose/SMPL 工具都 RGB 训), IR 提不出人体几何。SMPL 几何锚需两模态都提几何 → 死。** 加上 RegDB RGB/IR 非同时拍(几何不对应), 双重证死。
**kill-switch 价值兑现: 5min GPU 在实现前拦住, 省了 Swin 改造+几何对齐几天工。**

→ **转 fallback: Swin-VI 机制**。VI-ReID 至今 ResNet50 主导、无 Swin/CLIP 赢家(纯 ViT PMT 才 67.5 R1), 团队是 Swin/SOLIDER 专家。CAJ ResNet baseline 已搭好(POOL 76.80/69.14)→ 换 Swin-Small(SOLIDER pretrain)→ 强 backbone 在 VI-ReID 帮不帮? 帮=机制论文; 不帮=为什么不帮(模态 gap 吞掉 backbone 红利)也是机制问题。先做这个经验首验。

## 对照组 / 消融(扛"通用正则"质疑)
- baseline CAJ(双流, 无 SMPL)。
- +SMPL 几何对齐(完整)。
- ★**shuffle SMPL 几何**(跨人乱配几何)→ 增益应塌(证明是几何对齐, 非通用正则)。
- 单变量: 只加 SMPL 对齐 loss, 其余不动。

## 预期结果
- 过 CAJ baseline(~67 mAP), 冲及格线 ~73-75 mAP(SYSU all-search single-shot)。
- 天花板参考 IDKL 79.85。
- 失败最可能原因: (a) IR SMPL fit 太差(kill-switch 拦); (b) 几何对齐增益是通用正则(shuffle 对照拦)。

## 数据 / 设施
- RegDB(今天可下, github githubXin89/RegDB-dataset Drive) 先原型; SYSU-MM01(邮件 wuanc@mail.sysu.edu.cn)+ LLCM(zhangyk@stu.xmu.edu.cn)申请中。
- SMPL infra 在 lab-3090(ROMP/SMPLer-X)。
- venue: ICME/ICPR/Neurocomputing/PR Letters → 冲 ACCV/BMVC。

## 待办(远程回来后)
1. kill-switch: IR SMPL fit 可行性。
2. 复现 CAJ baseline(Swin backbone)。
3. design v1: 几何对齐 loss 的具体形式(关节热图对齐 / mesh 顶点对齐 / 姿态参数一致)。
4. 双审(Claude + Codex)→ 训练。
