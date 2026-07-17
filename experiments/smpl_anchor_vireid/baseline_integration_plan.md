# CAJ baseline 结构 + Swin/SMPL 集成计划 (2026-06-22)

baseline: mangye16/Cross-Modal-Re-ID-baseline。三层: 根=AGW base(train.py, SGD, ~47mAP); `ICCV21_CAJ/`=CAJ 强基线(train_ext.py, ~67mAP SYSU / 79mAP RegDB); PAMI23 暂不用。

## 双流架构(model.py)
- `visible_module`/`thermal_module`: 各一份 resnet50 **只跑 stem**(conv1→bn1→relu→maxpool, 64ch/stride4)=模态私有浅层。
- `base_resnet`(共享): 只跑 layer1-4(2048ch)=模态共享深层。
- `embed_net.forward(x1,x2,modal=0)`: modal0(训练)→ x1过visible/x2过thermal → **torch.cat((x1,x2),0) 沿 batch 拼**, 一起过共享 base(不是 channel 融合, RGB/IR 走同权重靠 loss 拉近); modal1/2(测试)按模态选 stem。
- 池化 GeM(p=3) → BN1d(2048) → Linear(2048,cls)。训练返回(x_pool 2048d, logits); 测试返回 l2norm 特征。

## 数据(data_loader/utils)
- batch=P×K 身份对齐: IdentitySampler 每块选 batch_size(8)身份 × num_pos(4), color/thermal 同步逐身份取(同ID非同瞬间)。有效 batch base=64 / CAJ=96。
- RegDBData 读 idx/train_visible_{trial}.txt(每行 path label)→ PIL load → resize(144,288)。**坑: __init__ 硬编码覆盖 data_dir='../Datasets/...'(改路径改这);Image.ANTIALIAS Pillow≥10 已删, 换 LANCZOS**。
- 测试: RegDB test_mode=[2,1] query=visible/gallery=thermal; SYSU=[1,2] gallery=RGB/query=IR。

## 损失 / 训练
- id(CE on logits)+ tri(TripletLoss_WRT/ADP on x_pool)。CAJ 加 KL(T=3, color-view0 ↔ thermal logits)。
- SGD wd5e-4 mom0.9 nesterov, lr0.1, warmup10+step(20×.1,50×.01)。base 81ep/CAJ 100ep, 每2ep eval。
- CAJ 关键: ChannelExchange(RGB→单通道/灰度造模态差)+ 每 color 出两视图 + KL 一致性。

## ★改造计划
### (a) ResNet50→Swin-Small(团队 SOLIDER 权重)
- 复用 `/Users/abdslm/Desktop/SOLIDER-REID/model/backbones/swin_transformer.py` 的 `swin_small_patch4_window7_224`(embed96, depths(2,2,18,2), 最终768)。forward 返回(global_feat[B,768], outs)。`base.init_weights(model_path)` 加载。
- 双流切点: Swin 天然 patch_embed | stages。private=各一份 patch_embed(返回 tokens+hw_shape); shared=stages+norm(消费 tokens+hw_shape, 必须传分辨率)。
- 改: pool_dim 2048→768(model.py L170 + train/test 多处硬编码); 删 Non_local(通道不兼容); bottleneck/classifier 768; **optimizer 必换 AdamW lr~3.5e-4 wd0.05 warmup+cosine(SGD lr0.1 会发散=最大坑)**。
- 三份 Swin(2 stem+1 shared)同一 SOLIDER ckpt 初始化。输入288×144→patch4→72×36 OK。

### (b) SMPL 几何对齐(LUPI, 测试丢 SMPL)
- 离线 ROMP 提几何(团队 [[smpl-infra-on-lab3090]] 已通), 存 train_color_geom.npy/train_thermal_geom.npy **与图像数组同序**。
- ★**几何用 pose θ/投影关节(24×2)/rot6d, 不用 β**(exp333 证 β≈随机)。
- 喂 data_loader 加一路(__getitem__ 返回 geom1/geom2), **配对铁律: geom 用和图同一 cIndex/tIndex 索引**。
- 对齐 loss **只在训练循环, 绝不进 model.forward**(LUPI 硬保证): 切 color/thermal 块, 几何门控 w_ij=softmax(-||geom_c_i - geom_t_j||) 同ID内, L_align=Σ w_ij·||f_c_i - f_t_j||²。total=id+tri+(kl)+λ·L_align。
- 测试 test() 只调 net 不碰 geom。

## ★风险(排序) + kill-switch
1. **IR 上 SMPL 是 OOD**(ROMP 训于 RGB, 热成像单通道关节可能不可靠; SYSU/RegDB 无像素对齐 RGB-IR 同瞬间对)=核心科学风险。→ kill-switch 先跑。定位为"姿态/视角兼容性门控 + 各模态独立几何", 低置信 SMPL confidence-gating 跳过。
2. β 无用(只用 pose/joints/2D 投影)。
3. 索引错位(geom 与图同序, off-by-one 静默污染)。
4. 几何别泄进 forward(否则测试丢不掉)。
5. **廉价 kill-switch**: 先只加"从编码特征回归特权几何"aux 头, 在 Swin-CAJ baseline 上看有无任何 mAP 增益; +0.0(像β)→ 早杀。

## 下一步
1. ★kill-switch A(更根本): ROMP 在 RegDB IR crop 上能否产出合理 SMPL/关节(检测率/置信/视觉)。fit garbage → SMPL 锚死, 转 Swin-VI 机制。
2. 复现 CAJ baseline(先 ResNet 确认跑通 ~67/79)→ 换 Swin。
3. design v1 + 双审 → 训练。
