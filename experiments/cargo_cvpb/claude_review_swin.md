# Claude Broad Review — Swin backbone(--backbone swin_small)

**审查对象**: afd_model.py 的 SwinBackboneReID(L76)+ AFDModel backbone 分支(L326 resnet50 / L359 swin)+ build_model;afd_train.py 的 --backbone(L1059)+ guard(L1231)+ build(L1280);smoke_swin_backbone.py
**日期**: 2026-06-23
**结论**: 审查通过(无 Critical/High/Medium)

## 审查范围(全范围)
a. 设计合理性(Swin backbone 冲 SOTA, 团队资产, 换 backbone 非加模块)
b. SwinBackboneReID 逐行(hook 契约 + grad 流)
c. resnet50 默认字节级不变
d. OVLI hook 适配(NCHW map)
e. 预训练权重加载 + avg-pool 强制

## 逐项
### 1. 设计合理性
弱 resnet50 都到 OVLI 52.37 超 VDT 42.76 → 换 SOLIDER Swin 强 backbone 冲 SOTA。两个涨点模块(SetVLAD/ACVP)证伪后的正确转向: 不加模块, 换 backbone。

### 2. SwinBackboneReID 逐行(L76-127)
- **self.layer4 = nn.Identity()**(L119): OVLI 的 model.layer4 forward-hook 点, 捕获 NCHW map(与 resnet 同契约)。✓
- **forward**(L121-127): swin(x) → (gfeat, outs), outs[-1] 经 Identity 路由 → (B,768,H,W) NCHW, **无 detach → grad 流 backbone→proj**。smoke[3] 验证 grad 到 patch_embed.projection.weight。✓
- **init_weights(pretrain_path)**(L113-115): SOLIDER teacher ckpt(backbone.* keys), strict=False。smoke[2] `<All keys matched successfully>`。✓
- out_dim=768(L112): swin_small num_features[-1]。256×128 → 8×4=32 tokens(= OVLI 默认网格, adaptive_pool no-op)。✓

### 3. resnet50 默认字节级不变
backbone='resnet50'(默认)→ 原 resnet50 body 逐字保留(L326 内)。新 args 追加签名末尾 → 现有 caller 不变。smoke[1] resnet50 in_planes 2048, logits/BN unit-norm 正常。✓

### 4. OVLI hook 适配 + avg-pool 强制
Swin forward 返回 NCHW(B,768,8,4)= OVLI hook 期望格式。smoke[3] 捕获 (16,768,8,4), tokens (16,32,256) per-token unit-norm, OVLI loss 1.3152。Swin 最后 map LayerNorm'd(有负值), GeM clamp(min=eps) 破坏负半 → **强制 avg-pool**(L376, review 抓到的好点)。✓

### 5. guard + 依赖
swin + --use_afd → error(L1231, AFD 插 resnet 浅层 Swin 没有)。_ensure_mmcv_stub 处理 mmcv 缺失(lazy import)。✓

## Findings
- **Critical/High/Medium: 无。Low: 无实质问题。**

## 结论
审查通过。Swin backbone 字节级保留 resnet50 默认 + OVLI hook 适配 NCHW + grad 流到 Swin + 预训练 teacher ckpt 加载(all keys matched)+ avg-pool 强制(GeM 不破坏 LayerNorm 负值)。smoke 4 组全过(lab-3090 真 swin_small.pth)。codex 审 + GPU 空即跑 Swin OVLI(`--backbone swin_small --swin_pretrain <repo>/pretrained/swin_small.pth`,可选 `--img_size 384 128` SOLIDER 原生分辨率)。
