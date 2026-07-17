# 实验 exp188: Occlusion-Asymmetric Self-Distillation (OA-SD)

## 动机
- 范式级创新方向：把 occluded ReID 重新定义为"学习遮挡不变表示"
- Teacher 看 clean image (pre-PLBOA) → clean structural tokens
- Student 看 occluded image (post-PLBOA) → degraded tokens
- Distillation: student tokens 逼近 teacher tokens（cosine distance）
- **使用 EMA teacher**：teacher 是 student 的 exponential moving average (decay=0.999)
- Teacher 用 eval mode 做 forward (no dropout/droppath)，每个 optimizer step 后 EMA 更新
- 与 PersonMAE 的区别：distill identity-level tokens 而非 reconstruct pixels
- 与我们失败的 exp048/091/092 的区别：软目标 + 同一图像的 clean/occluded 版本

## 核心假设
通过让 student (看到遮挡图像) 的 structural tokens 逼近 teacher (看到完整图像) 的 tokens，
模型学会产生"遮挡不变"的 body-part 表示。

## 技术方案

### 数据流
```
同一张图像:
  ├─ clean_img (pre-PLBOA) ─→ EMA Teacher forward (eval, no_grad) → clean tokens
  └─ occluded_img (post-PLBOA) ─→ Student forward (train) → degraded tokens
                                      ↓
                               Standard loss (CE/SupCon + triplet)
                                      +
                               Distillation loss (cosine distance to EMA teacher)
                                      ↓
                               After optimizer.step():
                               EMA update: teacher = 0.999 * teacher + 0.001 * student
```

### 修改文件
1. `config/defaults.py`: POSE_OA_SD, POSE_OA_SD_WEIGHT
2. `datasets/pose_dataset.py`: OA-SD mode 保存 pre-PLBOA clean image
3. `datasets/make_dataloader.py`: 设置 _oa_sd_mode flag
4. `processor/processor.py`: teacher forward (no_grad) + distillation loss

### Distillation Loss
对 per-token features (global + 6 structural tokens):
```python
d_loss = (1 - cosine_sim(student_token, teacher_token.detach())).mean()
```
逐 token 计算 cosine distance，平均后作为 distillation loss。

### 显存
- 2x model weights (student + EMA teacher copy)
- 2x forward pass (student with graph + teacher no graph)
- 预估 ~18-20GB (student ~10GB + teacher ~3GB + model copies ~4GB)
- **需要在本地 3090 24GB 上跑**

### EMA 参数
- decay = 0.999 (标准 DINO/BYOL 值)
- teacher 初始化 = student 的 deepcopy
- teacher 在 eval mode (无 dropout/droppath)
- 每个 optimizer step 后更新

## 对照组
- exp176 (SupCon T=0.05, 无 distillation): 64.1/75.5
- 消融变量: 仅增加 OA-SD distillation loss

## 预期
- 如果成立: 遮挡不变表示 → R1 提升（更鲁棒的 top-1 matching）
- 如果失败: distillation 信号可能与 SupCon/CE 梯度冲突
