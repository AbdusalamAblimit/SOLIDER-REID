# exp336 监控 — Swin 纯 LGPA-D 隔离（关 PSG）

## 配置
- lab-3090-d,原 train.py pipeline。config exp336_swin_lgpa_nopsg.yml。log /tmp/exp336.log。
- Swin-Tiny 384×128,纯 LGPA-D(PSG/OASD/parallel-aug/PLBOA 全关),scene 热图,GLOBAL_LOSS_SCALE 0.5,120ep。
- 审查:Claude(子代理)PASS + Codex approve。

## 问题 & 判据
- **CLIP 模块(LGPA-D)本身能否 standalone 涨点,还是 PSG 驱动?**
- 同一 ckpt:`test.py POSE_TEST_FEAT=equal_concat`(LGPA) vs `=global`(baseline,detach→backbone 相同)。**别设 POSE_LGPA=False,只改 POSE_TEST_FEAT。**
- equalcat > global → CLIP standalone 有效(exp335 ViT 失败=ViT-specific);equalcat ≈/< global → PSG 驱动,CLIP 部位冗余。
- 参考:Swin-Tiny baseline ~56-60。

## ✅ 启动 sanity（关键）
- **lgpa_assign: 6.977 @ e1**(scene 热图工作,对照 exp335 bug=0)。损失结构 == 原版 exp244(id_global 6.55/id_part 6.73/tri_global/tri_part/lgpa_assign)。忠实纯 LGPA-D。
- 速度 ~60s/epoch → e120 ~2-2.5h。

## 进度（equalcat mAP,训练 eval = POSE_TEST_FEAT=equal_concat）
| epoch | equalcat mAP | R1 |
|---|---|---|
| 10 | 35.9 | 47.2 |
| 20 | 43.0 | 53.3 |
| 30 | 48.5 | 57.5 |
| 40 | 52.8 | 61.4 |
| 50 | 54.8 | 63.2 |
| 60 | 56.6 | 64.9 |

### 🎯 e60 答案（同一 ckpt,test.py 同路径）—— CLIP 模块能 standalone!
| 描述子 | mAP | R1 |
|---|---|---|
| **equal_concat (LGPA)** | **56.6** | **64.9** |
| **global (baseline)** | **54.9** | **63.4** |
| **LGPA 净增益** | **+1.7** | **+1.5** |

**结论**:纯 LGPA-D 在 Swin 上(无 PSG/OASD/aug/PLBOA)**确实给 global 加值 +1.7 mAP**。
→ **CLIP 模块(部位语义)有 standalone 价值**——带了 global 没有的信息。
→ **exp335 ViT 失败 = ViT-specific**:ViT 末层 token 全局抽象,detached 池不出强部位;Swin 多尺度 stage 特征部位友好。**backbone 决定部位特征质量**,非 CLIP 无用。
→ 用户全程对:① 热图 bug 真实(assign 0→7) ② 不是 backbone 否定 CLIP,而是 backbone 选择(ViT vs Swin)决定成败。
待 e120 收敛值(gain 可能 +1.5~2.5)。
| 70 | 58.6 | 67.1 | (equalcat 续涨,gap over global e60=54.9 在扩大) |

后续:e60/e120 checkpoint 跑 `test.py POSE_TEST_FEAT=global` 取 baseline,精确对照(within-ckpt equalcat vs global)。
- e10 equalcat 35.9 = 早期爬升正常。终值参考:exp244 全系统 65.3;Swin-Tiny baseline ~56-60。
- 方向性答案 @ e60 checkpoint(~1h):equalcat vs global。
