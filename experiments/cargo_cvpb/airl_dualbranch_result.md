# AIRL kill-switch #4 — 全共享双分支结果 + 救援决策(2026-06-23)

## 全共享双分支(--airl_dualbranch)= FAIL(ep10 早期判定)
lab-3090,Swin,fuse_w=0.25。ep10 eval:
```
[mean] full=41.99  rec=42.04  FUSE=42.05  <- model-selection uses FUSE
```
- **FUSE − full = +0.06**(oracle #3 预期软融合 +1.46)。rec≈full≈FUSE,**两 head 坍缩成同一表示**。
- ep10 绝对值 42 是中段正常(同机 OVLI-Swin ep10=45→ep60=67;AIRL 正则压早期);**真红旗是 FUSE−full delta≈0,与 epoch 无关、应随训练放大却没有。**

## 根因(结构性,非训练不足)
#3 的 +1.46 来自**独立训练的 AIRL-model + clean baseline-model**(多样)。全共享双分支:**f_rec 的 ground-degradation consistency 梯度回流共享 backbone → 整个 backbone 被 consistency 塑形 → f_full 也变 robust → 'clean vs AIRL' 两极合一**。单 backbone 装不下两极 = 死穴。

## 救援决策(3 codex 收敛)
| 路线 | CCF-B | 选择 |
|------|-------|------|
| **(b) 梯度隔离双分支** | 当前 20-30%,做成 55-65% | **★ 主救援**:f_rec 靠后 stage 独立分叉 + consistency 不污染 clean trunk。实现中(subagent aeefac8564a034f93) |
| **(a) 2-model 互补融合** | 30-40% | **兜底**:+1.46 在手,需非ensemble证据(非seed-ensemble对照/错误集互补/principled fusion)+诚实报 2× cost |
| (c) 最小 AIRL A→G 专项 | 15-25% | 当诊断/消融,不当 headline(mean 打平伤) |
| (d) 放弃 | 10-20% | 不选,三关证据在手 |

**严格门槛(codex 定)**:梯度隔离版 fuse 比 best single head **≥+0.7~1.0 mean** 才算成,否则停,转 (a)。只给 (b) 1-2 个强分化实验,不补小变体。

## kill-switch 价值
ep10(全共享 1.5 小时)就看出 +0.06 死穴,没等 4.5 小时训完 ep60。结构诊断 + codex 救援同步,工程迭代不停。

## AIRL 全景(诚实)
三关全过(诊断 +13~19 / 最小机制 area 桶 +3.6~8.4 / fusion 上界 +1.46)+ #4 全共享失败但救援在跑。**机制本身没被证伪——是"单模型内化"的实现方式要换梯度隔离。** headline 钉死 "observation-limited recoverability + complementary evidence fusion",不吹优雅单模型也不吹 ensemble。

---

## 救援(b)实现完成(2026-06-23,`--airl_dualbranch_iso`,仅实现+smoke,未训练)

### 设计(分叉点 / 梯度隔离 / 数据流)
- **分叉点**:f_rec 从 Swin stage `iso_stage`(默认 3=末段,MGN 式"share 0-2、split 末段";可选 2=split 末两段)的**输入残差流**处分叉。`SwinBackboneReID` 加 `rec_stages`=`copy.deepcopy(swin.stages[iso_stage:])` + `rec_norm`(末段 norm 拷贝) + `rec_semantic_embed_w/b`(frozen 拷贝),全部挂在 `backbone_swin` 内 → 自动进缩放 Swin-LR 组(预训练权重,与 f_full 末段同配方)。
- **梯度隔离**(死穴的修复):`_forward_swin_split` 先跑完整 f_full stage 循环(在 stage `iso_stage` 输入处 **捕获 `x.detach()` + `semantic_weight.detach()`**),rec 拷贝在循环**之后**跑。detach 是隔离边界:f_rec 的 consistency + ID-CE 梯度只更新 rec 末段 + BNNeck_rec,**永不回流共享 trunk**。clean trunk + f_full 因此保持"干净极",f_rec 学"recover 极"。
- **f_full 完全不受污染**:① 共享 trunk 零梯度;② degraded 一致性 forward 用 `rec_only=True`,只算 f_rec 头 → f_full BNNeck running stats **不被 degraded ground 图更新**(比全共享版更干净,后者接受了这点 minor exposure);③ rec 拷贝在 f_full 循环后跑 → 训练期 DropPath RNG 序列对 f_full 也忠实(split f_full 图 == 原单图 forward,train/eval 都 max|d|=0)。
- **eval**:软融合不变 `cos=w·cos_rec+(1−w)·cos_full`,w=0.25,复用 `airl_dualbranch_eval`(它调 `model(return_dual=True)`,iso forward 的 want_iso 路径返回同样的 `(f_full,f_rec)` 元组)。一次 forward 两特征。

### 改动
- `afd_reid/afd_model.py`:`SwinBackboneReID`(iso_branch/iso_stage + `_run_rec_stages` + `_forward_swin_split` + `forward(return_rec,rec_only)`);`AFDModel`(`airl_dualbranch_iso`/`airl_iso_stage` flag+guard、`bottleneck_rec`/`classifier_rec` 复用、`_embed_rec`、forward 的 want_iso/rec_only 路径);`build_model` 透传 2 参。
- `cargo_cvpb/afd_train.py`:`--airl_dualbranch_iso`/`--airl_iso_stage` CLI + guard(互斥 airl/dualbranch、swin-only、stage∈[1,3]、standalone);print 行;优化器自检(rec 末段→Swin 组、rec BNNeck→full 组);训练环 `loss_ce_rec` 门控 + iso consistency 块(degraded forward `rec_only=True`);**`airl_lambda_eff` warmup 门控加 iso**(否则 consistency 乘 0);per-epoch log(AIRL-ISO);eval 派发。
- `cargo_cvpb/smoke_airl_iso.py`:14 项 numeric smoke,全过。

### 双审 + smoke
- **Claude broad review**:第 1 轮抓出 **Critical C1**——`airl_lambda_eff` 漏 iso flag → iso 运行时 consistency 恒乘 0、recover 信号从不训练(smoke 的 loss-only 检查结构上抓不到)。已修 + 加回归 smoke I11。结论 APPROVE(`claude_review_airl_iso.md`)。
- **Codex review(--search)**:第 1 轮 1 Medium(degraded forward 漏 BN-stat 进 f_full)+ 2 Low(DropPath RNG / smoke 覆盖)。全修(rec_only 路径 + 循环重排 + smoke I4/I12/I13)。第 2 轮 **approve,no findings**(`codex_review_airl_iso.md`)。
- novelty:无 exact prior("detached fork into independent copied late backbone stage + recover branch + fixed cosine prior fusion" for ReID);成分接近 MGN/GreyReID/cross-res ReID/ControlNet-locked-copy/SimSiam-stopgrad → novelty 钉在 CARGO aerial-ground 失败模式 + 证据,不吹全新 primitive。

### smoke 14 项(全过)
I1 OFF 字节级 / I2 split f_full==原图(eval) / I3 dual head+rec 扰动隔离 f_full / **I4 consistency 梯度只到 rec(trunk+f_full 零)+ rec_only degraded 不动 f_full BNNeck stats** / I5 f_full CE 不碰 rec / I6 LR 分组+三者都动 / I7 软融合 distmat / I8 NaN 安全 / I9 f_rec ID-CE grounds rec 且 trunk 零 / I10 fp32 consistency / **I11 trainer warmup 含 iso(回归)** / **I12 iso_stage=2 隔离** / **I13 train-mode DropPath RNG 忠实**。既有 `smoke_airl_dualbranch`(11/11)+`smoke_airl`(21/21)回归通过。

### 待办(下一步,未做)
GPU 空出后训练 iso(swin,iso_stage=3,fuse_w=0.25,60ep)。**kill-switch**:fuse mean 比 best single head **≥+0.7~1.0 mean** 才算成,否则停转救援(a)。该判据只在 C1 修复后有意义。建议首 epoch log 确认 AIRL-ISO `consistency=` 非零且 `lam_eff` 从~0 ramp。
