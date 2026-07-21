# 实验 exp410：PC²P（Pose-Complete CLIP Proxy Classifier）

## 动机

exp409 PCHM 已在唯一 fresh e120 得到 `57.0 mAP / 68.6 R1`：联合 pose×CLIP hard pair 能提高首位命中，
但没有改善完整排序。下一对象不能继续选单个正负边，而应直接重定义全部身份类别的训练几何。

旧 learned classifier 的每个身份权重由单图 batch 梯度逐步形成，并不显式拥有同一身份跨相机、跨姿态、跨部位
的完整支持。PC²P 把 `single-image support incomplete` 落到分类对象：用同一 PID 的五槽 region-isolated CLIP
视觉证据构造 pose-complete identity proxy，并用该冻结 proxy 直接替换 learned classifier。

## 核心假设

对每个训练身份 `y` 和解剖槽 `r`，先聚合所有 valid 图像的单位 CLIP descriptor：

```text
q[y,r] = normalize(mean_i feature[i,r]),  i 属于 y 且 valid[i,r]
P[y]   = normalize(mean_r q[y,r])
```

先逐槽归一化、再五槽等权合成，避免高频可见或大面积 torso 证据支配身份中心。训练时保持 D0 final global feature
`g`、BNNeck `f=BN(g)`、原 batch-hard triplet 和 pose loss不变，只把普通 learned classifier替换为：

```text
logits = f @ P.T
```

`P` 完全冻结；不增加 `Q`、projection、adapter、temperature、margin、loss weight或stage。删除可学习 `Q` 是机制
成立的必要条件：当702类少于768维时，`P @ Q` 几乎可以退化成任意 learned classifier。无 `Q` 后，proxy CE若
生效就必须通过 BNNeck 进入 `global_feat`、Stage-3 和 backbone；triplet和测试检索也读取同一 `global_feat`。

## 技术方案

1. 从已封存且完整验证的 exp409 per-image五槽 CLIP cache只读构造一个 **fresh exp410 bank asset**；不复用
   exp409 output/checkpoint，不启动CLIP重编码GPU任务。
2. builder用 official train 的真实 relabel映射把15,618条路径绑定到702个PID；输出：
   - `proxy[702,768]` FP32；
   - `slot_counts[702,5]`；
   - relabel→original PID映射；
   - source cache/RGB/pose/CLIP/builder/source SHA。
3. `PC2P_ENABLED=False` 时不得读取bank，模型初始化/state/forward/loss/eval必须保持D0 exact。
4. `PC2P_ENABLED=True` 时仅训练入口加载和验证bank；model training forward显式接收bank并用FP32 `F.linear`算
   logits。原 classifier不得被调用，bank不进state/optimizer且无梯度。
5. 不对BN feature额外L2 normalize：单位proxy与BN feature直接点积，让既有BN gamma提供自然尺度；禁止因首批
   logit大小增加temperature/scale。
6. inference完全不加载bank、CLIP或外部pose，仍返回原768维 `global_feat`。

## 对照组与强反事实

首个唯一训练arm只运行 `correct`；只有自然e120性能双门GO后才串行补matched controls：

- `D0`：sealed clean learned classifier基线；
- `wrong-RGB`：对correct的702个proxy row做固定无不动点PID置换，保持row集合、范数、数量和路径完全相同，
  只破坏PID–CLIP绑定；
- `generic`：每个PID直接聚合 full-image global CLIP embedding，身份仍正确但没有pose分槽/等槽补全；
- `zero`：702行全零，CE对feature梯度为零，只保留triplet和D0 pose路径；
- `random-code`：可选的语义盲唯一单位codebook，用于排除任意固定source key。它只在correct性能GO且
  wrong-RGB仍不足以解释时执行。

不得把“所有PID共享同一均值row”当generic：它与zero一样产生严格为零的feature CE梯度，不能归因pose。

## 预期结果与裁决

- 唯一fresh seed1234/e120必须同时严格超过clean D0 raw：
  - mAP `57.5587756578`
  - R1 `67.6923076923`
- 中间e10/20/.../120只记录同epoch mAP/R1/差值，不早停。
- 性能GO后，`correct`还必须胜过`wrong-RGB`与`generic`，才能声称收益来自正确PID–CLIP绑定和pose-complete
  aggregation；否则只能记作固定classifier或普通CLIP proxy收益。

## 风险与失败解释

1. **mAP或R1任一FAIL**：冻结PC²P，不调logit scale、loss、batch、proxy norm或增加projection。
2. **R1涨、mAP不涨**：说明固定proxy仍只改善局部混淆，没有解决完整排序；直接换对象。
3. **correct≈wrong-RGB/random-code**：模型把proxy当任意类代码，pose+CLIP语义所有权不成立。
4. **correct≈generic**：增益来自普通identity CLIP proxy，不能主张pose completion。
5. **CE被BN或其他head吸收**：CE-only梯度合同必须证明 `global_feat`、Stage-3、backbone非零；没有projection可绕过。
6. **固定proxy训练不稳定**：按事实封板，不增加temperature、scale或margin补救。

## 创新边界

- 问题门：PASS——把单图支持不完整具体化为“分类几何应由跨视图、跨部位身份支持集定义”。
- 机制门：CONDITIONAL PASS——冻结CLIP原型分类已有近邻；可争差分仅是“pose逐槽跨图补全的visual
  identity-set proxy、无adapter直接替换learned classifier、测试恢复原global descriptor”的整体。
- 证据门：PASS——D0、wrong-RGB、generic、zero/random-code可分别检验固定监督、PID绑定和pose completion。

因此PC²P只定位为C类会议候选，不声称“首次CLIP classifier”或“首次pose+CLIP”。
