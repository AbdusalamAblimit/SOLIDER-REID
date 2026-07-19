# 实验 exp394：证据预算化的 CLIP-Owned Executable Residual

## 当前状态

`DESIGN-FROZEN / FORMAL NO-START`。本实验不是exp393 Phase B续跑，不复用其训练checkpoint，
也不降低Phase A门槛。只复用已封板的Phase 0E rich teacher定义与codebook SHA。

## 动机

exp393 Phase A证明了两个不同事实：

1. random nonzero expert加ReZero能让alpha、token/context projection和expert都获得梯度；
2. 仅由ReID loss拥有的自由alpha最终仍停在`1e-4`，full−all-bypass只有
   `-0.000249709 mAP point`。

因此下一问题不是继续调alpha，而是：**怎样让rich CLIP evidence拥有production branch的方向，
同时让执行预算不能静默塌回identity，又不把CLIP直接蒸馏到final descriptor。**

## 核心假设

Phase 0E已证明centered 16维local evidence在全量held-out PID上高rank且绑定correct RGB/mask。
如果该evidence先在零执行预算阶段训练真实production expert的方向，再通过预注册、与clean D0活跃
residual能量匹配的有界预算执行，则：

- route应在all-bypass检索反事实中留下可见贡献；
- correct evidence应优于wrong/static/generic控制；
- CLIP仍不直接监督final ReID descriptor。

## 技术方案

### 1. Rich evidence student

沿用exp393 Phase 0E定义：每slot teacher code为region CLS减同图global、减fit-slot prior、共享固定
PCA-16并单位归一。学生从detached anchor source feature预测`e_student[B,5,16]`，用cosine与relation
loss学习；进入production router前detach，防止ID loss把evidence改写成identity code。

### 2. Production branch

```text
z_r        = MaskPool(F_consumer, stopgrad(mask_r))
h_r(p)     = GELU(T(F_p) + C(z_r) + E(stopgrad(e_student_r)))
b_r(p)     = Expert_r(h_r(p))
bhat_r(p)  = b_r(p) / stopgrad(RMS_channel(b_r(p)) + eps)
DeltaF(p)  = rho(epoch) * sum_r mask_r(p) * presence_r * bhat_r(p)
F'(p)      = F(p) + DeltaF(p)
```

`L_exec`仍作用于生产expert生成的pre-budget branch proposal，并用同权重、detached tokens重算，
只更新T/C/E/Expert与evidence head，不回流backbone或final descriptor。

### 3. 非塌缩执行预算

`rho(epoch)`不是可学习alpha：

- teacher阶段保持exact zero，初始化与baseline descriptor exact；
- handoff阶段从0线性升到固定`rho_star`；
- 此后固定，不接受ReID或CLIP梯度。

`rho_star`不得按性能搜索。它只允许通过一个train-only、固定128图的只读审计确定：取clean D0
seed1234 final两个活跃consumer的per-token applied-delta RMS中位数，匹配到本实验归一化branch的执行
能量。不得读取query/gallery或使用exp394中间指标。

固定预算只是防塌缩接口，不是创新或成功证据。若generic normalized route同样有效，不能归因CLIP。

## 梯度所有权

- frozen CLIP image/text encoder、PCA、teacher code：永久no-grad；
- evidence/mask/presence loss只更新anchor/student，不回流backbone；
- `L_exec`只更新推理保留的T/C/E/Expert与evidence head；
- ReID loss更新backbone、production router、BNNeck/classifier；
- `rho`固定无grad；
- final descriptor不接受CLIP feature/text/logit KD。

## 对照组与反事实

同一final checkpoint必须串行评测：

1. correct rich evidence；
2. all-router-bypass；
3. slot-mean/static evidence；
4. wrong-RGB evidence；
5. same-RGB wrong-mask与slot-cycle；
6. fixed random orthogonal code；
7. generic normalized route（删除evidence projection，保持预算与参数规模匹配）；
8. budget-only direction control；
9. RGB-only correct/shuffle/None/exploding pose exact。

## 执行门禁

### Phase 0R-S：synthetic/CPU exact

该contract必须是本地独立审计，不修改远端sealed execution repo、production model或config。

1. `rho=0`时full/bypass逐tensor exact；
2. NULL mask/presence exact identity；
3. budget schedule只由epoch决定，repeat exact；
4. branch RMS normalization finite，zero-mass slot exact zero；
5. correct/wrong evidence改变pre-budget branch，static code不伪造sample variation；
6. 分loss梯度所有权exact。

### Phase 0R-128：train-only预算冻结

只读clean D0固定128图，报告两个consumer applied-delta RMS分布、median/P95、repeat、RGB-only、吞吐和
SHA；冻结唯一`rho_star`。失败只阻断本接口，不调整样本或读取验证集。

### CUDA/AMP preflight

至少24步验证：teacher阶段branch因`L_exec`更新但descriptor保持exact identity；handoff后预算非零、
descriptor gap finite；correct/wrong/static branch可分；backbone/ID head/两个consumer均更新；strict
reload、teacher isolation、RGB-only、峰值显存全部PASS。

上述门未全部冻结前正式训练`NO-START`。

## 正式训练边界

若获授权：official clean Occluded-Duke、Swin-Tiny、batch64、seed1234、SGD lr0.0008、120 epoch、
eval10、checkpoint120；fresh、单变量bundle、唯一4090、final-only、自然跑满。不得续训、重复、换seed、
挑best或按中间性能调预算/loss。

## 最低成功条件

1. final相对clean D0 `>=+0.3 mAP`且R1不下降；
2. full−all-bypass `>=+0.2 mAP`；
3. correct相对wrong/static/generic至少一个关键mAP差`>=+0.1`，其余方向不反转；
4. 内部branch反事实与检索方向一致；
5. teacher-free、pose-free、strict finite、两个consumer独立贡献全部PASS。

任一FAIL只关闭当前evidence-normalized budget接口，不否定Phase 0E teacher或CLIP–TAPF总体。

## 创新边界

不能把固定预算、RMSNorm、PCA或local CLIP KD声称为贡献。只有当rich evidence对推理保留branch拥有
可反事实识别的方向控制，并同时改变final retrieval，才可争“counterfactually executable mediator”。
仍需与RegionCLIP、ALADIN、π-VL、PAFormer、ProFD的真实执行路径核对。
