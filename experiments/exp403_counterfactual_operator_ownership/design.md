# 实验 exp403：ELO-CUR 反事实算子所有权

> 当前状态：`DESIGN + STATIC CPU ONLY / GPU NO-START`

## 动机

exp401 证明 route alive；exp402 有效地否定了“当前 C0 route 使用 sample-specific rich evidence”这一解释。
直接原因不是 evidence 不变化，而是现有 router 允许 token/context/static expert 绕过 evidence：wrong-RGB
和 zero 都不劣于 correct。

exp403 不调 rho、loss 权重、batch、stage 数或 mask。它重定义 consumer 的结构对象和训练证据单位。

## 核心假设

若 evidence 直接拥有生产低秩算子的逐样本系数，并且 correct execution 必须相对 matched wrong-RGB、
NULL 和 generic execution 提高同 ID utility，则最终 descriptor 会形成可冻结复核的 sample-specific
semantic ownership；若仍不能形成，则关闭 ELO-CUR 路线，而不是继续调旧 C0。

## 技术方案

### 1. Evidence-owned shared low-rank operator

删除五个 slot-specific static experts。两个 late router 各使用一套跨五个 slot 共享的参数：

```text
h_k(p) = V(token_p) + C(local_context_k)
c_k    = H(evidence_k)                 # H 无 bias
s_k    = cos(Q(local_context_k), K(evidence_k))
g_k    = 1[e_k != 0] * sigmoid(s_k)

delta_k(p) = M_k(p) * presence_k * U(c_k ⊙ h_k(p)) * g_k
F'(p)      = F(p) + rho * sum_k delta_k(p)
```

`U/V/C/H/Q/K` 均跨 slot 共享；没有 `ModuleList[slot expert]`。`e=0` 时 `c=0` 且 `g=0`，所以无论
token/context 如何，NULL 都逐元素 exact identity。rho 固定沿用 exp401 的
`0.08075544983148575`，rank/evidence dim 均保持 `16`。

### 2. 兼容性序与完整执行效用

训练 batch 内为每个有效样本构造 deterministic donor：同 camera、不同 PID；无合法 donor 的样本只从
ownership loss mask 掉，不允许跨 camera 回退。四个 evidence arm 为：

1. correct；
2. matched wrong-RGB donor；
3. training-split frozen generic mean；
4. all-zero NULL。

兼容性头不是 auxiliary classifier：其 `g_k` 直接乘生产 delta。四臂预注册诊断顺序为：

```text
s(correct) >= s(wrong)   + 0.10
s(wrong)   >= s(generic) + 0.10
s(generic) >= s(NULL)    + 0.10
```

其中只有以下单边 ownership hinge 进入反向传播：

```text
L_compat = relu(0.10 + max(stopgrad(s(wrong)),
                              stopgrad(s(generic)),
                              stopgrad(s(NULL))) - s(correct))
```

`wrong>generic>NULL` 两个下游间隔只作冻结诊断，不伪装成可训练 loss。因为三个 reference 均
stop-gradient，若把 reference-reference hinge 加进目标，它们在数学上没有可用梯度；若开放 reference 梯度，
又会允许模型主动压低 control。该单边写法与最终 `correct-max(control)` 检索门严格同构。

完整执行 utility 使用 batch 内同 PID、排除自身的 correct descriptor prototype：

```text
u_i(branch) = cosine(descriptor_i(branch), stopgrad(positive_prototype_i))
L_CUR = mean_ref relu(0.05 + stopgrad(u_i(ref)) - u_i(correct))
ref in {wrong-RGB, generic, NULL}
```

counterfactual arms 的 descriptor、utility 与随机状态全部只作 stop-gradient reference；梯度不能通过降低
wrong/NULL utility 制造假 margin。只有 correct branch 可从 `L_CUR` 得到梯度。

### 3. 计算边界

主干 Stage 0–2 只执行一次。correct branch 按正常图执行 Stage 3 两个 block+router；三个 reference branch
从同一个 detached Stage-3 input 依次重放，恢复相同 RNG，使 stochastic depth 不成为差异来源。reference
branch 不保留 autograd graph。最终推理只执行 correct branch，仍为 RGB-only 单一 global descriptor。

### 4. loss 与梯度所有权

- frozen CLIP/pose teacher：不进 model/state/optimizer；
- mask/presence/evidence teacher loss：沿用 exp401 边界；
- ELO-CUR semantic components 与已有 mask/presence/evidence components做等权 mean，再由冻结外层
  `POSE_LOSS_WEIGHT=0.1`进入总 loss；不新增可调 loss scale；
- `L_compat/L_CUR` 只更新 correct execution 上的 student evidence、compatibility 与 shared operator；
- wrong/generic/NULL reference 不接收梯度；
- ReID loss只作用 correct descriptor；reference 不增加 CE/triplet 分支。

## 对照组

1. sealed clean D0 seed1234：raw `57.5587756578/67.6923076923/80.7692307692/84.5701357466`；
2. sealed exp401/402 current C0：route alive 但 semantic NO-GO；
3. 同一 exp403 checkpoint 的 wrong-RGB、generic、NULL、slot-cycle、wrong-mask；
4. 同一 checkpoint 的 all-router-bypass；
5. CPU negative mutant：evidence ignored；
6. CPU negative mutant：compatibility auxiliary-only，不进入 descriptor；
7. CPU negative mutant：reference 未 detach，可通过破坏 control 获利。

不会重跑 D0、exp401 或 exp402；它们只作为封板对照。

## 预期结果与正式门

只有后续 fresh seed1234 e120 同时满足，才判 `EVIDENCE_OPERATOR_OWNERSHIP_GO`：

1. full mAP `>=57.5587756578` 且 R1 `>=67.6923076923`（不低于 sealed clean D0）；
2. `correct - max(wrong-RGB, generic, NULL) >= +0.1 mAP point`；
3. `correct - all-router-bypass >= +0.1 mAP point`；
4. correct/wrong/generic/NULL descriptor arms finite、active，且训练/评测访问 teacher/pose/codebook 为 0；
5. checkpoint teacher-free、strict reload、两个 router 与 evidence head retained、RGB-only。

兼容性 ordinal 全 PASS 但最终 retrieval 未过，只能判 diagnostic proxy success / mechanism NO-GO。

## 风险与失败解释

1. **dynamic adapter 复述风险**：原子不新；若 final 强控制不成立，不能靠结构名包装。
2. **reference sabotage**：若 counterfactual arm 有梯度，结果无效，不允许修补同编号正式执行。
3. **compatibility shortcut**：若 ordinal 成立但 descriptor/retrieval 不分离，说明 gate 只是诊断头。
4. **generic content routing**：若 generic 复现 correct，sample ownership失败。
5. **算力/AMP**：reference 只重放 Stage 3 并 no-grad；若 batch64 OOM/非稳态，不改 batch 或 scale救场，
   该 execution 封板后新编号重设实现 contract。
6. **效果不足**：full 低于 D0 或 correct margin不足，只关闭 ELO-CUR，不永久否定 Phase0E/Phase0R。
