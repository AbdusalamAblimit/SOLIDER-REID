# 实验 exp404：SPK 固定语义乘积核描述子

> 当前状态：`CUDA V1 INVALID / V2 FAIL / DEFAULT-GRADSCALER V3 EXECUTION AUTHORIZED`

## 目标调整

用户已将投稿目标明确调整为C类会议。exp404仍遵守不重跑封板实验、不删不利control、不调rho/loss/batch救旧臂
等科学底线，但不再要求每个结构原子都达到B类主贡献级首创。创新主张上限冻结为：

1. open-set ReID中区分route alive与semantic ownership的问题定义；
2. 一个固定、不可学习绕过的语义乘积核描述子；
3. wrong/generic/NULL/all-bypass之外加入random-key与frequency-matched null的证据协议。

不声称发明InfoNCE、张量积、concept bottleneck或因果可识别性理论。

## 动机

exp402证明旧C0的sample evidence不拥有最终排序；exp403又证明即使evidence生成可执行低秩算子，global descriptor
仍可对该变化近乎不敏感。后续无编号诊断进一步证明，`correct > wrong > generic/NULL`本身也可能由随机source
key伪造。

exp404不再把evidence注入hidden residual。它把semantic factor放到最终768维欧氏描述子的固定乘积核中，使
classifier、triplet与测试距离读取同一个绑定后的对象；结构没有可学习projection、constant branch或additive
bypass。

## 核心假设

若正确的逐图semantic evidence确实补充了身份判别，固定乘积绑定应同时满足：

1. correct优于matched wrong-RGB、generic与NULL；
2. correct优于semantic-blind unique random-key；
3. correct优于类别数/频率匹配的random-cluster null；
4. 关闭乘积因子后检索下降。

若只击败wrong/NULL却不击败random controls，只能说明source authentication；若descriptor active但检索不分离，
则固定乘积也没有建立semantic ownership。

## 技术方案

### 1. 结构对象重置

- backbone固定Swin-Tiny；
- 保留D0 pose spatial path与现有训练期pose边界；
- rich CLIP teacher仍只在训练期生成五slot、16维target，student RGB anchor预测同形evidence；
- 删除C0 static semantic experts与exp403 ELO-CUR operator，不复用其compatibility/CUR目标；
- semantic evidence只在最终global feature处执行一次固定乘积绑定。

这不是给exp402/403调loss或rho，而是把机制对象从hidden residual改成final metric representation。

### 2. 固定Semantic Product Kernel（SPK）

对student evidence `e in R^(B,5,16)`及student presence `p`做确定性聚合：

```text
e_bar = sum_k p_k * e_k / max(sum_k p_k, 1)
a(e)  = 16 * softmax(e_bar)                 # 无temperature、无参数
F     = reshape(global_feat, B, 16, 48)     # 固定连续分组
D(e)  = reshape(a(e)[...,None] * F, B, 768)
```

`e=0`时`a(e)=1`，所以NULL与product-bypass在乘积前逐元素exact等于原global feature。正确分支的classification、
triplet、BNNeck和正式测试均只读取`D(e)`；不保留`global_feat + D(e)`、concat constant channel或可学习降维。

### 3. 训练与部署

- 训练仍只有correct execution接收ReID loss；不把wrong/random control训练成负例；
- mask/presence/evidence teacher supervision沿用冻结权重，不新增ownership loss；
- 推理时teacher/pose/codebook均不访问，只由单RGB预测`e/p`并输出一个固定768维descriptor；
- 标准欧氏/cosine评测不增加pair-specific scorer、re-ranking或第二输入。

### 4. 强反事实

同一最终checkpoint串行执行：

1. correct student evidence；
2. same-split/same-camera/different-PID wrong-RGB evidence；
3. train-split frozen generic mean；
4. NULL zero；
5. all-product-bypass（必须与NULL exact）；
6. unique random-key：每个sample使用hash确定的signed permutation，保持自身evidence范数与绝对值多重集；
7. frequency-matched random-cluster：8个semantic-blind共享类别，严格频率平衡并预注册PID/camera覆盖门；
8. wrong-mask与slot-cycle仅作归因补充，不替代上述主门。

random controls只在正式终审替换supplied factor，不参与训练，也不得因结果不利而删除。

## 对照组

1. sealed clean D0 seed1234：`57.5587756578/67.6923076923/80.7692307692/84.5701357466`；
2. sealed exp401：route alive弱正边界；
3. sealed exp402：current C0 semantic-interface NO-GO；
4. sealed exp403：ELO-CUR mechanism NO-GO；
5. exp404同checkpoint全部冻结反事实。

所有sealed编号只读引用，不重跑或补跑。

## 两级正式门

### C-track mechanism GO

唯一fresh seed1234/e120同时满足：

1. validity、strict reload、RGB-only、teacher-free、state/RNG/patch恢复全部PASS；
2. correct mAP `>=56.7`；
3. `correct - max(wrong-RGB,generic,NULL,random-key,random-cluster) >= +0.1 mAP point`；
4. `correct - all-product-bypass >= +0.1 mAP point`；
5. 所有arm finite/active，NULL与all-product-bypass exact，random controls分布合同PASS。

### C-track paper GO

在mechanism GO之外，再要求correct mAP/R1不低于sealed clean D0。若只过mechanism门，则最多形成C类机制/证据
候选，不声称超过主基线；若semantic或random-control门失败，则`SPK MECHANISM NO-GO`。

## 风险与失败解释

1. **乘积原子非首创**：贡献只允许落在ReID ownership对象与完整证据协议。
2. **uniform collapse**：student evidence若使`a(e)`近常数，NULL exact但route不活，判NO-GO。
3. **group hiding**：backbone可能把身份信息塞进近常权重组；random-key/random-cluster门用于抓该shortcut。
4. **语义与同ID变化冲突**：逐图appearance状态可能降低同ID跨视角相似度；不得用temperature/scale救场。
5. **random-cluster validity**：任何覆盖门未过只封板该执行，不降低门槛或同编号补跑。
6. **性能不足**：不按中间eval早停；最终未过两级门即如实封板。
