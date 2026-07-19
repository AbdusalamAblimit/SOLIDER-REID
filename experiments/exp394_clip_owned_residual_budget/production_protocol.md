# exp394 Production：证据预算化CLIP-Owned Executable Residual协议

## 当前状态

`STATIC PROTOCOL FROZEN / PHASE 0P SOURCE CONTRACT PASS / PRODUCTION STATIC-CPU PASS /
CUDA PROTOCOL FROZEN / PREFLIGHT IMPLEMENTATION STATIC-PASS / CUDA PREFLIGHT EXECUTION GO /
FORMAL NO-START`。

本协议冻结fresh implementation对象和反事实；production static/CPU现已封板，且未修改exp393 sealed
repo。后续指令现只授权一次冻结CUDA preflight，仍不授权正式训练。Phase 0R-S与0R-128 PASS只说明
代数/梯度接口和预算来源可执行，不证明本方法有效。

## 冻结基线seam

实现基线只能是exp393 RZ-C0 exact source `09340f76f84502f9018bee3c8eec005961b0a8cb`。本地当前
source与远端sealed repo下列八文件SHA exact一致：

| 文件 | SHA256 |
|---|---|
| `model/tapf.py` | `559b75f1aad9973828f7298789f50d6b8e7fd536d648423d3468ee5903f0f1ba` |
| `model/make_model.py` | `87603a7eb2f26d599d0d3e755fe9997ae168197351b2823bd5eff0b823e9f4b0` |
| `model/backbones/swin_transformer.py` | `b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef` |
| `processor/processor.py` | `5b0886cb16ec0e9020d39ed14bc119e8e35c88661148b7af8b1208c9edda4904` |
| `config/defaults.py` | `b67365bd7f238a3263abf165e863386dcde0766cfa38c7f89e885eb856f63005` |
| `datasets/pose_dataset.py` | `d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc` |
| `model/clip_semantic_teacher.py` | `50c2607394f81573788ade6c1337f173753763cd35d69925a4645dbee695de79` |
| RZ-C0 config | `f409cc069b6f3500e009e6d40681e8baf9547bb77b864e9f35a7ea02ca11d1a6` |

`swin_transformer.py`的Stage-3两个block→两个`tapf.apply_gate`是已验证production seam，必须保持blob
exact；不得增加stage或consumer。`datasets/pose_dataset.py`已提供pre-RE `teacher_rgb`且eval loader
RGB-only，也必须保持exact。默认config的TAPF与新开关均须关闭。

## Frozen rich teacher

训练期外置teacher沿用Phase 0E-FULL的PC-MBCLS readout与hard-owner五slot mask：

```text
raw[b,r]  = normalize(region_cls[b,r]) - normalize(global_cls[b])
code[b,r] = normalize((raw[b,r] - slot_mean[r]) @ shared_basis.T)
```

- code shape=`[B,5,16]`，invalid slot exact zero；
- full codebook路径固定为
  `/home/afr/reid-clean/audits/exp393_phase0e/phase0e_full_codebook.json`，SHA256=
  `fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a`；
- teacher验证definition、`slot_means[5,D]`、`shared_basis[16,D]`、basis orthogonality与全部finite；
- CLIP checkpoint SHA仍为
  `9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`，但新config禁止引用旧
  `/home/afr/SOLIDER-REID`路径；production前必须复制到新的clean canonical实体文件并复核SHA，
  禁止symlink/path mapping；
- 不加载text encoder/prompt，不恢复scalar q；PCA只作固定压缩器；
- teacher始终`inference_mode/eval/requires_grad=False`，在model/optimizer/checkpoint/eval之外；
- query/gallery与RGB-only inference不得构建teacher、读取pose或codebook。

## Student state与梯度边界

anchor trunk仍只吃`source_feature.detach()`。共享hidden产生pose、region mask和presence；新增
`evidence_head`从`hidden.detach()`预测`[B,5,16]`并L2 normalize。这样：

1. pose/mask/presence loss更新anchor trunk和各自head，不回流backbone；
2. evidence cosine/relation loss只更新evidence head，不回流anchor trunk/backbone；
3. production ReID forward使用`e_student.detach()`、`mask.detach()`、`presence.detach()`，故ReID只更新
   backbone、T/C/E/Expert和ID head，不把evidence student改写成identity code；
4. `L_exec`用同一T/C/E/Expert重算，但输入token/context/mask/presence全部detach，只让梯度到
   T/C/E/Expert与evidence head；
5. teacher code永久detach；final descriptor不接受CLIP feature/text/logit或descriptor KD。

不得新增训练后删除的projector吸收teacher loss。evidence head与T/C/E/Expert都必须保留在RGB-only
推理路径和checkpoint中。

## Production router

两个独立router各含共享到其内部的`T:768→16`、`C:768→16`、`E:16→16`及五个slot expert
`Expert_r:16→768`；不同consumer之间不共享权重。对每个router：

```text
z_r          = MaskPool(tokens, stopgrad(mask_r))
h_r(p)       = GELU(T(tokens_p) + C(z_r) + E(stopgrad(e_student_r)))
b_r(p)       = Expert_r(h_r(p))
bhat_r(p)    = b_r(p) / stopgrad(sqrt(mean_c(b_r(p)^2)) + 1e-6)
unit_delta_p = sum_r mask_r(p) * presence_r * bhat_r(p)
routed_p     = tokens_p + rho(epoch) * unit_delta_p
```

expert用局部保存/恢复RNG的small-nonzero初始化；T/C/E使用正常初始化。zero-mass或presence=0 slot的
`bhat/scatter`必须exact zero，NULL mask/presence必须identity。RMS在FP32计算、denominator stopgrad，
再转回token dtype；任何nonfinite立即FAIL。

`rho`只允许是由config读取的Python float，不注册parameter/buffer，不接受梯度：

```text
e1-e5: rho=0 exact
e6-e9: rho=rho_star*(epoch-5)/5
e10+:  rho=rho_star
eval:  rho=rho_star
rho_star=0.08075544983148575
```

teacher阶段虽然descriptor identity exact，`L_exec`必须更新真实production branch；handoff只打开已冻结
预算，不改变loss、温度、样本或teacher。

## Loss定义

对flatten后的valid slot集合，`relation(u)`定义为L2-normalize后完整pairwise cosine Gram；只统计两端
均valid且非对角pair，少于2个valid时返回与`u`相连的exact zero。

```text
L_evidence_cos = mean_valid(1 - cosine(e_student, teacher_code))
L_evidence_rel = MSE_valid_pairs(relation(e_student), relation(teacher_code))
v_k[b,r]       = MaskPool(pre-budget b_k[b,r], stopgrad(mask[b,r]))
L_exec_k       = MSE_valid_pairs(relation(v_k), relation(teacher_code))
L_exec         = mean(L_exec_0, L_exec_1)
L_semantic     = mean(L_mask, L_presence, L_evidence_cos,
                      L_evidence_rel, L_exec)
L_pose         = L_heatmap + L_confidence + L_semantic
L_total        = L_ReID + 0.1 * L_pose
```

不新增可调loss weight，沿用D0/Semantic C0的总`POSE_LOSS_WEIGHT=0.1`。`L_exec`必须监督pre-budget
`b_r`而非`bhat`、final delta或final descriptor，防止固定rho直接优化出表面检索差。

## Teacher→student与eval

- mask/presence沿用e1-e5 teacher、e6-e9线性handoff、e10+ student；
- evidence production输入从第一步起始终为student prediction，不把teacher code送入推理路径；
- teacher code只作`L_evidence/L_exec` target；
- eval固定student mask/presence/evidence与`rho_star`，`pose_batch=None`/shuffle/exploding必须exact；
- checkpoint strict reload后teacher/codebook/CLIP/text/pose均不在state。

## 同checkpoint串行反事实

1. `correct`：正常student evidence/mask/presence；
2. `all-bypass`与两个single-consumer bypass；
3. `static`：evidence exact zero（centered slot mean）；
4. `wrong-RGB`：按different-PID固定donor map只交换evidence，保留recipient RGB/mask；
5. `wrong-mask`：同RGB循环mask slot，保留evidence；
6. `slot-cycle`：循环evidence slot，保留mask；
7. `random-orthogonal`：固定seed正交矩阵右乘evidence，保持norm/relation但破坏已学坐标；
8. `generic-context-only`：令`E(evidence)=0`，保留已训练T/C/Expert、同一rho与全部参数文件；这只证明
   同checkpoint对evidence项的依赖，不能冒充独立训练的matched generic adapter；
9. `budget-only`：用固定seed、slot-specific、channel-RMS=1的随机方向替换learned proposal，保留
   mask/presence/rho；
10. correct start/end、state SHA、descriptor与hook restore exact。

如果候选最终通过，真正“matched generic normalized route”必须以后作为fresh独立消融训练，不能混入
当前单臂或拿同checkpoint context-only反事实替代。

## 允许修改范围与默认off

未来fresh implementation只允许目标性修改：

- `model/tapf.py`：新anchor/router/TAPF class；
- `model/clip_semantic_teacher.py`：新增rich image-only teacher，不改sealed scalar class语义；
- `model/make_model.py`：新开关选择新class；
- `processor/processor.py`：外置teacher target与冻结loss日志；
- `config/defaults.py`：默认关闭的新字段；
- 新exp394 config与独立preflight/audit脚本。

`model/backbones/swin_transformer.py`、`datasets/pose_dataset.py`、旧config与旧class必须blob exact。
普通baseline、D0、HT0、Semantic C0、RZ-C0的config-off构造/state/forward必须逐tensor parity。

## 下一门禁

1. 先运行独立AST/source contract，证明上述sealed seam、默认off和production-absent边界；
2. PASS只授权在fresh独立repo实现，不授权CUDA或训练；
3. 实现后先做static/CPU exact，随后至少24步真实batch64 CUDA/AMP；
4. CUDA门必须覆盖teacher阶段identity、branch更新、handoff descriptor gap、三类loss梯度所有权、两个
   consumer、correct/wrong/static区分、strict reload、teacher隔离、RGB-only与显存；
5. 全部门禁和资产SHA冻结后才可能授权fresh e120，仍需final all-bypass与强反事实裁决。

## Phase 0P source contract封板

独立AST/source contract在本地uv环境连续两次result与runner SHA exact，19项检查全部PASS：八个sealed
source/config SHA、TAPF与semantic默认off、production flag/config当前确实不存在、Swin Stage-3两个
live consumer、两个独立router、token/context/expert seam、anchor source detach、state进入router前
detach、teacher只在train构建且target no-grad/detach、checkpoint只保存model state、eval无teacher/pose
构建、pre-RE teacher RGB与frozen visual seam全部成立。

script/result/runner SHA256分别为
`27859a2ae0b5a1020b9a68cda5777ad332e05701c372940d584728eb5d60fae1`、
`496630c3e7ba1d76d4e49b6347f8741f6d799b56ca5321283792bda92fd4cb8d`、
`0e5070db2ec733e76139d7b41bd55cba724cf4cafba9dc5e23845e7c85be5eb5`。

裁决：`PHASE0P_SOURCE_PASS`，只授权从`09340f7`建立fresh独立implementation repo并实现本协议；
不得修改sealed repo，不授权CUDA/AMP preflight、正式训练或semantic multi-stage。

## Production static/CPU实现封板

实现严格只修改五个已授权source、新增exp394 config与独立CPU contract；
`swin_transformer.py`和`pose_dataset.py` SHA保持
`b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef`/
`d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc`。

CPU contract确认：e1–e5 rho exact zero，e6–e9按冻结公式线性打开，e10+/eval固定
`0.08075544983148575`；NULL mask/presence的normalized proposal、delta与descriptor均exact zero/
identity；两个consumer独立；FP32 channel RMS finite；correct/wrong/static evidence改变pre-budget
proposal；strict reload逐forward exact；teacher/CLIP/codebook/text不在model state。

四类分loss梯度所有权全部PASS：evidence cosine/relation只更新evidence head；mask/presence更新anchor
trunk与对应heads；`L_exec`更新evidence head和两个router的T/C/E/Expert而阻断backbone/anchor/ID；
ReID更新token backbone、两个router与ID head而阻断source anchor/evidence head。最终semantic loss逐
tensor等于五项无权重mean，总loss仍只通过原`POSE_LOSS_WEIGHT=0.1`接入。

最终script/result/runner SHA256分别为
`5be2980eb6a666f791ba5e3cd87bbabb7a0b9934bb44724e091cbbb7e4545cd1`、
`658ac1fd261ec09db618e9d658ae00fa3f0f7d7887b87e8716c601adbc8b0636`、
`658ac1fd261ec09db618e9d658ae00fa3f0f7d7887b87e8716c601adbc8b0636`；第二遍result/runner SHA完全一致。
裁决=`PRODUCTION_STATIC_CPU_SEALED_PASS / CUDA PREFLIGHT DESIGN GO / CUDA NO-START /
FORMAL NO-START`。

clean canonical CLIP实体已在CPU-only资产步骤落盘为
`/home/afr/reid-clean/weights/exp394_clip_l14_openclip_9ce2e8a8.safetensors`，确认regular file、非
symlink且SHA=`9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`；full codebook SHA仍为
`fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a`。该资产就绪不等于CUDA门通过。
