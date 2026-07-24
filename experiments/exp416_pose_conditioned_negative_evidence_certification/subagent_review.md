# exp416 PC-NEC子agent审查记录

## 审查边界

- 禁止Claude；
- 只拦截致命bug、变量混淆与exp395--415/公开强近邻整体同构；
- 不修改文件、不连接远端、不使用GPU；
- design PASS只授权无训练fuel audit实现，不授权训练。

最终design SHA256：

`d118fe038c96b067ea139c727980bc26529fe48c2131ebeabe3f817869619e67`

## 统计与证据路

首轮=`0B / 2H`：

1. pair-global/query-macro/PID-macro同时存在，但GO层级、D0+E组合公式、lambda与最强control未唯一；
2. PID-bootstrap的单位、重复权重、seed、分位数与是否重算非线性metric未冻结。

修复后：

- 唯一GO主指标为query metric的PID-macro；
- D0/E分别query内mid-rank，lambda固定`[0,.25,.5,.75,1]`，五折OOF按mAP→R1→最小lambda；
- 最强control按完整OOF逐指标先确定；
- query PID为cluster，P次有放回、PID/query两级等权、固定PCG64 seed/salt、10,000次、线性5%下界；
- UNDECIDED固定E=0并保留全分母。

最终=`PASS / 0B / 0H`。

## 机制与变量路

首轮=`BLOCK / 4H / 1 old-isomorphism`：

1. future certificate没有数学合同，可能实现成exp409 hard-pair、普通part metric或exp408 relation KD；
2. wrong-RGB donor可能与query/candidate PID冲突；
3. student-part层、pooling与归一化未冻结；
4. canonical-location control面积/coverage与neither命名不一致；
5. strongest-control bootstrap选择时序不唯一。

修复后：

- 冻结`z_qj`、detached `v/V`、全负身份`C/U` partition与set-LSE certificate loss；
- genuine identity不产生排斥梯度，证书只通过global similarity更新anchor与负身份support；
- 明确禁止pair deletion/top-k/Borda/triplet margin/relation KD；
- donor固定同时不同于query/candidate PID、同camera、全槽valid并按hash选择；
- student-part固定sealed D0 `featmaps[-1]`、area-interpolated mask weighted mean与L2 normalize；
- canonical-location CLIP/neither共享availability、逐槽固定高宽与frozen center；
- 最强control在完整OOF先确定，bootstrap不重选。

最终=`PASS / 0B / 0H / 0 old-isomorphism`。

## 公开近邻与创新边界路

最终=`PASS / 0B / 0H / 0 old-isomorphism`。

KPR/BPBreID在测试时保留part matching；PAT-CSL做part邻居/soft-label传播；Instruct-ReID以指令相似度调
triplet margin；exp409选择单个pair。PC-NEC当前只保留：

`fixed full bank + real common-visible slots + CLIP negative certificate + all-negative-identity set loss +
global-only eval`

这一整体窄差分。它只有C类条件资格，不是绝对首次声明。fuel audit本身不进入论文；future实现若退化为part
triplet、semantic margin、pair mining或test-time part matching，创新资格立即失效。

## 实现首轮复审与机制修正

首轮实现复审=`2B / 3H`：

1. geometry coverage kill-switch只存在于summary，stage3会无条件加载D0/CLIP；
2. audit在OOF/bootstrap完成后才创建once-only namespace；
3. geometry summary、candidate bank与fuel cache没有内容级交叉绑定；
4. static microbatch使用链式不等式，无法覆盖多数错配；
5. geometry/audit缺formal HEAD、source/input SHA与物理只读封存。

已修复：

- geometry NPZ内嵌gate/coverage/per-slot pair/PID receipt，stage3从当前bank与availability exact重算，
  任一门失败即在加载D0/CLIP前固定`NO-START`；
- audit在读取bank/geometry/fuel前创建fresh namespace并写`started.json`，异常写永久
  `failure.json / resume_allowed=false`；
- fuel cache内嵌`bank_sha256/geometry_sha256`，audit再次从当前bank与fuel availability重算全部coverage；
- candidate→geometry→fuel→audit逐级验证formal HEAD、receipt/manifest SHA和当前输入SHA；
- 修正microbatch显式三条件，并把成功/失败资产封存为文件`0444`、目录`0555`；
- 四个非D0 bootstrap改为每个replicate取七个paired control差的最小值，避免固定sample-selected arm的
  选择后偏差。

机制二次复审还发现future训练对象原式会主动吸引未决负身份，并混入集合大小；已在formal fuel读数前修正文档：

- `U`改为stop-gradient参考；
- `C/U`均用归一化logmeanexp；
- certificate descriptor必须与离线证书观察同一deterministic RGB view；
- fuel GO后、PK64前增加held-out-PID threshold-feasibility门。

最终design SHA256更新为：

`b0c4301c3bf34b2a59f329d0b31c261e23f58cb7b12b1624c7ffc96c6a03eed0`

上述更改均发生在任何formal asset/CUDA运行之前；当前仍为
`LOCAL STATIC PASS / FINAL READ-ONLY REGRESSION PENDING / FUEL NO-START / TRAINING NO-START`。

## Camera、主张范围与二值证书最终回归

独立只读回归首先指出一个BLOCKER：genuine固定为跨相机，而任意top-20跨PID impostor会使标签与
candidate-camera分布混淆。修复后，bank按每个query的genuine candidate-camera频数分层；每个出现的camera
先保留至少1个impostor，再按largest-remainder分配其余quota，并在对应camera stratum内取D0最近跨PID候选。
任一quota不足在读取pose/OpenCLIP前直接INVALID。构造器与audit均重算camera quota receipt，极端`1:100`
camera频数self-test已覆盖。

同轮两个HIGH已收口：

- 当前“CLIP”只指冻结OpenCLIP image encoder的region visual representation，不调用text encoder/prompt；
  正结果只能支持其相对当前matched controls的增量，不能声称语言语义或所有视觉foundation encoder不可替代；
- threshold-feasibility按raw PID五折cross-fitting，直接约束identity-level `max`后的二值证书：
  genuine identity的anchor-level family-wise误证率`<=1%`、负身份coverage的PID-macro mean`>=30%`、
  `C/U`同时非空anchor比例`>=80%`；任一失败=`THRESHOLD NO-GO / PK64 NO-START`。

第一名实现回归=`PASS / 0B / 0H`。该结论只说明D0 signal runner没有已知致命实现错误，不授权PK64/e120。

## 最终效能红队与候选裁决

后续两名独立红队分别裁决：

- `FUEL IMPLEMENTATION DIAGNOSTIC PASS / PC-NEC TRAINING CAUSALITY BLOCKED / 2B / 2H`；
- 当前D0 signal audit实现可PASS，但future PK64因consumer/domain错位BLOCKED。

BLOCKER 1是消费者变量混淆：fuel只验证E能否改善sealed D0，而future总损失以zero-owner为宿主；
exp411--414已经证明zero-owner会吸收大部分普通listwise增益，D0残差不等于宿主残差。

BLOCKER 2是identity/batch域未闭合：top-20 impostor image cache不能为随机PK64全部负身份pair提供证书，
也无法合法计算batch内16身份的30% coverage。HIGH则是修正后的`L_cert`仍属于pose×OpenCLIP决定
negative-identity subset额外梯度的家族，与exp409/412近邻；只有consumer-aligned residual gate通过，才保留
从singleton hard edge扩为all-certified-identity set的窄差分。

最终四候选裁决只保留修订PC-NEC：

1. sealed zero-owner上检验certificate对真实残余误排序身份的富集，correct须严格胜全部controls且
   PID-bootstrap下界`>0`；
2. 冻结确定性PK64 batch序列，全部`64×64` pair从按图pose/OpenCLIP cache动态得到证书，missing/fallback=`0`；
3. `P=16`、负身份数15，identity-level genuine误证`<=1%`、负身份coverage`>=30%`、
   `C/U`同时非空anchor`>=80%`；
4. exact `L_cert`无更新backward与独立zero-owner全排序梯度的归一化对齐
   `A=<g_rank,g_cert>/(||g_rank||||g_cert||)`，correct相对每个control的PID-bootstrap下界均`>0`。

PS-ODM不选，因为缺独立真实occluder资产且即便上涨也可能只是普通augmentation；APCO不选，因为与
adversarial erasing近乎同构，且无训练门容易被“攻击当然更难”自证。修订PC-NEC任一门失败即
`NO CANDIDATE`，不得自动递补二者。

最新design SHA256=
`6d062b2782abb1bd5c9fa36a1a7500ff3105c203d70e5e9e10c6d8e03622ef77`。

最终=`D0 DIAGNOSTIC CODE PASS / FORMAL HARD-BLOCKED / TRAINING NO-START`。
