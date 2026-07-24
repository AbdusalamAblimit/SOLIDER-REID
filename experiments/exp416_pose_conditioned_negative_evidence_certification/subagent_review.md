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
