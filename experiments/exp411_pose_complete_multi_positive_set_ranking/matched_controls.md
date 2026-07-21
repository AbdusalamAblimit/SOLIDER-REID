# exp411 PCMPSR matched controls 冻结设计

## 当前前提

唯一fresh correct arm已自然完成e120并封板为`PERFORMANCE GO / ATTRIBUTION PENDING`：
`58.8 mAP / 70.1 R1 / 82.1 R5 / 85.8 R10`，相对sealed clean D0为
`+1.2/+2.4/+1.3/+1.2`。该结果只证明PCMPSR整体训练对象有效，尚不能把增益归因于
pose×CLIP五槽owner；correct formal、config、cache和checkpoint均永久只读。

## 共同不变量

- Swin-Tiny、batch64、`P×K=16×4`、seed1234、fresh output、自然e120、不续训；
- learned CE、D0 pose loss、optimizer、schedule、augmentation、global descriptor和eval完全不变；
- 每个anchor×16身份仍使用同一类内位置排除规则和严格等大小的三图support；
- set ranking仍为无temperature、无margin的全身份`softplus(logmeanexp(D_pos-D_neg))`；
- 两臂仍严格加载同一fresh PCMPSR cache和增强后pose输入，避免因数据路径或运行图不同形成混淆；
- 只添加一个显式control-mode配置，默认值必须保持`correct`，且默认关闭PCMPSR时D0行为不变；
- 两个control严格串行，先`zero_owner`，自然封板后才允许fresh启动`wrong_rgb`。

## zero-owner 单变量

owner仍可按correct路径确定并用于首batch诊断，但不再进入student集合距离。冻结公式为：

`D_zero(i,z) = sum_{j in S(z,q)} d(g_i,g_j) / 3`。

因此相对correct唯一变化是删除五个slot-owner multiplicity及相应分母项；support、正负身份、student特征、
loss形式和所有其它训练量不变。若zero-owner e120不低于correct，则exp411的涨点更接近普通多正集合排序，
不得把性能归因于pose×CLIP owner。

## wrong-RGB 单变量

support、pose visibility、owner公式和八项集合距离均保持correct；只把owner选择所用的五槽CLIP feature/validity
按batch固定左移4行。PK64中每4行属于同一PID，因此shift=4逐行保证different-PID；support本身始终保持原PID，
不会制造错误正样本。冻结公式仍为correct的`3 support + 5 owner`，但owner由错误RGB槽证据决定。

若wrong-RGB e120不低于correct，则CLIP的正确图像/PID绑定没有形成可辨识贡献，pose+CLIP语义组织归因失败。

## 启动门与最终裁决

正式control启动前只执行一次必要合同：

1. PCMPSR关闭时新增配置不改变D0 state、RNG、forward或loss；未显式配置mode时correct输出exact；
2. zero-owner的support、类标签和正类index与correct exact，集合距离逐项等于三support手工均值；
3. wrong-RGB沿用既有different-PID shift，support exact且owner变化率非零；
4. 两臂loss均finite，并由各自isolated set loss向final descriptor、Stage-3和backbone提供finite非零梯度；
5. 固定MMPOSE-ABU真实PK64 CUDA/AMP路径取得实际参数update；
6. 一次独立智能体盲审只修`BLOCKER/HIGH`，最终`0B/0H`。

correct e120必须严格高于zero-owner和wrong-RGB各自e120的mAP与R1，才判
`POSE+CLIP SCIENTIFIC GO`并保留C类会议正面方法资格。任一control不低于correct时，完整保留性能事实和不利结果，
关闭pose+CLIP归因；禁止调owner、loss、shift、batch、temperature、margin或按中间点停止来救旧机制。
