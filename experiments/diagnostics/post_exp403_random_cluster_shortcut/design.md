# post-exp403 CPU诊断：频率匹配random-cluster假语义

> 状态：`DESIGN / CPU ONLY / NO EXP404 / GPU FORBIDDEN`

## 动机

random source-key诊断已经证明，逐样本随机key可伪造`correct > wrong > generic/NULL`。一个自然修正是要求
semantic state在跨身份样本间重复，使operator只能由共享类别而非sample checksum触发。但“重复”本身仍不等于
语义：频率匹配、PID无关的随机类别也具有共享支持。

本诊断检验更窄命题：若平衡随机cluster同样能形成强检索顺序，则后继离散属性、prototype或codebook机制不能只
击败unique random-key；还必须击败具有相同类别数与频率的semantic-blind random-cluster control。

## 核心假设

为384个toy sample分配8个完全平衡、与PID/camera无关的随机cluster。descriptor保留identity prototype，但
identity预算只由supplied cluster是否为真实共享类别、以及是否与host cluster相同决定：

```text
q(correct cluster) = 1.00
q(other real cluster) = 0.45
q(generic) = 0.18
q(NULL) = 0.05
descriptor = [q * identity, sqrt(1-q^2) * mismatch_nuisance]
```

四臂为correct、same-camera different-PID donor cluster、generic和NULL。再对全部cluster assignment做一次新的
频率保持随机置换。evidence-ignored mutant令所有臂`q=0.45`且supplied cluster不再进入nuisance，必须不能
通过强顺序。

## 解释边界

该toy只证明“跨样本共享 + 频率平衡”仍可产生非语义假阳性，不声称任何真实模型采用此shortcut，也不形成
方法候选。若PASS，未来候选若依赖离散attribute/prototype/codebook，除unique random-key外还必须加入
frequency-matched random-cluster或等价label-permutation control。

## 禁止边界

- 不读official数据、pose、cache、checkpoint或sealed asset；
- 不导入torch、不初始化CUDA、不启动远端进程；
- 不创建exp404，不修改exp394–403任何文件；
- 不把toy性能写成ReID性能结果。
