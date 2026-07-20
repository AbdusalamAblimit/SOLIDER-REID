# post-exp403 CPU诊断：随机source-key假所有权

> 状态：`DESIGN / CPU ONLY / NO EXP404 / GPU FORBIDDEN`

## 动机

exp402/403要求correct evidence优于wrong、generic与NULL，但这种顺序可能由“evidence是否来自当前RGB”的
checksum与“它是否像真实sample code”的validity detector制造，而不需要任何support/appearance语义。若随机
source key也能通过同类顺序，后续机制不能只以该顺序声明semantic ownership。

## 核心假设

构造与PID、camera和语义独立的逐样本随机key。descriptor保留identity payload，同时用两个非语义量控制固定
identity预算：

1. supplied key是否与host key exact匹配；
2. supplied key是否属于真实sample-key集合，而非generic/NULL。

若correct、wrong-real-key、generic/NULL仍形成强检索顺序，而随机key在same-ID与different-ID间没有几何差，
则说明现有反事实门存在source-authentication假阳性。

## 技术方案

纯CPU生成128个identity prototype、每ID一个query和两个gallery。每个sample独立获得16维单位随机key，key与
PID独立。输出descriptor为identity prototype与deterministic mismatch nuisance的正交直和；identity预算
`q`只由exact source match和real-key validity决定：

```text
q = 0.08 + 0.37 * is_real_sample_key + 0.55 * is_exact_host_key
descriptor = [q * identity, sqrt(1-q^2) * mismatch_nuisance]
```

四臂为correct、same-camera different-PID donor key、全体key均值generic和all-zero NULL。再随机置换所有
sample-key assignment，验证结论不依赖key语义。negative mutant把所有臂的`q`固定为`0.45`，必须无法稳定通过
顺序门。

## 预期解释

该诊断只证明一种existential shortcut，不声称真实exp402/403使用了它，也不形成方法候选。若PASS，未来候选除
原有controls外还必须保留semantic-blind random-key或等价null-semantic control；否则强顺序仍不足以证明语义。

## 禁止边界

- 不读official数据、pose、cache、checkpoint或sealed asset；
- 不导入torch、不初始化CUDA、不启动远端进程；
- 不创建exp404，不修改exp394–403任何文件；
- 不把toy性能写成ReID性能结果。
