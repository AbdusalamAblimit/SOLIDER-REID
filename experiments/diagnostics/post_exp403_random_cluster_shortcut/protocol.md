# 频率匹配random-cluster CPU诊断协议

## 冻结执行

- Python只能通过仓库uv环境执行，`CUDA_VISIBLE_DEVICES=''`；
- seed=`20260720`，identity=`128`，query/identity=`1`，gallery/identity=`2`；
- cluster=`8`，每cluster严格`48`个sample；identity dim=`64`，nuisance dim=`128`；
- 原始平衡随机assignment与一次独立频率保持置换各执行一次；
- 不访问GPU、网络、数据集或模型资产。

## 正合同

原始与置换执行必须同时满足：

1. correct mAP `>=0.99`；
2. `correct - wrong >=0.05` raw mAP；
3. `wrong - max(generic,NULL) >=0.05` raw mAP；
4. 每cluster count exact=`48`且覆盖至少`40`个PID和两个camera；
5. same-ID与matched different-PID donor的cluster碰撞率绝对差 `<0.10`；
6. donor全部same-camera、different-PID且无fixed point；
7. 所有descriptor finite、L2 norm误差 `<1e-10`。

## 反合同

令所有臂`q=0.45`且supplied cluster不再进入nuisance后，不允许同时通过correct floor与两个mAP margin。
mutant若仍通过则诊断无效。

## 裁决

- 正合同两遍PASS且mutant被抓：`FREQUENCY_MATCHED_RANDOM_CLUSTER_FALSE_SEMANTICS_DEMONSTRATED`；
- 否则：`DIAGNOSTIC_INCONCLUSIVE`。

无论结果如何均保持`NO EXP404 / GPU NO-START`。
