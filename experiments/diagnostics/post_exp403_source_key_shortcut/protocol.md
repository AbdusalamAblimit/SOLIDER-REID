# 随机source-key CPU诊断协议

## 冻结执行

- Python只能通过仓库uv环境执行；
- seed=`20260720`，identity=`128`，query/identity=`1`，gallery/identity=`2`；
- identity dim=`64`，key dim=`16`，nuisance dim=`128`；
- 原始key assignment与一次固定随机置换各执行一次；
- 不访问GPU、网络、数据集或模型资产。

## 正合同

原始与置换执行必须同时满足：

1. correct mAP `>=0.99`；
2. `correct - wrong >=0.05` raw mAP；
3. `wrong - max(generic,NULL) >=0.05` raw mAP；
4. random-key same-ID与different-ID cosine均值绝对差 `<0.05`；
5. donor全部same-camera、different-PID且无fixed point；
6. 所有descriptor finite、L2 norm误差 `<1e-10`。

## 反合同

关闭source-match/realness ownership、令所有臂`q=0.45`后，不允许同时通过上述两项mAP margin。mutant若仍
通过则诊断无效。

## 裁决

- 正合同两遍PASS且mutant被抓：`RANDOM_SOURCE_KEY_FALSE_OWNERSHIP_DEMONSTRATED`；
- 否则：`DIAGNOSTIC INCONCLUSIVE`。

无论结果如何均保持`NO EXP404 / GPU NO-START`。
