# exp400 rich-budget final production preflight协议

## 状态

`PROTOCOL-FROZEN / STATIC-CPU SEALED-PASS / CUDA FRESH-EXECUTION GO /
FORMAL NO-START`

## 上游封板

- exp399=`BASELINE_RELATIVE_STEADY_STATE_PASS / PRODUCTION PREFLIGHT GO`；
- result/runner/manifest SHA=
  `d5255fced4553c6d4669ce11a1644e1495340a590ee76e54f22139f547cb9cca`/
  `d5255fced4553c6d4669ce11a1644e1495340a590ee76e54f22139f547cb9cca`/
  `b719b3acdec3746dae8f602fc526564a08047ae5ad1a9e2c3a3865a973c2b12e`；
- exp394–399保持sealed；exp400不得加载exp399内存更新或任何checkpoint。

## 训练段

逐字保持exp399：32 attempts、e1/e6各16、tail8、同一materialized loader、matched RNG、D0/rich fresh
model/optimizer/default scaler、15组report、named state SHA、无scheduler/checkpoint。

## terminal段

rich final CPU state封存后：

1. 每个descriptor variant先strict restore final state与saved RNG；
2. monkeypatch仅用于diagnostic selective bypass，并在finally恢复；
3. epoch1比较full/all-bypass，epoch6比较full/all-bypass/bypass0/bypass1；
4. fresh reloaded model strict load同一state；
5. eval correct/shuffle/None/ExplodingPose/reloaded，要求RGB-only exact；
6. state、RNG、hooks、source/assets、teacher/codebook和finite终审；
7. 不保存state，不授权通过失败结果续训。

## static/CPU

连续两遍必须证明：exp399全部35项门不退化；terminal函数AST含strict load、finally restore、两个独立
bypass、ExplodingPose、RGB-only exact、teacher-free/finite/source/asset/state门；toy router/model验证
rho0 identity、两个consumer非零、selective bypass恢复和strict reload。CUDA未初始化且result byte-exact。

## formal授权

只有actual trajectory、validity、terminal三组门全部PASS，result内`formal_training_authorized=true`，才
直接启动唯一fresh e120。static PASS或partial actual PASS均不授权。
