# exp412 PSGC 独立设计盲审

## 首轮结论

独立审查结论为 `0 BLOCKER / 3 HIGH`。审查确认 PSGC 是新的 pose×CLIP backward-routing 对象，而不是恢复或
修改 exp411 已失败的 owner multiplicity；三个 HIGH 均要求收紧机制边界，不要求增加调参或重复训练前测试。

## HIGH 与闭环

1. **AMP dtype 可能破坏 forward exact**：若 FP32 路由场直接乘 FP16 feature，结果可能提升为 FP32。设计与实现
   已固定 `G.detach().to(X.device, X.dtype)`，接点只允许在 `norm3` 后、`avgpool` 前的 descriptor 分支；合同检查
   dtype、`torch.equal(X_route,X)`、score/feature/全部 loss exact。
2. **缺 q-only 联合归因控制**：pose-only 与 text-shuffle 不能排除正确 CLIP 标量单轴已足够。已新增 q-only，
   front 内 visibility 置常量而保留正确 q 和相同 pose field；科学 GO 要求 correct 同时严格胜 pose-only、q-only
   与 text-shuffle 的 raw mAP/R1。
3. **“梯度守恒/转移”表述过强**：`sum w=4`不能保证不同 feature field 与上游梯度下的真实梯度范数守恒。
   全文已收紧为“路由系数预算守恒/系数重分配”，公式不变且不再作真实梯度量守恒主张。

## 最终授权

三项 HIGH 已在设计和实现边界中闭环，当前=`0B/0H / IMPLEMENTATION AUTHORIZED / GPU NO-START`。后续独立
代码盲审仍只拦截致命接线、dtype、默认关闭或变量混淆问题，不扩展为多轮 preflight。
