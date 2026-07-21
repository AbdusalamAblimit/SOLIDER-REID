# exp410 PC²P 独立代码盲审

## 审查范围

- `design.md / protocol.md / monitor.md`；
- `config/defaults.py` 与 exp410 独立 config；
- `model/pose_complete_clip_proxy.py`；
- `model/make_model.py`、`processor/processor.py` 的 PC²P 接线；
- fresh bank builder 与真实 PK batch64 CUDA/AMP 合同脚本。

审查者未参与实现、未修改文件、未使用 GPU，只报告 `BLOCKER/HIGH`。

## 结论

`0 BLOCKER / 0 HIGH`。

审查确认：

1. default-off 不增加 state 或 RNG 消耗，合同包含同 RNG 的真实 forward/loss exact；
2. bank 的 PID 行顺序、official 路径、逐图 RGB 绑定、source provenance、SHA、shape、dtype、单位范数、
   五槽支持和行唯一性闭环，builder 与 loader 字段一致；
3. PC²P 与 PICRD/PCHM/semantic/SPK/ELO-CUR/hierarchical 路径互斥；
4. proxy 只是 processor 持有的冻结外部 tensor，不进 model state 或 optimizer；PC²P 分支不调用原 classifier；
5. `F.linear` 在 autocast 关闭区以 FP32 执行，CE 梯度合同覆盖 BNNeck、`base.norm3` 与 Stage-3；
6. eval 不接收也不加载 bank，仍返回原 global descriptor；
7. Torch 1.13 API 与 NumPy void-row 唯一性写法兼容。

因此实现允许进入 fresh bank 构建与唯一真实 batch64 CUDA/AMP 合同；合同通过后不再追加 static 测试，立即启动
唯一 fresh seed1234/e120 `correct` arm。
