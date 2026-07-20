# 随机source-key CPU诊断协议：execution v2

## v1封板与唯一修正

execution v1在metric前因query slot解析错误封板为`SEALED-INVALID`。v2仅修正donor lookup：

```text
query slot   = "q"
gallery slot = replica index
donor        = same camera + same slot + next PID modulo 128
```

v1脚本保持不变；v2通过独立入口覆盖该函数。除此之外完全继承`protocol.md`的seed、维度、四臂、正反门槛和
GPU/data禁止边界。v2是唯一允许的fresh修正执行，不补跑v1。
