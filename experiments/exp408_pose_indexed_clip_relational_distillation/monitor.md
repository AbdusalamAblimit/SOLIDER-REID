# exp408 PICRD 监控

> 当前：`DESIGN FROZEN / LITERATURE CONDITIONAL PASS / IMPLEMENTING / GPU NO-START`

## 2026-07-21：从 CAVT 切到直接训练对象

exp407 已按 `FORMAL SEALED-FAIL / VALIDITY FAILURE / SCIENCE NOT EVALUATED`封板，4090空闲。CAVT
不作科学否定，但连续三次消耗在测量合同后从活动主线移除；exp408禁止继续修donor/matcher。

代码审计确认旧rich路线的relation监督被`source_feature.detach()`和`hidden.detach()`双重阻断，且旧evidence
由全图GAP生成。近期论文/开源审计未发现PICRD整体同构实现，但pose+CLIP、part KD和batch relation原子高度拥挤，
故只定位为C类候选。最终冻结机制为逐槽跨batch relation、correct-vs-wrong/generic/zero训练内排序和Stage-2
直接反传；若退化为普通part KD则不运行。

下一步只实现必要代码和最小正反/梯度/CUDA-AMP检查，然后进入一次独立盲审。当前未创建cache、未启动GPU，
没有新增mAP/R1。

## 2026-07-21：设计盲审首轮1B/2H并修正

首轮审查发现四臂若各自使用valid集合，关系排序可能由缺失模式产生；另有负臂反传会造成循环论证、canonical
cache和e120 batch未冻结两项HIGH。设计现已修正为四臂共同`V_common`、control距离stop-gradient、固定
wrong offset=4、deterministic resize cache及16 PID×4图诊断manifest。等待代码实现后的独立盲审；GPU仍NO-START。

独立代码盲审首轮为`0B/1H`：processor外层AMP会让声明为FP32的Gram matmul/einsum重新落到FP16。已把
PICRD mask/pooling/relation整个块显式置于`autocast(enabled=False)`并把source/teacher转FP32；等待同一审查者闭环。

同一独立审查者已闭环为`0 BLOCKER / 0 HIGH`。本地语法/config merge、逐槽shape、共同valid、构造正例四臂顺序
和source非零有限梯度均PASS；没有启动GPU。下一步只做一次固定MMPOSE-ABU真实model CUDA/AMP update，然后立即
fresh生成cache并冻结SHA，不追加其它static。

固定MMPOSE-ABU真实batch64 CUDA/AMP检查PASS：PICRD块为FP32，`stage_grad_tensors=142`，首个
`base.patch_embed.projection.weight`梯度绝对和=`49.140007`，默认GradScaler=`65536`且一次optimizer真实更新。
四臂初值correct/wrong/generic/zero=`0.727728/0.727611/0.037120/0.725649`，generic明显更近，说明control
不是装饰性弱臂；训练必须实际扭转顺序。检查后GPU=`2 MiB/0%`。

首次cache-v1后台调用在顶层`from datasets...`立即因repo根未进入`sys.path`退出；log仅237 bytes，未读official/
pose、未初始化CUDA、未创建cache/diagnostic。v1目录冻结不复用。修复只在import前加入脚本解析出的repo root，
cache/config改用fresh `exp408-picrd-cache-v2`；等待同一独立审查者聚焦闭环后立即执行。
