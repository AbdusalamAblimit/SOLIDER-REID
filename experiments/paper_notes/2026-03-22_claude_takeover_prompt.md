# 2026-03-22 Claude 夜间接手提示词

把下面整段直接发给 Claude：

```text
你现在接手仓库 `/root/work/SOLIDER-REID` 的研究工作。

先完整阅读并严格遵守：
- `/root/work/SOLIDER-REID/AGENTS.md`
- `/root/work/SOLIDER-REID/CLAUDE.md`

然后先读：
- `/root/work/SOLIDER-REID/experiments/results.md`
- `/root/work/SOLIDER-REID/experiments/decisions.md`
- `/root/work/SOLIDER-REID/experiments/innovation_brainstorm.md`
- `/root/work/SOLIDER-REID/experiments/paper_materials/story.md`
- 当前最新实验的 `design.md / monitor.md`

你的工作方式：
1. 默认自主推进，不等我确认
2. 先读文档，再监控当前实验，再做决策
3. 先写设计，再改代码，再广范围审查，再启动训练
4. 每个实验都必须打足够多的行为日志，方便及时止损
5. 每个实验结束后，先补文档，再开下一个
6. 整晚尽量让本地和远程两台机器都在工作
7. 但两台机器不能跑几乎一样的东西，必须是两个真正不同的创新点，或同一主问题下两个强对照

当前方向约束：
1. 不要回到已经做烂的小模块堆叠路线
2. 不要把 scorer/gate/context 小修补继续当默认主线
3. 不要把 visibility、小 attention、小 GCN、小 recipe 调参重新当主创新
4. 当前必须继续记住 `exp109` 的核心发现：
   - 真正的 gap 仍然是 `single-image support incomplete`

当前最新状态：
1. `exp148 PCVT` 是当前最值得继续追的主线
   - 核心是：用 pose-defined complementary pseudo-views，把单图改写成“伪多 support 学习对象”
   - 它不是 retrieval trick，也不是 feature completion 小残差
   - 如果它还在跑，优先继续监控、记录、收口
2. `exp149 SCFA` 已快速判负
   - 不要继续在它上面补变体
   - 远程如果空闲，下一条实验必须是与 `PCVT` 真正不同的大方向

你必须特别遵守：
1. compact 后不要根据旧记忆恢复主线，必须重新读最新文档
2. 每个新实验都要做广范围 Claude 审查
   - 不只审代码 diff
   - 还要审：
     - 想法是否真的算新方向
     - 是否只是旧机制换名
     - train/test 是否对称
     - AMP 是否安全
     - 默认行为是否安全
     - 日志是否足以支撑及时止损
3. 没有 `claude_review.md`，不得启动训练

远程服务器连接方式：
```bash
sshpass -p 'aZKBF3qdSS59Wf4uveVQgEwWAtHAwbeg' ssh -p 29162 -o StrictHostKeyChecking=no root@i-2.gpushare.com
```

远程项目目录：
```bash
/root/work/SOLIDER-REID
```

工作目标：
1. 持续监控当前实验
2. 主动做决策：继续 / 止损 / 收尾 / 切新方向
3. 一旦某台机器空下来，马上设计下一条真正不同的新机制
4. 继续逼近一个足够支撑 B 类论文主贡献的核心创新点
```
