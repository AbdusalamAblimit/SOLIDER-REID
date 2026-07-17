# 实验 exp323: MLLM-as-Occlusion-Reasoner — pose-grounding A/B 廉价首验

> **来源**：post-PRCV「搬范式」路线（CLIP-ReID/Pose2ID 打法）。`paradigm-import-survey` 排名 #1。
> **性质**：**推理-only 廉价 kill-switch 首验**（frozen MLLM，零训练）。不正向就砍，转 exp324。
> **机器**：lab-3090-d（exp255 ckpt + Qwen2.5-VL-7B via hf-mirror + RTX 3090 idle）。

## 动机

- **搬范式**：重量级 ReID 成果的规律是"第一个把外部新范式搬进 ReID"（CLIP→CLIP-ReID、扩散→Pose2ID）。当前 VLM 社区**最热的"已知短板"**：CAPTURe(ICCV'25)/O-Bench/amodal-completion 一整批 benchmark 刚系统实锤 **MLLM 的遮挡/amodal 空间推理远低于人类、scaling+thinking 都补不上**。ReID 侧无人把这缺口和 occluded ReID 连起来。
- **我们独占的角度**：项目的 pose visibility + 5-part LGPA **正是那些 benchmark 证明 MLLM 自己做不到的 amodal grounding**——直接告诉 MLLM"哪些部位可见/被挡"。这把"又拍个 MLLM 上 ReID"的 me-too，变成**"先证缺口、再用 pose 修复"的问题级新定义**。
- **为什么先做这个廉价首验**：整条线的命脉是一个可证伪假设——"给 MLLM pose-visibility grounding 能显著改善它对遮挡 pair 的同人判定"。1-2 天近零成本先证伪，省整个 GPU slot。

## 核心假设

给 Qwen2.5-VL 注入 pose-visibility grounding（"query 中头/躯干可见、腿/脚被遮挡，只比可见部位"）后，它在**重遮挡** ReID pair 上的同人/异人判定准确率，**显著高于**裸 prompt（只给两张图问"同人?"）。且增益**集中在重遮挡子集**，而非整体均匀——证明收益来自遮挡推理而非通用判别。

## 技术方案（inference-only，零训练）

数据/格式已核实：`data/occluded_duke/pose_data/{query,gallery}/<img>.npz` 含 `visibility`(17,)、`visibility_binary`(17,)、keypoints；`visibility_summary.json` 已聚合。COCO-17 keypoint。

1. **重遮挡子集定义**：每张 query 的可见关键点数 = `visibility_binary.sum()`。取**可见数最低的子集**（如底部 30%，或 visible≤8，复用 exp109 分桶口径）作 heavy-occ query。
2. **候选 pair 构造**：用 exp255 ckpt 跑 `eval_fliptest_maxsim.py` 得每个 query 的 MaxSim ranking → 取 top-K(=10) gallery 候选（天然混 same/diff GT，且是"模型已经分不清"的难例）。GT same/diff 由 pid 给出。
3. **Qwen2.5-VL A/B**（同 pair、同模型、只换 prompt）：
   - **Prompt A（裸）**：两图输入 → "Are these the same person? Answer yes or no."
   - **Prompt B（pose-grounded）**：同两图 + 由 query visibility_binary→body-part 映射生成的 grounding："In the first image, visible parts: {head, torso}; occluded: {legs, feet}. Compare ONLY the mutually-visible parts. Are these the same person?"
4. **指标**：
   - 主：heavy-occ pair 上 A vs B 的同人/异人**判定准确率**（vs GT）。
   - 副：MLLM 判定当 re-ranker，heavy-occ 子集 rank-1/mAP 变化（A vs B vs 原 MaxSim）。
   - 红线#6 防御：**heavy-occ 子集增益 vs 整体增益**——增益必须集中在重遮挡组。
5. （可选）GPT-4o API 做 oracle 上界——若该机器无 OpenAI 访问则跳过。

## 预期结果

- 假设成立：B 准确率显著 > A（重遮挡组，期望差值 ≥ 5-10 个百分点），且 heavy-occ 增益 >> 整体。→ 缺口真实 + pose 能修 → 重量级信号成立 → 上 LoRA verifier + 全量评测。
- 失败最可能原因：(1) 低分辨率监控行人图上 MLLM 判定噪声大，A/B 都烂；(2) pose grounding 不改变判定（MLLM 不听 grounding 或本就用可见部位）；(3) 增益均匀分布（→ 通用重排，撞红线#6）。任一即砍，转 exp324。

## 对照组

- baseline = Prompt A（裸 MLLM 判定）。treatment = Prompt B（pose-grounded）。单变量 = prompt 是否含 pose visibility，其余全同。
- sanity：随机 same/diff pair 上 A/B（确认 MLLM 基础判别 non-trivial）；整体 vs heavy-occ 子集切分（红线#6）。

## 设计修正（2026-06-16，用户 methodological 指正）

**强模型当 baseline 不对**：GPT-5.5（codex）太强，可能自己已补上遮挡缺口 → B≈A → **假阴性**（天花板效应，不能 kill idea）；且方法真正部署/微调/蒸馏的是**小开源 MLLM**，GPT-5.5 在方法里只能当 oracle/teacher。

修正后角色：
- **被测主体 = 小开源模型**（Qwen2.5-VL-**3B**/7B）——遮挡缺口真实存在、且是可部署 regime。**决定性 kill-switch 在这里**。
- **GPT-5.5（codex）= oracle 上界**（已在跑，重定位；还回答"前沿模型是否也有残余缺口"的战略问题）。
- **做成 size→收益梯度**（3B 收益最大 > 7B > GPT-5.5 最小）：证明"pose grounding 补的正是小模型遮挡缺口"，把"强 baseline 问题"转成卖点。可部署 ReID 系统恰在小模型 regime。

**战略警告**：MLLM-reasoner 有点在和前沿赛跑（模型越强缺口越小，卖点随时间衰减）→ 方法须落到"蒸馏进小可部署网络"；exp324 DINO-correspondence 更 frontier-independent。梯度实验顺带量化衰减速度。

## Kill-switch / 下一步

- **小模型（3B/7B）B 显著 > A 且集中重遮挡** → 真信号 → exp323b：LoRA 微调小 Qwen verifier + 全量 re-rank + 可控遮挡(PLBOA)消融 + GPT-5.5 蒸馏 teacher。
- 小模型也 B≈A → 砍，转 exp324（DINO-correspondence），不沉没成本。（GPT-5.5 单独 B≈A 不算 kill，可能天花板。）
