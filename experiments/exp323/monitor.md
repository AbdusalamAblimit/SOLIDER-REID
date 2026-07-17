# exp323 monitor — Qwen2.5-VL-3B 视觉裁剪 A/B（lab-3090-d）

## 任务1：姿态视觉裁剪预处理（已完成）

**坐标映射确定方式**：读 jpg 实际尺寸发现 query `0525_c4_f0066009.jpg` 是 53×131，
npz `keypoints` 的 x∈[6.5,43.1]、y∈[40.6,126.7] 直接落在 [0,W]×[0,H] 内
→ **keypoints 已经在原图 jpg 像素空间，无需任何变换**。叠加 overlay 到放大 4× 的 jpg 上，
红点（head/torso 关键点）精确落在目标人脸/肩部，绿框正确框住目标人（排除上下其他人）→ 映射确认。

**人物选择（_pN 歧义）**：每张图有多个 `_p{N}.npz`（多人检测）。选 `visibility_binary` 与
pair 的 `query_visible_keypoints` 完全匹配的那个；多个匹配（14 例）时取 p0（置信度最高的主检测）。
全 288 对中 p0 直接匹配 286/288，任一候选匹配 288/288。

**裁剪**：query 用 `query_part_visibility` 的可见部位关键点求包围盒 +15% pad；
gallery 用同一部位集合、各自的关键点。单部位（如仅 head）会塌成 ~6px 小框 →
加 `MIN_SIDE=24` 最小边，对称扩张保证 VLM 可读。

**结果**：288/288 对成功裁剪，双图全部存在。
- query 全图 fallback 32 例 = 正好是 n_visible=0（无任何可见部位）的 32 对，合理。
- gallery 全图 fallback 3 例（gallery 该部位无关键点）。
- 加 MIN_SIDE 后 tiny crops（<8px）从 43 降到 0。
- 数据集均衡：n_visible 0-8 各 32 对；gt_same 144 同 /144 异（chance=50%）。

## 任务2：Qwen2.5-VL-3B A/B（进行中）

环境：系统 `python3`，torch 2.7.1+cu118，transformers 5.2.0，已装 qwen-vl-utils + accelerate(1.14.0)。
**踩坑**：任务说"模型已下"，实际 3B 缓存只有 2.8G 且 safetensors 是 `.incomplete`
（另一并行会话的 snapshot_download 还在拉，同时还在拉 7B 竞争带宽）。
首个 smoke test 因权重不全卡死（RSS 624M、GPU 0%）→ kill，等下载完成。
下载 ~1.5MB/s，7.51GB 总量，待补全后再跑全量。

### 三条件
- 甲(jia)：原始全图 + base prompt
- 乙(yi)：原始全图 + base + 可见/遮挡部位文字
- 丙(bing)：任务1裁剪图 + base prompt

## 任务2 最终结果（2026-06-16，run 84s/288对×3条件）

3B 下载补全后 model load 15.9s，推理 ~0.29s/对，全程 84s。

**核心结果：三条件全部 = 50.0% 准确率（=随机基线，144/144 均衡集）。**

| 条件 | 准确率 | YES | NO | UNK |
|------|--------|-----|----|----|
| 甲(裸) | 0.500 (144/288) | 0 | 288 | 0 |
| 乙(文字) | 0.500 (144/288) | 2 | 286 | 0 |
| 丙(视觉裁剪) | 0.500 (144/288) | 0 | 288 | 0 |

按 n_visible 0-8 分档：**每一档都恰好 0.500**（乙在 n_vis=2/3 档因 2 个 YES 各 ±0.03，net 0）。
heavy(≤4) vs light(≥5)：全 0.500。

**原因诊断（scripts/exp323_diag.py，12 个高可见对探针）**：
Qwen2.5-VL-3B 在"一个词 YES/NO"格式下有**压倒性 NO-bias**——
- forceB（强制"必须选、挑更可能的"）：12 对全 NO（含 6 个明显同人 pid 全可见），0 个 YES。
- reason（允许先推理）：同人对开始出现 "appear to be the same person..."，
  说明模型**并非完全不能区分**，但 80 token 内多被截断没到 ANSWER；约束成一个词就塌成 NO。

**结论**：3B 的 always-NO 退化使 A/B/C 对比**不可判**（三条件都被钉在 50%）。
视觉裁剪(丙)和文字(乙)相对裸图(甲)**均无提升**，但这是小模型 NO-bias 导致的天花板/地板效应，
不是方法本身被证伪。对照：同 288 对 GPT-5.5 裸=55.9% 文字=55.6%（文字也无效）。

**踩坑**：(1) 任务称"模型已下"实际 3B 缓存只有 2.8G/.incomplete，
另一会话并行拉 7B+3B 抢带宽（hf-mirror ~1.5MB/s），等下载补全花了 ~1h。
(2) scripts/ 非 package，从 /tmp 或 scripts/ 内跑需 sys.path 插 repo root。

**建议下一步**（若要救信号）：换 reasoning 输出格式（先推理后 ANSWER:，max_new_tokens≥128）
重跑 A/B/C，或换更大的可部署模型（7B 已下好）看 NO-bias 是否缓解、裁剪是否显出增益。

## 任务2 补充：reasoning 输出格式 A/B/C（exp323_qwen3b_reason，1792s/288对×3）

为解一词格式的 always-NO 地板效应，改用"先简要推理后 ANSWER: YES/NO"格式（max_new_tokens=128）重跑。
模型不再 always-NO，给出多样判定 → A/B/C **可判**。

**最终结果（UNK 计为错）：**

| 条件 | reasoning acc | YES | NO | UNK | vs 甲 |
|------|------|-----|----|----|----|
| 甲(裸) | **0.542** (156/288) | 154 | 134 | 0 | — |
| 乙(文字) | **0.493** (142/288) | 166 | 118 | 4 | **-4.9pt** |
| 丙(视觉裁剪) | **0.358** (103/288) | 108 | 109 | **71** | **-18.4pt** |

按 n_visible（甲/乙/丙）：n=0..8 无任何档显示丙>甲。丙在 n_vis=3/4/5/7 跌到 0.22-0.28。
heavy(≤4)：甲0.525 乙0.525 丙0.375；light(≥5)：甲0.562 乙0.453 丙0.336。
**增益未集中重遮挡，两个 treatment 一致变差。**

**为什么丙(裁剪)最差**：raw_bing 多为 "To determine... I will analyze... Clothing Color: - Image 1 shows..."
——裁剪去掉上下文后模型对每个碎片**长篇描述**，128 token 内常没到 ANSWER → 71 个 UNK。
裁剪不是让匹配更容易，而是**删了语境、把 token 预算耗在描述碎片上、判定更难收敛**。

**最终结论（kill-switch 信号明确）**：
- 文字 grounding（乙）对 3B **无帮助甚至有害**（一词 +0，reasoning -4.9pt）。
- 姿态视觉裁剪（丙）**显著有害**（reasoning -18.4pt，含 71 UNK）。
- 裸图（甲）reasoning 54.2%，接近 GPT-5.5 裸 56.5%——3B 裸任务并不显著更差，
  问题在于**两个 pose-guided 干预都不 work**，且裁剪明显伤害。
- 红线#6（增益须集中重遮挡）不满足：增益根本不存在，变差还更均匀/更全面。
→ "frozen 小 MLLM + pose 视觉裁剪/文字提示" 这条廉价首验**不正向，建议砍**（转 exp324 或换机制）。
（注：这是 frozen-MLLM zero-shot 的结论；若要继续 MLLM-reasoner 线，需 LoRA 微调让模型学会用裁剪/grounding，
但 frozen 证据已偏负，沉没成本警告。）
