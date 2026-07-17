# Post-PRCV Frozen-Feature 探索 —— 完整负结果 + 方法论教训（2026-06-24）

> 目标：post-PRCV 找一篇 CCF-B **方法稿**。资源：frozen Swin/SOLIDER（occluded_duke 73 / market 94.6 / CARGO 67）+ on-disk 数据（occluded_duke / market / MSMT / CARGO / AG-ReID.v2 / RSTPReid，**无视频**）。结论：**这条 frozen-feature + on-disk 路被穷尽地证负——6 个方法方向 cheap-kill 全死 + 视频 no-go + 唯一 analysis 发现严验证伪。零浪费训练。**

## 1. 六个方法方向（全部零训练 cheap-kill 死，未烧一次训练）
| # | 方向 | axis | re-frame | 死因 |
|---|---|---|---|---|
| 1 | **B 航拍包含** | accuracy 隐藏变量 | 航拍证据⊆地面（非对称信息包含） | 前提错：σ_aerial **<** σ_ground（TTA 方差反向），物理前提就不成立 |
| 2 | **GOPL** | accuracy 隐藏变量 | SMPL 共同可见度=positive 可靠性 | =occlusion-count 廉价代理；cov3d≈cov2d；坐实 SMPL 对 ReID 无独特信号（四连证） |
| 3 | **Gallery Hubness** | accuracy 隐藏变量 | gallery 负向 in-degree=残差失败拓扑变量 | remedy 被 k-reciprocal 完全占（+1.51 vs +10.98）；**且诊断本身后被证伪，见 §2** |
| 4 | **OSAC 谱过坍缩** | training 表示几何 | seen-ID 过坍缩压掉 unseen 可迁移证据 | 前提倒置：训练 effective rank **升**（55→68），身份信号在 top-PC 非谱尾，D1 反转，D2 +0.0000 |
| 5 | **RMA-TIReID** | text-image | 文本锚定到视觉身份流形（非局部对齐） | 边缘死：token-prototype≈color-only（非颜色 token 只值 ~1 mAP）；跨域流形弱（R1 49） |
| 6 | **Rank-Regret** | 效率/Pareto | cheap-vs-full 排名不一致路由算力 | 撞 CFPER（RI partial 控 #false 后≈0）+ Swin 无 cheap exit（算力集中 stage2 92%，cascade 省 1-5% 无意义） |

**视频（AG-VPReID tracklet）**：codex 判 no-go 8/10——temporal aggregation 成熟红海，AG-VPReID(CVPR25)/AG-VPReID.VIR(IJCB25)/VReID-XFD(2026) 占满，能廉价拿到的 re-frame 会被归类成 temporal attention，且需下载数据。

## 2. Hubness analysis 的完整起落（最值得记的一条）
- **诊断**：M(q)=Σ H_k 解释 AP 误差 rho+0.60（OD），D1 置换破、D3 控 norm/margin/camera/#pos 后仍 +0.60、D4 负向≠热门——**看起来是干净的真发现**。
- **failure-case 机制**：hub=非身份明亮场景（橙车）过度编码，跨 24 身份余弦抱团 0.166，27× kNN 富集——**像 mechanism-level 确证**。
- **★严验证伪（codex paper-review 后）**：
  - **P0 circular**：rho+0.60 含 query 自身对 H_k 的 self-loop，leave-one-query-out 后 +0.48，held-out split 后 **+0.33**（高估~半）。
  - **★P0c 决定性**：控一个 trivial 代理 `#false-in-topk`（=你自己 top-k 里错几个，**无需任何 hub/拓扑概念**）后，M(q) 偏相关塌到 **≈0（−0.06，两集一致）**；反向控 M 后 #false 仍 +0.51。**即"gallery 负向 in-degree / many-to-one 拓扑"相对"top-k 错的多少"无增量解释力。**
  - **P3 机制半塌**：去背景 H_k −47%，但去人 −87% 掉更多（与"非身份场景因子"预测相反）——hub 信号过半在人体 crop，非纯场景。
  - 残存 P4（高 M=k-reciprocal 修复最多 query，分箱 4.3×/4.5×）也可能只是 #false 代理。
- **教训**：原 D3 漏控了 `#false-in-topk` 这个最致命代理，导致一个 trivial 计数被包装成"拓扑诊断"。

## 3. 方法论教训（可复用）
1. **★任何 per-query 解释变量，对照必须包含 `#false-in-topk`/top-k 错误计数这个 trivial 代理**——它能解释大量 AP 误差且不需要任何机制概念，漏控它会让"诊断"假阳性。这是本 session 最硬的一条。
2. **零训练 cheap-kill 极其有效**：6 个方向全在动手前止损，零浪费训练。每个 re-frame 必须有一个"若失败则推翻叙事"的零训练诊断。
3. **frozen-feature 隐藏变量反复被成熟 test-time 工具碾压**：k-reciprocal/camera-aware re-ranking/hard-negative 覆盖了 retrieval/topology-side 的明显隐藏变量。新 accuracy 隐藏变量极难逃脱。
4. **SMPL/人体几何对 ReID 无超出 occlusion-count/视角的独特信号**（exp333/poseCLIP/SMPL-anchor/GOPL 四连证）——别再投。
5. **架构约束 kill 方法**：Swin 算力集中末段前一 stage，没有早退点——效率/early-exit 方法需 ResNet/多-exit 网络。
6. **背景/区域抑制是团队旧雷**（PSG/pose-mask 在强系统无价值）——failure-case 指向它时即死路。

## 4. 战略结论 + 建议
**现有资源（frozen Swin + on-disk image/text + 无视频）下，frozen-feature 找 method/diagnosis 的路已穷尽证负。** 要继续找 CCF-B 方法稿，只能上**新资源**（用户决策）：
- (a) **新数据/模态**：视频（AG-VPReID，但红海）/ 引入外部 FM（MLLM/DINO，但 [[fm-import-occluded-reid-closed]] 已证负）；
- (b) **换架构**：多-exit 网络做 efficiency 方法（Rank-Regret 的死因是 Swin 无 cheap exit，换架构可能复活——但需训练 + novelty 仍 7/10 偏弱）；
- (c) **换问题/项目**：这条 post-PRCV frozen-image 线证负到底；
- (d) 回审 CARGO OVLI 线（但 MaxSim<avg 的 headline、OVP 撞 CMPC 是已知死点）。

**不建议**：在现有 frozen-feature + on-disk 上再找 accuracy 隐藏变量（穷尽证负）；烧训练硬撞已 cheap-kill 否掉的方向。
