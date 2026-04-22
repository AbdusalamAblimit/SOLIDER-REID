# 实验协议

## 实验命名规范

```
exp{NNN}_{描述}  例: exp001_baseline, exp153_new_method
```

### 同一实验的变体命名

同一个实验的不同变体（超参调整、消融、重跑、环境变更等）**必须**用字母后缀区分，不得分配新实验号：

```
exp245a  — 第一个变体
exp245b  — 第二个变体
exp245g  — 第七个变体（按字母顺序递增）
```

**判断标准**：如果两个实验的核心方法相同，只是配置/种子/环境/超参不同，就属于同一实验的变体。

**文档要求**：
- 所有变体共享同一个 `experiments/exp{NNN}/` 目录
- monitor.md 中按变体分节记录
- results.md 中用 `exp{NNN}{x}` 标识各变体

**反例（已发生的问题）**：exp245 系列曾产生 245/245b/245c/.../245h_v2 等大量子实验名，
但 log 目录和文档分散，造成混乱。今后严格遵守此规范。

每个实验使用独立 `OUTPUT_DIR`: `./log/occluded_duke/{实验名}`

## 硬性 CLI override 默认

**所有训练启动**必须加 `TEST.IMS_PER_BATCH 64` (无论机器)。默认 256 在 Market (22k+ test images) / Occ-PTrack (多样本) 下 flip-test TTA 会触发 OOM, 即使 24GB 4090/3090 也不例外 (80 epoch fragmentation 累积)。

已发生过两次 OOM 因此规则 (2026-04-22 exp292 lab3090 e20 / exp293 lab4090 e80)。今后新训练命令模板:
```
python train.py --config_file xxx.yml \
  SOLVER.SEED NN \
  TEST.IMS_PER_BATCH 64 \
  OUTPUT_DIR ./log/.../expNNN \
  <其他 override>
```

## 实验设计文档（design.md）

每个实验**启动训练前**，必须先写 `experiments/exp{NNN}/design.md`：

```markdown
# 实验 exp{NNN}: {实验名称}

## 动机
- 为什么做？解决什么问题？基于哪些前序实验/论文？

## 核心假设
- 一句话描述

## 技术方案
- 修改了哪些文件？新增了哪些模块？
- 数据流（从输入到输出）
- 关键超参数及选择依据

## 预期结果
- 假设成立时预期 mAP/R1 变化
- 如果失败最可能的原因

## 对照组
- 对照的 baseline 实验
- 消融变量（只改一个）
```

## 代码原则

- **插件式设计**：新模块在 `models/modules/` 下，通过 config 开关控制
- **不破坏 baseline**：默认 config 必须能复现 baseline
- **单变量**：每次实验只改一个变量
- **可复现**：config 文件、随机种子、commit hash 必须记录

## 代码提交

```bash
git commit -m "exp{NNN}: {简短描述}"
```

## 实验审查制度（Claude Broad Review）

每个实验**启动训练前**，必须做广范围 Opus 4.6 子代理审查。

### 审查范围（缺一不可）

a. `design.md` — 合理性、单变量原则、假设是否清晰。**特别注意：如果实验只改了配置参数或几行代码，审查必须质疑"这是否只是小调参？是否在逃避真正的创新？"**
b. 所有新增/修改代码 — 逐行阅读，检查 bug、runtime error
c. 配置文件 — config 选项是否正确引用
d. `config/defaults.py` — 新默认值是否安全、是否破坏已有实验
e. processor — loss 计算、特征提取、评估逻辑
f. 与前序实验的对照 — 消融变量隔离性

### 审查行为要求

- **逐行阅读**所有新增/修改代码
- **逐行对比**配置文件与对照组差异
- **验证数据流**：forward/backward pass 每一步
- **检查边界**：None、设备/dtype/shape 不匹配
- **检查优化器**：新参数是否被正确加入
- **检查交互**：是否影响已有实验可复现性
- 发现可疑之处必须标记，宁可误报不可漏报

### 流程（双审查制：Claude + Codex）

**第一轮：Claude Broad Review（Opus 子代理）**

1. 审查 → 发现问题 → 主 agent 修改代码
2. 重新审查 → 直到明确通过
3. **二次审查的范围与一次审查完全相同**，不是只审查修复的问题。每轮审查都是完整的全范围审查。
4. 结论记录在 `experiments/exp{NNN}/claude_review.md`（或 `claude_review_v{K}.md`）
5. 审查分级：Critical / High / Medium / Low，所有级别都必须修复

**第二轮：Codex Review（GPT 代码审查）**

6. Claude 审查通过后，运行 `/codex:review` 对 git diff 做独立代码审查
7. Codex 输出结构化审查结果（verdict + findings with severity）
8. 将 Codex 审查结果保存到 `experiments/exp{NNN}/codex_review.md`
9. 如果 verdict 不是 `approve`：修复所有 findings → 重新运行 `/codex:review` → 循环直到 approve
10. **两层审查都通过才能训练**（hook 同时检查 claude_review 和 codex_review）

### Codex 审查结果保存格式

```markdown
# Codex Review — exp{NNN}

**Verdict**: approve / needs-attention
**Date**: {YYYY-MM-DD HH:MM}
**Review round**: {第几轮}

## Findings
{从 /codex:review 输出直接复制}

## 结论
{verdict: approve 时写} codex 审查通过
```

### 硬性阻断条件（hook 检查全部）

- `design.md` 存在
- `claude_review.md` 包含"审查通过"且 ≥30 行
- `codex_review.md` 包含 "verdict.*approve" 或 "codex 审查通过"
- **严禁未通过双审查就启动训练**

## 新实验默认风格

1. 先写设计，再改代码
2. 日志必须够重（能观察模块是否工作、塌缩、过强/过弱）
3. Agent 审查必须广范围（审想法新颖性、train/test 对称、AMP 安全、日志充分性）
