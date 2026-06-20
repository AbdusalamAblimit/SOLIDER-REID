# 实验 exp355: PGPD (Pose-Guided Prompt-Prototype Distillation)

## 动机
20-codex 调研剩余最干净的弱赌注(file14)。PC-SOR(空间归属)已死。PGPD 走**训练端蒸馏**, 不碰空间、不进 CLIP 对齐(避开吸收), 也不外挂描述子(避开冗余)。
核心赌注: **CLIP-ReID 的 ID-prompt simplex 里藏着"类间邻域几何"(这张图离哪些别的 ID 近)——这是 ID-CE/triplet 给不了、LGPA 部位分支也表达不了的软信息。** 用 pose 选一个"更完整的同 ID teacher", 把它在 prompt simplex 上的暗分布(对其他 ID 的相似度)蒸馏给遮挡 student → 遮挡 student 继承完整视图的干净 ID 边界。

## 为什么可能避开今晚的坑
- **不被吸收**: 不在 CLIP i2t/t2i 对齐里加 pose 通路。pose 只做 (a) 选 teacher (b) 构造遮挡 (c) GRL 去姿态。蒸馏目标是 prompt simplex 上的软分布(已有量), 无新带参 ID 通路。
- **不冗余**: 暗知识 = 类间邻域几何, ID-CE/triplet(只推正类/拉负类硬标签)和 LGPA(部位判别)都不提供这个软关系。
- **测试端零开销**: 无 test-time pose, 无架构改, 描述子还是 global。

## 核心机制(数据流)
基于 exp341(CLIP-ReID 可学习 ID prompt + i2t/t2i SupCon, Swin-Tiny)。新增:
1. **pose 完整度**: `comp_i = target_heatmaps[i].amax(dim=(1,2)).sum()`(每图**目标人**可见关键点强度和, detach)。**用 target_heatmaps 非 scene(max-merge 多人)**——否则多人遮挡图里干扰者会虚高完整度误导 teacher 选择(Codex Medium)。target 不可用时 fallback scene。
2. **batch 内 teacher 选择**: 对每个 student i, 在**同 ID** 且 `comp_j > comp_i` 中选 comp 最大的 j 当 teacher。无更完整同 ID → 该图无蒸馏(w=0)。
3. **ID-prototype logits**: batch 唯一 ID 的 prototype `P_uniq = clip_id_prompt(unique_labels)` (P, D); `logits = normalize(clip_id_proj(global)) @ normalize(P_uniq).t() / τ` (B, P)。
4. **暗知识 KL(对硬负)**: 对 student i / teacher j, 屏蔽各自真 ID 那列(只留 P-1 硬负), `L_dark_i = KL(log_softmax(logits_i_masked) ‖ softmax(stopgrad(logits_j_masked)))`。teacher 分布 stop-grad(teacher 只当目标不被拉)。
5. **权重**: `w_i = sigmoid_conf_i · clamp(comp_j − comp_i, 0)`, 归一化到均值 1。
6. **(可选 flag)GRL pose-adversarial**: 小 head 从 global 预测 pose 向量, 梯度反转 → global 去姿态捷径。默认关, 当 ablation。

## 损失
`L = L_clipreid(原 i2t/t2i, 不动) + λ_dark · mean(w · L_dark) [+ λ_adv · L_pose_adv]`
λ_dark 默认 0.5, τ 默认 0.1。λ_adv 默认 0(关)。

## 预期
若暗知识真带新信息: exp355 > exp341 59.8(纯噪声外, 至少 +0.3~0.5)。
失败可能: batch 内同 ID 完整度差异小(teacher≈student, w≈0 蒸馏空转); P=16 暗分布太小(只 15 硬负); 软关系对 occluded-duke 不 load-bearing。

## 对照 / 消融(关键)
- **基线 exp341**(CLIP prompt, 无蒸馏)= 59.8。单变量 = PGPD 蒸馏开关。
- **★必做控制 exp355r: random-erase teacher 配对**(teacher 随机选同 ID, 不按 pose 完整度)→ 隔离"pose 选 teacher"的价值。若 exp355 ≈ exp355r, 则 pose 无贡献(只是 KD 正则)。
- ablation: +GRL(λ_adv>0); λ_dark sweep。

## 实现文件
- `model/modules/clip_id_prompt.py`: 加 `pgpd_dark_loss(image_feat, uniq_protos, logits 计算)` 工具(或在 pose_backbone_model 内联)。
- `model/pose_backbone_model.py`: clip_id_loss 块后加 PGPD 块(teacher 选择 + dark KL)。需 scene_heatmaps(已有)、label、global。
- `config/defaults.py`: POSE_PGPD(bool), POSE_PGPD_W(0.5), POSE_PGPD_TAU(0.1), POSE_PGPD_RANDOM_TEACHER(bool, 控制), POSE_PGPD_GRL_W(0.0)。
- config: exp355_pgpd.yml = exp341 + POSE_PGPD True。

## 审查重点
- teacher 选择无同 ID 更完整图时 w=0 安全(不 NaN)。
- KL 方向对(student 学 teacher, teacher stop-grad)。
- 屏蔽真 ID 列正确(student/teacher 各自真 ID)。
- batch 唯一 ID prototype 提取与 label 映射对齐。
- 不破坏 exp341 baseline(POSE_PGPD False 时完全等价 exp341)。
- AMP 安全(KL/softmax 数值)。

## 实现备注 (首版)
- **首版只做核心暗知识蒸馏**(pose 选 teacher + dark KL)。GRL pose-adversarial **延后**(若核心有正信号再加 exp355-grl ablation), 故未加 POSE_PGPD_GRL_W flag。
- 实现: `model/pose_backbone_model.py::_pgpd_loss`(新方法); forward clip_id_loss 块 exp341 base 后调用; `config/defaults.py` 加 POSE_PGPD/_W/_TAU/_RANDOM_TEACHER; config `exp355_pgpd.yml`。
- 数据流确认: PGPD 加到 clip_id_loss → line 973 (LGPA-off 路径) 返回 {'clip_id_loss'} → processor line 1300 `loss += clip_id_w * clip_id_loss`。POSE_PGPD False 时 _pgpd_loss 不调用 → 完全等价 exp341。
- NaN 防护: P<3 跳过; 屏蔽真ID列后 0*(-inf) 用 masked_fill(prod,0) + nan_to_num; fp32 softmax; w.sum().clamp(min=1e-6)。

## ★ 结果 (2026-06-21): FAILED -1.2
exp355 PGPD(pose teacher)= global 58.6 / equal_concat 58.6, **vs exp341 59.8 = -1.2**。PGPD 弱赌注失败且为负。
**为何**: dark-KD 逼遮挡 student 匹配完整 teacher 的硬负分布——可能过度约束(遮挡 student 本该更不确定)或与主 i2t/t2i supcon(+2.2)竞争。teacher coverage 48/64 mean_w 0.61 证明 PGPD 激活非空转, 故是机制本身负, 非空转。
控制 exp355r(random teacher)~1.5h 后出 → 隔离: ≈exp355 则蒸馏本身负(pose 选无关); >exp355 则 pose 选 teacher 反而更糟。
