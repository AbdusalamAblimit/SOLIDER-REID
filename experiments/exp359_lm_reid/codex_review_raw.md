Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019efb4a-1dc7-73d0-9717-8840e1cdc47b
--------
user
Review the LM-ReID fine-tuning script: the single file matching experiments/cargo_cvpb/cvpb_lm_reid_*.py (exp359; design at experiments/exp359_lm_reid/design.md). 它 fine-tune frozen SOLIDER PoseBackboneModel(pose_dict=None → 纯 backbone global feat)成 lattice-marginalized embedding 抗采样格点(sub-pixel phase / +-1 LR-px bbox / antialias kernel)。请先 cat 该文件 + model/pose_backbone_model.py 的 forward。逐行审:
(1) runtime bug / shape / device / dtype;
(2) 用 pose_dict=None 取 out[0]=cls_score,out[1]=global_feat 是否对(对照 model/pose_backbone_model.py forward 默认 return = cls_score,global_feat,featmaps,None; 确认 pose 分支都靠 scene_heatmaps/pose_dict is not None 跳过, _lgpa_fixed_bands 默认 False);
(3) AMP 安全: autocast 仅包 forward, 损失 .float() 在 fp32, GradScaler scale/step/update;
(4) 4 loss 数学正确: L_id=CE+batch_hard_triplet; L_marg=-log(mean_l softmax(cls^l)[y])+triplet(mean_l gf); L_cons=mean(1-cos(z^l,sg z_mu))+beta*KL(p^l||sg p_mu) forward-KL; L_adv=GRL 对 variant-slot 判别器;
(5) image-major reshape: x=xb.view(B*M,..), y_rep=y.repeat_interleave(M), cls/gf.view(B,M,..), slot=arange(M).repeat(B) 是否全部对齐到 [i*M+m]=img i variant m;
(6) batch_hard_triplet 的 d_ap=(d*is_pos).max / d_an=(d+is_pos*1e9).min 掩码正确性;
(7) PKSampler 抽样;
(8) torch.save(model.state_dict()) 与 SOLIDER load_param 兼容;
(9) 对照隔离: lattice 生成复制自 cvpb_lattice_killswitch.py(byte-identical 降质), eval 复用 kill-switch.
也用 web search 查 lattice-marginalization / sub-pixel sampling uncertainty for low-res ReID 是否有 exact 先例。输出: Verdict (approve / needs-attention) + findings (severity Critical/High/Medium/Low, 每条带行号+修法).
codex
我会先按你的要求定位 exp359 脚本，读取设计文档、脚本全文和 `PoseBackboneModel.forward`，然后对照逐项做代码审查；最后再补一轮 web 检索确认是否有精确先例。
exec
/bin/zsh -lc "sed -n '1,240p' experiments/exp359_lm_reid/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
# 实验 exp359: LM-ReID (Lattice-Marginalized ReID)

> 后 PRCV「换问题」阶段第一个 GO 的 method 候选（9 个零训练 cheap-kill 全死后）。零训练 kill-switch 已 GO（d8 lattice, agent a68e6），method-design codex 判 **CCF-B 7/10**（训练版超 frozen ensemble → 8/10）。本实验做训练版，证明 lattice-marginalization 是 **method 不是 ensemble trick**。
> 设计来源：`experiments/cargo_cvpb/litreview2/pivot/clean/lattice_method_design.txt`（完整 codex 设计）+ `cvpb_lattice_result.md`（kill-switch 数据）。

## 动机

低分辨率 ReID 的传统视角：LR = 模糊/缺细节，解法 = SR / resolution-invariant feature。**我们重定义**：一部分 LR 失败不是"缺信息"，而是 **采样格点不确定性（sampling-lattice uncertainty）**——同一个 HR 身份在不同合法的 LR 采样格点（sub-pixel phase / bbox alignment / downsample kernel）下，落到不同 embedding 区域，导致 rank-1 身份翻转。

### 零训练 kill-switch 证据（GO）
frozen exp260b Market，K=9 lattice variants ensemble，HR gallery / LR query：

| h | rank-1 flip% | single LR | lat-MaxSim | **LATgain** | **LAT−TTA** |
|---|---|---|---|---|---|
| 16 | 74.9% | 42.65 | 46.87 | **+4.23** | **+3.04** |
| 24 | 31.3% | 69.31 | 72.98 | **+3.67** | **+2.68** |
| 32 | 9.7% | 81.93 | 83.98 | +2.05 | +1.44 |
| 48 | 1.2% | 90.44 | 91.02 | +0.58 | +0.41 |

两个生死控制都过：**vs 普通 TTA**（同 K 同 MaxSim，lattice 比 TTA 多 +3.04，证 lattice-specific 非 TTA 换名；lat-MaxSim 46.9 > tta 43.8 > single 42.6，无"输给 trivial baseline"问题）；**vs #false-in-topk**（phase-var partial +0.19/+0.21 存活）。机制：phase-var 是 TTA-var 的 **7.5×**；74.9% h=16 query 跨 lattice 翻转 rank-1，随 h 升单调衰减到 1.2%（h=48 机制消失 = 低分辨率专用现象）。

### 诚实 caveat（写法要求）
phase-var 作 per-query 失败预测器**不干净**（控 LR-severity 后 partial 塌到 ≈0，与 per-image LR 失真共线）。**GO 靠的是 interventional 结果**（ensemble +4.2 / LAT−TTA +3.0 是直接测量）。故事写成 "lattice sensitivity 是 **mechanism-level nuisance**, 不是 standalone failure predictor"，方法是对所有 severe-LR query 做 marginalization（不是预测哪张失败）。

## 核心假设

训练一个 lattice-marginalized embedding（对 lattice variants 身份稳定）+ 推理 K-marginalization，在 h=16 上比 frozen lattice ensemble **再高 +0.8~2.0 mAP** → 证明它学到了 lattice-invariance（是 method），不是 ensemble trick。

## 技术方案

### 数据流
1. 正常 ReID baseline（Market，exp260b 同配置）。
2. fine-tune：HR train image 在线生成 LR lattice variants `x^l = U(D_l(x))`，l ∈ {sub-pixel phase, bbox jitter, downsample kernel}。每图每 iter 采样 M=2-4 variants，eval 用 K=9。
3. h 混合训练 h ∈ {16,24,32}，severe-biased（但不只训 h=16）。

### LM-ReID loss
```
z^l = norm(fθ(T_l(x)));  p^l = softmax(W z^l);  z^μ = norm(mean_l z^l);  p^μ = mean_l p^l
L_id   = mean_l [ CE(p^l, y) + Triplet(z^l, y) ]
L_marg = -log[ mean_l p^l[y] ] + Triplet(z^μ, y)                 # marginal likelihood（主贡献）
L_cons = mean_l (1 - cos(z^l, sg(z^μ))) + β·mean_l KL(p^l || sg(p^μ))  # consistency to mean
L_adv  = GRL-CE(Dφ(z^l), lattice_label_l)                        # 弱：去掉 embedding 中可预测 lattice label
L = L_id + λ_m·L_marg + λ_c·L_cons + λ_a·L_adv
```
默认 λ_m=1.0, λ_c=0.2, β=0.5, λ_a=0.02–0.05（warmup 后开）。**L_adv 弱辅助非主贡献**（太强会擦身份边缘细节，必须 ablation）。

### 推理 K-marginalization
```
s(q, g) = τ·log[ 1/K Σ_l exp( cos(f(T_l(q)), f(g)) / τ ) ]
```
τ→0 接近 lat-MaxSim（主推，因 lat-MaxSim 46.9 > mean），τ 大接近 mean（消融）。

## 预期结果

**过线（决定 method vs trick）**：
- h=16：训练版 > frozen ensemble **+0.8~2.0 mAP**；> single +5~7；> TTA +2~3.5。
- h=24：稳定收益。
- h=32：允许 marginal 不负。

失败最可能原因：训练版只 ≈ frozen ensemble（没学到额外 lattice-invariance）→ 沦为 test-time ensemble trick，不成方法稿。备选投稿角度：同等 mAP 下 K 从 9 降到 3 或 single inference 保留大部分收益。

## 对照组

- single LR（canonical bicubic，固定一个）。
- 普通 K-TTA（同 K，random crop/flip/color/resize）。
- **frozen lattice ensemble**（零训练 K=9，= kill-switch 的 +4.23，这是训练版必须超过的硬线）。
- （成稿）k-reciprocal / SR-based / VPFA。

消融：marg only / marg+cons / marg+cons+adv；τ sweep；K=1/3/5/9 曲线；phase-only vs +bbox+kernel。

## 协议 / benchmark

- 合成：Market/MSMT，gallery HR，query LR h=16/24/32，canonical LR single baseline，K=9（3×3 phase 主，bbox/kernel ablation，不无限扩 K）。所有 TTA 对照 **K-matched**。
- 标准 CR-ReID（成稿补）：MLR-Market / MLR-CUHK03 / CAVIAR（PS-HRNet 用过）。
- 新指标：PRF@1（phase rank-flip rate）、Flip Entropy、LEG（lattice ensemble gain）、LOTG（lattice-over-TTA gain）、query ΔAP。按 h 分报 + paired bootstrap 95% CI + K=1/3/5/9 曲线 + compute cost。PRF 随 h（74.9%→31%→10%→1.2%）是强故事线。

## 撞车边界（novelty，codex 5 路联网）

- **VPFA**（2510.00936，最近邻）：CR-ReID = feature-space resolution direction，Vector Panning LR→pseudo-HR。hidden variable 是 resolution gap/feature direction，**不是 sampling lattice，不做 lattice marginalization**。必须正面对比。
- LRAR（2207.13037）：resolution-adaptive representation，占"resolution adaptive"词 → 我们 novelty 写 sampling-lattice uncertainty。
- RFD（2109.07871）：multi-res gallery distillation，占"resolution-invariant distillation"，没占 lattice intervention。
- BlurPool（1904.11486）：anti-aliasing/shift-invariance 已老 → **不能说首次发现 aliasing 影响**。**能声称**：首次在 LR person ReID 把采样格点作隐藏变量 + rank-level intervention 证明 + lattice-specific marginalization 解决检索身份翻转。
- FlipReID（2105.05639）：ReID 常用 flip-mean TTA → **必须反复强调非 TTA 换名**（同 K vs-TTA 控制，lattice 多 +3.04）。

## 风险与定位

operating point 低（只在 h≤24 强 +3-4 mAP，h=32 marginal）→ 主动收窄定位 **"severe low-resolution / cross-resolution ReID under sampling-lattice uncertainty"**，不写成通用 ReID 鲁棒性稿。h=32 marginal 反而支持机制（分辨率升高 → lattice uncertainty 消退）。真正风险 = 训练版打不过 frozen ensemble，那就只是 ensemble trick。

## 审查 / 训练协议

1. 本 design.md（已写）。
2. 实现 LM-ReID（插件式，config 开关，lattice aug 复用 `cvpb_lattice_killswitch.py` 的 LR 生成）。
3. Claude 广审（Opus 子代理）→ `claude_review.md`；Codex 审（`codex --search exec`）→ `codex_review.md`。**两层通过才训练**。
4. smoke（几 iter 确认各 loss 分量下降无泄漏）。
5. 训练（lab-3090，Market，h 混合，BS=64 不改，TEST.IMS_PER_BATCH 64，PYTHONUNBUFFERED=1 nohup）。
6. eval（**test.py 不用 train.py**）：LM-ReID(K marg) vs single / TTA / frozen-ensemble，h=16/24/32 分报，LEG/LOTG/PRF。

exec
/bin/zsh -lc "rg --files experiments/cargo_cvpb | rg 'experiments/cargo_cvpb/cvpb_lm_reid_.*\\.py"'$'"'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/cargo_cvpb/cvpb_lm_reid_train.py

exec
/bin/zsh -lc 'rg -n "def forward|_lgpa_fixed_bands|scene_heatmaps|pose_dict|return cls_score" model/pose_backbone_model.py' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
191:            self._lgpa_fixed_bands = getattr(cfg.MODEL, 'POSE_LGPA_FIXED_BANDS', False)
198:            if self._lgpa_fixed_bands:
457:    def _run_backbone_with_psg(self, x, scene_heatmaps, pose_dict=None):
467:        if getattr(self, 'use_pose_patch_embed', False) and scene_heatmaps is not None:
470:            hm = F.interpolate(scene_heatmaps, size=(H_hw, W_hw),
477:        if getattr(self, 'use_pose_prompt', False) and scene_heatmaps is not None:
480:            hm = F.interpolate(scene_heatmaps, size=(H_hw, W_hw),
520:                    stage, x, hw_shape, scene_heatmaps, stage_idx=i)
548:    def _run_stage_with_psg(self, stage, x, hw_shape, scene_heatmaps,
558:            if scene_heatmaps is not None and key in self.psg_modules_dict:
559:                x = self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)
562:            if getattr(self, 'use_paa', False) and scene_heatmaps is not None and key in getattr(self, 'paa_modules_dict', {}):
563:                x = self.paa_modules_dict[key](x, hw_shape, scene_heatmaps)
591:    def _pgpd_loss(self, img_proj, txt_proto, label, scene_heatmaps, target_heatmaps=None):
616:        # Use target_heatmaps (target person only), NOT scene_heatmaps (max-merged over all
620:        comp_hm = target_heatmaps if target_heatmaps is not None else scene_heatmaps
655:    def _pcmsc_loss(self, featmap, img, scene_heatmaps):
670:        pose = F.interpolate(scene_heatmaps.float(), size=(H, W), mode='bilinear', align_corners=False)
696:    def _lgpa_heatmap(self, scene_heatmaps, B, device):
701:        if getattr(self, '_lgpa_fixed_bands', False):
703:        return scene_heatmaps
705:    def forward(self, x, label=None, cam_label=None, view_label=None,
706:                pose_dict=None):
708:        scene_heatmaps = None
710:        if pose_dict is not None:
711:            scene_heatmaps, _, target_heatmaps, _ = self._prepare_pose(pose_dict)
715:            if self.training and getattr(self, 'use_pose_shuffle', False) and scene_heatmaps is not None:
716:                Bp = scene_heatmaps.shape[0]
719:                    ar = torch.arange(Bp, device=scene_heatmaps.device)
726:                    scene_heatmaps = scene_heatmaps[perm]
733:            if self.training and getattr(self, 'use_pose_channel_shuffle', False) and scene_heatmaps is not None:
734:                Kc = scene_heatmaps.shape[1]
735:                cperm = torch.argsort(torch.rand(scene_heatmaps.shape[0], Kc, device=scene_heatmaps.device), dim=1)
736:                idx = cperm[:, :, None, None].expand(-1, -1, scene_heatmaps.shape[2], scene_heatmaps.shape[3])
737:                scene_heatmaps = torch.gather(scene_heatmaps, 1, idx)
742:        # Substitute scene_heatmaps with target_heatmaps so all downstream
746:        # (default), scene_heatmaps keeps its original max-merged value.
748:            scene_heatmaps = target_heatmaps
751:        if self.training and scene_heatmaps is not None and self.pose_dropout_p > 0:
752:            keep_mask = (torch.rand(scene_heatmaps.shape[0], 1, 1, 1,
753:                                    device=scene_heatmaps.device) >= self.pose_dropout_p)
754:            scene_heatmaps = scene_heatmaps * keep_mask.float()
757:        global_feat, featmaps = self._run_backbone_with_psg(x, scene_heatmaps)
777:                pose_vec = scene_heatmaps.float().mean(dim=(2, 3)) \
778:                    if (getattr(self.clip_id_prompt, 'pose_cond', False) and scene_heatmaps is not None) else None
781:                if getattr(self, 'use_clip_id_part_guided', False) and scene_heatmaps is not None:
783:                    part_feats = self.pose_guided_part_pool(featmaps[-1], scene_heatmaps)  # (B, nP, C)
792:                    if getattr(self, 'use_clip_id_noparam_pool', False) and scene_heatmaps is not None:
793:                        feat_for_clip = self.pose_weighted_pool(featmaps[-1], scene_heatmaps)
794:                    elif getattr(self, 'use_clip_id_pose_guided', False) and scene_heatmaps is not None:
795:                        feat_for_clip = self.pose_guided_pool(featmaps[-1], scene_heatmaps)
803:                    if getattr(self, 'use_pgpd', False) and scene_heatmaps is not None:
804:                        clip_id_loss = clip_id_loss + self._pgpd_loss(img_proj, txt_proto, label, scene_heatmaps, target_heatmaps)
807:                    if getattr(self, 'use_clip_id_occ_repel', False) and scene_heatmaps is not None:
808:                        occ_feat = self.pose_weighted_pool(featmaps[-1], scene_heatmaps, invert=True)
815:            if getattr(self, 'use_pcmsc', False) and scene_heatmaps is not None and self.training:
816:                pcmsc = self._pcmsc_loss(featmaps[-1], x, scene_heatmaps)
820:            if getattr(self, 'use_vcsr', False) and scene_heatmaps is not None:
823:                    vcsr_input, scene_heatmaps, return_cls=True)
828:            elif getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None \
834:                kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
835:                sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
837:                    spatial_tokens, (H_fm, W_fm), scene_heatmaps,
870:                    hm_r = F.interpolate(scene_heatmaps, size=(H_fm, W_fm),
905:            elif getattr(self, 'use_lgpa', False) and (scene_heatmaps is not None or getattr(self, '_lgpa_fixed_bands', False)):
908:                lgpa_hm = self._lgpa_heatmap(scene_heatmaps, x.shape[0], x.device)
916:                if self.use_skeleton_gcn and pose_dict is not None:
920:                        feat_map_detached, pose_dict, return_cls=True, label=label,
933:            elif getattr(self, 'use_ppa', False) and scene_heatmaps is not None:
936:                    featmaps[-1], scene_heatmaps, return_cls=True)
940:                if self.use_skeleton_gcn and pose_dict is not None:
944:                        feat_map_detached, pose_dict, return_cls=True, label=label,
956:            elif self.use_skeleton_gcn and pose_dict is not None:
967:                if getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None \
971:                    kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
972:                    sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
974:                        spatial_tokens, (H_fm, W_fm), scene_heatmaps,
1000:                        spatial_tokens, scene_heatmaps, fH=H_d, fW=W_d)
1007:                    feat_map_detached, pose_dict, return_cls=True, label=label,
1018:                    kp_coords = pose_dict['keypoints'][:, 0, :, :]  # (B, 17, 2) person 0
1019:                    kp_scores = pose_dict['scores'][:, 0, :]        # (B, 17) person 0
1054:                        hm_r = F.interpolate(scene_heatmaps, size=(featmaps[-1].shape[2], featmaps[-1].shape[3]),
1068:                    kp_coords = pose_dict['keypoints'][:, 0, :, :]  # (B, 17, 2)
1087:                return cls_score, global_feat, featmaps, None, {'clip_id_loss': clip_id_loss}
1088:            return cls_score, global_feat, featmaps, None
1100:            if getattr(self, 'use_vcsr', False) and scene_heatmaps is not None and \
1103:                    featmaps[-1], scene_heatmaps, return_cls=False)
1106:            # LGPA test path — uses scene_heatmaps (same as PPA for fair comparison)
1107:            elif getattr(self, 'use_lgpa', False) and (scene_heatmaps is not None or getattr(self, '_lgpa_fixed_bands', False)) and \
1109:                lgpa_hm = self._lgpa_heatmap(scene_heatmaps, x.shape[0], x.device)
1114:                if self.use_skeleton_gcn and pose_dict is not None:
1116:                        featmaps[-1], pose_dict, return_cls=False)
1123:            elif getattr(self, 'use_ppa', False) and scene_heatmaps is not None and \
1126:                    featmaps[-1], scene_heatmaps, return_cls=False)
1129:                if self.use_skeleton_gcn and pose_dict is not None:
1131:                        featmaps[-1], pose_dict, return_cls=False)
1137:            elif getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None and \
1141:                kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
1142:                sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
1144:                    spatial_tokens, (H_fm, W_fm), scene_heatmaps,
1153:                    hm_r = F.interpolate(scene_heatmaps, size=(H_fm, W_fm),
1176:            elif self.use_skeleton_gcn and pose_dict is not None and \
1180:                if getattr(self, 'use_fsdc', False) and scene_heatmaps is not None:
1184:                        spatial_tokens, scene_heatmaps, fH=H_d, fW=W_d)
1188:                    feat_for_gcn, pose_dict, return_cls=False,
1191:                if getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None \
1195:                    kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
1196:                    sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
1198:                        spatial_tokens, (H_fm, W_fm), scene_heatmaps,
1207:                    kp_coords = pose_dict['keypoints'][:, 0, :, :]
1208:                    kp_scores = pose_dict['scores'][:, 0, :]
1244:    def _prepare_pose(pose_dict):
1248:            scene_heatmaps: (B, 17, H, W) merged scene-level heatmap
1253:        heatmaps = pose_dict['heatmaps']
1254:        scores = pose_dict['scores']
1255:        person_mask = pose_dict['person_mask']
1257:        scene_heatmaps = merge_person_heatmaps(heatmaps, person_mask)
1273:        return scene_heatmaps, scene_scores, target_heatmaps, diff_heatmaps

codex
已定位到唯一脚本 `experiments/cargo_cvpb/cvpb_lm_reid_train.py`，设计文档显示它确实是 exp359 的训练版 LM-ReID。接下来我会用带行号输出审查，便于后面直接给每条 finding 的修法。
exec
/bin/zsh -lc 'nl -ba experiments/cargo_cvpb/cvpb_lm_reid_train.py' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""LM-ReID (Lattice-Marginalized ReID) — TRAINING  (exp359, Market, fine-tune exp260b).
     3	
     4	HYPOTHESIS: a lattice-marginalized embedding (trained to be INVARIANT to the sampling
     5	lattice — sub-pixel phase / +/-1 LR-pixel bbox / antialias kernel) BEATS the zero-training
     6	frozen K-phase ensemble (the kill-switch's +4.23 mAP @ h=16). If the trained model clears
     7	frozen-ensemble +0.8~2.0 @ h=16 -> it is a METHOD; if it only ~= the frozen ensemble ->
     8	it is an ensemble trick (honest fail, report as such).
     9	
    10	Design: experiments/exp359_lm_reid/design.md  (method-design codex, CCF-B 7/10).
    11	Loss:   L = L_id + lam_marg*L_marg + lam_cons*L_cons + lam_adv*L_adv.
    12	        L_id   = mean_l [ CE(cls^l, y) + Triplet(gf^l, y) ]            (per-variant ReID)
    13	        L_marg = -log[ mean_l softmax(cls^l)[y] ] + Triplet(mean_l gf^l, y)  (marginal lik.)
    14	        L_cons = mean_l (1-cos(z^l, sg(z_mu))) + beta*mean_l KL(p^l || sg(p_mu))  (lattice inv.)
    15	        L_adv  = GRL: a discriminator that predicts the lattice-variant index from z is
    16	                 reversed, so z carries NO predictable lattice label (weak, warmup-gated).
    17	
    18	EVAL is done SEPARATELY (apples-to-apples, byte-identical to the GO kill-switch):
    19	    cvpb_lattice_killswitch.py --ckpt <this output>/transformer_<ep>.pth
    20	    -> compare the fine-tuned single / lat-mean / lat-max vs the FROZEN lat-MaxSim 46.87.
    21	
    22	Backbone fine-tune uses pose_dict=None (POSE DISABLED, identical to the frozen baseline's
    23	PSG-off global feat); all pose-conditional branches are skipped inside the model forward.
    24	
    25	Run (lab-3090):
    26	    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 \
    27	      /root/miniconda3/envs/solider-reid/bin/python \
    28	      experiments/cargo_cvpb/cvpb_lm_reid_train.py \
    29	      --epochs 40 --out log/market1501/exp359_lm_reid 2>&1 | tee /tmp/exp359_lm_reid.log
    30	    # smoke first:  --epochs 1 --smoke_ids 32 --workers 4
    31	"""
    32	import os, sys, time, argparse, random, math
    33	import numpy as np
    34	from PIL import Image
    35	
    36	_here = os.path.dirname(os.path.abspath(__file__))
    37	_repo = os.path.abspath(os.path.join(_here, '..', '..'))
    38	sys.path.insert(0, _repo)
    39	
    40	ap = argparse.ArgumentParser()
    41	ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
    42	ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
    43	ap.add_argument('--data_root', default='data')
    44	ap.add_argument('--out', default='log/market1501/exp359_lm_reid')
    45	ap.add_argument('--heights', type=int, nargs='+', default=[16, 24, 32])
    46	ap.add_argument('--height_p', type=float, nargs='+', default=[0.5, 0.3, 0.2],
    47	                help='sampling prob per height (severe-biased)')
    48	ap.add_argument('--M', type=int, default=2, help='#lattice variants per image per iter')
    49	ap.add_argument('--P', type=int, default=16, help='#ids per batch')
    50	ap.add_argument('--Kins', type=int, default=4, help='#instances per id per batch (P*Kins=BS=64)')
    51	ap.add_argument('--epochs', type=int, default=40)
    52	ap.add_argument('--lr', type=float, default=3.5e-3)
    53	ap.add_argument('--weight_decay', type=float, default=1e-4)
    54	ap.add_argument('--warmup', type=int, default=5)
    55	ap.add_argument('--lam_marg', type=float, default=1.0)
    56	ap.add_argument('--lam_cons', type=float, default=0.2)
    57	ap.add_argument('--beta_kl', type=float, default=0.5)
    58	ap.add_argument('--lam_adv', type=float, default=0.0, help='0 disables L_adv (weak aux)')
    59	ap.add_argument('--adv_start', type=int, default=10)
    60	ap.add_argument('--margin', type=float, default=0.3)
    61	ap.add_argument('--workers', type=int, default=8)
    62	ap.add_argument('--seed', type=int, default=42)
    63	ap.add_argument('--smoke_ids', type=int, default=0, help='cap #train ids for a fast smoke')
    64	ap.add_argument('--smoke_iters', type=int, default=0, help='cap iters/epoch for smoke')
    65	ap.add_argument('--save_every', type=int, default=20)
    66	cli = ap.parse_args()
    67	
    68	random.seed(cli.seed); np.random.seed(cli.seed)
    69	SIZE_TEST = (384, 128)                       # (H, W) model input / HR canvas
    70	PIXEL_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    71	PIXEL_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    72	_KERNELS = {'bicubic': Image.BICUBIC, 'bilinear': Image.BILINEAR, 'lanczos': Image.LANCZOS,
    73	            'box': Image.BOX, 'hamming': Image.HAMMING, 'nearest': Image.NEAREST}
    74	
    75	# =========================================================================== #
    76	# data list (Market train; relabel pids to 0..N-1)
    77	# =========================================================================== #
    78	import re, glob
    79	_PAT = re.compile(r'([-\d]+)_c(\d)')
    80	
    81	
    82	def list_train(dir_path):
    83	    raw = []
    84	    pids = set()
    85	    for p in sorted(glob.glob(os.path.join(dir_path, '*.jpg'))):
    86	        pid, cam = map(int, _PAT.search(p).groups())
    87	        if pid == -1:
    88	            continue
    89	        raw.append([p, pid, cam - 1]); pids.add(pid)
    90	    pid2lbl = {pid: i for i, pid in enumerate(sorted(pids))}
    91	    items = [[p, pid2lbl[pid], cam] for (p, pid, cam) in raw]
    92	    return items, len(pids)
    93	
    94	
    95	# =========================================================================== #
    96	# LR + lattice variant generation (COPIED verbatim from cvpb_lattice_killswitch.py
    97	# so the training-time degradation is byte-identical to the GO kill-switch eval).
    98	# =========================================================================== #
    99	def _to_target_aspect(img):
   100	    return img.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)
   101	
   102	
   103	def make_lr(hr_img, h, kernel='bicubic'):
   104	    w = max(1, int(round(h * SIZE_TEST[1] / SIZE_TEST[0])))
   105	    small = hr_img.resize((w, h), _KERNELS[kernel])
   106	    return small.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)
   107	
   108	
   109	def make_lattice_variants(hr_img, h, K, rng):
   110	    """K plausible PHASE/CROP/KERNEL variants of the SAME hr image at height h.
   111	    variant 0 = canonical deterministic bicubic LR (single-LR baseline)."""
   112	    W_hr, H_hr = hr_img.size
   113	    hr_per_lr_y = H_hr / float(h)
   114	    hr_per_lr_x = W_hr / float(max(1, round(h / 3.0)))
   115	    variants = [make_lr(hr_img, h, 'bicubic')]
   116	    kernels_cycle = ['bicubic', 'bilinear', 'lanczos', 'box', 'hamming']
   117	    for j in range(1, K):
   118	        mode = j % 3
   119	        kern = kernels_cycle[j % len(kernels_cycle)]
   120	        if mode == 0:
   121	            dx = rng.uniform(-0.5, 0.5) * hr_per_lr_x
   122	            dy = rng.uniform(-0.5, 0.5) * hr_per_lr_y
   123	            shifted = hr_img.transform((W_hr, H_hr), Image.AFFINE, (1, 0, dx, 0, 1, dy),
   124	                                       resample=Image.BICUBIC)
   125	            v = make_lr(shifted, h, kern)
   126	        elif mode == 1:
   127	            sx = int(round(rng.choice([-1, 0, 1]) * hr_per_lr_x))
   128	            sy = int(round(rng.choice([-1, 0, 1]) * hr_per_lr_y))
   129	            left = max(0, sx); upper = max(0, sy)
   130	            right = W_hr + min(0, sx); lower = H_hr + min(0, sy)
   131	            if right - left < 4 or lower - upper < 4:
   132	                left, upper, right, lower = 0, 0, W_hr, H_hr
   133	            cropped = hr_img.crop((left, upper, right, lower)).resize((W_hr, H_hr), Image.BICUBIC)
   134	            v = make_lr(cropped, h, kern)
   135	        else:
   136	            ez = rng.choice([-1, 1]) * 0.5 * hr_per_lr_y
   137	            box = (-ez, -ez * (W_hr / H_hr), W_hr + ez, H_hr + ez * (W_hr / H_hr)) if ez > 0 \
   138	                else (abs(ez), abs(ez) * (W_hr / H_hr), W_hr - abs(ez), H_hr - abs(ez) * (W_hr / H_hr))
   139	            l, u, r, b = (int(round(v_)) for v_ in box)
   140	            pad = max(0, -l, -u, r - W_hr, b - H_hr) + 1
   141	            canvas = Image.new('RGB', (W_hr + 2 * pad, H_hr + 2 * pad), (0, 0, 0))
   142	            canvas.paste(hr_img, (pad, pad))
   143	            cropped = canvas.crop((l + pad, u + pad, r + pad, b + pad)).resize((W_hr, H_hr), Image.BICUBIC)
   144	            v = make_lr(cropped, h, kern)
   145	        variants.append(v)
   146	    return variants
   147	
   148	
   149	def pil_to_tensor_np(img):
   150	    arr = np.asarray(img, dtype=np.float32) / 255.0
   151	    arr = (arr - PIXEL_MEAN) / PIXEL_STD
   152	    return arr.transpose(2, 0, 1)
   153	
   154	
   155	# =========================================================================== #
   156	# PK dataset: each item = ONE person image -> M lattice LR variants (random height).
   157	# A PK batch sampler draws P ids x Kins instances; collate stacks [B, M, C, H, W].
   158	# =========================================================================== #
   159	import torch
   160	from torch.utils.data import Dataset, DataLoader, Sampler
   161	from datasets.bases import read_image
   162	
   163	
   164	class LatticeTrainSet(Dataset):
   165	    def __init__(self, items):
   166	        self.items = items
   167	
   168	    def __len__(self):
   169	        return len(self.items)
   170	
   171	    def __getitem__(self, idx):
   172	        p, lbl, cam = self.items[idx]
   173	        # per-sample rng so workers differ but run is reproducible-ish
   174	        rng = np.random.RandomState((cli.seed + idx * 2654435761) % (2**32))
   175	        h = int(rng.choice(cli.heights, p=np.array(cli.height_p) / np.sum(cli.height_p)))
   176	        hr = _to_target_aspect(read_image(p))
   177	        vs = make_lattice_variants(hr, h, cli.M, rng)        # M PIL @ 384x128
   178	        t = np.stack([pil_to_tensor_np(v) for v in vs], 0)   # [M,3,H,W]
   179	        return torch.from_numpy(t), int(lbl)
   180	
   181	
   182	class PKSampler(Sampler):
   183	    """Yield flat indices in P-id x Kins-instance blocks (one 'batch' = P*Kins)."""
   184	    def __init__(self, items, P, Kins, num_iters=None):
   185	        self.P, self.Kins = P, Kins
   186	        self.by_pid = {}
   187	        for i, (_, lbl, _) in enumerate(items):
   188	            self.by_pid.setdefault(lbl, []).append(i)
   189	        self.pids = list(self.by_pid.keys())
   190	        self.length = (len(items) // (P * Kins)) if num_iters is None else num_iters
   191	        self._n_items = len(items)
   192	
   193	    def __len__(self):
   194	        return self.length * self.P * self.Kins
   195	
   196	    def __iter__(self):
   197	        flat = []
   198	        for _ in range(self.length):
   199	            chosen = random.sample(self.pids, self.P) if len(self.pids) >= self.P \
   200	                else [random.choice(self.pids) for _ in range(self.P)]
   201	            for pid in chosen:
   202	                pool = self.by_pid[pid]
   203	                if len(pool) >= self.Kins:
   204	                    flat.extend(random.sample(pool, self.Kins))
   205	                else:
   206	                    flat.extend([random.choice(pool) for _ in range(self.Kins)])
   207	        return iter(flat)
   208	
   209	
   210	def collate(batch):
   211	    ts = torch.stack([b[0] for b in batch], 0)               # [B,M,3,H,W]
   212	    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
   213	    return ts, ys
   214	
   215	
   216	# =========================================================================== #
   217	# losses
   218	# =========================================================================== #
   219	def euclidean_dist(x, y):
   220	    m, n = x.size(0), y.size(0)
   221	    xx = (x ** 2).sum(1, keepdim=True).expand(m, n)
   222	    yy = (y ** 2).sum(1, keepdim=True).expand(n, m).t()
   223	    d = xx + yy - 2 * x @ y.t()
   224	    return d.clamp(min=1e-12).sqrt()
   225	
   226	
   227	def batch_hard_triplet(feat, labels, margin):
   228	    """standard batch-hard triplet (hardest pos / hardest neg) on euclidean feat."""
   229	    d = euclidean_dist(feat, feat)
   230	    N = labels.size(0)
   231	    is_pos = labels[:, None].eq(labels[None, :])
   232	    is_neg = ~is_pos
   233	    # hardest positive: max same-id dist (diagonal is 0/same-id, included is fine)
   234	    d_ap = (d * is_pos.float()).max(1)[0]
   235	    # hardest negative: min diff-id dist
   236	    d_an = (d + is_pos.float() * 1e9).min(1)[0]
   237	    y = torch.ones_like(d_ap)
   238	    return torch.nn.functional.margin_ranking_loss(d_an, d_ap, y, margin=margin)
   239	
   240	
   241	class GradReverse(torch.autograd.Function):
   242	    @staticmethod
   243	    def forward(ctx, x, lamb):
   244	        ctx.lamb = lamb
   245	        return x.view_as(x)
   246	
   247	    @staticmethod
   248	    def backward(ctx, g):
   249	        return -ctx.lamb * g, None
   250	
   251	
   252	# =========================================================================== #
   253	# model (TRAINABLE; pose disabled -> plain backbone global feat + classifier)
   254	# =========================================================================== #
   255	def build_trainable_model():
   256	    from config import cfg
   257	    from model import make_model
   258	    from datasets.market1501 import Market1501
   259	    cfg.merge_from_file(os.path.join(_repo, cli.config))
   260	    cfg.merge_from_list([
   261	        'MODEL.POSE_TEST_FEAT', 'global',
   262	        'TEST.NECK_FEAT', 'after',
   263	        'TEST.FEAT_NORM', 'yes',
   264	    ])
   265	    cfg.freeze()
   266	    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
   267	    ds = Market1501(root=os.path.join(_repo, cli.data_root), verbose=False)
   268	    model = make_model(cfg, num_class=ds.num_train_pids, camera_num=ds.num_train_cams,
   269	                       view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
   270	    model.load_param(os.path.join(_repo, cli.ckpt))
   271	    print(f"[model] loaded {cli.ckpt}; num_cls={ds.num_train_pids}; pose DISABLED (pose_dict=None)",
   272	          flush=True)
   273	    return model.cuda(), ds.num_train_pids
   274	
   275	
   276	def main():
   277	    print("#" * 88)
   278	    print("# LM-ReID TRAINING (exp359) — fine-tune exp260b for lattice-marginalized embedding")
   279	    print("#" * 88)
   280	    t_items, n_cls = list_train(os.path.join(_repo, cli.data_root, 'market1501', 'bounding_box_train'))
   281	    if cli.smoke_ids > 0:
   282	        keep = set(sorted({it[1] for it in t_items})[:cli.smoke_ids])
   283	        t_items = [it for it in t_items if it[1] in keep]
   284	        # relabel again to contiguous
   285	        remap = {l: i for i, l in enumerate(sorted({it[1] for it in t_items}))}
   286	        t_items = [[p, remap[l], c] for (p, l, c) in t_items]
   287	        n_cls = len(remap)
   288	    print(f"[data] #train_img={len(t_items)} #ids(for sampler)={len({it[1] for it in t_items})} "
   289	          f"n_cls={n_cls}  M={cli.M} P={cli.P} Kins={cli.Kins} (BS={cli.P*cli.Kins}) heights={cli.heights}")
   290	
   291	    model, _ = build_trainable_model()
   292	    in_planes = model.bottleneck.weight.shape[0]
   293	    print(f"[model] embedding dim (bottleneck) = {in_planes}")
   294	
   295	    # lattice discriminator for L_adv (predict which of M variant slots): tiny MLP
   296	    disc = torch.nn.Sequential(torch.nn.Linear(in_planes, 256), torch.nn.ReLU(inplace=True),
   297	                               torch.nn.Linear(256, cli.M)).cuda()
   298	
   299	    params = [p for p in model.parameters() if p.requires_grad] + list(disc.parameters())
   300	    opt = torch.optim.SGD(params, lr=cli.lr, momentum=0.9, weight_decay=cli.weight_decay, nesterov=True)
   301	    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
   302	    scaler = torch.cuda.amp.GradScaler()    # AMP: B*M=128 Swin-base fwd+bwd would be tight in fp32
   303	
   304	    ds = LatticeTrainSet(t_items)
   305	    iters_per_epoch = (len(t_items) // (cli.P * cli.Kins))
   306	    if cli.smoke_iters > 0:
   307	        iters_per_epoch = min(iters_per_epoch, cli.smoke_iters)
   308	    sampler = PKSampler(t_items, cli.P, cli.Kins, num_iters=iters_per_epoch)
   309	    loader = DataLoader(ds, batch_size=cli.P * cli.Kins, sampler=sampler, num_workers=cli.workers,
   310	                        collate_fn=collate, pin_memory=True, drop_last=True)
   311	
   312	    os.makedirs(os.path.join(_repo, cli.out), exist_ok=True)
   313	
   314	    def lr_at(ep):
   315	        if ep < cli.warmup:
   316	            return cli.lr * (ep + 1) / max(1, cli.warmup)
   317	        prog = (ep - cli.warmup) / max(1, cli.epochs - cli.warmup)
   318	        return 0.5 * cli.lr * (1 + math.cos(math.pi * prog))
   319	
   320	    print(f"[train] iters/epoch={iters_per_epoch}  epochs={cli.epochs}  lr={cli.lr}", flush=True)
   321	    for ep in range(cli.epochs):
   322	        model.train()
   323	        for g in opt.param_groups:
   324	            g['lr'] = lr_at(ep)
   325	        adv_lamb = cli.lam_adv if (cli.lam_adv > 0 and ep >= cli.adv_start) else 0.0
   326	        agg = {k: 0.0 for k in ('L', 'id', 'marg', 'cons', 'adv', 'acc')}
   327	        t0 = time.time()
   328	        for it, (xb, yb) in enumerate(loader):
   329	            B, M = xb.shape[0], xb.shape[1]
   330	            x = xb.view(B * M, *xb.shape[2:]).cuda(non_blocking=True)
   331	            y = yb.cuda(non_blocking=True)
   332	            y_rep = y.repeat_interleave(M)                    # [B*M]
   333	            cam0 = torch.zeros(B * M, dtype=torch.long, device=x.device)
   334	            with torch.cuda.amp.autocast():
   335	                out = model(x, label=y_rep, cam_label=cam0, view_label=cam0, pose_dict=None)
   336	            # pose-OFF training return is (cls_score, global_feat, featmaps, None); be robust to the
   337	            # list form ([cls_score]+heads, [global_feat]+heads) in case a pose branch ever fires.
   338	            # losses computed in fp32 (.float()) for numerical safety (log / KL underflow in fp16).
   339	            cls = (out[0][0] if isinstance(out[0], (list, tuple)) else out[0]).float()   # [B*M, n_cls]
   340	            gf = (out[1][0] if isinstance(out[1], (list, tuple)) else out[1]).float()    # [B*M, D]
   341	            D = gf.shape[1]
   342	
   343	            # ---- L_id (per-variant CE + triplet) ----
   344	            L_ce = ce(cls, y_rep)
   345	            L_tri = batch_hard_triplet(gf, y_rep, cli.margin)
   346	            L_id = L_ce + L_tri
   347	
   348	            # ---- reshape to [B, M, .] ----
   349	            cls_bm = cls.view(B, M, -1)
   350	            gf_bm = gf.view(B, M, D)
   351	
   352	            # ---- L_marg (marginal likelihood + triplet on mean feat) ----
   353	            p_bm = torch.softmax(cls_bm, dim=-1)             # [B,M,C]
   354	            p_mean = p_bm.mean(1)                            # [B,C]
   355	            ll = torch.log(p_mean.gather(1, y[:, None]).clamp_min(1e-8)).squeeze(1)  # [B]
   356	            gf_mean = gf_bm.mean(1)                          # [B,D]
   357	            L_marg = -ll.mean() + batch_hard_triplet(gf_mean, y, cli.margin)
   358	
   359	            # ---- L_cons (lattice invariance: pull each variant to the mean) ----
   360	            z = torch.nn.functional.normalize(gf_bm, dim=-1)            # [B,M,D]
   361	            z_mu = torch.nn.functional.normalize(gf_bm.mean(1), dim=-1).detach()  # [B,D]
   362	            cos_term = (1.0 - (z * z_mu[:, None, :]).sum(-1)).mean()
   363	            # KL(p^l || sg(p_mu)) — forward KL (design): pull each variant's prediction to the mean.
   364	            logp_l = torch.log_softmax(cls_bm, dim=-1)                  # [B,M,C] = log p^l
   365	            log_pmu = torch.log(p_mean.detach().clamp_min(1e-8))[:, None, :]   # [B,1,C] = log sg(p_mu)
   366	            kl_term = (p_bm * (logp_l - log_pmu)).sum(-1).mean()        # p_bm = softmax(cls_bm) = p^l
   367	            L_cons = cos_term + cli.beta_kl * kl_term
   368	
   369	            # ---- L_adv (GRL: remove predictable lattice-slot label from z) ----
   370	            if adv_lamb > 0:
   371	                slot = torch.arange(M, device=x.device).repeat(B)      # [B*M] variant index
   372	                zr = GradReverse.apply(torch.nn.functional.normalize(gf, dim=-1), adv_lamb)
   373	                L_adv = ce(disc(zr), slot)
   374	            else:
   375	                L_adv = torch.zeros((), device=x.device)
   376	
   377	            loss = L_id + cli.lam_marg * L_marg + cli.lam_cons * L_cons + adv_lamb * L_adv
   378	            opt.zero_grad(set_to_none=True)
   379	            scaler.scale(loss).backward()
   380	            scaler.step(opt)
   381	            scaler.update()
   382	
   383	            with torch.no_grad():
   384	                acc = (cls.argmax(1) == y_rep).float().mean().item()
   385	            agg['L'] += loss.item(); agg['id'] += L_id.item(); agg['marg'] += L_marg.item()
   386	            agg['cons'] += L_cons.item(); agg['adv'] += float(L_adv.item()); agg['acc'] += acc
   387	            if (it + 1) % 50 == 0 or cli.smoke_iters > 0:
   388	                n = it + 1
   389	                print(f"  ep{ep} it{n}/{iters_per_epoch} L={agg['L']/n:.3f} id={agg['id']/n:.3f} "
   390	                      f"marg={agg['marg']/n:.3f} cons={agg['cons']/n:.3f} adv={agg['adv']/n:.3f} "
   391	                      f"acc={agg['acc']/n:.3f} lr={lr_at(ep):.2e}", flush=True)
   392	        n = max(1, it + 1)
   393	        print(f"[epoch {ep}] L={agg['L']/n:.3f} id={agg['id']/n:.3f} marg={agg['marg']/n:.3f} "
   394	              f"cons={agg['cons']/n:.3f} adv={agg['adv']/n:.3f} acc={agg['acc']/n:.3f} "
   395	              f"({time.time()-t0:.0f}s)", flush=True)
   396	        if (ep + 1) % cli.save_every == 0 or (ep + 1) == cli.epochs:
   397	            sp = os.path.join(_repo, cli.out, f'transformer_{ep+1}.pth')
   398	            torch.save(model.state_dict(), sp)
   399	            print(f"[save] {sp}", flush=True)
   400	    print("[done] LM-ReID training complete. EVAL via cvpb_lattice_killswitch.py --ckpt <out>.")
   401	
   402	
   403	if __name__ == '__main__':
   404	    main()

exec
/bin/zsh -lc "nl -ba model/pose_backbone_model.py | sed -n '430,540p'" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "nl -ba model/pose_backbone_model.py | sed -n '680,1225p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   430	            lpcs_delta_scale = getattr(cfg.MODEL, 'POSE_LPCS_DELTA_SCALE', 0.5)
   431	            lpcs_head_mode = getattr(cfg.MODEL, 'POSE_LPCS_HEAD_MODE', 'residual')
   432	            lpcs_context_mode = getattr(cfg.MODEL, 'POSE_LPCS_CONTEXT_MODE', 'none')
   433	            if lpcs_context_mode in ('query_ctx', 'comp_ctx'):
   434	                lpcs_input_dim = 11
   435	            else:
   436	                lpcs_input_dim = 6
   437	            if lpcs_head_mode == 'residual_conf':
   438	                self.lpcs_head = PairResidualConfidenceScorer(
   439	                    input_dim=lpcs_input_dim,
   440	                    hidden_dim=lpcs_hidden,
   441	                    delta_scale=lpcs_delta_scale,
   442	                )
   443	            else:
   444	                self.lpcs_head = PairResidualScorer(
   445	                    input_dim=lpcs_input_dim,
   446	                    hidden_dim=lpcs_hidden,
   447	                    delta_scale=lpcs_delta_scale,
   448	                )
   449	            lpcs_params = sum(p.numel() for p in self.lpcs_head.parameters())
   450	            print(f'[LPCS] Learned Pair Correction Scorer enabled: '
   451	                  f'head_mode={lpcs_head_mode}, hidden={lpcs_hidden}, delta_scale={lpcs_delta_scale}, '
   452	                  f'context_mode={lpcs_context_mode}, params={lpcs_params}')
   453	
   454	        # Store backbone's semantic weight for manual forward
   455	        self._semantic_weight_val = semantic_weight
   456	
   457	    def _run_backbone_with_psg(self, x, scene_heatmaps, pose_dict=None):
   458	        """Run backbone forward with PSG injection in configured stages.
   459	
   460	        Manually iterates backbone stages, inserting PSG after each block
   461	        in the configured stages.
   462	        """
   463	        # Patch embedding
   464	        x, hw_shape = self.base.patch_embed(x)
   465	
   466	        # PAPE: add pose patch embedding (early pose injection)
   467	        if getattr(self, 'use_pose_patch_embed', False) and scene_heatmaps is not None:
   468	            H_hw, W_hw = hw_shape
   469	            # Resize heatmaps to match post-PatchEmbed spatial dims
   470	            hm = F.interpolate(scene_heatmaps, size=(H_hw, W_hw),
   471	                               mode='bilinear', align_corners=False)
   472	            pose_tokens = self.pose_patch_embed(hm)  # (B, C, H, W)
   473	            pose_tokens = pose_tokens.flatten(2).transpose(1, 2)  # (B, N, C)
   474	            x = x + pose_tokens.to(x.dtype)  # AMP safety
   475	
   476	        # Pose Prompt: KPR-style argmax part ID → learnable embedding → additive
   477	        if getattr(self, 'use_pose_prompt', False) and scene_heatmaps is not None:
   478	            H_hw, W_hw = hw_shape
   479	            # Resize 17-channel heatmaps to patch resolution
   480	            hm = F.interpolate(scene_heatmaps, size=(H_hw, W_hw),
   481	                               mode='bilinear', align_corners=False)  # (B, 17, H, W)
   482	            # Heatmaps are already [0,1] (ViTPose MSE-trained output, not logits)
   483	            # Only clamp float16 rounding artifacts (tiny negatives)
   484	            hm = hm.clamp(min=0)
   485	            # Background channel: 1 - max keypoint confidence
   486	            bg = 1.0 - hm.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
   487	            hm_with_bg = torch.cat([bg, hm], dim=1)  # (B, 18, H, W)
   488	            # Argmax → part ID per patch (detach: no gradient through heatmaps)
   489	            part_ids = hm_with_bg.detach().argmax(dim=1)  # (B, H, W) values in [0, 17]
   490	            part_ids = part_ids.reshape(part_ids.shape[0], -1)  # (B, N)
   491	            # Stochastic prompt drop during training (use empty prompt = all background)
   492	            if self.training and self.pose_prompt_drop > 0:
   493	                drop_mask = torch.rand(part_ids.shape[0], 1, device=part_ids.device) < self.pose_prompt_drop
   494	                part_ids = torch.where(drop_mask.expand_as(part_ids),
   495	                                       torch.zeros_like(part_ids), part_ids)  # 0 = background
   496	            # Lookup learnable embeddings, scale, and add to patch tokens
   497	            prompt_embeds = self.pose_prompt_embed(part_ids)  # (B, N, C)
   498	            prompt_embeds = prompt_embeds.to(x.dtype)  # AMP safety: match float16/32
   499	            scale = torch.sigmoid(self.pose_prompt_scale)  # learnable injection strength
   500	            x = x + scale * prompt_embeds
   501	
   502	        if self.base.use_abs_pos_embed:
   503	            x = x + self.base.absolute_pos_embed
   504	        x = self.base.drop_after_pos(x)
   505	
   506	        # Build semantic weight tensor
   507	        sw_val = self._semantic_weight_val
   508	        if self.base.semantic_weight >= 0:
   509	            w = torch.ones(x.shape[0], 1, device=x.device) * sw_val
   510	            w = torch.cat([w, 1 - w], dim=-1)
   511	            sem_weight = w
   512	        else:
   513	            sem_weight = None
   514	
   515	        outs = []
   516	        for i, stage in enumerate(self.base.stages):
   517	            if i in self.psg_stage_indices:
   518	                # Stage with PSG: manually run blocks with injection
   519	                x, hw_shape, out, out_hw_shape = self._run_stage_with_psg(
   520	                    stage, x, hw_shape, scene_heatmaps, stage_idx=i)
   521	            else:
   522	                # Normal stage: run without modification
   523	                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
   524	
   525	            # Apply semantic weight (from SOLIDER pretraining)
   526	            if sem_weight is not None:
   527	                sw = self.base.semantic_embed_w[i](sem_weight).unsqueeze(1)
   528	                sb = self.base.semantic_embed_b[i](sem_weight).unsqueeze(1)
   529	                x = x * self.base.softplus(sw) + sb
   530	
   531	            if i in self.base.out_indices:
   532	                norm_layer = getattr(self.base, f'norm{i}')
   533	                out = norm_layer(out)
   534	                out = out.view(-1, *out_hw_shape,
   535	                               self.base.num_features[i]).permute(0, 3, 1,
   536	                                                                   2).contiguous()
   537	                outs.append(out)
   538	
   539	        # Pooling
   540	        featmap = outs[-1]  # (B, C, fH, fW)

 succeeded in 0ms:
   680	        mask = (region_of_token.unsqueeze(0) == sel.unsqueeze(1))        # (B, HW) bool
   681	        mt = self.pcmsc_mask_token.view(1, 1, C).to(tokens.dtype)
   682	        tok_masked = torch.where(mask.unsqueeze(-1), mt, tokens)
   683	        # decoder: the selected region's query reconstructs from the (masked) token set
   684	        q = self.pcmsc_query[sel].unsqueeze(1).to(tokens.dtype)          # (B,1,C)
   685	        R = self.pcmsc_decoder(q, tok_masked, tok_masked)[0].squeeze(1)  # (B, C)
   686	        R = F.normalize(self.pcmsc_proj(R).float(), dim=-1)             # (B, clip_dim) fp32
   687	        tgt = target[torch.arange(B, device=device), sel]               # (B, clip_dim)
   688	        cos = (R * tgt).sum(-1)
   689	        loss = (1.0 - cos).mean()
   690	        if not getattr(self, '_pcmsc_logged', False):
   691	            self._pcmsc_logged = True
   692	            print('[PC-MSC] first-call diag: sel-region hist %s, mean cos %.3f'
   693	                  % (torch.bincount(sel, minlength=3).tolist(), float(cos.mean())))
   694	        return self.pcmsc_w * loss
   695	
   696	    def _lgpa_heatmap(self, scene_heatmaps, B, device):
   697	        """Select the heatmap fed to the LGPA head: None (no-pose), fixed canonical
   698	        (fixed-bands), or per-image scene heatmaps (default)."""
   699	        if getattr(self, '_lgpa_no_pose', False):
   700	            return None
   701	        if getattr(self, '_lgpa_fixed_bands', False):
   702	            return self._canonical_heatmap(B, device)
   703	        return scene_heatmaps
   704	
   705	    def forward(self, x, label=None, cam_label=None, view_label=None,
   706	                pose_dict=None):
   707	        # Prepare pose
   708	        scene_heatmaps = None
   709	        target_heatmaps = None
   710	        if pose_dict is not None:
   711	            scene_heatmaps, _, target_heatmaps, _ = self._prepare_pose(pose_dict)
   712	            # exp357 pose-shuffle kill-switch: training-only cross-image permutation of the pose
   713	            # within the batch (each image gets ANOTHER image's real pose). Tests whether the
   714	            # CORRECT pose spatial content is causal for the LGPA gain. Test path uses true pose.
   715	            if self.training and getattr(self, 'use_pose_shuffle', False) and scene_heatmaps is not None:
   716	                Bp = scene_heatmaps.shape[0]
   717	                if Bp > 1:
   718	                    # derangement: NO image keeps its own pose (Codex: randperm leaves ~1 fixed point)
   719	                    ar = torch.arange(Bp, device=scene_heatmaps.device)
   720	                    perm = torch.randperm(Bp, device=ar.device)
   721	                    tries = 0
   722	                    while bool((perm == ar).any()) and tries < 8:
   723	                        perm = torch.randperm(Bp, device=ar.device); tries += 1
   724	                    if bool((perm == ar).any()):
   725	                        perm = torch.roll(ar, 1, 0)     # guaranteed-derangement fallback (cyclic shift)
   726	                    scene_heatmaps = scene_heatmaps[perm]
   727	                    if target_heatmaps is not None:
   728	                        target_heatmaps = target_heatmaps[perm]
   729	            # exp358 cross-PART (channel) shuffle: per-image permutation of the K keypoint
   730	            # channels — destroys anatomical part identity while keeping each image's own spatial
   731	            # support. Complements exp357 (cross-image): together they isolate whether the LGPA
   732	            # gain needs correct pose-image correspondence and/or correct anatomical part assignment.
   733	            if self.training and getattr(self, 'use_pose_channel_shuffle', False) and scene_heatmaps is not None:
   734	                Kc = scene_heatmaps.shape[1]
   735	                cperm = torch.argsort(torch.rand(scene_heatmaps.shape[0], Kc, device=scene_heatmaps.device), dim=1)
   736	                idx = cperm[:, :, None, None].expand(-1, -1, scene_heatmaps.shape[2], scene_heatmaps.shape[3])
   737	                scene_heatmaps = torch.gather(scene_heatmaps, 1, idx)
   738	                if target_heatmaps is not None:
   739	                    target_heatmaps = torch.gather(target_heatmaps, 1, idx)
   740	
   741	        # Target-only heatmap swap (multi-person disambiguation).
   742	        # Substitute scene_heatmaps with target_heatmaps so all downstream
   743	        # pose-aware modules (PSG/LGPA/VCSR/PPA/STR/FSDC/etc.) receive the
   744	        # target-person (index 0) signal instead of max-merged scene.
   745	        # No other code path is touched: when use_target_heatmap is False
   746	        # (default), scene_heatmaps keeps its original max-merged value.
   747	        if self.use_target_heatmap and target_heatmaps is not None:
   748	            scene_heatmaps = target_heatmaps
   749	
   750	        # Stochastic Pose Dropout: zero out heatmaps per-sample during training
   751	        if self.training and scene_heatmaps is not None and self.pose_dropout_p > 0:
   752	            keep_mask = (torch.rand(scene_heatmaps.shape[0], 1, 1, 1,
   753	                                    device=scene_heatmaps.device) >= self.pose_dropout_p)
   754	            scene_heatmaps = scene_heatmaps * keep_mask.float()
   755	
   756	        # Run backbone with PSG injection
   757	        global_feat, featmaps = self._run_backbone_with_psg(x, scene_heatmaps)
   758	
   759	        if self.reduce_feat_dim:
   760	            global_feat = self.fcneck(global_feat)
   761	
   762	        feat = self.bottleneck(global_feat)
   763	
   764	        if self.training:
   765	            feat_cls = self.dropout(feat)
   766	            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
   767	                cls_score = self.classifier(feat_cls, label)
   768	            else:
   769	                cls_score = self.classifier(feat_cls)
   770	
   771	            # CLIP-ReID ID-prompt contrastive (the WORKING CLIP mechanism): align global feat
   772	            # to per-ID learnable text prototypes via SupCon i2t/t2i.
   773	            clip_id_loss = None
   774	            if getattr(self, 'use_clip_id_prompt', False) and label is not None:
   775	                from .modules.clip_id_prompt import supcon_i2t
   776	                # Option B: per-image pose conditions the prompt (pose_vec None unless pose_cond)
   777	                pose_vec = scene_heatmaps.float().mean(dim=(2, 3)) \
   778	                    if (getattr(self.clip_id_prompt, 'pose_cond', False) and scene_heatmaps is not None) else None
   779	                txt_proto = self.clip_id_prompt(label, pose_vec)  # (B, clip_dim)
   780	                t = self.clip_id_temp
   781	                if getattr(self, 'use_clip_id_part_guided', False) and scene_heatmaps is not None:
   782	                    # Option C: K pose-localized part features, each aligned to the ID prototype
   783	                    part_feats = self.pose_guided_part_pool(featmaps[-1], scene_heatmaps)  # (B, nP, C)
   784	                    clip_id_loss = 0.0
   785	                    for kp in range(part_feats.shape[1]):
   786	                        ipk = self.clip_id_proj(part_feats[:, kp])
   787	                        clip_id_loss = clip_id_loss + supcon_i2t(ipk, txt_proto, label, t) \
   788	                            + supcon_i2t(txt_proto, ipk, label, t)
   789	                    clip_id_loss = clip_id_loss / part_feats.shape[1]
   790	                else:
   791	                    # exp347 (param-free de-occluded) / Option A (pose-guided pooled) / exp341 (raw global)
   792	                    if getattr(self, 'use_clip_id_noparam_pool', False) and scene_heatmaps is not None:
   793	                        feat_for_clip = self.pose_weighted_pool(featmaps[-1], scene_heatmaps)
   794	                    elif getattr(self, 'use_clip_id_pose_guided', False) and scene_heatmaps is not None:
   795	                        feat_for_clip = self.pose_guided_pool(featmaps[-1], scene_heatmaps)
   796	                    else:
   797	                        feat_for_clip = global_feat
   798	                    img_proj = self.clip_id_proj(feat_for_clip)   # (B, clip_dim)
   799	                    clip_id_loss = supcon_i2t(img_proj, txt_proto, label, t) \
   800	                        + supcon_i2t(txt_proto, img_proj, label, t)
   801	                    # exp355 PGPD: pose selects a more-complete same-ID teacher; distill its
   802	                    # soft distribution over the batch's other-ID prototypes to this student.
   803	                    if getattr(self, 'use_pgpd', False) and scene_heatmaps is not None:
   804	                        clip_id_loss = clip_id_loss + self._pgpd_loss(img_proj, txt_proto, label, scene_heatmaps, target_heatmaps)
   805	                    # exp348: occluder repulsion — push the occluder-region (low-visibility) feature
   806	                    # away from the ID prototype (penalize only positive similarity → make it neutral).
   807	                    if getattr(self, 'use_clip_id_occ_repel', False) and scene_heatmaps is not None:
   808	                        occ_feat = self.pose_weighted_pool(featmaps[-1], scene_heatmaps, invert=True)
   809	                        occ_proj = torch.nn.functional.normalize(self.clip_id_proj(occ_feat), dim=1)
   810	                        tp = torch.nn.functional.normalize(txt_proto, dim=1)
   811	                        repel = (occ_proj * tp).sum(1).clamp(min=0).mean()
   812	                        clip_id_loss = clip_id_loss + self.clip_id_occ_repel_w * repel
   813	
   814	            # exp356 PC-MSC: pose-masked CLIP-semantic completion (training-only regularizer)
   815	            if getattr(self, 'use_pcmsc', False) and scene_heatmaps is not None and self.training:
   816	                pcmsc = self._pcmsc_loss(featmaps[-1], x, scene_heatmaps)
   817	                clip_id_loss = pcmsc if clip_id_loss is None else clip_id_loss + pcmsc
   818	
   819	            # VCSR: Visibility-Conditional Semantic Routing (detached)
   820	            if getattr(self, 'use_vcsr', False) and scene_heatmaps is not None:
   821	                vcsr_input = featmaps[-1].detach()
   822	                vcsr_cls_scores, vcsr_feats, vcsr_data = self.vcsr_head(
   823	                    vcsr_input, scene_heatmaps, return_cls=True)
   824	                kp_data = vcsr_data
   825	                return [cls_score] + vcsr_cls_scores, [global_feat] + vcsr_feats, featmaps, None, kp_data
   826	
   827	            # Part branch: STD-PR (structural tokens) only — when GCN is NOT also enabled
   828	            elif getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None \
   829	                    and not self.use_skeleton_gcn:
   830	                feat_map_detached = featmaps[-1].detach()
   831	                B_fm, C_fm, H_fm, W_fm = feat_map_detached.shape
   832	                spatial_tokens = feat_map_detached.flatten(2).transpose(1, 2)  # (B, H*W, C)
   833	                # Pass keypoints for anchor-sampled query initialization
   834	                kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
   835	                sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
   836	                router_out = self.structural_router(
   837	                    spatial_tokens, (H_fm, W_fm), scene_heatmaps,
   838	                    keypoints=kp_p0, scores=sc_p0,
   839	                    input_size=tuple(x.shape[2:]))
   840	                # Unpack: with self-attn returns (refined, stats, raw), without returns (tokens, stats)
   841	                if getattr(self, 'str_self_attn', False):
   842	                    structural_tokens, str_stats, raw_tokens = router_out
   843	                else:
   844	                    structural_tokens, str_stats = router_out
   845	                    raw_tokens = structural_tokens
   846	                # PLTD: Part-Level Token Dropout — randomly zero out tokens during training
   847	                part_drop_p = getattr(self, 'str_part_drop', 0.0)
   848	                if self.training and part_drop_p > 0:
   849	                    B_tok, K_tok, C_tok = structural_tokens.shape
   850	                    # Each token independently dropped with probability p
   851	                    # Ensure at least 2 tokens survive per sample
   852	                    drop_mask = torch.rand(B_tok, K_tok, 1, device=structural_tokens.device) >= part_drop_p
   853	                    # Guarantee minimum 2 tokens survive
   854	                    alive = drop_mask.squeeze(-1).sum(dim=1)  # (B,)
   855	                    for b_idx in range(B_tok):
   856	                        if alive[b_idx] < 2:
   857	                            # Randomly revive tokens until we have 2
   858	                            dead = (~drop_mask[b_idx].squeeze(-1)).nonzero(as_tuple=True)[0]
   859	                            revive = dead[torch.randperm(len(dead))[:2 - int(alive[b_idx].item())]]
   860	                            drop_mask[b_idx, revive] = True
   861	                    structural_tokens = structural_tokens * drop_mask.float()
   862	                    raw_tokens = raw_tokens * drop_mask.float()
   863	                    n_dropped = (1 - drop_mask.float()).sum() / (B_tok * K_tok)
   864	                    str_stats['pltd_drop'] = n_dropped.item()
   865	                # Part feature: confidence-weighted pooling from heatmap response
   866	                K_str = structural_tokens.shape[1]
   867	                if K_str == 6:
   868	                    # 6-part groups: compute per-part heatmap visibility
   869	                    _pg = [[0,1,2,3,4],[5,6,11,12],[5,7,9],[6,8,10],[11,13,15],[12,14,16]]
   870	                    hm_r = F.interpolate(scene_heatmaps, size=(H_fm, W_fm),
   871	                                        mode='bilinear', align_corners=False)
   872	                    pw = []
   873	                    for g in _pg:
   874	                        pw.append(hm_r[:, g].mean(dim=(1,2,3)))  # (B,)
   875	                    part_w = torch.stack(pw, dim=1)  # (B, 6)
   876	                    part_w = part_w / part_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
   877	                    str_feat = (structural_tokens * part_w.unsqueeze(2)).sum(dim=1)
   878	                else:
   879	                    str_feat = structural_tokens.mean(dim=1)  # fallback
   880	                # Per-token or pooled classification
   881	                if getattr(self, 'str_per_token', False):
   882	                    # DPTL dual-path: CE on raw tokens (diversity), triplet on refined tokens (coherence)
   883	                    str_cls_list = []
   884	                    str_feat_list = []
   885	                    # CE path uses raw tokens (independent, diverse)
   886	                    ce_tokens = raw_tokens
   887	                    # Triplet/test path uses refined tokens (contextualized)
   888	                    tri_tokens = structural_tokens
   889	                    for k in range(ce_tokens.shape[1]):
   890	                        tok_k = ce_tokens[:, k]  # (B, C)
   891	                        tok_bn = self.structural_router.part_bn(tok_k)
   892	                        str_cls_list.append(self.str_classifier(tok_bn))
   893	                        str_feat_list.append(tri_tokens[:, k])  # refined for triplet
   894	                    kp_data = {'str_stats': str_stats}
   895	                    if K_str == 6:
   896	                        kp_data['part_visibility'] = part_w  # (B, 6) per-part visibility weights
   897	                    return [cls_score] + str_cls_list, [global_feat] + str_feat_list, featmaps, None, kp_data
   898	                else:
   899	                    # Pooled: all tokens averaged
   900	                    str_feat_bn = self.structural_router.part_bn(str_feat)
   901	                    str_cls = self.str_classifier(str_feat_bn)
   902	                    kp_data = {'str_stats': str_stats}
   903	                    return [cls_score, str_cls], [global_feat, str_feat], featmaps, None, kp_data
   904	
   905	            elif getattr(self, 'use_lgpa', False) and (scene_heatmaps is not None or getattr(self, '_lgpa_fixed_bands', False)):
   906	                # LGPA: CLIP cross-attention part assignment
   907	                lgpa_input = featmaps[-1].detach() if getattr(self, '_lgpa_detach', False) else featmaps[-1]
   908	                lgpa_hm = self._lgpa_heatmap(scene_heatmaps, x.shape[0], x.device)
   909	                lgpa_cls_scores, lgpa_feats, lgpa_data = self.clip_part_head(
   910	                    lgpa_input, lgpa_hm, return_cls=True)
   911	                kp_data = lgpa_data
   912	                if clip_id_loss is not None:
   913	                    kp_data['clip_id_loss'] = clip_id_loss   # carry CLIP-ID-prompt loss through LGPA path
   914	
   915	                # LGPA + GCN dual branch: also run GCN on detached features
   916	                if self.use_skeleton_gcn and pose_dict is not None:
   917	                    feat_map_detached = featmaps[-1].detach()
   918	                    _s2_feat = featmaps[-2].detach() if len(featmaps) >= 2 else None
   919	                    gcn_cls_scores, gcn_feats, gcn_data = self.skeleton_head(
   920	                        feat_map_detached, pose_dict, return_cls=True, label=label,
   921	                        stage2_feat=_s2_feat)
   922	                    if gcn_data and 'kp_feats' in gcn_data:
   923	                        kp_data['gcn_kp_feats'] = gcn_data['kp_feats']
   924	                        kp_data['gcn_kp_weights'] = gcn_data['kp_weights']
   925	                        if 'vcn_stats' in gcn_data:
   926	                            kp_data['vcn_stats'] = gcn_data['vcn_stats']
   927	                    return ([cls_score] + lgpa_cls_scores + gcn_cls_scores,
   928	                            [global_feat] + lgpa_feats + gcn_feats,
   929	                            featmaps, None, kp_data)
   930	
   931	                return [cls_score] + lgpa_cls_scores, [global_feat] + lgpa_feats, featmaps, None, kp_data
   932	
   933	            elif getattr(self, 'use_ppa', False) and scene_heatmaps is not None:
   934	                # PPA: Pose-Prompted Part-Assignment Head (end-to-end, NOT detached)
   935	                ppa_cls_scores, ppa_feats, ppa_data = self.part_assignment_head(
   936	                    featmaps[-1], scene_heatmaps, return_cls=True)
   937	                kp_data = ppa_data
   938	
   939	                # PPA + GCN dual branch: also run GCN on detached features
   940	                if self.use_skeleton_gcn and pose_dict is not None:
   941	                    feat_map_detached = featmaps[-1].detach()
   942	                    _s2_feat = featmaps[-2].detach() if len(featmaps) >= 2 else None
   943	                    gcn_cls_scores, gcn_feats, gcn_data = self.skeleton_head(
   944	                        feat_map_detached, pose_dict, return_cls=True, label=label,
   945	                        stage2_feat=_s2_feat)
   946	                    # Merge: PPA kp_data takes priority, add GCN kp_feats for MaxSim
   947	                    if gcn_data and 'kp_feats' in gcn_data:
   948	                        kp_data['gcn_kp_feats'] = gcn_data['kp_feats']
   949	                        kp_data['gcn_kp_weights'] = gcn_data['kp_weights']
   950	                    return ([cls_score] + ppa_cls_scores + gcn_cls_scores,
   951	                            [global_feat] + ppa_feats + gcn_feats,
   952	                            featmaps, None, kp_data)
   953	
   954	                return [cls_score] + ppa_cls_scores, [global_feat] + ppa_feats, featmaps, None, kp_data
   955	
   956	            elif self.use_skeleton_gcn and pose_dict is not None:
   957	                # GSPB: Gradient-scaled part branch
   958	                # scale=0 → detach (default), scale=1 → non-detach
   959	                _gs = getattr(self, '_part_grad_scale', 0.0)
   960	                if _gs > 0:
   961	                    feat_map_detached = featmaps[-1].detach() + _gs * (featmaps[-1] - featmaps[-1].detach())
   962	                else:
   963	                    feat_map_detached = featmaps[-1].detach()
   964	
   965	                # Dual Part Branch: also run STD-PR for per-token SupCon if both are enabled
   966	                dual_branch_active = False
   967	                if getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None \
   968	                        and getattr(self, 'str_per_token', False):
   969	                    B_fm, C_fm, H_fm, W_fm = feat_map_detached.shape
   970	                    spatial_tokens = feat_map_detached.flatten(2).transpose(1, 2)
   971	                    kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
   972	                    sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
   973	                    router_out = self.structural_router(
   974	                        spatial_tokens, (H_fm, W_fm), scene_heatmaps,
   975	                        keypoints=kp_p0, scores=sc_p0,
   976	                        input_size=tuple(x.shape[2:]))
   977	                    if getattr(self, 'str_self_attn', False):
   978	                        structural_tokens, str_stats, raw_tokens = router_out
   979	                    else:
   980	                        structural_tokens, str_stats = router_out
   981	                        raw_tokens = structural_tokens
   982	                    # Per-token classification for SupCon
   983	                    ce_tokens = raw_tokens
   984	                    tri_tokens = structural_tokens
   985	                    str_cls_list = []
   986	                    str_feat_list = []
   987	                    for k in range(ce_tokens.shape[1]):
   988	                        tok_k = ce_tokens[:, k]
   989	                        tok_bn = self.structural_router.part_bn(tok_k)
   990	                        str_cls_list.append(self.str_classifier(tok_bn))
   991	                        str_feat_list.append(tri_tokens[:, k])
   992	                    dual_branch_active = True
   993	
   994	                # FSDC: Feature-Space Diffusion Completion
   995	                fsdc_loss = None
   996	                if getattr(self, 'use_fsdc', False):
   997	                    B_d, C_d, H_d, W_d = feat_map_detached.shape
   998	                    spatial_tokens = feat_map_detached.flatten(2).transpose(1, 2)  # (B, N, C)
   999	                    completed_tokens, fsdc_loss, fsdc_stats = self.feature_denoiser(
  1000	                        spatial_tokens, scene_heatmaps, fH=H_d, fW=W_d)
  1001	                    # Reshape back to feature map
  1002	                    feat_map_detached = completed_tokens.transpose(1, 2).reshape(B_d, C_d, H_d, W_d)
  1003	
  1004	                # Pass Stage 2 features for KAMP/MRKF multi-scale fusion
  1005	                _s2_feat = featmaps[-2].detach() if len(featmaps) >= 2 else None
  1006	                gcn_cls_scores, gcn_feats, kp_data = self.skeleton_head(
  1007	                    feat_map_detached, pose_dict, return_cls=True, label=label,
  1008	                    stage2_feat=_s2_feat)
  1009	                # Store FSDC loss in kp_data for processor
  1010	                if fsdc_loss is not None:
  1011	                    if kp_data is None:
  1012	                        kp_data = {}
  1013	                    kp_data['fsdc_loss'] = fsdc_loss
  1014	                    kp_data['fsdc_stats'] = fsdc_stats
  1015	
  1016	                # PNIS: normalize GCN feature by subtracting pose offset
  1017	                if getattr(self, 'use_pose_normalize', False) and len(gcn_feats) > 0:
  1018	                    kp_coords = pose_dict['keypoints'][:, 0, :, :]  # (B, 17, 2) person 0
  1019	                    kp_scores = pose_dict['scores'][:, 0, :]        # (B, 17) person 0
  1020	                    # Normalize coordinates to [0,1]
  1021	                    img_h, img_w = x.shape[2], x.shape[3]
  1022	                    kp_coords_norm = kp_coords.clone()
  1023	                    kp_coords_norm[:, :, 0] = kp_coords_norm[:, :, 0] / max(img_w, 1)
  1024	                    kp_coords_norm[:, :, 1] = kp_coords_norm[:, :, 1] / max(img_h, 1)
  1025	                    identity_feat, pn_stats = self.pose_normalizer(
  1026	                        gcn_feats[0], kp_coords_norm, kp_scores)
  1027	                    gcn_feats[0] = identity_feat
  1028	                    if kp_data is None:
  1029	                        kp_data = {}
  1030	                    kp_data['pn_stats'] = pn_stats
  1031	
  1032	                # SPLADE: auxiliary sparse classification (does NOT modify gcn lists)
  1033	                if getattr(self, 'use_splade', False) and len(gcn_feats) > 0:
  1034	                    sparse_feat, sparsity = self.sparse_head(gcn_feats[0])
  1035	                    sparse_cls = self.sparse_classifier(sparse_feat)
  1036	                    if kp_data is None:
  1037	                        kp_data = {}
  1038	                    kp_data['splade_cls'] = sparse_cls      # separate CE loss in processor
  1039	                    kp_data['splade_sparsity'] = sparsity
  1040	                    kp_data['splade_reg'] = sparse_feat.mean()  # sparsity regularization
  1041	
  1042	                # Dual Part Branch: combine GCN + STD-PR per-token outputs
  1043	                if dual_branch_active:
  1044	                    # Return: [global, str_tok1..6, gcn] for both scores and feats
  1045	                    # SupCon operates on str_tok1..6, GCN provides architecture via gcn
  1046	                    if kp_data is None:
  1047	                        kp_data = {}
  1048	                    kp_data['str_stats'] = str_stats
  1049	                    kp_data['num_str_tokens'] = len(str_feat_list)  # SupCon uses feat[1:1+num_str_tokens]
  1050	                    # part_visibility for STD-PR tokens
  1051	                    K_str = len(str_feat_list)
  1052	                    if K_str == 6:
  1053	                        _pg = [[0,1,2,3,4],[5,6,11,12],[5,7,9],[6,8,10],[11,13,15],[12,14,16]]
  1054	                        hm_r = F.interpolate(scene_heatmaps, size=(featmaps[-1].shape[2], featmaps[-1].shape[3]),
  1055	                                            mode='bilinear', align_corners=False)
  1056	                        pw = [hm_r[:, g].mean(dim=(1,2,3)) for g in _pg]
  1057	                        part_w = torch.stack(pw, dim=1)
  1058	                        part_w = part_w / part_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
  1059	                        kp_data['part_visibility'] = part_w
  1060	                    return ([cls_score] + str_cls_list + gcn_cls_scores,
  1061	                            [global_feat] + str_feat_list + gcn_feats,
  1062	                            featmaps, None, kp_data)
  1063	
  1064	                # BA-PKC: sample keypoint features from NON-detached feature map
  1065	                # Gradients flow to backbone, unlike GCN which uses detached features
  1066	                if getattr(self, 'ba_pkc', False) or getattr(self, 'bt_pkd', False):
  1067	                    raw_fm = featmaps[-1]  # (B, C, fH, fW) — NOT detached!
  1068	                    kp_coords = pose_dict['keypoints'][:, 0, :, :]  # (B, 17, 2)
  1069	                    input_h, input_w = x.shape[2], x.shape[3]
  1070	                    grid_x = (kp_coords[:, :, 0] / input_w * 2 - 1).clamp(-1, 1)
  1071	                    grid_y = (kp_coords[:, :, 1] / input_h * 2 - 1).clamp(-1, 1)
  1072	                    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, 17, 1, 2)
  1073	                    sampled = F.grid_sample(raw_fm, grid, mode='bilinear',
  1074	                                            padding_mode='border', align_corners=True)
  1075	                    ba_kp_feats = sampled.squeeze(-1).permute(0, 2, 1)  # (B, 17, C)
  1076	                    if kp_data is None:
  1077	                        kp_data = {}
  1078	                    if getattr(self, 'ba_pkc', False):
  1079	                        kp_data['ba_kp_feats'] = ba_kp_feats
  1080	                    if getattr(self, 'bt_pkd', False):
  1081	                        kp_data['bt_kp_feats'] = ba_kp_feats  # non-detached for distillation
  1082	
  1083	                # Return lists -> triggers list-loss path (implicit 0.5x global)
  1084	                return [cls_score] + gcn_cls_scores, [global_feat] + gcn_feats, featmaps, None, kp_data
  1085	
  1086	            if clip_id_loss is not None:
  1087	                return cls_score, global_feat, featmaps, None, {'clip_id_loss': clip_id_loss}
  1088	            return cls_score, global_feat, featmaps, None
  1089	        else:
  1090	            if self.neck_feat == 'after':
  1091	                test_feat = feat
  1092	            else:
  1093	                test_feat = global_feat
  1094	
  1095	            # Part branch test features
  1096	            gcn_feats = None
  1097	            aux_data = {}
  1098	
  1099	            # VCSR test path
  1100	            if getattr(self, 'use_vcsr', False) and scene_heatmaps is not None and \
  1101	                    getattr(self, 'pose_test_feat', 'global') != 'global':
  1102	                _, vcsr_feats, aux_data = self.vcsr_head(
  1103	                    featmaps[-1], scene_heatmaps, return_cls=False)
  1104	                gcn_feats = vcsr_feats
  1105	
  1106	            # LGPA test path — uses scene_heatmaps (same as PPA for fair comparison)
  1107	            elif getattr(self, 'use_lgpa', False) and (scene_heatmaps is not None or getattr(self, '_lgpa_fixed_bands', False)) and \
  1108	                    getattr(self, 'pose_test_feat', 'global') != 'global':
  1109	                lgpa_hm = self._lgpa_heatmap(scene_heatmaps, x.shape[0], x.device)
  1110	                _, lgpa_feats, aux_data = self.clip_part_head(
  1111	                    featmaps[-1], lgpa_hm, return_cls=False)
  1112	                gcn_feats = lgpa_feats  # [pooled, part1..partK]
  1113	                # LGPA + GCN dual: also get GCN features
  1114	                if self.use_skeleton_gcn and pose_dict is not None:
  1115	                    _, gcn_only_feats, gcn_aux = self.skeleton_head(
  1116	                        featmaps[-1], pose_dict, return_cls=False)
  1117	                    gcn_feats = lgpa_feats + gcn_only_feats
  1118	                    if gcn_aux and 'kp_feats' in gcn_aux:
  1119	                        aux_data['gcn_kp_feats'] = gcn_aux['kp_feats']
  1120	                        aux_data['gcn_kp_weights'] = gcn_aux['kp_weights']
  1121	
  1122	            # PPA test path
  1123	            elif getattr(self, 'use_ppa', False) and scene_heatmaps is not None and \
  1124	                    getattr(self, 'pose_test_feat', 'global') != 'global':
  1125	                _, ppa_feats, aux_data = self.part_assignment_head(
  1126	                    featmaps[-1], scene_heatmaps, return_cls=False)
  1127	                gcn_feats = ppa_feats  # [pooled, part1..partK]
  1128	                # PPA + GCN dual: also get GCN features
  1129	                if self.use_skeleton_gcn and pose_dict is not None:
  1130	                    _, gcn_only_feats, gcn_aux = self.skeleton_head(
  1131	                        featmaps[-1], pose_dict, return_cls=False)
  1132	                    gcn_feats = ppa_feats + gcn_only_feats
  1133	                    if gcn_aux and 'kp_feats' in gcn_aux:
  1134	                        aux_data['gcn_kp_feats'] = gcn_aux['kp_feats']
  1135	                        aux_data['gcn_kp_weights'] = gcn_aux['kp_weights']
  1136	
  1137	            elif getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None and \
  1138	                    getattr(self, 'pose_test_feat', 'global') != 'global' and not self.use_skeleton_gcn:
  1139	                B_fm, C_fm, H_fm, W_fm = featmaps[-1].shape
  1140	                spatial_tokens = featmaps[-1].flatten(2).transpose(1, 2)
  1141	                kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
  1142	                sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
  1143	                router_out = self.structural_router(
  1144	                    spatial_tokens, (H_fm, W_fm), scene_heatmaps,
  1145	                    keypoints=kp_p0, scores=sc_p0,
  1146	                    input_size=tuple(x.shape[2:]))
  1147	                # Use refined tokens (first return) for test features
  1148	                structural_tokens = router_out[0]
  1149	                # Confidence-weighted pooling (same as training)
  1150	                K_str = structural_tokens.shape[1]
  1151	                if K_str == 6:
  1152	                    _pg = [[0,1,2,3,4],[5,6,11,12],[5,7,9],[6,8,10],[11,13,15],[12,14,16]]
  1153	                    hm_r = F.interpolate(scene_heatmaps, size=(H_fm, W_fm),
  1154	                                        mode='bilinear', align_corners=False)
  1155	                    pw = [hm_r[:, g].mean(dim=(1,2,3)) for g in _pg]
  1156	                    part_w = torch.stack(pw, dim=1)
  1157	                    part_w = part_w / part_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
  1158	                    str_feat = (structural_tokens * part_w.unsqueeze(2)).sum(dim=1)
  1159	                else:
  1160	                    str_feat = structural_tokens.mean(dim=1)
  1161	                if self.pose_test_feat in ('maxsim', 'maxsim_hybrid',
  1162	                                          'cvk_hybrid', 'cvk_only'):
  1163	                    # Return structural tokens as kp_feats for set matching
  1164	                    K = structural_tokens.shape[1]
  1165	                    test_feat = {
  1166	                        'mode': self.pose_test_feat,
  1167	                        'global_feat': test_feat,
  1168	                        'kp_feats': structural_tokens,  # (B, K, C)
  1169	                        'kp_weights': torch.ones(structural_tokens.shape[0], K,
  1170	                                                 device=structural_tokens.device),
  1171	                    }
  1172	                    return test_feat, featmaps
  1173	                # Per-token training uses pooled test feature (better than per-token concat)
  1174	                # Confidence-weighted pool captures the right signal; per-token concat dilutes it
  1175	                gcn_feats = [str_feat]  # equal_concat: global + pooled_part
  1176	            elif self.use_skeleton_gcn and pose_dict is not None and \
  1177	                    getattr(self, 'pose_test_feat', 'global') != 'global':
  1178	                # FSDC: complete occluded tokens at test time
  1179	                feat_for_gcn = featmaps[-1]
  1180	                if getattr(self, 'use_fsdc', False) and scene_heatmaps is not None:
  1181	                    B_d, C_d, H_d, W_d = feat_for_gcn.shape
  1182	                    spatial_tokens = feat_for_gcn.flatten(2).transpose(1, 2)
  1183	                    completed, _, _ = self.feature_denoiser(
  1184	                        spatial_tokens, scene_heatmaps, fH=H_d, fW=W_d)
  1185	                    feat_for_gcn = completed.transpose(1, 2).reshape(B_d, C_d, H_d, W_d)
  1186	                _s2_test = featmaps[-2] if len(featmaps) >= 2 else None
  1187	                _, gcn_feats, aux_data = self.skeleton_head(
  1188	                    feat_for_gcn, pose_dict, return_cls=False,
  1189	                    stage2_feat=_s2_test)
  1190	                # Dual Part Branch test: also add STD-PR per-token features
  1191	                if getattr(self, 'use_structural_routing', False) and scene_heatmaps is not None \
  1192	                        and getattr(self, 'str_per_token', False):
  1193	                    B_fm, C_fm, H_fm, W_fm = featmaps[-1].shape
  1194	                    spatial_tokens = featmaps[-1].flatten(2).transpose(1, 2)
  1195	                    kp_p0 = pose_dict['keypoints'][:, 0] if pose_dict is not None else None
  1196	                    sc_p0 = pose_dict['scores'][:, 0] if pose_dict is not None else None
  1197	                    router_out = self.structural_router(
  1198	                        spatial_tokens, (H_fm, W_fm), scene_heatmaps,
  1199	                        keypoints=kp_p0, scores=sc_p0,
  1200	                        input_size=tuple(x.shape[2:]))
  1201	                    structural_tokens = router_out[0]
  1202	                    # Add each structural token to gcn_feats for equal_concat
  1203	                    for k in range(structural_tokens.shape[1]):
  1204	                        gcn_feats.append(structural_tokens[:, k])
  1205	                # PNIS: normalize test features too
  1206	                if getattr(self, 'use_pose_normalize', False) and gcn_feats is not None and len(gcn_feats) > 0:
  1207	                    kp_coords = pose_dict['keypoints'][:, 0, :, :]
  1208	                    kp_scores = pose_dict['scores'][:, 0, :]
  1209	                    img_h, img_w = x.shape[2], x.shape[3]
  1210	                    kp_coords_norm = kp_coords.clone()
  1211	                    kp_coords_norm[:, :, 0] /= max(img_w, 1)
  1212	                    kp_coords_norm[:, :, 1] /= max(img_h, 1)
  1213	                    identity_feat, _ = self.pose_normalizer(gcn_feats[0], kp_coords_norm, kp_scores)
  1214	                    gcn_feats[0] = identity_feat
  1215	                # SPLADE: training-only auxiliary, no test-time feature change
  1216	
  1217	            # Assemble test features from global + part branch
  1218	            if gcn_feats is not None:
  1219	                if self.pose_test_feat == 'gcn_only':
  1220	                    test_feat = torch.cat(gcn_feats, dim=1)
  1221	                elif self.pose_test_feat == 'equal_concat':
  1222	                    g_norm = F.normalize(test_feat, p=2, dim=1)
  1223	                    p_norm = [F.normalize(f, p=2, dim=1) for f in gcn_feats]
  1224	                    test_feat = torch.cat([g_norm] + p_norm, dim=1)
  1225	                elif self.pose_test_feat == 'part_only':

codex
`PoseBackboneModel.forward` 的训练路径确认到：`pose_dict=None` 时 `scene_heatmaps` 保持 `None`，训练默认在没有 CLIP/pose 分支触发时返回 `(cls_score, global_feat, featmaps, None)`；LGPA 只有 `scene_heatmaps is not None` 或 `_lgpa_fixed_bands` 打开才会进。现在我会补查配置默认值、kill-switch 生成函数、load/save 兼容和 sampler 细节。
exec
/bin/zsh -lc "nl -ba model/pose_backbone_model.py | sed -n '140,215p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   140	            print('[exp358] POSE_CHANNEL_SHUFFLE kill-switch ON: training-only per-image keypoint-channel permutation')
   141	        self.use_target_heatmap = getattr(cfg.MODEL, 'POSE_USE_TARGET_HEATMAP', False)
   142	        if self.use_target_heatmap:
   143	            print('[POSE] POSE_USE_TARGET_HEATMAP=True: '
   144	                  'pose modules (PSG/LGPA/VCSR/PPA/STR/FSDC/...) will receive '
   145	                  'person-0 (target) heatmap instead of max-merged scene heatmap.')
   146	
   147	        # GSPB: Gradient-Scaled Part Branch
   148	        self._part_grad_scale = float(getattr(cfg.MODEL, 'POSE_PART_GRAD_SCALE', 0.0))
   149	        if self._part_grad_scale > 0:
   150	            print(f'[GSPB] Part branch gradient scale: {self._part_grad_scale}')
   151	
   152	        # BA-PKC: Backbone-Aware Per-Keypoint Contrastive
   153	        self.ba_pkc = getattr(cfg.MODEL, 'POSE_BA_PKC', False)
   154	        if self.ba_pkc:
   155	            print('[BA-PKC] Backbone-aware per-keypoint contrastive enabled')
   156	
   157	        # BT-PKD: Backbone-Through Per-Keypoint Distillation
   158	        self.bt_pkd = getattr(cfg.MODEL, 'POSE_BT_PKD', False)
   159	        if self.bt_pkd:
   160	            print('[BT-PKD] Backbone-through per-keypoint distillation enabled')
   161	
   162	        # VCSR: Visibility-Conditional Semantic Routing (dynamic part gating)
   163	        self.use_vcsr = getattr(cfg.MODEL, 'POSE_VCSR', False)
   164	        if self.use_vcsr:
   165	            from .modules.vcsr_head import VCSRHead
   166	            self.vcsr_head = VCSRHead(
   167	                feat_dim=self.in_planes,
   168	                num_classes=num_classes,
   169	                clip_dim=int(getattr(cfg.MODEL, 'POSE_LGPA_CLIP_DIM', 512)),
   170	                num_heads=int(getattr(cfg.MODEL, 'POSE_LGPA_NUM_HEADS', 8)),
   171	                pose_mask_temp=float(getattr(cfg.MODEL, 'POSE_LGPA_POSE_TEMP', 1.0)),
   172	                vis_threshold=float(getattr(cfg.MODEL, 'POSE_VCSR_VIS_THR', 0.3)),
   173	            )
   174	            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
   175	
   176	        # LGPA: Language-Grounded Part Assignment (CLIP + cross-attention + pose)
   177	        self.use_lgpa = getattr(cfg.MODEL, 'POSE_LGPA', False)
   178	        if self.use_lgpa and getattr(cfg.MODEL, 'POSE_PPA', False):
   179	            raise ValueError('POSE_LGPA and POSE_PPA cannot both be enabled')
   180	        if self.use_lgpa:
   181	            from .modules.clip_part_head import CLIPPartHead
   182	            self.clip_part_head = CLIPPartHead(
   183	                feat_dim=self.in_planes,
   184	                num_classes=num_classes,
   185	                clip_dim=int(getattr(cfg.MODEL, 'POSE_LGPA_CLIP_DIM', 512)),
   186	                num_heads=int(getattr(cfg.MODEL, 'POSE_LGPA_NUM_HEADS', 8)),
   187	                pose_mask_temp=float(getattr(cfg.MODEL, 'POSE_LGPA_POSE_TEMP', 1.0)),
   188	            )
   189	            self._lgpa_detach = getattr(cfg.MODEL, 'POSE_LGPA_DETACH', False)
   190	            self._lgpa_no_pose = getattr(cfg.MODEL, 'POSE_LGPA_NO_POSE', False)
   191	            self._lgpa_fixed_bands = getattr(cfg.MODEL, 'POSE_LGPA_FIXED_BANDS', False)
   192	            self._canon_hm_cache = None  # (1,17,H,W) canonical pedestrian pose, built lazily
   193	            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
   194	            if self._lgpa_detach:
   195	                print('[LGPA] Running on DETACHED features (no gradient to backbone)')
   196	            if self._lgpa_no_pose:
   197	                print('[LGPA] NO-POSE ablation: heatmaps=None -> no pose-bias/assign/visibility (pure CLIP-text parts)')
   198	            if self._lgpa_fixed_bands:
   199	                print('[LGPA] FIXED-BANDS: per-image pose replaced by a FIXED canonical pedestrian pose '
   200	                      '(fixed CLIP text + fixed anatomical prior, NO per-image pose)')
   201	            if getattr(cfg.MODEL, 'POSE_LGPA_RANDOM_TEXT', False):
   202	                # Attribution ablation: replace CLIP text prototypes with FIXED random unit vectors.
   203	                # If part_only(random) ~= part_only(CLIP), the CLIP semantics contribute ~0 (shell).
   204	                _g = torch.Generator().manual_seed(42)
   205	                _rand = F.normalize(torch.randn(
   206	                    self.clip_part_head.num_labels, self.clip_part_head.clip_dim,
   207	                    generator=_g), p=2, dim=-1)
   208	                with torch.no_grad():
   209	                    self.clip_part_head.clip_text_features.copy_(_rand.float())
   210	                print('[LGPA] RANDOM-TEXT ablation: CLIP text prototypes -> FIXED random vectors (seed 42)')
   211	
   212	        # CLIP-ReID-style learnable ID prompts (the WORKING CLIP mechanism, vs dead fixed part text)
   213	        self.use_clip_id_prompt = getattr(cfg.MODEL, 'POSE_CLIP_ID_PROMPT', False)
   214	        if self.use_clip_id_prompt:
   215	            from .modules.clip_id_prompt import CLIPIDPromptLearner

exec
/bin/zsh -lc 'rg -n "def _to_target_aspect|def make_lr|def make_lattice_variants|PIXEL_MEAN|SIZE_TEST|KERNELS|_KERNELS" experiments/cargo_cvpb/cvpb_lattice_killswitch.py experiments/cargo_cvpb/cvpb_lm_reid_train.py' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/cargo_cvpb/cvpb_lm_reid_train.py:69:SIZE_TEST = (384, 128)                       # (H, W) model input / HR canvas
experiments/cargo_cvpb/cvpb_lm_reid_train.py:70:PIXEL_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
experiments/cargo_cvpb/cvpb_lm_reid_train.py:72:_KERNELS = {'bicubic': Image.BICUBIC, 'bilinear': Image.BILINEAR, 'lanczos': Image.LANCZOS,
experiments/cargo_cvpb/cvpb_lm_reid_train.py:99:def _to_target_aspect(img):
experiments/cargo_cvpb/cvpb_lm_reid_train.py:100:    return img.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)
experiments/cargo_cvpb/cvpb_lm_reid_train.py:103:def make_lr(hr_img, h, kernel='bicubic'):
experiments/cargo_cvpb/cvpb_lm_reid_train.py:104:    w = max(1, int(round(h * SIZE_TEST[1] / SIZE_TEST[0])))
experiments/cargo_cvpb/cvpb_lm_reid_train.py:105:    small = hr_img.resize((w, h), _KERNELS[kernel])
experiments/cargo_cvpb/cvpb_lm_reid_train.py:106:    return small.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)
experiments/cargo_cvpb/cvpb_lm_reid_train.py:109:def make_lattice_variants(hr_img, h, K, rng):
experiments/cargo_cvpb/cvpb_lm_reid_train.py:151:    arr = (arr - PIXEL_MEAN) / PIXEL_STD
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:75:SIZE_TEST = (384, 128)       # (H, W) the model input
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:76:PIXEL_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:80:_KERNELS = {
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:110:def _to_target_aspect(img):
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:113:    return img.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:116:def make_lr(hr_img, h, kernel='bicubic'):
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:120:    w = max(1, int(round(h * SIZE_TEST[1] / SIZE_TEST[0])))   # h*128/384 = h/3
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:121:    k = _KERNELS[kernel]
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:123:    return small.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:126:def make_lattice_variants(hr_img, h, K, rng):
experiments/cargo_cvpb/cvpb_lattice_killswitch.py:209:    arr = (arr - PIXEL_MEAN) / PIXEL_STD

exec
/bin/zsh -lc "rg -n \"def load_param|load_param\\(\" -S model" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
model/make_model.py:91:            self.base.load_param(model_path)
model/make_model.py:113:            self.load_param(model_path)
model/make_model.py:142:    def load_param(self, trained_path):
model/make_model.py:155:    #  def load_param(self, trained_path):
model/make_model.py:244:        #    self.load_param(model_path)
model/make_model.py:268:    def load_param(self, trained_path):
model/make_model.py:302:            self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
model/make_model.py:435:    def load_param(self, trained_path):
model/backbones/vit_pytorch.py:341:    def load_param(self, model_path,hw_ratio):
model/backbones/resnet_ibn_a.py:157:    def load_param(self, model_path):
model/backbones/resnet.py:127:    def load_param(self, model_path):

exec
/bin/zsh -lc 'rg -n "POSE_LGPA_FIXED_BANDS|POSE_TEST_FEAT|USE_LGPA|USE_SKELETON|USE_PSG|use_lgpa|use_skeleton_gcn|psg_stage" config configs model/pose_backbone_model.py' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
model/pose_backbone_model.py:40:        psg_stages = list(getattr(cfg.MODEL, 'POSE_PSG_STAGES', [-1]))
model/pose_backbone_model.py:43:        self.psg_stage_indices = set()
model/pose_backbone_model.py:44:        for s in psg_stages:
model/pose_backbone_model.py:46:            self.psg_stage_indices.add(idx)
model/pose_backbone_model.py:53:        for stage_idx in sorted(self.psg_stage_indices):
model/pose_backbone_model.py:67:        if last_stage_idx in self.psg_stage_indices:
model/pose_backbone_model.py:77:            for stage_idx in sorted(self.psg_stage_indices):
model/pose_backbone_model.py:174:            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
model/pose_backbone_model.py:177:        self.use_lgpa = getattr(cfg.MODEL, 'POSE_LGPA', False)
model/pose_backbone_model.py:178:        if self.use_lgpa and getattr(cfg.MODEL, 'POSE_PPA', False):
model/pose_backbone_model.py:180:        if self.use_lgpa:
model/pose_backbone_model.py:191:            self._lgpa_fixed_bands = getattr(cfg.MODEL, 'POSE_LGPA_FIXED_BANDS', False)
model/pose_backbone_model.py:193:            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
model/pose_backbone_model.py:295:            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
model/pose_backbone_model.py:313:        self.use_skeleton_gcn = getattr(cfg.MODEL, 'POSE_SKELETON_GCN', False)
model/pose_backbone_model.py:314:        if self.use_skeleton_gcn:
model/pose_backbone_model.py:339:            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'concat_scaled')
model/pose_backbone_model.py:357:        if self.use_vcnorm and not self.use_skeleton_gcn:
model/pose_backbone_model.py:393:            self.pose_test_feat = getattr(cfg.MODEL, 'POSE_TEST_FEAT', 'equal_concat')
model/pose_backbone_model.py:402:        if self.use_splade and self.use_skeleton_gcn:
model/pose_backbone_model.py:419:            if not self.use_skeleton_gcn:
model/pose_backbone_model.py:427:            if not self.use_skeleton_gcn:
model/pose_backbone_model.py:517:            if i in self.psg_stage_indices:
model/pose_backbone_model.py:829:                    and not self.use_skeleton_gcn:
model/pose_backbone_model.py:905:            elif getattr(self, 'use_lgpa', False) and (scene_heatmaps is not None or getattr(self, '_lgpa_fixed_bands', False)):
model/pose_backbone_model.py:916:                if self.use_skeleton_gcn and pose_dict is not None:
model/pose_backbone_model.py:940:                if self.use_skeleton_gcn and pose_dict is not None:
model/pose_backbone_model.py:956:            elif self.use_skeleton_gcn and pose_dict is not None:
model/pose_backbone_model.py:1107:            elif getattr(self, 'use_lgpa', False) and (scene_heatmaps is not None or getattr(self, '_lgpa_fixed_bands', False)) and \
model/pose_backbone_model.py:1114:                if self.use_skeleton_gcn and pose_dict is not None:
model/pose_backbone_model.py:1129:                if self.use_skeleton_gcn and pose_dict is not None:
model/pose_backbone_model.py:1138:                    getattr(self, 'pose_test_feat', 'global') != 'global' and not self.use_skeleton_gcn:
model/pose_backbone_model.py:1176:            elif self.use_skeleton_gcn and pose_dict is not None and \
config/defaults.py:93:_C.MODEL.POSE_TEST_FEAT = 'concat_scaled'  # 'concat_scaled', 'part_only', 'equal_concat', 'cvk_only', 'cvk_hybrid', 'maxsim'
config/defaults.py:224:_C.MODEL.POSE_LGPA_FIXED_BANDS = False    # Fixed-semantics: replace per-image pose with a FIXED canonical pedestrian pose (fixed CLIP text + fixed anatomical prior, no per-image pose)
configs/market/pose_psg_lgpa_gcn_base.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/market/pose_vcnorm_base_control.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/market/pose_pds_sg_gcn.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/market/prcv_best_tiny.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/market/prcv_best_base.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/market/pose_vcnorm_base.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/market/pose_pds_sg_gcn_small.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/market/prcv_best_small.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/market/pose_pds_sg_gcn_base.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_posetrack/prcv_best_small.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_posetrack/prcv_best_base.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_posetrack/prcv_best_tiny.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_roa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_sckd_min4.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_pacd.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_ms.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_sasa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_roa_vcga.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/prcv_best_tiny.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_ptd.yml:28:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_lgpa_detach.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp337_swin_lgpa_nopose.yml:32:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_lgpa_gcn_base.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_ttsfr.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_pgam.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_pgtm.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_sckd_up07_freeze30.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr_pertoken_pgmpoa.yml:29:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_kdl.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pamc.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_xcad.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_plboa_roa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_evidential.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr_pertoken_plboa_pape.yml:28:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_pds.yml:24:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr_plboa_roa.yml:28:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pke.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr_pertoken_plboa.yml:27:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr_pertoken_plboa_pape_ms_supcon_small.yml:31:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_pfm.yml:24:  POSE_TEST_FEAT: 'part_only'  # part-only proven best
configs/occluded_duke/pose_psg_gcn_lpcs.yml:23:  POSE_TEST_FEAT: 'cvk_residual'
configs/occluded_duke/exp340_swin_lgpa_fixedbands.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp340_swin_lgpa_fixedbands.yml:26:  POSE_LGPA_FIXED_BANDS: True                # ★ 固定 canonical pose 替代 per-image pose
configs/occluded_duke/exp340_swin_lgpa_fixedbands.yml:32:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pair_top_resid_scrd.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/exp351_undetach_deocc.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp351_undetach_deocc.yml:37:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_roa_nopaa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr_pertoken_plboa_pltd.yml:28:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp340b_fixedbands_undetach.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp340b_fixedbands_undetach.yml:26:  POSE_LGPA_FIXED_BANDS: True                # ★ 固定 canonical pose 替代 per-image pose
configs/occluded_duke/exp340b_fixedbands_undetach.yml:32:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_ltcs.yml:23:  POSE_TEST_FEAT: 'cvk_adaptive'
configs/occluded_duke/pose_psg_stdpr_pertoken_plboa_pape_ms.yml:29:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp037_lka.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_gl025.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_paa_pqtd.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/prcv_best_small.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_pds_stopgrad.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pgam_t05.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_sgre.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp045_seed42_cvk_verify.yml:23:  POSE_TEST_FEAT: 'cvk_hybrid'
configs/occluded_duke/pose_pds_delayed_stopgrad.yml:27:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_lpcs_rank_decay.yml:23:  POSE_TEST_FEAT: 'cvk_residual'
configs/occluded_duke/pose_psg_gcn_pnis.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pair_top_residual_kl_scrd.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/pose_psg_gcn_paa_pisd.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_roa_base.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_maxsim_add.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pair_top_exact_scrd_freeze30.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/exp035b_kpw_score_visibility.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp341_clip_id_prompt.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp341_clip_id_prompt.yml:34:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_gradient_occ.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_sasa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_rrc.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp041_cvk_weight_sweep.yml:23:  POSE_TEST_FEAT: 'cvk_hybrid'
configs/occluded_duke/exp344_pose_cond_prompt.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp344_pose_cond_prompt.yml:35:  POSE_TEST_FEAT: 'global'
configs/occluded_duke/pose_psg_gcn_paa_sc.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr.yml:24:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_noscale.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr_pertoken_plboa_pape3.yml:29:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp353_undetach_noclip.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp353_undetach_noclip.yml:35:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_plboa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_apg.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_st.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_pamn.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_roa_sgmt.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_pds_sg_gcn.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_parallel.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp341base_noprompt.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp341base_noprompt.yml:34:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp035c_kpw_visibility.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_lpcs_delta_top.yml:23:  POSE_TEST_FEAT: 'cvk_residual'
configs/occluded_duke/pose_psg_gcn_paa_film.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_sgmt50.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp355_pgpd.yml:37:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_lpcs_comp_ctx.yml:23:  POSE_TEST_FEAT: 'cvk_residual'
configs/occluded_duke/pose_psg_gcn_pgam_s23.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp349_small_full_clip.yml:33:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_cipgfr.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp358_pose_channel_shuffle.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp358_pose_channel_shuffle.yml:36:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_maxsim_add_w1.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_evidential_scaled.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/exp342_clip_id_pose.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp342_clip_id_pose.yml:35:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_sgmt.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_mrkf.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp348_occ_repel.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp348_occ_repel.yml:38:  POSE_TEST_FEAT: 'global'
configs/occluded_duke/pose_stage2_parts.yml:23:  POSE_TEST_FEAT: 'part_only'
configs/occluded_duke/pose_psg_stdpr_plboa_200ep.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_maxsim_add.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_splade.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_gl025.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gilt.yml:25:  POSE_TEST_FEAT: 'global'
configs/occluded_duke/prcv_best_base.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_lpcs_hard_rank.yml:23:  POSE_TEST_FEAT: 'cvk_residual'
configs/occluded_duke/pose_psg_part.yml:23:  POSE_TEST_FEAT: 'part_only'
configs/occluded_duke/exp044_exp030a_seed42_rebuild.yml:23:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/pose_psg_gcn_gkd.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_rrpaa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_pkp.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp042_pair_case_analysis.yml:23:  POSE_TEST_FEAT: 'cvk_hybrid'
configs/occluded_duke/pose_psg_gcn_sckd_diag_up07.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_scrd_freeze30.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/pose_psg_keypoint_pool.yml:26:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/pose_psg_gcn_sckd_up07_freeze20.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp355r_pgpd_random.yml:37:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_part_dominant.yml:24:  POSE_TEST_FEAT: 'part_only'  # only use part features at test time
configs/occluded_duke/pose_psg_gcn_sgw_scrd.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/pose_psg_gcn_paa_b128.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pcl.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp340c_fixedbands_randomtext.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp340c_fixedbands_randomtext.yml:26:  POSE_LGPA_FIXED_BANDS: True
configs/occluded_duke/exp340c_fixedbands_randomtext.yml:33:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_tdpc.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_ps.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp357_pose_shuffle.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp357_pose_shuffle.yml:36:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr_plboa.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_sckd_up07.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_vcga.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_lpcs_fix.yml:23:  POSE_TEST_FEAT: 'cvk_residual'
configs/occluded_duke/pose_pds_sg_gcn_small.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_body_random.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_plboa_roa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp336_swin_lgpa_nopsg.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp336_swin_lgpa_nopsg.yml:31:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_plboa_200ep.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_dpf.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_lpcs_conf.yml:23:  POSE_TEST_FEAT: 'cvk_residual'
configs/occluded_duke/pose_psg_gcn_paml.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp349b_small_undetach_clip.yml:33:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_pds_stopgrad_nopsg.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pnis_plboa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp036_kp_triplet.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_lku.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp040_exp030a_cvk_verify.yml:23:  POSE_TEST_FEAT: 'cvk_hybrid'
configs/occluded_duke/pose_psg_gcn_pcvt_random.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_pgfi.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_lsrm.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_lgpa_gcn512_2stage_small.yml:30:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pair_queue_scrd.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/pose_psg_gcn_skc.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pair_top_exact_scrd.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/pose_psg_gcn_pair_delta_scrd.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/pose_psg_lgpa.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_csrd.yml:23:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/pose_psg_stdpr_pertoken17_plboa.yml:27:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp342c_global1x.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp342c_global1x.yml:35:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_pds_stopgrad_partlr.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp345_pose_part_clip.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp345_pose_part_clip.yml:35:  POSE_TEST_FEAT: 'global'
configs/occluded_duke/pose_psg_gcn_lpcs_query_ctx.yml:23:  POSE_TEST_FEAT: 'cvk_residual'
configs/occluded_duke/pose_psg_gcn_paa_pcqa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pcvt.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp046_exp030a_seed2024_rebuild.yml:23:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/exp035d_kpw_binary_visibility.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_sckd.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr_pertoken_plboa_pape_ms_supcon.yml:31:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_scfr.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pair_delta_scrd_a4.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/pose_psg_vcsr.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr_17.yml:24:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_plboa.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paroa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_scrd.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/exp356_pcmsc.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp356_pcmsc.yml:37:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pgam.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_scfa.yml:23:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/exp342b_undetach.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp342b_undetach.yml:35:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_pds_sg_gcn_base.yml:26:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_roa_pgam.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pvat.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_paa_mm.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_lpcs_delta_top_fix.yml:23:  POSE_TEST_FEAT: 'cvk_residual'
configs/occluded_duke/pose_psg_gcn_maxsim.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp347_noparam_deocc.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp347_noparam_deocc.yml:36:  POSE_TEST_FEAT: 'global'
configs/occluded_duke/pose_psg_gcn_kprpe.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp350_undetach_clean.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp350_undetach_clean.yml:35:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp352_undetach_morepartw.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp352_undetach_morepartw.yml:36:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp047_csgt_triplet.yml:24:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/exp035a_kpw_score.yml:25:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_sasa_alpha1.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp039_cvk_eval.yml:23:  POSE_TEST_FEAT: 'cvk_hybrid'
configs/occluded_duke/pose_psg_stdpr_plboa_paa.yml:28:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp343_pose_guided_clip.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp343_pose_guided_clip.yml:35:  POSE_TEST_FEAT: 'global'
configs/occluded_duke/pose_psg_gcn_roa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/exp048_sgmkc.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/pose_psg_gcn_pair_top_scrd.yml:25:  POSE_TEST_FEAT: 'concat_scaled'
configs/occluded_duke/exp356r_pcmsc_random.yml:4:# 判据：同一 ckpt，test.py POSE_TEST_FEAT=equal_concat(LGPA) vs =global(baseline，detached→==baseline)。
configs/occluded_duke/exp356r_pcmsc_random.yml:37:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_stdpr_pertoken_plboa_dptl.yml:28:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_pke_roa.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_scrc.yml:23:  POSE_TEST_FEAT: 'equal_concat'
configs/occluded_duke/pose_psg_gcn_maxsim_hard.yml:23:  POSE_TEST_FEAT: 'equal_concat'

codex
初步看到两类需要重点确认的点：一是脚本注释写“byte-identical”但训练端 `make_lattice_variants` 可能不是从 kill-switch 原样复制；二是 `L_adv` 的 GRL 系数可能被乘了两次。我会把这些和 load/save、配置隔离逐项钉实。
exec
/bin/zsh -lc "nl -ba configs/market/pose_psg_lgpa_gcn_base.yml | sed -n '1,80p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
     1	MODEL:
     2	  PRETRAIN_HW_RATIO: 2
     3	  PRETRAIN_CHOICE: 'self'
     4	  PRETRAIN_PATH: 'pretrained/swin_base.pth'
     5	  METRIC_LOSS_TYPE: 'triplet'
     6	  IF_LABELSMOOTH: 'off'
     7	  IF_WITH_CENTER: 'no'
     8	  NAME: 'transformer'
     9	  NO_MARGIN: True
    10	  DEVICE_ID: ('0')
    11	  TRANSFORMER_TYPE: 'swin_base_patch4_window7_224'
    12	  STRIDE_SIZE: [16, 16]
    13	  SEMANTIC_WEIGHT: 0.2
    14	  WITH_CP: True
    15	  POSE_ENABLED: True
    16	  POSE_DATA_DIR: 'data/market1501'
    17	  POSE_HEATMAP_SIZE: [96, 32]
    18	  POSE_BACKBONE_PSG: True
    19	  POSE_PFM_HIDDEN: 64
    20	  POSE_LGPA: True
    21	  POSE_LGPA_CLIP_DIM: 512
    22	  POSE_LGPA_NUM_HEADS: 8
    23	  POSE_LGPA_POSE_TEMP: 1.0
    24	  POSE_LGPA_ASSIGN_WEIGHT: 0.5
    25	  POSE_LGPA_DETACH: True
    26	  POSE_SKELETON_GCN: True
    27	  POSE_GCN_LAYERS: 2
    28	  POSE_GCN_HIDDEN: 512
    29	  POSE_PSG_STAGES: [-2, -1]
    30	  POSE_TEST_FEAT: 'equal_concat'
    31	  GLOBAL_LOSS_SCALE: 0.5
    32	  POSE_LOWER_BODY_OCC: False
    33	  POSE_PARALLEL_AUG: True
    34	  POSE_OA_SD: True
    35	  POSE_OA_SD_WEIGHT: 1.0
    36	  POSE_OA_SD_EMA_DECAY: 0.999
    37	
    38	INPUT:
    39	  SIZE_TRAIN: [384, 128]
    40	  SIZE_TEST: [384, 128]
    41	  PROB: 0.5
    42	  RE_PROB: 0.5
    43	  PADDING: 10
    44	  PIXEL_MEAN: [0.5, 0.5, 0.5]
    45	  PIXEL_STD: [0.5, 0.5, 0.5]
    46	
    47	DATASETS:
    48	  NAMES: ('market1501')
    49	  ROOT_DIR: ('data')
    50	
    51	DATALOADER:
    52	  SAMPLER: 'softmax_triplet'
    53	  NUM_INSTANCE: 4
    54	  NUM_WORKERS: 8
    55	
    56	SOLVER:
    57	  OPTIMIZER_NAME: 'SGD'
    58	  MAX_EPOCHS: 120
    59	  BASE_LR: 0.0004
    60	  WARMUP_EPOCHS: 20
    61	  IMS_PER_BATCH: 64
    62	  WARMUP_METHOD: 'cosine'
    63	  LARGE_FC_LR: False
    64	  CHECKPOINT_PERIOD: 20
    65	  LOG_PERIOD: 20
    66	  EVAL_PERIOD: 10
    67	  WEIGHT_DECAY: 1e-4
    68	  WEIGHT_DECAY_BIAS: 1e-4
    69	  BIAS_LR_FACTOR: 2
    70	
    71	TEST:
    72	  EVAL: True
    73	  IMS_PER_BATCH: 128
    74	  RE_RANKING: False
    75	  WEIGHT: ''
    76	  NECK_FEAT: 'before'
    77	  FEAT_NORM: 'yes'
    78	
    79	OUTPUT_DIR: './log/market1501/exp260_base_gcn512_2stage'

exec
/bin/zsh -lc "nl -ba config/defaults.py | sed -n '180,235p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   180	
   181	# MaxSim triplet: set-to-set metric learning
   182	_C.MODEL.POSE_MAXSIM_TRIPLET = False
   183	_C.MODEL.POSE_MAXSIM_TRIPLET_TEMP = 0.05
   184	_C.MODEL.POSE_MAXSIM_TRIPLET_ADDITIVE = False  # If True, add to pooled triplet instead of replacing
   185	_C.MODEL.POSE_MAXSIM_TRIPLET_WEIGHT = 0.25     # Weight when additive
   186	
   187	# Parallel augmentation (3-view training)
   188	_C.MODEL.POSE_PARALLEL_AUG = False
   189	_C.MODEL.POSE_OA_SD = False               # Occlusion-Asymmetric Self-Distillation
   190	_C.MODEL.POSE_OA_SD_WEIGHT = 1.0          # Distillation loss weight
   191	_C.MODEL.POSE_OA_SD_EMA_DECAY = 0.999    # EMA teacher decay rate
   192	_C.MODEL.POSE_OA_SD_GLOBAL_ONLY = False   # Only distill global feature (not per-token)
   193	
   194	# OA-RD: Occlusion-Asymmetric Relational Distillation
   195	_C.MODEL.POSE_OA_RD = False                # Enable relational distillation (distill pairwise similarity structure)
   196	_C.MODEL.POSE_OA_RD_TEMP = 0.1            # Temperature for softmax on similarity matrix
   197	_C.MODEL.POSE_OA_RD_WEIGHT = 1.0          # Weight of relational distillation loss
   198	
   199	# KAMP: Keypoint-Anchored Multi-Scale Part features
   200	_C.MODEL.POSE_MULTI_SCALE_KP = False      # Enable multi-scale keypoint sampling
   201	_C.MODEL.POSE_MULTI_SCALE_STAGES = [2, 3] # Which stages to use (0-indexed, 3=last)
   202	
   203	# PADPQ: Pose-Anchored Deformable Part Queries — learned offsets around keypoints
   204	_C.MODEL.POSE_DEFORMABLE_SAMPLE = False    # Enable deformable keypoint sampling
   205	_C.MODEL.POSE_DEFORMABLE_K = 4            # Number of offset sampling points per keypoint
   206	
   207	# Per-body-part independent training (KPR-inspired)
   208	_C.MODEL.POSE_GCN_PER_PART = False        # Split 17 keypoints into 6 body parts, each with own classifier
   209	
   210	# PPA: Pose-Prompted Part-Assignment Head — end-to-end learnable part assignment
   211	_C.MODEL.POSE_PPA = False                 # Enable PPA (replaces GCN Part branch)
   212	_C.MODEL.POSE_PPA_NUM_PARTS = 5           # Number of body parts (5)
   213	_C.MODEL.POSE_PPA_ASSIGN_WEIGHT = 0.5     # Assignment loss weight
   214	_C.MODEL.POSE_PPA_GILT = False            # GiLt mode: Part triplet only, no Part CE
   215	
   216	# LGPA: Language-Grounded Part Assignment — CLIP text prototypes + cross-attention + pose masks
   217	_C.MODEL.POSE_LGPA = False                # Enable LGPA (replaces PPA / GCN Part branch)
   218	_C.MODEL.POSE_LGPA_CLIP_DIM = 512        # CLIP text feature dimension (ViT-B-32 = 512)
   219	_C.MODEL.POSE_LGPA_NUM_HEADS = 8         # Cross-attention heads
   220	_C.MODEL.POSE_LGPA_POSE_TEMP = 1.0       # Pose mask temperature
   221	_C.MODEL.POSE_LGPA_ASSIGN_WEIGHT = 0.5   # Assignment supervision loss weight
   222	_C.MODEL.POSE_LGPA_DETACH = False         # Detach features before LGPA (no gradient to backbone)
   223	_C.MODEL.POSE_LGPA_NO_POSE = False        # Ablation: pass None heatmaps to LGPA (no pose-bias/assign/visibility) = pure CLIP-text parts
   224	_C.MODEL.POSE_LGPA_FIXED_BANDS = False    # Fixed-semantics: replace per-image pose with a FIXED canonical pedestrian pose (fixed CLIP text + fixed anatomical prior, no per-image pose)
   225	_C.MODEL.POSE_LGPA_RANDOM_TEXT = False     # Attribution ablation: replace CLIP text prototypes with FIXED random vectors (tests if CLIP semantics contribute vs pure structure)    # Fixed-semantics: replace per-image pose with a FIXED canonical pedestrian pose (fixed CLIP text + fixed anatomical prior, no per-image pose)
   226	_C.MODEL.POSE_CLIP_ID_PROMPT = False      # CLIP-ReID-style learnable ID text prompts (CoOp) + i2t/t2i contrastive — the WORKING CLIP mechanism
   227	_C.MODEL.POSE_CLIP_ID_ARCH = 'ViT-L-14'
   228	_C.MODEL.POSE_CLIP_ID_PRETRAINED = 'openai'  # path to local open_clip safetensors on the training machine
   229	_C.MODEL.POSE_CLIP_ID_TEMP = 0.07
   230	_C.MODEL.POSE_CLIP_ID_WEIGHT = 1.0
   231	_C.MODEL.POSE_CLIP_ID_POSE_GUIDED = False  # Option A: i2t/t2i aligns a pose-bias pooled feature (not raw global)
   232	_C.MODEL.POSE_CLIP_ID_POSE_TEMP = 1.0
   233	_C.MODEL.POSE_CLIP_ID_POSE_PROMPT = False  # Option B: per-image pose conditions the prompt context
   234	_C.MODEL.POSE_CLIP_ID_PART_GUIDED = False  # Option C: K pose-localized part features aligned to ID prototype
   235	_C.MODEL.POSE_CLIP_ID_NOPARAM_POOL = False  # exp347: param-free de-occluded pooling for alignment

exec
/bin/zsh -lc "nl -ba model/make_model.py | sed -n '120,180p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   120	        if self.reduce_feat_dim:
   121	            global_feat = self.fcneck(global_feat)
   122	
   123	        if self.neck == 'no':
   124	            feat = global_feat
   125	        elif self.neck == 'bnneck':
   126	            feat = self.bottleneck(global_feat)
   127	        if self.dropout_rate > 0:
   128	            feat = self.dropout(feat)
   129	
   130	        if self.training:
   131	            if self.cos_layer:
   132	                cls_score = self.arcface(feat, label)
   133	            else:
   134	                cls_score = self.classifier(feat)
   135	            return cls_score, global_feat
   136	        else:
   137	            if self.neck_feat == 'after':
   138	                return feat
   139	            else:
   140	                return global_feat
   141	
   142	    def load_param(self, trained_path):
   143	        param_dict = torch.load(trained_path, weights_only=False)
   144	        if 'state_dict' in param_dict:
   145	            param_dict = param_dict['state_dict']
   146	        for i in param_dict:
   147	            if 'classifier' in i:
   148	                continue
   149	            elif 'module' in i:
   150	                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
   151	            else:
   152	                self.state_dict()[i].copy_(param_dict[i])
   153	        print('Loading pretrained model from {}'.format(trained_path))
   154	
   155	    #  def load_param(self, trained_path):
   156	        #  param_dict = torch.load(trained_path, map_location='cpu', weights_only=False)
   157	        #  for i in param_dict:
   158	            #  try:
   159	                #  self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
   160	            #  except:
   161	                #  continue
   162	        #  print('Loading pretrained model from {}'.format(trained_path))
   163	
   164	
   165	class build_transformer(nn.Module):
   166	    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
   167	        super(build_transformer, self).__init__()
   168	        last_stride = cfg.MODEL.LAST_STRIDE
   169	        model_path = cfg.MODEL.PRETRAIN_PATH
   170	        model_name = cfg.MODEL.NAME
   171	        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
   172	        self.cos_layer = cfg.MODEL.COS_LAYER
   173	        self.neck = cfg.MODEL.NECK
   174	        self.neck_feat = cfg.TEST.NECK_FEAT
   175	        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
   176	        self.feat_dim = cfg.MODEL.FEAT_DIM
   177	        self.dropout_rate = cfg.MODEL.DROPOUT_RATE
   178	
   179	        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
   180	

exec
/bin/zsh -lc "nl -ba model/make_model.py | sed -n '400,460p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   400	        local_feat_3 = b3_local_feat[:, 0]
   401	
   402	        # lf_4
   403	        b4_local_feat = x[:, patch_length*3:patch_length*4]
   404	        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
   405	        local_feat_4 = b4_local_feat[:, 0]
   406	
   407	        feat = self.bottleneck(global_feat)
   408	
   409	        local_feat_1_bn = self.bottleneck_1(local_feat_1)
   410	        local_feat_2_bn = self.bottleneck_2(local_feat_2)
   411	        local_feat_3_bn = self.bottleneck_3(local_feat_3)
   412	        local_feat_4_bn = self.bottleneck_4(local_feat_4)
   413	
   414	        if self.training:
   415	            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
   416	                cls_score = self.classifier(feat, label)
   417	            else:
   418	                cls_score = self.classifier(feat)
   419	                cls_score_1 = self.classifier_1(local_feat_1_bn)
   420	                cls_score_2 = self.classifier_2(local_feat_2_bn)
   421	                cls_score_3 = self.classifier_3(local_feat_3_bn)
   422	                cls_score_4 = self.classifier_4(local_feat_4_bn)
   423	            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
   424	                        cls_score_4
   425	                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
   426	                            local_feat_4]  # global feature for triplet loss
   427	        else:
   428	            if self.neck_feat == 'after':
   429	                return torch.cat(
   430	                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
   431	            else:
   432	                return torch.cat(
   433	                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)
   434	
   435	    def load_param(self, trained_path):
   436	        param_dict = torch.load(trained_path, weights_only=False)
   437	        for i in param_dict:
   438	            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
   439	        print('Loading pretrained model from {}'.format(trained_path))
   440	
   441	
   442	
   443	__factory_T_type = {
   444	    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
   445	    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
   446	    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
   447	    'deit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
   448	    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
   449	    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
   450	    'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
   451	}
   452	
   453	def make_model(cfg, num_class, camera_num, view_num, semantic_weight):
   454	    if cfg.MODEL.NAME == 'transformer':
   455	        if cfg.MODEL.JPM:
   456	            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
   457	            print('===========building transformer with JPM module ===========')
   458	        elif cfg.MODEL.POSE_ENABLED:
   459	            if getattr(cfg.MODEL, 'POSE_DUAL_STREAM', False):
   460	                from .pose_dual_stream_model import PoseDualStreamModel

exec
/bin/zsh -lc "nl -ba experiments/cargo_cvpb/cvpb_lattice_killswitch.py | sed -n '70,160p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
    70	ap.add_argument('--reuse_gallery', action='store_true')
    71	cli = ap.parse_args()
    72	np.random.seed(cli.seed)
    73	RNG = np.random.RandomState(cli.seed)
    74	
    75	SIZE_TEST = (384, 128)       # (H, W) the model input
    76	PIXEL_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    77	PIXEL_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    78	
    79	# PIL resample kernels for the "different antialias kernel" lattice axis
    80	_KERNELS = {
    81	    'bicubic': Image.BICUBIC,
    82	    'bilinear': Image.BILINEAR,
    83	    'lanczos': Image.LANCZOS,
    84	    'box': Image.BOX,
    85	    'hamming': Image.HAMMING,
    86	    'nearest': Image.NEAREST,
    87	}
    88	
    89	
    90	# =========================================================================== #
    91	# dataset list (parse Market dirs directly; no dataloader needed)
    92	# =========================================================================== #
    93	import re, glob
    94	_PAT = re.compile(r'([-\d]+)_c(\d)')
    95	
    96	
    97	def list_split(dir_path):
    98	    items = []
    99	    for p in sorted(glob.glob(os.path.join(dir_path, '*.jpg'))):
   100	        pid, cam = map(int, _PAT.search(p).groups())
   101	        if pid == -1:
   102	            continue
   103	        items.append((p, pid, cam - 1))
   104	    return items
   105	
   106	
   107	# =========================================================================== #
   108	# LR + lattice variant generation  (all in PIL space, from the ORIGINAL image)
   109	# =========================================================================== #
   110	def _to_target_aspect(img):
   111	    """Resize the original crop to the model's 384x128 (3:1) HR canvas with BICUBIC.
   112	    This is the 'HR' reference everything is degraded from (gallery also uses this)."""
   113	    return img.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)
   114	
   115	
   116	def make_lr(hr_img, h, kernel='bicubic'):
   117	    """Deterministic synthetic LR: HR(384x128) --down--> (h, w) --up--> 384x128.
   118	    w preserves the 3:1 canvas aspect: w = round(h/3).  Returns a 384x128 PIL image
   119	    (degrade-then-restore-size, the standard CR-ReID synthetic LR convention)."""
   120	    w = max(1, int(round(h * SIZE_TEST[1] / SIZE_TEST[0])))   # h*128/384 = h/3
   121	    k = _KERNELS[kernel]
   122	    small = hr_img.resize((w, h), k)
   123	    return small.resize((SIZE_TEST[1], SIZE_TEST[0]), Image.BICUBIC)
   124	
   125	
   126	def make_lattice_variants(hr_img, h, K, rng):
   127	    """K plausible PHASE/CROP/KERNEL variants of the SAME hr image at height h.
   128	
   129	    Each variant perturbs the SAMPLING LATTICE relative to the scene by a SUB-LR-pixel
   130	    amount, then forms the LR image.  The depicted person is (almost) the same extent;
   131	    only WHICH hr pixels land on each LR sample point changes.  Axes:
   132	      - sub-pixel phase shift  (fractional HR translate before downsample)
   133	      - +/-1 LR-pixel bbox crop shift / expand (integer LR-pixel = h/.. HR pixels)
   134	      - antialias kernel choice
   135	
   136	    variant 0 is ALWAYS the canonical deterministic bicubic LR (no perturbation) so the
   137	    single-LR baseline == variants[0].
   138	    Returns list of K PIL images (each 384x128)."""
   139	    W_hr, H_hr = hr_img.size                      # 128, 384
   140	    # how many HR pixels correspond to 1 LR pixel at this height
   141	    hr_per_lr_y = H_hr / float(h)                  # 384/h
   142	    hr_per_lr_x = W_hr / float(max(1, round(h / 3.0)))  # 128/(h/3) ~ 3
   143	    variants = [make_lr(hr_img, h, 'bicubic')]     # 0: canonical
   144	    kernels_cycle = ['bicubic', 'bilinear', 'lanczos', 'box', 'hamming']
   145	    for j in range(1, K):
   146	        # --- pick a lattice perturbation type round-robin so the K cover all axes ---
   147	        mode = j % 3
   148	        kern = kernels_cycle[j % len(kernels_cycle)]
   149	        if mode == 0:
   150	            # sub-pixel phase: fractional shift of up to +/-0.5 LR pixel (in HR px)
   151	            dx = rng.uniform(-0.5, 0.5) * hr_per_lr_x
   152	            dy = rng.uniform(-0.5, 0.5) * hr_per_lr_y
   153	            shifted = hr_img.transform(
   154	                (W_hr, H_hr), Image.AFFINE, (1, 0, dx, 0, 1, dy),
   155	                resample=Image.BICUBIC)
   156	            v = make_lr(shifted, h, kern)
   157	        elif mode == 1:
   158	            # +/-1 LR-pixel bbox crop shift: crop the HR by an integer # of LR pixels on
   159	            # each side then resize back to the HR canvas (== shifting the bbox window).
   160	            sx = int(round(rng.choice([-1, 0, 1]) * hr_per_lr_x))

exec
/bin/zsh -lc 'rg -n "--ckpt|load_param|torch.load|state_dict|MODEL.POSE_TEST_FEAT|pose_dict=None|make_lattice_variants|lat-MaxSim|MaxSim|mean" experiments/cargo_cvpb/cvpb_lattice_killswitch.py' in /Users/abdslm/Desktop/SOLIDER-REID
 exited 2 in 0ms:
rg: unrecognized flag --ckpt|load_param|torch.load|state_dict|MODEL.POSE_TEST_FEAT|pose_dict

codex
`configs/market/pose_psg_lgpa_gcn_base.yml` 本身是 PSG+LGPA+GCN 配置，但在训练脚本里强制传 `pose_dict=None`，因此 GCN 路径不会进；PSG 注入函数也只在有 heatmap 时改 token。LGPA fixed-bands 默认是 `False`，这个隔离成立。
exec
/bin/zsh -lc "nl -ba experiments/cargo_cvpb/cvpb_lattice_killswitch.py | sed -n '156,235p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   156	            v = make_lr(shifted, h, kern)
   157	        elif mode == 1:
   158	            # +/-1 LR-pixel bbox crop shift: crop the HR by an integer # of LR pixels on
   159	            # each side then resize back to the HR canvas (== shifting the bbox window).
   160	            sx = int(round(rng.choice([-1, 0, 1]) * hr_per_lr_x))
   161	            sy = int(round(rng.choice([-1, 0, 1]) * hr_per_lr_y))
   162	            left = max(0, sx); upper = max(0, sy)
   163	            right = W_hr + min(0, sx); lower = H_hr + min(0, sy)
   164	            if right - left < 4 or lower - upper < 4:
   165	                left, upper, right, lower = 0, 0, W_hr, H_hr
   166	            cropped = hr_img.crop((left, upper, right, lower)).resize(
   167	                (W_hr, H_hr), Image.BICUBIC)
   168	            v = make_lr(cropped, h, kern)
   169	        else:
   170	            # bbox expand / contract by 1 LR pixel (zoom in/out a touch) + kernel swap
   171	            ez = rng.choice([-1, 1]) * 0.5 * hr_per_lr_y   # expand/contract in HR px
   172	            box = (-ez, -ez * (W_hr / H_hr), W_hr + ez, H_hr + ez * (W_hr / H_hr)) \
   173	                if ez > 0 else (abs(ez), abs(ez) * (W_hr / H_hr),
   174	                                W_hr - abs(ez), H_hr - abs(ez) * (W_hr / H_hr))
   175	            # PIL crop on a fractional/negative box: emulate via paste on padded canvas
   176	            l, u, r, b = box
   177	            l, u, r, b = int(round(l)), int(round(u)), int(round(r)), int(round(b))
   178	            pad = max(0, -l, -u, r - W_hr, b - H_hr) + 1
   179	            canvas = Image.new('RGB', (W_hr + 2 * pad, H_hr + 2 * pad), (0, 0, 0))
   180	            canvas.paste(hr_img, (pad, pad))
   181	            cropped = canvas.crop((l + pad, u + pad, r + pad, b + pad)).resize(
   182	                (W_hr, H_hr), Image.BICUBIC)
   183	            v = make_lr(cropped, h, kern)
   184	        variants.append(v)
   185	    return variants
   186	
   187	
   188	def make_tta_variants(lr_img, K, rng, pad=10):
   189	    """ORDINARY TTA control: K views of ONE bicubic LR image via pad+RandomCrop (+ hflip).
   190	    NO lattice/phase semantics -- the standard cheap test-time augmentation.  variant 0 ==
   191	    the un-augmented LR so the single-LR baseline is shared with the lattice path."""
   192	    W, H = lr_img.size
   193	    out = [lr_img]                                  # 0: identity (== single LR)
   194	    for j in range(1, K):
   195	        canvas = Image.new('RGB', (W + 2 * pad, H + 2 * pad), (0, 0, 0))
   196	        canvas.paste(lr_img, (pad, pad))
   197	        cx = rng.randint(0, 2 * pad + 1)
   198	        cy = rng.randint(0, 2 * pad + 1)
   199	        crop = canvas.crop((cx, cy, cx + W, cy + H))
   200	        if rng.rand() < 0.5:
   201	            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
   202	        out.append(crop)
   203	    return out
   204	
   205	
   206	def pil_to_tensor_np(img):
   207	    """PIL 384x128 -> normalized CHW float32 ndarray (matches dataset _image_to_tensor)."""
   208	    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC
   209	    arr = (arr - PIXEL_MEAN) / PIXEL_STD
   210	    return arr.transpose(2, 0, 1)                            # CHW
   211	
   212	
   213	# =========================================================================== #
   214	# frozen model (pose DISABLED at test: pose_dict=None -> plain SOLIDER global feat)
   215	# =========================================================================== #
   216	class FrozenExtractor:
   217	    def __init__(self):
   218	        import torch
   219	        from config import cfg
   220	        from model import make_model
   221	        from datasets.market1501 import Market1501
   222	        self.torch = torch
   223	        cfg.merge_from_file(os.path.join(_repo, cli.config))
   224	        cfg.merge_from_list([
   225	            'TEST.WEIGHT', os.path.join(_repo, cli.ckpt),
   226	            'MODEL.POSE_TEST_FEAT', 'global',     # single clean global vector
   227	            'TEST.NECK_FEAT', 'after',            # trained BN-neck eval feature
   228	            'TEST.FEAT_NORM', 'yes',
   229	        ])  # leave PRETRAIN_* as in config; load_param(ckpt) overwrites all weights anyway
   230	        cfg.freeze()
   231	        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
   232	        # need num_classes/camera_num/view_num to build the head; read from Market meta.
   233	        ds = Market1501(root=os.path.join(_repo, cli.data_root), verbose=False)
   234	        model = make_model(cfg, num_class=ds.num_train_pids,
   235	                           camera_num=ds.num_train_cams, view_num=1,

exec
/bin/zsh -lc "rg -n -e \"--ckpt|load_param|torch\\.load|state_dict|POSE_TEST_FEAT|pose_dict=None|make_lattice_variants|lat-MaxSim|MaxSim|lat-|mean\" experiments/cargo_cvpb/cvpb_lattice_killswitch.py" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
17:        measure mean pairwise (1-cos) of their frozen features.  (does the lattice move
25:    (D) ensemble mAP: K-phase feature-mean / MaxSim vs a SINGLE deterministic bicubic LR.
28:    (C1) vs ORDINARY TTA: the SAME K, the SAME fusion (mean / MaxSim), but the K views are
47:      --ckpt   log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth \
61:ap.add_argument('--ckpt', default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
126:def make_lattice_variants(hr_img, h, K, rng):
214:# frozen model (pose DISABLED at test: pose_dict=None -> plain SOLIDER global feat)
226:            'MODEL.POSE_TEST_FEAT', 'global',     # single clean global vector
229:        ])  # leave PRETRAIN_* as in config; load_param(ckpt) overwrites all weights anyway
237:        model.load_param(os.path.join(_repo, cli.ckpt))
240:        print(f"[model] loaded {cli.ckpt}; POSE_TEST_FEAT=global; pose DISABLED at test "
241:              f"(pose_dict=None -> plain backbone global feat). num_cls={ds.num_train_pids}",
256:                out = self.model(t, cam_label=cam, view_label=view, pose_dict=None)
283:    all_cmc = np.asarray(all_cmc).mean(0)
284:    return dict(mAP=float(np.mean(all_AP)) * 100, r1=float(all_cmc[0]) * 100,
325:    rx -= rx.mean(); ry -= ry.mean()
406:            lat_pils = [make_lattice_variants(hr_q[i], h, cli.K, RNG) for i in range(cs, ce)]
420:        # mean over queries of mean pairwise (1-cos) among the K lattice variants.
421:        def mean_pairwise_dist(F):       # F: [Nq,K,D] L2-normed
425:            return pd.mean(1)                        # [Nq]
426:        phase_var = mean_pairwise_dist(f_lat)        # per-query lattice spread
427:        tta_var = mean_pairwise_dist(f_tta)
430:        print(f"  (A) same-image phase feature variance (mean pairwise 1-cos over K):")
431:        print(f"      lattice phase-var  mean={phase_var.mean():.4f}  median={np.median(phase_var):.4f}  "
433:        print(f"      ordinary TTA  var  mean={tta_var.mean():.4f}  (reference)")
434:        print(f"      single-LR -> HR feat drift  mean={lr_hr_drift.mean():.4f}  "
448:        top1_agree = (top1 == top1[:, [0]]).mean(1)   # [Nq]  (1.0 = perfectly stable)
453:        jac10 = np.array([np.mean([jacc(top10[i, 0], top10[i, j]) for j in range(1, cli.K)])
459:        print(f"      top1 stays==canonical : mean={top1_agree.mean():.3f}  "
461:        print(f"      top10 Jaccard(canon,j): mean={jac10.mean():.3f}  "
463:        print(f"      #distinct rank-1 IDs over K phases: mean={id_flip.mean():.2f}  "
465:              f"frac queries with >=2 = {100*(id_flip>=2).mean():.1f}%")
471:        # phase-lattice ENSEMBLE: feature-mean (renormed) and MaxSim
472:        f_lat_mean = f_lat.mean(1)
473:        f_lat_mean /= (np.linalg.norm(f_lat_mean, axis=1, keepdims=True) + 1e-12)
474:        d_lat_mean = 1.0 - f_lat_mean @ gf.T
475:        r_lat_mean = eval_map(d_lat_mean, q_pid, q_cam, g_pid, g_cam)
476:        # MaxSim: per (q,g) take the BEST sim over the K query variants
480:        f_tta_mean = f_tta.mean(1)
481:        f_tta_mean /= (np.linalg.norm(f_tta_mean, axis=1, keepdims=True) + 1e-12)
482:        r_tta_mean = eval_map(1.0 - f_tta_mean @ gf.T, q_pid, q_cam, g_pid, g_cam)
488:        print(f"      phase-lattice  mean-feat     : mAP={r_lat_mean['mAP']:.3f}  R1={r_lat_mean['r1']:.3f}  "
489:              f"(d{r_lat_mean['mAP']-r_single['mAP']:+.3f})")
490:        print(f"      phase-lattice  MaxSim        : mAP={r_lat_max['mAP']:.3f}  R1={r_lat_max['r1']:.3f}  "
493:        print(f"      ordinary-TTA   mean-feat     : mAP={r_tta_mean['mAP']:.3f}  R1={r_tta_mean['r1']:.3f}  "
494:              f"(d{r_tta_mean['mAP']-r_single['mAP']:+.3f})")
495:        print(f"      ordinary-TTA   MaxSim        : mAP={r_tta_max['mAP']:.3f}  R1={r_tta_max['r1']:.3f}  "
497:        best_lat = max(r_lat_mean['mAP'], r_lat_max['mAP'])
498:        best_tta = max(r_tta_mean['mAP'], r_tta_max['mAP'])
499:        print(f"      >> phase-lattice best gain = {best_lat-r_single['mAP']:+.3f}   "
501:              f"LATTICE-MINUS-TTA = {best_lat-best_tta:+.3f}  (must be clearly >0 to live)")
531:            phase_var=phase_var.mean(), tta_var=tta_var.mean(),
532:            top1_agree=top1_agree.mean(), jac10=jac10.mean(), id_flip=id_flip.mean(),
533:            frac_flip=100 * (id_flip >= 2).mean(),
535:            mAP_lat_mean=r_lat_mean['mAP'], mAP_lat_max=r_lat_max['mAP'],
536:            mAP_tta_mean=r_tta_mean['mAP'], mAP_tta_max=r_tta_max['mAP'],
548:           "single  lat-mean lat-max  tta-max | LATgain TTAgain LAT-TTA | "
555:              f"{s['mAP_single']:6.2f} {s['mAP_lat_mean']:7.2f} {s['mAP_lat_max']:7.2f} {s['mAP_tta_max']:7.2f}| "

exec
/bin/zsh -lc "nl -ba model/pose_backbone_model.py | sed -n '1,80p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
     1	"""Pose-guided ReID model with backbone-internal pose injection.
     2	
     3	Instead of post-hoc part pooling, injects pose information directly into
     4	the backbone's feature extraction process via Pose Spatial Gates (PSG)
     5	applied between Stage 3 blocks.
     6	
     7	This changes HOW features are formed, not just how they're pooled.
     8	"""
     9	import torch
    10	import torch.nn as nn
    11	import torch.nn.functional as F
    12	from .make_model import build_transformer
    13	from .modules.pose_spatial_gate import PoseSpatialGate
    14	from .modules.pose_utils import merge_person_heatmaps
    15	from .modules.skeleton_gcn import SkeletonGCNHead
    16	from .modules.pose_additive_adapter import PoseAdditiveAdapter
    17	from .modules.pair_adaptive_fusion import (
    18	    PairAdaptiveFusionHead,
    19	    PairResidualConfidenceScorer,
    20	    PairResidualScorer,
    21	)
    22	
    23	
    24	class PoseBackboneModel(build_transformer):
    25	    """ReID model with pose injection inside backbone.
    26	
    27	    Architecture:
    28	    - Swin backbone Stages 0-2: unchanged
    29	    - Stage 3: PSG applied after each SwinBlock
    30	    - Global feature (GAP -> BN -> classifier)
    31	    - Optional: Skeleton GCN part branch, PAA adapter, LTCS/LPCS heads
    32	
    33	    Test feature = global feature (pose-aware).
    34	    """
    35	
    36	    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
    37	        super().__init__(num_classes, camera_num, view_num, cfg, factory, semantic_weight)
    38	
    39	        # Determine which stages get pose injection
    40	        psg_stages = list(getattr(cfg.MODEL, 'POSE_PSG_STAGES', [-1]))
    41	        num_backbone_stages = len(self.base.stages)
    42	        # Resolve negative indices
    43	        self.psg_stage_indices = set()
    44	        for s in psg_stages:
    45	            idx = s if s >= 0 else num_backbone_stages + s
    46	            self.psg_stage_indices.add(idx)
    47	
    48	        hidden_dim = getattr(cfg.MODEL, 'POSE_PFM_HIDDEN', 64)
    49	        spatial_conv = getattr(cfg.MODEL, 'POSE_PSG_SPATIAL', False)
    50	
    51	        # PSG-only mode: create gate modules per stage
    52	        self.psg_modules_dict = nn.ModuleDict()
    53	        for stage_idx in sorted(self.psg_stage_indices):
    54	            stage = self.base.stages[stage_idx]
    55	            feat_ch = self.base.num_features[stage_idx]
    56	            for block_idx in range(len(stage.blocks)):
    57	                key = f's{stage_idx}_b{block_idx}'
    58	                self.psg_modules_dict[key] = PoseSpatialGate(
    59	                    pose_channels=17,
    60	                    feat_channels=feat_ch,
    61	                    hidden_dim=hidden_dim,
    62	                    spatial_conv=spatial_conv,
    63	                )
    64	
    65	        # Backward compatibility: also keep psg_modules list for Stage 3
    66	        last_stage_idx = num_backbone_stages - 1
    67	        if last_stage_idx in self.psg_stage_indices:
    68	            self.psg_modules = nn.ModuleList([
    69	                self.psg_modules_dict[f's{last_stage_idx}_b{j}']
    70	                for j in range(len(self.base.stages[last_stage_idx].blocks))
    71	            ])
    72	
    73	        # PAA (Pose Additive Adapter): additive injection alongside PSG
    74	        self.use_paa = getattr(cfg.MODEL, 'POSE_ADDITIVE_ADAPTER', False)
    75	        if self.use_paa:
    76	            self.paa_modules_dict = nn.ModuleDict()
    77	            for stage_idx in sorted(self.psg_stage_indices):
    78	                stage = self.base.stages[stage_idx]
    79	                feat_ch = self.base.num_features[stage_idx]
    80	                for block_idx in range(len(stage.blocks)):

exec
/bin/zsh -lc "nl -ba model/make_model.py | sed -n '453,485p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   453	def make_model(cfg, num_class, camera_num, view_num, semantic_weight):
   454	    if cfg.MODEL.NAME == 'transformer':
   455	        if cfg.MODEL.JPM:
   456	            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
   457	            print('===========building transformer with JPM module ===========')
   458	        elif cfg.MODEL.POSE_ENABLED:
   459	            if getattr(cfg.MODEL, 'POSE_DUAL_STREAM', False):
   460	                from .pose_dual_stream_model import PoseDualStreamModel
   461	                model = PoseDualStreamModel(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
   462	                print('===========building Pose Dual Stream (PDS) transformer===========')
   463	            elif getattr(cfg.MODEL, 'POSE_PSG_PART', False):
   464	                from .pose_psg_part_model import PosePSGPartModel
   465	                model = PosePSGPartModel(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
   466	                print('===========building PSG + Part Pooling transformer===========')
   467	            elif getattr(cfg.MODEL, 'POSE_BACKBONE_PSG', False):
   468	                from .pose_backbone_model import PoseBackboneModel
   469	                model = PoseBackboneModel(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
   470	                print('===========building pose-backbone (PSG) transformer===========')
   471	            else:
   472	                from .pose_model import PoseReIDModel
   473	                model = PoseReIDModel(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
   474	                print('===========building pose-guided transformer===========')
   475	        else:
   476	            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, semantic_weight)
   477	            print('===========building transformer===========')
   478	    else:
   479	        model = Backbone(num_class, cfg)
   480	        print('===========building ResNet===========')
   481	    return model

codex
保存兼容性正在看 `build_transformer.load_param`：这个模型继承的是 `build_transformer` 的加载逻辑，而训练脚本保存的是纯 `state_dict`。需要确认加载时是否会跳过 classifier 以及是否会因新增 `disc` 不在 checkpoint 中而受影响。
exec
/bin/zsh -lc "nl -ba experiments/cargo_cvpb/cvpb_lattice_killswitch.py | sed -n '216,268p'" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc "nl -ba model/make_model.py | sed -n '250,285p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   250	        feat = self.bottleneck(global_feat)
   251	        feat_cls = self.dropout(feat)
   252	
   253	        if self.training:
   254	            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
   255	                cls_score = self.classifier(feat_cls, label)
   256	            else:
   257	                cls_score = self.classifier(feat_cls)
   258	
   259	            return cls_score, global_feat, featmaps  # global feature for triplet loss
   260	        else:
   261	            if self.neck_feat == 'after':
   262	                # print("Test with feature after BN")
   263	                return feat, featmaps
   264	            else:
   265	                # print("Test with feature before BN")
   266	                return global_feat, featmaps
   267	
   268	    def load_param(self, trained_path):
   269	        param_dict = torch.load(trained_path, map_location='cpu', weights_only=False)
   270	        for i in param_dict:
   271	            try:
   272	                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
   273	            except:
   274	                continue
   275	        print('Loading pretrained model from {}'.format(trained_path))
   276	
   277	
   278	class build_transformer_local(nn.Module):
   279	    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
   280	        super(build_transformer_local, self).__init__()
   281	        model_path = cfg.MODEL.PRETRAIN_PATH
   282	        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
   283	        self.cos_layer = cfg.MODEL.COS_LAYER
   284	        self.neck = cfg.MODEL.NECK
   285	        self.neck_feat = cfg.TEST.NECK_FEAT

 succeeded in 0ms:
   216	class FrozenExtractor:
   217	    def __init__(self):
   218	        import torch
   219	        from config import cfg
   220	        from model import make_model
   221	        from datasets.market1501 import Market1501
   222	        self.torch = torch
   223	        cfg.merge_from_file(os.path.join(_repo, cli.config))
   224	        cfg.merge_from_list([
   225	            'TEST.WEIGHT', os.path.join(_repo, cli.ckpt),
   226	            'MODEL.POSE_TEST_FEAT', 'global',     # single clean global vector
   227	            'TEST.NECK_FEAT', 'after',            # trained BN-neck eval feature
   228	            'TEST.FEAT_NORM', 'yes',
   229	        ])  # leave PRETRAIN_* as in config; load_param(ckpt) overwrites all weights anyway
   230	        cfg.freeze()
   231	        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
   232	        # need num_classes/camera_num/view_num to build the head; read from Market meta.
   233	        ds = Market1501(root=os.path.join(_repo, cli.data_root), verbose=False)
   234	        model = make_model(cfg, num_class=ds.num_train_pids,
   235	                           camera_num=ds.num_train_cams, view_num=1,
   236	                           semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
   237	        model.load_param(os.path.join(_repo, cli.ckpt))
   238	        self.model = model.cuda().eval()
   239	        self.cam_dummy = None
   240	        print(f"[model] loaded {cli.ckpt}; POSE_TEST_FEAT=global; pose DISABLED at test "
   241	              f"(pose_dict=None -> plain backbone global feat). num_cls={ds.num_train_pids}",
   242	              flush=True)
   243	
   244	    def feats_from_pil(self, pil_list):
   245	        """Batched frozen features for a list of 384x128 PIL images. L2-normalized."""
   246	        torch = self.torch
   247	        feats = []
   248	        B = cli.batch
   249	        for s in range(0, len(pil_list), B):
   250	            chunk = pil_list[s:s + B]
   251	            arr = np.stack([pil_to_tensor_np(im) for im in chunk], 0)
   252	            t = torch.from_numpy(arr).cuda(non_blocking=True)
   253	            cam = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
   254	            view = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
   255	            with torch.no_grad():
   256	                out = self.model(t, cam_label=cam, view_label=view, pose_dict=None)
   257	            feat = out[0] if isinstance(out, (tuple, list)) else out
   258	            assert torch.is_tensor(feat) and feat.dim() == 2, \
   259	                f"expected single global vector, got {type(feat)} {getattr(feat,'shape',None)}"
   260	            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
   261	            feats.append(feat.cpu().numpy().astype(np.float32))
   262	        return np.concatenate(feats, 0)
   263	
   264	
   265	# =========================================================================== #
   266	# eval helpers (Market protocol: drop same pid&cam junk)
   267	# =========================================================================== #
   268	def eval_map(dist, q_pid, q_cam, g_pid, g_cam, max_rank=10):

exec
/bin/zsh -lc "nl -ba experiments/cargo_cvpb/cvpb_lattice_killswitch.py | sed -n '396,505p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   396	
   397	        # ============ build feats for every variant ==================
   398	        # STREAMING over query chunks so we never hold all K*Nq PIL images in RAM at once
   399	        # (materializing 2*K*Nq decoded 384x128 images = ~18GB at K=9,Nq=3368 -> OOM).
   400	        # phase-lattice variants -> f_lat [Nq,K,D]; ordinary-TTA -> f_tta [Nq,K,D].
   401	        Nq = len(hr_q)
   402	        f_lat_chunks, f_tta_chunks = [], []
   403	        CHUNK = 256                                  # queries per chunk -> 256*9 = 2304 PILs peak
   404	        for cs in range(0, Nq, CHUNK):
   405	            ce = min(cs + CHUNK, Nq)
   406	            lat_pils = [make_lattice_variants(hr_q[i], h, cli.K, RNG) for i in range(cs, ce)]
   407	            single_lr = [v[0] for v in lat_pils]
   408	            tta_pils = [make_tta_variants(single_lr[k], cli.K, RNG) for k in range(len(lat_pils))]
   409	            flat_lat = [im for vs in lat_pils for im in vs]
   410	            flat_tta = [im for vs in tta_pils for im in vs]
   411	            f_lat_chunks.append(ext.feats_from_pil(flat_lat).reshape(ce - cs, cli.K, -1))
   412	            f_tta_chunks.append(ext.feats_from_pil(flat_tta).reshape(ce - cs, cli.K, -1))
   413	            del lat_pils, single_lr, tta_pils, flat_lat, flat_tta
   414	        f_lat = np.concatenate(f_lat_chunks, 0)      # [Nq,K,D]
   415	        f_tta = np.concatenate(f_tta_chunks, 0)
   416	        del f_lat_chunks, f_tta_chunks
   417	        f_single = f_lat[:, 0, :]                                     # == canonical LR feat
   418	
   419	        # ---------- (A) same-image phase feature variance ----------
   420	        # mean over queries of mean pairwise (1-cos) among the K lattice variants.
   421	        def mean_pairwise_dist(F):       # F: [Nq,K,D] L2-normed
   422	            G = F @ np.transpose(F, (0, 2, 1))      # [Nq,K,K] cos
   423	            iu = np.triu_indices(F.shape[1], k=1)
   424	            pd = 1.0 - G[:, iu[0], iu[1]]            # [Nq, n_pairs]
   425	            return pd.mean(1)                        # [Nq]
   426	        phase_var = mean_pairwise_dist(f_lat)        # per-query lattice spread
   427	        tta_var = mean_pairwise_dist(f_tta)
   428	        # also: drift of canonical LR from HR (how far one LR is from the true HR feat)
   429	        lr_hr_drift = 1.0 - (f_single * hr_qf).sum(1)
   430	        print(f"  (A) same-image phase feature variance (mean pairwise 1-cos over K):")
   431	        print(f"      lattice phase-var  mean={phase_var.mean():.4f}  median={np.median(phase_var):.4f}  "
   432	              f"p90={np.quantile(phase_var,0.9):.4f}")
   433	        print(f"      ordinary TTA  var  mean={tta_var.mean():.4f}  (reference)")
   434	        print(f"      single-LR -> HR feat drift  mean={lr_hr_drift.mean():.4f}  "
   435	              f"(absolute LR distortion)")
   436	
   437	        # ---------- (B) rank volatility across phases ----------
   438	        # for each variant get top-10 gallery (raw kNN, no junk removal needed for volatility).
   439	        # argpartition (not full argsort) keeps the transient allocation small.
   440	        sims = f_lat @ gf.T                          # [Nq,K,Ng]  (~1.9GB f32)
   441	        part = np.argpartition(-sims, kth=10, axis=2)[:, :, :10]      # top-10 unordered
   442	        rows = np.arange(Nq)[:, None, None]; kk = np.arange(cli.K)[None, :, None]
   443	        ord10 = np.argsort(-sims[rows, kk, part], axis=2)             # order within 10
   444	        top10 = np.take_along_axis(part, ord10, axis=2)               # [Nq,K,10] sorted
   445	        del sims, part, ord10                        # free the big arrays before ensemble
   446	        top1 = top10[:, :, 0]                         # [Nq,K]
   447	        # top1 agreement: fraction of variants whose top1 == canonical-variant top1
   448	        top1_agree = (top1 == top1[:, [0]]).mean(1)   # [Nq]  (1.0 = perfectly stable)
   449	        # top10 Jaccard between canonical variant and each other, averaged
   450	        def jacc(a, b):
   451	            sa, sb = set(a.tolist()), set(b.tolist())
   452	            return len(sa & sb) / float(len(sa | sb))
   453	        jac10 = np.array([np.mean([jacc(top10[i, 0], top10[i, j]) for j in range(1, cli.K)])
   454	                          for i in range(len(hr_q))])
   455	        # ID-level top1 flip: does the IDENTITY of rank-1 change across phases?
   456	        top1_pid = g_pid[top1]                        # [Nq,K]
   457	        id_flip = np.array([len(np.unique(top1_pid[i])) for i in range(len(hr_q))])  # #distinct top1 IDs
   458	        print(f"  (B) rank volatility across the K phases:")
   459	        print(f"      top1 stays==canonical : mean={top1_agree.mean():.3f}  "
   460	              f"(1.0=stable; lower=more volatile)")
   461	        print(f"      top10 Jaccard(canon,j): mean={jac10.mean():.3f}  "
   462	              f"(1.0=identical sets)")
   463	        print(f"      #distinct rank-1 IDs over K phases: mean={id_flip.mean():.2f}  "
   464	              f"(>1 => the retrieved identity FLIPS with sampling phase)  "
   465	              f"frac queries with >=2 = {100*(id_flip>=2).mean():.1f}%")
   466	
   467	        # ---------- (D) ensemble mAP ----------
   468	        # single LR baseline
   469	        d_single = 1.0 - f_single @ gf.T
   470	        r_single = eval_map(d_single, q_pid, q_cam, g_pid, g_cam)
   471	        # phase-lattice ENSEMBLE: feature-mean (renormed) and MaxSim
   472	        f_lat_mean = f_lat.mean(1)
   473	        f_lat_mean /= (np.linalg.norm(f_lat_mean, axis=1, keepdims=True) + 1e-12)
   474	        d_lat_mean = 1.0 - f_lat_mean @ gf.T
   475	        r_lat_mean = eval_map(d_lat_mean, q_pid, q_cam, g_pid, g_cam)
   476	        # MaxSim: per (q,g) take the BEST sim over the K query variants
   477	        sim_lat_max = (f_lat @ gf.T).max(1)          # [Nq,Ng]
   478	        r_lat_max = eval_map(1.0 - sim_lat_max, q_pid, q_cam, g_pid, g_cam)
   479	        # ordinary-TTA ENSEMBLE (the life/death control), SAME fusions
   480	        f_tta_mean = f_tta.mean(1)
   481	        f_tta_mean /= (np.linalg.norm(f_tta_mean, axis=1, keepdims=True) + 1e-12)
   482	        r_tta_mean = eval_map(1.0 - f_tta_mean @ gf.T, q_pid, q_cam, g_pid, g_cam)
   483	        sim_tta_max = (f_tta @ gf.T).max(1)
   484	        r_tta_max = eval_map(1.0 - sim_tta_max, q_pid, q_cam, g_pid, g_cam)
   485	
   486	        print(f"  (D) ENSEMBLE mAP (K={cli.K}):")
   487	        print(f"      single bicubic LR            : mAP={r_single['mAP']:.3f}  R1={r_single['r1']:.3f}")
   488	        print(f"      phase-lattice  mean-feat     : mAP={r_lat_mean['mAP']:.3f}  R1={r_lat_mean['r1']:.3f}  "
   489	              f"(d{r_lat_mean['mAP']-r_single['mAP']:+.3f})")
   490	        print(f"      phase-lattice  MaxSim        : mAP={r_lat_max['mAP']:.3f}  R1={r_lat_max['r1']:.3f}  "
   491	              f"(d{r_lat_max['mAP']-r_single['mAP']:+.3f})")
   492	        print(f"      ----  LIFE/DEATH CONTROL (ordinary TTA, same K & fusion)  ----")
   493	        print(f"      ordinary-TTA   mean-feat     : mAP={r_tta_mean['mAP']:.3f}  R1={r_tta_mean['r1']:.3f}  "
   494	              f"(d{r_tta_mean['mAP']-r_single['mAP']:+.3f})")
   495	        print(f"      ordinary-TTA   MaxSim        : mAP={r_tta_max['mAP']:.3f}  R1={r_tta_max['r1']:.3f}  "
   496	              f"(d{r_tta_max['mAP']-r_single['mAP']:+.3f})")
   497	        best_lat = max(r_lat_mean['mAP'], r_lat_max['mAP'])
   498	        best_tta = max(r_tta_mean['mAP'], r_tta_max['mAP'])
   499	        print(f"      >> phase-lattice best gain = {best_lat-r_single['mAP']:+.3f}   "
   500	              f"ordinary-TTA best gain = {best_tta-r_single['mAP']:+.3f}   "
   501	              f"LATTICE-MINUS-TTA = {best_lat-best_tta:+.3f}  (must be clearly >0 to live)")
   502	
   503	        # ---------- (C) does phase variance EXPLAIN failure? ----------
   504	        ap_single = per_query_ap(d_single, q_pid, q_cam, g_pid, g_cam)
   505	        err = 1.0 - ap_single

exec
/bin/zsh -lc "rg -n \"freeze|frozen|requires_grad|for p in model|eval\\(|train\\(\" experiments/cargo_cvpb/cvpb_lm_reid_train.py experiments/exp359_lm_reid/design.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/exp359_lm_reid/design.md:3:> 后 PRCV「换问题」阶段第一个 GO 的 method 候选（9 个零训练 cheap-kill 全死后）。零训练 kill-switch 已 GO（d8 lattice, agent a68e6），method-design codex 判 **CCF-B 7/10**（训练版超 frozen ensemble → 8/10）。本实验做训练版，证明 lattice-marginalization 是 **method 不是 ensemble trick**。
experiments/exp359_lm_reid/design.md:11:frozen exp260b Market，K=9 lattice variants ensemble，HR gallery / LR query：
experiments/exp359_lm_reid/design.md:27:训练一个 lattice-marginalized embedding（对 lattice variants 身份稳定）+ 推理 K-marginalization，在 h=16 上比 frozen lattice ensemble **再高 +0.8~2.0 mAP** → 证明它学到了 lattice-invariance（是 method），不是 ensemble trick。
experiments/exp359_lm_reid/design.md:56:- h=16：训练版 > frozen ensemble **+0.8~2.0 mAP**；> single +5~7；> TTA +2~3.5。
experiments/exp359_lm_reid/design.md:60:失败最可能原因：训练版只 ≈ frozen ensemble（没学到额外 lattice-invariance）→ 沦为 test-time ensemble trick，不成方法稿。备选投稿角度：同等 mAP 下 K 从 9 降到 3 或 single inference 保留大部分收益。
experiments/exp359_lm_reid/design.md:66:- **frozen lattice ensemble**（零训练 K=9，= kill-switch 的 +4.23，这是训练版必须超过的硬线）。
experiments/exp359_lm_reid/design.md:87:operating point 低（只在 h≤24 强 +3-4 mAP，h=32 marginal）→ 主动收窄定位 **"severe low-resolution / cross-resolution ReID under sampling-lattice uncertainty"**，不写成通用 ReID 鲁棒性稿。h=32 marginal 反而支持机制（分辨率升高 → lattice uncertainty 消退）。真正风险 = 训练版打不过 frozen ensemble，那就只是 ensemble trick。
experiments/exp359_lm_reid/design.md:96:6. eval（**test.py 不用 train.py**）：LM-ReID(K marg) vs single / TTA / frozen-ensemble，h=16/24/32 分报，LEG/LOTG/PRF。
experiments/cargo_cvpb/cvpb_lm_reid_train.py:6:frozen K-phase ensemble (the kill-switch's +4.23 mAP @ h=16). If the trained model clears
experiments/cargo_cvpb/cvpb_lm_reid_train.py:7:frozen-ensemble +0.8~2.0 @ h=16 -> it is a METHOD; if it only ~= the frozen ensemble ->
experiments/cargo_cvpb/cvpb_lm_reid_train.py:22:Backbone fine-tune uses pose_dict=None (POSE DISABLED, identical to the frozen baseline's
experiments/cargo_cvpb/cvpb_lm_reid_train.py:82:def list_train(dir_path):
experiments/cargo_cvpb/cvpb_lm_reid_train.py:265:    cfg.freeze()
experiments/cargo_cvpb/cvpb_lm_reid_train.py:280:    t_items, n_cls = list_train(os.path.join(_repo, cli.data_root, 'market1501', 'bounding_box_train'))
experiments/cargo_cvpb/cvpb_lm_reid_train.py:299:    params = [p for p in model.parameters() if p.requires_grad] + list(disc.parameters())
experiments/cargo_cvpb/cvpb_lm_reid_train.py:322:        model.train()

codex
`torch.save(model.state_dict())` 和 kill-switch 的 `model.load_param()` 是兼容的：加载函数直接迭代 checkpoint key 并复制，`disc` 没保存也不影响 eval。接下来我会集中审 loss 和 batch/variant 对齐，尤其是 GRL 与 KL。
exec
/bin/zsh -lc "nl -ba model/pose_backbone_model.py | sed -n '540,590p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   540	        featmap = outs[-1]  # (B, C, fH, fW)
   541	
   542	        # Standard GAP
   543	        global_feat = self.base.avgpool(featmap)
   544	        global_feat = torch.flatten(global_feat, 1)
   545	
   546	        return global_feat, outs
   547	
   548	    def _run_stage_with_psg(self, stage, x, hw_shape, scene_heatmaps,
   549	                            stage_idx=None):
   550	        """Run a stage's blocks with PSG and optional PAA injection."""
   551	        for block_idx, block in enumerate(stage.blocks):
   552	            key = f's{stage_idx}_b{block_idx}'
   553	
   554	            # Run the Swin block
   555	            x = block(x, hw_shape)
   556	
   557	            # PSG: apply gate after block
   558	            if scene_heatmaps is not None and key in self.psg_modules_dict:
   559	                x = self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)
   560	
   561	            # PAA: apply additive adapter after PSG
   562	            if getattr(self, 'use_paa', False) and scene_heatmaps is not None and key in getattr(self, 'paa_modules_dict', {}):
   563	                x = self.paa_modules_dict[key](x, hw_shape, scene_heatmaps)
   564	
   565	        # Handle downsample (Stage 3 has no downsample in Swin)
   566	        if stage.downsample:
   567	            x_down, down_hw_shape = stage.downsample(x, hw_shape)
   568	            return x_down, down_hw_shape, x, hw_shape
   569	        else:
   570	            return x, hw_shape, x, hw_shape
   571	
   572	    def _canonical_heatmap(self, B, device):
   573	        """Fixed canonical upright-pedestrian COCO-17 pose heatmap (NO per-image info).
   574	        FIXED-BANDS mode feeds this in place of per-image pose, giving the CLIP-text part
   575	        queries a fixed anatomical localization prior (head top -> ankles bottom)."""
   576	        if self._canon_hm_cache is None:
   577	            H, W = 96, 32
   578	            KP = [(0.50, 0.06), (0.46, 0.05), (0.54, 0.05), (0.42, 0.06), (0.58, 0.06),
   579	                  (0.36, 0.18), (0.64, 0.18), (0.32, 0.32), (0.68, 0.32), (0.30, 0.45), (0.70, 0.45),
   580	                  (0.40, 0.50), (0.60, 0.50), (0.41, 0.72), (0.59, 0.72), (0.42, 0.95), (0.58, 0.95)]
   581	            ys = torch.arange(H, dtype=torch.float32).view(H, 1)
   582	            xs = torch.arange(W, dtype=torch.float32).view(1, W)
   583	            hm = torch.zeros(1, 17, H, W)
   584	            sx, sy = 0.12 * W, 0.05 * H
   585	            for k, (nx, ny) in enumerate(KP):
   586	                cx, cy = nx * W, ny * H
   587	                hm[0, k] = torch.exp(-(((xs - cx) ** 2) / (2 * sx ** 2) + ((ys - cy) ** 2) / (2 * sy ** 2)))
   588	            self._canon_hm_cache = hm
   589	        return self._canon_hm_cache.to(device).expand(B, 17, -1, -1).contiguous()
   590	

exec
/bin/zsh -lc 'rg -n "def __init__|self.ID_LOSS_TYPE|label_smooth|classifier|bottleneck|BatchNorm|global_feat" model/make_model.py config/defaults.py configs/market/pose_psg_lgpa_gcn_base.yml' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
model/make_model.py:50:    elif classname.find('BatchNorm') != -1:
model/make_model.py:55:def weights_init_classifier(m):
model/make_model.py:64:    def __init__(self, num_classes, cfg):
model/make_model.py:102:        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
model/make_model.py:103:        self.classifier.apply(weights_init_classifier)
model/make_model.py:105:        self.bottleneck = nn.BatchNorm1d(self.in_planes)
model/make_model.py:106:        self.bottleneck.bias.requires_grad_(False)
model/make_model.py:107:        self.bottleneck.apply(weights_init_kaiming)
model/make_model.py:118:        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
model/make_model.py:119:        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
model/make_model.py:121:            global_feat = self.fcneck(global_feat)
model/make_model.py:124:            feat = global_feat
model/make_model.py:126:            feat = self.bottleneck(global_feat)
model/make_model.py:134:                cls_score = self.classifier(feat)
model/make_model.py:135:            return cls_score, global_feat
model/make_model.py:140:                return global_feat
model/make_model.py:147:            if 'classifier' in i:
model/make_model.py:166:    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
model/make_model.py:212:        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
model/make_model.py:213:        if self.ID_LOSS_TYPE == 'arcface':
model/make_model.py:214:            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
model/make_model.py:215:            self.classifier = Arcface(self.in_planes, self.num_classes,
model/make_model.py:217:        elif self.ID_LOSS_TYPE == 'cosface':
model/make_model.py:218:            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
model/make_model.py:219:            self.classifier = Cosface(self.in_planes, self.num_classes,
model/make_model.py:221:        elif self.ID_LOSS_TYPE == 'amsoftmax':
model/make_model.py:222:            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
model/make_model.py:223:            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
model/make_model.py:225:        elif self.ID_LOSS_TYPE == 'circle':
model/make_model.py:226:            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
model/make_model.py:227:            self.classifier = CircleLoss(self.in_planes, self.num_classes,
model/make_model.py:234:            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
model/make_model.py:235:            self.classifier.apply(weights_init_classifier)
model/make_model.py:237:        self.bottleneck = nn.BatchNorm1d(self.in_planes)
model/make_model.py:238:        self.bottleneck.bias.requires_grad_(False)
model/make_model.py:239:        self.bottleneck.apply(weights_init_kaiming)
model/make_model.py:247:        global_feat, featmaps = self.base(x)
model/make_model.py:249:            global_feat = self.fcneck(global_feat)
model/make_model.py:250:        feat = self.bottleneck(global_feat)
model/make_model.py:254:            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
model/make_model.py:255:                cls_score = self.classifier(feat_cls, label)
model/make_model.py:257:                cls_score = self.classifier(feat_cls)
model/make_model.py:259:            return cls_score, global_feat, featmaps  # global feature for triplet loss
model/make_model.py:266:                return global_feat, featmaps
model/make_model.py:279:    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
model/make_model.py:317:        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
model/make_model.py:318:        if self.ID_LOSS_TYPE == 'arcface':
model/make_model.py:319:            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
model/make_model.py:320:            self.classifier = Arcface(self.in_planes, self.num_classes,
model/make_model.py:322:        elif self.ID_LOSS_TYPE == 'cosface':
model/make_model.py:323:            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
model/make_model.py:324:            self.classifier = Cosface(self.in_planes, self.num_classes,
model/make_model.py:326:        elif self.ID_LOSS_TYPE == 'amsoftmax':
model/make_model.py:327:            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
model/make_model.py:328:            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
model/make_model.py:330:        elif self.ID_LOSS_TYPE == 'circle':
model/make_model.py:331:            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
model/make_model.py:332:            self.classifier = CircleLoss(self.in_planes, self.num_classes,
model/make_model.py:335:            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
model/make_model.py:336:            self.classifier.apply(weights_init_classifier)
model/make_model.py:337:            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
model/make_model.py:338:            self.classifier_1.apply(weights_init_classifier)
model/make_model.py:339:            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
model/make_model.py:340:            self.classifier_2.apply(weights_init_classifier)
model/make_model.py:341:            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
model/make_model.py:342:            self.classifier_3.apply(weights_init_classifier)
model/make_model.py:343:            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
model/make_model.py:344:            self.classifier_4.apply(weights_init_classifier)
model/make_model.py:346:        self.bottleneck = nn.BatchNorm1d(self.in_planes)
model/make_model.py:347:        self.bottleneck.bias.requires_grad_(False)
model/make_model.py:348:        self.bottleneck.apply(weights_init_kaiming)
model/make_model.py:349:        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
model/make_model.py:350:        self.bottleneck_1.bias.requires_grad_(False)
model/make_model.py:351:        self.bottleneck_1.apply(weights_init_kaiming)
model/make_model.py:352:        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
model/make_model.py:353:        self.bottleneck_2.bias.requires_grad_(False)
model/make_model.py:354:        self.bottleneck_2.apply(weights_init_kaiming)
model/make_model.py:355:        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
model/make_model.py:356:        self.bottleneck_3.bias.requires_grad_(False)
model/make_model.py:357:        self.bottleneck_3.apply(weights_init_kaiming)
model/make_model.py:358:        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
model/make_model.py:359:        self.bottleneck_4.bias.requires_grad_(False)
model/make_model.py:360:        self.bottleneck_4.apply(weights_init_kaiming)
model/make_model.py:376:        global_feat = b1_feat[:, 0]
model/make_model.py:407:        feat = self.bottleneck(global_feat)
model/make_model.py:409:        local_feat_1_bn = self.bottleneck_1(local_feat_1)
model/make_model.py:410:        local_feat_2_bn = self.bottleneck_2(local_feat_2)
model/make_model.py:411:        local_feat_3_bn = self.bottleneck_3(local_feat_3)
model/make_model.py:412:        local_feat_4_bn = self.bottleneck_4(local_feat_4)
model/make_model.py:415:            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
model/make_model.py:416:                cls_score = self.classifier(feat, label)
model/make_model.py:418:                cls_score = self.classifier(feat)
model/make_model.py:419:                cls_score_1 = self.classifier_1(local_feat_1_bn)
model/make_model.py:420:                cls_score_2 = self.classifier_2(local_feat_2_bn)
model/make_model.py:421:                cls_score_3 = self.classifier_3(local_feat_3_bn)
model/make_model.py:422:                cls_score_4 = self.classifier_4(local_feat_4_bn)
model/make_model.py:425:                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
model/make_model.py:433:                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)
config/defaults.py:208:_C.MODEL.POSE_GCN_PER_PART = False        # Split 17 keypoints into 6 body parts, each with own classifier

exec
/bin/zsh -lc 'rg -n "MODEL.DEVICE_ID|CUDA_VISIBLE_DEVICES|os.environ" train.py processor model datasets experiments/cargo_cvpb/cvpb_lm_reid_train.py' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
experiments/cargo_cvpb/cvpb_lm_reid_train.py:266:    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
train.py:68:    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

codex
从模型 forward 看，`out[0]`/`out[1]` 取法在本配置下有效：训练态 `PoseBackboneModel` line 1088 返回 `cls_score, global_feat, featmaps, None`，而脚本对 list/tuple 的 defensive unwrap 也不会误伤当前路径。我要再查一下 backbone 是否依赖 `cam_label/view_label`，以及 `M=2` 时 loss/sampler 有没有退化问题。
exec
/bin/zsh -lc 'rg -n "ID_LOSS_TYPE|NO_MARGIN|MODEL.NECK|DROPOUT_RATE|NECK_FEAT" config/defaults.py configs/market/pose_psg_lgpa_gcn_base.yml' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
configs/market/pose_psg_lgpa_gcn_base.yml:9:  NO_MARGIN: True
configs/market/pose_psg_lgpa_gcn_base.yml:76:  NECK_FEAT: 'before'
config/defaults.py:35:_C.MODEL.NECK = 'bnneck'
config/defaults.py:39:_C.MODEL.ID_LOSS_TYPE = 'softmax'
config/defaults.py:47:_C.MODEL.NO_MARGIN = False
config/defaults.py:53:_C.MODEL.DROPOUT_RATE = 0.0
config/defaults.py:471:_C.TEST.NECK_FEAT = 'after'

exec
/bin/zsh -lc "rg -n \"class Arcface|class Cosface|class AMSoftmax|class CircleLoss|def forward\\(self,.*label\" loss model -S" in /Users/abdslm/Desktop/SOLIDER-REID
exec
/bin/zsh -lc 'rg -n "cam_label|view_label" model backbones -S' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
model/pose_psg_part_model.py:33:    def forward(self, x, label=None, cam_label=None, view_label=None,
model/modules/clip_id_prompt.py:80:    def forward(self, label, pose=None):
model/modules/skeleton_gcn.py:757:    def forward(self, feat_map, pose_dict, return_cls=True, label=None,
loss/metric_learning.py:45:class CircleLoss(nn.Module):
loss/metric_learning.py:76:class Arcface(nn.Module):
loss/metric_learning.py:101:    def forward(self, input, label):
loss/metric_learning.py:124:class Cosface(nn.Module):
loss/metric_learning.py:143:    def forward(self, input, label):
loss/metric_learning.py:166:class AMSoftmax(nn.Module):
loss/arcface.py:35:    def forward(self, input, label):
loss/arcface.py:52:class CircleLoss(nn.Module):
model/pose_dual_stream_model.py:222:    def forward(self, x, label=None, cam_label=None, view_label=None,
model/pose_backbone_model.py:705:    def forward(self, x, label=None, cam_label=None, view_label=None,
model/make_model.py:116:    def forward(self, x, label=None, **kwargs):  # label is unused if self.cos_layer == 'no'
model/make_model.py:246:    def forward(self, x, label=None, cam_label= None, view_label=None):
model/make_model.py:370:    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
model/pose_model.py:58:    def forward(self, x, label=None, cam_label=None, view_label=None,
loss/center_loss.py:29:    def forward(self, x, labels):
model/backbones/resnet.py:115:    def forward(self, x, cam_label=None):
loss/supcon_loss.py:26:    def forward(self, features, labels):
model/backbones/vit_pytorch.py:337:    def forward(self, x, cam_label=None, view_label=None):

 exited 2 in 0ms:
rg: backbones: No such file or directory (os error 2)
model/pose_psg_part_model.py:33:    def forward(self, x, label=None, cam_label=None, view_label=None,
model/pose_backbone_model.py:705:    def forward(self, x, label=None, cam_label=None, view_label=None,
model/pose_dual_stream_model.py:222:    def forward(self, x, label=None, cam_label=None, view_label=None,
model/backbones/vit_pytorch.py:337:    def forward(self, x, cam_label=None, view_label=None):
model/backbones/vit_pytorch.py:338:        x = self.forward_features(x, cam_label, view_label)
model/backbones/resnet.py:115:    def forward(self, x, cam_label=None):
model/make_model.py:246:    def forward(self, x, label=None, cam_label= None, view_label=None):
model/make_model.py:370:    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
model/make_model.py:372:        features = self.base(x, cam_label=cam_label, view_label=view_label)
model/pose_model.py:58:    def forward(self, x, label=None, cam_label=None, view_label=None,

codex
`cam_label/view_label` 目前传全 0 不会破坏这个配置：Swin forward 支持这些参数，但是否实际用取决于 camera/view embedding 开关；我会继续看实现，避免隐藏的 device/shape 假设漏掉。
exec
/bin/zsh -lc "nl -ba model/backbones/vit_pytorch.py | sed -n '300,345p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   300	
   301	    def reset_classifier(self, num_classes, global_pool=''):
   302	        self.num_classes = num_classes
   303	        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
   304	
   305	    def forward_features(self, x, camera_id, view_id):
   306	        B = x.shape[0]
   307	        x = self.patch_embed(x)
   308	
   309	        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
   310	        x = torch.cat((cls_tokens, x), dim=1)
   311	
   312	        if self.cam_num > 0 and self.view_num > 0:
   313	            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
   314	        elif self.cam_num > 0:
   315	            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
   316	        elif self.view_num > 0:
   317	            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
   318	        else:
   319	            x = x + self.pos_embed
   320	
   321	        x = self.pos_drop(x)
   322	
   323	        if self.local_feature:
   324	            for blk in self.blocks[:-1]:
   325	                x = blk(x)
   326	            return x
   327	        else:
   328	            for blk in self.blocks:
   329	                x = blk(x)
   330	
   331	            x = self.norm(x)
   332	        if self.gem_pool:
   333	            gf = self.gem(x[:,1:].permute(0,2,1)).squeeze()
   334	            return x[:, 0] + gf
   335	        return x[:, 0]
   336	
   337	    def forward(self, x, cam_label=None, view_label=None):
   338	        x = self.forward_features(x, cam_label, view_label)
   339	        return x
   340	
   341	    def load_param(self, model_path,hw_ratio):
   342	        param_dict = torch.load(model_path, map_location='cpu', weights_only=False)
   343	        count=0
   344	        if 'model' in param_dict:
   345	            param_dict = param_dict['model']

exec
/bin/zsh -lc "nl -ba model/make_model.py | sed -n '188,245p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
   188	            view_num = 0
   189	
   190	        convert_weights = True if pretrain_choice == 'imagenet' else False
   191	        backbone_kwargs = dict(
   192	            img_size=cfg.INPUT.SIZE_TRAIN,
   193	            drop_path_rate=cfg.MODEL.DROP_PATH,
   194	            drop_rate=cfg.MODEL.DROP_OUT,
   195	            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
   196	            pretrained=model_path,
   197	            convert_weights=convert_weights,
   198	            semantic_weight=semantic_weight,
   199	        )
   200	        # Gradient checkpointing is implemented in the Swin backbone only.
   201	        if cfg.MODEL.TRANSFORMER_TYPE.startswith('swin'):
   202	            backbone_kwargs['with_cp'] = cfg.MODEL.WITH_CP
   203	        elif cfg.MODEL.WITH_CP:
   204	            print('WITH_CP is ignored for non-Swin backbones in this codebase')
   205	
   206	        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](**backbone_kwargs)
   207	        if model_path != '':
   208	            self.base.init_weights(model_path)
   209	        self.in_planes = self.base.num_features[-1]
   210	
   211	        self.num_classes = num_classes
   212	        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
   213	        if self.ID_LOSS_TYPE == 'arcface':
   214	            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
   215	            self.classifier = Arcface(self.in_planes, self.num_classes,
   216	                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
   217	        elif self.ID_LOSS_TYPE == 'cosface':
   218	            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
   219	            self.classifier = Cosface(self.in_planes, self.num_classes,
   220	                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
   221	        elif self.ID_LOSS_TYPE == 'amsoftmax':
   222	            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
   223	            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
   224	                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
   225	        elif self.ID_LOSS_TYPE == 'circle':
   226	            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
   227	            self.classifier = CircleLoss(self.in_planes, self.num_classes,
   228	                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
   229	        else:
   230	            if self.reduce_feat_dim:
   231	                self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
   232	                self.fcneck.apply(weights_init_xavier)
   233	                self.in_planes = cfg.MODEL.FEAT_DIM
   234	            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
   235	            self.classifier.apply(weights_init_classifier)
   236	
   237	        self.bottleneck = nn.BatchNorm1d(self.in_planes)
   238	        self.bottleneck.bias.requires_grad_(False)
   239	        self.bottleneck.apply(weights_init_kaiming)
   240	
   241	        self.dropout = nn.Dropout(self.dropout_rate)
   242	
   243	        #if pretrain_choice == 'self':
   244	        #    self.load_param(model_path)
   245	

exec
/bin/zsh -lc 'rg -n "cam_label|view_label|camera|view|sie|SIE|semantic" model/backbones/vit_pytorch.py config/defaults.py configs/market/pose_psg_lgpa_gcn_base.yml' in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
config/defaults.py:73:# SIE Parameter
config/defaults.py:74:_C.MODEL.SIE_COE = 3.0
config/defaults.py:75:_C.MODEL.SIE_CAMERA = False
config/defaults.py:76:_C.MODEL.SIE_VIEW = False
config/defaults.py:187:# Parallel augmentation (3-view training)
config/defaults.py:224:_C.MODEL.POSE_LGPA_FIXED_BANDS = False    # Fixed-semantics: replace per-image pose with a FIXED canonical pedestrian pose (fixed CLIP text + fixed anatomical prior, no per-image pose)
config/defaults.py:225:_C.MODEL.POSE_LGPA_RANDOM_TEXT = False     # Attribution ablation: replace CLIP text prototypes with FIXED random vectors (tests if CLIP semantics contribute vs pure structure)    # Fixed-semantics: replace per-image pose with a FIXED canonical pedestrian pose (fixed CLIP text + fixed anatomical prior, no per-image pose)
config/defaults.py:244:# exp356 PC-MSC: pose-conditioned masked semantic completion (training-only)
config/defaults.py:246:                                     # part's frozen-CLIP-visual semantic from the visible context
model/backbones/vit_pytorch.py:231:    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, view=0,drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), local_feature=False, sie_xishu =1.0, hw_ratio=1, gem_pool = False, stem_conv=False):
model/backbones/vit_pytorch.py:244:        self.cam_num = camera
model/backbones/vit_pytorch.py:245:        self.view_num = view
model/backbones/vit_pytorch.py:246:        self.sie_xishu = sie_xishu
model/backbones/vit_pytorch.py:251:        # Initialize SIE Embedding
model/backbones/vit_pytorch.py:252:        if camera > 1 and view > 1:
model/backbones/vit_pytorch.py:253:            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
model/backbones/vit_pytorch.py:254:            trunc_normal_(self.sie_embed, std=.02)
model/backbones/vit_pytorch.py:255:        elif camera > 1:
model/backbones/vit_pytorch.py:256:            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
model/backbones/vit_pytorch.py:257:            trunc_normal_(self.sie_embed, std=.02)
model/backbones/vit_pytorch.py:258:        elif view > 1:
model/backbones/vit_pytorch.py:259:            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
model/backbones/vit_pytorch.py:260:            trunc_normal_(self.sie_embed, std=.02)
model/backbones/vit_pytorch.py:305:    def forward_features(self, x, camera_id, view_id):
model/backbones/vit_pytorch.py:312:        if self.cam_num > 0 and self.view_num > 0:
model/backbones/vit_pytorch.py:313:            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
model/backbones/vit_pytorch.py:315:            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
model/backbones/vit_pytorch.py:316:        elif self.view_num > 0:
model/backbones/vit_pytorch.py:317:            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
model/backbones/vit_pytorch.py:337:    def forward(self, x, cam_label=None, view_label=None):
model/backbones/vit_pytorch.py:338:        x = self.forward_features(x, cam_label, view_label)
model/backbones/vit_pytorch.py:401:def vit_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
model/backbones/vit_pytorch.py:402:    model = TransReID(img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, camera=camera, view=view, drop_path_rate=drop_path_rate, sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)
model/backbones/vit_pytorch.py:405:def vit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
model/backbones/vit_pytorch.py:406:    model = TransReID(img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,drop_path_rate=drop_path_rate, camera=camera, view=view, sie_xishu=sie_xishu, local_feature=local_feature,  **kwargs)

web search: 
web search: "lattice marginalization" person re-identification low resolution sub-pixel sampling
web search: 
web search: low resolution person re-identification sampling uncertainty aliasing subpixel
web search: 
web search: "subpixel" "Re-ID" "low resolution"
web search: 
web search: "sampling-grid" "person re-identification"
web search: 
web search: "Making Convolutional Networks Shift-Invariant Again" BlurPool anti-aliasing
web search: 
web search: "VPFA" "person re-identification" "Vector Panning"
web search: 
web search: "sub-pixel" "sampling uncertainty" deep learning image recognition
web search: 
web search: "phase" "ReID" "low-resolution"
web search: 
web search: https://arxiv.org/abs/2510.00936
web search: 
web search: 'resolution-invariant' in https://arxiv.org/abs/2207.13037
codex
**Verdict: needs-attention**

No Critical runtime/shape/device bug found, but I would not launch this as-is because two training-design issues can invalidate exp359.

**Findings**

- **High** [cvpb_lm_reid_train.py:173](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:173): per-sample RNG is reset from `seed + idx`, so each image gets the same height and same lattice variants every epoch. This contradicts design “每图每 iter 采样 M variants”.
  Fix: include epoch/worker/call randomness, e.g. dataset `set_epoch(ep)` plus `seed ^ epoch ^ idx`, or use worker-seeded `np.random`/`torch.initial_seed()` so variants change per iteration while remaining reproducible.

- **High** [cvpb_lm_reid_train.py:48](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:48), [cvpb_lm_reid_train.py:118](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:118): default `M=2` plus `mode = j % 3` trains only canonical + bbox-shift/bilinear. It does not train sub-pixel phase or zoom/antialias axes despite the method claim.
  Fix: make real training default `M>=4` at minimum, or sample perturbation mode/kernel for each noncanonical variant; if keeping `L_adv`, store actual lattice labels.

- **Medium** [cvpb_lm_reid_train.py:372](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:372), [cvpb_lm_reid_train.py:377](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:377): GRL adversarial loss is lambda-scaled twice for the backbone gradient: once inside `GradReverse(..., adv_lamb)` and once in `+ adv_lamb * L_adv`. With `0.02-0.05`, the adversarial signal becomes `λ²`.
  Fix: either `GradReverse.apply(z, 1.0)` and keep `adv_lamb * L_adv`, or keep GRL lambda and add unweighted `L_adv`.

- **Medium** [cvpb_lm_reid_train.py:345](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:345): `L_id` triplet is computed over flattened `B*M`, not `mean_l Triplet(gf[:, l], y)`. This treats other variants of the same image as positives and is not exactly the stated per-lattice-slot formula.
  Fix if strict design matters: reshape first and average `batch_hard_triplet(gf_bm[:, m], y, margin)` over `m`.

- **Low** [cvpb_lm_reid_train.py:96](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:96): lattice generation appears output-equivalent to kill-switch, but not literally byte-identical source.  
  Fix: move shared degradation functions into one imported helper, or add a hash/regression test comparing generated PIL bytes for fixed seed.

**Checks That Look Correct**

- `pose_dict=None` path is correct. `PoseBackboneModel.forward` returns `(cls_score, global_feat, featmaps, None)` at [pose_backbone_model.py:1088](/Users/abdslm/Desktop/SOLIDER-REID/model/pose_backbone_model.py:1088). Pose/LGPA/GCN branches are skipped because `scene_heatmaps is None`; `_lgpa_fixed_bands` default is `False`.
- AMP is basically safe: autocast wraps only model forward; `cls/gf.float()` makes loss fp32; `GradScaler.scale/step/update` is used.
- Image-major layout is aligned: `xb.view(B*M, ...)`, `y.repeat_interleave(M)`, `view(B,M,...)`, and `arange(M).repeat(B)` all match `[i*M+m]`.
- `batch_hard_triplet` masks are correct under default PK sampling.
- `PKSampler` default gives positives and negatives (`P=16,Kins=4`).
- `torch.save(model.state_dict())` is compatible with SOLIDER `load_param`, which copies matching keys and ignores failures.

**Web Search**

I did not find an exact prior for “sampling-lattice hidden variable / sub-pixel phase marginalization” in low-res ReID. Closest works model resolution gap or resolution-invariant features, not lattice phase marginalization: VPFA, LRAR, RFD; BlurPool covers general anti-aliasing/shift-invariance but not ReID lattice marginalization. Sources: [VPFA](https://arxiv.org/abs/2510.00936), [LRAR](https://arxiv.org/abs/2207.13037), [RFD](https://arxiv.org/abs/2109.07871), [BlurPool](https://arxiv.org/abs/1904.11486).
tokens used
184,386
**Verdict: needs-attention**

No Critical runtime/shape/device bug found, but I would not launch this as-is because two training-design issues can invalidate exp359.

**Findings**

- **High** [cvpb_lm_reid_train.py:173](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:173): per-sample RNG is reset from `seed + idx`, so each image gets the same height and same lattice variants every epoch. This contradicts design “每图每 iter 采样 M variants”.
  Fix: include epoch/worker/call randomness, e.g. dataset `set_epoch(ep)` plus `seed ^ epoch ^ idx`, or use worker-seeded `np.random`/`torch.initial_seed()` so variants change per iteration while remaining reproducible.

- **High** [cvpb_lm_reid_train.py:48](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:48), [cvpb_lm_reid_train.py:118](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:118): default `M=2` plus `mode = j % 3` trains only canonical + bbox-shift/bilinear. It does not train sub-pixel phase or zoom/antialias axes despite the method claim.
  Fix: make real training default `M>=4` at minimum, or sample perturbation mode/kernel for each noncanonical variant; if keeping `L_adv`, store actual lattice labels.

- **Medium** [cvpb_lm_reid_train.py:372](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:372), [cvpb_lm_reid_train.py:377](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:377): GRL adversarial loss is lambda-scaled twice for the backbone gradient: once inside `GradReverse(..., adv_lamb)` and once in `+ adv_lamb * L_adv`. With `0.02-0.05`, the adversarial signal becomes `λ²`.
  Fix: either `GradReverse.apply(z, 1.0)` and keep `adv_lamb * L_adv`, or keep GRL lambda and add unweighted `L_adv`.

- **Medium** [cvpb_lm_reid_train.py:345](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:345): `L_id` triplet is computed over flattened `B*M`, not `mean_l Triplet(gf[:, l], y)`. This treats other variants of the same image as positives and is not exactly the stated per-lattice-slot formula.
  Fix if strict design matters: reshape first and average `batch_hard_triplet(gf_bm[:, m], y, margin)` over `m`.

- **Low** [cvpb_lm_reid_train.py:96](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/cvpb_lm_reid_train.py:96): lattice generation appears output-equivalent to kill-switch, but not literally byte-identical source.  
  Fix: move shared degradation functions into one imported helper, or add a hash/regression test comparing generated PIL bytes for fixed seed.

**Checks That Look Correct**

- `pose_dict=None` path is correct. `PoseBackboneModel.forward` returns `(cls_score, global_feat, featmaps, None)` at [pose_backbone_model.py:1088](/Users/abdslm/Desktop/SOLIDER-REID/model/pose_backbone_model.py:1088). Pose/LGPA/GCN branches are skipped because `scene_heatmaps is None`; `_lgpa_fixed_bands` default is `False`.
- AMP is basically safe: autocast wraps only model forward; `cls/gf.float()` makes loss fp32; `GradScaler.scale/step/update` is used.
- Image-major layout is aligned: `xb.view(B*M, ...)`, `y.repeat_interleave(M)`, `view(B,M,...)`, and `arange(M).repeat(B)` all match `[i*M+m]`.
- `batch_hard_triplet` masks are correct under default PK sampling.
- `PKSampler` default gives positives and negatives (`P=16,Kins=4`).
- `torch.save(model.state_dict())` is compatible with SOLIDER `load_param`, which copies matching keys and ignores failures.

**Web Search**

I did not find an exact prior for “sampling-lattice hidden variable / sub-pixel phase marginalization” in low-res ReID. Closest works model resolution gap or resolution-invariant features, not lattice phase marginalization: VPFA, LRAR, RFD; BlurPool covers general anti-aliasing/shift-invariance but not ReID lattice marginalization. Sources: [VPFA](https://arxiv.org/abs/2510.00936), [LRAR](https://arxiv.org/abs/2207.13037), [RFD](https://arxiv.org/abs/2109.07871), [BlurPool](https://arxiv.org/abs/1904.11486).
