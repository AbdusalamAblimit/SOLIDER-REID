Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019ef6ae-0c30-7ae1-8e9c-2bfa5350a3eb
--------
user
你是 10 名独立审查员中的第 8 名,对 AIRL 做全量代码 + log 审查(不是只看你的 lens,要全量,但本轮额外深挖下面这个重点)。
背景: AIRL = aerial-ground person ReID 的 degradation-consistency 双分支方法。
- code/afd_model.py: 模型 AFDModel,含 _forward_swin_split / iso dual-branch (f_full 干净头 + f_rec recover 头)。
- code/afd_train.py: 训练 + eval。do_inference 双分支 eval 报 full/rec/FUSE 各方向。
- config: --airl_dualbranch_iso --airl_iso_stage 3 --airl_iso_trunk_recce 0 --airl_fuse_w 0.25;soft-fuse: cos = 0.25*cos(f_rec) + 0.75*cos(f_full)。
- logs/: agreidv2_airl_4090.log (AIRL final, lab-4090, torch2.6) / agreidv2_airl_iso.log (AIRL, lab-3090, torch2.7, 跑到 ep40+) / agreidv2_baseline_4090.log (baseline seed1, lab-4090) / agreidv2_baseline_4090_s2.log (baseline seed2, lab-4090) / agreidv2_baseline.log (baseline, lab-3090)。
- 每 epoch eval 打两行: [A->G] full mAP= rec mAP= FUSE mAP= 和 [G->A] 同。baseline 不带 full/rec(单特征)。

我(主控)反复改判过这个结果,可能有错,请独立核这 3 条待验结论:
(1) AIRL net gain (FUSE mean vs baseline final mean) ≈ 融合增益 FUSE−full ~+1.5,双数据集一致;
(2) AIRL 的 f_full 头 ≈ baseline ± seed 噪声(±1.2):CARGO 时 full 头走运比 baseline +0.6 → net +2.37;AG-ReID.v2 lab-4090 时 full 头背运 −1.2 → net ≈ 0(我一度据此判 AIRL 死,后改判是 seed 噪声);
(3) 方向特化(rec/full 哪个头帮 A->G/G->A 哪个检索方向)是 CARGO-specific,第二数据集方向翻号/消失。

== 本轮重点 lens ==
数值/AMP: 有无 AMP autocast?f_full/f_rec 是否都 L2-normalize?cos 数值稳定吗?有无潜在 NaN/Inf?fuse_w 的 dtype/device 对吗?

== 要求 ==
逐行读 code/afd_model.py 和 code/afd_train.py 的相关部分,并对照 logs/ 里的真实数字。
输出格式:
1. Verdict: approve(代码与结果可信)/ needs-attention(有问题)
2. Findings: 每条给 severity(Critical/High/Medium/Low)+ 文件:行号 + 具体问题 + 为什么
3. 对主控 3 条待验结论的独立判断: 逐条 同意 / 反对 / log数据不支持,给依据
4. 本轮 lens 的专项结论
用 web search 查相关 novelty/先例。中文输出,务实不客套。
codex
我会先按仓库要求读当前研究文档，然后逐行核 `afd_model.py` / `afd_train.py` 的 AIRL 路径和 logs 里的真实数字。因为你明确要求查 novelty/先例，我也会用 web search 做一轮外部对照。
exec
/bin/zsh -lc "pwd && rg --files | sed -n '1,160p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
exec
/bin/zsh -lc 'for f in experiments/results.md experiments/decisions.md experiments/innovation_brainstorm.md experiments/paper_materials/story.md; do if [ -f "$f" ]; then printf '"'%s\\n' \""'$f"; wc -l "$f"; fi; done' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
reviews/codex_6.md
reviews/codex_2.md
reviews/codex_3.md
reviews/codex_7.md
reviews/codex_10.md
reviews/codex_8.md
reviews/codex_9.md
reviews/codex_4.md
reviews/codex_5.md
reviews/codex_1.md
logs/agreidv2_baseline_4090_s2.log
logs/agreidv2_airl_iso.log
logs/agreidv2_baseline_4090.log
logs/agreidv2_baseline.log
logs/agreidv2_airl_4090.log
code/afd_train.py
code/afd_model.py
launch_10codex.sh

 succeeded in 0ms:
exec
/bin/zsh -lc "ls -la code logs experiments | sed -n '1,220p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
ls: experiments: No such file or directory
code:
total 392
drwxr-xr-x@ 4 abdslm  staff     128 Jun 24 06:43 .
drwxr-xr-x@ 6 abdslm  staff     192 Jun 24 06:50 ..
-rw-r--r--@ 1 abdslm  staff   47042 Jun 23 19:16 afd_model.py
-rw-r--r--@ 1 abdslm  staff  150590 Jun 23 22:25 afd_train.py

logs:
total 848
drwxr-xr-x@ 7 abdslm  staff     224 Jun 24 06:43 .
drwxr-xr-x@ 6 abdslm  staff     192 Jun 24 06:50 ..
-rw-r--r--@ 1 abdslm  staff  108760 Jun 24 02:55 agreidv2_airl_4090.log
-rw-r--r--@ 1 abdslm  staff   90527 Jun 24 06:42 agreidv2_airl_iso.log
-rw-r--r--@ 1 abdslm  staff   75012 Jun 24 03:04 agreidv2_baseline.log
-rw-r--r--@ 1 abdslm  staff   74870 Jun 24 05:01 agreidv2_baseline_4090.log
-rw-r--r--@ 1 abdslm  staff   70386 Jun 24 06:43 agreidv2_baseline_4090_s2.log

codex
这个 bundle 里没有 `experiments/` 文档目录，只有 `code/`、`logs/` 和已有 review 文件。我先不读其他审查员结论，避免把独立判断污染掉。
exec
/bin/zsh -lc 'wc -l code/afd_model.py code/afd_train.py logs/*.log' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
     898 code/afd_model.py
    2445 code/afd_train.py
     981 logs/agreidv2_airl_4090.log
     811 logs/agreidv2_airl_iso.log
     957 logs/agreidv2_baseline.log
     956 logs/agreidv2_baseline_4090.log
     896 logs/agreidv2_baseline_4090_s2.log
    7944 total

exec
/bin/zsh -lc "rg -n \"AIRL|airl|AFDModel|dualbranch|iso|_forward_swin_split|f_full|f_rec|normalize|F\\.normalize|autocast|GradScaler|amp|nan|inf|fuse|do_inference|cos\" code/afd_model.py code/afd_train.py" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
code/afd_model.py:7:    train returns (global_feat, logits); test returns L2-normalized BN feature.
code/afd_model.py:17:      training-time regularizer (no test-time cost). Two switchable sub-mechanisms:
code/afd_model.py:81:    Exposes the SAME hook contract as the resnet50 AFDModel:
code/afd_model.py:86:        (AFDModel pools it -> BNNeck), mirroring resnet's ``_forward_backbone``.
code/afd_model.py:96:                 drop_path_rate=0.1, iso_branch=False, iso_stage=3,
code/afd_model.py:97:                 iso_trunk_recce=True):
code/afd_model.py:123:        # ---- AIRL gradient-isolated dual-branch (f_rec independent late stage) --
code/afd_model.py:124:        # iso_branch=True forks a SECOND last-stage path (f_rec) off the shared
code/afd_model.py:125:        # residual stream at the input of stage `iso_stage`.  The rec path is an
code/afd_model.py:126:        # INDEPENDENT deep-copy of swin.stages[iso_stage:] (+ that stage's output
code/afd_model.py:128:        # iso_trunk_recce (the trunk-undersupervision FIX):
code/afd_model.py:131:        #     detach()ed -> the AIRL degradation-consistency gradient updates ONLY the
code/afd_model.py:133:        #     isolation invariant that keeps f_rec a specialised "recover expert" and
code/afd_model.py:134:        #     protects the clean trunk + f_full from being pulled toward degradation-
code/afd_model.py:135:        #     robustness.  Holds for BOTH settings of iso_trunk_recce.
code/afd_model.py:138:        #       - iso_trunk_recce=True  (default, the FIX): the fork feed is NOT
code/afd_model.py:139:        #         detached, so f_rec's CLEAN ID-CE gradient FLOWS BACK into the shared
code/afd_model.py:140:        #         trunk.  Diagnosis (codex consensus): the original full-detach iso cut
code/afd_model.py:141:        #         the trunk's extra identity supervision (f_rec's clean ID-CE only
code/afd_model.py:142:        #         updated the detached rec tail), leaving f_full WEAKER than even the
code/afd_model.py:145:        #         identity supervision -> strengthens f_full, while the degradation-
code/afd_model.py:146:        #         consistency stays detached (above) -> f_rec stays specialised.
code/afd_model.py:147:        #       - iso_trunk_recce=False (ablation): the clean fork feed is ALSO
code/afd_model.py:148:        #         detached -> the ORIGINAL full-isolation iso (clean ID-CE + consistency
code/afd_model.py:149:        #         both severed from the trunk).  Kept for the controlled comparison
code/afd_model.py:152:        # OFF (iso_branch=False) -> nothing is built and the forward is byte-for-byte
code/afd_model.py:154:        self.iso_branch = bool(iso_branch)
code/afd_model.py:155:        self.iso_stage = int(iso_stage)
code/afd_model.py:156:        # iso_trunk_recce: whether the CLEAN rec ID-CE gradient reflows into the
code/afd_model.py:158:        # (False, original full-isolation ablation).  No effect when iso_branch off.
code/afd_model.py:159:        self.iso_trunk_recce = bool(iso_trunk_recce)
code/afd_model.py:160:        if self.iso_branch:
code/afd_model.py:162:            if not (1 <= self.iso_stage <= n_stages - 1):
code/afd_model.py:164:                    f"iso_stage must be in [1, {n_stages - 1}] (fork after the "
code/afd_model.py:165:                    f"shared early stages, before the last); got {self.iso_stage}")
code/afd_model.py:166:            # the rec branch re-runs stages [iso_stage .. last] on its OWN copy.
code/afd_model.py:167:            # deep-copy preserves the pretrained weights as the f_rec init (same
code/afd_model.py:168:            # starting point as f_full's stages -> divergence comes from training,
code/afd_model.py:169:            # not from a random re-init that would cripple f_rec).
code/afd_model.py:171:                copy.deepcopy(self.swin.stages[i]) for i in range(self.iso_stage,
code/afd_model.py:174:            # an independent copy so f_rec gets its own LayerNorm (matches the
code/afd_model.py:175:            # f_full norm recipe; reshaped exactly like swin.forward does).
code/afd_model.py:182:            # rec branch runs stages [iso_stage..last] so it needs those indices.
code/afd_model.py:186:                    for i in range(self.iso_stage, n_stages))
code/afd_model.py:189:                    for i in range(self.iso_stage, n_stages))
code/afd_model.py:199:            # single layer4 hook (on the f_full map) is unaffected.
code/afd_model.py:203:        """Run the INDEPENDENT rec copy of stages [iso_stage..last] on the residual
code/afd_model.py:206:        The caller (_forward_swin_split) decides whether `x` is detached: the DEGRADED
code/afd_model.py:207:        consistency pass always passes a detached fork (gradient isolation), while the
code/afd_model.py:208:        CLEAN pass with iso_trunk_recce=True passes a NON-detached fork so the clean
code/afd_model.py:209:        f_rec ID-CE reflows into the shared trunk.  This method itself is agnostic to
code/afd_model.py:215:        rec map is computed the same way f_full's map is -- the ONLY differences are
code/afd_model.py:222:            i = self.iso_stage + j               # absolute stage index
code/afd_model.py:236:    def _forward_swin_split(self, x, rec_only=False):
code/afd_model.py:239:        Returns (f_full_map, f_rec_map).  The shared patch_embed + ALL f_full stages
code/afd_model.py:241:        training-time stochastic-depth / DropPath RNG sequence f_full sees is
code/afd_model.py:243:        not interleaved, so it cannot perturb f_full's RNG draws); the residual
code/afd_model.py:244:        stream at the input of stage `iso_stage` is captured and fed through the
code/afd_model.py:251:            (the isolation invariant, independent of iso_trunk_recce).
code/afd_model.py:253:            self.iso_trunk_recce is False (original full-isolation ablation).  When
code/afd_model.py:254:            iso_trunk_recce is True (the fix, default) the clean fork feed is NOT
code/afd_model.py:255:            detached, so f_rec's CLEAN ID-CE gradient reflows into the shared trunk
code/afd_model.py:256:            (extra identity supervision that strengthens f_full).  The
code/afd_model.py:260:        rec_only=True: skip the f_full BNNeck-side work entirely is done by the
code/afd_model.py:262:        degraded consistency pass avoid keeping the f_full map's grad graph -- we
code/afd_model.py:264:        f_full's last-stage norm/grad, so full_map is returned detached to make the
code/afd_model.py:265:        "f_full untouched by the degraded pass" contract explicit and cheap.
code/afd_model.py:268:        # build the per-sample semantic weight exactly like SwinTransformer.forward
code/afd_model.py:282:        #   * degraded consistency pass (rec_only) -> ALWAYS detach (isolation
code/afd_model.py:284:        #   * clean pass -> detach ONLY when iso_trunk_recce is False (original
code/afd_model.py:285:        #     full-isolation ablation); when True (the fix) keep the graph so the
code/afd_model.py:287:        detach_fork = bool(rec_only) or (not self.iso_trunk_recce)
code/afd_model.py:293:            # iso_stage, snapshot this stream (the gradient-isolation boundary) to
code/afd_model.py:297:            if i == self.iso_stage:
code/afd_model.py:314:            # the degraded consistency pass only needs f_rec; detach f_full's map so
code/afd_model.py:315:            # no f_full grad graph is built and the contract "the degraded pass does
code/afd_model.py:316:            # not train f_full" is explicit.  (running stats of self.bottleneck are
code/afd_model.py:318:            # BatchNorm forward on it; see AFDModel.forward rec_only path.)
code/afd_model.py:321:        # per detach_fork (degraded/ablation -> isolated; clean+fix -> grad reflows to
code/afd_model.py:324:        # multiply x*softplus(sw)+sb keeps x's graph).  Run AFTER the f_full loop so
code/afd_model.py:325:        # f_full's RNG is unchanged.
code/afd_model.py:332:        # Default path (return_rec=False OR iso off): SwinTransformer.forward ->
code/afd_model.py:336:        if not (self.iso_branch and return_rec):
code/afd_model.py:340:        # iso dual-branch path: run the split forward -> (f_full map, f_rec map).
code/afd_model.py:344:        # iso_trunk_recce=True (clean f_rec ID-CE reflows -> extra trunk supervision),
code/afd_model.py:345:        # else detached (full-isolation ablation).  See _forward_swin_split.
code/afd_model.py:346:        full_map, rec_map = self._forward_swin_split(x, rec_only=rec_only)
code/afd_model.py:387:        x = x.clamp(min=self.eps).pow(self.p)
code/afd_model.py:420:    Sum of the three equals x (up to fp error). Done per-sample via 2D FFT over (H,W).
code/afd_model.py:512:        """Per-sample: with prob high_drop_p, drop the high band (keep low+mid)."""
code/afd_model.py:528:class AFDModel(nn.Module):
code/afd_model.py:539:                 airl_dualbranch=False,
code/afd_model.py:540:                 airl_dualbranch_iso=False, airl_iso_stage=3,
code/afd_model.py:541:                 airl_iso_trunk_recce=True):
code/afd_model.py:545:        # AIRL dual-branch: a SECOND BNNeck head (bottleneck_rec + classifier_rec)
code/afd_model.py:546:        # over the SAME shared backbone feature map.  f_full (the original head)
code/afd_model.py:547:        # keeps full-resolution identity evidence (protects G->A); f_rec (this
code/afd_model.py:548:        # second head) additionally carries the AIRL ground-degradation
code/afd_model.py:551:        # heads' cosine scores are SOFT-fused at the distance-matrix level
code/afd_model.py:552:        # (cos = w*cos_rec + (1-w)*cos_full) -- ONE forward yields both features.
code/afd_model.py:555:        self.airl_dualbranch = bool(airl_dualbranch)
code/afd_model.py:556:        # AIRL gradient-isolated dual-branch: the SAME two-head + soft-fusion idea
code/afd_model.py:557:        # as airl_dualbranch, but f_rec is NOT a BNNeck over the shared global_feat;
code/afd_model.py:559:        # trunk feature (see SwinBackboneReID.iso_branch).  This severs the f_rec
code/afd_model.py:560:        # consistency gradient from the shared trunk so the clean trunk + f_full are
code/afd_model.py:563:        # the shared airl_dualbranch (same eval/loss contract, different f_rec path).
code/afd_model.py:564:        self.airl_dualbranch_iso = bool(airl_dualbranch_iso)
code/afd_model.py:565:        self.airl_iso_stage = int(airl_iso_stage)
code/afd_model.py:566:        # airl_iso_trunk_recce: route the CLEAN f_rec ID-CE gradient back into the
code/afd_model.py:568:        # original full-isolation iso where the clean fork feed is also detached
code/afd_model.py:569:        # (False). The degradation-consistency stays trunk-isolated either way.
code/afd_model.py:570:        self.airl_iso_trunk_recce = bool(airl_iso_trunk_recce)
code/afd_model.py:571:        if self.airl_dualbranch_iso:
code/afd_model.py:572:            assert not self.airl_dualbranch, (
code/afd_model.py:573:                "airl_dualbranch_iso and airl_dualbranch are mutually exclusive "
code/afd_model.py:574:                "(shared vs gradient-isolated f_rec; pick one).")
code/afd_model.py:576:                "airl_dualbranch_iso requires backbone='swin_small' (the rec branch "
code/afd_model.py:590:                resnet.layer4[0].downsample[0].stride = (1, 1)
code/afd_model.py:627:                iso_branch=self.airl_dualbranch_iso,
code/afd_model.py:628:                iso_stage=self.airl_iso_stage,
code/afd_model.py:629:                iso_trunk_recce=self.airl_iso_trunk_recce)
code/afd_model.py:635:            # clamp(min=eps) would destroy the negative half -> force avg pooling
code/afd_model.py:642:        # BNNeck (f_full -- the original head: full-resolution identity evidence)
code/afd_model.py:650:        # AIRL dual-branch: a SECOND independent BNNeck head (f_rec).  Same structure
code/afd_model.py:651:        # / init recipe as f_full (frozen-bias BNNeck + bias-free classifier), but its
code/afd_model.py:652:        # OWN parameters so the two heads can specialise (f_rec absorbs the
code/afd_model.py:653:        # degradation-consistency signal, f_full stays clean).
code/afd_model.py:654:        #   * airl_dualbranch     : f_rec pools the SHARED global_feat (fully shared
code/afd_model.py:656:        #   * airl_dualbranch_iso : f_rec pools the INDEPENDENT rec last-stage map
code/afd_model.py:657:        #                           (gradient-isolated trunk -> heads re-diverge).
code/afd_model.py:660:        if self.airl_dualbranch or self.airl_dualbranch_iso:
code/afd_model.py:718:        """rec map -> pooled rec feat -> BNNeck_rec feat (independent f_rec head).
code/afd_model.py:720:        Used only by the gradient-isolated dual-branch: the rec map already comes
code/afd_model.py:722:        here keeps the whole f_rec head isolated from the shared trunk.
code/afd_model.py:734:               When airl_dualbranch is on, the dict ALSO carries the f_rec head's
code/afd_model.py:737:               the f_rec ID-CE + degradation-consistency.
code/afd_model.py:738:        Eval : returns the L2-normalized f_full BN feature (single head); when
code/afd_model.py:739:               airl_dualbranch is on AND return_dual=True, returns the tuple
code/afd_model.py:740:               (f_full_norm, f_rec_norm) so the dual-branch eval can SOFT-fuse
code/afd_model.py:741:               the two cosine scores.  return_dual defaults to False, so the
code/afd_model.py:744:               airl_dualbranch_iso: identical (f_full_norm, f_rec_norm) eval tuple
code/afd_model.py:745:               and bn_feat_rec/logits_rec train keys, but f_rec is pooled from the
code/afd_model.py:746:               INDEPENDENT rec last-stage map (gradient-isolated trunk) instead of
code/afd_model.py:750:        # ---- AIRL gradient-isolated dual-branch path -------------------------- #
code/afd_model.py:751:        # When iso is on we need BOTH the f_full map and the independent rec map.
code/afd_model.py:752:        # The Swin split forward returns both in one pass; f_full pools the shared
code/afd_model.py:753:        # map (bn_feat/global_feat) and f_rec pools the rec map through bottleneck_rec.
code/afd_model.py:754:        # Fork-feed gradient regime (see _forward_swin_split): the DEGRADED pass
code/afd_model.py:756:        # reaches the trunk; the CLEAN pass detaches only when iso_trunk_recce is
code/afd_model.py:757:        # False -- with the fix (True) the clean f_rec ID-CE reflows into the trunk
code/afd_model.py:758:        # (extra identity supervision), while the consistency stays trunk-isolated.
code/afd_model.py:761:        # otherwise want_iso would be False and the rec_only dict request would
code/afd_model.py:762:        # silently fall through to the single f_full eval tensor.
code/afd_model.py:763:        want_iso = self.airl_dualbranch_iso and (self.training or return_dual
code/afd_model.py:765:        if want_iso:
code/afd_model.py:768:            # rec_only (the degraded consistency pass): compute ONLY the f_rec head.
code/afd_model.py:769:            # f_full's BNNeck is NOT run on the degraded images, so self.bottleneck's
code/afd_model.py:771:            # f_full eval head) -- f_full is a true clean expert -- and the f_full
code/afd_model.py:783:                # eval: ONE forward -> two L2-normalized features (f_full, f_rec).
code/afd_model.py:784:                return (F.normalize(bn_feat, dim=1),
code/afd_model.py:785:                        F.normalize(bn_feat_rec, dim=1))
code/afd_model.py:787:                'global_feat': global_feat,   # f_full triplet (shared trunk)
code/afd_model.py:791:                # f_rec head over the INDEPENDENT rec map (own ID-CE + AIRL
code/afd_model.py:806:            f_full = F.normalize(bn_feat, dim=1)
code/afd_model.py:807:            if return_dual and self.airl_dualbranch:
code/afd_model.py:809:                # two L2-normalized features for the distmat-level soft fusion.
code/afd_model.py:811:                return f_full, F.normalize(bn_feat_rec, dim=1)
code/afd_model.py:812:            return f_full
code/afd_model.py:821:        if self.airl_dualbranch:
code/afd_model.py:822:            # f_rec head: its OWN BNNeck + classifier on the shared global_feat.
code/afd_model.py:823:            # The train loop applies f_rec ID-CE (so f_rec is a valid identity
code/afd_model.py:824:            # space) PLUS the AIRL degradation-consistency (so it is robust to a
code/afd_model.py:826:            # NOT duplicated for f_rec (single backbone-level triplet).
code/afd_model.py:877:    return AFDModel(
code/afd_model.py:894:        airl_dualbranch=getattr(args, 'airl_dualbranch', False),
code/afd_model.py:895:        airl_dualbranch_iso=getattr(args, 'airl_dualbranch_iso', False),
code/afd_model.py:896:        airl_iso_stage=getattr(args, 'airl_iso_stage', 3),
code/afd_model.py:897:        airl_iso_trunk_recce=getattr(args, 'airl_iso_trunk_recce', True),
code/afd_train.py:19:            sample-to-sample, in-batch, opposite-view *retrieval* loss whose
code/afd_train.py:20:            score is a hybrid of (a) global cosine and (b) a symmetric
code/afd_train.py:37:                delta_ij = cos(z_i, P[y_j, view_j]) - cos(z_i, P[y_i, view_j]),
code/afd_train.py:38:            map it to a weight w_ij = clamp(1 - gamma*sigmoid((delta_ij-margin)/
code/afd_train.py:57:  samples in the OPPOSITE view (1-v) and the negatives are the opposite-view
code/afd_train.py:58:  samples of OTHER pids.  Same-view samples are excluded as candidates entirely
code/afd_train.py:60:      score(i,j) = alpha * cos(g_i, g_j)
code/afd_train.py:82:  ramp lambda over the first N epochs so the (randomly-initialised) projection
code/afd_train.py:94:    PK sampler P=16 x K=4 (bs=64), AdamW lr 3.5e-4, 10-ep warmup + cosine, 60 ep.
code/afd_train.py:99:    L2-normalized BNNeck feature in a register_buffer of shape
code/afd_train.py:102:      2) add an InfoNCE loss pulling each sample toward its OWN pid's
code/afd_train.py:104:         prototypes:  CE( cos(z, P[:, opp_view]) / tau ,  y ).
code/afd_train.py:107:    Cold-start handling: a sample only contributes to the OVP loss once its own
code/afd_train.py:151:                           RandomIdentitySampler, filter_by_view)
code/afd_train.py:169:    Prototypes are L2-normalized; features used for update/loss are the
code/afd_train.py:170:    L2-normalized BNNeck features (detached for the EMA update).
code/afd_train.py:188:        re-normalize the prototype.
code/afd_train.py:201:                gmean = F.normalize(gmean, dim=0)
code/afd_train.py:208:                    self.bank[pid_i, v] = F.normalize(self.bank[pid_i, v], dim=0)
code/afd_train.py:211:        """InfoNCE: each sample classified against ALL pids' opposite-view
code/afd_train.py:215:        Returns a scalar loss (0 if no sample has an initialized own-opp proto).
code/afd_train.py:218:        opp = 1 - views                                   # opposite view per sample
code/afd_train.py:222:            # samples whose opposite view == v_opp (i.e. samples from view 1-v_opp)
code/afd_train.py:232:            # only keep samples whose OWN opposite-view proto exists
code/afd_train.py:240:            logits = logits.masked_fill(~valid.view(1, -1), float('-inf'))
code/afd_train.py:255:    EXACTLY to <mean_k tok_i, mean_k tok_j> -- the gram of the per-sample MEAN of
code/afd_train.py:262:    random aggregated vector is meaningless and drags the cross-view cosine down.
code/afd_train.py:271:    residual.  `mean_k(tok)` is the UN-normalized mean of the K L2-normed tokens
code/afd_train.py:280:    Each `*_residual()` returns a per-sample (B,out_dim) residual that is itself
code/afd_train.py:286:    autocast(enabled=False)); every reduction is NaN-safe.
code/afd_train.py:289:                    aggregation -> intra-normalized (C x D) VLAD -> linear to
code/afd_train.py:299:                    normalize, flatten -> linear (token second-order statistics).
code/afd_train.py:371:            vlad = F.normalize(vlad, dim=2)                       # intra-norm /cluster
code/afd_train.py:373:            vlad = F.normalize(vlad, dim=1)                       # global VLAD L2-norm
code/afd_train.py:385:            den = g.sum(dim=1).clamp(min=self.eps)                # (B,1) >0 -> NaN-safe
code/afd_train.py:397:        gate zero-init, so at step 0 the output == the un-normalized mean of the K
code/afd_train.py:399:        L2-norm), so its gram is the UN-normalized `mean @ mean.T` == the 52.37
code/afd_train.py:401:        pooling REPLACES the mean (random-init), and aggregate_tokens L2-normalizes
code/afd_train.py:402:        it (cosine gram) -> the collapsing control.  Both branches are
code/afd_train.py:425:    There is NO prototype / memory / EMA: the loss is a pure sample-to-sample,
code/afd_train.py:439:        self.alpha = float(alpha)             # weight on global cosine in score
code/afd_train.py:456:        #     candidate set is restricted to opposite-view samples (the headline
code/afd_train.py:459:        #     candidates = ALL other samples (incl. same view); positives =
code/afd_train.py:461:        #     A plain all-view token-set supervised-contrastive control isolating
code/afd_train.py:504:        # Set-pooling of the K tokens into a single per-sample vector (the
code/afd_train.py:509:        #     gram of the per-sample MEAN of the K tokens (the current best mech).
code/afd_train.py:513:        #     the gram of those aggregated vectors (residual mode: UN-normalized
code/afd_train.py:515:        #     L2-normed cosine gram).  The match / pool / align / topk / thresh
code/afd_train.py:561:        projection runs in fp32 (numerical safety for the cos/MaxSim/logsumexp
code/afd_train.py:562:        downstream); tokens are L2-normalized per token.
code/afd_train.py:569:        # (the cached map may be fp16 under autocast).
code/afd_train.py:574:        tok = F.normalize(tok, dim=2)                               # per-token L2
code/afd_train.py:587:        The 52.37 avg/mean path computes its cross-view gram as the UN-normalized
code/afd_train.py:588:        `mean_k(tok) @ mean_k(tok).T` (the per-sample mean of the K L2-normed
code/afd_train.py:589:        tokens is NOT re-normalized -- see --ovli_match avg --ovli_pool mean, which
code/afd_train.py:595:            UN-normalized mean gram == the 52.37 avg/mean-pool gram BYTE-FOR-BYTE
code/afd_train.py:597:            here would instead give `<normalize(mean), normalize(mean)>` (diag==1),
code/afd_train.py:602:            REPLACES the mean, so it IS L2-normalized -> the cross-view gram is a
code/afd_train.py:603:            cosine in [-1,1] (the original standalone convention; this branch is
code/afd_train.py:608:            # raw mean(+residual): UN-normalized gram == 52.37 avg/mean-pool path.
code/afd_train.py:610:        return F.normalize(a, dim=1)           # standalone: unit vectors -> cosine gram
code/afd_train.py:644:            mean_above = summed / cnt.clamp(min=1).to(tmax.dtype)
code/afd_train.py:670:                                    finite floor (never -inf) so an empty row (can
code/afd_train.py:673:                                    clamps the per-row count to >=1.
code/afd_train.py:689:        cnt = mask4.sum(dim=other_dim).clamp(min=1).to(sim.dtype)
code/afd_train.py:705:        order-invariant aggregation: msim = gram of the per-sample aggregated
code/afd_train.py:707:        UN-normalized `mean(+residual) @ .T`, so at gate_res==0 it is byte-equal to
code/afd_train.py:709:        the L2-normed cosine gram.  setpool == 'mean' (default) leaves the original
code/afd_train.py:715:            return a @ a.t()                       # (B,B) cosine gram, symmetric
code/afd_train.py:760:        delta_ij = cos(z_i, P[y_j, view_j]) - cos(z_i, P[y_i, view_j])
code/afd_train.py:763:                    negative; z_i, P[.] L2-normed so cos == dot).
code/afd_train.py:764:        w_ij     = clamp(1 - gamma * sigmoid((delta_ij - margin)/eta), wmin, 1.0)
code/afd_train.py:765:        bias_ij  = log(w_ij)   (w_ij >= wmin > 0 -> finite, never -inf/NaN).
code/afd_train.py:768:        # work in fp32 for the cos/sigmoid/clamp/log numerics (gfeat may be fp32
code/afd_train.py:784:        cos_self = (zb * proto_self).sum(-1)                       # (B,B) cos(z_i,P[y_i,vj])
code/afd_train.py:785:        cos_neg = (zb * proto_neg).sum(-1)                        # (B,B) cos(z_i,P[y_j,vj])
code/afd_train.py:786:        delta = cos_neg - cos_self                                 # (B,B) ambiguity
code/afd_train.py:788:        w = torch.clamp(1.0 - gamma * s, min=wmin, max=1.0)        # (B,B) in [wmin,1]
code/afd_train.py:790:        # initialised (never-seen prototype is a zero vector -> meaningless cos);
code/afd_train.py:823:            samples (view != view_i); positives = opposite-view same-pid samples.
code/afd_train.py:825:            samples (any view, excl. self); positives = same-pid any view.
code/afd_train.py:842:        # pairwise hybrid score in fp32 (cos in [-1,1], maxsim in [-1,1])
code/afd_train.py:854:        #   allview=True  (control): all-view -> candidates = every other sample.
code/afd_train.py:874:        # Use a large finite floor (not -inf) so a fully-masked row can never
code/afd_train.py:875:        # yield -inf, and so (-inf)-(-inf)=nan can never appear even before the
code/afd_train.py:918:    the OVLI hook and reranks by score = alpha*cos(global) + (1-alpha)*MaxSim.
code/afd_train.py:927:    def extract(samples):
code/afd_train.py:929:        loader = _bel(samples, args)
code/afd_train.py:960:        # used by the train loss -> residual mode = UN-normalized mean(+residual)
code/afd_train.py:961:        # gram == 52.37 avg/mean path at gate_res==0; standalone = cosine gram),
code/afd_train.py:962:        # NOT the token-set MaxSim.  Aggregate query/gallery tokens in sample-row
code/afd_train.py:1009:            results[tag] = {'global': (float('nan'), float('nan')),
code/afd_train.py:1010:                            'rerank': (float('nan'), float('nan'))}
code/afd_train.py:1012:        qf = F.normalize(qf, dim=1)
code/afd_train.py:1013:        gf = F.normalize(gf, dim=1)
code/afd_train.py:1014:        gsim = (qf @ gf.t()).numpy()                      # (Nq,Ng) cosine
code/afd_train.py:1015:        # global-only (rank by cosine distance == -gsim)
code/afd_train.py:1017:        # rerank: alpha*cos + (1-alpha)*MaxSim, rank by descending hybrid
code/afd_train.py:1029:def airl_dualbranch_eval(model, dataset, args, device):
code/afd_train.py:1030:    """AIRL dual-branch eval: extract BOTH heads (f_full, f_rec) in ONE forward
code/afd_train.py:1031:    and report f_full-only, f_rec-only, and the SOFT-FUSED cosine ranking
code/afd_train.py:1032:    (cos = w*cos_rec + (1-w)*cos_full, w = args.airl_fuse_w, fixed) for A->G and
code/afd_train.py:1034:    fusion: cos_rec replaces the AIRL-model cosine, cos_full replaces the
code/afd_train.py:1035:    baseline-model cosine, and they share ONE backbone forward.
code/afd_train.py:1039:    f_full number reproduces run_cross_view_eval's A<->G mAP bit-for-bit (same
code/afd_train.py:1041:    Returns {tag: {'full': (mAP,R1), 'rec': (mAP,R1), 'fuse': (mAP,R1)}}.
code/afd_train.py:1051:    def extract(samples):
code/afd_train.py:1052:        loader = _bel(samples, args)
code/afd_train.py:1058:            # ONE forward -> two L2-normalized features (f_full, f_rec).
code/afd_train.py:1059:            f_full, f_rec = model(imgs, view_idx=vidx, return_dual=True)
code/afd_train.py:1060:            ffs.append(f_full.cpu())
code/afd_train.py:1061:            frs.append(f_rec.cpu())
code/afd_train.py:1071:    w = args.airl_fuse_w
code/afd_train.py:1081:            nan2 = (float('nan'), float('nan'))
code/afd_train.py:1082:            results[tag] = {'full': nan2, 'rec': nan2, 'fuse': nan2}
code/afd_train.py:1084:        # features are already L2-normalized at eval; renormalize defensively so
code/afd_train.py:1085:        # the cosine == the gram of unit vectors (matches eval_market exactly).
code/afd_train.py:1086:        q_full = F.normalize(q_full, dim=1); g_full = F.normalize(g_full, dim=1)
code/afd_train.py:1087:        q_rec = F.normalize(q_rec, dim=1);   g_rec = F.normalize(g_rec, dim=1)
code/afd_train.py:1088:        s_full = (q_full @ g_full.t()).numpy()        # (Nq,Ng) cosine, f_full
code/afd_train.py:1089:        s_rec = (q_rec @ g_rec.t()).numpy()           # (Nq,Ng) cosine, f_rec
code/afd_train.py:1090:        # soft fusion: cos = w*cos_rec + (1-w)*cos_full -> distance = 2 - 2*cos
code/afd_train.py:1091:        # (identical to kill-switch #3 GATE 5; cosine in [-1,1] -> dist in [0,4]).
code/afd_train.py:1094:        dm_fuse = (2.0 - 2.0 * (w * s_rec + (1.0 - w) * s_full))
code/afd_train.py:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
code/afd_train.py:1099:                        'fuse': (zmap, zr1)}
code/afd_train.py:1104:# AIRL: Aerial Identity Recoverability Learning (resolution-degradation
code/afd_train.py:1107:# Motivation (new_angle_AIRL.md, kill-switch #1 PASS): on CARGO the aerial->ground
code/afd_train.py:1114:# AIRL turns that diagnostic into a training signal WITHOUT any cross-view
code/afd_train.py:1117:#   1. For each GROUND image (high-res) sample a "pixel budget" = a scale ratio
code/afd_train.py:1121:#      "if this person were shot from a UAV, how much information would survive".
code/afd_train.py:1125:#      (KL on logits, or cosine/MSE on the L2-normed BNNeck feature).  Intuition:
code/afd_train.py:1128:#   3. total = CE + triplet + airl_lambda_eff * consistency.
code/afd_train.py:1133:#   * --airl OFF (default) => NO degradation, NO extra forward, NO loss term =>
code/afd_train.py:1134:#     the baseline is reproduced BYTE-FOR-BYTE (the whole AIRL block is skipped).
code/afd_train.py:1135:#   * The consistency loss runs in TRUE fp32 (autocast disabled) for KL/cosine
code/afd_train.py:1137:#   * AIRL is a TRAIN-time loss only; eval is unchanged (train/test symmetric).
code/afd_train.py:1140:def airl_degrade(imgs, min_scale, blur=False, generator=None):
code/afd_train.py:1141:    """Resolution-degrade a NORMALIZED image batch to a sampled pixel budget.
code/afd_train.py:1144:    dataloader; degradation is a linear resample in normalized space, which is a
code/afd_train.py:1150:    degradation).  The image is bilinearly DOWN-sampled to (round(s*H), round(s*W))
code/afd_train.py:1151:    (>=1 px) then bilinearly UP-sampled back to (H, W), so the output keeps the
code/afd_train.py:1153:    light 3x3 avg-pool blur (stride 1, reflect pad) is applied AFTER the up-sample
code/afd_train.py:1159:    so the resample is done one image at a time (B is small, bs<=64).
code/afd_train.py:1191:def airl_consistency_loss(logits_o, bn_o, logits_d, bn_d, mode='kl', tau=4.0):
code/afd_train.py:1204:      mode='feat': 1 - cos(bn_o.detach, bn_d) on the L2-normed BNNeck feature
code/afd_train.py:1207:    Returns a finite non-negative scalar.  A light finite guard (nan_to_num) is
code/afd_train.py:1210:    pathological forward ever produced inf/NaN logits.
code/afd_train.py:1213:        zo = F.normalize(bn_o.float(), dim=1).detach()
code/afd_train.py:1214:        zd = F.normalize(bn_d.float(), dim=1)
code/afd_train.py:1215:        # 1 - cosine in [0,2]; mean over batch.  (== 0.5*||zo-zd||^2 on unit vecs.)
code/afd_train.py:1217:        return torch.nan_to_num(out)
code/afd_train.py:1227:    return torch.nan_to_num(kl * (tau * tau))
code/afd_train.py:1257:    ap.add_argument('--no_amp', action='store_true', help='disable mixed precision')
code/afd_train.py:1315:                    help='score = alpha*cos(global) + (1-alpha)*sym_MaxSim(tokens)')
code/afd_train.py:1334:                    help='k for --ovli_pool topk (clamped to [1, K] tokens)')
code/afd_train.py:1340:                         'the OVLI loss -> candidates become ALL other samples '
code/afd_train.py:1371:    # projected tokens into one per-sample vector (the cross-view score is then
code/afd_train.py:1372:    # the cosine gram of those vectors).  This REPLACES the fixed "mean over
code/afd_train.py:1424:                    help='ACVP max softening strength: w_ij = clamp(1 - gamma*'
code/afd_train.py:1435:                    help='ACVP linear gamma warmup over this many epochs (ramp 0 -> '
code/afd_train.py:1437:    # --- AIRL (Aerial Identity Recoverability Learning) -- resolution-degradation
code/afd_train.py:1440:    # consistency = loss); the optimizer is untouched.  See airl_degrade /
code/afd_train.py:1441:    # airl_consistency_loss above.  Independent of OVP/OVLI/ACVP (can co-run, but
code/afd_train.py:1442:    # the headline AIRL run is --airl alone on the plain baseline).
code/afd_train.py:1443:    ap.add_argument('--airl', action='store_true',
code/afd_train.py:1444:                    help='enable AIRL: per-image resolution degradation (to a '
code/afd_train.py:1445:                         'sampled aerial-scale pixel budget) + original/degraded '
code/afd_train.py:1449:    ap.add_argument('--airl_lambda', type=float, default=0.5,
code/afd_train.py:1450:                    help='weight of the AIRL consistency loss '
code/afd_train.py:1451:                         '(total = CE + triplet + airl_lambda_eff * consistency).')
code/afd_train.py:1452:    ap.add_argument('--airl_min_scale', type=float, default=0.25,
code/afd_train.py:1454:                         ' 1]); s*H x s*W is the down-sampled pixel budget before '
code/afd_train.py:1455:                         'up-sampling back. 0.25 ~ the aerial small bucket (aerial '
code/afd_train.py:1457:    ap.add_argument('--airl_consistency', default='kl', choices=['kl', 'feat'],
code/afd_train.py:1460:                         "1 - cosine on the L2-normed BNNeck feature.")
code/afd_train.py:1461:    ap.add_argument('--airl_tau', type=float, default=4.0,
code/afd_train.py:1462:                    help='softmax temperature for --airl_consistency kl (Hinton '
code/afd_train.py:1464:    ap.add_argument('--airl_blur', action='store_true',
code/afd_train.py:1466:                         'up-sample (UAV optical-blur proxy; no extra params).')
code/afd_train.py:1467:    ap.add_argument('--airl_warmup', type=int, default=5,
code/afd_train.py:1468:                    help='linear AIRL lambda warmup over this many epochs (ramp 0 -> '
code/afd_train.py:1469:                         'airl_lambda) so the consistency term opens gently.')
code/afd_train.py:1470:    # --- AIRL dual-branch (resolvability branch): the COMPLETE AIRL mechanism for
code/afd_train.py:1472:    # (f_rec) on the shared backbone: f_full keeps full-resolution identity
code/afd_train.py:1473:    # evidence (protects G->A), f_rec gets its own ID-CE PLUS the AIRL
code/afd_train.py:1475:    # evidence, serves A->G).  At eval the two heads' cosine scores are
code/afd_train.py:1476:    # SOFT-fused at the distance-matrix level:
code/afd_train.py:1477:    #     cos = airl_fuse_w * cos(f_rec) + (1 - airl_fuse_w) * cos(f_full)
code/afd_train.py:1483:    # a clean (f_full) and a recover (f_rec) evidence head DIVERGE, combined by a
code/afd_train.py:1490:    ap.add_argument('--airl_dualbranch', action='store_true',
code/afd_train.py:1491:                    help='enable the AIRL dual-branch (resolvability branch): a '
code/afd_train.py:1492:                         'second BNNeck head f_rec (own ID-CE + AIRL degradation '
code/afd_train.py:1493:                         'consistency) alongside the clean f_full head, soft-fused '
code/afd_train.py:1494:                         'at eval (cos = w*cos_rec + (1-w)*cos_full). One forward, '
code/afd_train.py:1496:    ap.add_argument('--airl_fuse_w', type=float, default=0.25,
code/afd_train.py:1497:                    help='fixed global fusion weight on the f_rec cosine at eval '
code/afd_train.py:1498:                         '(cos = airl_fuse_w*cos_rec + (1-airl_fuse_w)*cos_full); '
code/afd_train.py:1504:    ap.add_argument('--airl_dualbranch_iso', action='store_true',
code/afd_train.py:1505:                    help='gradient-ISOLATED AIRL dual-branch (rescue of the failed '
code/afd_train.py:1506:                         'fully-shared --airl_dualbranch): f_rec is a BNNeck over an '
code/afd_train.py:1508:                         'iso_stage (not the shared global_feat). The degradation-'
code/afd_train.py:1511:                         'degraded pass forks off a DETACHED trunk), so f_rec stays a '
code/afd_train.py:1513:                         'toward degradation-robustness) is avoided. The CLEAN f_rec '
code/afd_train.py:1514:                         'ID-CE routing is governed by --airl_iso_trunk_recce: default '
code/afd_train.py:1516:                         'supervision -> strengthens the otherwise-weak f_full); 0 = '
code/afd_train.py:1517:                         'original full-isolation (clean ID-CE also detached). '
code/afd_train.py:1519:                         'contract as --airl_dualbranch (shares its AIRL hyperparams '
code/afd_train.py:1520:                         '+ --airl_fuse_w). Default OFF reproduces the baseline.')
code/afd_train.py:1521:    ap.add_argument('--airl_iso_stage', type=int, default=3,
code/afd_train.py:1522:                    help='Swin stage index the f_rec branch forks AFTER (the rec '
code/afd_train.py:1523:                         'branch re-runs stages [iso_stage..last] on its own deep-'
code/afd_train.py:1526:                         'iso_stage=3 (default) = share stages 0-2, split ONLY the '
code/afd_train.py:1527:                         'last stage (MGN-style, cheapest); iso_stage=2 = split the '
code/afd_train.py:1528:                         'last two stages (more f_rec divergence capacity, heavier). '
code/afd_train.py:1529:                         'Must be in [1,3]. Only used with --airl_dualbranch_iso.')
code/afd_train.py:1531:    # iso left f_full WEAK (ep20 45.56 < baseline 48.98 < even fully-shared f_rec
code/afd_train.py:1532:    # 47.39): f_rec's clean ID-CE only updated the DETACHED rec tail, so the shared
code/afd_train.py:1534:    # got from BOTH heads' ID-CE.  --airl_iso_trunk_recce 1 (default) re-routes ONLY
code/afd_train.py:1536:    # supervision -> strengthens f_full) while keeping the degradation-CONSISTENCY
code/afd_train.py:1537:    # gradient detached from the trunk (so f_rec stays a specialised recover pole --
code/afd_train.py:1538:    # the isolation that the iso variant exists for).  0 = the ORIGINAL full-isolation
code/afd_train.py:1539:    # iso (clean ID-CE also detached), kept for the controlled ablation.  Only used
code/afd_train.py:1540:    # with --airl_dualbranch_iso.
code/afd_train.py:1541:    ap.add_argument('--airl_iso_trunk_recce', type=int, default=1, choices=[0, 1],
code/afd_train.py:1542:                    help='1 (default, the FIX): route the CLEAN f_rec ID-CE gradient '
code/afd_train.py:1544:                         'strengthens the weak f_full); the degradation-consistency '
code/afd_train.py:1545:                         'gradient stays DETACHED from the trunk (f_rec stays '
code/afd_train.py:1546:                         'specialised). 0: original full-isolation iso (clean ID-CE '
code/afd_train.py:1548:                         'without --airl_dualbranch_iso.')
code/afd_train.py:1554:    args.airl_iso_trunk_recce = bool(args.airl_iso_trunk_recce)
code/afd_train.py:1557:    # --backbone swin_small (caught in AFDModel too, but fail fast at parse time).
code/afd_train.py:1563:    # InfoNCE vs sample-to-sample late-interaction retrieval).  All three modes
code/afd_train.py:1587:    # produce inf/NaN.  Enforce wmin in (0,1], eta>0, gamma>=0 at parse time so a
code/afd_train.py:1600:    # AIRL numeric-safety guard: min_scale in (0,1] so the down-sampled budget is a
code/afd_train.py:1602:    if args.airl and not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
code/afd_train.py:1603:        ap.error("--airl_min_scale must be in (0,1] (per-image scale ratio s in "
code/afd_train.py:1604:                 f"[min_scale,1]); got {args.airl_min_scale}.")
code/afd_train.py:1605:    if args.airl and not (args.airl_tau > 0.0):
code/afd_train.py:1606:        ap.error(f"--airl_tau must be > 0 (softmax temperature); got {args.airl_tau}.")
code/afd_train.py:1608:    # AIRL dual-branch guards.  --airl (single-head consistency) and
code/afd_train.py:1609:    # --airl_dualbranch (two-head, consistency on f_rec only) are two DIFFERENT
code/afd_train.py:1610:    # AIRL instantiations of the SAME degrade+consistency primitive; running both
code/afd_train.py:1611:    # would apply consistency twice (to the single head AND to f_rec) and muddy
code/afd_train.py:1613:    # --airl_lambda / --airl_min_scale / --airl_consistency / --airl_tau /
code/afd_train.py:1614:    # --airl_blur / --airl_warmup (the consistency on f_rec is the SAME function),
code/afd_train.py:1615:    # and they are validated the same way (so a stray bad --airl_min_scale with
code/afd_train.py:1616:    # only --airl_dualbranch still fails fast).
code/afd_train.py:1617:    if args.airl_dualbranch:
code/afd_train.py:1618:        if args.airl:
code/afd_train.py:1619:            ap.error("--airl_dualbranch and --airl are mutually exclusive (both "
code/afd_train.py:1620:                     "apply the AIRL degradation-consistency; dual-branch applies "
code/afd_train.py:1621:                     "it to the f_rec head only). Pick one.")
code/afd_train.py:1622:        if not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
code/afd_train.py:1623:            ap.error("--airl_min_scale must be in (0,1] (used by --airl_dualbranch "
code/afd_train.py:1624:                     f"too); got {args.airl_min_scale}.")
code/afd_train.py:1625:        if not (args.airl_tau > 0.0):
code/afd_train.py:1626:            ap.error("--airl_tau must be > 0 (used by --airl_dualbranch too); got "
code/afd_train.py:1627:                     f"{args.airl_tau}.")
code/afd_train.py:1628:        if not (0.0 <= args.airl_fuse_w <= 1.0):
code/afd_train.py:1629:            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
code/afd_train.py:1630:                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
code/afd_train.py:1636:        if args.airl_fuse_w != 0.25:
code/afd_train.py:1637:            print(f"[AIRL-DUAL][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
code/afd_train.py:1640:        # The dual-branch is the standalone headline AIRL mechanism; keep its
code/afd_train.py:1642:        # (they target a different gap and would confound the f_rec specialisation).
code/afd_train.py:1644:            ap.error("--airl_dualbranch is run standalone (headline AIRL); do not "
code/afd_train.py:1647:    # AIRL gradient-isolated dual-branch guards.  This is the RESCUE variant: the
code/afd_train.py:1648:    # SAME degrade+consistency+soft-fusion contract as --airl_dualbranch, but f_rec
code/afd_train.py:1651:    #   * shares the AIRL hyperparams (--airl_lambda/min_scale/consistency/tau/blur/
code/afd_train.py:1652:    #     warmup) and --airl_fuse_w, validated identically;
code/afd_train.py:1653:    #   * is mutually exclusive with BOTH --airl (single-head) and --airl_dualbranch
code/afd_train.py:1654:    #     (fully-shared) -- three distinct AIRL instantiations, one at a time;
code/afd_train.py:1656:    #   * runs standalone (no OVP/OVLI), same as --airl_dualbranch.
code/afd_train.py:1657:    if args.airl_dualbranch_iso:
code/afd_train.py:1658:        if args.airl or args.airl_dualbranch:
code/afd_train.py:1659:            ap.error("--airl_dualbranch_iso is mutually exclusive with --airl and "
code/afd_train.py:1660:                     "--airl_dualbranch (three distinct AIRL instantiations; the "
code/afd_train.py:1661:                     "iso variant forks an independent late stage off a detached "
code/afd_train.py:1664:            ap.error("--airl_dualbranch_iso requires --backbone swin_small (the rec "
code/afd_train.py:1666:        if not (1 <= args.airl_iso_stage <= 3):
code/afd_train.py:1667:            ap.error("--airl_iso_stage must be in [1,3] (swin_small has 4 stages "
code/afd_train.py:1669:                     f"{args.airl_iso_stage}.")
code/afd_train.py:1670:        if not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
code/afd_train.py:1671:            ap.error("--airl_min_scale must be in (0,1] (used by "
code/afd_train.py:1672:                     f"--airl_dualbranch_iso too); got {args.airl_min_scale}.")
code/afd_train.py:1673:        if not (args.airl_tau > 0.0):
code/afd_train.py:1674:            ap.error("--airl_tau must be > 0 (used by --airl_dualbranch_iso too); "
code/afd_train.py:1675:                     f"got {args.airl_tau}.")
code/afd_train.py:1676:        if not (0.0 <= args.airl_fuse_w <= 1.0):
code/afd_train.py:1677:            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
code/afd_train.py:1678:                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
code/afd_train.py:1679:        if args.airl_fuse_w != 0.25:
code/afd_train.py:1680:            print(f"[AIRL-ISO][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
code/afd_train.py:1684:            ap.error("--airl_dualbranch_iso is run standalone (headline AIRL); do "
code/afd_train.py:1716:    print(f"  airl={args.airl} (lambda={args.airl_lambda} "
code/afd_train.py:1717:          f"min_scale={args.airl_min_scale} consistency={args.airl_consistency} "
code/afd_train.py:1718:          f"tau={args.airl_tau} blur={args.airl_blur} warmup={args.airl_warmup}) "
code/afd_train.py:1721:    print(f"  airl_dualbranch={args.airl_dualbranch} (fuse_w={args.airl_fuse_w} "
code/afd_train.py:1722:          f"lambda={args.airl_lambda} min_scale={args.airl_min_scale} "
code/afd_train.py:1723:          f"consistency={args.airl_consistency} tau={args.airl_tau} "
code/afd_train.py:1724:          f"blur={args.airl_blur} warmup={args.airl_warmup}) "
code/afd_train.py:1725:          f"[resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL "
code/afd_train.py:1726:          f"consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; "
code/afd_train.py:1728:    print(f"  airl_dualbranch_iso={args.airl_dualbranch_iso} "
code/afd_train.py:1729:          f"(iso_stage={args.airl_iso_stage} trunk_recce={args.airl_iso_trunk_recce} "
code/afd_train.py:1730:          f"fuse_w={args.airl_fuse_w} "
code/afd_train.py:1731:          f"lambda={args.airl_lambda} min_scale={args.airl_min_scale} "
code/afd_train.py:1732:          f"consistency={args.airl_consistency} tau={args.airl_tau} "
code/afd_train.py:1733:          f"blur={args.airl_blur} warmup={args.airl_warmup}) "
code/afd_train.py:1734:          f"[GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late "
code/afd_train.py:1737:          f"CLEAN f_rec ID-CE into the trunk (extra identity supervision -> "
code/afd_train.py:1738:          f"strengthens f_full); trunk_recce=0 = original full-isolation (clean "
code/afd_train.py:1739:          f"ID-CE also detached). f_rec learns the recover pole; same soft-fusion "
code/afd_train.py:1742:          f"epochs={args.epochs} warmup={args.warmup_epochs} amp={not args.no_amp}")
code/afd_train.py:1752:        # and their mean with no change to the eval / AIRL-iso code.
code/afd_train.py:1758:    sampler = RandomIdentitySampler(dataset.train, batch_size, args.K)
code/afd_train.py:1759:    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
code/afd_train.py:1813:    # a constant-output fixed point (last-stage map off-diag cos +0.99 -> all
code/afd_train.py:1906:    # AIRL dual-branch self-check: the SECOND BNNeck head (bottleneck_rec +
code/afd_train.py:1909:    # f_rec must actually train, with its OWN params, at the FULL head LR even on
code/afd_train.py:1911:    if args.airl_dualbranch:
code/afd_train.py:1915:        # bottleneck_rec.bias has requires_grad_=False (frozen, like f_full's BN
code/afd_train.py:1920:        assert rec_in, "AIRL dual-branch f_rec head params NOT in optimizer!"
code/afd_train.py:1921:        # on Swin, f_rec must be at the FULL head LR (not the backbone factor):
code/afd_train.py:1922:        # both rec params are random-init heads, identical to f_full's BNNeck.
code/afd_train.py:1924:        print(f"  [AIRL-DUAL] f_rec head (bottleneck_rec + classifier_rec) params "
code/afd_train.py:1926:              f"trainable tensors); eval soft-fusion cos=w*cos_rec+(1-w)*cos_full "
code/afd_train.py:1927:              f"w={args.airl_fuse_w}")
code/afd_train.py:1928:    # AIRL gradient-isolated dual-branch self-check: BNNeck_rec + classifier_rec are
code/afd_train.py:1931:    # LR group (pretrained weights, same as f_full's stages).  Assert both placements
code/afd_train.py:1933:    if args.airl_dualbranch_iso:
code/afd_train.py:1940:        assert rec_head_in, "AIRL-ISO f_rec head params NOT in optimizer!"
code/afd_train.py:1947:        assert rec_stage_in, "AIRL-ISO rec late-stage params NOT in optimizer!"
code/afd_train.py:1949:        # backbone weight, byte-identical recipe to f_full's stages).  Find which
code/afd_train.py:1957:            assert rec_stage_in_swin, ("AIRL-ISO rec late stage NOT in the scaled "
code/afd_train.py:1960:            assert rec_head_in_full, ("AIRL-ISO rec BNNeck head NOT in the full-LR "
code/afd_train.py:1968:        recce_msg = ("trunk_recce=1 (clean f_rec ID-CE REFLOWS to trunk; degraded "
code/afd_train.py:1969:                     "consistency stays detached)" if args.airl_iso_trunk_recce
code/afd_train.py:1971:                          "original full-isolation)")
code/afd_train.py:1972:        print(f"  [AIRL-ISO] iso_stage={args.airl_iso_stage}: rec late stage "
code/afd_train.py:1975:              f"[{lr_msg}]; degradation-consistency grad isolated from shared trunk "
code/afd_train.py:1976:              f"(detached degraded pass at stage-{args.airl_iso_stage} input); "
code/afd_train.py:1978:              f"cos=w*cos_rec+(1-w)*cos_full w={args.airl_fuse_w}")
code/afd_train.py:1980:    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)
code/afd_train.py:1996:        # ACVP: linear gamma warmup (ramp 0 -> acvp_gamma over acvp_warmup epochs)
code/afd_train.py:1999:        # AIRL: linear lambda warmup (ramp 0 -> airl_lambda over airl_warmup epochs)
code/afd_train.py:2000:        # so the resolution-consistency term opens gently.  Shared by ALL THREE AIRL
code/afd_train.py:2001:        # instantiations (mutually exclusive): the single-head --airl, the fully-
code/afd_train.py:2002:        # shared dual-branch --airl_dualbranch, AND the gradient-isolated dual-branch
code/afd_train.py:2003:        # --airl_dualbranch_iso (same consistency function, same warmup).  MUST list
code/afd_train.py:2004:        # all three: the flags are mutually exclusive, so omitting iso here would
code/afd_train.py:2005:        # leave airl_lambda_eff==0 every epoch on an iso run and silently zero out
code/afd_train.py:2006:        # the f_rec consistency gradient (the whole mechanism being tested).
code/afd_train.py:2007:        airl_lambda_eff = (args.airl_lambda * min(1.0, epoch / max(1, args.airl_warmup))) \
code/afd_train.py:2008:            if (args.airl or args.airl_dualbranch or args.airl_dualbranch_iso) else 0.0
code/afd_train.py:2011:                  'airl': 0.0, 'airl_scale': 0.0, 'airl_n_ground': 0.0,
code/afd_train.py:2028:            with torch.amp.autocast('cuda', enabled=not args.no_amp):
code/afd_train.py:2038:                # AIRL dual-branch: the f_rec head needs its OWN identity grounding
code/afd_train.py:2040:                # trained on consistency alone would be unidentified).  Add f_rec's
code/afd_train.py:2041:                # ID cross-entropy (SAME label-smoothing CE as f_full); the global
code/afd_train.py:2042:                # triplet stays on the SHARED global_feat (NOT duplicated for f_rec).
code/afd_train.py:2043:                # The f_rec degradation-consistency is added below (fp32 block).
code/afd_train.py:2045:                if args.airl_dualbranch or args.airl_dualbranch_iso:
code/afd_train.py:2046:                    # f_rec ID grounding.  For --airl_dualbranch f_rec reads the
code/afd_train.py:2047:                    # shared global_feat; for --airl_dualbranch_iso it reads the
code/afd_train.py:2051:                    #   * --airl_dualbranch       -> the shared trunk (fully shared).
code/afd_train.py:2052:                    #   * --airl_dualbranch_iso, trunk_recce=1 (FIX) -> the shared
code/afd_train.py:2053:                    #     trunk TOO: model.forward ran the iso clean pass with a
code/afd_train.py:2055:                    #     trunk (extra identity supervision -> strengthens f_full)
code/afd_train.py:2057:                    #     stays trunk-isolated.
code/afd_train.py:2058:                    #   * --airl_dualbranch_iso, trunk_recce=0 -> the isolated rec
code/afd_train.py:2059:                    #     stage only (original full-isolation: clean fork detached).
code/afd_train.py:2065:                    # OVP loss in fp32 for numerical safety (cosine + softmax)
code/afd_train.py:2066:                    z = F.normalize(bn.float(), dim=1)
code/afd_train.py:2070:            # OVLI: compute in TRUE fp32 (autocast disabled) -- the cos/MaxSim/
code/afd_train.py:2072:            # the autocast forward already cached the fp16 layer4 map) keeps the
code/afd_train.py:2078:                with torch.amp.autocast('cuda', enabled=False):
code/afd_train.py:2079:                    # global feature for the score: normalized BN feat (matches
code/afd_train.py:2081:                    g_ovli = F.normalize(bn.float(), dim=1)
code/afd_train.py:2085:                        # the warmup-ramped gamma so the OVLI denominator softens
code/afd_train.py:2103:            # AIRL: resolution-degradation consistency.  ASYMMETRIC by design --
code/afd_train.py:2105:            # a sampled aerial-scale pixel budget, run ONE extra forward through the
code/afd_train.py:2109:            # low-budget aerial samples would just be all-view self-degradation and
code/afd_train.py:2111:            # params; train-time only.  Empty-ground batch -> loss_airl=0 (no extra
code/afd_train.py:2114:            loss_airl = torch.zeros((), device=device)
code/afd_train.py:2115:            airl_scale_mean = torch.zeros((), device=device)
code/afd_train.py:2117:            if args.airl:
code/afd_train.py:2126:                # -> skip AIRL this step (loss_airl stays 0, no extra forward).  The
code/afd_train.py:2127:                # ID-balanced RandomIdentitySampler makes a <2-ground batch a rare
code/afd_train.py:2133:                    # second forward runs under the SAME autocast as the original so
code/afd_train.py:2136:                        deg_imgs, deg_scales = airl_degrade(
code/afd_train.py:2137:                            imgs_g, args.airl_min_scale, blur=args.airl_blur)
code/afd_train.py:2138:                        airl_scale_mean = deg_scales.mean()
code/afd_train.py:2139:                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
code/afd_train.py:2142:                    with torch.amp.autocast('cuda', enabled=False):
code/afd_train.py:2146:                        loss_airl = airl_consistency_loss(
code/afd_train.py:2149:                            mode=args.airl_consistency, tau=args.airl_tau)
code/afd_train.py:2150:                    loss = loss + airl_lambda_eff * loss_airl
code/afd_train.py:2151:                # n_ground < 2 -> too few ground rows this batch: loss_airl stays 0,
code/afd_train.py:2155:            # AIRL dual-branch: the SAME ground-only degradation-consistency, but
code/afd_train.py:2156:            # applied ONLY to the f_rec head (logits_rec / bn_feat_rec).  f_full is
code/afd_train.py:2159:            # f_rec is pulled toward its own clean prediction under the low pixel
code/afd_train.py:2161:            # model(deg_imgs) pass (the model has no rec-only path), so f_full's
code/afd_train.py:2163:            # for stat tracking only -- exactly as in the --airl single-head path
code/afd_train.py:2166:            # empirically by kill-switch #4, and matching --airl keeps the ablation
code/afd_train.py:2168:            # --airl above; the only difference is the HEAD the consistency reads.
code/afd_train.py:2169:            # Mutually exclusive with --airl, so loss_airl is 0 unless dual-branch.
code/afd_train.py:2170:            if args.airl_dualbranch:
code/afd_train.py:2177:                        deg_imgs, deg_scales = airl_degrade(
code/afd_train.py:2178:                            imgs_g, args.airl_min_scale, blur=args.airl_blur)
code/afd_train.py:2179:                        airl_scale_mean = deg_scales.mean()
code/afd_train.py:2180:                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
code/afd_train.py:2183:                    with torch.amp.autocast('cuda', enabled=False):
code/afd_train.py:2184:                        # consistency on the f_rec head ONLY: degraded f_rec
code/afd_train.py:2185:                        # prediction (grad on) -> clean f_rec one (detached target
code/afd_train.py:2187:                        loss_airl = airl_consistency_loss(
code/afd_train.py:2190:                            mode=args.airl_consistency, tau=args.airl_tau)
code/afd_train.py:2191:                    loss = loss + airl_lambda_eff * loss_airl
code/afd_train.py:2192:                # n_ground < 2 -> skip (same size-1 BatchNorm1d guard as --airl).
code/afd_train.py:2194:            # AIRL gradient-isolated dual-branch: the SAME ground-only degradation-
code/afd_train.py:2195:            # consistency on the f_rec head.  The DEGRADED side (out_d) comes from a
code/afd_train.py:2197:            # trunk (model.forward / _forward_swin_split), and the CLEAN side
code/afd_train.py:2199:            # airl_consistency_loss.  So the consistency gradient flows ONLY through
code/afd_train.py:2201:            # detach BEFORE the shared trunk -- the clean trunk + f_full receive ZERO
code/afd_train.py:2202:            # consistency gradient (smoke I4) REGARDLESS of --airl_iso_trunk_recce
code/afd_train.py:2206:            # f_rec specialises as the recover pole.  The degraded forward uses
code/afd_train.py:2207:            # rec_only=True: it computes ONLY the f_rec head (the rec late stage +
code/afd_train.py:2208:            # BNNeck_rec), so f_full's BNNeck running stats are NOT updated on the
code/afd_train.py:2209:            # degraded images -> f_full stays a TRUE clean expert (no degraded-ground
code/afd_train.py:2210:            # stat leak, unlike the shared --airl_dualbranch which accepts that minor
code/afd_train.py:2211:            # exposure) and the f_full pool+classifier is skipped
code/afd_train.py:2212:            # (cheaper).  Mutually exclusive with --airl / --airl_dualbranch, so this
code/afd_train.py:2213:            # block fires only for the iso variant.
code/afd_train.py:2214:            if args.airl_dualbranch_iso:
code/afd_train.py:2221:                        deg_imgs, deg_scales = airl_degrade(
code/afd_train.py:2222:                            imgs_g, args.airl_min_scale, blur=args.airl_blur)
code/afd_train.py:2223:                        airl_scale_mean = deg_scales.mean()
code/afd_train.py:2224:                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
code/afd_train.py:2225:                        # rec_only -> dict with ONLY logits_rec / bn_feat_rec (f_full
code/afd_train.py:2228:                    with torch.amp.autocast('cuda', enabled=False):
code/afd_train.py:2229:                        # consistency on the ISOLATED f_rec head: degraded f_rec
code/afd_train.py:2230:                        # prediction (grad on, into the rec stage only) -> clean f_rec
code/afd_train.py:2232:                        loss_airl = airl_consistency_loss(
code/afd_train.py:2235:                            mode=args.airl_consistency, tau=args.airl_tau)
code/afd_train.py:2236:                    loss = loss + airl_lambda_eff * loss_airl
code/afd_train.py:2237:                # n_ground < 2 -> skip (same size-1 BatchNorm1d guard as --airl).
code/afd_train.py:2253:                    ovp.update(F.normalize(bn.detach().float(), dim=1),
code/afd_train.py:2261:                    acvp_mem.update(F.normalize(bn.detach().float(), dim=1),
code/afd_train.py:2283:            meters['airl'] += float(loss_airl) * bs
code/afd_train.py:2284:            meters['airl_scale'] += float(airl_scale_mean) * bs
code/afd_train.py:2285:            meters['airl_n_ground'] += n_ground
code/afd_train.py:2296:                if args.airl:
code/afd_train.py:2297:                    extra += f" AIRL: {meters['airl'] / seen:.4f}"
code/afd_train.py:2298:                if args.airl_dualbranch or args.airl_dualbranch_iso:
code/afd_train.py:2300:                              f" AIRL_rec: {meters['airl'] / seen:.4f}")
code/afd_train.py:2337:        # AIRL per-epoch kill-switch log: lam_eff (warmup-ramped weight),
code/afd_train.py:2341:        # (mean sampled pixel-budget ratio, ~ (1+min_scale)/2; sanity that the
code/afd_train.py:2342:        # degradation actually fires), n_ground (total GROUND samples degraded this
code/afd_train.py:2343:        # epoch -- AIRL is asymmetric: only views==1 ground rows get degraded; this
code/afd_train.py:2345:        # airl_collapse flag if the consistency loss is ~0 from the start
code/afd_train.py:2346:        # (degradation not biting) / NaN/inf, OR n_ground==0 all epoch (mask wrong /
code/afd_train.py:2347:        # no ground sampled => AIRL silently did nothing).
code/afd_train.py:2348:        airl_msg = ""
code/afd_train.py:2349:        if args.airl:
code/afd_train.py:2350:            a_cons = meters['airl'] / seen
code/afd_train.py:2351:            a_scale = meters['airl_scale'] / seen
code/afd_train.py:2352:            n_g = int(meters['airl_n_ground'])
code/afd_train.py:2355:            airl_msg = (f" AIRL[lam_eff={airl_lambda_eff:.3f} "
code/afd_train.py:2358:        # AIRL dual-branch per-epoch log: f_rec ID-CE (must converge like f_full's
code/afd_train.py:2359:        # CE -> f_rec is a valid identity space), the f_rec degradation-consistency
code/afd_train.py:2360:        # (same trend-down expectation as --airl), deg_scale_mean and n_ground.
code/afd_train.py:2363:        if args.airl_dualbranch or args.airl_dualbranch_iso:
code/afd_train.py:2364:            a_cons = meters['airl'] / seen
code/afd_train.py:2365:            a_scale = meters['airl_scale'] / seen
code/afd_train.py:2367:            n_g = int(meters['airl_n_ground'])
code/afd_train.py:2371:            tag = "AIRL-ISO" if args.airl_dualbranch_iso else "AIRL-DUAL"
code/afd_train.py:2372:            airl_msg = (f" {tag}[lam_eff={airl_lambda_eff:.3f} "
code/afd_train.py:2377:              f"Acc={meters['acc'] / seen:.3f}{ovp_msg}{ovli_msg}{acvp_msg}{airl_msg}")
code/afd_train.py:2395:            # AIRL dual-branch: report f_full-only, f_rec-only, and the SOFT-FUSED
code/afd_train.py:2396:            # mean (cos = w*cos_rec + (1-w)*cos_full).  The run_cross_view_eval
code/afd_train.py:2397:            # number above is the f_full-only head (model() returns f_full at eval);
code/afd_train.py:2398:            # the HEADLINE = the fused mean, which is what model-selection uses (the
code/afd_train.py:2399:            # whole point of the dual-branch is the fusion, not f_full alone).
code/afd_train.py:2400:            if args.airl_dualbranch or args.airl_dualbranch_iso:
code/afd_train.py:2401:                # airl_dualbranch_eval calls model(return_dual=True) which yields
code/afd_train.py:2402:                # (f_full, f_rec) for BOTH the shared and the iso variant (the iso
code/afd_train.py:2403:                # forward's want_iso path returns the same tuple), so the soft-fusion
code/afd_train.py:2405:                dual = airl_dualbranch_eval(model, dataset, args, device)
code/afd_train.py:2406:                _ev_tag = ("AIRL-ISO dual-branch" if args.airl_dualbranch_iso
code/afd_train.py:2407:                           else "AIRL dual-branch")
code/afd_train.py:2408:                print(f"  ---- {_ev_tag} (fuse_w={args.airl_fuse_w}) "
code/afd_train.py:2415:                          f"FUSE mAP={dual[tag]['fuse'][0]:.2f} "
code/afd_train.py:2416:                          f"R1={dual[tag]['fuse'][1]:.2f}")
code/afd_train.py:2419:                fuse_mean = (dual['A->G']['fuse'][0] + dual['G->A']['fuse'][0]) / 2
code/afd_train.py:2421:                      f"FUSE={fuse_mean:.2f}  <- model-selection uses FUSE")
code/afd_train.py:2422:                # override model-selection metric with the fused mean
code/afd_train.py:2423:                mean_map = fuse_mean

exec
/bin/zsh -lc "rg -n \"A->G|G->A|mAP|Rank-1|Epoch\\[|Validation Results|baseline|AIRL|Namespace|airl\" logs/*.log" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
logs/agreidv2_baseline_4090_s2.log:2:CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
logs/agreidv2_baseline_4090_s2.log:7:  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
logs/agreidv2_baseline_4090_s2.log:8:  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
logs/agreidv2_baseline_4090_s2.log:9:  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_baseline_4090_s2.log:11:  out_dir=/home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090_s2
logs/agreidv2_baseline_4090_s2.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_baseline_4090_s2.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_baseline_4090_s2.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_baseline_4090_s2.log:25:Epoch[1] Iter[50/786] Loss: 37.861 CE: 6.693 Tri: 31.168 Acc: 0.001 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:26:Epoch[1] Iter[100/786] Loss: 31.890 CE: 6.692 Tri: 25.199 Acc: 0.001 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:27:Epoch[1] Iter[150/786] Loss: 28.812 CE: 6.690 Tri: 22.122 Acc: 0.002 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:28:Epoch[1] Iter[200/786] Loss: 26.822 CE: 6.688 Tri: 20.133 Acc: 0.003 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:29:Epoch[1] Iter[250/786] Loss: 25.430 CE: 6.686 Tri: 18.744 Acc: 0.005 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:30:Epoch[1] Iter[300/786] Loss: 24.218 CE: 6.684 Tri: 17.534 Acc: 0.008 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:31:Epoch[1] Iter[350/786] Loss: 23.361 CE: 6.682 Tri: 16.679 Acc: 0.010 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:32:Epoch[1] Iter[400/786] Loss: 22.603 CE: 6.680 Tri: 15.924 Acc: 0.014 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:33:Epoch[1] Iter[450/786] Loss: 21.925 CE: 6.677 Tri: 15.247 Acc: 0.020 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:34:Epoch[1] Iter[500/786] Loss: 21.337 CE: 6.675 Tri: 14.663 Acc: 0.028 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:35:Epoch[1] Iter[550/786] Loss: 20.808 CE: 6.672 Tri: 14.135 Acc: 0.035 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:36:Epoch[1] Iter[600/786] Loss: 20.303 CE: 6.669 Tri: 13.633 Acc: 0.047 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:37:Epoch[1] Iter[650/786] Loss: 19.866 CE: 6.666 Tri: 13.200 Acc: 0.063 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:38:Epoch[1] Iter[700/786] Loss: 19.460 CE: 6.662 Tri: 12.798 Acc: 0.082 LR: 3.50e-07
logs/agreidv2_baseline_4090_s2.log:39:Epoch[1] done in 91.2s  Loss=19.118 Acc=0.109
logs/agreidv2_baseline_4090_s2.log:40:Epoch[2] Iter[50/786] Loss: 11.933 CE: 6.610 Tri: 5.324 Acc: 0.118 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:41:Epoch[2] Iter[100/786] Loss: 11.104 CE: 6.571 Tri: 4.533 Acc: 0.169 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:42:Epoch[2] Iter[150/786] Loss: 10.708 CE: 6.537 Tri: 4.170 Acc: 0.201 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:43:Epoch[2] Iter[200/786] Loss: 10.402 CE: 6.504 Tri: 3.898 Acc: 0.230 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:44:Epoch[2] Iter[250/786] Loss: 10.152 CE: 6.471 Tri: 3.681 Acc: 0.253 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:45:Epoch[2] Iter[300/786] Loss: 9.988 CE: 6.439 Tri: 3.550 Acc: 0.277 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:46:Epoch[2] Iter[350/786] Loss: 9.798 CE: 6.405 Tri: 3.392 Acc: 0.300 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:47:Epoch[2] Iter[400/786] Loss: 9.641 CE: 6.372 Tri: 3.269 Acc: 0.323 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:48:Epoch[2] Iter[450/786] Loss: 9.469 CE: 6.337 Tri: 3.132 Acc: 0.343 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:49:Epoch[2] Iter[500/786] Loss: 9.310 CE: 6.301 Tri: 3.009 Acc: 0.364 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:50:Epoch[2] Iter[550/786] Loss: 9.170 CE: 6.263 Tri: 2.906 Acc: 0.384 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:51:Epoch[2] Iter[600/786] Loss: 9.037 CE: 6.223 Tri: 2.814 Acc: 0.406 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:52:Epoch[2] Iter[650/786] Loss: 8.918 CE: 6.180 Tri: 2.738 Acc: 0.428 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:53:Epoch[2] Iter[700/786] Loss: 8.800 CE: 6.130 Tri: 2.670 Acc: 0.450 LR: 3.82e-06
logs/agreidv2_baseline_4090_s2.log:54:Epoch[2] done in 91.5s  Loss=8.680 Acc=0.474
logs/agreidv2_baseline_4090_s2.log:55:Epoch[3] Iter[50/786] Loss: 7.091 CE: 5.601 Tri: 1.491 Acc: 0.304 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:56:Epoch[3] Iter[100/786] Loss: 6.853 CE: 5.496 Tri: 1.357 Acc: 0.341 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:57:Epoch[3] Iter[150/786] Loss: 6.762 CE: 5.425 Tri: 1.337 Acc: 0.370 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:58:Epoch[3] Iter[200/786] Loss: 6.659 CE: 5.352 Tri: 1.307 Acc: 0.400 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:59:Epoch[3] Iter[250/786] Loss: 6.539 CE: 5.270 Tri: 1.269 Acc: 0.432 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:60:Epoch[3] Iter[300/786] Loss: 6.409 CE: 5.186 Tri: 1.223 Acc: 0.462 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:61:Epoch[3] Iter[350/786] Loss: 6.293 CE: 5.100 Tri: 1.193 Acc: 0.488 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:62:Epoch[3] Iter[400/786] Loss: 6.187 CE: 5.020 Tri: 1.167 Acc: 0.508 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:63:Epoch[3] Iter[450/786] Loss: 6.071 CE: 4.930 Tri: 1.141 Acc: 0.531 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:64:Epoch[3] Iter[500/786] Loss: 5.965 CE: 4.842 Tri: 1.122 Acc: 0.550 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:65:Epoch[3] Iter[550/786] Loss: 5.855 CE: 4.752 Tri: 1.103 Acc: 0.570 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:66:Epoch[3] Iter[600/786] Loss: 5.736 CE: 4.654 Tri: 1.082 Acc: 0.587 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:67:Epoch[3] Iter[650/786] Loss: 5.615 CE: 4.552 Tri: 1.062 Acc: 0.604 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:68:Epoch[3] Iter[700/786] Loss: 5.479 CE: 4.437 Tri: 1.043 Acc: 0.622 LR: 7.28e-06
logs/agreidv2_baseline_4090_s2.log:69:Epoch[3] done in 91.4s  Loss=5.334 Acc=0.640
logs/agreidv2_baseline_4090_s2.log:70:Epoch[4] Iter[50/786] Loss: 4.582 CE: 3.840 Tri: 0.742 Acc: 0.463 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:71:Epoch[4] Iter[100/786] Loss: 4.444 CE: 3.706 Tri: 0.738 Acc: 0.513 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:72:Epoch[4] Iter[150/786] Loss: 4.301 CE: 3.592 Tri: 0.709 Acc: 0.554 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:73:Epoch[4] Iter[200/786] Loss: 4.194 CE: 3.492 Tri: 0.702 Acc: 0.587 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:74:Epoch[4] Iter[250/786] Loss: 4.093 CE: 3.395 Tri: 0.698 Acc: 0.617 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:75:Epoch[4] Iter[300/786] Loss: 3.982 CE: 3.296 Tri: 0.686 Acc: 0.645 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:76:Epoch[4] Iter[350/786] Loss: 3.882 CE: 3.204 Tri: 0.678 Acc: 0.669 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:77:Epoch[4] Iter[400/786] Loss: 3.776 CE: 3.117 Tri: 0.659 Acc: 0.691 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:78:Epoch[4] Iter[450/786] Loss: 3.687 CE: 3.034 Tri: 0.653 Acc: 0.710 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:79:Epoch[4] Iter[500/786] Loss: 3.596 CE: 2.955 Tri: 0.641 Acc: 0.727 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:80:Epoch[4] Iter[550/786] Loss: 3.507 CE: 2.877 Tri: 0.630 Acc: 0.742 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:81:Epoch[4] Iter[600/786] Loss: 3.421 CE: 2.803 Tri: 0.618 Acc: 0.755 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:82:Epoch[4] Iter[650/786] Loss: 3.339 CE: 2.731 Tri: 0.608 Acc: 0.767 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:83:Epoch[4] Iter[700/786] Loss: 3.249 CE: 2.655 Tri: 0.594 Acc: 0.779 LR: 1.07e-05
logs/agreidv2_baseline_4090_s2.log:84:Epoch[4] done in 91.0s  Loss=3.182 Acc=0.788
logs/agreidv2_baseline_4090_s2.log:85:Epoch[5] Iter[50/786] Loss: 2.884 CE: 2.438 Tri: 0.446 Acc: 0.742 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:86:Epoch[5] Iter[100/786] Loss: 2.896 CE: 2.389 Tri: 0.506 Acc: 0.761 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:87:Epoch[5] Iter[150/786] Loss: 2.790 CE: 2.303 Tri: 0.487 Acc: 0.785 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:88:Epoch[5] Iter[200/786] Loss: 2.727 CE: 2.239 Tri: 0.488 Acc: 0.805 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:89:Epoch[5] Iter[250/786] Loss: 2.654 CE: 2.177 Tri: 0.477 Acc: 0.821 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:90:Epoch[5] Iter[300/786] Loss: 2.593 CE: 2.125 Tri: 0.468 Acc: 0.834 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:91:Epoch[5] Iter[350/786] Loss: 2.530 CE: 2.073 Tri: 0.456 Acc: 0.845 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:92:Epoch[5] Iter[400/786] Loss: 2.482 CE: 2.032 Tri: 0.450 Acc: 0.853 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:93:Epoch[5] Iter[450/786] Loss: 2.431 CE: 1.994 Tri: 0.437 Acc: 0.861 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:94:Epoch[5] Iter[500/786] Loss: 2.387 CE: 1.959 Tri: 0.428 Acc: 0.867 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:95:Epoch[5] Iter[550/786] Loss: 2.350 CE: 1.928 Tri: 0.422 Acc: 0.873 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:96:Epoch[5] Iter[600/786] Loss: 2.322 CE: 1.902 Tri: 0.420 Acc: 0.877 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:97:Epoch[5] Iter[650/786] Loss: 2.290 CE: 1.875 Tri: 0.415 Acc: 0.882 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:98:Epoch[5] Iter[700/786] Loss: 2.258 CE: 1.847 Tri: 0.411 Acc: 0.887 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:99:Epoch[5] done in 91.2s  Loss=2.228 Acc=0.892
logs/agreidv2_baseline_4090_s2.log:100:Epoch[6] Iter[50/786] Loss: 2.307 CE: 1.898 Tri: 0.409 Acc: 0.853 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:101:Epoch[6] Iter[100/786] Loss: 2.189 CE: 1.816 Tri: 0.373 Acc: 0.876 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:102:Epoch[6] Iter[150/786] Loss: 2.120 CE: 1.770 Tri: 0.350 Acc: 0.887 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:103:Epoch[6] Iter[200/786] Loss: 2.069 CE: 1.731 Tri: 0.338 Acc: 0.897 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:104:Epoch[6] Iter[250/786] Loss: 2.047 CE: 1.708 Tri: 0.339 Acc: 0.903 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:105:Epoch[6] Iter[300/786] Loss: 2.018 CE: 1.686 Tri: 0.332 Acc: 0.908 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:106:Epoch[6] Iter[350/786] Loss: 1.995 CE: 1.666 Tri: 0.329 Acc: 0.912 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:107:Epoch[6] Iter[400/786] Loss: 1.981 CE: 1.652 Tri: 0.330 Acc: 0.915 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:108:Epoch[6] Iter[450/786] Loss: 1.961 CE: 1.637 Tri: 0.324 Acc: 0.919 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:109:Epoch[6] Iter[500/786] Loss: 1.938 CE: 1.622 Tri: 0.316 Acc: 0.922 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:110:Epoch[6] Iter[550/786] Loss: 1.923 CE: 1.609 Tri: 0.314 Acc: 0.924 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:111:Epoch[6] Iter[600/786] Loss: 1.903 CE: 1.596 Tri: 0.307 Acc: 0.927 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:112:Epoch[6] Iter[650/786] Loss: 1.889 CE: 1.585 Tri: 0.304 Acc: 0.929 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:113:Epoch[6] Iter[700/786] Loss: 1.876 CE: 1.572 Tri: 0.303 Acc: 0.931 LR: 1.77e-05
logs/agreidv2_baseline_4090_s2.log:114:Epoch[6] done in 90.5s  Loss=1.862 Acc=0.933
logs/agreidv2_baseline_4090_s2.log:115:Epoch[7] Iter[50/786] Loss: 1.873 CE: 1.614 Tri: 0.259 Acc: 0.914 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:116:Epoch[7] Iter[100/786] Loss: 1.858 CE: 1.579 Tri: 0.279 Acc: 0.926 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:117:Epoch[7] Iter[150/786] Loss: 1.855 CE: 1.568 Tri: 0.287 Acc: 0.928 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:118:Epoch[7] Iter[200/786] Loss: 1.830 CE: 1.551 Tri: 0.279 Acc: 0.932 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:119:Epoch[7] Iter[250/786] Loss: 1.816 CE: 1.538 Tri: 0.277 Acc: 0.935 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:120:Epoch[7] Iter[300/786] Loss: 1.805 CE: 1.530 Tri: 0.275 Acc: 0.937 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:121:Epoch[7] Iter[350/786] Loss: 1.793 CE: 1.524 Tri: 0.270 Acc: 0.938 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:122:Epoch[7] Iter[400/786] Loss: 1.785 CE: 1.516 Tri: 0.269 Acc: 0.939 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:123:Epoch[7] Iter[450/786] Loss: 1.779 CE: 1.510 Tri: 0.269 Acc: 0.940 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:124:Epoch[7] Iter[500/786] Loss: 1.769 CE: 1.503 Tri: 0.266 Acc: 0.941 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:125:Epoch[7] Iter[550/786] Loss: 1.759 CE: 1.497 Tri: 0.263 Acc: 0.942 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:126:Epoch[7] Iter[600/786] Loss: 1.752 CE: 1.491 Tri: 0.261 Acc: 0.944 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:127:Epoch[7] Iter[650/786] Loss: 1.741 CE: 1.483 Tri: 0.258 Acc: 0.945 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:128:Epoch[7] Iter[700/786] Loss: 1.729 CE: 1.474 Tri: 0.255 Acc: 0.947 LR: 2.11e-05
logs/agreidv2_baseline_4090_s2.log:129:Epoch[7] done in 91.6s  Loss=1.715 Acc=0.949
logs/agreidv2_baseline_4090_s2.log:130:Epoch[8] Iter[50/786] Loss: 1.839 CE: 1.553 Tri: 0.286 Acc: 0.924 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:131:Epoch[8] Iter[100/786] Loss: 1.802 CE: 1.529 Tri: 0.273 Acc: 0.932 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:132:Epoch[8] Iter[150/786] Loss: 1.763 CE: 1.512 Tri: 0.252 Acc: 0.935 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:133:Epoch[8] Iter[200/786] Loss: 1.741 CE: 1.498 Tri: 0.243 Acc: 0.940 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:134:Epoch[8] Iter[250/786] Loss: 1.722 CE: 1.485 Tri: 0.237 Acc: 0.942 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:135:Epoch[8] Iter[300/786] Loss: 1.709 CE: 1.475 Tri: 0.233 Acc: 0.944 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:136:Epoch[8] Iter[350/786] Loss: 1.697 CE: 1.469 Tri: 0.228 Acc: 0.946 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:137:Epoch[8] Iter[400/786] Loss: 1.697 CE: 1.465 Tri: 0.232 Acc: 0.947 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:138:Epoch[8] Iter[450/786] Loss: 1.694 CE: 1.461 Tri: 0.233 Acc: 0.948 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:139:Epoch[8] Iter[500/786] Loss: 1.689 CE: 1.456 Tri: 0.233 Acc: 0.949 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:140:Epoch[8] Iter[550/786] Loss: 1.683 CE: 1.451 Tri: 0.232 Acc: 0.949 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:141:Epoch[8] Iter[600/786] Loss: 1.678 CE: 1.445 Tri: 0.233 Acc: 0.951 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:142:Epoch[8] Iter[650/786] Loss: 1.673 CE: 1.440 Tri: 0.233 Acc: 0.952 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:143:Epoch[8] Iter[700/786] Loss: 1.663 CE: 1.433 Tri: 0.230 Acc: 0.953 LR: 2.46e-05
logs/agreidv2_baseline_4090_s2.log:144:Epoch[8] done in 91.8s  Loss=1.647 Acc=0.955
logs/agreidv2_baseline_4090_s2.log:145:Epoch[9] Iter[50/786] Loss: 1.758 CE: 1.530 Tri: 0.227 Acc: 0.928 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:146:Epoch[9] Iter[100/786] Loss: 1.719 CE: 1.491 Tri: 0.228 Acc: 0.939 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:147:Epoch[9] Iter[150/786] Loss: 1.707 CE: 1.477 Tri: 0.230 Acc: 0.943 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:148:Epoch[9] Iter[200/786] Loss: 1.684 CE: 1.465 Tri: 0.219 Acc: 0.946 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:149:Epoch[9] Iter[250/786] Loss: 1.683 CE: 1.461 Tri: 0.223 Acc: 0.946 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:150:Epoch[9] Iter[300/786] Loss: 1.682 CE: 1.457 Tri: 0.225 Acc: 0.947 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:151:Epoch[9] Iter[350/786] Loss: 1.684 CE: 1.457 Tri: 0.227 Acc: 0.947 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:152:Epoch[9] Iter[400/786] Loss: 1.680 CE: 1.453 Tri: 0.227 Acc: 0.948 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:153:Epoch[9] Iter[450/786] Loss: 1.671 CE: 1.446 Tri: 0.225 Acc: 0.949 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:154:Epoch[9] Iter[500/786] Loss: 1.664 CE: 1.441 Tri: 0.223 Acc: 0.950 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:155:Epoch[9] Iter[550/786] Loss: 1.658 CE: 1.436 Tri: 0.222 Acc: 0.950 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:156:Epoch[9] Iter[600/786] Loss: 1.654 CE: 1.433 Tri: 0.222 Acc: 0.950 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:157:Epoch[9] Iter[650/786] Loss: 1.643 CE: 1.426 Tri: 0.217 Acc: 0.952 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:158:Epoch[9] Iter[700/786] Loss: 1.631 CE: 1.417 Tri: 0.215 Acc: 0.954 LR: 2.81e-05
logs/agreidv2_baseline_4090_s2.log:159:Epoch[9] done in 90.9s  Loss=1.619 Acc=0.955
logs/agreidv2_baseline_4090_s2.log:160:Epoch[10] Iter[50/786] Loss: 1.676 CE: 1.486 Tri: 0.190 Acc: 0.938 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:161:Epoch[10] Iter[100/786] Loss: 1.659 CE: 1.459 Tri: 0.201 Acc: 0.945 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:162:Epoch[10] Iter[150/786] Loss: 1.630 CE: 1.443 Tri: 0.187 Acc: 0.948 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:163:Epoch[10] Iter[200/786] Loss: 1.632 CE: 1.436 Tri: 0.196 Acc: 0.950 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:164:Epoch[10] Iter[250/786] Loss: 1.631 CE: 1.428 Tri: 0.202 Acc: 0.951 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:165:Epoch[10] Iter[300/786] Loss: 1.624 CE: 1.424 Tri: 0.200 Acc: 0.952 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:166:Epoch[10] Iter[350/786] Loss: 1.623 CE: 1.423 Tri: 0.200 Acc: 0.952 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:167:Epoch[10] Iter[400/786] Loss: 1.621 CE: 1.421 Tri: 0.201 Acc: 0.952 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:168:Epoch[10] Iter[450/786] Loss: 1.616 CE: 1.417 Tri: 0.199 Acc: 0.953 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:169:Epoch[10] Iter[500/786] Loss: 1.612 CE: 1.414 Tri: 0.198 Acc: 0.954 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:170:Epoch[10] Iter[550/786] Loss: 1.605 CE: 1.410 Tri: 0.195 Acc: 0.954 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:171:Epoch[10] Iter[600/786] Loss: 1.599 CE: 1.403 Tri: 0.196 Acc: 0.955 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:172:Epoch[10] Iter[650/786] Loss: 1.591 CE: 1.397 Tri: 0.194 Acc: 0.956 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:173:Epoch[10] Iter[700/786] Loss: 1.582 CE: 1.390 Tri: 0.192 Acc: 0.958 LR: 3.15e-05
logs/agreidv2_baseline_4090_s2.log:174:Epoch[10] done in 91.5s  Loss=1.568 Acc=0.959
logs/agreidv2_baseline_4090_s2.log:176:    [A->G] mAP=74.43  R1=83.49  R5=89.90  mINP=45.26
logs/agreidv2_baseline_4090_s2.log:177:    [G->A] mAP=74.11  R1=83.66  R5=90.50  mINP=36.04
logs/agreidv2_baseline_4090_s2.log:178:    [mean] mAP=74.27  R1=83.57
logs/agreidv2_baseline_4090_s2.log:179:    * new best mean mAP=74.27 (epoch 10) saved
logs/agreidv2_baseline_4090_s2.log:180:Epoch[11] Iter[50/786] Loss: 1.648 CE: 1.481 Tri: 0.167 Acc: 0.937 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:181:Epoch[11] Iter[100/786] Loss: 1.639 CE: 1.459 Tri: 0.180 Acc: 0.943 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:182:Epoch[11] Iter[150/786] Loss: 1.623 CE: 1.444 Tri: 0.179 Acc: 0.947 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:183:Epoch[11] Iter[200/786] Loss: 1.616 CE: 1.439 Tri: 0.177 Acc: 0.948 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:184:Epoch[11] Iter[250/786] Loss: 1.623 CE: 1.436 Tri: 0.187 Acc: 0.948 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:185:Epoch[11] Iter[300/786] Loss: 1.613 CE: 1.428 Tri: 0.184 Acc: 0.950 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:186:Epoch[11] Iter[350/786] Loss: 1.607 CE: 1.423 Tri: 0.184 Acc: 0.951 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:187:Epoch[11] Iter[400/786] Loss: 1.599 CE: 1.417 Tri: 0.182 Acc: 0.952 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:188:Epoch[11] Iter[450/786] Loss: 1.596 CE: 1.413 Tri: 0.183 Acc: 0.953 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:189:Epoch[11] Iter[500/786] Loss: 1.587 CE: 1.406 Tri: 0.180 Acc: 0.953 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:190:Epoch[11] Iter[550/786] Loss: 1.582 CE: 1.403 Tri: 0.179 Acc: 0.954 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:191:Epoch[11] Iter[600/786] Loss: 1.576 CE: 1.398 Tri: 0.178 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:192:Epoch[11] Iter[650/786] Loss: 1.567 CE: 1.391 Tri: 0.176 Acc: 0.956 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:193:Epoch[11] Iter[700/786] Loss: 1.559 CE: 1.384 Tri: 0.174 Acc: 0.957 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:194:Epoch[11] done in 91.8s  Loss=1.551 Acc=0.959
logs/agreidv2_baseline_4090_s2.log:195:Epoch[12] Iter[50/786] Loss: 1.765 CE: 1.536 Tri: 0.230 Acc: 0.927 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:196:Epoch[12] Iter[100/786] Loss: 1.671 CE: 1.473 Tri: 0.198 Acc: 0.942 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:197:Epoch[12] Iter[150/786] Loss: 1.647 CE: 1.451 Tri: 0.196 Acc: 0.945 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:198:Epoch[12] Iter[200/786] Loss: 1.623 CE: 1.434 Tri: 0.189 Acc: 0.949 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:199:Epoch[12] Iter[250/786] Loss: 1.596 CE: 1.418 Tri: 0.177 Acc: 0.952 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:200:Epoch[12] Iter[300/786] Loss: 1.581 CE: 1.407 Tri: 0.174 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:201:Epoch[12] Iter[350/786] Loss: 1.578 CE: 1.403 Tri: 0.175 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:202:Epoch[12] Iter[400/786] Loss: 1.578 CE: 1.401 Tri: 0.178 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:203:Epoch[12] Iter[450/786] Loss: 1.572 CE: 1.396 Tri: 0.175 Acc: 0.956 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:204:Epoch[12] Iter[500/786] Loss: 1.565 CE: 1.392 Tri: 0.173 Acc: 0.956 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:205:Epoch[12] Iter[550/786] Loss: 1.552 CE: 1.384 Tri: 0.168 Acc: 0.958 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:206:Epoch[12] Iter[600/786] Loss: 1.548 CE: 1.379 Tri: 0.168 Acc: 0.958 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:207:Epoch[12] Iter[650/786] Loss: 1.539 CE: 1.374 Tri: 0.165 Acc: 0.959 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:208:Epoch[12] Iter[700/786] Loss: 1.529 CE: 1.366 Tri: 0.163 Acc: 0.960 LR: 3.50e-05
logs/agreidv2_baseline_4090_s2.log:209:Epoch[12] done in 91.3s  Loss=1.516 Acc=0.962
logs/agreidv2_baseline_4090_s2.log:210:Epoch[13] Iter[50/786] Loss: 1.654 CE: 1.487 Tri: 0.168 Acc: 0.934 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:211:Epoch[13] Iter[100/786] Loss: 1.618 CE: 1.451 Tri: 0.166 Acc: 0.942 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:212:Epoch[13] Iter[150/786] Loss: 1.572 CE: 1.418 Tri: 0.155 Acc: 0.951 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:213:Epoch[13] Iter[200/786] Loss: 1.547 CE: 1.396 Tri: 0.151 Acc: 0.955 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:214:Epoch[13] Iter[250/786] Loss: 1.535 CE: 1.386 Tri: 0.149 Acc: 0.957 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:215:Epoch[13] Iter[300/786] Loss: 1.527 CE: 1.381 Tri: 0.147 Acc: 0.958 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:216:Epoch[13] Iter[350/786] Loss: 1.517 CE: 1.373 Tri: 0.143 Acc: 0.960 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:217:Epoch[13] Iter[400/786] Loss: 1.505 CE: 1.366 Tri: 0.139 Acc: 0.961 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:218:Epoch[13] Iter[450/786] Loss: 1.496 CE: 1.359 Tri: 0.136 Acc: 0.962 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:219:Epoch[13] Iter[500/786] Loss: 1.488 CE: 1.354 Tri: 0.134 Acc: 0.963 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:220:Epoch[13] Iter[550/786] Loss: 1.484 CE: 1.349 Tri: 0.135 Acc: 0.964 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:221:Epoch[13] Iter[600/786] Loss: 1.478 CE: 1.344 Tri: 0.134 Acc: 0.964 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:222:Epoch[13] Iter[650/786] Loss: 1.471 CE: 1.338 Tri: 0.132 Acc: 0.965 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:223:Epoch[13] Iter[700/786] Loss: 1.464 CE: 1.332 Tri: 0.132 Acc: 0.966 LR: 3.49e-05
logs/agreidv2_baseline_4090_s2.log:224:Epoch[13] done in 92.1s  Loss=1.452 Acc=0.968
logs/agreidv2_baseline_4090_s2.log:225:Epoch[14] Iter[50/786] Loss: 1.576 CE: 1.425 Tri: 0.151 Acc: 0.952 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:226:Epoch[14] Iter[100/786] Loss: 1.518 CE: 1.387 Tri: 0.131 Acc: 0.958 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:227:Epoch[14] Iter[150/786] Loss: 1.495 CE: 1.369 Tri: 0.126 Acc: 0.960 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:228:Epoch[14] Iter[200/786] Loss: 1.482 CE: 1.357 Tri: 0.126 Acc: 0.963 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:229:Epoch[14] Iter[250/786] Loss: 1.478 CE: 1.352 Tri: 0.125 Acc: 0.964 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:230:Epoch[14] Iter[300/786] Loss: 1.471 CE: 1.347 Tri: 0.123 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:231:Epoch[14] Iter[350/786] Loss: 1.469 CE: 1.344 Tri: 0.125 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:232:Epoch[14] Iter[400/786] Loss: 1.465 CE: 1.342 Tri: 0.124 Acc: 0.964 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:233:Epoch[14] Iter[450/786] Loss: 1.461 CE: 1.337 Tri: 0.124 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:234:Epoch[14] Iter[500/786] Loss: 1.458 CE: 1.334 Tri: 0.124 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:235:Epoch[14] Iter[550/786] Loss: 1.452 CE: 1.329 Tri: 0.123 Acc: 0.966 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:236:Epoch[14] Iter[600/786] Loss: 1.448 CE: 1.325 Tri: 0.123 Acc: 0.967 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:237:Epoch[14] Iter[650/786] Loss: 1.443 CE: 1.320 Tri: 0.123 Acc: 0.968 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:238:Epoch[14] Iter[700/786] Loss: 1.434 CE: 1.314 Tri: 0.121 Acc: 0.969 LR: 3.47e-05
logs/agreidv2_baseline_4090_s2.log:239:Epoch[14] done in 91.9s  Loss=1.423 Acc=0.970
logs/agreidv2_baseline_4090_s2.log:240:Epoch[15] Iter[50/786] Loss: 1.579 CE: 1.431 Tri: 0.148 Acc: 0.942 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:241:Epoch[15] Iter[100/786] Loss: 1.538 CE: 1.397 Tri: 0.142 Acc: 0.951 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:242:Epoch[15] Iter[150/786] Loss: 1.508 CE: 1.375 Tri: 0.133 Acc: 0.956 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:243:Epoch[15] Iter[200/786] Loss: 1.497 CE: 1.365 Tri: 0.132 Acc: 0.958 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:244:Epoch[15] Iter[250/786] Loss: 1.487 CE: 1.356 Tri: 0.131 Acc: 0.960 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:245:Epoch[15] Iter[300/786] Loss: 1.473 CE: 1.347 Tri: 0.126 Acc: 0.961 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:246:Epoch[15] Iter[350/786] Loss: 1.465 CE: 1.340 Tri: 0.125 Acc: 0.963 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:247:Epoch[15] Iter[400/786] Loss: 1.453 CE: 1.332 Tri: 0.121 Acc: 0.964 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:248:Epoch[15] Iter[450/786] Loss: 1.448 CE: 1.328 Tri: 0.121 Acc: 0.965 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:249:Epoch[15] Iter[500/786] Loss: 1.440 CE: 1.322 Tri: 0.118 Acc: 0.966 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:250:Epoch[15] Iter[550/786] Loss: 1.436 CE: 1.318 Tri: 0.118 Acc: 0.967 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:251:Epoch[15] Iter[600/786] Loss: 1.430 CE: 1.313 Tri: 0.116 Acc: 0.968 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:252:Epoch[15] Iter[650/786] Loss: 1.423 CE: 1.308 Tri: 0.115 Acc: 0.969 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:253:Epoch[15] Iter[700/786] Loss: 1.414 CE: 1.301 Tri: 0.113 Acc: 0.970 LR: 3.45e-05
logs/agreidv2_baseline_4090_s2.log:254:Epoch[15] done in 91.8s  Loss=1.402 Acc=0.971
logs/agreidv2_baseline_4090_s2.log:255:Epoch[16] Iter[50/786] Loss: 1.508 CE: 1.398 Tri: 0.110 Acc: 0.947 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:256:Epoch[16] Iter[100/786] Loss: 1.467 CE: 1.365 Tri: 0.102 Acc: 0.956 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:257:Epoch[16] Iter[150/786] Loss: 1.447 CE: 1.347 Tri: 0.100 Acc: 0.960 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:258:Epoch[16] Iter[200/786] Loss: 1.441 CE: 1.338 Tri: 0.103 Acc: 0.962 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:259:Epoch[16] Iter[250/786] Loss: 1.442 CE: 1.333 Tri: 0.109 Acc: 0.963 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:260:Epoch[16] Iter[300/786] Loss: 1.438 CE: 1.328 Tri: 0.109 Acc: 0.964 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:261:Epoch[16] Iter[350/786] Loss: 1.430 CE: 1.322 Tri: 0.108 Acc: 0.965 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:262:Epoch[16] Iter[400/786] Loss: 1.427 CE: 1.319 Tri: 0.109 Acc: 0.966 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:263:Epoch[16] Iter[450/786] Loss: 1.423 CE: 1.315 Tri: 0.108 Acc: 0.966 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:264:Epoch[16] Iter[500/786] Loss: 1.415 CE: 1.309 Tri: 0.107 Acc: 0.968 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:265:Epoch[16] Iter[550/786] Loss: 1.409 CE: 1.303 Tri: 0.105 Acc: 0.969 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:266:Epoch[16] Iter[600/786] Loss: 1.403 CE: 1.298 Tri: 0.105 Acc: 0.970 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:267:Epoch[16] Iter[650/786] Loss: 1.396 CE: 1.292 Tri: 0.104 Acc: 0.971 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:268:Epoch[16] Iter[700/786] Loss: 1.390 CE: 1.287 Tri: 0.103 Acc: 0.972 LR: 3.41e-05
logs/agreidv2_baseline_4090_s2.log:269:Epoch[16] done in 91.3s  Loss=1.380 Acc=0.973
logs/agreidv2_baseline_4090_s2.log:270:Epoch[17] Iter[50/786] Loss: 1.495 CE: 1.382 Tri: 0.113 Acc: 0.952 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:271:Epoch[17] Iter[100/786] Loss: 1.465 CE: 1.355 Tri: 0.111 Acc: 0.958 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:272:Epoch[17] Iter[150/786] Loss: 1.444 CE: 1.336 Tri: 0.109 Acc: 0.962 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:273:Epoch[17] Iter[200/786] Loss: 1.423 CE: 1.321 Tri: 0.102 Acc: 0.965 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:274:Epoch[17] Iter[250/786] Loss: 1.403 CE: 1.309 Tri: 0.094 Acc: 0.968 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:275:Epoch[17] Iter[300/786] Loss: 1.395 CE: 1.301 Tri: 0.094 Acc: 0.969 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:276:Epoch[17] Iter[350/786] Loss: 1.388 CE: 1.296 Tri: 0.093 Acc: 0.970 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:277:Epoch[17] Iter[400/786] Loss: 1.393 CE: 1.296 Tri: 0.097 Acc: 0.969 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:278:Epoch[17] Iter[450/786] Loss: 1.387 CE: 1.292 Tri: 0.096 Acc: 0.970 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:279:Epoch[17] Iter[500/786] Loss: 1.384 CE: 1.289 Tri: 0.095 Acc: 0.971 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:280:Epoch[17] Iter[550/786] Loss: 1.383 CE: 1.286 Tri: 0.097 Acc: 0.971 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:281:Epoch[17] Iter[600/786] Loss: 1.380 CE: 1.283 Tri: 0.097 Acc: 0.972 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:282:Epoch[17] Iter[650/786] Loss: 1.373 CE: 1.278 Tri: 0.095 Acc: 0.973 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:283:Epoch[17] Iter[700/786] Loss: 1.365 CE: 1.272 Tri: 0.093 Acc: 0.974 LR: 3.38e-05
logs/agreidv2_baseline_4090_s2.log:284:Epoch[17] done in 91.4s  Loss=1.355 Acc=0.975
logs/agreidv2_baseline_4090_s2.log:285:Epoch[18] Iter[50/786] Loss: 1.443 CE: 1.342 Tri: 0.101 Acc: 0.965 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:286:Epoch[18] Iter[100/786] Loss: 1.429 CE: 1.319 Tri: 0.110 Acc: 0.969 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:287:Epoch[18] Iter[150/786] Loss: 1.405 CE: 1.304 Tri: 0.101 Acc: 0.969 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:288:Epoch[18] Iter[200/786] Loss: 1.391 CE: 1.295 Tri: 0.096 Acc: 0.971 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:289:Epoch[18] Iter[250/786] Loss: 1.380 CE: 1.287 Tri: 0.094 Acc: 0.973 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:290:Epoch[18] Iter[300/786] Loss: 1.374 CE: 1.282 Tri: 0.092 Acc: 0.974 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:291:Epoch[18] Iter[350/786] Loss: 1.374 CE: 1.280 Tri: 0.094 Acc: 0.973 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:292:Epoch[18] Iter[400/786] Loss: 1.370 CE: 1.277 Tri: 0.094 Acc: 0.973 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:293:Epoch[18] Iter[450/786] Loss: 1.361 CE: 1.271 Tri: 0.090 Acc: 0.974 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:294:Epoch[18] Iter[500/786] Loss: 1.358 CE: 1.269 Tri: 0.089 Acc: 0.974 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:295:Epoch[18] Iter[550/786] Loss: 1.353 CE: 1.265 Tri: 0.088 Acc: 0.975 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:296:Epoch[18] Iter[600/786] Loss: 1.349 CE: 1.262 Tri: 0.087 Acc: 0.975 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:297:Epoch[18] Iter[650/786] Loss: 1.345 CE: 1.258 Tri: 0.087 Acc: 0.976 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:298:Epoch[18] Iter[700/786] Loss: 1.339 CE: 1.253 Tri: 0.086 Acc: 0.977 LR: 3.33e-05
logs/agreidv2_baseline_4090_s2.log:299:Epoch[18] done in 91.4s  Loss=1.331 Acc=0.978
logs/agreidv2_baseline_4090_s2.log:300:Epoch[19] Iter[50/786] Loss: 1.460 CE: 1.344 Tri: 0.116 Acc: 0.963 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:301:Epoch[19] Iter[100/786] Loss: 1.423 CE: 1.318 Tri: 0.104 Acc: 0.968 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:302:Epoch[19] Iter[150/786] Loss: 1.403 CE: 1.304 Tri: 0.099 Acc: 0.971 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:303:Epoch[19] Iter[200/786] Loss: 1.390 CE: 1.294 Tri: 0.097 Acc: 0.971 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:304:Epoch[19] Iter[250/786] Loss: 1.379 CE: 1.287 Tri: 0.092 Acc: 0.972 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:305:Epoch[19] Iter[300/786] Loss: 1.369 CE: 1.280 Tri: 0.089 Acc: 0.972 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:306:Epoch[19] Iter[350/786] Loss: 1.361 CE: 1.275 Tri: 0.086 Acc: 0.973 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:307:Epoch[19] Iter[400/786] Loss: 1.356 CE: 1.272 Tri: 0.084 Acc: 0.973 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:308:Epoch[19] Iter[450/786] Loss: 1.345 CE: 1.266 Tri: 0.079 Acc: 0.974 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:309:Epoch[19] Iter[500/786] Loss: 1.339 CE: 1.261 Tri: 0.078 Acc: 0.975 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:310:Epoch[19] Iter[550/786] Loss: 1.333 CE: 1.257 Tri: 0.077 Acc: 0.975 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:311:Epoch[19] Iter[600/786] Loss: 1.328 CE: 1.252 Tri: 0.076 Acc: 0.976 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:312:Epoch[19] Iter[650/786] Loss: 1.322 CE: 1.247 Tri: 0.075 Acc: 0.977 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:313:Epoch[19] Iter[700/786] Loss: 1.315 CE: 1.242 Tri: 0.073 Acc: 0.978 LR: 3.28e-05
logs/agreidv2_baseline_4090_s2.log:314:Epoch[19] done in 92.1s  Loss=1.305 Acc=0.979
logs/agreidv2_baseline_4090_s2.log:315:Epoch[20] Iter[50/786] Loss: 1.389 CE: 1.317 Tri: 0.072 Acc: 0.967 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:316:Epoch[20] Iter[100/786] Loss: 1.365 CE: 1.286 Tri: 0.078 Acc: 0.973 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:317:Epoch[20] Iter[150/786] Loss: 1.353 CE: 1.274 Tri: 0.080 Acc: 0.974 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:318:Epoch[20] Iter[200/786] Loss: 1.341 CE: 1.264 Tri: 0.077 Acc: 0.976 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:319:Epoch[20] Iter[250/786] Loss: 1.329 CE: 1.256 Tri: 0.073 Acc: 0.976 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:320:Epoch[20] Iter[300/786] Loss: 1.325 CE: 1.252 Tri: 0.074 Acc: 0.977 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:321:Epoch[20] Iter[350/786] Loss: 1.320 CE: 1.249 Tri: 0.071 Acc: 0.976 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:322:Epoch[20] Iter[400/786] Loss: 1.318 CE: 1.247 Tri: 0.070 Acc: 0.977 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:323:Epoch[20] Iter[450/786] Loss: 1.317 CE: 1.246 Tri: 0.071 Acc: 0.977 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:324:Epoch[20] Iter[500/786] Loss: 1.311 CE: 1.241 Tri: 0.069 Acc: 0.978 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:325:Epoch[20] Iter[550/786] Loss: 1.309 CE: 1.239 Tri: 0.070 Acc: 0.978 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:326:Epoch[20] Iter[600/786] Loss: 1.306 CE: 1.236 Tri: 0.071 Acc: 0.979 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:327:Epoch[20] Iter[650/786] Loss: 1.303 CE: 1.233 Tri: 0.070 Acc: 0.979 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:328:Epoch[20] Iter[700/786] Loss: 1.297 CE: 1.228 Tri: 0.069 Acc: 0.980 LR: 3.23e-05
logs/agreidv2_baseline_4090_s2.log:329:Epoch[20] done in 91.0s  Loss=1.292 Acc=0.980
logs/agreidv2_baseline_4090_s2.log:331:    [A->G] mAP=70.89  R1=80.73  R5=88.07  mINP=40.20
logs/agreidv2_baseline_4090_s2.log:332:    [G->A] mAP=70.82  R1=81.23  R5=87.36  mINP=32.87
logs/agreidv2_baseline_4090_s2.log:333:    [mean] mAP=70.85  R1=80.98
logs/agreidv2_baseline_4090_s2.log:334:Epoch[21] Iter[50/786] Loss: 1.385 CE: 1.304 Tri: 0.080 Acc: 0.970 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:335:Epoch[21] Iter[100/786] Loss: 1.374 CE: 1.293 Tri: 0.081 Acc: 0.968 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:336:Epoch[21] Iter[150/786] Loss: 1.352 CE: 1.275 Tri: 0.078 Acc: 0.972 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:337:Epoch[21] Iter[200/786] Loss: 1.337 CE: 1.263 Tri: 0.073 Acc: 0.973 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:338:Epoch[21] Iter[250/786] Loss: 1.326 CE: 1.256 Tri: 0.070 Acc: 0.975 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:339:Epoch[21] Iter[300/786] Loss: 1.315 CE: 1.248 Tri: 0.066 Acc: 0.976 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:340:Epoch[21] Iter[350/786] Loss: 1.311 CE: 1.244 Tri: 0.067 Acc: 0.977 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:341:Epoch[21] Iter[400/786] Loss: 1.304 CE: 1.239 Tri: 0.065 Acc: 0.978 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:342:Epoch[21] Iter[450/786] Loss: 1.298 CE: 1.235 Tri: 0.064 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:343:Epoch[21] Iter[500/786] Loss: 1.295 CE: 1.231 Tri: 0.064 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:344:Epoch[21] Iter[550/786] Loss: 1.292 CE: 1.228 Tri: 0.064 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:345:Epoch[21] Iter[600/786] Loss: 1.287 CE: 1.225 Tri: 0.062 Acc: 0.980 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:346:Epoch[21] Iter[650/786] Loss: 1.282 CE: 1.221 Tri: 0.061 Acc: 0.981 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:347:Epoch[21] Iter[700/786] Loss: 1.274 CE: 1.215 Tri: 0.059 Acc: 0.982 LR: 3.17e-05
logs/agreidv2_baseline_4090_s2.log:348:Epoch[21] done in 91.4s  Loss=1.267 Acc=0.982
logs/agreidv2_baseline_4090_s2.log:349:Epoch[22] Iter[50/786] Loss: 1.354 CE: 1.271 Tri: 0.082 Acc: 0.974 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:350:Epoch[22] Iter[100/786] Loss: 1.341 CE: 1.257 Tri: 0.084 Acc: 0.976 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:351:Epoch[22] Iter[150/786] Loss: 1.322 CE: 1.244 Tri: 0.078 Acc: 0.979 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:352:Epoch[22] Iter[200/786] Loss: 1.313 CE: 1.241 Tri: 0.071 Acc: 0.978 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:353:Epoch[22] Iter[250/786] Loss: 1.301 CE: 1.235 Tri: 0.067 Acc: 0.978 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:354:Epoch[22] Iter[300/786] Loss: 1.297 CE: 1.230 Tri: 0.068 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:355:Epoch[22] Iter[350/786] Loss: 1.298 CE: 1.228 Tri: 0.070 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:356:Epoch[22] Iter[400/786] Loss: 1.295 CE: 1.227 Tri: 0.069 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:357:Epoch[22] Iter[450/786] Loss: 1.290 CE: 1.223 Tri: 0.067 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:358:Epoch[22] Iter[500/786] Loss: 1.288 CE: 1.221 Tri: 0.066 Acc: 0.981 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:359:Epoch[22] Iter[550/786] Loss: 1.282 CE: 1.217 Tri: 0.064 Acc: 0.981 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:360:Epoch[22] Iter[600/786] Loss: 1.277 CE: 1.214 Tri: 0.063 Acc: 0.982 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:361:Epoch[22] Iter[650/786] Loss: 1.271 CE: 1.210 Tri: 0.061 Acc: 0.983 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:362:Epoch[22] Iter[700/786] Loss: 1.267 CE: 1.206 Tri: 0.061 Acc: 0.983 LR: 3.10e-05
logs/agreidv2_baseline_4090_s2.log:363:Epoch[22] done in 91.6s  Loss=1.260 Acc=0.984
logs/agreidv2_baseline_4090_s2.log:364:Epoch[23] Iter[50/786] Loss: 1.351 CE: 1.285 Tri: 0.065 Acc: 0.971 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:365:Epoch[23] Iter[100/786] Loss: 1.327 CE: 1.257 Tri: 0.070 Acc: 0.975 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:366:Epoch[23] Iter[150/786] Loss: 1.310 CE: 1.248 Tri: 0.062 Acc: 0.975 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:367:Epoch[23] Iter[200/786] Loss: 1.303 CE: 1.241 Tri: 0.062 Acc: 0.976 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:368:Epoch[23] Iter[250/786] Loss: 1.296 CE: 1.235 Tri: 0.061 Acc: 0.977 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:369:Epoch[23] Iter[300/786] Loss: 1.292 CE: 1.230 Tri: 0.062 Acc: 0.978 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:370:Epoch[23] Iter[350/786] Loss: 1.288 CE: 1.227 Tri: 0.061 Acc: 0.979 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:371:Epoch[23] Iter[400/786] Loss: 1.285 CE: 1.224 Tri: 0.061 Acc: 0.979 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:372:Epoch[23] Iter[450/786] Loss: 1.280 CE: 1.220 Tri: 0.060 Acc: 0.980 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:373:Epoch[23] Iter[500/786] Loss: 1.275 CE: 1.216 Tri: 0.058 Acc: 0.981 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:374:Epoch[23] Iter[550/786] Loss: 1.270 CE: 1.213 Tri: 0.057 Acc: 0.981 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:375:Epoch[23] Iter[600/786] Loss: 1.267 CE: 1.210 Tri: 0.058 Acc: 0.982 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:376:Epoch[23] Iter[650/786] Loss: 1.264 CE: 1.207 Tri: 0.057 Acc: 0.982 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:377:Epoch[23] Iter[700/786] Loss: 1.259 CE: 1.203 Tri: 0.056 Acc: 0.983 LR: 3.03e-05
logs/agreidv2_baseline_4090_s2.log:378:Epoch[23] done in 91.7s  Loss=1.251 Acc=0.983
logs/agreidv2_baseline_4090_s2.log:379:Epoch[24] Iter[50/786] Loss: 1.341 CE: 1.269 Tri: 0.072 Acc: 0.973 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:380:Epoch[24] Iter[100/786] Loss: 1.319 CE: 1.250 Tri: 0.069 Acc: 0.975 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:381:Epoch[24] Iter[150/786] Loss: 1.308 CE: 1.240 Tri: 0.068 Acc: 0.977 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:382:Epoch[24] Iter[200/786] Loss: 1.291 CE: 1.230 Tri: 0.061 Acc: 0.978 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:383:Epoch[24] Iter[250/786] Loss: 1.286 CE: 1.226 Tri: 0.060 Acc: 0.979 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:384:Epoch[24] Iter[300/786] Loss: 1.280 CE: 1.221 Tri: 0.059 Acc: 0.980 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:385:Epoch[24] Iter[350/786] Loss: 1.274 CE: 1.217 Tri: 0.058 Acc: 0.980 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:386:Epoch[24] Iter[400/786] Loss: 1.267 CE: 1.211 Tri: 0.056 Acc: 0.981 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:387:Epoch[24] Iter[450/786] Loss: 1.262 CE: 1.208 Tri: 0.054 Acc: 0.981 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:388:Epoch[24] Iter[500/786] Loss: 1.259 CE: 1.205 Tri: 0.054 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:389:Epoch[24] Iter[550/786] Loss: 1.255 CE: 1.202 Tri: 0.053 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:390:Epoch[24] Iter[600/786] Loss: 1.253 CE: 1.200 Tri: 0.054 Acc: 0.983 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:391:Epoch[24] Iter[650/786] Loss: 1.249 CE: 1.197 Tri: 0.053 Acc: 0.983 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:392:Epoch[24] Iter[700/786] Loss: 1.244 CE: 1.192 Tri: 0.052 Acc: 0.984 LR: 2.95e-05
logs/agreidv2_baseline_4090_s2.log:393:Epoch[24] done in 92.1s  Loss=1.237 Acc=0.984
logs/agreidv2_baseline_4090_s2.log:394:Epoch[25] Iter[50/786] Loss: 1.325 CE: 1.258 Tri: 0.067 Acc: 0.975 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:395:Epoch[25] Iter[100/786] Loss: 1.305 CE: 1.237 Tri: 0.068 Acc: 0.980 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:396:Epoch[25] Iter[150/786] Loss: 1.286 CE: 1.222 Tri: 0.064 Acc: 0.981 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:397:Epoch[25] Iter[200/786] Loss: 1.281 CE: 1.218 Tri: 0.063 Acc: 0.981 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:398:Epoch[25] Iter[250/786] Loss: 1.273 CE: 1.211 Tri: 0.062 Acc: 0.982 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:399:Epoch[25] Iter[300/786] Loss: 1.264 CE: 1.207 Tri: 0.057 Acc: 0.982 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:400:Epoch[25] Iter[350/786] Loss: 1.259 CE: 1.203 Tri: 0.056 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:401:Epoch[25] Iter[400/786] Loss: 1.255 CE: 1.201 Tri: 0.054 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:402:Epoch[25] Iter[450/786] Loss: 1.251 CE: 1.198 Tri: 0.053 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:403:Epoch[25] Iter[500/786] Loss: 1.250 CE: 1.196 Tri: 0.054 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:404:Epoch[25] Iter[550/786] Loss: 1.247 CE: 1.194 Tri: 0.053 Acc: 0.984 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:405:Epoch[25] Iter[600/786] Loss: 1.244 CE: 1.191 Tri: 0.053 Acc: 0.984 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:406:Epoch[25] Iter[650/786] Loss: 1.238 CE: 1.187 Tri: 0.051 Acc: 0.985 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:407:Epoch[25] Iter[700/786] Loss: 1.231 CE: 1.182 Tri: 0.049 Acc: 0.985 LR: 2.87e-05
logs/agreidv2_baseline_4090_s2.log:408:Epoch[25] done in 91.4s  Loss=1.225 Acc=0.986
logs/agreidv2_baseline_4090_s2.log:409:Epoch[26] Iter[50/786] Loss: 1.291 CE: 1.236 Tri: 0.055 Acc: 0.978 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:410:Epoch[26] Iter[100/786] Loss: 1.271 CE: 1.221 Tri: 0.050 Acc: 0.980 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:411:Epoch[26] Iter[150/786] Loss: 1.260 CE: 1.210 Tri: 0.051 Acc: 0.981 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:412:Epoch[26] Iter[200/786] Loss: 1.251 CE: 1.201 Tri: 0.050 Acc: 0.982 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:413:Epoch[26] Iter[250/786] Loss: 1.248 CE: 1.199 Tri: 0.049 Acc: 0.982 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:414:Epoch[26] Iter[300/786] Loss: 1.245 CE: 1.196 Tri: 0.049 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:415:Epoch[26] Iter[350/786] Loss: 1.241 CE: 1.193 Tri: 0.048 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:416:Epoch[26] Iter[400/786] Loss: 1.237 CE: 1.190 Tri: 0.047 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:417:Epoch[26] Iter[450/786] Loss: 1.232 CE: 1.187 Tri: 0.045 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:418:Epoch[26] Iter[500/786] Loss: 1.230 CE: 1.185 Tri: 0.045 Acc: 0.985 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:419:Epoch[26] Iter[550/786] Loss: 1.225 CE: 1.182 Tri: 0.043 Acc: 0.985 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:420:Epoch[26] Iter[600/786] Loss: 1.220 CE: 1.179 Tri: 0.041 Acc: 0.985 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:421:Epoch[26] Iter[650/786] Loss: 1.217 CE: 1.176 Tri: 0.041 Acc: 0.986 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:422:Epoch[26] Iter[700/786] Loss: 1.211 CE: 1.172 Tri: 0.040 Acc: 0.986 LR: 2.78e-05
logs/agreidv2_baseline_4090_s2.log:423:Epoch[26] done in 91.7s  Loss=1.206 Acc=0.987
logs/agreidv2_baseline_4090_s2.log:424:Epoch[27] Iter[50/786] Loss: 1.279 CE: 1.235 Tri: 0.043 Acc: 0.974 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:425:Epoch[27] Iter[100/786] Loss: 1.248 CE: 1.208 Tri: 0.040 Acc: 0.980 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:426:Epoch[27] Iter[150/786] Loss: 1.233 CE: 1.194 Tri: 0.039 Acc: 0.983 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:427:Epoch[27] Iter[200/786] Loss: 1.230 CE: 1.189 Tri: 0.041 Acc: 0.983 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:428:Epoch[27] Iter[250/786] Loss: 1.225 CE: 1.185 Tri: 0.039 Acc: 0.984 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:429:Epoch[27] Iter[300/786] Loss: 1.221 CE: 1.181 Tri: 0.040 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:430:Epoch[27] Iter[350/786] Loss: 1.218 CE: 1.179 Tri: 0.039 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:431:Epoch[27] Iter[400/786] Loss: 1.216 CE: 1.177 Tri: 0.039 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:432:Epoch[27] Iter[450/786] Loss: 1.214 CE: 1.175 Tri: 0.039 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:433:Epoch[27] Iter[500/786] Loss: 1.213 CE: 1.174 Tri: 0.039 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:434:Epoch[27] Iter[550/786] Loss: 1.210 CE: 1.171 Tri: 0.039 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:435:Epoch[27] Iter[600/786] Loss: 1.206 CE: 1.168 Tri: 0.038 Acc: 0.987 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:436:Epoch[27] Iter[650/786] Loss: 1.205 CE: 1.166 Tri: 0.039 Acc: 0.987 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:437:Epoch[27] Iter[700/786] Loss: 1.201 CE: 1.163 Tri: 0.038 Acc: 0.987 LR: 2.69e-05
logs/agreidv2_baseline_4090_s2.log:438:Epoch[27] done in 91.9s  Loss=1.196 Acc=0.988
logs/agreidv2_baseline_4090_s2.log:439:Epoch[28] Iter[50/786] Loss: 1.276 CE: 1.230 Tri: 0.046 Acc: 0.977 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:440:Epoch[28] Iter[100/786] Loss: 1.245 CE: 1.206 Tri: 0.039 Acc: 0.983 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:441:Epoch[28] Iter[150/786] Loss: 1.243 CE: 1.201 Tri: 0.041 Acc: 0.981 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:442:Epoch[28] Iter[200/786] Loss: 1.233 CE: 1.193 Tri: 0.040 Acc: 0.983 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:443:Epoch[28] Iter[250/786] Loss: 1.225 CE: 1.186 Tri: 0.039 Acc: 0.984 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:444:Epoch[28] Iter[300/786] Loss: 1.220 CE: 1.182 Tri: 0.038 Acc: 0.984 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:445:Epoch[28] Iter[350/786] Loss: 1.220 CE: 1.180 Tri: 0.040 Acc: 0.984 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:446:Epoch[28] Iter[400/786] Loss: 1.217 CE: 1.177 Tri: 0.040 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:447:Epoch[28] Iter[450/786] Loss: 1.214 CE: 1.174 Tri: 0.039 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:448:Epoch[28] Iter[500/786] Loss: 1.212 CE: 1.172 Tri: 0.040 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:449:Epoch[28] Iter[550/786] Loss: 1.209 CE: 1.170 Tri: 0.039 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:450:Epoch[28] Iter[600/786] Loss: 1.205 CE: 1.167 Tri: 0.038 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:451:Epoch[28] Iter[650/786] Loss: 1.201 CE: 1.164 Tri: 0.037 Acc: 0.987 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:452:Epoch[28] Iter[700/786] Loss: 1.197 CE: 1.160 Tri: 0.036 Acc: 0.987 LR: 2.59e-05
logs/agreidv2_baseline_4090_s2.log:453:Epoch[28] done in 91.7s  Loss=1.191 Acc=0.988
logs/agreidv2_baseline_4090_s2.log:454:Epoch[29] Iter[50/786] Loss: 1.246 CE: 1.208 Tri: 0.039 Acc: 0.982 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:455:Epoch[29] Iter[100/786] Loss: 1.227 CE: 1.189 Tri: 0.038 Acc: 0.985 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:456:Epoch[29] Iter[150/786] Loss: 1.220 CE: 1.184 Tri: 0.037 Acc: 0.985 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:457:Epoch[29] Iter[200/786] Loss: 1.214 CE: 1.177 Tri: 0.037 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:458:Epoch[29] Iter[250/786] Loss: 1.209 CE: 1.174 Tri: 0.035 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:459:Epoch[29] Iter[300/786] Loss: 1.205 CE: 1.171 Tri: 0.034 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:460:Epoch[29] Iter[350/786] Loss: 1.199 CE: 1.167 Tri: 0.032 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:461:Epoch[29] Iter[400/786] Loss: 1.195 CE: 1.164 Tri: 0.031 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:462:Epoch[29] Iter[450/786] Loss: 1.191 CE: 1.161 Tri: 0.030 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:463:Epoch[29] Iter[500/786] Loss: 1.188 CE: 1.159 Tri: 0.029 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:464:Epoch[29] Iter[550/786] Loss: 1.183 CE: 1.156 Tri: 0.028 Acc: 0.988 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:465:Epoch[29] Iter[600/786] Loss: 1.181 CE: 1.153 Tri: 0.028 Acc: 0.988 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:466:Epoch[29] Iter[650/786] Loss: 1.177 CE: 1.150 Tri: 0.027 Acc: 0.989 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:467:Epoch[29] Iter[700/786] Loss: 1.173 CE: 1.147 Tri: 0.026 Acc: 0.989 LR: 2.50e-05
logs/agreidv2_baseline_4090_s2.log:468:Epoch[29] done in 91.1s  Loss=1.168 Acc=0.990
logs/agreidv2_baseline_4090_s2.log:469:Epoch[30] Iter[50/786] Loss: 1.243 CE: 1.200 Tri: 0.042 Acc: 0.975 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:470:Epoch[30] Iter[100/786] Loss: 1.220 CE: 1.181 Tri: 0.039 Acc: 0.980 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:471:Epoch[30] Iter[150/786] Loss: 1.205 CE: 1.169 Tri: 0.037 Acc: 0.984 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:472:Epoch[30] Iter[200/786] Loss: 1.199 CE: 1.164 Tri: 0.035 Acc: 0.985 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:473:Epoch[30] Iter[250/786] Loss: 1.196 CE: 1.160 Tri: 0.036 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:474:Epoch[30] Iter[300/786] Loss: 1.192 CE: 1.158 Tri: 0.034 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:475:Epoch[30] Iter[350/786] Loss: 1.188 CE: 1.155 Tri: 0.033 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:476:Epoch[30] Iter[400/786] Loss: 1.185 CE: 1.153 Tri: 0.033 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:477:Epoch[30] Iter[450/786] Loss: 1.182 CE: 1.150 Tri: 0.032 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:478:Epoch[30] Iter[500/786] Loss: 1.180 CE: 1.149 Tri: 0.032 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:479:Epoch[30] Iter[550/786] Loss: 1.177 CE: 1.146 Tri: 0.031 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:480:Epoch[30] Iter[600/786] Loss: 1.177 CE: 1.146 Tri: 0.032 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:481:Epoch[30] Iter[650/786] Loss: 1.175 CE: 1.143 Tri: 0.032 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:482:Epoch[30] Iter[700/786] Loss: 1.171 CE: 1.140 Tri: 0.031 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_baseline_4090_s2.log:483:Epoch[30] done in 92.1s  Loss=1.165 Acc=0.989
logs/agreidv2_baseline_4090_s2.log:485:    [A->G] mAP=72.48  R1=81.49  R5=89.05  mINP=44.27
logs/agreidv2_baseline_4090_s2.log:486:    [G->A] mAP=73.37  R1=82.44  R5=89.12  mINP=36.52
logs/agreidv2_baseline_4090_s2.log:487:    [mean] mAP=72.93  R1=81.97
logs/agreidv2_baseline_4090_s2.log:488:Epoch[31] Iter[50/786] Loss: 1.230 CE: 1.187 Tri: 0.043 Acc: 0.985 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:489:Epoch[31] Iter[100/786] Loss: 1.206 CE: 1.171 Tri: 0.035 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:490:Epoch[31] Iter[150/786] Loss: 1.195 CE: 1.163 Tri: 0.032 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:491:Epoch[31] Iter[200/786] Loss: 1.191 CE: 1.160 Tri: 0.032 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:492:Epoch[31] Iter[250/786] Loss: 1.185 CE: 1.155 Tri: 0.031 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:493:Epoch[31] Iter[300/786] Loss: 1.182 CE: 1.152 Tri: 0.030 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:494:Epoch[31] Iter[350/786] Loss: 1.179 CE: 1.150 Tri: 0.030 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:495:Epoch[31] Iter[400/786] Loss: 1.178 CE: 1.148 Tri: 0.030 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:496:Epoch[31] Iter[450/786] Loss: 1.174 CE: 1.145 Tri: 0.029 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:497:Epoch[31] Iter[500/786] Loss: 1.172 CE: 1.143 Tri: 0.029 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:498:Epoch[31] Iter[550/786] Loss: 1.169 CE: 1.140 Tri: 0.028 Acc: 0.990 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:499:Epoch[31] Iter[600/786] Loss: 1.168 CE: 1.139 Tri: 0.029 Acc: 0.990 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:500:Epoch[31] Iter[650/786] Loss: 1.165 CE: 1.137 Tri: 0.028 Acc: 0.990 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:501:Epoch[31] Iter[700/786] Loss: 1.161 CE: 1.134 Tri: 0.027 Acc: 0.990 LR: 2.29e-05
logs/agreidv2_baseline_4090_s2.log:502:Epoch[31] done in 90.9s  Loss=1.157 Acc=0.991
logs/agreidv2_baseline_4090_s2.log:503:Epoch[32] Iter[50/786] Loss: 1.212 CE: 1.179 Tri: 0.033 Acc: 0.981 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:504:Epoch[32] Iter[100/786] Loss: 1.195 CE: 1.166 Tri: 0.030 Acc: 0.982 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:505:Epoch[32] Iter[150/786] Loss: 1.184 CE: 1.156 Tri: 0.028 Acc: 0.985 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:506:Epoch[32] Iter[200/786] Loss: 1.178 CE: 1.150 Tri: 0.028 Acc: 0.986 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:507:Epoch[32] Iter[250/786] Loss: 1.174 CE: 1.146 Tri: 0.027 Acc: 0.986 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:508:Epoch[32] Iter[300/786] Loss: 1.173 CE: 1.145 Tri: 0.028 Acc: 0.987 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:509:Epoch[32] Iter[350/786] Loss: 1.169 CE: 1.142 Tri: 0.027 Acc: 0.988 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:510:Epoch[32] Iter[400/786] Loss: 1.167 CE: 1.139 Tri: 0.027 Acc: 0.988 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:511:Epoch[32] Iter[450/786] Loss: 1.166 CE: 1.138 Tri: 0.027 Acc: 0.988 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:512:Epoch[32] Iter[500/786] Loss: 1.163 CE: 1.136 Tri: 0.027 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:513:Epoch[32] Iter[550/786] Loss: 1.162 CE: 1.134 Tri: 0.027 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:514:Epoch[32] Iter[600/786] Loss: 1.159 CE: 1.132 Tri: 0.027 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:515:Epoch[32] Iter[650/786] Loss: 1.157 CE: 1.130 Tri: 0.026 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:516:Epoch[32] Iter[700/786] Loss: 1.153 CE: 1.128 Tri: 0.026 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_baseline_4090_s2.log:517:Epoch[32] done in 91.8s  Loss=1.149 Acc=0.990
logs/agreidv2_baseline_4090_s2.log:518:Epoch[33] Iter[50/786] Loss: 1.205 CE: 1.172 Tri: 0.034 Acc: 0.985 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:519:Epoch[33] Iter[100/786] Loss: 1.183 CE: 1.155 Tri: 0.028 Acc: 0.987 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:520:Epoch[33] Iter[150/786] Loss: 1.176 CE: 1.148 Tri: 0.027 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:521:Epoch[33] Iter[200/786] Loss: 1.168 CE: 1.143 Tri: 0.026 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:522:Epoch[33] Iter[250/786] Loss: 1.164 CE: 1.139 Tri: 0.025 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:523:Epoch[33] Iter[300/786] Loss: 1.161 CE: 1.136 Tri: 0.024 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:524:Epoch[33] Iter[350/786] Loss: 1.158 CE: 1.134 Tri: 0.024 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:525:Epoch[33] Iter[400/786] Loss: 1.155 CE: 1.132 Tri: 0.023 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:526:Epoch[33] Iter[450/786] Loss: 1.153 CE: 1.130 Tri: 0.022 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:527:Epoch[33] Iter[500/786] Loss: 1.153 CE: 1.130 Tri: 0.023 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:528:Epoch[33] Iter[550/786] Loss: 1.150 CE: 1.128 Tri: 0.023 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:529:Epoch[33] Iter[600/786] Loss: 1.148 CE: 1.126 Tri: 0.022 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:530:Epoch[33] Iter[650/786] Loss: 1.145 CE: 1.124 Tri: 0.022 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:531:Epoch[33] Iter[700/786] Loss: 1.141 CE: 1.120 Tri: 0.021 Acc: 0.991 LR: 2.08e-05
logs/agreidv2_baseline_4090_s2.log:532:Epoch[33] done in 91.4s  Loss=1.137 Acc=0.991
logs/agreidv2_baseline_4090_s2.log:533:Epoch[34] Iter[50/786] Loss: 1.159 CE: 1.143 Tri: 0.016 Acc: 0.988 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:534:Epoch[34] Iter[100/786] Loss: 1.154 CE: 1.136 Tri: 0.018 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:535:Epoch[34] Iter[150/786] Loss: 1.152 CE: 1.132 Tri: 0.020 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:536:Epoch[34] Iter[200/786] Loss: 1.147 CE: 1.127 Tri: 0.020 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:537:Epoch[34] Iter[250/786] Loss: 1.146 CE: 1.126 Tri: 0.020 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:538:Epoch[34] Iter[300/786] Loss: 1.145 CE: 1.124 Tri: 0.021 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:539:Epoch[34] Iter[350/786] Loss: 1.144 CE: 1.123 Tri: 0.021 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:540:Epoch[34] Iter[400/786] Loss: 1.142 CE: 1.122 Tri: 0.020 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:541:Epoch[34] Iter[450/786] Loss: 1.141 CE: 1.121 Tri: 0.020 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:542:Epoch[34] Iter[500/786] Loss: 1.140 CE: 1.120 Tri: 0.020 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:543:Epoch[34] Iter[550/786] Loss: 1.138 CE: 1.119 Tri: 0.019 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:544:Epoch[34] Iter[600/786] Loss: 1.137 CE: 1.117 Tri: 0.020 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:545:Epoch[34] Iter[650/786] Loss: 1.135 CE: 1.116 Tri: 0.020 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:546:Epoch[34] Iter[700/786] Loss: 1.133 CE: 1.114 Tri: 0.019 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline_4090_s2.log:547:Epoch[34] done in 91.2s  Loss=1.131 Acc=0.992
logs/agreidv2_baseline_4090_s2.log:548:Epoch[35] Iter[50/786] Loss: 1.178 CE: 1.153 Tri: 0.025 Acc: 0.985 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:549:Epoch[35] Iter[100/786] Loss: 1.168 CE: 1.141 Tri: 0.026 Acc: 0.988 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:550:Epoch[35] Iter[150/786] Loss: 1.158 CE: 1.135 Tri: 0.023 Acc: 0.989 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:551:Epoch[35] Iter[200/786] Loss: 1.159 CE: 1.134 Tri: 0.025 Acc: 0.989 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:552:Epoch[35] Iter[250/786] Loss: 1.155 CE: 1.131 Tri: 0.024 Acc: 0.989 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:553:Epoch[35] Iter[300/786] Loss: 1.152 CE: 1.129 Tri: 0.024 Acc: 0.990 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:554:Epoch[35] Iter[350/786] Loss: 1.149 CE: 1.126 Tri: 0.023 Acc: 0.990 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:555:Epoch[35] Iter[400/786] Loss: 1.147 CE: 1.124 Tri: 0.023 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:556:Epoch[35] Iter[450/786] Loss: 1.145 CE: 1.122 Tri: 0.023 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:557:Epoch[35] Iter[500/786] Loss: 1.141 CE: 1.120 Tri: 0.021 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:558:Epoch[35] Iter[550/786] Loss: 1.139 CE: 1.118 Tri: 0.021 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:559:Epoch[35] Iter[600/786] Loss: 1.136 CE: 1.116 Tri: 0.020 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:560:Epoch[35] Iter[650/786] Loss: 1.134 CE: 1.114 Tri: 0.020 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:561:Epoch[35] Iter[700/786] Loss: 1.131 CE: 1.111 Tri: 0.020 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_baseline_4090_s2.log:562:Epoch[35] done in 91.2s  Loss=1.127 Acc=0.993
logs/agreidv2_baseline_4090_s2.log:563:Epoch[36] Iter[50/786] Loss: 1.163 CE: 1.139 Tri: 0.024 Acc: 0.987 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:564:Epoch[36] Iter[100/786] Loss: 1.145 CE: 1.126 Tri: 0.019 Acc: 0.989 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:565:Epoch[36] Iter[150/786] Loss: 1.143 CE: 1.121 Tri: 0.022 Acc: 0.990 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:566:Epoch[36] Iter[200/786] Loss: 1.139 CE: 1.118 Tri: 0.021 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:567:Epoch[36] Iter[250/786] Loss: 1.139 CE: 1.117 Tri: 0.021 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:568:Epoch[36] Iter[300/786] Loss: 1.136 CE: 1.115 Tri: 0.021 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:569:Epoch[36] Iter[350/786] Loss: 1.134 CE: 1.114 Tri: 0.021 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:570:Epoch[36] Iter[400/786] Loss: 1.133 CE: 1.112 Tri: 0.021 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:571:Epoch[36] Iter[450/786] Loss: 1.131 CE: 1.111 Tri: 0.020 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:572:Epoch[36] Iter[500/786] Loss: 1.129 CE: 1.110 Tri: 0.019 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:573:Epoch[36] Iter[550/786] Loss: 1.129 CE: 1.109 Tri: 0.020 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:574:Epoch[36] Iter[600/786] Loss: 1.128 CE: 1.108 Tri: 0.020 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:575:Epoch[36] Iter[650/786] Loss: 1.125 CE: 1.106 Tri: 0.019 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:576:Epoch[36] Iter[700/786] Loss: 1.122 CE: 1.104 Tri: 0.019 Acc: 0.993 LR: 1.75e-05
logs/agreidv2_baseline_4090_s2.log:577:Epoch[36] done in 91.7s  Loss=1.118 Acc=0.993
logs/agreidv2_baseline_4090_s2.log:578:Epoch[37] Iter[50/786] Loss: 1.150 CE: 1.131 Tri: 0.019 Acc: 0.989 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:579:Epoch[37] Iter[100/786] Loss: 1.137 CE: 1.120 Tri: 0.017 Acc: 0.990 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:580:Epoch[37] Iter[150/786] Loss: 1.134 CE: 1.116 Tri: 0.018 Acc: 0.990 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:581:Epoch[37] Iter[200/786] Loss: 1.134 CE: 1.114 Tri: 0.020 Acc: 0.990 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:582:Epoch[37] Iter[250/786] Loss: 1.131 CE: 1.112 Tri: 0.018 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:583:Epoch[37] Iter[300/786] Loss: 1.128 CE: 1.110 Tri: 0.018 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:584:Epoch[37] Iter[350/786] Loss: 1.126 CE: 1.108 Tri: 0.018 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:585:Epoch[37] Iter[400/786] Loss: 1.125 CE: 1.108 Tri: 0.017 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:586:Epoch[37] Iter[450/786] Loss: 1.123 CE: 1.107 Tri: 0.017 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:587:Epoch[37] Iter[500/786] Loss: 1.121 CE: 1.105 Tri: 0.016 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:588:Epoch[37] Iter[550/786] Loss: 1.120 CE: 1.104 Tri: 0.016 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:589:Epoch[37] Iter[600/786] Loss: 1.119 CE: 1.102 Tri: 0.016 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:590:Epoch[37] Iter[650/786] Loss: 1.117 CE: 1.101 Tri: 0.017 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:591:Epoch[37] Iter[700/786] Loss: 1.115 CE: 1.099 Tri: 0.016 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_baseline_4090_s2.log:592:Epoch[37] done in 92.1s  Loss=1.111 Acc=0.993
logs/agreidv2_baseline_4090_s2.log:593:Epoch[38] Iter[50/786] Loss: 1.124 CE: 1.111 Tri: 0.014 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:594:Epoch[38] Iter[100/786] Loss: 1.129 CE: 1.110 Tri: 0.019 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:595:Epoch[38] Iter[150/786] Loss: 1.122 CE: 1.106 Tri: 0.015 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:596:Epoch[38] Iter[200/786] Loss: 1.124 CE: 1.106 Tri: 0.017 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:597:Epoch[38] Iter[250/786] Loss: 1.123 CE: 1.106 Tri: 0.017 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:598:Epoch[38] Iter[300/786] Loss: 1.125 CE: 1.105 Tri: 0.019 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:599:Epoch[38] Iter[350/786] Loss: 1.123 CE: 1.104 Tri: 0.018 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:600:Epoch[38] Iter[400/786] Loss: 1.122 CE: 1.104 Tri: 0.018 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:601:Epoch[38] Iter[450/786] Loss: 1.119 CE: 1.102 Tri: 0.017 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:602:Epoch[38] Iter[500/786] Loss: 1.118 CE: 1.101 Tri: 0.017 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:603:Epoch[38] Iter[550/786] Loss: 1.115 CE: 1.099 Tri: 0.016 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:604:Epoch[38] Iter[600/786] Loss: 1.113 CE: 1.097 Tri: 0.016 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:605:Epoch[38] Iter[650/786] Loss: 1.111 CE: 1.096 Tri: 0.015 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:606:Epoch[38] Iter[700/786] Loss: 1.108 CE: 1.093 Tri: 0.015 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090_s2.log:607:Epoch[38] done in 91.6s  Loss=1.105 Acc=0.993
logs/agreidv2_baseline_4090_s2.log:608:Epoch[39] Iter[50/786] Loss: 1.133 CE: 1.114 Tri: 0.019 Acc: 0.991 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:609:Epoch[39] Iter[100/786] Loss: 1.133 CE: 1.112 Tri: 0.021 Acc: 0.991 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:610:Epoch[39] Iter[150/786] Loss: 1.124 CE: 1.107 Tri: 0.017 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:611:Epoch[39] Iter[200/786] Loss: 1.119 CE: 1.103 Tri: 0.016 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:612:Epoch[39] Iter[250/786] Loss: 1.116 CE: 1.100 Tri: 0.016 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:613:Epoch[39] Iter[300/786] Loss: 1.115 CE: 1.100 Tri: 0.015 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:614:Epoch[39] Iter[350/786] Loss: 1.113 CE: 1.098 Tri: 0.015 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:615:Epoch[39] Iter[400/786] Loss: 1.112 CE: 1.097 Tri: 0.015 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:616:Epoch[39] Iter[450/786] Loss: 1.109 CE: 1.095 Tri: 0.014 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:617:Epoch[39] Iter[500/786] Loss: 1.108 CE: 1.094 Tri: 0.014 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:618:Epoch[39] Iter[550/786] Loss: 1.107 CE: 1.093 Tri: 0.015 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:619:Epoch[39] Iter[600/786] Loss: 1.106 CE: 1.091 Tri: 0.014 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:620:Epoch[39] Iter[650/786] Loss: 1.103 CE: 1.089 Tri: 0.014 Acc: 0.994 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:621:Epoch[39] Iter[700/786] Loss: 1.102 CE: 1.088 Tri: 0.014 Acc: 0.994 LR: 1.42e-05
logs/agreidv2_baseline_4090_s2.log:622:Epoch[39] done in 91.3s  Loss=1.099 Acc=0.994
logs/agreidv2_baseline_4090_s2.log:623:Epoch[40] Iter[50/786] Loss: 1.114 CE: 1.103 Tri: 0.011 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:624:Epoch[40] Iter[100/786] Loss: 1.111 CE: 1.101 Tri: 0.010 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:625:Epoch[40] Iter[150/786] Loss: 1.110 CE: 1.097 Tri: 0.012 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:626:Epoch[40] Iter[200/786] Loss: 1.108 CE: 1.095 Tri: 0.013 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:627:Epoch[40] Iter[250/786] Loss: 1.107 CE: 1.094 Tri: 0.013 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:628:Epoch[40] Iter[300/786] Loss: 1.107 CE: 1.093 Tri: 0.014 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:629:Epoch[40] Iter[350/786] Loss: 1.105 CE: 1.092 Tri: 0.014 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:630:Epoch[40] Iter[400/786] Loss: 1.103 CE: 1.090 Tri: 0.013 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:631:Epoch[40] Iter[450/786] Loss: 1.102 CE: 1.089 Tri: 0.013 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:632:Epoch[40] Iter[500/786] Loss: 1.103 CE: 1.089 Tri: 0.014 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:633:Epoch[40] Iter[550/786] Loss: 1.102 CE: 1.088 Tri: 0.014 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:634:Epoch[40] Iter[600/786] Loss: 1.101 CE: 1.087 Tri: 0.013 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:635:Epoch[40] Iter[650/786] Loss: 1.099 CE: 1.086 Tri: 0.013 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:636:Epoch[40] Iter[700/786] Loss: 1.097 CE: 1.085 Tri: 0.013 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline_4090_s2.log:637:Epoch[40] done in 91.4s  Loss=1.094 Acc=0.994
logs/agreidv2_baseline_4090_s2.log:639:    [A->G] mAP=76.85  R1=84.76  R5=91.26  mINP=49.81
logs/agreidv2_baseline_4090_s2.log:640:    [G->A] mAP=76.72  R1=85.04  R5=90.17  mINP=42.94
logs/agreidv2_baseline_4090_s2.log:641:    [mean] mAP=76.79  R1=84.90
logs/agreidv2_baseline_4090_s2.log:642:    * new best mean mAP=76.79 (epoch 40) saved
logs/agreidv2_baseline_4090_s2.log:643:Epoch[41] Iter[50/786] Loss: 1.114 CE: 1.102 Tri: 0.012 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:644:Epoch[41] Iter[100/786] Loss: 1.110 CE: 1.096 Tri: 0.014 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:645:Epoch[41] Iter[150/786] Loss: 1.108 CE: 1.093 Tri: 0.015 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:646:Epoch[41] Iter[200/786] Loss: 1.106 CE: 1.093 Tri: 0.013 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:647:Epoch[41] Iter[250/786] Loss: 1.105 CE: 1.092 Tri: 0.013 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:648:Epoch[41] Iter[300/786] Loss: 1.103 CE: 1.090 Tri: 0.013 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:649:Epoch[41] Iter[350/786] Loss: 1.101 CE: 1.090 Tri: 0.012 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:650:Epoch[41] Iter[400/786] Loss: 1.101 CE: 1.089 Tri: 0.012 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:651:Epoch[41] Iter[450/786] Loss: 1.100 CE: 1.088 Tri: 0.012 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:652:Epoch[41] Iter[500/786] Loss: 1.099 CE: 1.087 Tri: 0.012 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:653:Epoch[41] Iter[550/786] Loss: 1.097 CE: 1.085 Tri: 0.012 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:654:Epoch[41] Iter[600/786] Loss: 1.095 CE: 1.083 Tri: 0.011 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:655:Epoch[41] Iter[650/786] Loss: 1.093 CE: 1.082 Tri: 0.011 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:656:Epoch[41] Iter[700/786] Loss: 1.091 CE: 1.080 Tri: 0.010 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090_s2.log:657:Epoch[41] done in 91.0s  Loss=1.089 Acc=0.995
logs/agreidv2_baseline_4090_s2.log:658:Epoch[42] Iter[50/786] Loss: 1.112 CE: 1.100 Tri: 0.012 Acc: 0.990 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:659:Epoch[42] Iter[100/786] Loss: 1.097 CE: 1.088 Tri: 0.009 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:660:Epoch[42] Iter[150/786] Loss: 1.098 CE: 1.089 Tri: 0.009 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:661:Epoch[42] Iter[200/786] Loss: 1.097 CE: 1.087 Tri: 0.010 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:662:Epoch[42] Iter[250/786] Loss: 1.096 CE: 1.086 Tri: 0.010 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:663:Epoch[42] Iter[300/786] Loss: 1.097 CE: 1.085 Tri: 0.012 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:664:Epoch[42] Iter[350/786] Loss: 1.097 CE: 1.084 Tri: 0.013 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:665:Epoch[42] Iter[400/786] Loss: 1.096 CE: 1.083 Tri: 0.013 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:666:Epoch[42] Iter[450/786] Loss: 1.096 CE: 1.083 Tri: 0.013 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:667:Epoch[42] Iter[500/786] Loss: 1.095 CE: 1.082 Tri: 0.012 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:668:Epoch[42] Iter[550/786] Loss: 1.093 CE: 1.081 Tri: 0.012 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:669:Epoch[42] Iter[600/786] Loss: 1.092 CE: 1.080 Tri: 0.012 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:670:Epoch[42] Iter[650/786] Loss: 1.090 CE: 1.078 Tri: 0.012 Acc: 0.995 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:671:Epoch[42] Iter[700/786] Loss: 1.088 CE: 1.077 Tri: 0.011 Acc: 0.995 LR: 1.11e-05
logs/agreidv2_baseline_4090_s2.log:672:Epoch[42] done in 91.7s  Loss=1.085 Acc=0.995
logs/agreidv2_baseline_4090_s2.log:673:Epoch[43] Iter[50/786] Loss: 1.095 CE: 1.085 Tri: 0.009 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:674:Epoch[43] Iter[100/786] Loss: 1.095 CE: 1.082 Tri: 0.013 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:675:Epoch[43] Iter[150/786] Loss: 1.100 CE: 1.084 Tri: 0.016 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:676:Epoch[43] Iter[200/786] Loss: 1.094 CE: 1.081 Tri: 0.013 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:677:Epoch[43] Iter[250/786] Loss: 1.091 CE: 1.080 Tri: 0.011 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:678:Epoch[43] Iter[300/786] Loss: 1.091 CE: 1.080 Tri: 0.011 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:679:Epoch[43] Iter[350/786] Loss: 1.090 CE: 1.080 Tri: 0.011 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:680:Epoch[43] Iter[400/786] Loss: 1.090 CE: 1.079 Tri: 0.011 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:681:Epoch[43] Iter[450/786] Loss: 1.087 CE: 1.078 Tri: 0.010 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:682:Epoch[43] Iter[500/786] Loss: 1.086 CE: 1.077 Tri: 0.009 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:683:Epoch[43] Iter[550/786] Loss: 1.085 CE: 1.075 Tri: 0.010 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:684:Epoch[43] Iter[600/786] Loss: 1.084 CE: 1.075 Tri: 0.009 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:685:Epoch[43] Iter[650/786] Loss: 1.082 CE: 1.073 Tri: 0.009 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:686:Epoch[43] Iter[700/786] Loss: 1.080 CE: 1.072 Tri: 0.009 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090_s2.log:687:Epoch[43] done in 92.0s  Loss=1.078 Acc=0.995
logs/agreidv2_baseline_4090_s2.log:688:Epoch[44] Iter[50/786] Loss: 1.090 CE: 1.082 Tri: 0.008 Acc: 0.996 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:689:Epoch[44] Iter[100/786] Loss: 1.090 CE: 1.082 Tri: 0.007 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:690:Epoch[44] Iter[150/786] Loss: 1.093 CE: 1.084 Tri: 0.009 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:691:Epoch[44] Iter[200/786] Loss: 1.091 CE: 1.082 Tri: 0.009 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:692:Epoch[44] Iter[250/786] Loss: 1.089 CE: 1.080 Tri: 0.009 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:693:Epoch[44] Iter[300/786] Loss: 1.086 CE: 1.078 Tri: 0.009 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:694:Epoch[44] Iter[350/786] Loss: 1.086 CE: 1.077 Tri: 0.009 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:695:Epoch[44] Iter[400/786] Loss: 1.085 CE: 1.076 Tri: 0.009 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:696:Epoch[44] Iter[450/786] Loss: 1.084 CE: 1.075 Tri: 0.009 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:697:Epoch[44] Iter[500/786] Loss: 1.082 CE: 1.073 Tri: 0.009 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:698:Epoch[44] Iter[550/786] Loss: 1.081 CE: 1.072 Tri: 0.008 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:699:Epoch[44] Iter[600/786] Loss: 1.079 CE: 1.071 Tri: 0.008 Acc: 0.996 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:700:Epoch[44] Iter[650/786] Loss: 1.078 CE: 1.070 Tri: 0.008 Acc: 0.996 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:701:Epoch[44] Iter[700/786] Loss: 1.076 CE: 1.068 Tri: 0.008 Acc: 0.996 LR: 9.07e-06
logs/agreidv2_baseline_4090_s2.log:702:Epoch[44] done in 91.7s  Loss=1.074 Acc=0.996
logs/agreidv2_baseline_4090_s2.log:703:Epoch[45] Iter[50/786] Loss: 1.093 CE: 1.081 Tri: 0.011 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:704:Epoch[45] Iter[100/786] Loss: 1.090 CE: 1.079 Tri: 0.011 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:705:Epoch[45] Iter[150/786] Loss: 1.091 CE: 1.078 Tri: 0.013 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:706:Epoch[45] Iter[200/786] Loss: 1.090 CE: 1.078 Tri: 0.011 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:707:Epoch[45] Iter[250/786] Loss: 1.087 CE: 1.077 Tri: 0.011 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:708:Epoch[45] Iter[300/786] Loss: 1.086 CE: 1.076 Tri: 0.010 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:709:Epoch[45] Iter[350/786] Loss: 1.085 CE: 1.075 Tri: 0.011 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:710:Epoch[45] Iter[400/786] Loss: 1.083 CE: 1.073 Tri: 0.010 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:711:Epoch[45] Iter[450/786] Loss: 1.081 CE: 1.072 Tri: 0.009 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:712:Epoch[45] Iter[500/786] Loss: 1.080 CE: 1.071 Tri: 0.009 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:713:Epoch[45] Iter[550/786] Loss: 1.080 CE: 1.070 Tri: 0.009 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:714:Epoch[45] Iter[600/786] Loss: 1.078 CE: 1.069 Tri: 0.009 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:715:Epoch[45] Iter[650/786] Loss: 1.077 CE: 1.068 Tri: 0.009 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:716:Epoch[45] Iter[700/786] Loss: 1.075 CE: 1.067 Tri: 0.009 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline_4090_s2.log:717:Epoch[45] done in 91.4s  Loss=1.073 Acc=0.996
logs/agreidv2_baseline_4090_s2.log:718:Epoch[46] Iter[50/786] Loss: 1.098 CE: 1.085 Tri: 0.013 Acc: 0.993 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:719:Epoch[46] Iter[100/786] Loss: 1.089 CE: 1.078 Tri: 0.011 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:720:Epoch[46] Iter[150/786] Loss: 1.084 CE: 1.074 Tri: 0.010 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:721:Epoch[46] Iter[200/786] Loss: 1.082 CE: 1.073 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:722:Epoch[46] Iter[250/786] Loss: 1.080 CE: 1.071 Tri: 0.009 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:723:Epoch[46] Iter[300/786] Loss: 1.079 CE: 1.070 Tri: 0.009 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:724:Epoch[46] Iter[350/786] Loss: 1.079 CE: 1.070 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:725:Epoch[46] Iter[400/786] Loss: 1.078 CE: 1.069 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:726:Epoch[46] Iter[450/786] Loss: 1.077 CE: 1.068 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:727:Epoch[46] Iter[500/786] Loss: 1.077 CE: 1.068 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:728:Epoch[46] Iter[550/786] Loss: 1.076 CE: 1.067 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:729:Epoch[46] Iter[600/786] Loss: 1.074 CE: 1.066 Tri: 0.009 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:730:Epoch[46] Iter[650/786] Loss: 1.073 CE: 1.064 Tri: 0.008 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:731:Epoch[46] Iter[700/786] Loss: 1.071 CE: 1.063 Tri: 0.008 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_baseline_4090_s2.log:732:Epoch[46] done in 91.5s  Loss=1.069 Acc=0.996
logs/agreidv2_baseline_4090_s2.log:733:Epoch[47] Iter[50/786] Loss: 1.085 CE: 1.078 Tri: 0.007 Acc: 0.993 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:734:Epoch[47] Iter[100/786] Loss: 1.086 CE: 1.079 Tri: 0.007 Acc: 0.992 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:735:Epoch[47] Iter[150/786] Loss: 1.084 CE: 1.076 Tri: 0.007 Acc: 0.993 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:736:Epoch[47] Iter[200/786] Loss: 1.084 CE: 1.075 Tri: 0.009 Acc: 0.994 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:737:Epoch[47] Iter[250/786] Loss: 1.080 CE: 1.071 Tri: 0.009 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:738:Epoch[47] Iter[300/786] Loss: 1.077 CE: 1.069 Tri: 0.008 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:739:Epoch[47] Iter[350/786] Loss: 1.077 CE: 1.069 Tri: 0.008 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:740:Epoch[47] Iter[400/786] Loss: 1.076 CE: 1.068 Tri: 0.008 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:741:Epoch[47] Iter[450/786] Loss: 1.076 CE: 1.067 Tri: 0.009 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:742:Epoch[47] Iter[500/786] Loss: 1.074 CE: 1.065 Tri: 0.009 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:743:Epoch[47] Iter[550/786] Loss: 1.073 CE: 1.065 Tri: 0.008 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:744:Epoch[47] Iter[600/786] Loss: 1.072 CE: 1.064 Tri: 0.008 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:745:Epoch[47] Iter[650/786] Loss: 1.070 CE: 1.063 Tri: 0.008 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:746:Epoch[47] Iter[700/786] Loss: 1.069 CE: 1.061 Tri: 0.008 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline_4090_s2.log:747:Epoch[47] done in 90.9s  Loss=1.068 Acc=0.996
logs/agreidv2_baseline_4090_s2.log:748:Epoch[48] Iter[50/786] Loss: 1.083 CE: 1.075 Tri: 0.008 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:749:Epoch[48] Iter[100/786] Loss: 1.079 CE: 1.071 Tri: 0.007 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:750:Epoch[48] Iter[150/786] Loss: 1.077 CE: 1.069 Tri: 0.007 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:751:Epoch[48] Iter[200/786] Loss: 1.076 CE: 1.068 Tri: 0.007 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:752:Epoch[48] Iter[250/786] Loss: 1.076 CE: 1.068 Tri: 0.008 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:753:Epoch[48] Iter[300/786] Loss: 1.075 CE: 1.067 Tri: 0.008 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:754:Epoch[48] Iter[350/786] Loss: 1.073 CE: 1.066 Tri: 0.007 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:755:Epoch[48] Iter[400/786] Loss: 1.072 CE: 1.065 Tri: 0.007 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:756:Epoch[48] Iter[450/786] Loss: 1.071 CE: 1.064 Tri: 0.007 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:757:Epoch[48] Iter[500/786] Loss: 1.070 CE: 1.063 Tri: 0.007 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:758:Epoch[48] Iter[550/786] Loss: 1.068 CE: 1.061 Tri: 0.007 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:759:Epoch[48] Iter[600/786] Loss: 1.067 CE: 1.060 Tri: 0.006 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:760:Epoch[48] Iter[650/786] Loss: 1.066 CE: 1.059 Tri: 0.006 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:761:Epoch[48] Iter[700/786] Loss: 1.064 CE: 1.058 Tri: 0.006 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_baseline_4090_s2.log:762:Epoch[48] done in 90.8s  Loss=1.062 Acc=0.997
logs/agreidv2_baseline_4090_s2.log:763:Epoch[49] Iter[50/786] Loss: 1.075 CE: 1.068 Tri: 0.007 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:764:Epoch[49] Iter[100/786] Loss: 1.073 CE: 1.065 Tri: 0.008 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:765:Epoch[49] Iter[150/786] Loss: 1.072 CE: 1.065 Tri: 0.007 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:766:Epoch[49] Iter[200/786] Loss: 1.070 CE: 1.064 Tri: 0.007 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:767:Epoch[49] Iter[250/786] Loss: 1.069 CE: 1.062 Tri: 0.007 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:768:Epoch[49] Iter[300/786] Loss: 1.070 CE: 1.062 Tri: 0.008 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:769:Epoch[49] Iter[350/786] Loss: 1.068 CE: 1.061 Tri: 0.007 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:770:Epoch[49] Iter[400/786] Loss: 1.068 CE: 1.061 Tri: 0.007 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:771:Epoch[49] Iter[450/786] Loss: 1.067 CE: 1.060 Tri: 0.007 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:772:Epoch[49] Iter[500/786] Loss: 1.067 CE: 1.060 Tri: 0.007 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:773:Epoch[49] Iter[550/786] Loss: 1.066 CE: 1.059 Tri: 0.006 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:774:Epoch[49] Iter[600/786] Loss: 1.065 CE: 1.059 Tri: 0.006 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:775:Epoch[49] Iter[650/786] Loss: 1.064 CE: 1.058 Tri: 0.006 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:776:Epoch[49] Iter[700/786] Loss: 1.063 CE: 1.057 Tri: 0.006 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090_s2.log:777:Epoch[49] done in 90.9s  Loss=1.061 Acc=0.997
logs/agreidv2_baseline_4090_s2.log:778:Epoch[50] Iter[50/786] Loss: 1.072 CE: 1.066 Tri: 0.007 Acc: 0.994 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:779:Epoch[50] Iter[100/786] Loss: 1.076 CE: 1.067 Tri: 0.009 Acc: 0.994 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:780:Epoch[50] Iter[150/786] Loss: 1.073 CE: 1.066 Tri: 0.007 Acc: 0.994 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:781:Epoch[50] Iter[200/786] Loss: 1.071 CE: 1.064 Tri: 0.006 Acc: 0.994 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:782:Epoch[50] Iter[250/786] Loss: 1.069 CE: 1.063 Tri: 0.007 Acc: 0.995 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:783:Epoch[50] Iter[300/786] Loss: 1.067 CE: 1.061 Tri: 0.006 Acc: 0.995 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:784:Epoch[50] Iter[350/786] Loss: 1.066 CE: 1.060 Tri: 0.006 Acc: 0.995 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:785:Epoch[50] Iter[400/786] Loss: 1.066 CE: 1.060 Tri: 0.006 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:786:Epoch[50] Iter[450/786] Loss: 1.065 CE: 1.059 Tri: 0.006 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:787:Epoch[50] Iter[500/786] Loss: 1.064 CE: 1.058 Tri: 0.006 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:788:Epoch[50] Iter[550/786] Loss: 1.063 CE: 1.057 Tri: 0.006 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:789:Epoch[50] Iter[600/786] Loss: 1.062 CE: 1.056 Tri: 0.006 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:790:Epoch[50] Iter[650/786] Loss: 1.060 CE: 1.055 Tri: 0.005 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:791:Epoch[50] Iter[700/786] Loss: 1.059 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090_s2.log:792:Epoch[50] done in 91.7s  Loss=1.057 Acc=0.997
logs/agreidv2_baseline_4090_s2.log:794:    [A->G] mAP=79.31  R1=86.67  R5=91.68  mINP=54.35
logs/agreidv2_baseline_4090_s2.log:795:    [G->A] mAP=79.77  R1=87.69  R5=92.10  mINP=46.88
logs/agreidv2_baseline_4090_s2.log:796:    [mean] mAP=79.54  R1=87.18
logs/agreidv2_baseline_4090_s2.log:797:    * new best mean mAP=79.54 (epoch 50) saved
logs/agreidv2_baseline_4090_s2.log:798:Epoch[51] Iter[50/786] Loss: 1.066 CE: 1.060 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:799:Epoch[51] Iter[100/786] Loss: 1.067 CE: 1.060 Tri: 0.007 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:800:Epoch[51] Iter[150/786] Loss: 1.067 CE: 1.060 Tri: 0.007 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:801:Epoch[51] Iter[200/786] Loss: 1.066 CE: 1.060 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:802:Epoch[51] Iter[250/786] Loss: 1.065 CE: 1.059 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:803:Epoch[51] Iter[300/786] Loss: 1.063 CE: 1.058 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:804:Epoch[51] Iter[350/786] Loss: 1.062 CE: 1.057 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:805:Epoch[51] Iter[400/786] Loss: 1.062 CE: 1.056 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:806:Epoch[51] Iter[450/786] Loss: 1.061 CE: 1.055 Tri: 0.005 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:807:Epoch[51] Iter[500/786] Loss: 1.060 CE: 1.055 Tri: 0.005 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:808:Epoch[51] Iter[550/786] Loss: 1.059 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:809:Epoch[51] Iter[600/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.997 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:810:Epoch[51] Iter[650/786] Loss: 1.057 CE: 1.053 Tri: 0.005 Acc: 0.997 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:811:Epoch[51] Iter[700/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 3.34e-06
logs/agreidv2_baseline_4090_s2.log:812:Epoch[51] done in 91.4s  Loss=1.055 Acc=0.997
logs/agreidv2_baseline_4090_s2.log:813:Epoch[52] Iter[50/786] Loss: 1.065 CE: 1.058 Tri: 0.007 Acc: 0.995 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:814:Epoch[52] Iter[100/786] Loss: 1.068 CE: 1.058 Tri: 0.009 Acc: 0.995 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:815:Epoch[52] Iter[150/786] Loss: 1.065 CE: 1.058 Tri: 0.007 Acc: 0.995 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:816:Epoch[52] Iter[200/786] Loss: 1.061 CE: 1.056 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:817:Epoch[52] Iter[250/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:818:Epoch[52] Iter[300/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:819:Epoch[52] Iter[350/786] Loss: 1.060 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:820:Epoch[52] Iter[400/786] Loss: 1.059 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:821:Epoch[52] Iter[450/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:822:Epoch[52] Iter[500/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:823:Epoch[52] Iter[550/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:824:Epoch[52] Iter[600/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:825:Epoch[52] Iter[650/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:826:Epoch[52] Iter[700/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_baseline_4090_s2.log:827:Epoch[52] done in 91.7s  Loss=1.053 Acc=0.997
logs/agreidv2_baseline_4090_s2.log:828:Epoch[53] Iter[50/786] Loss: 1.071 CE: 1.061 Tri: 0.009 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:829:Epoch[53] Iter[100/786] Loss: 1.068 CE: 1.060 Tri: 0.008 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:830:Epoch[53] Iter[150/786] Loss: 1.067 CE: 1.060 Tri: 0.007 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:831:Epoch[53] Iter[200/786] Loss: 1.067 CE: 1.059 Tri: 0.008 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:832:Epoch[53] Iter[250/786] Loss: 1.066 CE: 1.058 Tri: 0.008 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:833:Epoch[53] Iter[300/786] Loss: 1.064 CE: 1.057 Tri: 0.007 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:834:Epoch[53] Iter[350/786] Loss: 1.063 CE: 1.056 Tri: 0.007 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:835:Epoch[53] Iter[400/786] Loss: 1.062 CE: 1.056 Tri: 0.006 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:836:Epoch[53] Iter[450/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:837:Epoch[53] Iter[500/786] Loss: 1.060 CE: 1.054 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:838:Epoch[53] Iter[550/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:839:Epoch[53] Iter[600/786] Loss: 1.058 CE: 1.052 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:840:Epoch[53] Iter[650/786] Loss: 1.057 CE: 1.051 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:841:Epoch[53] Iter[700/786] Loss: 1.056 CE: 1.050 Tri: 0.005 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline_4090_s2.log:842:Epoch[53] done in 91.2s  Loss=1.054 Acc=0.997
logs/agreidv2_baseline_4090_s2.log:843:Epoch[54] Iter[50/786] Loss: 1.067 CE: 1.062 Tri: 0.005 Acc: 0.995 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:844:Epoch[54] Iter[100/786] Loss: 1.063 CE: 1.059 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:845:Epoch[54] Iter[150/786] Loss: 1.060 CE: 1.057 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:846:Epoch[54] Iter[200/786] Loss: 1.060 CE: 1.056 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:847:Epoch[54] Iter[250/786] Loss: 1.060 CE: 1.056 Tri: 0.004 Acc: 0.995 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:848:Epoch[54] Iter[300/786] Loss: 1.060 CE: 1.056 Tri: 0.004 Acc: 0.995 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:849:Epoch[54] Iter[350/786] Loss: 1.059 CE: 1.055 Tri: 0.004 Acc: 0.995 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:850:Epoch[54] Iter[400/786] Loss: 1.058 CE: 1.054 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:851:Epoch[54] Iter[450/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:852:Epoch[54] Iter[500/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:853:Epoch[54] Iter[550/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:854:Epoch[54] Iter[600/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:855:Epoch[54] Iter[650/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:856:Epoch[54] Iter[700/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_baseline_4090_s2.log:857:Epoch[54] done in 90.8s  Loss=1.052 Acc=0.997
logs/agreidv2_baseline_4090_s2.log:858:Epoch[55] Iter[50/786] Loss: 1.063 CE: 1.058 Tri: 0.005 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:859:Epoch[55] Iter[100/786] Loss: 1.062 CE: 1.057 Tri: 0.006 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:860:Epoch[55] Iter[150/786] Loss: 1.064 CE: 1.056 Tri: 0.008 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:861:Epoch[55] Iter[200/786] Loss: 1.061 CE: 1.054 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:862:Epoch[55] Iter[250/786] Loss: 1.062 CE: 1.055 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:863:Epoch[55] Iter[300/786] Loss: 1.061 CE: 1.054 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:864:Epoch[55] Iter[350/786] Loss: 1.060 CE: 1.053 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:865:Epoch[55] Iter[400/786] Loss: 1.059 CE: 1.052 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:866:Epoch[55] Iter[450/786] Loss: 1.059 CE: 1.052 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:867:Epoch[55] Iter[500/786] Loss: 1.058 CE: 1.051 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:868:Epoch[55] Iter[550/786] Loss: 1.057 CE: 1.050 Tri: 0.007 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:869:Epoch[55] Iter[600/786] Loss: 1.055 CE: 1.049 Tri: 0.006 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:870:Epoch[55] Iter[650/786] Loss: 1.054 CE: 1.048 Tri: 0.006 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:871:Epoch[55] Iter[700/786] Loss: 1.053 CE: 1.048 Tri: 0.006 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline_4090_s2.log:872:Epoch[55] done in 90.8s  Loss=1.052 Acc=0.997
logs/agreidv2_baseline_4090_s2.log:873:Epoch[56] Iter[50/786] Loss: 1.059 CE: 1.052 Tri: 0.007 Acc: 0.998 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:874:Epoch[56] Iter[100/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:875:Epoch[56] Iter[150/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:876:Epoch[56] Iter[200/786] Loss: 1.057 CE: 1.053 Tri: 0.005 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:877:Epoch[56] Iter[250/786] Loss: 1.058 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:878:Epoch[56] Iter[300/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:879:Epoch[56] Iter[350/786] Loss: 1.058 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:880:Epoch[56] Iter[400/786] Loss: 1.057 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:881:Epoch[56] Iter[450/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:882:Epoch[56] Iter[500/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:883:Epoch[56] Iter[550/786] Loss: 1.054 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:884:Epoch[56] Iter[600/786] Loss: 1.053 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:885:Epoch[56] Iter[650/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:886:Epoch[56] Iter[700/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090_s2.log:887:Epoch[56] done in 91.8s  Loss=1.049 Acc=0.997
logs/agreidv2_baseline_4090_s2.log:888:Epoch[57] Iter[50/786] Loss: 1.053 CE: 1.050 Tri: 0.003 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_baseline_4090_s2.log:889:Epoch[57] Iter[100/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.995 LR: 5.50e-07
logs/agreidv2_baseline_4090_s2.log:890:Epoch[57] Iter[150/786] Loss: 1.055 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090_s2.log:891:Epoch[57] Iter[200/786] Loss: 1.054 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090_s2.log:892:Epoch[57] Iter[250/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090_s2.log:893:Epoch[57] Iter[300/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090_s2.log:894:Epoch[57] Iter[350/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090_s2.log:895:Epoch[57] Iter[400/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090_s2.log:896:Epoch[57] Iter[450/786] Loss: 1.053 CE: 1.050 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:2:CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
logs/agreidv2_baseline_4090.log:7:  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
logs/agreidv2_baseline_4090.log:8:  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
logs/agreidv2_baseline_4090.log:9:  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_baseline_4090.log:11:  out_dir=/home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090
logs/agreidv2_baseline_4090.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_baseline_4090.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_baseline_4090.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_baseline_4090.log:25:Epoch[1] Iter[50/786] Loss: 38.559 CE: 6.693 Tri: 31.866 Acc: 0.001 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:26:Epoch[1] Iter[100/786] Loss: 31.717 CE: 6.692 Tri: 25.025 Acc: 0.002 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:27:Epoch[1] Iter[150/786] Loss: 28.642 CE: 6.690 Tri: 21.952 Acc: 0.002 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:28:Epoch[1] Iter[200/786] Loss: 26.615 CE: 6.689 Tri: 19.926 Acc: 0.004 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:29:Epoch[1] Iter[250/786] Loss: 25.297 CE: 6.686 Tri: 18.611 Acc: 0.005 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:30:Epoch[1] Iter[300/786] Loss: 24.205 CE: 6.684 Tri: 17.521 Acc: 0.008 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:31:Epoch[1] Iter[350/786] Loss: 23.287 CE: 6.682 Tri: 16.605 Acc: 0.010 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:32:Epoch[1] Iter[400/786] Loss: 22.465 CE: 6.680 Tri: 15.785 Acc: 0.014 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:33:Epoch[1] Iter[450/786] Loss: 21.761 CE: 6.677 Tri: 15.084 Acc: 0.020 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:34:Epoch[1] Iter[500/786] Loss: 21.109 CE: 6.675 Tri: 14.434 Acc: 0.028 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:35:Epoch[1] Iter[550/786] Loss: 20.511 CE: 6.672 Tri: 13.839 Acc: 0.038 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:36:Epoch[1] Iter[600/786] Loss: 20.012 CE: 6.669 Tri: 13.343 Acc: 0.049 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:37:Epoch[1] Iter[650/786] Loss: 19.529 CE: 6.666 Tri: 12.863 Acc: 0.064 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:38:Epoch[1] Iter[700/786] Loss: 19.089 CE: 6.662 Tri: 12.427 Acc: 0.086 LR: 3.50e-07
logs/agreidv2_baseline_4090.log:39:Epoch[1] done in 91.5s  Loss=18.756 Acc=0.113
logs/agreidv2_baseline_4090.log:40:Epoch[2] Iter[50/786] Loss: 11.898 CE: 6.607 Tri: 5.291 Acc: 0.137 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:41:Epoch[2] Iter[100/786] Loss: 11.336 CE: 6.577 Tri: 4.759 Acc: 0.171 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:42:Epoch[2] Iter[150/786] Loss: 10.828 CE: 6.543 Tri: 4.285 Acc: 0.207 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:43:Epoch[2] Iter[200/786] Loss: 10.525 CE: 6.509 Tri: 4.015 Acc: 0.237 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:44:Epoch[2] Iter[250/786] Loss: 10.265 CE: 6.474 Tri: 3.791 Acc: 0.268 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:45:Epoch[2] Iter[300/786] Loss: 10.016 CE: 6.440 Tri: 3.576 Acc: 0.293 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:46:Epoch[2] Iter[350/786] Loss: 9.828 CE: 6.408 Tri: 3.420 Acc: 0.312 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:47:Epoch[2] Iter[400/786] Loss: 9.630 CE: 6.374 Tri: 3.256 Acc: 0.337 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:48:Epoch[2] Iter[450/786] Loss: 9.449 CE: 6.339 Tri: 3.109 Acc: 0.357 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:49:Epoch[2] Iter[500/786] Loss: 9.298 CE: 6.303 Tri: 2.995 Acc: 0.378 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:50:Epoch[2] Iter[550/786] Loss: 9.149 CE: 6.264 Tri: 2.886 Acc: 0.401 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:51:Epoch[2] Iter[600/786] Loss: 9.000 CE: 6.222 Tri: 2.778 Acc: 0.423 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:52:Epoch[2] Iter[650/786] Loss: 8.868 CE: 6.178 Tri: 2.689 Acc: 0.445 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:53:Epoch[2] Iter[700/786] Loss: 8.735 CE: 6.125 Tri: 2.609 Acc: 0.469 LR: 3.82e-06
logs/agreidv2_baseline_4090.log:54:Epoch[2] done in 91.2s  Loss=8.616 Acc=0.490
logs/agreidv2_baseline_4090.log:55:Epoch[3] Iter[50/786] Loss: 7.078 CE: 5.631 Tri: 1.447 Acc: 0.304 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:56:Epoch[3] Iter[100/786] Loss: 6.952 CE: 5.535 Tri: 1.417 Acc: 0.356 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:57:Epoch[3] Iter[150/786] Loss: 6.785 CE: 5.449 Tri: 1.336 Acc: 0.391 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:58:Epoch[3] Iter[200/786] Loss: 6.652 CE: 5.358 Tri: 1.294 Acc: 0.432 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:59:Epoch[3] Iter[250/786] Loss: 6.539 CE: 5.273 Tri: 1.265 Acc: 0.459 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:60:Epoch[3] Iter[300/786] Loss: 6.411 CE: 5.192 Tri: 1.219 Acc: 0.481 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:61:Epoch[3] Iter[350/786] Loss: 6.294 CE: 5.108 Tri: 1.185 Acc: 0.505 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:62:Epoch[3] Iter[400/786] Loss: 6.172 CE: 5.016 Tri: 1.155 Acc: 0.529 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:63:Epoch[3] Iter[450/786] Loss: 6.058 CE: 4.929 Tri: 1.129 Acc: 0.549 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:64:Epoch[3] Iter[500/786] Loss: 5.933 CE: 4.837 Tri: 1.095 Acc: 0.569 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:65:Epoch[3] Iter[550/786] Loss: 5.812 CE: 4.746 Tri: 1.066 Acc: 0.584 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:66:Epoch[3] Iter[600/786] Loss: 5.699 CE: 4.649 Tri: 1.049 Acc: 0.601 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:67:Epoch[3] Iter[650/786] Loss: 5.568 CE: 4.541 Tri: 1.027 Acc: 0.618 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:68:Epoch[3] Iter[700/786] Loss: 5.424 CE: 4.419 Tri: 1.005 Acc: 0.636 LR: 7.28e-06
logs/agreidv2_baseline_4090.log:69:Epoch[3] done in 90.6s  Loss=5.292 Acc=0.651
logs/agreidv2_baseline_4090.log:70:Epoch[4] Iter[50/786] Loss: 4.542 CE: 3.787 Tri: 0.756 Acc: 0.480 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:71:Epoch[4] Iter[100/786] Loss: 4.394 CE: 3.686 Tri: 0.708 Acc: 0.509 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:72:Epoch[4] Iter[150/786] Loss: 4.257 CE: 3.562 Tri: 0.695 Acc: 0.553 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:73:Epoch[4] Iter[200/786] Loss: 4.128 CE: 3.455 Tri: 0.673 Acc: 0.590 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:74:Epoch[4] Iter[250/786] Loss: 3.987 CE: 3.338 Tri: 0.649 Acc: 0.625 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:75:Epoch[4] Iter[300/786] Loss: 3.886 CE: 3.247 Tri: 0.639 Acc: 0.650 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:76:Epoch[4] Iter[350/786] Loss: 3.787 CE: 3.162 Tri: 0.626 Acc: 0.673 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:77:Epoch[4] Iter[400/786] Loss: 3.698 CE: 3.079 Tri: 0.620 Acc: 0.694 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:78:Epoch[4] Iter[450/786] Loss: 3.622 CE: 3.005 Tri: 0.617 Acc: 0.711 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:79:Epoch[4] Iter[500/786] Loss: 3.533 CE: 2.925 Tri: 0.608 Acc: 0.727 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:80:Epoch[4] Iter[550/786] Loss: 3.453 CE: 2.852 Tri: 0.601 Acc: 0.742 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:81:Epoch[4] Iter[600/786] Loss: 3.379 CE: 2.782 Tri: 0.596 Acc: 0.755 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:82:Epoch[4] Iter[650/786] Loss: 3.303 CE: 2.712 Tri: 0.591 Acc: 0.767 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:83:Epoch[4] Iter[700/786] Loss: 3.222 CE: 2.640 Tri: 0.582 Acc: 0.779 LR: 1.07e-05
logs/agreidv2_baseline_4090.log:84:Epoch[4] done in 90.8s  Loss=3.152 Acc=0.789
logs/agreidv2_baseline_4090.log:85:Epoch[5] Iter[50/786] Loss: 2.885 CE: 2.466 Tri: 0.419 Acc: 0.728 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:86:Epoch[5] Iter[100/786] Loss: 2.799 CE: 2.361 Tri: 0.438 Acc: 0.765 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:87:Epoch[5] Iter[150/786] Loss: 2.753 CE: 2.292 Tri: 0.460 Acc: 0.790 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:88:Epoch[5] Iter[200/786] Loss: 2.664 CE: 2.211 Tri: 0.453 Acc: 0.811 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:89:Epoch[5] Iter[250/786] Loss: 2.628 CE: 2.159 Tri: 0.469 Acc: 0.826 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:90:Epoch[5] Iter[300/786] Loss: 2.568 CE: 2.106 Tri: 0.462 Acc: 0.839 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:91:Epoch[5] Iter[350/786] Loss: 2.508 CE: 2.061 Tri: 0.447 Acc: 0.848 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:92:Epoch[5] Iter[400/786] Loss: 2.458 CE: 2.019 Tri: 0.439 Acc: 0.858 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:93:Epoch[5] Iter[450/786] Loss: 2.418 CE: 1.985 Tri: 0.433 Acc: 0.864 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:94:Epoch[5] Iter[500/786] Loss: 2.385 CE: 1.953 Tri: 0.431 Acc: 0.871 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:95:Epoch[5] Iter[550/786] Loss: 2.340 CE: 1.921 Tri: 0.420 Acc: 0.877 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:96:Epoch[5] Iter[600/786] Loss: 2.311 CE: 1.892 Tri: 0.419 Acc: 0.882 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:97:Epoch[5] Iter[650/786] Loss: 2.276 CE: 1.863 Tri: 0.413 Acc: 0.887 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:98:Epoch[5] Iter[700/786] Loss: 2.242 CE: 1.834 Tri: 0.407 Acc: 0.892 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:99:Epoch[5] done in 90.9s  Loss=2.212 Acc=0.896
logs/agreidv2_baseline_4090.log:100:Epoch[6] Iter[50/786] Loss: 2.193 CE: 1.849 Tri: 0.344 Acc: 0.863 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:101:Epoch[6] Iter[100/786] Loss: 2.164 CE: 1.814 Tri: 0.350 Acc: 0.875 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:102:Epoch[6] Iter[150/786] Loss: 2.116 CE: 1.771 Tri: 0.345 Acc: 0.886 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:103:Epoch[6] Iter[200/786] Loss: 2.088 CE: 1.739 Tri: 0.349 Acc: 0.894 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:104:Epoch[6] Iter[250/786] Loss: 2.058 CE: 1.711 Tri: 0.347 Acc: 0.902 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:105:Epoch[6] Iter[300/786] Loss: 2.031 CE: 1.689 Tri: 0.342 Acc: 0.907 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:106:Epoch[6] Iter[350/786] Loss: 2.002 CE: 1.669 Tri: 0.332 Acc: 0.912 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:107:Epoch[6] Iter[400/786] Loss: 1.983 CE: 1.655 Tri: 0.328 Acc: 0.915 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:108:Epoch[6] Iter[450/786] Loss: 1.963 CE: 1.639 Tri: 0.324 Acc: 0.919 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:109:Epoch[6] Iter[500/786] Loss: 1.948 CE: 1.626 Tri: 0.322 Acc: 0.922 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:110:Epoch[6] Iter[550/786] Loss: 1.928 CE: 1.612 Tri: 0.316 Acc: 0.924 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:111:Epoch[6] Iter[600/786] Loss: 1.933 CE: 1.603 Tri: 0.330 Acc: 0.926 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:112:Epoch[6] Iter[650/786] Loss: 1.942 CE: 1.597 Tri: 0.344 Acc: 0.927 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:113:Epoch[6] Iter[700/786] Loss: 1.931 CE: 1.587 Tri: 0.345 Acc: 0.929 LR: 1.77e-05
logs/agreidv2_baseline_4090.log:114:Epoch[6] done in 91.8s  Loss=1.912 Acc=0.932
logs/agreidv2_baseline_4090.log:115:Epoch[7] Iter[50/786] Loss: 1.960 CE: 1.643 Tri: 0.317 Acc: 0.912 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:116:Epoch[7] Iter[100/786] Loss: 1.925 CE: 1.611 Tri: 0.314 Acc: 0.919 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:117:Epoch[7] Iter[150/786] Loss: 1.915 CE: 1.598 Tri: 0.316 Acc: 0.923 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:118:Epoch[7] Iter[200/786] Loss: 1.886 CE: 1.578 Tri: 0.307 Acc: 0.927 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:119:Epoch[7] Iter[250/786] Loss: 1.853 CE: 1.561 Tri: 0.292 Acc: 0.931 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:120:Epoch[7] Iter[300/786] Loss: 1.836 CE: 1.551 Tri: 0.286 Acc: 0.932 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:121:Epoch[7] Iter[350/786] Loss: 1.812 CE: 1.536 Tri: 0.276 Acc: 0.936 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:122:Epoch[7] Iter[400/786] Loss: 1.799 CE: 1.527 Tri: 0.272 Acc: 0.938 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:123:Epoch[7] Iter[450/786] Loss: 1.783 CE: 1.517 Tri: 0.266 Acc: 0.940 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:124:Epoch[7] Iter[500/786] Loss: 1.773 CE: 1.509 Tri: 0.263 Acc: 0.942 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:125:Epoch[7] Iter[550/786] Loss: 1.760 CE: 1.501 Tri: 0.258 Acc: 0.944 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:126:Epoch[7] Iter[600/786] Loss: 1.751 CE: 1.493 Tri: 0.258 Acc: 0.945 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:127:Epoch[7] Iter[650/786] Loss: 1.740 CE: 1.486 Tri: 0.254 Acc: 0.947 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:128:Epoch[7] Iter[700/786] Loss: 1.727 CE: 1.476 Tri: 0.251 Acc: 0.948 LR: 2.11e-05
logs/agreidv2_baseline_4090.log:129:Epoch[7] done in 90.7s  Loss=1.714 Acc=0.950
logs/agreidv2_baseline_4090.log:130:Epoch[8] Iter[50/786] Loss: 1.836 CE: 1.548 Tri: 0.288 Acc: 0.926 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:131:Epoch[8] Iter[100/786] Loss: 1.783 CE: 1.517 Tri: 0.266 Acc: 0.931 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:132:Epoch[8] Iter[150/786] Loss: 1.755 CE: 1.498 Tri: 0.258 Acc: 0.935 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:133:Epoch[8] Iter[200/786] Loss: 1.742 CE: 1.489 Tri: 0.253 Acc: 0.938 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:134:Epoch[8] Iter[250/786] Loss: 1.731 CE: 1.482 Tri: 0.249 Acc: 0.940 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:135:Epoch[8] Iter[300/786] Loss: 1.719 CE: 1.476 Tri: 0.242 Acc: 0.942 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:136:Epoch[8] Iter[350/786] Loss: 1.710 CE: 1.471 Tri: 0.239 Acc: 0.944 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:137:Epoch[8] Iter[400/786] Loss: 1.706 CE: 1.466 Tri: 0.240 Acc: 0.945 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:138:Epoch[8] Iter[450/786] Loss: 1.696 CE: 1.460 Tri: 0.235 Acc: 0.946 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:139:Epoch[8] Iter[500/786] Loss: 1.690 CE: 1.457 Tri: 0.234 Acc: 0.947 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:140:Epoch[8] Iter[550/786] Loss: 1.680 CE: 1.449 Tri: 0.231 Acc: 0.948 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:141:Epoch[8] Iter[600/786] Loss: 1.675 CE: 1.445 Tri: 0.230 Acc: 0.949 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:142:Epoch[8] Iter[650/786] Loss: 1.668 CE: 1.439 Tri: 0.229 Acc: 0.950 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:143:Epoch[8] Iter[700/786] Loss: 1.658 CE: 1.431 Tri: 0.227 Acc: 0.951 LR: 2.46e-05
logs/agreidv2_baseline_4090.log:144:Epoch[8] done in 91.7s  Loss=1.643 Acc=0.953
logs/agreidv2_baseline_4090.log:145:Epoch[9] Iter[50/786] Loss: 1.798 CE: 1.525 Tri: 0.273 Acc: 0.932 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:146:Epoch[9] Iter[100/786] Loss: 1.741 CE: 1.495 Tri: 0.247 Acc: 0.939 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:147:Epoch[9] Iter[150/786] Loss: 1.723 CE: 1.485 Tri: 0.239 Acc: 0.940 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:148:Epoch[9] Iter[200/786] Loss: 1.705 CE: 1.473 Tri: 0.233 Acc: 0.944 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:149:Epoch[9] Iter[250/786] Loss: 1.681 CE: 1.461 Tri: 0.220 Acc: 0.946 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:150:Epoch[9] Iter[300/786] Loss: 1.667 CE: 1.452 Tri: 0.215 Acc: 0.948 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:151:Epoch[9] Iter[350/786] Loss: 1.649 CE: 1.442 Tri: 0.207 Acc: 0.950 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:152:Epoch[9] Iter[400/786] Loss: 1.631 CE: 1.433 Tri: 0.198 Acc: 0.952 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:153:Epoch[9] Iter[450/786] Loss: 1.625 CE: 1.428 Tri: 0.197 Acc: 0.953 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:154:Epoch[9] Iter[500/786] Loss: 1.624 CE: 1.425 Tri: 0.199 Acc: 0.953 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:155:Epoch[9] Iter[550/786] Loss: 1.622 CE: 1.421 Tri: 0.201 Acc: 0.954 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:156:Epoch[9] Iter[600/786] Loss: 1.618 CE: 1.416 Tri: 0.202 Acc: 0.955 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:157:Epoch[9] Iter[650/786] Loss: 1.611 CE: 1.410 Tri: 0.201 Acc: 0.956 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:158:Epoch[9] Iter[700/786] Loss: 1.603 CE: 1.403 Tri: 0.200 Acc: 0.957 LR: 2.81e-05
logs/agreidv2_baseline_4090.log:159:Epoch[9] done in 91.5s  Loss=1.588 Acc=0.958
logs/agreidv2_baseline_4090.log:160:Epoch[10] Iter[50/786] Loss: 1.724 CE: 1.503 Tri: 0.221 Acc: 0.935 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:161:Epoch[10] Iter[100/786] Loss: 1.710 CE: 1.482 Tri: 0.228 Acc: 0.940 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:162:Epoch[10] Iter[150/786] Loss: 1.673 CE: 1.459 Tri: 0.215 Acc: 0.944 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:163:Epoch[10] Iter[200/786] Loss: 1.666 CE: 1.453 Tri: 0.213 Acc: 0.945 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:164:Epoch[10] Iter[250/786] Loss: 1.647 CE: 1.441 Tri: 0.206 Acc: 0.948 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:165:Epoch[10] Iter[300/786] Loss: 1.630 CE: 1.432 Tri: 0.199 Acc: 0.950 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:166:Epoch[10] Iter[350/786] Loss: 1.626 CE: 1.428 Tri: 0.197 Acc: 0.951 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:167:Epoch[10] Iter[400/786] Loss: 1.623 CE: 1.425 Tri: 0.199 Acc: 0.952 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:168:Epoch[10] Iter[450/786] Loss: 1.618 CE: 1.419 Tri: 0.199 Acc: 0.953 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:169:Epoch[10] Iter[500/786] Loss: 1.613 CE: 1.416 Tri: 0.198 Acc: 0.953 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:170:Epoch[10] Iter[550/786] Loss: 1.606 CE: 1.410 Tri: 0.196 Acc: 0.954 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:171:Epoch[10] Iter[600/786] Loss: 1.600 CE: 1.406 Tri: 0.194 Acc: 0.954 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:172:Epoch[10] Iter[650/786] Loss: 1.595 CE: 1.401 Tri: 0.194 Acc: 0.955 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:173:Epoch[10] Iter[700/786] Loss: 1.584 CE: 1.392 Tri: 0.192 Acc: 0.957 LR: 3.15e-05
logs/agreidv2_baseline_4090.log:174:Epoch[10] done in 91.1s  Loss=1.571 Acc=0.958
logs/agreidv2_baseline_4090.log:176:    [A->G] mAP=73.40  R1=82.85  R5=89.26  mINP=44.92
logs/agreidv2_baseline_4090.log:177:    [G->A] mAP=72.69  R1=83.43  R5=89.45  mINP=35.03
logs/agreidv2_baseline_4090.log:178:    [mean] mAP=73.04  R1=83.14
logs/agreidv2_baseline_4090.log:179:    * new best mean mAP=73.04 (epoch 10) saved
logs/agreidv2_baseline_4090.log:180:Epoch[11] Iter[50/786] Loss: 1.728 CE: 1.512 Tri: 0.216 Acc: 0.927 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:181:Epoch[11] Iter[100/786] Loss: 1.648 CE: 1.468 Tri: 0.179 Acc: 0.942 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:182:Epoch[11] Iter[150/786] Loss: 1.617 CE: 1.440 Tri: 0.177 Acc: 0.948 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:183:Epoch[11] Iter[200/786] Loss: 1.611 CE: 1.432 Tri: 0.179 Acc: 0.949 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:184:Epoch[11] Iter[250/786] Loss: 1.616 CE: 1.431 Tri: 0.185 Acc: 0.949 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:185:Epoch[11] Iter[300/786] Loss: 1.606 CE: 1.425 Tri: 0.180 Acc: 0.950 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:186:Epoch[11] Iter[350/786] Loss: 1.600 CE: 1.421 Tri: 0.179 Acc: 0.951 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:187:Epoch[11] Iter[400/786] Loss: 1.598 CE: 1.416 Tri: 0.182 Acc: 0.952 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:188:Epoch[11] Iter[450/786] Loss: 1.589 CE: 1.410 Tri: 0.179 Acc: 0.953 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:189:Epoch[11] Iter[500/786] Loss: 1.583 CE: 1.406 Tri: 0.177 Acc: 0.954 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:190:Epoch[11] Iter[550/786] Loss: 1.574 CE: 1.400 Tri: 0.174 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:191:Epoch[11] Iter[600/786] Loss: 1.569 CE: 1.395 Tri: 0.174 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:192:Epoch[11] Iter[650/786] Loss: 1.560 CE: 1.387 Tri: 0.173 Acc: 0.957 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:193:Epoch[11] Iter[700/786] Loss: 1.548 CE: 1.379 Tri: 0.169 Acc: 0.958 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:194:Epoch[11] done in 91.0s  Loss=1.536 Acc=0.960
logs/agreidv2_baseline_4090.log:195:Epoch[12] Iter[50/786] Loss: 1.686 CE: 1.493 Tri: 0.193 Acc: 0.936 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:196:Epoch[12] Iter[100/786] Loss: 1.640 CE: 1.458 Tri: 0.183 Acc: 0.942 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:197:Epoch[12] Iter[150/786] Loss: 1.619 CE: 1.441 Tri: 0.178 Acc: 0.945 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:198:Epoch[12] Iter[200/786] Loss: 1.597 CE: 1.427 Tri: 0.170 Acc: 0.947 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:199:Epoch[12] Iter[250/786] Loss: 1.593 CE: 1.420 Tri: 0.173 Acc: 0.950 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:200:Epoch[12] Iter[300/786] Loss: 1.581 CE: 1.411 Tri: 0.170 Acc: 0.952 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:201:Epoch[12] Iter[350/786] Loss: 1.572 CE: 1.404 Tri: 0.168 Acc: 0.953 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:202:Epoch[12] Iter[400/786] Loss: 1.562 CE: 1.397 Tri: 0.165 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:203:Epoch[12] Iter[450/786] Loss: 1.554 CE: 1.392 Tri: 0.162 Acc: 0.956 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:204:Epoch[12] Iter[500/786] Loss: 1.546 CE: 1.385 Tri: 0.161 Acc: 0.957 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:205:Epoch[12] Iter[550/786] Loss: 1.542 CE: 1.380 Tri: 0.161 Acc: 0.958 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:206:Epoch[12] Iter[600/786] Loss: 1.538 CE: 1.376 Tri: 0.162 Acc: 0.959 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:207:Epoch[12] Iter[650/786] Loss: 1.536 CE: 1.371 Tri: 0.164 Acc: 0.959 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:208:Epoch[12] Iter[700/786] Loss: 1.525 CE: 1.363 Tri: 0.162 Acc: 0.961 LR: 3.50e-05
logs/agreidv2_baseline_4090.log:209:Epoch[12] done in 91.8s  Loss=1.511 Acc=0.962
logs/agreidv2_baseline_4090.log:210:Epoch[13] Iter[50/786] Loss: 1.607 CE: 1.438 Tri: 0.169 Acc: 0.944 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:211:Epoch[13] Iter[100/786] Loss: 1.567 CE: 1.412 Tri: 0.155 Acc: 0.950 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:212:Epoch[13] Iter[150/786] Loss: 1.550 CE: 1.398 Tri: 0.152 Acc: 0.953 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:213:Epoch[13] Iter[200/786] Loss: 1.546 CE: 1.389 Tri: 0.157 Acc: 0.954 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:214:Epoch[13] Iter[250/786] Loss: 1.541 CE: 1.384 Tri: 0.156 Acc: 0.955 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:215:Epoch[13] Iter[300/786] Loss: 1.531 CE: 1.377 Tri: 0.154 Acc: 0.957 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:216:Epoch[13] Iter[350/786] Loss: 1.521 CE: 1.371 Tri: 0.150 Acc: 0.959 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:217:Epoch[13] Iter[400/786] Loss: 1.517 CE: 1.367 Tri: 0.150 Acc: 0.959 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:218:Epoch[13] Iter[450/786] Loss: 1.507 CE: 1.361 Tri: 0.146 Acc: 0.960 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:219:Epoch[13] Iter[500/786] Loss: 1.503 CE: 1.357 Tri: 0.146 Acc: 0.961 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:220:Epoch[13] Iter[550/786] Loss: 1.499 CE: 1.354 Tri: 0.145 Acc: 0.961 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:221:Epoch[13] Iter[600/786] Loss: 1.493 CE: 1.348 Tri: 0.144 Acc: 0.962 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:222:Epoch[13] Iter[650/786] Loss: 1.483 CE: 1.341 Tri: 0.142 Acc: 0.963 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:223:Epoch[13] Iter[700/786] Loss: 1.475 CE: 1.335 Tri: 0.140 Acc: 0.964 LR: 3.49e-05
logs/agreidv2_baseline_4090.log:224:Epoch[13] done in 90.6s  Loss=1.466 Acc=0.965
logs/agreidv2_baseline_4090.log:225:Epoch[14] Iter[50/786] Loss: 1.575 CE: 1.447 Tri: 0.128 Acc: 0.939 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:226:Epoch[14] Iter[100/786] Loss: 1.519 CE: 1.402 Tri: 0.117 Acc: 0.950 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:227:Epoch[14] Iter[150/786] Loss: 1.507 CE: 1.384 Tri: 0.124 Acc: 0.955 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:228:Epoch[14] Iter[200/786] Loss: 1.497 CE: 1.375 Tri: 0.122 Acc: 0.956 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:229:Epoch[14] Iter[250/786] Loss: 1.485 CE: 1.365 Tri: 0.120 Acc: 0.958 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:230:Epoch[14] Iter[300/786] Loss: 1.474 CE: 1.356 Tri: 0.119 Acc: 0.960 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:231:Epoch[14] Iter[350/786] Loss: 1.464 CE: 1.349 Tri: 0.116 Acc: 0.962 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:232:Epoch[14] Iter[400/786] Loss: 1.463 CE: 1.345 Tri: 0.118 Acc: 0.963 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:233:Epoch[14] Iter[450/786] Loss: 1.460 CE: 1.341 Tri: 0.119 Acc: 0.964 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:234:Epoch[14] Iter[500/786] Loss: 1.460 CE: 1.339 Tri: 0.120 Acc: 0.964 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:235:Epoch[14] Iter[550/786] Loss: 1.457 CE: 1.335 Tri: 0.121 Acc: 0.964 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:236:Epoch[14] Iter[600/786] Loss: 1.450 CE: 1.329 Tri: 0.121 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:237:Epoch[14] Iter[650/786] Loss: 1.444 CE: 1.323 Tri: 0.121 Acc: 0.966 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:238:Epoch[14] Iter[700/786] Loss: 1.437 CE: 1.316 Tri: 0.120 Acc: 0.967 LR: 3.47e-05
logs/agreidv2_baseline_4090.log:239:Epoch[14] done in 91.6s  Loss=1.425 Acc=0.969
logs/agreidv2_baseline_4090.log:240:Epoch[15] Iter[50/786] Loss: 1.512 CE: 1.405 Tri: 0.106 Acc: 0.946 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:241:Epoch[15] Iter[100/786] Loss: 1.486 CE: 1.374 Tri: 0.111 Acc: 0.954 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:242:Epoch[15] Iter[150/786] Loss: 1.474 CE: 1.357 Tri: 0.116 Acc: 0.960 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:243:Epoch[15] Iter[200/786] Loss: 1.461 CE: 1.349 Tri: 0.113 Acc: 0.962 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:244:Epoch[15] Iter[250/786] Loss: 1.452 CE: 1.340 Tri: 0.112 Acc: 0.963 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:245:Epoch[15] Iter[300/786] Loss: 1.449 CE: 1.335 Tri: 0.114 Acc: 0.964 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:246:Epoch[15] Iter[350/786] Loss: 1.448 CE: 1.332 Tri: 0.116 Acc: 0.965 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:247:Epoch[15] Iter[400/786] Loss: 1.442 CE: 1.327 Tri: 0.115 Acc: 0.966 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:248:Epoch[15] Iter[450/786] Loss: 1.435 CE: 1.323 Tri: 0.112 Acc: 0.966 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:249:Epoch[15] Iter[500/786] Loss: 1.430 CE: 1.318 Tri: 0.111 Acc: 0.967 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:250:Epoch[15] Iter[550/786] Loss: 1.424 CE: 1.314 Tri: 0.110 Acc: 0.968 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:251:Epoch[15] Iter[600/786] Loss: 1.422 CE: 1.311 Tri: 0.111 Acc: 0.968 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:252:Epoch[15] Iter[650/786] Loss: 1.416 CE: 1.306 Tri: 0.110 Acc: 0.969 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:253:Epoch[15] Iter[700/786] Loss: 1.409 CE: 1.300 Tri: 0.109 Acc: 0.970 LR: 3.45e-05
logs/agreidv2_baseline_4090.log:254:Epoch[15] done in 91.6s  Loss=1.399 Acc=0.971
logs/agreidv2_baseline_4090.log:255:Epoch[16] Iter[50/786] Loss: 1.513 CE: 1.395 Tri: 0.118 Acc: 0.950 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:256:Epoch[16] Iter[100/786] Loss: 1.462 CE: 1.355 Tri: 0.106 Acc: 0.959 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:257:Epoch[16] Iter[150/786] Loss: 1.444 CE: 1.340 Tri: 0.104 Acc: 0.963 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:258:Epoch[16] Iter[200/786] Loss: 1.431 CE: 1.331 Tri: 0.100 Acc: 0.964 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:259:Epoch[16] Iter[250/786] Loss: 1.426 CE: 1.322 Tri: 0.104 Acc: 0.966 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:260:Epoch[16] Iter[300/786] Loss: 1.420 CE: 1.318 Tri: 0.103 Acc: 0.967 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:261:Epoch[16] Iter[350/786] Loss: 1.419 CE: 1.314 Tri: 0.105 Acc: 0.967 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:262:Epoch[16] Iter[400/786] Loss: 1.415 CE: 1.311 Tri: 0.105 Acc: 0.968 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:263:Epoch[16] Iter[450/786] Loss: 1.414 CE: 1.308 Tri: 0.106 Acc: 0.968 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:264:Epoch[16] Iter[500/786] Loss: 1.408 CE: 1.303 Tri: 0.104 Acc: 0.969 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:265:Epoch[16] Iter[550/786] Loss: 1.402 CE: 1.299 Tri: 0.104 Acc: 0.970 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:266:Epoch[16] Iter[600/786] Loss: 1.397 CE: 1.295 Tri: 0.102 Acc: 0.970 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:267:Epoch[16] Iter[650/786] Loss: 1.391 CE: 1.291 Tri: 0.101 Acc: 0.971 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:268:Epoch[16] Iter[700/786] Loss: 1.383 CE: 1.285 Tri: 0.099 Acc: 0.972 LR: 3.41e-05
logs/agreidv2_baseline_4090.log:269:Epoch[16] done in 91.6s  Loss=1.376 Acc=0.973
logs/agreidv2_baseline_4090.log:270:Epoch[17] Iter[50/786] Loss: 1.507 CE: 1.383 Tri: 0.125 Acc: 0.952 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:271:Epoch[17] Iter[100/786] Loss: 1.479 CE: 1.356 Tri: 0.124 Acc: 0.960 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:272:Epoch[17] Iter[150/786] Loss: 1.451 CE: 1.336 Tri: 0.115 Acc: 0.964 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:273:Epoch[17] Iter[200/786] Loss: 1.440 CE: 1.327 Tri: 0.113 Acc: 0.965 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:274:Epoch[17] Iter[250/786] Loss: 1.427 CE: 1.318 Tri: 0.109 Acc: 0.967 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:275:Epoch[17] Iter[300/786] Loss: 1.418 CE: 1.310 Tri: 0.108 Acc: 0.967 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:276:Epoch[17] Iter[350/786] Loss: 1.412 CE: 1.304 Tri: 0.108 Acc: 0.968 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:277:Epoch[17] Iter[400/786] Loss: 1.406 CE: 1.300 Tri: 0.107 Acc: 0.969 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:278:Epoch[17] Iter[450/786] Loss: 1.403 CE: 1.297 Tri: 0.106 Acc: 0.970 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:279:Epoch[17] Iter[500/786] Loss: 1.397 CE: 1.293 Tri: 0.104 Acc: 0.971 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:280:Epoch[17] Iter[550/786] Loss: 1.390 CE: 1.288 Tri: 0.102 Acc: 0.971 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:281:Epoch[17] Iter[600/786] Loss: 1.384 CE: 1.284 Tri: 0.101 Acc: 0.972 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:282:Epoch[17] Iter[650/786] Loss: 1.375 CE: 1.278 Tri: 0.097 Acc: 0.973 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:283:Epoch[17] Iter[700/786] Loss: 1.368 CE: 1.272 Tri: 0.096 Acc: 0.974 LR: 3.38e-05
logs/agreidv2_baseline_4090.log:284:Epoch[17] done in 91.3s  Loss=1.358 Acc=0.975
logs/agreidv2_baseline_4090.log:285:Epoch[18] Iter[50/786] Loss: 1.397 CE: 1.327 Tri: 0.070 Acc: 0.966 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:286:Epoch[18] Iter[100/786] Loss: 1.386 CE: 1.309 Tri: 0.077 Acc: 0.968 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:287:Epoch[18] Iter[150/786] Loss: 1.372 CE: 1.293 Tri: 0.079 Acc: 0.971 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:288:Epoch[18] Iter[200/786] Loss: 1.366 CE: 1.285 Tri: 0.081 Acc: 0.972 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:289:Epoch[18] Iter[250/786] Loss: 1.370 CE: 1.284 Tri: 0.086 Acc: 0.972 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:290:Epoch[18] Iter[300/786] Loss: 1.371 CE: 1.282 Tri: 0.089 Acc: 0.973 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:291:Epoch[18] Iter[350/786] Loss: 1.363 CE: 1.278 Tri: 0.085 Acc: 0.973 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:292:Epoch[18] Iter[400/786] Loss: 1.360 CE: 1.274 Tri: 0.086 Acc: 0.974 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:293:Epoch[18] Iter[450/786] Loss: 1.359 CE: 1.273 Tri: 0.086 Acc: 0.974 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:294:Epoch[18] Iter[500/786] Loss: 1.354 CE: 1.269 Tri: 0.085 Acc: 0.974 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:295:Epoch[18] Iter[550/786] Loss: 1.352 CE: 1.267 Tri: 0.085 Acc: 0.975 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:296:Epoch[18] Iter[600/786] Loss: 1.346 CE: 1.263 Tri: 0.084 Acc: 0.975 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:297:Epoch[18] Iter[650/786] Loss: 1.339 CE: 1.257 Tri: 0.082 Acc: 0.976 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:298:Epoch[18] Iter[700/786] Loss: 1.330 CE: 1.251 Tri: 0.079 Acc: 0.977 LR: 3.33e-05
logs/agreidv2_baseline_4090.log:299:Epoch[18] done in 91.0s  Loss=1.322 Acc=0.978
logs/agreidv2_baseline_4090.log:300:Epoch[19] Iter[50/786] Loss: 1.404 CE: 1.319 Tri: 0.084 Acc: 0.966 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:301:Epoch[19] Iter[100/786] Loss: 1.387 CE: 1.298 Tri: 0.089 Acc: 0.971 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:302:Epoch[19] Iter[150/786] Loss: 1.377 CE: 1.291 Tri: 0.085 Acc: 0.971 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:303:Epoch[19] Iter[200/786] Loss: 1.361 CE: 1.280 Tri: 0.081 Acc: 0.973 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:304:Epoch[19] Iter[250/786] Loss: 1.359 CE: 1.276 Tri: 0.083 Acc: 0.973 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:305:Epoch[19] Iter[300/786] Loss: 1.352 CE: 1.270 Tri: 0.082 Acc: 0.974 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:306:Epoch[19] Iter[350/786] Loss: 1.347 CE: 1.268 Tri: 0.080 Acc: 0.975 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:307:Epoch[19] Iter[400/786] Loss: 1.346 CE: 1.266 Tri: 0.080 Acc: 0.975 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:308:Epoch[19] Iter[450/786] Loss: 1.343 CE: 1.264 Tri: 0.080 Acc: 0.975 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:309:Epoch[19] Iter[500/786] Loss: 1.340 CE: 1.260 Tri: 0.079 Acc: 0.975 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:310:Epoch[19] Iter[550/786] Loss: 1.336 CE: 1.257 Tri: 0.079 Acc: 0.976 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:311:Epoch[19] Iter[600/786] Loss: 1.333 CE: 1.254 Tri: 0.079 Acc: 0.976 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:312:Epoch[19] Iter[650/786] Loss: 1.336 CE: 1.252 Tri: 0.084 Acc: 0.976 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:313:Epoch[19] Iter[700/786] Loss: 1.330 CE: 1.248 Tri: 0.082 Acc: 0.977 LR: 3.28e-05
logs/agreidv2_baseline_4090.log:314:Epoch[19] done in 91.0s  Loss=1.323 Acc=0.978
logs/agreidv2_baseline_4090.log:315:Epoch[20] Iter[50/786] Loss: 1.479 CE: 1.369 Tri: 0.110 Acc: 0.949 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:316:Epoch[20] Iter[100/786] Loss: 1.426 CE: 1.320 Tri: 0.106 Acc: 0.962 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:317:Epoch[20] Iter[150/786] Loss: 1.412 CE: 1.308 Tri: 0.104 Acc: 0.965 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:318:Epoch[20] Iter[200/786] Loss: 1.394 CE: 1.295 Tri: 0.099 Acc: 0.968 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:319:Epoch[20] Iter[250/786] Loss: 1.376 CE: 1.285 Tri: 0.092 Acc: 0.970 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:320:Epoch[20] Iter[300/786] Loss: 1.369 CE: 1.278 Tri: 0.091 Acc: 0.970 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:321:Epoch[20] Iter[350/786] Loss: 1.364 CE: 1.273 Tri: 0.090 Acc: 0.971 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:322:Epoch[20] Iter[400/786] Loss: 1.358 CE: 1.268 Tri: 0.090 Acc: 0.972 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:323:Epoch[20] Iter[450/786] Loss: 1.352 CE: 1.265 Tri: 0.088 Acc: 0.973 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:324:Epoch[20] Iter[500/786] Loss: 1.347 CE: 1.261 Tri: 0.086 Acc: 0.974 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:325:Epoch[20] Iter[550/786] Loss: 1.338 CE: 1.255 Tri: 0.083 Acc: 0.974 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:326:Epoch[20] Iter[600/786] Loss: 1.330 CE: 1.250 Tri: 0.080 Acc: 0.975 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:327:Epoch[20] Iter[650/786] Loss: 1.322 CE: 1.244 Tri: 0.078 Acc: 0.976 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:328:Epoch[20] Iter[700/786] Loss: 1.314 CE: 1.238 Tri: 0.075 Acc: 0.977 LR: 3.23e-05
logs/agreidv2_baseline_4090.log:329:Epoch[20] done in 91.5s  Loss=1.304 Acc=0.978
logs/agreidv2_baseline_4090.log:331:    [A->G] mAP=72.37  R1=81.58  R5=89.05  mINP=43.54
logs/agreidv2_baseline_4090.log:332:    [G->A] mAP=73.14  R1=82.83  R5=88.85  mINP=36.99
logs/agreidv2_baseline_4090.log:333:    [mean] mAP=72.75  R1=82.20
logs/agreidv2_baseline_4090.log:334:Epoch[21] Iter[50/786] Loss: 1.379 CE: 1.306 Tri: 0.072 Acc: 0.961 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:335:Epoch[21] Iter[100/786] Loss: 1.354 CE: 1.282 Tri: 0.072 Acc: 0.967 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:336:Epoch[21] Iter[150/786] Loss: 1.337 CE: 1.266 Tri: 0.071 Acc: 0.970 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:337:Epoch[21] Iter[200/786] Loss: 1.324 CE: 1.253 Tri: 0.071 Acc: 0.974 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:338:Epoch[21] Iter[250/786] Loss: 1.313 CE: 1.246 Tri: 0.067 Acc: 0.976 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:339:Epoch[21] Iter[300/786] Loss: 1.307 CE: 1.240 Tri: 0.067 Acc: 0.977 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:340:Epoch[21] Iter[350/786] Loss: 1.303 CE: 1.237 Tri: 0.066 Acc: 0.977 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:341:Epoch[21] Iter[400/786] Loss: 1.299 CE: 1.234 Tri: 0.065 Acc: 0.978 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:342:Epoch[21] Iter[450/786] Loss: 1.295 CE: 1.231 Tri: 0.064 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:343:Epoch[21] Iter[500/786] Loss: 1.293 CE: 1.229 Tri: 0.064 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:344:Epoch[21] Iter[550/786] Loss: 1.291 CE: 1.227 Tri: 0.064 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:345:Epoch[21] Iter[600/786] Loss: 1.288 CE: 1.225 Tri: 0.063 Acc: 0.980 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:346:Epoch[21] Iter[650/786] Loss: 1.283 CE: 1.221 Tri: 0.062 Acc: 0.980 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:347:Epoch[21] Iter[700/786] Loss: 1.277 CE: 1.216 Tri: 0.061 Acc: 0.981 LR: 3.17e-05
logs/agreidv2_baseline_4090.log:348:Epoch[21] done in 91.2s  Loss=1.271 Acc=0.982
logs/agreidv2_baseline_4090.log:349:Epoch[22] Iter[50/786] Loss: 1.395 CE: 1.304 Tri: 0.091 Acc: 0.968 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:350:Epoch[22] Iter[100/786] Loss: 1.352 CE: 1.272 Tri: 0.080 Acc: 0.971 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:351:Epoch[22] Iter[150/786] Loss: 1.336 CE: 1.261 Tri: 0.075 Acc: 0.972 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:352:Epoch[22] Iter[200/786] Loss: 1.331 CE: 1.254 Tri: 0.078 Acc: 0.974 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:353:Epoch[22] Iter[250/786] Loss: 1.322 CE: 1.247 Tri: 0.075 Acc: 0.975 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:354:Epoch[22] Iter[300/786] Loss: 1.313 CE: 1.241 Tri: 0.072 Acc: 0.976 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:355:Epoch[22] Iter[350/786] Loss: 1.304 CE: 1.235 Tri: 0.069 Acc: 0.977 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:356:Epoch[22] Iter[400/786] Loss: 1.295 CE: 1.230 Tri: 0.065 Acc: 0.978 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:357:Epoch[22] Iter[450/786] Loss: 1.287 CE: 1.224 Tri: 0.063 Acc: 0.979 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:358:Epoch[22] Iter[500/786] Loss: 1.282 CE: 1.220 Tri: 0.061 Acc: 0.979 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:359:Epoch[22] Iter[550/786] Loss: 1.280 CE: 1.218 Tri: 0.062 Acc: 0.979 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:360:Epoch[22] Iter[600/786] Loss: 1.276 CE: 1.216 Tri: 0.061 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:361:Epoch[22] Iter[650/786] Loss: 1.271 CE: 1.212 Tri: 0.059 Acc: 0.981 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:362:Epoch[22] Iter[700/786] Loss: 1.266 CE: 1.208 Tri: 0.058 Acc: 0.981 LR: 3.10e-05
logs/agreidv2_baseline_4090.log:363:Epoch[22] done in 90.7s  Loss=1.260 Acc=0.982
logs/agreidv2_baseline_4090.log:364:Epoch[23] Iter[50/786] Loss: 1.327 CE: 1.261 Tri: 0.066 Acc: 0.973 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:365:Epoch[23] Iter[100/786] Loss: 1.302 CE: 1.242 Tri: 0.060 Acc: 0.975 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:366:Epoch[23] Iter[150/786] Loss: 1.295 CE: 1.235 Tri: 0.061 Acc: 0.976 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:367:Epoch[23] Iter[200/786] Loss: 1.289 CE: 1.230 Tri: 0.059 Acc: 0.977 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:368:Epoch[23] Iter[250/786] Loss: 1.285 CE: 1.225 Tri: 0.060 Acc: 0.978 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:369:Epoch[23] Iter[300/786] Loss: 1.279 CE: 1.221 Tri: 0.058 Acc: 0.979 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:370:Epoch[23] Iter[350/786] Loss: 1.274 CE: 1.217 Tri: 0.057 Acc: 0.979 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:371:Epoch[23] Iter[400/786] Loss: 1.269 CE: 1.214 Tri: 0.056 Acc: 0.980 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:372:Epoch[23] Iter[450/786] Loss: 1.266 CE: 1.212 Tri: 0.054 Acc: 0.980 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:373:Epoch[23] Iter[500/786] Loss: 1.263 CE: 1.209 Tri: 0.054 Acc: 0.981 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:374:Epoch[23] Iter[550/786] Loss: 1.260 CE: 1.207 Tri: 0.053 Acc: 0.981 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:375:Epoch[23] Iter[600/786] Loss: 1.255 CE: 1.203 Tri: 0.052 Acc: 0.982 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:376:Epoch[23] Iter[650/786] Loss: 1.253 CE: 1.201 Tri: 0.052 Acc: 0.982 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:377:Epoch[23] Iter[700/786] Loss: 1.249 CE: 1.197 Tri: 0.052 Acc: 0.983 LR: 3.03e-05
logs/agreidv2_baseline_4090.log:378:Epoch[23] done in 91.6s  Loss=1.243 Acc=0.983
logs/agreidv2_baseline_4090.log:379:Epoch[24] Iter[50/786] Loss: 1.324 CE: 1.263 Tri: 0.061 Acc: 0.978 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:380:Epoch[24] Iter[100/786] Loss: 1.310 CE: 1.245 Tri: 0.066 Acc: 0.978 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:381:Epoch[24] Iter[150/786] Loss: 1.307 CE: 1.238 Tri: 0.069 Acc: 0.978 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:382:Epoch[24] Iter[200/786] Loss: 1.289 CE: 1.228 Tri: 0.061 Acc: 0.980 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:383:Epoch[24] Iter[250/786] Loss: 1.287 CE: 1.225 Tri: 0.062 Acc: 0.980 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:384:Epoch[24] Iter[300/786] Loss: 1.283 CE: 1.222 Tri: 0.062 Acc: 0.981 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:385:Epoch[24] Iter[350/786] Loss: 1.279 CE: 1.218 Tri: 0.061 Acc: 0.981 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:386:Epoch[24] Iter[400/786] Loss: 1.275 CE: 1.216 Tri: 0.059 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:387:Epoch[24] Iter[450/786] Loss: 1.271 CE: 1.213 Tri: 0.058 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:388:Epoch[24] Iter[500/786] Loss: 1.268 CE: 1.211 Tri: 0.057 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:389:Epoch[24] Iter[550/786] Loss: 1.263 CE: 1.208 Tri: 0.055 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:390:Epoch[24] Iter[600/786] Loss: 1.260 CE: 1.205 Tri: 0.055 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:391:Epoch[24] Iter[650/786] Loss: 1.255 CE: 1.201 Tri: 0.054 Acc: 0.983 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:392:Epoch[24] Iter[700/786] Loss: 1.250 CE: 1.197 Tri: 0.053 Acc: 0.983 LR: 2.95e-05
logs/agreidv2_baseline_4090.log:393:Epoch[24] done in 91.3s  Loss=1.243 Acc=0.984
logs/agreidv2_baseline_4090.log:394:Epoch[25] Iter[50/786] Loss: 1.310 CE: 1.250 Tri: 0.060 Acc: 0.974 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:395:Epoch[25] Iter[100/786] Loss: 1.290 CE: 1.232 Tri: 0.058 Acc: 0.977 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:396:Epoch[25] Iter[150/786] Loss: 1.281 CE: 1.223 Tri: 0.058 Acc: 0.979 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:397:Epoch[25] Iter[200/786] Loss: 1.269 CE: 1.214 Tri: 0.055 Acc: 0.981 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:398:Epoch[25] Iter[250/786] Loss: 1.264 CE: 1.211 Tri: 0.053 Acc: 0.980 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:399:Epoch[25] Iter[300/786] Loss: 1.256 CE: 1.206 Tri: 0.050 Acc: 0.981 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:400:Epoch[25] Iter[350/786] Loss: 1.253 CE: 1.203 Tri: 0.050 Acc: 0.982 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:401:Epoch[25] Iter[400/786] Loss: 1.248 CE: 1.199 Tri: 0.049 Acc: 0.982 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:402:Epoch[25] Iter[450/786] Loss: 1.246 CE: 1.197 Tri: 0.049 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:403:Epoch[25] Iter[500/786] Loss: 1.242 CE: 1.194 Tri: 0.048 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:404:Epoch[25] Iter[550/786] Loss: 1.238 CE: 1.190 Tri: 0.048 Acc: 0.984 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:405:Epoch[25] Iter[600/786] Loss: 1.234 CE: 1.187 Tri: 0.047 Acc: 0.984 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:406:Epoch[25] Iter[650/786] Loss: 1.229 CE: 1.183 Tri: 0.045 Acc: 0.985 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:407:Epoch[25] Iter[700/786] Loss: 1.225 CE: 1.180 Tri: 0.045 Acc: 0.985 LR: 2.87e-05
logs/agreidv2_baseline_4090.log:408:Epoch[25] done in 90.9s  Loss=1.219 Acc=0.986
logs/agreidv2_baseline_4090.log:409:Epoch[26] Iter[50/786] Loss: 1.287 CE: 1.227 Tri: 0.060 Acc: 0.977 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:410:Epoch[26] Iter[100/786] Loss: 1.273 CE: 1.217 Tri: 0.056 Acc: 0.978 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:411:Epoch[26] Iter[150/786] Loss: 1.260 CE: 1.206 Tri: 0.053 Acc: 0.981 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:412:Epoch[26] Iter[200/786] Loss: 1.247 CE: 1.197 Tri: 0.050 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:413:Epoch[26] Iter[250/786] Loss: 1.247 CE: 1.195 Tri: 0.052 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:414:Epoch[26] Iter[300/786] Loss: 1.242 CE: 1.191 Tri: 0.050 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:415:Epoch[26] Iter[350/786] Loss: 1.239 CE: 1.189 Tri: 0.050 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:416:Epoch[26] Iter[400/786] Loss: 1.236 CE: 1.187 Tri: 0.049 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:417:Epoch[26] Iter[450/786] Loss: 1.232 CE: 1.185 Tri: 0.047 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:418:Epoch[26] Iter[500/786] Loss: 1.228 CE: 1.182 Tri: 0.046 Acc: 0.985 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:419:Epoch[26] Iter[550/786] Loss: 1.224 CE: 1.179 Tri: 0.045 Acc: 0.985 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:420:Epoch[26] Iter[600/786] Loss: 1.218 CE: 1.175 Tri: 0.043 Acc: 0.986 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:421:Epoch[26] Iter[650/786] Loss: 1.214 CE: 1.172 Tri: 0.042 Acc: 0.986 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:422:Epoch[26] Iter[700/786] Loss: 1.209 CE: 1.169 Tri: 0.041 Acc: 0.987 LR: 2.78e-05
logs/agreidv2_baseline_4090.log:423:Epoch[26] done in 91.4s  Loss=1.203 Acc=0.987
logs/agreidv2_baseline_4090.log:424:Epoch[27] Iter[50/786] Loss: 1.275 CE: 1.219 Tri: 0.056 Acc: 0.978 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:425:Epoch[27] Iter[100/786] Loss: 1.250 CE: 1.202 Tri: 0.048 Acc: 0.982 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:426:Epoch[27] Iter[150/786] Loss: 1.248 CE: 1.197 Tri: 0.051 Acc: 0.983 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:427:Epoch[27] Iter[200/786] Loss: 1.238 CE: 1.191 Tri: 0.047 Acc: 0.984 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:428:Epoch[27] Iter[250/786] Loss: 1.230 CE: 1.185 Tri: 0.045 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:429:Epoch[27] Iter[300/786] Loss: 1.224 CE: 1.181 Tri: 0.043 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:430:Epoch[27] Iter[350/786] Loss: 1.219 CE: 1.178 Tri: 0.041 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:431:Epoch[27] Iter[400/786] Loss: 1.215 CE: 1.175 Tri: 0.040 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:432:Epoch[27] Iter[450/786] Loss: 1.214 CE: 1.173 Tri: 0.041 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:433:Epoch[27] Iter[500/786] Loss: 1.214 CE: 1.173 Tri: 0.041 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:434:Epoch[27] Iter[550/786] Loss: 1.212 CE: 1.171 Tri: 0.041 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:435:Epoch[27] Iter[600/786] Loss: 1.208 CE: 1.169 Tri: 0.040 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:436:Epoch[27] Iter[650/786] Loss: 1.203 CE: 1.165 Tri: 0.038 Acc: 0.987 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:437:Epoch[27] Iter[700/786] Loss: 1.198 CE: 1.161 Tri: 0.037 Acc: 0.987 LR: 2.69e-05
logs/agreidv2_baseline_4090.log:438:Epoch[27] done in 91.1s  Loss=1.192 Acc=0.988
logs/agreidv2_baseline_4090.log:439:Epoch[28] Iter[50/786] Loss: 1.245 CE: 1.204 Tri: 0.041 Acc: 0.981 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:440:Epoch[28] Iter[100/786] Loss: 1.237 CE: 1.199 Tri: 0.038 Acc: 0.982 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:441:Epoch[28] Iter[150/786] Loss: 1.239 CE: 1.196 Tri: 0.042 Acc: 0.981 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:442:Epoch[28] Iter[200/786] Loss: 1.241 CE: 1.194 Tri: 0.047 Acc: 0.981 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:443:Epoch[28] Iter[250/786] Loss: 1.240 CE: 1.191 Tri: 0.049 Acc: 0.982 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:444:Epoch[28] Iter[300/786] Loss: 1.236 CE: 1.187 Tri: 0.048 Acc: 0.983 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:445:Epoch[28] Iter[350/786] Loss: 1.233 CE: 1.184 Tri: 0.049 Acc: 0.983 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:446:Epoch[28] Iter[400/786] Loss: 1.228 CE: 1.180 Tri: 0.048 Acc: 0.984 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:447:Epoch[28] Iter[450/786] Loss: 1.223 CE: 1.176 Tri: 0.046 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:448:Epoch[28] Iter[500/786] Loss: 1.219 CE: 1.173 Tri: 0.046 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:449:Epoch[28] Iter[550/786] Loss: 1.215 CE: 1.171 Tri: 0.045 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:450:Epoch[28] Iter[600/786] Loss: 1.210 CE: 1.167 Tri: 0.043 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:451:Epoch[28] Iter[650/786] Loss: 1.207 CE: 1.165 Tri: 0.042 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:452:Epoch[28] Iter[700/786] Loss: 1.202 CE: 1.161 Tri: 0.041 Acc: 0.987 LR: 2.59e-05
logs/agreidv2_baseline_4090.log:453:Epoch[28] done in 91.6s  Loss=1.196 Acc=0.988
logs/agreidv2_baseline_4090.log:454:Epoch[29] Iter[50/786] Loss: 1.254 CE: 1.204 Tri: 0.050 Acc: 0.982 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:455:Epoch[29] Iter[100/786] Loss: 1.233 CE: 1.191 Tri: 0.042 Acc: 0.984 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:456:Epoch[29] Iter[150/786] Loss: 1.222 CE: 1.182 Tri: 0.041 Acc: 0.985 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:457:Epoch[29] Iter[200/786] Loss: 1.216 CE: 1.175 Tri: 0.041 Acc: 0.985 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:458:Epoch[29] Iter[250/786] Loss: 1.213 CE: 1.172 Tri: 0.041 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:459:Epoch[29] Iter[300/786] Loss: 1.207 CE: 1.168 Tri: 0.039 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:460:Epoch[29] Iter[350/786] Loss: 1.203 CE: 1.165 Tri: 0.038 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:461:Epoch[29] Iter[400/786] Loss: 1.199 CE: 1.162 Tri: 0.036 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:462:Epoch[29] Iter[450/786] Loss: 1.195 CE: 1.159 Tri: 0.036 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:463:Epoch[29] Iter[500/786] Loss: 1.193 CE: 1.158 Tri: 0.036 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:464:Epoch[29] Iter[550/786] Loss: 1.191 CE: 1.155 Tri: 0.036 Acc: 0.988 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:465:Epoch[29] Iter[600/786] Loss: 1.188 CE: 1.153 Tri: 0.035 Acc: 0.988 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:466:Epoch[29] Iter[650/786] Loss: 1.186 CE: 1.151 Tri: 0.035 Acc: 0.988 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:467:Epoch[29] Iter[700/786] Loss: 1.182 CE: 1.148 Tri: 0.034 Acc: 0.988 LR: 2.50e-05
logs/agreidv2_baseline_4090.log:468:Epoch[29] done in 91.5s  Loss=1.177 Acc=0.989
logs/agreidv2_baseline_4090.log:469:Epoch[30] Iter[50/786] Loss: 1.219 CE: 1.181 Tri: 0.038 Acc: 0.984 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:470:Epoch[30] Iter[100/786] Loss: 1.198 CE: 1.169 Tri: 0.029 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:471:Epoch[30] Iter[150/786] Loss: 1.199 CE: 1.163 Tri: 0.036 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:472:Epoch[30] Iter[200/786] Loss: 1.199 CE: 1.163 Tri: 0.036 Acc: 0.985 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:473:Epoch[30] Iter[250/786] Loss: 1.198 CE: 1.161 Tri: 0.038 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:474:Epoch[30] Iter[300/786] Loss: 1.194 CE: 1.157 Tri: 0.036 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:475:Epoch[30] Iter[350/786] Loss: 1.191 CE: 1.155 Tri: 0.036 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:476:Epoch[30] Iter[400/786] Loss: 1.188 CE: 1.154 Tri: 0.035 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:477:Epoch[30] Iter[450/786] Loss: 1.186 CE: 1.152 Tri: 0.034 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:478:Epoch[30] Iter[500/786] Loss: 1.184 CE: 1.150 Tri: 0.034 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:479:Epoch[30] Iter[550/786] Loss: 1.182 CE: 1.148 Tri: 0.034 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:480:Epoch[30] Iter[600/786] Loss: 1.179 CE: 1.146 Tri: 0.033 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:481:Epoch[30] Iter[650/786] Loss: 1.176 CE: 1.144 Tri: 0.032 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:482:Epoch[30] Iter[700/786] Loss: 1.173 CE: 1.141 Tri: 0.032 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_baseline_4090.log:483:Epoch[30] done in 91.4s  Loss=1.167 Acc=0.989
logs/agreidv2_baseline_4090.log:485:    [A->G] mAP=71.72  R1=80.60  R5=87.90  mINP=43.02
logs/agreidv2_baseline_4090.log:486:    [G->A] mAP=73.24  R1=83.16  R5=88.96  mINP=35.74
logs/agreidv2_baseline_4090.log:487:    [mean] mAP=72.48  R1=81.88
logs/agreidv2_baseline_4090.log:488:Epoch[31] Iter[50/786] Loss: 1.214 CE: 1.184 Tri: 0.031 Acc: 0.983 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:489:Epoch[31] Iter[100/786] Loss: 1.198 CE: 1.170 Tri: 0.028 Acc: 0.984 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:490:Epoch[31] Iter[150/786] Loss: 1.194 CE: 1.165 Tri: 0.029 Acc: 0.984 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:491:Epoch[31] Iter[200/786] Loss: 1.188 CE: 1.160 Tri: 0.027 Acc: 0.985 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:492:Epoch[31] Iter[250/786] Loss: 1.186 CE: 1.157 Tri: 0.028 Acc: 0.986 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:493:Epoch[31] Iter[300/786] Loss: 1.184 CE: 1.155 Tri: 0.029 Acc: 0.986 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:494:Epoch[31] Iter[350/786] Loss: 1.183 CE: 1.153 Tri: 0.030 Acc: 0.986 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:495:Epoch[31] Iter[400/786] Loss: 1.179 CE: 1.150 Tri: 0.029 Acc: 0.987 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:496:Epoch[31] Iter[450/786] Loss: 1.175 CE: 1.147 Tri: 0.028 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:497:Epoch[31] Iter[500/786] Loss: 1.173 CE: 1.145 Tri: 0.027 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:498:Epoch[31] Iter[550/786] Loss: 1.170 CE: 1.143 Tri: 0.027 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:499:Epoch[31] Iter[600/786] Loss: 1.167 CE: 1.141 Tri: 0.026 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:500:Epoch[31] Iter[650/786] Loss: 1.163 CE: 1.138 Tri: 0.026 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:501:Epoch[31] Iter[700/786] Loss: 1.160 CE: 1.135 Tri: 0.025 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_baseline_4090.log:502:Epoch[31] done in 90.5s  Loss=1.156 Acc=0.990
logs/agreidv2_baseline_4090.log:503:Epoch[32] Iter[50/786] Loss: 1.188 CE: 1.161 Tri: 0.027 Acc: 0.985 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:504:Epoch[32] Iter[100/786] Loss: 1.178 CE: 1.152 Tri: 0.026 Acc: 0.988 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:505:Epoch[32] Iter[150/786] Loss: 1.177 CE: 1.148 Tri: 0.029 Acc: 0.987 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:506:Epoch[32] Iter[200/786] Loss: 1.176 CE: 1.145 Tri: 0.030 Acc: 0.988 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:507:Epoch[32] Iter[250/786] Loss: 1.171 CE: 1.142 Tri: 0.028 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:508:Epoch[32] Iter[300/786] Loss: 1.172 CE: 1.141 Tri: 0.030 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:509:Epoch[32] Iter[350/786] Loss: 1.167 CE: 1.139 Tri: 0.028 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:510:Epoch[32] Iter[400/786] Loss: 1.165 CE: 1.136 Tri: 0.028 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:511:Epoch[32] Iter[450/786] Loss: 1.164 CE: 1.135 Tri: 0.028 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:512:Epoch[32] Iter[500/786] Loss: 1.162 CE: 1.134 Tri: 0.028 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:513:Epoch[32] Iter[550/786] Loss: 1.160 CE: 1.133 Tri: 0.027 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:514:Epoch[32] Iter[600/786] Loss: 1.157 CE: 1.131 Tri: 0.026 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:515:Epoch[32] Iter[650/786] Loss: 1.155 CE: 1.129 Tri: 0.026 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:516:Epoch[32] Iter[700/786] Loss: 1.151 CE: 1.126 Tri: 0.025 Acc: 0.991 LR: 2.19e-05
logs/agreidv2_baseline_4090.log:517:Epoch[32] done in 91.6s  Loss=1.146 Acc=0.991
logs/agreidv2_baseline_4090.log:518:Epoch[33] Iter[50/786] Loss: 1.187 CE: 1.160 Tri: 0.028 Acc: 0.986 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:519:Epoch[33] Iter[100/786] Loss: 1.172 CE: 1.149 Tri: 0.023 Acc: 0.987 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:520:Epoch[33] Iter[150/786] Loss: 1.163 CE: 1.142 Tri: 0.021 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:521:Epoch[33] Iter[200/786] Loss: 1.155 CE: 1.136 Tri: 0.019 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:522:Epoch[33] Iter[250/786] Loss: 1.157 CE: 1.136 Tri: 0.021 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:523:Epoch[33] Iter[300/786] Loss: 1.156 CE: 1.134 Tri: 0.021 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:524:Epoch[33] Iter[350/786] Loss: 1.159 CE: 1.134 Tri: 0.025 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:525:Epoch[33] Iter[400/786] Loss: 1.156 CE: 1.133 Tri: 0.023 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:526:Epoch[33] Iter[450/786] Loss: 1.158 CE: 1.133 Tri: 0.025 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:527:Epoch[33] Iter[500/786] Loss: 1.157 CE: 1.132 Tri: 0.025 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:528:Epoch[33] Iter[550/786] Loss: 1.156 CE: 1.130 Tri: 0.026 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:529:Epoch[33] Iter[600/786] Loss: 1.154 CE: 1.129 Tri: 0.025 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:530:Epoch[33] Iter[650/786] Loss: 1.151 CE: 1.127 Tri: 0.025 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:531:Epoch[33] Iter[700/786] Loss: 1.148 CE: 1.124 Tri: 0.024 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_baseline_4090.log:532:Epoch[33] done in 91.6s  Loss=1.143 Acc=0.991
logs/agreidv2_baseline_4090.log:533:Epoch[34] Iter[50/786] Loss: 1.187 CE: 1.160 Tri: 0.027 Acc: 0.986 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:534:Epoch[34] Iter[100/786] Loss: 1.172 CE: 1.148 Tri: 0.024 Acc: 0.987 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:535:Epoch[34] Iter[150/786] Loss: 1.177 CE: 1.147 Tri: 0.030 Acc: 0.987 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:536:Epoch[34] Iter[200/786] Loss: 1.168 CE: 1.141 Tri: 0.026 Acc: 0.988 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:537:Epoch[34] Iter[250/786] Loss: 1.161 CE: 1.137 Tri: 0.024 Acc: 0.988 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:538:Epoch[34] Iter[300/786] Loss: 1.155 CE: 1.133 Tri: 0.022 Acc: 0.988 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:539:Epoch[34] Iter[350/786] Loss: 1.152 CE: 1.130 Tri: 0.022 Acc: 0.989 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:540:Epoch[34] Iter[400/786] Loss: 1.150 CE: 1.128 Tri: 0.022 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:541:Epoch[34] Iter[450/786] Loss: 1.147 CE: 1.125 Tri: 0.022 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:542:Epoch[34] Iter[500/786] Loss: 1.144 CE: 1.123 Tri: 0.021 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:543:Epoch[34] Iter[550/786] Loss: 1.143 CE: 1.121 Tri: 0.021 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:544:Epoch[34] Iter[600/786] Loss: 1.141 CE: 1.120 Tri: 0.021 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:545:Epoch[34] Iter[650/786] Loss: 1.139 CE: 1.118 Tri: 0.021 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:546:Epoch[34] Iter[700/786] Loss: 1.136 CE: 1.116 Tri: 0.020 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline_4090.log:547:Epoch[34] done in 91.6s  Loss=1.133 Acc=0.992
logs/agreidv2_baseline_4090.log:548:Epoch[35] Iter[50/786] Loss: 1.173 CE: 1.149 Tri: 0.023 Acc: 0.986 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:549:Epoch[35] Iter[100/786] Loss: 1.162 CE: 1.138 Tri: 0.024 Acc: 0.988 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:550:Epoch[35] Iter[150/786] Loss: 1.157 CE: 1.134 Tri: 0.022 Acc: 0.987 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:551:Epoch[35] Iter[200/786] Loss: 1.149 CE: 1.129 Tri: 0.019 Acc: 0.988 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:552:Epoch[35] Iter[250/786] Loss: 1.144 CE: 1.126 Tri: 0.018 Acc: 0.989 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:553:Epoch[35] Iter[300/786] Loss: 1.140 CE: 1.123 Tri: 0.017 Acc: 0.990 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:554:Epoch[35] Iter[350/786] Loss: 1.138 CE: 1.120 Tri: 0.017 Acc: 0.990 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:555:Epoch[35] Iter[400/786] Loss: 1.136 CE: 1.119 Tri: 0.017 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:556:Epoch[35] Iter[450/786] Loss: 1.136 CE: 1.118 Tri: 0.017 Acc: 0.990 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:557:Epoch[35] Iter[500/786] Loss: 1.133 CE: 1.116 Tri: 0.017 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:558:Epoch[35] Iter[550/786] Loss: 1.131 CE: 1.115 Tri: 0.016 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:559:Epoch[35] Iter[600/786] Loss: 1.129 CE: 1.113 Tri: 0.016 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:560:Epoch[35] Iter[650/786] Loss: 1.128 CE: 1.111 Tri: 0.017 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:561:Epoch[35] Iter[700/786] Loss: 1.126 CE: 1.109 Tri: 0.017 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_baseline_4090.log:562:Epoch[35] done in 91.6s  Loss=1.122 Acc=0.992
logs/agreidv2_baseline_4090.log:563:Epoch[36] Iter[50/786] Loss: 1.162 CE: 1.141 Tri: 0.020 Acc: 0.989 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:564:Epoch[36] Iter[100/786] Loss: 1.156 CE: 1.135 Tri: 0.021 Acc: 0.989 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:565:Epoch[36] Iter[150/786] Loss: 1.151 CE: 1.129 Tri: 0.021 Acc: 0.989 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:566:Epoch[36] Iter[200/786] Loss: 1.145 CE: 1.125 Tri: 0.020 Acc: 0.990 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:567:Epoch[36] Iter[250/786] Loss: 1.142 CE: 1.121 Tri: 0.021 Acc: 0.990 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:568:Epoch[36] Iter[300/786] Loss: 1.138 CE: 1.119 Tri: 0.020 Acc: 0.990 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:569:Epoch[36] Iter[350/786] Loss: 1.134 CE: 1.116 Tri: 0.018 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:570:Epoch[36] Iter[400/786] Loss: 1.130 CE: 1.113 Tri: 0.017 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:571:Epoch[36] Iter[450/786] Loss: 1.128 CE: 1.111 Tri: 0.017 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:572:Epoch[36] Iter[500/786] Loss: 1.126 CE: 1.110 Tri: 0.017 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:573:Epoch[36] Iter[550/786] Loss: 1.124 CE: 1.108 Tri: 0.016 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:574:Epoch[36] Iter[600/786] Loss: 1.123 CE: 1.107 Tri: 0.016 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:575:Epoch[36] Iter[650/786] Loss: 1.120 CE: 1.105 Tri: 0.016 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:576:Epoch[36] Iter[700/786] Loss: 1.119 CE: 1.103 Tri: 0.016 Acc: 0.993 LR: 1.75e-05
logs/agreidv2_baseline_4090.log:577:Epoch[36] done in 90.8s  Loss=1.116 Acc=0.993
logs/agreidv2_baseline_4090.log:578:Epoch[37] Iter[50/786] Loss: 1.152 CE: 1.129 Tri: 0.023 Acc: 0.990 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:579:Epoch[37] Iter[100/786] Loss: 1.141 CE: 1.121 Tri: 0.020 Acc: 0.990 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:580:Epoch[37] Iter[150/786] Loss: 1.133 CE: 1.114 Tri: 0.019 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:581:Epoch[37] Iter[200/786] Loss: 1.129 CE: 1.111 Tri: 0.018 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:582:Epoch[37] Iter[250/786] Loss: 1.126 CE: 1.108 Tri: 0.018 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:583:Epoch[37] Iter[300/786] Loss: 1.129 CE: 1.109 Tri: 0.020 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:584:Epoch[37] Iter[350/786] Loss: 1.126 CE: 1.108 Tri: 0.019 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:585:Epoch[37] Iter[400/786] Loss: 1.124 CE: 1.105 Tri: 0.018 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:586:Epoch[37] Iter[450/786] Loss: 1.123 CE: 1.104 Tri: 0.019 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:587:Epoch[37] Iter[500/786] Loss: 1.120 CE: 1.103 Tri: 0.018 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:588:Epoch[37] Iter[550/786] Loss: 1.119 CE: 1.101 Tri: 0.017 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:589:Epoch[37] Iter[600/786] Loss: 1.117 CE: 1.100 Tri: 0.017 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:590:Epoch[37] Iter[650/786] Loss: 1.115 CE: 1.098 Tri: 0.017 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:591:Epoch[37] Iter[700/786] Loss: 1.112 CE: 1.096 Tri: 0.016 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_baseline_4090.log:592:Epoch[37] done in 91.7s  Loss=1.109 Acc=0.993
logs/agreidv2_baseline_4090.log:593:Epoch[38] Iter[50/786] Loss: 1.132 CE: 1.117 Tri: 0.015 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:594:Epoch[38] Iter[100/786] Loss: 1.133 CE: 1.115 Tri: 0.018 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:595:Epoch[38] Iter[150/786] Loss: 1.127 CE: 1.110 Tri: 0.017 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:596:Epoch[38] Iter[200/786] Loss: 1.123 CE: 1.107 Tri: 0.016 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:597:Epoch[38] Iter[250/786] Loss: 1.119 CE: 1.104 Tri: 0.015 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:598:Epoch[38] Iter[300/786] Loss: 1.119 CE: 1.103 Tri: 0.016 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:599:Epoch[38] Iter[350/786] Loss: 1.118 CE: 1.102 Tri: 0.016 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:600:Epoch[38] Iter[400/786] Loss: 1.117 CE: 1.101 Tri: 0.016 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:601:Epoch[38] Iter[450/786] Loss: 1.116 CE: 1.101 Tri: 0.015 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:602:Epoch[38] Iter[500/786] Loss: 1.114 CE: 1.099 Tri: 0.015 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:603:Epoch[38] Iter[550/786] Loss: 1.112 CE: 1.097 Tri: 0.014 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:604:Epoch[38] Iter[600/786] Loss: 1.110 CE: 1.096 Tri: 0.014 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:605:Epoch[38] Iter[650/786] Loss: 1.108 CE: 1.094 Tri: 0.014 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:606:Epoch[38] Iter[700/786] Loss: 1.106 CE: 1.092 Tri: 0.013 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline_4090.log:607:Epoch[38] done in 92.1s  Loss=1.103 Acc=0.994
logs/agreidv2_baseline_4090.log:608:Epoch[39] Iter[50/786] Loss: 1.139 CE: 1.125 Tri: 0.015 Acc: 0.989 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:609:Epoch[39] Iter[100/786] Loss: 1.136 CE: 1.118 Tri: 0.018 Acc: 0.990 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:610:Epoch[39] Iter[150/786] Loss: 1.125 CE: 1.110 Tri: 0.015 Acc: 0.991 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:611:Epoch[39] Iter[200/786] Loss: 1.124 CE: 1.107 Tri: 0.017 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:612:Epoch[39] Iter[250/786] Loss: 1.122 CE: 1.105 Tri: 0.016 Acc: 0.991 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:613:Epoch[39] Iter[300/786] Loss: 1.119 CE: 1.103 Tri: 0.016 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:614:Epoch[39] Iter[350/786] Loss: 1.117 CE: 1.101 Tri: 0.016 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:615:Epoch[39] Iter[400/786] Loss: 1.116 CE: 1.100 Tri: 0.016 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:616:Epoch[39] Iter[450/786] Loss: 1.113 CE: 1.098 Tri: 0.016 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:617:Epoch[39] Iter[500/786] Loss: 1.111 CE: 1.096 Tri: 0.015 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:618:Epoch[39] Iter[550/786] Loss: 1.109 CE: 1.094 Tri: 0.015 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:619:Epoch[39] Iter[600/786] Loss: 1.107 CE: 1.093 Tri: 0.014 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:620:Epoch[39] Iter[650/786] Loss: 1.106 CE: 1.091 Tri: 0.014 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:621:Epoch[39] Iter[700/786] Loss: 1.104 CE: 1.090 Tri: 0.014 Acc: 0.994 LR: 1.42e-05
logs/agreidv2_baseline_4090.log:622:Epoch[39] done in 91.6s  Loss=1.101 Acc=0.994
logs/agreidv2_baseline_4090.log:623:Epoch[40] Iter[50/786] Loss: 1.117 CE: 1.106 Tri: 0.011 Acc: 0.990 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:624:Epoch[40] Iter[100/786] Loss: 1.120 CE: 1.103 Tri: 0.017 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:625:Epoch[40] Iter[150/786] Loss: 1.117 CE: 1.100 Tri: 0.017 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:626:Epoch[40] Iter[200/786] Loss: 1.117 CE: 1.099 Tri: 0.018 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:627:Epoch[40] Iter[250/786] Loss: 1.115 CE: 1.097 Tri: 0.018 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:628:Epoch[40] Iter[300/786] Loss: 1.113 CE: 1.096 Tri: 0.017 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:629:Epoch[40] Iter[350/786] Loss: 1.110 CE: 1.094 Tri: 0.016 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:630:Epoch[40] Iter[400/786] Loss: 1.107 CE: 1.093 Tri: 0.015 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:631:Epoch[40] Iter[450/786] Loss: 1.108 CE: 1.092 Tri: 0.016 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:632:Epoch[40] Iter[500/786] Loss: 1.105 CE: 1.091 Tri: 0.015 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:633:Epoch[40] Iter[550/786] Loss: 1.103 CE: 1.089 Tri: 0.015 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:634:Epoch[40] Iter[600/786] Loss: 1.102 CE: 1.088 Tri: 0.014 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:635:Epoch[40] Iter[650/786] Loss: 1.100 CE: 1.086 Tri: 0.014 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:636:Epoch[40] Iter[700/786] Loss: 1.099 CE: 1.085 Tri: 0.014 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline_4090.log:637:Epoch[40] done in 91.1s  Loss=1.096 Acc=0.994
logs/agreidv2_baseline_4090.log:639:    [A->G] mAP=76.21  R1=83.83  R5=91.00  mINP=48.95
logs/agreidv2_baseline_4090.log:640:    [G->A] mAP=77.13  R1=85.42  R5=91.17  mINP=42.14
logs/agreidv2_baseline_4090.log:641:    [mean] mAP=76.67  R1=84.63
logs/agreidv2_baseline_4090.log:642:    * new best mean mAP=76.67 (epoch 40) saved
logs/agreidv2_baseline_4090.log:643:Epoch[41] Iter[50/786] Loss: 1.131 CE: 1.108 Tri: 0.023 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:644:Epoch[41] Iter[100/786] Loss: 1.120 CE: 1.102 Tri: 0.018 Acc: 0.992 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:645:Epoch[41] Iter[150/786] Loss: 1.112 CE: 1.097 Tri: 0.015 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:646:Epoch[41] Iter[200/786] Loss: 1.110 CE: 1.094 Tri: 0.015 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:647:Epoch[41] Iter[250/786] Loss: 1.108 CE: 1.093 Tri: 0.014 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:648:Epoch[41] Iter[300/786] Loss: 1.106 CE: 1.091 Tri: 0.014 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:649:Epoch[41] Iter[350/786] Loss: 1.104 CE: 1.090 Tri: 0.014 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:650:Epoch[41] Iter[400/786] Loss: 1.103 CE: 1.089 Tri: 0.014 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:651:Epoch[41] Iter[450/786] Loss: 1.102 CE: 1.087 Tri: 0.014 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:652:Epoch[41] Iter[500/786] Loss: 1.099 CE: 1.086 Tri: 0.013 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:653:Epoch[41] Iter[550/786] Loss: 1.098 CE: 1.085 Tri: 0.013 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:654:Epoch[41] Iter[600/786] Loss: 1.096 CE: 1.083 Tri: 0.013 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:655:Epoch[41] Iter[650/786] Loss: 1.095 CE: 1.082 Tri: 0.012 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:656:Epoch[41] Iter[700/786] Loss: 1.092 CE: 1.080 Tri: 0.012 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline_4090.log:657:Epoch[41] done in 91.9s  Loss=1.089 Acc=0.995
logs/agreidv2_baseline_4090.log:658:Epoch[42] Iter[50/786] Loss: 1.101 CE: 1.093 Tri: 0.008 Acc: 0.991 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:659:Epoch[42] Iter[100/786] Loss: 1.109 CE: 1.091 Tri: 0.017 Acc: 0.990 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:660:Epoch[42] Iter[150/786] Loss: 1.105 CE: 1.090 Tri: 0.015 Acc: 0.991 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:661:Epoch[42] Iter[200/786] Loss: 1.102 CE: 1.089 Tri: 0.013 Acc: 0.992 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:662:Epoch[42] Iter[250/786] Loss: 1.100 CE: 1.087 Tri: 0.013 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:663:Epoch[42] Iter[300/786] Loss: 1.099 CE: 1.086 Tri: 0.013 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:664:Epoch[42] Iter[350/786] Loss: 1.098 CE: 1.085 Tri: 0.013 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:665:Epoch[42] Iter[400/786] Loss: 1.096 CE: 1.084 Tri: 0.012 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:666:Epoch[42] Iter[450/786] Loss: 1.094 CE: 1.083 Tri: 0.012 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:667:Epoch[42] Iter[500/786] Loss: 1.093 CE: 1.082 Tri: 0.011 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:668:Epoch[42] Iter[550/786] Loss: 1.092 CE: 1.081 Tri: 0.011 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:669:Epoch[42] Iter[600/786] Loss: 1.090 CE: 1.079 Tri: 0.011 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:670:Epoch[42] Iter[650/786] Loss: 1.088 CE: 1.078 Tri: 0.010 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:671:Epoch[42] Iter[700/786] Loss: 1.086 CE: 1.076 Tri: 0.010 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline_4090.log:672:Epoch[42] done in 91.0s  Loss=1.084 Acc=0.995
logs/agreidv2_baseline_4090.log:673:Epoch[43] Iter[50/786] Loss: 1.100 CE: 1.090 Tri: 0.010 Acc: 0.992 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:674:Epoch[43] Iter[100/786] Loss: 1.092 CE: 1.085 Tri: 0.007 Acc: 0.993 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:675:Epoch[43] Iter[150/786] Loss: 1.091 CE: 1.084 Tri: 0.008 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:676:Epoch[43] Iter[200/786] Loss: 1.091 CE: 1.081 Tri: 0.010 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:677:Epoch[43] Iter[250/786] Loss: 1.091 CE: 1.082 Tri: 0.010 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:678:Epoch[43] Iter[300/786] Loss: 1.091 CE: 1.081 Tri: 0.010 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:679:Epoch[43] Iter[350/786] Loss: 1.089 CE: 1.079 Tri: 0.010 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:680:Epoch[43] Iter[400/786] Loss: 1.088 CE: 1.078 Tri: 0.010 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:681:Epoch[43] Iter[450/786] Loss: 1.088 CE: 1.078 Tri: 0.010 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:682:Epoch[43] Iter[500/786] Loss: 1.088 CE: 1.078 Tri: 0.010 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:683:Epoch[43] Iter[550/786] Loss: 1.086 CE: 1.077 Tri: 0.009 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:684:Epoch[43] Iter[600/786] Loss: 1.085 CE: 1.076 Tri: 0.009 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:685:Epoch[43] Iter[650/786] Loss: 1.083 CE: 1.075 Tri: 0.008 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:686:Epoch[43] Iter[700/786] Loss: 1.081 CE: 1.073 Tri: 0.008 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline_4090.log:687:Epoch[43] done in 91.2s  Loss=1.079 Acc=0.994
logs/agreidv2_baseline_4090.log:688:Epoch[44] Iter[50/786] Loss: 1.096 CE: 1.087 Tri: 0.009 Acc: 0.991 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:689:Epoch[44] Iter[100/786] Loss: 1.093 CE: 1.083 Tri: 0.011 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:690:Epoch[44] Iter[150/786] Loss: 1.092 CE: 1.082 Tri: 0.010 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:691:Epoch[44] Iter[200/786] Loss: 1.093 CE: 1.080 Tri: 0.013 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:692:Epoch[44] Iter[250/786] Loss: 1.093 CE: 1.079 Tri: 0.014 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:693:Epoch[44] Iter[300/786] Loss: 1.090 CE: 1.078 Tri: 0.013 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:694:Epoch[44] Iter[350/786] Loss: 1.089 CE: 1.077 Tri: 0.012 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:695:Epoch[44] Iter[400/786] Loss: 1.087 CE: 1.075 Tri: 0.012 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:696:Epoch[44] Iter[450/786] Loss: 1.086 CE: 1.074 Tri: 0.012 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:697:Epoch[44] Iter[500/786] Loss: 1.085 CE: 1.073 Tri: 0.012 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:698:Epoch[44] Iter[550/786] Loss: 1.083 CE: 1.072 Tri: 0.011 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:699:Epoch[44] Iter[600/786] Loss: 1.082 CE: 1.071 Tri: 0.011 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:700:Epoch[44] Iter[650/786] Loss: 1.080 CE: 1.070 Tri: 0.010 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:701:Epoch[44] Iter[700/786] Loss: 1.079 CE: 1.069 Tri: 0.010 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline_4090.log:702:Epoch[44] done in 91.2s  Loss=1.077 Acc=0.995
logs/agreidv2_baseline_4090.log:703:Epoch[45] Iter[50/786] Loss: 1.101 CE: 1.084 Tri: 0.017 Acc: 0.993 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:704:Epoch[45] Iter[100/786] Loss: 1.094 CE: 1.081 Tri: 0.013 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:705:Epoch[45] Iter[150/786] Loss: 1.090 CE: 1.079 Tri: 0.012 Acc: 0.993 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:706:Epoch[45] Iter[200/786] Loss: 1.086 CE: 1.077 Tri: 0.010 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:707:Epoch[45] Iter[250/786] Loss: 1.084 CE: 1.075 Tri: 0.009 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:708:Epoch[45] Iter[300/786] Loss: 1.083 CE: 1.074 Tri: 0.009 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:709:Epoch[45] Iter[350/786] Loss: 1.080 CE: 1.073 Tri: 0.008 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:710:Epoch[45] Iter[400/786] Loss: 1.080 CE: 1.072 Tri: 0.008 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:711:Epoch[45] Iter[450/786] Loss: 1.079 CE: 1.071 Tri: 0.008 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:712:Epoch[45] Iter[500/786] Loss: 1.077 CE: 1.069 Tri: 0.008 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:713:Epoch[45] Iter[550/786] Loss: 1.076 CE: 1.069 Tri: 0.008 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:714:Epoch[45] Iter[600/786] Loss: 1.075 CE: 1.068 Tri: 0.008 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:715:Epoch[45] Iter[650/786] Loss: 1.074 CE: 1.066 Tri: 0.007 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:716:Epoch[45] Iter[700/786] Loss: 1.072 CE: 1.065 Tri: 0.007 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline_4090.log:717:Epoch[45] done in 91.6s  Loss=1.070 Acc=0.996
logs/agreidv2_baseline_4090.log:718:Epoch[46] Iter[50/786] Loss: 1.086 CE: 1.076 Tri: 0.010 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:719:Epoch[46] Iter[100/786] Loss: 1.089 CE: 1.078 Tri: 0.011 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:720:Epoch[46] Iter[150/786] Loss: 1.087 CE: 1.077 Tri: 0.010 Acc: 0.994 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:721:Epoch[46] Iter[200/786] Loss: 1.083 CE: 1.074 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:722:Epoch[46] Iter[250/786] Loss: 1.082 CE: 1.073 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:723:Epoch[46] Iter[300/786] Loss: 1.081 CE: 1.072 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:724:Epoch[46] Iter[350/786] Loss: 1.078 CE: 1.070 Tri: 0.008 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:725:Epoch[46] Iter[400/786] Loss: 1.078 CE: 1.070 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:726:Epoch[46] Iter[450/786] Loss: 1.079 CE: 1.069 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:727:Epoch[46] Iter[500/786] Loss: 1.078 CE: 1.068 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:728:Epoch[46] Iter[550/786] Loss: 1.077 CE: 1.068 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:729:Epoch[46] Iter[600/786] Loss: 1.075 CE: 1.067 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:730:Epoch[46] Iter[650/786] Loss: 1.074 CE: 1.066 Tri: 0.009 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:731:Epoch[46] Iter[700/786] Loss: 1.072 CE: 1.064 Tri: 0.008 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_baseline_4090.log:732:Epoch[46] done in 90.8s  Loss=1.070 Acc=0.996
logs/agreidv2_baseline_4090.log:733:Epoch[47] Iter[50/786] Loss: 1.082 CE: 1.073 Tri: 0.009 Acc: 0.994 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:734:Epoch[47] Iter[100/786] Loss: 1.079 CE: 1.071 Tri: 0.008 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:735:Epoch[47] Iter[150/786] Loss: 1.078 CE: 1.070 Tri: 0.007 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:736:Epoch[47] Iter[200/786] Loss: 1.076 CE: 1.069 Tri: 0.007 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:737:Epoch[47] Iter[250/786] Loss: 1.075 CE: 1.067 Tri: 0.008 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:738:Epoch[47] Iter[300/786] Loss: 1.075 CE: 1.067 Tri: 0.008 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:739:Epoch[47] Iter[350/786] Loss: 1.075 CE: 1.066 Tri: 0.008 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:740:Epoch[47] Iter[400/786] Loss: 1.075 CE: 1.066 Tri: 0.009 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:741:Epoch[47] Iter[450/786] Loss: 1.073 CE: 1.065 Tri: 0.008 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:742:Epoch[47] Iter[500/786] Loss: 1.073 CE: 1.065 Tri: 0.008 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:743:Epoch[47] Iter[550/786] Loss: 1.072 CE: 1.064 Tri: 0.008 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:744:Epoch[47] Iter[600/786] Loss: 1.070 CE: 1.063 Tri: 0.008 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:745:Epoch[47] Iter[650/786] Loss: 1.069 CE: 1.061 Tri: 0.007 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:746:Epoch[47] Iter[700/786] Loss: 1.067 CE: 1.060 Tri: 0.007 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline_4090.log:747:Epoch[47] done in 91.2s  Loss=1.065 Acc=0.996
logs/agreidv2_baseline_4090.log:748:Epoch[48] Iter[50/786] Loss: 1.079 CE: 1.076 Tri: 0.003 Acc: 0.993 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:749:Epoch[48] Iter[100/786] Loss: 1.076 CE: 1.071 Tri: 0.005 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:750:Epoch[48] Iter[150/786] Loss: 1.075 CE: 1.069 Tri: 0.006 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:751:Epoch[48] Iter[200/786] Loss: 1.073 CE: 1.068 Tri: 0.006 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:752:Epoch[48] Iter[250/786] Loss: 1.073 CE: 1.067 Tri: 0.006 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:753:Epoch[48] Iter[300/786] Loss: 1.072 CE: 1.066 Tri: 0.007 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:754:Epoch[48] Iter[350/786] Loss: 1.071 CE: 1.065 Tri: 0.006 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:755:Epoch[48] Iter[400/786] Loss: 1.070 CE: 1.063 Tri: 0.006 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:756:Epoch[48] Iter[450/786] Loss: 1.069 CE: 1.063 Tri: 0.006 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:757:Epoch[48] Iter[500/786] Loss: 1.068 CE: 1.062 Tri: 0.006 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:758:Epoch[48] Iter[550/786] Loss: 1.066 CE: 1.061 Tri: 0.006 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:759:Epoch[48] Iter[600/786] Loss: 1.065 CE: 1.060 Tri: 0.005 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:760:Epoch[48] Iter[650/786] Loss: 1.064 CE: 1.058 Tri: 0.005 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:761:Epoch[48] Iter[700/786] Loss: 1.063 CE: 1.057 Tri: 0.005 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_baseline_4090.log:762:Epoch[48] done in 90.9s  Loss=1.061 Acc=0.996
logs/agreidv2_baseline_4090.log:763:Epoch[49] Iter[50/786] Loss: 1.083 CE: 1.072 Tri: 0.011 Acc: 0.993 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:764:Epoch[49] Iter[100/786] Loss: 1.078 CE: 1.068 Tri: 0.009 Acc: 0.993 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:765:Epoch[49] Iter[150/786] Loss: 1.076 CE: 1.066 Tri: 0.010 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:766:Epoch[49] Iter[200/786] Loss: 1.073 CE: 1.065 Tri: 0.008 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:767:Epoch[49] Iter[250/786] Loss: 1.072 CE: 1.063 Tri: 0.009 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:768:Epoch[49] Iter[300/786] Loss: 1.071 CE: 1.062 Tri: 0.008 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:769:Epoch[49] Iter[350/786] Loss: 1.070 CE: 1.062 Tri: 0.008 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:770:Epoch[49] Iter[400/786] Loss: 1.068 CE: 1.061 Tri: 0.007 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:771:Epoch[49] Iter[450/786] Loss: 1.067 CE: 1.060 Tri: 0.007 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:772:Epoch[49] Iter[500/786] Loss: 1.066 CE: 1.059 Tri: 0.007 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:773:Epoch[49] Iter[550/786] Loss: 1.065 CE: 1.059 Tri: 0.007 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:774:Epoch[49] Iter[600/786] Loss: 1.064 CE: 1.058 Tri: 0.006 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:775:Epoch[49] Iter[650/786] Loss: 1.063 CE: 1.057 Tri: 0.006 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:776:Epoch[49] Iter[700/786] Loss: 1.062 CE: 1.056 Tri: 0.006 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline_4090.log:777:Epoch[49] done in 91.5s  Loss=1.060 Acc=0.996
logs/agreidv2_baseline_4090.log:778:Epoch[50] Iter[50/786] Loss: 1.064 CE: 1.062 Tri: 0.001 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:779:Epoch[50] Iter[100/786] Loss: 1.067 CE: 1.063 Tri: 0.004 Acc: 0.994 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:780:Epoch[50] Iter[150/786] Loss: 1.064 CE: 1.060 Tri: 0.004 Acc: 0.995 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:781:Epoch[50] Iter[200/786] Loss: 1.065 CE: 1.060 Tri: 0.005 Acc: 0.995 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:782:Epoch[50] Iter[250/786] Loss: 1.064 CE: 1.059 Tri: 0.005 Acc: 0.995 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:783:Epoch[50] Iter[300/786] Loss: 1.063 CE: 1.058 Tri: 0.005 Acc: 0.995 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:784:Epoch[50] Iter[350/786] Loss: 1.061 CE: 1.057 Tri: 0.004 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:785:Epoch[50] Iter[400/786] Loss: 1.062 CE: 1.056 Tri: 0.005 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:786:Epoch[50] Iter[450/786] Loss: 1.061 CE: 1.056 Tri: 0.005 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:787:Epoch[50] Iter[500/786] Loss: 1.060 CE: 1.055 Tri: 0.005 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:788:Epoch[50] Iter[550/786] Loss: 1.059 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:789:Epoch[50] Iter[600/786] Loss: 1.059 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:790:Epoch[50] Iter[650/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:791:Epoch[50] Iter[700/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_baseline_4090.log:792:Epoch[50] done in 91.5s  Loss=1.056 Acc=0.997
logs/agreidv2_baseline_4090.log:794:    [A->G] mAP=79.75  R1=86.59  R5=92.44  mINP=55.00
logs/agreidv2_baseline_4090.log:795:    [G->A] mAP=80.22  R1=88.51  R5=92.21  mINP=46.92
logs/agreidv2_baseline_4090.log:796:    [mean] mAP=79.98  R1=87.55
logs/agreidv2_baseline_4090.log:797:    * new best mean mAP=79.98 (epoch 50) saved
logs/agreidv2_baseline_4090.log:798:Epoch[51] Iter[50/786] Loss: 1.065 CE: 1.061 Tri: 0.004 Acc: 0.998 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:799:Epoch[51] Iter[100/786] Loss: 1.066 CE: 1.061 Tri: 0.005 Acc: 0.995 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:800:Epoch[51] Iter[150/786] Loss: 1.067 CE: 1.060 Tri: 0.007 Acc: 0.995 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:801:Epoch[51] Iter[200/786] Loss: 1.066 CE: 1.059 Tri: 0.007 Acc: 0.995 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:802:Epoch[51] Iter[250/786] Loss: 1.065 CE: 1.058 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:803:Epoch[51] Iter[300/786] Loss: 1.064 CE: 1.058 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:804:Epoch[51] Iter[350/786] Loss: 1.063 CE: 1.057 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:805:Epoch[51] Iter[400/786] Loss: 1.062 CE: 1.056 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:806:Epoch[51] Iter[450/786] Loss: 1.061 CE: 1.056 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:807:Epoch[51] Iter[500/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:808:Epoch[51] Iter[550/786] Loss: 1.060 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:809:Epoch[51] Iter[600/786] Loss: 1.059 CE: 1.054 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:810:Epoch[51] Iter[650/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.997 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:811:Epoch[51] Iter[700/786] Loss: 1.058 CE: 1.052 Tri: 0.005 Acc: 0.997 LR: 3.34e-06
logs/agreidv2_baseline_4090.log:812:Epoch[51] done in 91.5s  Loss=1.056 Acc=0.997
logs/agreidv2_baseline_4090.log:813:Epoch[52] Iter[50/786] Loss: 1.074 CE: 1.066 Tri: 0.008 Acc: 0.992 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:814:Epoch[52] Iter[100/786] Loss: 1.066 CE: 1.060 Tri: 0.006 Acc: 0.995 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:815:Epoch[52] Iter[150/786] Loss: 1.063 CE: 1.057 Tri: 0.005 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:816:Epoch[52] Iter[200/786] Loss: 1.066 CE: 1.059 Tri: 0.007 Acc: 0.995 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:817:Epoch[52] Iter[250/786] Loss: 1.063 CE: 1.057 Tri: 0.006 Acc: 0.995 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:818:Epoch[52] Iter[300/786] Loss: 1.062 CE: 1.056 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:819:Epoch[52] Iter[350/786] Loss: 1.062 CE: 1.056 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:820:Epoch[52] Iter[400/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:821:Epoch[52] Iter[450/786] Loss: 1.060 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:822:Epoch[52] Iter[500/786] Loss: 1.060 CE: 1.054 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:823:Epoch[52] Iter[550/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:824:Epoch[52] Iter[600/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:825:Epoch[52] Iter[650/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:826:Epoch[52] Iter[700/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline_4090.log:827:Epoch[52] done in 91.7s  Loss=1.055 Acc=0.996
logs/agreidv2_baseline_4090.log:828:Epoch[53] Iter[50/786] Loss: 1.072 CE: 1.062 Tri: 0.010 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:829:Epoch[53] Iter[100/786] Loss: 1.069 CE: 1.060 Tri: 0.009 Acc: 0.994 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:830:Epoch[53] Iter[150/786] Loss: 1.067 CE: 1.059 Tri: 0.008 Acc: 0.994 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:831:Epoch[53] Iter[200/786] Loss: 1.063 CE: 1.056 Tri: 0.007 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:832:Epoch[53] Iter[250/786] Loss: 1.062 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:833:Epoch[53] Iter[300/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:834:Epoch[53] Iter[350/786] Loss: 1.060 CE: 1.054 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:835:Epoch[53] Iter[400/786] Loss: 1.059 CE: 1.054 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:836:Epoch[53] Iter[450/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:837:Epoch[53] Iter[500/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.997 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:838:Epoch[53] Iter[550/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:839:Epoch[53] Iter[600/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.997 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:840:Epoch[53] Iter[650/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.997 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:841:Epoch[53] Iter[700/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 2.16e-06
logs/agreidv2_baseline_4090.log:842:Epoch[53] done in 91.8s  Loss=1.051 Acc=0.997
logs/agreidv2_baseline_4090.log:843:Epoch[54] Iter[50/786] Loss: 1.068 CE: 1.059 Tri: 0.008 Acc: 0.995 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:844:Epoch[54] Iter[100/786] Loss: 1.061 CE: 1.056 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:845:Epoch[54] Iter[150/786] Loss: 1.061 CE: 1.056 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:846:Epoch[54] Iter[200/786] Loss: 1.060 CE: 1.055 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:847:Epoch[54] Iter[250/786] Loss: 1.059 CE: 1.054 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:848:Epoch[54] Iter[300/786] Loss: 1.058 CE: 1.054 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:849:Epoch[54] Iter[350/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:850:Epoch[54] Iter[400/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:851:Epoch[54] Iter[450/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:852:Epoch[54] Iter[500/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:853:Epoch[54] Iter[550/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:854:Epoch[54] Iter[600/786] Loss: 1.054 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:855:Epoch[54] Iter[650/786] Loss: 1.054 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:856:Epoch[54] Iter[700/786] Loss: 1.053 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_baseline_4090.log:857:Epoch[54] done in 91.6s  Loss=1.051 Acc=0.997
logs/agreidv2_baseline_4090.log:858:Epoch[55] Iter[50/786] Loss: 1.056 CE: 1.053 Tri: 0.003 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:859:Epoch[55] Iter[100/786] Loss: 1.062 CE: 1.057 Tri: 0.005 Acc: 0.995 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:860:Epoch[55] Iter[150/786] Loss: 1.060 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:861:Epoch[55] Iter[200/786] Loss: 1.060 CE: 1.054 Tri: 0.006 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:862:Epoch[55] Iter[250/786] Loss: 1.059 CE: 1.054 Tri: 0.006 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:863:Epoch[55] Iter[300/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:864:Epoch[55] Iter[350/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:865:Epoch[55] Iter[400/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:866:Epoch[55] Iter[450/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:867:Epoch[55] Iter[500/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:868:Epoch[55] Iter[550/786] Loss: 1.053 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:869:Epoch[55] Iter[600/786] Loss: 1.053 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:870:Epoch[55] Iter[650/786] Loss: 1.052 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:871:Epoch[55] Iter[700/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline_4090.log:872:Epoch[55] done in 91.6s  Loss=1.049 Acc=0.997
logs/agreidv2_baseline_4090.log:873:Epoch[56] Iter[50/786] Loss: 1.064 CE: 1.057 Tri: 0.007 Acc: 0.993 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:874:Epoch[56] Iter[100/786] Loss: 1.060 CE: 1.055 Tri: 0.005 Acc: 0.995 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:875:Epoch[56] Iter[150/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:876:Epoch[56] Iter[200/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:877:Epoch[56] Iter[250/786] Loss: 1.056 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:878:Epoch[56] Iter[300/786] Loss: 1.054 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:879:Epoch[56] Iter[350/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:880:Epoch[56] Iter[400/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:881:Epoch[56] Iter[450/786] Loss: 1.052 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:882:Epoch[56] Iter[500/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:883:Epoch[56] Iter[550/786] Loss: 1.051 CE: 1.048 Tri: 0.003 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:884:Epoch[56] Iter[600/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:885:Epoch[56] Iter[650/786] Loss: 1.050 CE: 1.046 Tri: 0.003 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:886:Epoch[56] Iter[700/786] Loss: 1.049 CE: 1.045 Tri: 0.003 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline_4090.log:887:Epoch[56] done in 92.0s  Loss=1.047 Acc=0.997
logs/agreidv2_baseline_4090.log:888:Epoch[57] Iter[50/786] Loss: 1.060 CE: 1.058 Tri: 0.002 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:889:Epoch[57] Iter[100/786] Loss: 1.058 CE: 1.054 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:890:Epoch[57] Iter[150/786] Loss: 1.056 CE: 1.053 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:891:Epoch[57] Iter[200/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:892:Epoch[57] Iter[250/786] Loss: 1.059 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:893:Epoch[57] Iter[300/786] Loss: 1.059 CE: 1.053 Tri: 0.005 Acc: 0.995 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:894:Epoch[57] Iter[350/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.995 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:895:Epoch[57] Iter[400/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:896:Epoch[57] Iter[450/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:897:Epoch[57] Iter[500/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:898:Epoch[57] Iter[550/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:899:Epoch[57] Iter[600/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:900:Epoch[57] Iter[650/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:901:Epoch[57] Iter[700/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_baseline_4090.log:902:Epoch[57] done in 91.5s  Loss=1.049 Acc=0.997
logs/agreidv2_baseline_4090.log:903:Epoch[58] Iter[50/786] Loss: 1.060 CE: 1.056 Tri: 0.004 Acc: 0.993 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:904:Epoch[58] Iter[100/786] Loss: 1.058 CE: 1.054 Tri: 0.004 Acc: 0.994 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:905:Epoch[58] Iter[150/786] Loss: 1.058 CE: 1.054 Tri: 0.004 Acc: 0.994 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:906:Epoch[58] Iter[200/786] Loss: 1.058 CE: 1.053 Tri: 0.004 Acc: 0.995 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:907:Epoch[58] Iter[250/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.995 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:908:Epoch[58] Iter[300/786] Loss: 1.057 CE: 1.052 Tri: 0.004 Acc: 0.995 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:909:Epoch[58] Iter[350/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.995 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:910:Epoch[58] Iter[400/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:911:Epoch[58] Iter[450/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:912:Epoch[58] Iter[500/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:913:Epoch[58] Iter[550/786] Loss: 1.053 CE: 1.048 Tri: 0.004 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:914:Epoch[58] Iter[600/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:915:Epoch[58] Iter[650/786] Loss: 1.050 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:916:Epoch[58] Iter[700/786] Loss: 1.049 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
logs/agreidv2_baseline_4090.log:917:Epoch[58] done in 91.9s  Loss=1.048 Acc=0.997
logs/agreidv2_baseline_4090.log:918:Epoch[59] Iter[50/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.998 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:919:Epoch[59] Iter[100/786] Loss: 1.054 CE: 1.051 Tri: 0.003 Acc: 0.998 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:920:Epoch[59] Iter[150/786] Loss: 1.054 CE: 1.051 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:921:Epoch[59] Iter[200/786] Loss: 1.054 CE: 1.051 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:922:Epoch[59] Iter[250/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:923:Epoch[59] Iter[300/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.996 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:924:Epoch[59] Iter[350/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:925:Epoch[59] Iter[400/786] Loss: 1.054 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:926:Epoch[59] Iter[450/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:927:Epoch[59] Iter[500/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:928:Epoch[59] Iter[550/786] Loss: 1.052 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:929:Epoch[59] Iter[600/786] Loss: 1.051 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:930:Epoch[59] Iter[650/786] Loss: 1.050 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:931:Epoch[59] Iter[700/786] Loss: 1.049 CE: 1.045 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline_4090.log:932:Epoch[59] done in 91.5s  Loss=1.047 Acc=0.998
logs/agreidv2_baseline_4090.log:933:Epoch[60] Iter[50/786] Loss: 1.057 CE: 1.054 Tri: 0.003 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:934:Epoch[60] Iter[100/786] Loss: 1.056 CE: 1.053 Tri: 0.003 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:935:Epoch[60] Iter[150/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:936:Epoch[60] Iter[200/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:937:Epoch[60] Iter[250/786] Loss: 1.056 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:938:Epoch[60] Iter[300/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:939:Epoch[60] Iter[350/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:940:Epoch[60] Iter[400/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:941:Epoch[60] Iter[450/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:942:Epoch[60] Iter[500/786] Loss: 1.051 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:943:Epoch[60] Iter[550/786] Loss: 1.050 CE: 1.047 Tri: 0.003 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:944:Epoch[60] Iter[600/786] Loss: 1.049 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:945:Epoch[60] Iter[650/786] Loss: 1.049 CE: 1.045 Tri: 0.003 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:946:Epoch[60] Iter[700/786] Loss: 1.048 CE: 1.045 Tri: 0.003 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline_4090.log:947:Epoch[60] done in 91.9s  Loss=1.047 Acc=0.997
logs/agreidv2_baseline_4090.log:949:    [A->G] mAP=80.73  R1=87.35  R5=93.00  mINP=56.91
logs/agreidv2_baseline_4090.log:950:    [G->A] mAP=81.42  R1=89.07  R5=93.21  mINP=48.32
logs/agreidv2_baseline_4090.log:951:    [mean] mAP=81.08  R1=88.21
logs/agreidv2_baseline_4090.log:952:    * new best mean mAP=81.08 (epoch 60) saved
logs/agreidv2_baseline_4090.log:954:Training finished. Best mean A<->G mAP=81.08 @ epoch 60
logs/agreidv2_baseline_4090.log:955:Checkpoints in /home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090
logs/agreidv2_baseline.log:2:CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
logs/agreidv2_baseline.log:7:  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
logs/agreidv2_baseline.log:8:  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
logs/agreidv2_baseline.log:9:  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_baseline.log:11:  out_dir=/root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_baseline
logs/agreidv2_baseline.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_baseline.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_baseline.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_baseline.log:25:Epoch[1] Iter[50/786] Loss: 38.416 CE: 6.693 Tri: 31.723 Acc: 0.001 LR: 3.50e-07
logs/agreidv2_baseline.log:26:Epoch[1] Iter[100/786] Loss: 32.119 CE: 6.692 Tri: 25.427 Acc: 0.001 LR: 3.50e-07
logs/agreidv2_baseline.log:27:Epoch[1] Iter[150/786] Loss: 28.813 CE: 6.690 Tri: 22.123 Acc: 0.001 LR: 3.50e-07
logs/agreidv2_baseline.log:28:Epoch[1] Iter[200/786] Loss: 26.827 CE: 6.688 Tri: 20.138 Acc: 0.002 LR: 3.50e-07
logs/agreidv2_baseline.log:29:Epoch[1] Iter[250/786] Loss: 25.425 CE: 6.686 Tri: 18.739 Acc: 0.003 LR: 3.50e-07
logs/agreidv2_baseline.log:30:Epoch[1] Iter[300/786] Loss: 24.211 CE: 6.684 Tri: 17.527 Acc: 0.006 LR: 3.50e-07
logs/agreidv2_baseline.log:31:Epoch[1] Iter[350/786] Loss: 23.244 CE: 6.682 Tri: 16.562 Acc: 0.010 LR: 3.50e-07
logs/agreidv2_baseline.log:32:Epoch[1] Iter[400/786] Loss: 22.418 CE: 6.679 Tri: 15.739 Acc: 0.015 LR: 3.50e-07
logs/agreidv2_baseline.log:33:Epoch[1] Iter[450/786] Loss: 21.663 CE: 6.677 Tri: 14.986 Acc: 0.020 LR: 3.50e-07
logs/agreidv2_baseline.log:34:Epoch[1] Iter[500/786] Loss: 20.981 CE: 6.675 Tri: 14.306 Acc: 0.026 LR: 3.50e-07
logs/agreidv2_baseline.log:35:Epoch[1] Iter[550/786] Loss: 20.362 CE: 6.672 Tri: 13.690 Acc: 0.037 LR: 3.50e-07
logs/agreidv2_baseline.log:36:Epoch[1] Iter[600/786] Loss: 19.852 CE: 6.669 Tri: 13.183 Acc: 0.050 LR: 3.50e-07
logs/agreidv2_baseline.log:37:Epoch[1] Iter[650/786] Loss: 19.402 CE: 6.666 Tri: 12.736 Acc: 0.064 LR: 3.50e-07
logs/agreidv2_baseline.log:38:Epoch[1] Iter[700/786] Loss: 18.988 CE: 6.662 Tri: 12.326 Acc: 0.087 LR: 3.50e-07
logs/agreidv2_baseline.log:39:Epoch[1] done in 188.7s  Loss=18.648 Acc=0.115
logs/agreidv2_baseline.log:40:Epoch[2] Iter[50/786] Loss: 11.796 CE: 6.602 Tri: 5.195 Acc: 0.147 LR: 3.82e-06
logs/agreidv2_baseline.log:41:Epoch[2] Iter[100/786] Loss: 11.154 CE: 6.570 Tri: 4.584 Acc: 0.179 LR: 3.82e-06
logs/agreidv2_baseline.log:42:Epoch[2] Iter[150/786] Loss: 10.801 CE: 6.539 Tri: 4.262 Acc: 0.209 LR: 3.82e-06
logs/agreidv2_baseline.log:43:Epoch[2] Iter[200/786] Loss: 10.474 CE: 6.505 Tri: 3.969 Acc: 0.236 LR: 3.82e-06
logs/agreidv2_baseline.log:44:Epoch[2] Iter[250/786] Loss: 10.190 CE: 6.472 Tri: 3.717 Acc: 0.261 LR: 3.82e-06
logs/agreidv2_baseline.log:45:Epoch[2] Iter[300/786] Loss: 9.963 CE: 6.439 Tri: 3.524 Acc: 0.285 LR: 3.82e-06
logs/agreidv2_baseline.log:46:Epoch[2] Iter[350/786] Loss: 9.770 CE: 6.406 Tri: 3.364 Acc: 0.307 LR: 3.82e-06
logs/agreidv2_baseline.log:47:Epoch[2] Iter[400/786] Loss: 9.618 CE: 6.371 Tri: 3.247 Acc: 0.328 LR: 3.82e-06
logs/agreidv2_baseline.log:48:Epoch[2] Iter[450/786] Loss: 9.454 CE: 6.338 Tri: 3.116 Acc: 0.348 LR: 3.82e-06
logs/agreidv2_baseline.log:49:Epoch[2] Iter[500/786] Loss: 9.300 CE: 6.303 Tri: 2.997 Acc: 0.367 LR: 3.82e-06
logs/agreidv2_baseline.log:50:Epoch[2] Iter[550/786] Loss: 9.147 CE: 6.265 Tri: 2.882 Acc: 0.389 LR: 3.82e-06
logs/agreidv2_baseline.log:51:Epoch[2] Iter[600/786] Loss: 9.012 CE: 6.225 Tri: 2.787 Acc: 0.410 LR: 3.82e-06
logs/agreidv2_baseline.log:52:Epoch[2] Iter[650/786] Loss: 8.881 CE: 6.179 Tri: 2.701 Acc: 0.432 LR: 3.82e-06
logs/agreidv2_baseline.log:53:Epoch[2] Iter[700/786] Loss: 8.757 CE: 6.129 Tri: 2.629 Acc: 0.454 LR: 3.82e-06
logs/agreidv2_baseline.log:54:Epoch[2] done in 188.5s  Loss=8.628 Acc=0.478
logs/agreidv2_baseline.log:55:Epoch[3] Iter[50/786] Loss: 7.029 CE: 5.593 Tri: 1.436 Acc: 0.316 LR: 7.28e-06
logs/agreidv2_baseline.log:56:Epoch[3] Iter[100/786] Loss: 6.936 CE: 5.523 Tri: 1.412 Acc: 0.344 LR: 7.28e-06
logs/agreidv2_baseline.log:57:Epoch[3] Iter[150/786] Loss: 6.745 CE: 5.432 Tri: 1.313 Acc: 0.389 LR: 7.28e-06
logs/agreidv2_baseline.log:58:Epoch[3] Iter[200/786] Loss: 6.620 CE: 5.335 Tri: 1.284 Acc: 0.429 LR: 7.28e-06
logs/agreidv2_baseline.log:59:Epoch[3] Iter[250/786] Loss: 6.483 CE: 5.249 Tri: 1.234 Acc: 0.459 LR: 7.28e-06
logs/agreidv2_baseline.log:60:Epoch[3] Iter[300/786] Loss: 6.370 CE: 5.167 Tri: 1.204 Acc: 0.482 LR: 7.28e-06
logs/agreidv2_baseline.log:61:Epoch[3] Iter[350/786] Loss: 6.265 CE: 5.089 Tri: 1.176 Acc: 0.503 LR: 7.28e-06
logs/agreidv2_baseline.log:62:Epoch[3] Iter[400/786] Loss: 6.161 CE: 5.009 Tri: 1.151 Acc: 0.524 LR: 7.28e-06
logs/agreidv2_baseline.log:63:Epoch[3] Iter[450/786] Loss: 6.053 CE: 4.923 Tri: 1.129 Acc: 0.543 LR: 7.28e-06
logs/agreidv2_baseline.log:64:Epoch[3] Iter[500/786] Loss: 5.947 CE: 4.836 Tri: 1.112 Acc: 0.561 LR: 7.28e-06
logs/agreidv2_baseline.log:65:Epoch[3] Iter[550/786] Loss: 5.824 CE: 4.742 Tri: 1.082 Acc: 0.581 LR: 7.28e-06
logs/agreidv2_baseline.log:66:Epoch[3] Iter[600/786] Loss: 5.702 CE: 4.644 Tri: 1.057 Acc: 0.598 LR: 7.28e-06
logs/agreidv2_baseline.log:67:Epoch[3] Iter[650/786] Loss: 5.573 CE: 4.538 Tri: 1.035 Acc: 0.615 LR: 7.28e-06
logs/agreidv2_baseline.log:68:Epoch[3] Iter[700/786] Loss: 5.439 CE: 4.423 Tri: 1.016 Acc: 0.632 LR: 7.28e-06
logs/agreidv2_baseline.log:69:Epoch[3] done in 189.2s  Loss=5.308 Acc=0.648
logs/agreidv2_baseline.log:70:Epoch[4] Iter[50/786] Loss: 4.444 CE: 3.759 Tri: 0.684 Acc: 0.477 LR: 1.07e-05
logs/agreidv2_baseline.log:71:Epoch[4] Iter[100/786] Loss: 4.296 CE: 3.632 Tri: 0.663 Acc: 0.516 LR: 1.07e-05
logs/agreidv2_baseline.log:72:Epoch[4] Iter[150/786] Loss: 4.223 CE: 3.546 Tri: 0.676 Acc: 0.552 LR: 1.07e-05
logs/agreidv2_baseline.log:73:Epoch[4] Iter[200/786] Loss: 4.102 CE: 3.441 Tri: 0.661 Acc: 0.589 LR: 1.07e-05
logs/agreidv2_baseline.log:74:Epoch[4] Iter[250/786] Loss: 3.980 CE: 3.337 Tri: 0.643 Acc: 0.625 LR: 1.07e-05
logs/agreidv2_baseline.log:75:Epoch[4] Iter[300/786] Loss: 3.883 CE: 3.252 Tri: 0.631 Acc: 0.649 LR: 1.07e-05
logs/agreidv2_baseline.log:76:Epoch[4] Iter[350/786] Loss: 3.797 CE: 3.166 Tri: 0.630 Acc: 0.672 LR: 1.07e-05
logs/agreidv2_baseline.log:77:Epoch[4] Iter[400/786] Loss: 3.711 CE: 3.083 Tri: 0.628 Acc: 0.693 LR: 1.07e-05
logs/agreidv2_baseline.log:78:Epoch[4] Iter[450/786] Loss: 3.624 CE: 3.002 Tri: 0.622 Acc: 0.712 LR: 1.07e-05
logs/agreidv2_baseline.log:79:Epoch[4] Iter[500/786] Loss: 3.543 CE: 2.925 Tri: 0.618 Acc: 0.728 LR: 1.07e-05
logs/agreidv2_baseline.log:80:Epoch[4] Iter[550/786] Loss: 3.460 CE: 2.851 Tri: 0.610 Acc: 0.742 LR: 1.07e-05
logs/agreidv2_baseline.log:81:Epoch[4] Iter[600/786] Loss: 3.384 CE: 2.780 Tri: 0.604 Acc: 0.756 LR: 1.07e-05
logs/agreidv2_baseline.log:82:Epoch[4] Iter[650/786] Loss: 3.301 CE: 2.708 Tri: 0.593 Acc: 0.768 LR: 1.07e-05
logs/agreidv2_baseline.log:83:Epoch[4] Iter[700/786] Loss: 3.220 CE: 2.636 Tri: 0.583 Acc: 0.779 LR: 1.07e-05
logs/agreidv2_baseline.log:84:Epoch[4] done in 191.5s  Loss=3.139 Acc=0.791
logs/agreidv2_baseline.log:85:Epoch[5] Iter[50/786] Loss: 2.979 CE: 2.444 Tri: 0.535 Acc: 0.740 LR: 1.42e-05
logs/agreidv2_baseline.log:86:Epoch[5] Iter[100/786] Loss: 2.921 CE: 2.380 Tri: 0.542 Acc: 0.759 LR: 1.42e-05
logs/agreidv2_baseline.log:87:Epoch[5] Iter[150/786] Loss: 2.823 CE: 2.304 Tri: 0.519 Acc: 0.782 LR: 1.42e-05
logs/agreidv2_baseline.log:88:Epoch[5] Iter[200/786] Loss: 2.738 CE: 2.238 Tri: 0.500 Acc: 0.801 LR: 1.42e-05
logs/agreidv2_baseline.log:89:Epoch[5] Iter[250/786] Loss: 2.660 CE: 2.174 Tri: 0.486 Acc: 0.818 LR: 1.42e-05
logs/agreidv2_baseline.log:90:Epoch[5] Iter[300/786] Loss: 2.580 CE: 2.112 Tri: 0.468 Acc: 0.834 LR: 1.42e-05
logs/agreidv2_baseline.log:91:Epoch[5] Iter[350/786] Loss: 2.513 CE: 2.062 Tri: 0.451 Acc: 0.846 LR: 1.42e-05
logs/agreidv2_baseline.log:92:Epoch[5] Iter[400/786] Loss: 2.458 CE: 2.019 Tri: 0.439 Acc: 0.856 LR: 1.42e-05
logs/agreidv2_baseline.log:93:Epoch[5] Iter[450/786] Loss: 2.418 CE: 1.982 Tri: 0.436 Acc: 0.863 LR: 1.42e-05
logs/agreidv2_baseline.log:94:Epoch[5] Iter[500/786] Loss: 2.378 CE: 1.949 Tri: 0.429 Acc: 0.870 LR: 1.42e-05
logs/agreidv2_baseline.log:95:Epoch[5] Iter[550/786] Loss: 2.344 CE: 1.919 Tri: 0.425 Acc: 0.876 LR: 1.42e-05
logs/agreidv2_baseline.log:96:Epoch[5] Iter[600/786] Loss: 2.311 CE: 1.890 Tri: 0.421 Acc: 0.881 LR: 1.42e-05
logs/agreidv2_baseline.log:97:Epoch[5] Iter[650/786] Loss: 2.275 CE: 1.862 Tri: 0.413 Acc: 0.886 LR: 1.42e-05
logs/agreidv2_baseline.log:98:Epoch[5] Iter[700/786] Loss: 2.246 CE: 1.837 Tri: 0.409 Acc: 0.890 LR: 1.42e-05
logs/agreidv2_baseline.log:99:Epoch[5] done in 191.6s  Loss=2.211 Acc=0.895
logs/agreidv2_baseline.log:100:Epoch[6] Iter[50/786] Loss: 2.163 CE: 1.840 Tri: 0.323 Acc: 0.863 LR: 1.77e-05
logs/agreidv2_baseline.log:101:Epoch[6] Iter[100/786] Loss: 2.164 CE: 1.809 Tri: 0.355 Acc: 0.873 LR: 1.77e-05
logs/agreidv2_baseline.log:102:Epoch[6] Iter[150/786] Loss: 2.118 CE: 1.770 Tri: 0.348 Acc: 0.884 LR: 1.77e-05
logs/agreidv2_baseline.log:103:Epoch[6] Iter[200/786] Loss: 2.084 CE: 1.740 Tri: 0.344 Acc: 0.894 LR: 1.77e-05
logs/agreidv2_baseline.log:104:Epoch[6] Iter[250/786] Loss: 2.061 CE: 1.716 Tri: 0.345 Acc: 0.900 LR: 1.77e-05
logs/agreidv2_baseline.log:105:Epoch[6] Iter[300/786] Loss: 2.028 CE: 1.692 Tri: 0.335 Acc: 0.905 LR: 1.77e-05
logs/agreidv2_baseline.log:106:Epoch[6] Iter[350/786] Loss: 2.010 CE: 1.676 Tri: 0.334 Acc: 0.909 LR: 1.77e-05
logs/agreidv2_baseline.log:107:Epoch[6] Iter[400/786] Loss: 1.984 CE: 1.658 Tri: 0.326 Acc: 0.914 LR: 1.77e-05
logs/agreidv2_baseline.log:108:Epoch[6] Iter[450/786] Loss: 1.961 CE: 1.641 Tri: 0.321 Acc: 0.917 LR: 1.77e-05
logs/agreidv2_baseline.log:109:Epoch[6] Iter[500/786] Loss: 1.945 CE: 1.628 Tri: 0.318 Acc: 0.919 LR: 1.77e-05
logs/agreidv2_baseline.log:110:Epoch[6] Iter[550/786] Loss: 1.933 CE: 1.617 Tri: 0.315 Acc: 0.921 LR: 1.77e-05
logs/agreidv2_baseline.log:111:Epoch[6] Iter[600/786] Loss: 1.917 CE: 1.604 Tri: 0.313 Acc: 0.924 LR: 1.77e-05
logs/agreidv2_baseline.log:112:Epoch[6] Iter[650/786] Loss: 1.896 CE: 1.590 Tri: 0.306 Acc: 0.927 LR: 1.77e-05
logs/agreidv2_baseline.log:113:Epoch[6] Iter[700/786] Loss: 1.879 CE: 1.576 Tri: 0.303 Acc: 0.930 LR: 1.77e-05
logs/agreidv2_baseline.log:114:Epoch[6] Iter[750/786] Loss: 1.858 CE: 1.560 Tri: 0.298 Acc: 0.933 LR: 1.77e-05
logs/agreidv2_baseline.log:115:Epoch[6] done in 192.5s  Loss=1.858 Acc=0.933
logs/agreidv2_baseline.log:116:Epoch[7] Iter[50/786] Loss: 1.890 CE: 1.629 Tri: 0.262 Acc: 0.910 LR: 2.11e-05
logs/agreidv2_baseline.log:117:Epoch[7] Iter[100/786] Loss: 1.841 CE: 1.588 Tri: 0.253 Acc: 0.921 LR: 2.11e-05
logs/agreidv2_baseline.log:118:Epoch[7] Iter[150/786] Loss: 1.852 CE: 1.578 Tri: 0.274 Acc: 0.923 LR: 2.11e-05
logs/agreidv2_baseline.log:119:Epoch[7] Iter[200/786] Loss: 1.845 CE: 1.564 Tri: 0.281 Acc: 0.927 LR: 2.11e-05
logs/agreidv2_baseline.log:120:Epoch[7] Iter[250/786] Loss: 1.850 CE: 1.558 Tri: 0.292 Acc: 0.929 LR: 2.11e-05
logs/agreidv2_baseline.log:121:Epoch[7] Iter[300/786] Loss: 1.850 CE: 1.552 Tri: 0.298 Acc: 0.930 LR: 2.11e-05
logs/agreidv2_baseline.log:122:Epoch[7] Iter[350/786] Loss: 1.831 CE: 1.541 Tri: 0.291 Acc: 0.932 LR: 2.11e-05
logs/agreidv2_baseline.log:123:Epoch[7] Iter[400/786] Loss: 1.817 CE: 1.532 Tri: 0.285 Acc: 0.935 LR: 2.11e-05
logs/agreidv2_baseline.log:124:Epoch[7] Iter[450/786] Loss: 1.805 CE: 1.525 Tri: 0.281 Acc: 0.936 LR: 2.11e-05
logs/agreidv2_baseline.log:125:Epoch[7] Iter[500/786] Loss: 1.789 CE: 1.515 Tri: 0.274 Acc: 0.939 LR: 2.11e-05
logs/agreidv2_baseline.log:126:Epoch[7] Iter[550/786] Loss: 1.779 CE: 1.508 Tri: 0.271 Acc: 0.940 LR: 2.11e-05
logs/agreidv2_baseline.log:127:Epoch[7] Iter[600/786] Loss: 1.770 CE: 1.502 Tri: 0.268 Acc: 0.941 LR: 2.11e-05
logs/agreidv2_baseline.log:128:Epoch[7] Iter[650/786] Loss: 1.758 CE: 1.493 Tri: 0.265 Acc: 0.943 LR: 2.11e-05
logs/agreidv2_baseline.log:129:Epoch[7] Iter[700/786] Loss: 1.748 CE: 1.485 Tri: 0.263 Acc: 0.944 LR: 2.11e-05
logs/agreidv2_baseline.log:130:Epoch[7] done in 191.4s  Loss=1.732 Acc=0.947
logs/agreidv2_baseline.log:131:Epoch[8] Iter[50/786] Loss: 1.836 CE: 1.559 Tri: 0.277 Acc: 0.917 LR: 2.46e-05
logs/agreidv2_baseline.log:132:Epoch[8] Iter[100/786] Loss: 1.778 CE: 1.519 Tri: 0.259 Acc: 0.932 LR: 2.46e-05
logs/agreidv2_baseline.log:133:Epoch[8] Iter[150/786] Loss: 1.738 CE: 1.497 Tri: 0.241 Acc: 0.937 LR: 2.46e-05
logs/agreidv2_baseline.log:134:Epoch[8] Iter[200/786] Loss: 1.727 CE: 1.491 Tri: 0.236 Acc: 0.938 LR: 2.46e-05
logs/agreidv2_baseline.log:135:Epoch[8] Iter[250/786] Loss: 1.710 CE: 1.479 Tri: 0.231 Acc: 0.942 LR: 2.46e-05
logs/agreidv2_baseline.log:136:Epoch[8] Iter[300/786] Loss: 1.701 CE: 1.472 Tri: 0.229 Acc: 0.943 LR: 2.46e-05
logs/agreidv2_baseline.log:137:Epoch[8] Iter[350/786] Loss: 1.700 CE: 1.470 Tri: 0.231 Acc: 0.945 LR: 2.46e-05
logs/agreidv2_baseline.log:138:Epoch[8] Iter[400/786] Loss: 1.689 CE: 1.462 Tri: 0.227 Acc: 0.946 LR: 2.46e-05
logs/agreidv2_baseline.log:139:Epoch[8] Iter[450/786] Loss: 1.683 CE: 1.457 Tri: 0.226 Acc: 0.947 LR: 2.46e-05
logs/agreidv2_baseline.log:140:Epoch[8] Iter[500/786] Loss: 1.675 CE: 1.452 Tri: 0.224 Acc: 0.948 LR: 2.46e-05
logs/agreidv2_baseline.log:141:Epoch[8] Iter[550/786] Loss: 1.670 CE: 1.447 Tri: 0.222 Acc: 0.949 LR: 2.46e-05
logs/agreidv2_baseline.log:142:Epoch[8] Iter[600/786] Loss: 1.660 CE: 1.440 Tri: 0.220 Acc: 0.951 LR: 2.46e-05
logs/agreidv2_baseline.log:143:Epoch[8] Iter[650/786] Loss: 1.651 CE: 1.433 Tri: 0.218 Acc: 0.952 LR: 2.46e-05
logs/agreidv2_baseline.log:144:Epoch[8] Iter[700/786] Loss: 1.643 CE: 1.427 Tri: 0.217 Acc: 0.953 LR: 2.46e-05
logs/agreidv2_baseline.log:145:Epoch[8] done in 190.6s  Loss=1.631 Acc=0.955
logs/agreidv2_baseline.log:146:Epoch[9] Iter[50/786] Loss: 1.766 CE: 1.525 Tri: 0.241 Acc: 0.931 LR: 2.81e-05
logs/agreidv2_baseline.log:147:Epoch[9] Iter[100/786] Loss: 1.710 CE: 1.492 Tri: 0.218 Acc: 0.939 LR: 2.81e-05
logs/agreidv2_baseline.log:148:Epoch[9] Iter[150/786] Loss: 1.699 CE: 1.477 Tri: 0.223 Acc: 0.943 LR: 2.81e-05
logs/agreidv2_baseline.log:149:Epoch[9] Iter[200/786] Loss: 1.683 CE: 1.464 Tri: 0.218 Acc: 0.945 LR: 2.81e-05
logs/agreidv2_baseline.log:150:Epoch[9] Iter[250/786] Loss: 1.677 CE: 1.455 Tri: 0.222 Acc: 0.948 LR: 2.81e-05
logs/agreidv2_baseline.log:151:Epoch[9] Iter[300/786] Loss: 1.670 CE: 1.450 Tri: 0.221 Acc: 0.948 LR: 2.81e-05
logs/agreidv2_baseline.log:152:Epoch[9] Iter[350/786] Loss: 1.658 CE: 1.441 Tri: 0.217 Acc: 0.950 LR: 2.81e-05
logs/agreidv2_baseline.log:153:Epoch[9] Iter[400/786] Loss: 1.645 CE: 1.434 Tri: 0.210 Acc: 0.951 LR: 2.81e-05
logs/agreidv2_baseline.log:154:Epoch[9] Iter[450/786] Loss: 1.637 CE: 1.428 Tri: 0.209 Acc: 0.952 LR: 2.81e-05
logs/agreidv2_baseline.log:155:Epoch[9] Iter[500/786] Loss: 1.627 CE: 1.423 Tri: 0.205 Acc: 0.953 LR: 2.81e-05
logs/agreidv2_baseline.log:156:Epoch[9] Iter[550/786] Loss: 1.624 CE: 1.420 Tri: 0.204 Acc: 0.954 LR: 2.81e-05
logs/agreidv2_baseline.log:157:Epoch[9] Iter[600/786] Loss: 1.620 CE: 1.415 Tri: 0.205 Acc: 0.954 LR: 2.81e-05
logs/agreidv2_baseline.log:158:Epoch[9] Iter[650/786] Loss: 1.614 CE: 1.410 Tri: 0.204 Acc: 0.955 LR: 2.81e-05
logs/agreidv2_baseline.log:159:Epoch[9] Iter[700/786] Loss: 1.603 CE: 1.401 Tri: 0.202 Acc: 0.957 LR: 2.81e-05
logs/agreidv2_baseline.log:160:Epoch[9] done in 189.5s  Loss=1.593 Acc=0.959
logs/agreidv2_baseline.log:161:Epoch[10] Iter[50/786] Loss: 1.759 CE: 1.504 Tri: 0.255 Acc: 0.932 LR: 3.15e-05
logs/agreidv2_baseline.log:162:Epoch[10] Iter[100/786] Loss: 1.692 CE: 1.467 Tri: 0.225 Acc: 0.944 LR: 3.15e-05
logs/agreidv2_baseline.log:163:Epoch[10] Iter[150/786] Loss: 1.658 CE: 1.449 Tri: 0.209 Acc: 0.947 LR: 3.15e-05
logs/agreidv2_baseline.log:164:Epoch[10] Iter[200/786] Loss: 1.634 CE: 1.436 Tri: 0.198 Acc: 0.951 LR: 3.15e-05
logs/agreidv2_baseline.log:165:Epoch[10] Iter[250/786] Loss: 1.619 CE: 1.428 Tri: 0.191 Acc: 0.953 LR: 3.15e-05
logs/agreidv2_baseline.log:166:Epoch[10] Iter[300/786] Loss: 1.606 CE: 1.420 Tri: 0.186 Acc: 0.954 LR: 3.15e-05
logs/agreidv2_baseline.log:167:Epoch[10] Iter[350/786] Loss: 1.603 CE: 1.418 Tri: 0.185 Acc: 0.954 LR: 3.15e-05
logs/agreidv2_baseline.log:168:Epoch[10] Iter[400/786] Loss: 1.595 CE: 1.412 Tri: 0.182 Acc: 0.955 LR: 3.15e-05
logs/agreidv2_baseline.log:169:Epoch[10] Iter[450/786] Loss: 1.594 CE: 1.409 Tri: 0.185 Acc: 0.955 LR: 3.15e-05
logs/agreidv2_baseline.log:170:Epoch[10] Iter[500/786] Loss: 1.589 CE: 1.405 Tri: 0.184 Acc: 0.956 LR: 3.15e-05
logs/agreidv2_baseline.log:171:Epoch[10] Iter[550/786] Loss: 1.587 CE: 1.402 Tri: 0.185 Acc: 0.956 LR: 3.15e-05
logs/agreidv2_baseline.log:172:Epoch[10] Iter[600/786] Loss: 1.582 CE: 1.398 Tri: 0.184 Acc: 0.957 LR: 3.15e-05
logs/agreidv2_baseline.log:173:Epoch[10] Iter[650/786] Loss: 1.576 CE: 1.393 Tri: 0.183 Acc: 0.957 LR: 3.15e-05
logs/agreidv2_baseline.log:174:Epoch[10] Iter[700/786] Loss: 1.565 CE: 1.385 Tri: 0.180 Acc: 0.959 LR: 3.15e-05
logs/agreidv2_baseline.log:175:Epoch[10] done in 190.8s  Loss=1.553 Acc=0.960
logs/agreidv2_baseline.log:177:    [A->G] mAP=73.39  R1=82.60  R5=89.47  mINP=45.05
logs/agreidv2_baseline.log:178:    [G->A] mAP=73.97  R1=82.99  R5=89.51  mINP=35.92
logs/agreidv2_baseline.log:179:    [mean] mAP=73.68  R1=82.80
logs/agreidv2_baseline.log:180:    * new best mean mAP=73.68 (epoch 10) saved
logs/agreidv2_baseline.log:181:Epoch[11] Iter[50/786] Loss: 1.726 CE: 1.516 Tri: 0.210 Acc: 0.926 LR: 3.50e-05
logs/agreidv2_baseline.log:182:Epoch[11] Iter[100/786] Loss: 1.649 CE: 1.463 Tri: 0.186 Acc: 0.941 LR: 3.50e-05
logs/agreidv2_baseline.log:183:Epoch[11] Iter[150/786] Loss: 1.630 CE: 1.448 Tri: 0.182 Acc: 0.945 LR: 3.50e-05
logs/agreidv2_baseline.log:184:Epoch[11] Iter[200/786] Loss: 1.626 CE: 1.442 Tri: 0.184 Acc: 0.946 LR: 3.50e-05
logs/agreidv2_baseline.log:185:Epoch[11] Iter[250/786] Loss: 1.604 CE: 1.430 Tri: 0.174 Acc: 0.948 LR: 3.50e-05
logs/agreidv2_baseline.log:186:Epoch[11] Iter[300/786] Loss: 1.605 CE: 1.426 Tri: 0.179 Acc: 0.949 LR: 3.50e-05
logs/agreidv2_baseline.log:187:Epoch[11] Iter[350/786] Loss: 1.605 CE: 1.424 Tri: 0.181 Acc: 0.949 LR: 3.50e-05
logs/agreidv2_baseline.log:188:Epoch[11] Iter[400/786] Loss: 1.599 CE: 1.419 Tri: 0.180 Acc: 0.951 LR: 3.50e-05
logs/agreidv2_baseline.log:189:Epoch[11] Iter[450/786] Loss: 1.590 CE: 1.413 Tri: 0.177 Acc: 0.952 LR: 3.50e-05
logs/agreidv2_baseline.log:190:Epoch[11] Iter[500/786] Loss: 1.586 CE: 1.409 Tri: 0.176 Acc: 0.953 LR: 3.50e-05
logs/agreidv2_baseline.log:191:Epoch[11] Iter[550/786] Loss: 1.576 CE: 1.403 Tri: 0.173 Acc: 0.954 LR: 3.50e-05
logs/agreidv2_baseline.log:192:Epoch[11] Iter[600/786] Loss: 1.575 CE: 1.399 Tri: 0.176 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_baseline.log:193:Epoch[11] Iter[650/786] Loss: 1.568 CE: 1.393 Tri: 0.175 Acc: 0.956 LR: 3.50e-05
logs/agreidv2_baseline.log:194:Epoch[11] Iter[700/786] Loss: 1.558 CE: 1.385 Tri: 0.173 Acc: 0.957 LR: 3.50e-05
logs/agreidv2_baseline.log:195:Epoch[11] done in 188.0s  Loss=1.546 Acc=0.959
logs/agreidv2_baseline.log:196:Epoch[12] Iter[50/786] Loss: 1.628 CE: 1.482 Tri: 0.146 Acc: 0.932 LR: 3.50e-05
logs/agreidv2_baseline.log:197:Epoch[12] Iter[100/786] Loss: 1.629 CE: 1.465 Tri: 0.164 Acc: 0.936 LR: 3.50e-05
logs/agreidv2_baseline.log:198:Epoch[12] Iter[150/786] Loss: 1.597 CE: 1.439 Tri: 0.157 Acc: 0.943 LR: 3.50e-05
logs/agreidv2_baseline.log:199:Epoch[12] Iter[200/786] Loss: 1.582 CE: 1.423 Tri: 0.159 Acc: 0.947 LR: 3.50e-05
logs/agreidv2_baseline.log:200:Epoch[12] Iter[250/786] Loss: 1.571 CE: 1.415 Tri: 0.157 Acc: 0.949 LR: 3.50e-05
logs/agreidv2_baseline.log:201:Epoch[12] Iter[300/786] Loss: 1.568 CE: 1.407 Tri: 0.161 Acc: 0.952 LR: 3.50e-05
logs/agreidv2_baseline.log:202:Epoch[12] Iter[350/786] Loss: 1.560 CE: 1.400 Tri: 0.160 Acc: 0.953 LR: 3.50e-05
logs/agreidv2_baseline.log:203:Epoch[12] Iter[400/786] Loss: 1.558 CE: 1.396 Tri: 0.163 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_baseline.log:204:Epoch[12] Iter[450/786] Loss: 1.552 CE: 1.391 Tri: 0.161 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_baseline.log:205:Epoch[12] Iter[500/786] Loss: 1.545 CE: 1.385 Tri: 0.159 Acc: 0.957 LR: 3.50e-05
logs/agreidv2_baseline.log:206:Epoch[12] Iter[550/786] Loss: 1.540 CE: 1.380 Tri: 0.159 Acc: 0.958 LR: 3.50e-05
logs/agreidv2_baseline.log:207:Epoch[12] Iter[600/786] Loss: 1.531 CE: 1.374 Tri: 0.157 Acc: 0.959 LR: 3.50e-05
logs/agreidv2_baseline.log:208:Epoch[12] Iter[650/786] Loss: 1.525 CE: 1.368 Tri: 0.157 Acc: 0.960 LR: 3.50e-05
logs/agreidv2_baseline.log:209:Epoch[12] Iter[700/786] Loss: 1.519 CE: 1.362 Tri: 0.157 Acc: 0.961 LR: 3.50e-05
logs/agreidv2_baseline.log:210:Epoch[12] done in 187.8s  Loss=1.507 Acc=0.963
logs/agreidv2_baseline.log:211:Epoch[13] Iter[50/786] Loss: 1.634 CE: 1.461 Tri: 0.173 Acc: 0.938 LR: 3.49e-05
logs/agreidv2_baseline.log:212:Epoch[13] Iter[100/786] Loss: 1.569 CE: 1.419 Tri: 0.150 Acc: 0.949 LR: 3.49e-05
logs/agreidv2_baseline.log:213:Epoch[13] Iter[150/786] Loss: 1.549 CE: 1.399 Tri: 0.150 Acc: 0.953 LR: 3.49e-05
logs/agreidv2_baseline.log:214:Epoch[13] Iter[200/786] Loss: 1.540 CE: 1.394 Tri: 0.146 Acc: 0.954 LR: 3.49e-05
logs/agreidv2_baseline.log:215:Epoch[13] Iter[250/786] Loss: 1.534 CE: 1.389 Tri: 0.145 Acc: 0.954 LR: 3.49e-05
logs/agreidv2_baseline.log:216:Epoch[13] Iter[300/786] Loss: 1.524 CE: 1.383 Tri: 0.141 Acc: 0.955 LR: 3.49e-05
logs/agreidv2_baseline.log:217:Epoch[13] Iter[350/786] Loss: 1.516 CE: 1.376 Tri: 0.139 Acc: 0.957 LR: 3.49e-05
logs/agreidv2_baseline.log:218:Epoch[13] Iter[400/786] Loss: 1.508 CE: 1.370 Tri: 0.138 Acc: 0.958 LR: 3.49e-05
logs/agreidv2_baseline.log:219:Epoch[13] Iter[450/786] Loss: 1.502 CE: 1.365 Tri: 0.137 Acc: 0.959 LR: 3.49e-05
logs/agreidv2_baseline.log:220:Epoch[13] Iter[500/786] Loss: 1.497 CE: 1.360 Tri: 0.137 Acc: 0.960 LR: 3.49e-05
logs/agreidv2_baseline.log:221:Epoch[13] Iter[550/786] Loss: 1.492 CE: 1.355 Tri: 0.137 Acc: 0.961 LR: 3.49e-05
logs/agreidv2_baseline.log:222:Epoch[13] Iter[600/786] Loss: 1.486 CE: 1.350 Tri: 0.136 Acc: 0.962 LR: 3.49e-05
logs/agreidv2_baseline.log:223:Epoch[13] Iter[650/786] Loss: 1.480 CE: 1.345 Tri: 0.135 Acc: 0.963 LR: 3.49e-05
logs/agreidv2_baseline.log:224:Epoch[13] Iter[700/786] Loss: 1.471 CE: 1.338 Tri: 0.134 Acc: 0.964 LR: 3.49e-05
logs/agreidv2_baseline.log:225:Epoch[13] done in 189.1s  Loss=1.462 Acc=0.966
logs/agreidv2_baseline.log:226:Epoch[14] Iter[50/786] Loss: 1.577 CE: 1.426 Tri: 0.151 Acc: 0.946 LR: 3.47e-05
logs/agreidv2_baseline.log:227:Epoch[14] Iter[100/786] Loss: 1.524 CE: 1.389 Tri: 0.134 Acc: 0.955 LR: 3.47e-05
logs/agreidv2_baseline.log:228:Epoch[14] Iter[150/786] Loss: 1.515 CE: 1.379 Tri: 0.136 Acc: 0.956 LR: 3.47e-05
logs/agreidv2_baseline.log:229:Epoch[14] Iter[200/786] Loss: 1.502 CE: 1.371 Tri: 0.131 Acc: 0.958 LR: 3.47e-05
logs/agreidv2_baseline.log:230:Epoch[14] Iter[250/786] Loss: 1.491 CE: 1.360 Tri: 0.131 Acc: 0.961 LR: 3.47e-05
logs/agreidv2_baseline.log:231:Epoch[14] Iter[300/786] Loss: 1.482 CE: 1.353 Tri: 0.129 Acc: 0.962 LR: 3.47e-05
logs/agreidv2_baseline.log:232:Epoch[14] Iter[350/786] Loss: 1.471 CE: 1.347 Tri: 0.124 Acc: 0.963 LR: 3.47e-05
logs/agreidv2_baseline.log:233:Epoch[14] Iter[400/786] Loss: 1.469 CE: 1.342 Tri: 0.127 Acc: 0.964 LR: 3.47e-05
logs/agreidv2_baseline.log:234:Epoch[14] Iter[450/786] Loss: 1.461 CE: 1.337 Tri: 0.124 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_baseline.log:235:Epoch[14] Iter[500/786] Loss: 1.461 CE: 1.335 Tri: 0.126 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_baseline.log:236:Epoch[14] Iter[550/786] Loss: 1.459 CE: 1.331 Tri: 0.128 Acc: 0.966 LR: 3.47e-05
logs/agreidv2_baseline.log:237:Epoch[14] Iter[600/786] Loss: 1.455 CE: 1.327 Tri: 0.128 Acc: 0.966 LR: 3.47e-05
logs/agreidv2_baseline.log:238:Epoch[14] Iter[650/786] Loss: 1.449 CE: 1.322 Tri: 0.127 Acc: 0.967 LR: 3.47e-05
logs/agreidv2_baseline.log:239:Epoch[14] Iter[700/786] Loss: 1.443 CE: 1.316 Tri: 0.126 Acc: 0.968 LR: 3.47e-05
logs/agreidv2_baseline.log:240:Epoch[14] done in 191.2s  Loss=1.434 Acc=0.969
logs/agreidv2_baseline.log:241:Epoch[15] Iter[50/786] Loss: 1.598 CE: 1.436 Tri: 0.162 Acc: 0.942 LR: 3.45e-05
logs/agreidv2_baseline.log:242:Epoch[15] Iter[100/786] Loss: 1.532 CE: 1.393 Tri: 0.139 Acc: 0.952 LR: 3.45e-05
logs/agreidv2_baseline.log:243:Epoch[15] Iter[150/786] Loss: 1.508 CE: 1.374 Tri: 0.135 Acc: 0.957 LR: 3.45e-05
logs/agreidv2_baseline.log:244:Epoch[15] Iter[200/786] Loss: 1.491 CE: 1.362 Tri: 0.129 Acc: 0.960 LR: 3.45e-05
logs/agreidv2_baseline.log:245:Epoch[15] Iter[250/786] Loss: 1.471 CE: 1.348 Tri: 0.123 Acc: 0.962 LR: 3.45e-05
logs/agreidv2_baseline.log:246:Epoch[15] Iter[300/786] Loss: 1.462 CE: 1.341 Tri: 0.121 Acc: 0.963 LR: 3.45e-05
logs/agreidv2_baseline.log:247:Epoch[15] Iter[350/786] Loss: 1.456 CE: 1.337 Tri: 0.119 Acc: 0.963 LR: 3.45e-05
logs/agreidv2_baseline.log:248:Epoch[15] Iter[400/786] Loss: 1.449 CE: 1.333 Tri: 0.116 Acc: 0.964 LR: 3.45e-05
logs/agreidv2_baseline.log:249:Epoch[15] Iter[450/786] Loss: 1.440 CE: 1.326 Tri: 0.114 Acc: 0.965 LR: 3.45e-05
logs/agreidv2_baseline.log:250:Epoch[15] Iter[500/786] Loss: 1.435 CE: 1.322 Tri: 0.113 Acc: 0.966 LR: 3.45e-05
logs/agreidv2_baseline.log:251:Epoch[15] Iter[550/786] Loss: 1.432 CE: 1.318 Tri: 0.114 Acc: 0.967 LR: 3.45e-05
logs/agreidv2_baseline.log:252:Epoch[15] Iter[600/786] Loss: 1.428 CE: 1.314 Tri: 0.114 Acc: 0.968 LR: 3.45e-05
logs/agreidv2_baseline.log:253:Epoch[15] Iter[650/786] Loss: 1.420 CE: 1.308 Tri: 0.112 Acc: 0.969 LR: 3.45e-05
logs/agreidv2_baseline.log:254:Epoch[15] Iter[700/786] Loss: 1.411 CE: 1.301 Tri: 0.110 Acc: 0.970 LR: 3.45e-05
logs/agreidv2_baseline.log:255:Epoch[15] done in 190.5s  Loss=1.401 Acc=0.971
logs/agreidv2_baseline.log:256:Epoch[16] Iter[50/786] Loss: 1.487 CE: 1.376 Tri: 0.111 Acc: 0.958 LR: 3.41e-05
logs/agreidv2_baseline.log:257:Epoch[16] Iter[100/786] Loss: 1.449 CE: 1.346 Tri: 0.103 Acc: 0.962 LR: 3.41e-05
logs/agreidv2_baseline.log:258:Epoch[16] Iter[150/786] Loss: 1.439 CE: 1.338 Tri: 0.101 Acc: 0.963 LR: 3.41e-05
logs/agreidv2_baseline.log:259:Epoch[16] Iter[200/786] Loss: 1.441 CE: 1.332 Tri: 0.109 Acc: 0.964 LR: 3.41e-05
logs/agreidv2_baseline.log:260:Epoch[16] Iter[250/786] Loss: 1.435 CE: 1.327 Tri: 0.108 Acc: 0.965 LR: 3.41e-05
logs/agreidv2_baseline.log:261:Epoch[16] Iter[300/786] Loss: 1.433 CE: 1.323 Tri: 0.110 Acc: 0.965 LR: 3.41e-05
logs/agreidv2_baseline.log:262:Epoch[16] Iter[350/786] Loss: 1.429 CE: 1.319 Tri: 0.109 Acc: 0.965 LR: 3.41e-05
logs/agreidv2_baseline.log:263:Epoch[16] Iter[400/786] Loss: 1.424 CE: 1.314 Tri: 0.110 Acc: 0.966 LR: 3.41e-05
logs/agreidv2_baseline.log:264:Epoch[16] Iter[450/786] Loss: 1.417 CE: 1.310 Tri: 0.107 Acc: 0.967 LR: 3.41e-05
logs/agreidv2_baseline.log:265:Epoch[16] Iter[500/786] Loss: 1.412 CE: 1.305 Tri: 0.106 Acc: 0.968 LR: 3.41e-05
logs/agreidv2_baseline.log:266:Epoch[16] Iter[550/786] Loss: 1.407 CE: 1.302 Tri: 0.106 Acc: 0.969 LR: 3.41e-05
logs/agreidv2_baseline.log:267:Epoch[16] Iter[600/786] Loss: 1.401 CE: 1.297 Tri: 0.104 Acc: 0.970 LR: 3.41e-05
logs/agreidv2_baseline.log:268:Epoch[16] Iter[650/786] Loss: 1.392 CE: 1.290 Tri: 0.101 Acc: 0.971 LR: 3.41e-05
logs/agreidv2_baseline.log:269:Epoch[16] Iter[700/786] Loss: 1.384 CE: 1.284 Tri: 0.100 Acc: 0.972 LR: 3.41e-05
logs/agreidv2_baseline.log:270:Epoch[16] done in 190.0s  Loss=1.377 Acc=0.973
logs/agreidv2_baseline.log:271:Epoch[17] Iter[50/786] Loss: 1.473 CE: 1.370 Tri: 0.103 Acc: 0.956 LR: 3.38e-05
logs/agreidv2_baseline.log:272:Epoch[17] Iter[100/786] Loss: 1.450 CE: 1.342 Tri: 0.108 Acc: 0.963 LR: 3.38e-05
logs/agreidv2_baseline.log:273:Epoch[17] Iter[150/786] Loss: 1.431 CE: 1.326 Tri: 0.104 Acc: 0.966 LR: 3.38e-05
logs/agreidv2_baseline.log:274:Epoch[17] Iter[200/786] Loss: 1.417 CE: 1.314 Tri: 0.103 Acc: 0.969 LR: 3.38e-05
logs/agreidv2_baseline.log:275:Epoch[17] Iter[250/786] Loss: 1.405 CE: 1.305 Tri: 0.100 Acc: 0.970 LR: 3.38e-05
logs/agreidv2_baseline.log:276:Epoch[17] Iter[300/786] Loss: 1.400 CE: 1.300 Tri: 0.101 Acc: 0.971 LR: 3.38e-05
logs/agreidv2_baseline.log:277:Epoch[17] Iter[350/786] Loss: 1.396 CE: 1.297 Tri: 0.099 Acc: 0.971 LR: 3.38e-05
logs/agreidv2_baseline.log:278:Epoch[17] Iter[400/786] Loss: 1.390 CE: 1.292 Tri: 0.098 Acc: 0.972 LR: 3.38e-05
logs/agreidv2_baseline.log:279:Epoch[17] Iter[450/786] Loss: 1.387 CE: 1.289 Tri: 0.098 Acc: 0.973 LR: 3.38e-05
logs/agreidv2_baseline.log:280:Epoch[17] Iter[500/786] Loss: 1.384 CE: 1.286 Tri: 0.098 Acc: 0.973 LR: 3.38e-05
logs/agreidv2_baseline.log:281:Epoch[17] Iter[550/786] Loss: 1.382 CE: 1.284 Tri: 0.099 Acc: 0.973 LR: 3.38e-05
logs/agreidv2_baseline.log:282:Epoch[17] Iter[600/786] Loss: 1.377 CE: 1.280 Tri: 0.097 Acc: 0.974 LR: 3.38e-05
logs/agreidv2_baseline.log:283:Epoch[17] Iter[650/786] Loss: 1.370 CE: 1.275 Tri: 0.095 Acc: 0.974 LR: 3.38e-05
logs/agreidv2_baseline.log:284:Epoch[17] Iter[700/786] Loss: 1.363 CE: 1.269 Tri: 0.094 Acc: 0.975 LR: 3.38e-05
logs/agreidv2_baseline.log:285:Epoch[17] done in 191.3s  Loss=1.354 Acc=0.976
logs/agreidv2_baseline.log:286:Epoch[18] Iter[50/786] Loss: 1.444 CE: 1.350 Tri: 0.094 Acc: 0.960 LR: 3.33e-05
logs/agreidv2_baseline.log:287:Epoch[18] Iter[100/786] Loss: 1.405 CE: 1.317 Tri: 0.089 Acc: 0.965 LR: 3.33e-05
logs/agreidv2_baseline.log:288:Epoch[18] Iter[150/786] Loss: 1.387 CE: 1.301 Tri: 0.087 Acc: 0.967 LR: 3.33e-05
logs/agreidv2_baseline.log:289:Epoch[18] Iter[200/786] Loss: 1.384 CE: 1.293 Tri: 0.090 Acc: 0.969 LR: 3.33e-05
logs/agreidv2_baseline.log:290:Epoch[18] Iter[250/786] Loss: 1.380 CE: 1.288 Tri: 0.091 Acc: 0.970 LR: 3.33e-05
logs/agreidv2_baseline.log:291:Epoch[18] Iter[300/786] Loss: 1.373 CE: 1.284 Tri: 0.088 Acc: 0.971 LR: 3.33e-05
logs/agreidv2_baseline.log:292:Epoch[18] Iter[350/786] Loss: 1.369 CE: 1.280 Tri: 0.089 Acc: 0.972 LR: 3.33e-05
logs/agreidv2_baseline.log:293:Epoch[18] Iter[400/786] Loss: 1.362 CE: 1.275 Tri: 0.087 Acc: 0.973 LR: 3.33e-05
logs/agreidv2_baseline.log:294:Epoch[18] Iter[450/786] Loss: 1.357 CE: 1.271 Tri: 0.086 Acc: 0.974 LR: 3.33e-05
logs/agreidv2_baseline.log:295:Epoch[18] Iter[500/786] Loss: 1.350 CE: 1.268 Tri: 0.083 Acc: 0.974 LR: 3.33e-05
logs/agreidv2_baseline.log:296:Epoch[18] Iter[550/786] Loss: 1.347 CE: 1.264 Tri: 0.082 Acc: 0.975 LR: 3.33e-05
logs/agreidv2_baseline.log:297:Epoch[18] Iter[600/786] Loss: 1.342 CE: 1.260 Tri: 0.082 Acc: 0.976 LR: 3.33e-05
logs/agreidv2_baseline.log:298:Epoch[18] Iter[650/786] Loss: 1.337 CE: 1.255 Tri: 0.081 Acc: 0.976 LR: 3.33e-05
logs/agreidv2_baseline.log:299:Epoch[18] Iter[700/786] Loss: 1.331 CE: 1.250 Tri: 0.080 Acc: 0.977 LR: 3.33e-05
logs/agreidv2_baseline.log:300:Epoch[18] done in 191.0s  Loss=1.323 Acc=0.978
logs/agreidv2_baseline.log:301:Epoch[19] Iter[50/786] Loss: 1.414 CE: 1.337 Tri: 0.077 Acc: 0.959 LR: 3.28e-05
logs/agreidv2_baseline.log:302:Epoch[19] Iter[100/786] Loss: 1.378 CE: 1.306 Tri: 0.072 Acc: 0.968 LR: 3.28e-05
logs/agreidv2_baseline.log:303:Epoch[19] Iter[150/786] Loss: 1.367 CE: 1.291 Tri: 0.076 Acc: 0.972 LR: 3.28e-05
logs/agreidv2_baseline.log:304:Epoch[19] Iter[200/786] Loss: 1.358 CE: 1.282 Tri: 0.076 Acc: 0.973 LR: 3.28e-05
logs/agreidv2_baseline.log:305:Epoch[19] Iter[250/786] Loss: 1.352 CE: 1.275 Tri: 0.077 Acc: 0.974 LR: 3.28e-05
logs/agreidv2_baseline.log:306:Epoch[19] Iter[300/786] Loss: 1.352 CE: 1.273 Tri: 0.079 Acc: 0.974 LR: 3.28e-05
logs/agreidv2_baseline.log:307:Epoch[19] Iter[350/786] Loss: 1.345 CE: 1.269 Tri: 0.076 Acc: 0.975 LR: 3.28e-05
logs/agreidv2_baseline.log:308:Epoch[19] Iter[400/786] Loss: 1.344 CE: 1.268 Tri: 0.076 Acc: 0.974 LR: 3.28e-05
logs/agreidv2_baseline.log:309:Epoch[19] Iter[450/786] Loss: 1.342 CE: 1.265 Tri: 0.076 Acc: 0.974 LR: 3.28e-05
logs/agreidv2_baseline.log:310:Epoch[19] Iter[500/786] Loss: 1.339 CE: 1.262 Tri: 0.077 Acc: 0.975 LR: 3.28e-05
logs/agreidv2_baseline.log:311:Epoch[19] Iter[550/786] Loss: 1.337 CE: 1.259 Tri: 0.078 Acc: 0.975 LR: 3.28e-05
logs/agreidv2_baseline.log:312:Epoch[19] Iter[600/786] Loss: 1.332 CE: 1.255 Tri: 0.077 Acc: 0.976 LR: 3.28e-05
logs/agreidv2_baseline.log:313:Epoch[19] Iter[650/786] Loss: 1.327 CE: 1.251 Tri: 0.075 Acc: 0.976 LR: 3.28e-05
logs/agreidv2_baseline.log:314:Epoch[19] Iter[700/786] Loss: 1.319 CE: 1.245 Tri: 0.074 Acc: 0.977 LR: 3.28e-05
logs/agreidv2_baseline.log:315:Epoch[19] done in 191.6s  Loss=1.311 Acc=0.978
logs/agreidv2_baseline.log:316:Epoch[20] Iter[50/786] Loss: 1.389 CE: 1.327 Tri: 0.062 Acc: 0.962 LR: 3.23e-05
logs/agreidv2_baseline.log:317:Epoch[20] Iter[100/786] Loss: 1.374 CE: 1.301 Tri: 0.073 Acc: 0.969 LR: 3.23e-05
logs/agreidv2_baseline.log:318:Epoch[20] Iter[150/786] Loss: 1.363 CE: 1.292 Tri: 0.071 Acc: 0.969 LR: 3.23e-05
logs/agreidv2_baseline.log:319:Epoch[20] Iter[200/786] Loss: 1.358 CE: 1.282 Tri: 0.075 Acc: 0.970 LR: 3.23e-05
logs/agreidv2_baseline.log:320:Epoch[20] Iter[250/786] Loss: 1.351 CE: 1.275 Tri: 0.077 Acc: 0.972 LR: 3.23e-05
logs/agreidv2_baseline.log:321:Epoch[20] Iter[300/786] Loss: 1.343 CE: 1.268 Tri: 0.075 Acc: 0.974 LR: 3.23e-05
logs/agreidv2_baseline.log:322:Epoch[20] Iter[350/786] Loss: 1.334 CE: 1.261 Tri: 0.073 Acc: 0.975 LR: 3.23e-05
logs/agreidv2_baseline.log:323:Epoch[20] Iter[400/786] Loss: 1.326 CE: 1.256 Tri: 0.070 Acc: 0.976 LR: 3.23e-05
logs/agreidv2_baseline.log:324:Epoch[20] Iter[450/786] Loss: 1.322 CE: 1.251 Tri: 0.071 Acc: 0.976 LR: 3.23e-05
logs/agreidv2_baseline.log:325:Epoch[20] Iter[500/786] Loss: 1.319 CE: 1.249 Tri: 0.070 Acc: 0.977 LR: 3.23e-05
logs/agreidv2_baseline.log:326:Epoch[20] Iter[550/786] Loss: 1.314 CE: 1.244 Tri: 0.070 Acc: 0.977 LR: 3.23e-05
logs/agreidv2_baseline.log:327:Epoch[20] Iter[600/786] Loss: 1.310 CE: 1.241 Tri: 0.069 Acc: 0.978 LR: 3.23e-05
logs/agreidv2_baseline.log:328:Epoch[20] Iter[650/786] Loss: 1.305 CE: 1.237 Tri: 0.069 Acc: 0.979 LR: 3.23e-05
logs/agreidv2_baseline.log:329:Epoch[20] Iter[700/786] Loss: 1.300 CE: 1.232 Tri: 0.068 Acc: 0.979 LR: 3.23e-05
logs/agreidv2_baseline.log:330:Epoch[20] done in 191.5s  Loss=1.291 Acc=0.980
logs/agreidv2_baseline.log:332:    [A->G] mAP=71.25  R1=80.22  R5=87.69  mINP=41.91
logs/agreidv2_baseline.log:333:    [G->A] mAP=71.19  R1=80.73  R5=87.85  mINP=34.72
logs/agreidv2_baseline.log:334:    [mean] mAP=71.22  R1=80.47
logs/agreidv2_baseline.log:335:Epoch[21] Iter[50/786] Loss: 1.418 CE: 1.320 Tri: 0.098 Acc: 0.965 LR: 3.17e-05
logs/agreidv2_baseline.log:336:Epoch[21] Iter[100/786] Loss: 1.375 CE: 1.288 Tri: 0.087 Acc: 0.971 LR: 3.17e-05
logs/agreidv2_baseline.log:337:Epoch[21] Iter[150/786] Loss: 1.358 CE: 1.273 Tri: 0.085 Acc: 0.974 LR: 3.17e-05
logs/agreidv2_baseline.log:338:Epoch[21] Iter[200/786] Loss: 1.343 CE: 1.265 Tri: 0.078 Acc: 0.975 LR: 3.17e-05
logs/agreidv2_baseline.log:339:Epoch[21] Iter[250/786] Loss: 1.332 CE: 1.258 Tri: 0.075 Acc: 0.975 LR: 3.17e-05
logs/agreidv2_baseline.log:340:Epoch[21] Iter[300/786] Loss: 1.326 CE: 1.253 Tri: 0.073 Acc: 0.975 LR: 3.17e-05
logs/agreidv2_baseline.log:341:Epoch[21] Iter[350/786] Loss: 1.323 CE: 1.251 Tri: 0.073 Acc: 0.975 LR: 3.17e-05
logs/agreidv2_baseline.log:342:Epoch[21] Iter[400/786] Loss: 1.318 CE: 1.247 Tri: 0.071 Acc: 0.976 LR: 3.17e-05
logs/agreidv2_baseline.log:343:Epoch[21] Iter[450/786] Loss: 1.313 CE: 1.243 Tri: 0.070 Acc: 0.976 LR: 3.17e-05
logs/agreidv2_baseline.log:344:Epoch[21] Iter[500/786] Loss: 1.310 CE: 1.240 Tri: 0.070 Acc: 0.977 LR: 3.17e-05
logs/agreidv2_baseline.log:345:Epoch[21] Iter[550/786] Loss: 1.307 CE: 1.237 Tri: 0.070 Acc: 0.977 LR: 3.17e-05
logs/agreidv2_baseline.log:346:Epoch[21] Iter[600/786] Loss: 1.305 CE: 1.235 Tri: 0.071 Acc: 0.977 LR: 3.17e-05
logs/agreidv2_baseline.log:347:Epoch[21] Iter[650/786] Loss: 1.302 CE: 1.231 Tri: 0.071 Acc: 0.978 LR: 3.17e-05
logs/agreidv2_baseline.log:348:Epoch[21] Iter[700/786] Loss: 1.295 CE: 1.226 Tri: 0.069 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_baseline.log:349:Epoch[21] done in 187.1s  Loss=1.289 Acc=0.979
logs/agreidv2_baseline.log:350:Epoch[22] Iter[50/786] Loss: 1.398 CE: 1.309 Tri: 0.089 Acc: 0.967 LR: 3.10e-05
logs/agreidv2_baseline.log:351:Epoch[22] Iter[100/786] Loss: 1.353 CE: 1.277 Tri: 0.076 Acc: 0.972 LR: 3.10e-05
logs/agreidv2_baseline.log:352:Epoch[22] Iter[150/786] Loss: 1.335 CE: 1.260 Tri: 0.074 Acc: 0.975 LR: 3.10e-05
logs/agreidv2_baseline.log:353:Epoch[22] Iter[200/786] Loss: 1.321 CE: 1.250 Tri: 0.071 Acc: 0.978 LR: 3.10e-05
logs/agreidv2_baseline.log:354:Epoch[22] Iter[250/786] Loss: 1.313 CE: 1.243 Tri: 0.070 Acc: 0.979 LR: 3.10e-05
logs/agreidv2_baseline.log:355:Epoch[22] Iter[300/786] Loss: 1.304 CE: 1.236 Tri: 0.068 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_baseline.log:356:Epoch[22] Iter[350/786] Loss: 1.296 CE: 1.231 Tri: 0.066 Acc: 0.981 LR: 3.10e-05
logs/agreidv2_baseline.log:357:Epoch[22] Iter[400/786] Loss: 1.292 CE: 1.228 Tri: 0.064 Acc: 0.981 LR: 3.10e-05
logs/agreidv2_baseline.log:358:Epoch[22] Iter[450/786] Loss: 1.289 CE: 1.224 Tri: 0.065 Acc: 0.981 LR: 3.10e-05
logs/agreidv2_baseline.log:359:Epoch[22] Iter[500/786] Loss: 1.284 CE: 1.221 Tri: 0.063 Acc: 0.982 LR: 3.10e-05
logs/agreidv2_baseline.log:360:Epoch[22] Iter[550/786] Loss: 1.282 CE: 1.219 Tri: 0.063 Acc: 0.982 LR: 3.10e-05
logs/agreidv2_baseline.log:361:Epoch[22] Iter[600/786] Loss: 1.278 CE: 1.215 Tri: 0.062 Acc: 0.982 LR: 3.10e-05
logs/agreidv2_baseline.log:362:Epoch[22] Iter[650/786] Loss: 1.274 CE: 1.212 Tri: 0.061 Acc: 0.983 LR: 3.10e-05
logs/agreidv2_baseline.log:363:Epoch[22] Iter[700/786] Loss: 1.269 CE: 1.208 Tri: 0.060 Acc: 0.983 LR: 3.10e-05
logs/agreidv2_baseline.log:364:Epoch[22] done in 189.0s  Loss=1.261 Acc=0.984
logs/agreidv2_baseline.log:365:Epoch[23] Iter[50/786] Loss: 1.339 CE: 1.272 Tri: 0.067 Acc: 0.974 LR: 3.03e-05
logs/agreidv2_baseline.log:366:Epoch[23] Iter[100/786] Loss: 1.326 CE: 1.256 Tri: 0.070 Acc: 0.976 LR: 3.03e-05
logs/agreidv2_baseline.log:367:Epoch[23] Iter[150/786] Loss: 1.317 CE: 1.247 Tri: 0.070 Acc: 0.977 LR: 3.03e-05
logs/agreidv2_baseline.log:368:Epoch[23] Iter[200/786] Loss: 1.308 CE: 1.239 Tri: 0.069 Acc: 0.978 LR: 3.03e-05
logs/agreidv2_baseline.log:369:Epoch[23] Iter[250/786] Loss: 1.300 CE: 1.233 Tri: 0.067 Acc: 0.979 LR: 3.03e-05
logs/agreidv2_baseline.log:370:Epoch[23] Iter[300/786] Loss: 1.296 CE: 1.229 Tri: 0.067 Acc: 0.979 LR: 3.03e-05
logs/agreidv2_baseline.log:371:Epoch[23] Iter[350/786] Loss: 1.287 CE: 1.223 Tri: 0.064 Acc: 0.980 LR: 3.03e-05
logs/agreidv2_baseline.log:372:Epoch[23] Iter[400/786] Loss: 1.282 CE: 1.220 Tri: 0.062 Acc: 0.980 LR: 3.03e-05
logs/agreidv2_baseline.log:373:Epoch[23] Iter[450/786] Loss: 1.279 CE: 1.217 Tri: 0.061 Acc: 0.980 LR: 3.03e-05
logs/agreidv2_baseline.log:374:Epoch[23] Iter[500/786] Loss: 1.273 CE: 1.214 Tri: 0.059 Acc: 0.981 LR: 3.03e-05
logs/agreidv2_baseline.log:375:Epoch[23] Iter[550/786] Loss: 1.268 CE: 1.210 Tri: 0.058 Acc: 0.981 LR: 3.03e-05
logs/agreidv2_baseline.log:376:Epoch[23] Iter[600/786] Loss: 1.262 CE: 1.207 Tri: 0.056 Acc: 0.982 LR: 3.03e-05
logs/agreidv2_baseline.log:377:Epoch[23] Iter[650/786] Loss: 1.261 CE: 1.204 Tri: 0.057 Acc: 0.982 LR: 3.03e-05
logs/agreidv2_baseline.log:378:Epoch[23] Iter[700/786] Loss: 1.255 CE: 1.200 Tri: 0.055 Acc: 0.983 LR: 3.03e-05
logs/agreidv2_baseline.log:379:Epoch[23] done in 189.3s  Loss=1.249 Acc=0.983
logs/agreidv2_baseline.log:380:Epoch[24] Iter[50/786] Loss: 1.330 CE: 1.274 Tri: 0.056 Acc: 0.972 LR: 2.95e-05
logs/agreidv2_baseline.log:381:Epoch[24] Iter[100/786] Loss: 1.302 CE: 1.250 Tri: 0.052 Acc: 0.975 LR: 2.95e-05
logs/agreidv2_baseline.log:382:Epoch[24] Iter[150/786] Loss: 1.293 CE: 1.240 Tri: 0.054 Acc: 0.976 LR: 2.95e-05
logs/agreidv2_baseline.log:383:Epoch[24] Iter[200/786] Loss: 1.290 CE: 1.231 Tri: 0.059 Acc: 0.978 LR: 2.95e-05
logs/agreidv2_baseline.log:384:Epoch[24] Iter[250/786] Loss: 1.286 CE: 1.226 Tri: 0.059 Acc: 0.978 LR: 2.95e-05
logs/agreidv2_baseline.log:385:Epoch[24] Iter[300/786] Loss: 1.278 CE: 1.221 Tri: 0.057 Acc: 0.979 LR: 2.95e-05
logs/agreidv2_baseline.log:386:Epoch[24] Iter[350/786] Loss: 1.273 CE: 1.216 Tri: 0.057 Acc: 0.980 LR: 2.95e-05
logs/agreidv2_baseline.log:387:Epoch[24] Iter[400/786] Loss: 1.269 CE: 1.214 Tri: 0.055 Acc: 0.980 LR: 2.95e-05
logs/agreidv2_baseline.log:388:Epoch[24] Iter[450/786] Loss: 1.266 CE: 1.211 Tri: 0.055 Acc: 0.981 LR: 2.95e-05
logs/agreidv2_baseline.log:389:Epoch[24] Iter[500/786] Loss: 1.260 CE: 1.207 Tri: 0.053 Acc: 0.981 LR: 2.95e-05
logs/agreidv2_baseline.log:390:Epoch[24] Iter[550/786] Loss: 1.256 CE: 1.204 Tri: 0.052 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_baseline.log:391:Epoch[24] Iter[600/786] Loss: 1.251 CE: 1.201 Tri: 0.050 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_baseline.log:392:Epoch[24] Iter[650/786] Loss: 1.247 CE: 1.197 Tri: 0.050 Acc: 0.983 LR: 2.95e-05
logs/agreidv2_baseline.log:393:Epoch[24] Iter[700/786] Loss: 1.242 CE: 1.193 Tri: 0.049 Acc: 0.983 LR: 2.95e-05
logs/agreidv2_baseline.log:394:Epoch[24] done in 191.9s  Loss=1.234 Acc=0.984
logs/agreidv2_baseline.log:395:Epoch[25] Iter[50/786] Loss: 1.304 CE: 1.241 Tri: 0.063 Acc: 0.976 LR: 2.87e-05
logs/agreidv2_baseline.log:396:Epoch[25] Iter[100/786] Loss: 1.276 CE: 1.222 Tri: 0.054 Acc: 0.980 LR: 2.87e-05
logs/agreidv2_baseline.log:397:Epoch[25] Iter[150/786] Loss: 1.262 CE: 1.211 Tri: 0.050 Acc: 0.982 LR: 2.87e-05
logs/agreidv2_baseline.log:398:Epoch[25] Iter[200/786] Loss: 1.256 CE: 1.206 Tri: 0.050 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_baseline.log:399:Epoch[25] Iter[250/786] Loss: 1.257 CE: 1.205 Tri: 0.052 Acc: 0.982 LR: 2.87e-05
logs/agreidv2_baseline.log:400:Epoch[25] Iter[300/786] Loss: 1.256 CE: 1.204 Tri: 0.052 Acc: 0.982 LR: 2.87e-05
logs/agreidv2_baseline.log:401:Epoch[25] Iter[350/786] Loss: 1.250 CE: 1.199 Tri: 0.050 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_baseline.log:402:Epoch[25] Iter[400/786] Loss: 1.246 CE: 1.196 Tri: 0.051 Acc: 0.984 LR: 2.87e-05
logs/agreidv2_baseline.log:403:Epoch[25] Iter[450/786] Loss: 1.242 CE: 1.192 Tri: 0.049 Acc: 0.984 LR: 2.87e-05
logs/agreidv2_baseline.log:404:Epoch[25] Iter[500/786] Loss: 1.238 CE: 1.190 Tri: 0.049 Acc: 0.985 LR: 2.87e-05
logs/agreidv2_baseline.log:405:Epoch[25] Iter[550/786] Loss: 1.236 CE: 1.188 Tri: 0.048 Acc: 0.985 LR: 2.87e-05
logs/agreidv2_baseline.log:406:Epoch[25] Iter[600/786] Loss: 1.232 CE: 1.185 Tri: 0.047 Acc: 0.986 LR: 2.87e-05
logs/agreidv2_baseline.log:407:Epoch[25] Iter[650/786] Loss: 1.228 CE: 1.181 Tri: 0.046 Acc: 0.986 LR: 2.87e-05
logs/agreidv2_baseline.log:408:Epoch[25] Iter[700/786] Loss: 1.224 CE: 1.178 Tri: 0.046 Acc: 0.986 LR: 2.87e-05
logs/agreidv2_baseline.log:409:Epoch[25] done in 190.0s  Loss=1.218 Acc=0.987
logs/agreidv2_baseline.log:410:Epoch[26] Iter[50/786] Loss: 1.285 CE: 1.227 Tri: 0.058 Acc: 0.976 LR: 2.78e-05
logs/agreidv2_baseline.log:411:Epoch[26] Iter[100/786] Loss: 1.257 CE: 1.212 Tri: 0.046 Acc: 0.980 LR: 2.78e-05
logs/agreidv2_baseline.log:412:Epoch[26] Iter[150/786] Loss: 1.248 CE: 1.203 Tri: 0.044 Acc: 0.981 LR: 2.78e-05
logs/agreidv2_baseline.log:413:Epoch[26] Iter[200/786] Loss: 1.247 CE: 1.201 Tri: 0.046 Acc: 0.982 LR: 2.78e-05
logs/agreidv2_baseline.log:414:Epoch[26] Iter[250/786] Loss: 1.241 CE: 1.196 Tri: 0.045 Acc: 0.982 LR: 2.78e-05
logs/agreidv2_baseline.log:415:Epoch[26] Iter[300/786] Loss: 1.238 CE: 1.194 Tri: 0.044 Acc: 0.982 LR: 2.78e-05
logs/agreidv2_baseline.log:416:Epoch[26] Iter[350/786] Loss: 1.236 CE: 1.192 Tri: 0.044 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_baseline.log:417:Epoch[26] Iter[400/786] Loss: 1.234 CE: 1.189 Tri: 0.045 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_baseline.log:418:Epoch[26] Iter[450/786] Loss: 1.233 CE: 1.187 Tri: 0.046 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_baseline.log:419:Epoch[26] Iter[500/786] Loss: 1.231 CE: 1.185 Tri: 0.046 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_baseline.log:420:Epoch[26] Iter[550/786] Loss: 1.228 CE: 1.182 Tri: 0.046 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_baseline.log:421:Epoch[26] Iter[600/786] Loss: 1.226 CE: 1.180 Tri: 0.045 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_baseline.log:422:Epoch[26] Iter[650/786] Loss: 1.222 CE: 1.177 Tri: 0.045 Acc: 0.985 LR: 2.78e-05
logs/agreidv2_baseline.log:423:Epoch[26] Iter[700/786] Loss: 1.216 CE: 1.172 Tri: 0.043 Acc: 0.985 LR: 2.78e-05
logs/agreidv2_baseline.log:424:Epoch[26] done in 190.1s  Loss=1.210 Acc=0.986
logs/agreidv2_baseline.log:425:Epoch[27] Iter[50/786] Loss: 1.258 CE: 1.217 Tri: 0.041 Acc: 0.980 LR: 2.69e-05
logs/agreidv2_baseline.log:426:Epoch[27] Iter[100/786] Loss: 1.250 CE: 1.206 Tri: 0.044 Acc: 0.981 LR: 2.69e-05
logs/agreidv2_baseline.log:427:Epoch[27] Iter[150/786] Loss: 1.244 CE: 1.198 Tri: 0.047 Acc: 0.982 LR: 2.69e-05
logs/agreidv2_baseline.log:428:Epoch[27] Iter[200/786] Loss: 1.237 CE: 1.191 Tri: 0.046 Acc: 0.983 LR: 2.69e-05
logs/agreidv2_baseline.log:429:Epoch[27] Iter[250/786] Loss: 1.234 CE: 1.188 Tri: 0.046 Acc: 0.983 LR: 2.69e-05
logs/agreidv2_baseline.log:430:Epoch[27] Iter[300/786] Loss: 1.233 CE: 1.186 Tri: 0.047 Acc: 0.984 LR: 2.69e-05
logs/agreidv2_baseline.log:431:Epoch[27] Iter[350/786] Loss: 1.227 CE: 1.182 Tri: 0.045 Acc: 0.984 LR: 2.69e-05
logs/agreidv2_baseline.log:432:Epoch[27] Iter[400/786] Loss: 1.221 CE: 1.178 Tri: 0.042 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_baseline.log:433:Epoch[27] Iter[450/786] Loss: 1.215 CE: 1.174 Tri: 0.040 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline.log:434:Epoch[27] Iter[500/786] Loss: 1.212 CE: 1.172 Tri: 0.040 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline.log:435:Epoch[27] Iter[550/786] Loss: 1.211 CE: 1.170 Tri: 0.041 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_baseline.log:436:Epoch[27] Iter[600/786] Loss: 1.208 CE: 1.168 Tri: 0.041 Acc: 0.987 LR: 2.69e-05
logs/agreidv2_baseline.log:437:Epoch[27] Iter[650/786] Loss: 1.205 CE: 1.165 Tri: 0.040 Acc: 0.987 LR: 2.69e-05
logs/agreidv2_baseline.log:438:Epoch[27] Iter[700/786] Loss: 1.200 CE: 1.161 Tri: 0.039 Acc: 0.988 LR: 2.69e-05
logs/agreidv2_baseline.log:439:Epoch[27] done in 190.1s  Loss=1.195 Acc=0.988
logs/agreidv2_baseline.log:440:Epoch[28] Iter[50/786] Loss: 1.250 CE: 1.208 Tri: 0.042 Acc: 0.977 LR: 2.59e-05
logs/agreidv2_baseline.log:441:Epoch[28] Iter[100/786] Loss: 1.232 CE: 1.192 Tri: 0.040 Acc: 0.982 LR: 2.59e-05
logs/agreidv2_baseline.log:442:Epoch[28] Iter[150/786] Loss: 1.227 CE: 1.186 Tri: 0.041 Acc: 0.983 LR: 2.59e-05
logs/agreidv2_baseline.log:443:Epoch[28] Iter[200/786] Loss: 1.221 CE: 1.181 Tri: 0.040 Acc: 0.984 LR: 2.59e-05
logs/agreidv2_baseline.log:444:Epoch[28] Iter[250/786] Loss: 1.219 CE: 1.179 Tri: 0.041 Acc: 0.984 LR: 2.59e-05
logs/agreidv2_baseline.log:445:Epoch[28] Iter[300/786] Loss: 1.217 CE: 1.176 Tri: 0.041 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_baseline.log:446:Epoch[28] Iter[350/786] Loss: 1.215 CE: 1.174 Tri: 0.041 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_baseline.log:447:Epoch[28] Iter[400/786] Loss: 1.210 CE: 1.171 Tri: 0.039 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_baseline.log:448:Epoch[28] Iter[450/786] Loss: 1.208 CE: 1.169 Tri: 0.039 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_baseline.log:449:Epoch[28] Iter[500/786] Loss: 1.204 CE: 1.166 Tri: 0.038 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_baseline.log:450:Epoch[28] Iter[550/786] Loss: 1.201 CE: 1.164 Tri: 0.037 Acc: 0.987 LR: 2.59e-05
logs/agreidv2_baseline.log:451:Epoch[28] Iter[600/786] Loss: 1.198 CE: 1.161 Tri: 0.036 Acc: 0.987 LR: 2.59e-05
logs/agreidv2_baseline.log:452:Epoch[28] Iter[650/786] Loss: 1.194 CE: 1.159 Tri: 0.035 Acc: 0.987 LR: 2.59e-05
logs/agreidv2_baseline.log:453:Epoch[28] Iter[700/786] Loss: 1.190 CE: 1.155 Tri: 0.035 Acc: 0.988 LR: 2.59e-05
logs/agreidv2_baseline.log:454:Epoch[28] done in 190.8s  Loss=1.185 Acc=0.988
logs/agreidv2_baseline.log:455:Epoch[29] Iter[50/786] Loss: 1.222 CE: 1.193 Tri: 0.029 Acc: 0.984 LR: 2.50e-05
logs/agreidv2_baseline.log:456:Epoch[29] Iter[100/786] Loss: 1.223 CE: 1.186 Tri: 0.038 Acc: 0.983 LR: 2.50e-05
logs/agreidv2_baseline.log:457:Epoch[29] Iter[150/786] Loss: 1.224 CE: 1.182 Tri: 0.042 Acc: 0.984 LR: 2.50e-05
logs/agreidv2_baseline.log:458:Epoch[29] Iter[200/786] Loss: 1.218 CE: 1.177 Tri: 0.040 Acc: 0.985 LR: 2.50e-05
logs/agreidv2_baseline.log:459:Epoch[29] Iter[250/786] Loss: 1.212 CE: 1.173 Tri: 0.039 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_baseline.log:460:Epoch[29] Iter[300/786] Loss: 1.209 CE: 1.169 Tri: 0.039 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_baseline.log:461:Epoch[29] Iter[350/786] Loss: 1.205 CE: 1.166 Tri: 0.039 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_baseline.log:462:Epoch[29] Iter[400/786] Loss: 1.200 CE: 1.163 Tri: 0.037 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline.log:463:Epoch[29] Iter[450/786] Loss: 1.199 CE: 1.161 Tri: 0.037 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline.log:464:Epoch[29] Iter[500/786] Loss: 1.195 CE: 1.159 Tri: 0.036 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline.log:465:Epoch[29] Iter[550/786] Loss: 1.193 CE: 1.158 Tri: 0.035 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline.log:466:Epoch[29] Iter[600/786] Loss: 1.192 CE: 1.156 Tri: 0.036 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline.log:467:Epoch[29] Iter[650/786] Loss: 1.189 CE: 1.154 Tri: 0.035 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_baseline.log:468:Epoch[29] Iter[700/786] Loss: 1.186 CE: 1.151 Tri: 0.035 Acc: 0.988 LR: 2.50e-05
logs/agreidv2_baseline.log:469:Epoch[29] done in 191.6s  Loss=1.180 Acc=0.988
logs/agreidv2_baseline.log:470:Epoch[30] Iter[50/786] Loss: 1.223 CE: 1.186 Tri: 0.037 Acc: 0.984 LR: 2.39e-05
logs/agreidv2_baseline.log:471:Epoch[30] Iter[100/786] Loss: 1.213 CE: 1.178 Tri: 0.035 Acc: 0.984 LR: 2.39e-05
logs/agreidv2_baseline.log:472:Epoch[30] Iter[150/786] Loss: 1.212 CE: 1.172 Tri: 0.040 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_baseline.log:473:Epoch[30] Iter[200/786] Loss: 1.215 CE: 1.172 Tri: 0.043 Acc: 0.985 LR: 2.39e-05
logs/agreidv2_baseline.log:474:Epoch[30] Iter[250/786] Loss: 1.209 CE: 1.168 Tri: 0.041 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_baseline.log:475:Epoch[30] Iter[300/786] Loss: 1.204 CE: 1.163 Tri: 0.040 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_baseline.log:476:Epoch[30] Iter[350/786] Loss: 1.200 CE: 1.160 Tri: 0.040 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_baseline.log:477:Epoch[30] Iter[400/786] Loss: 1.197 CE: 1.159 Tri: 0.039 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_baseline.log:478:Epoch[30] Iter[450/786] Loss: 1.195 CE: 1.157 Tri: 0.039 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_baseline.log:479:Epoch[30] Iter[500/786] Loss: 1.193 CE: 1.155 Tri: 0.038 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_baseline.log:480:Epoch[30] Iter[550/786] Loss: 1.189 CE: 1.151 Tri: 0.037 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_baseline.log:481:Epoch[30] Iter[600/786] Loss: 1.186 CE: 1.150 Tri: 0.036 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_baseline.log:482:Epoch[30] Iter[650/786] Loss: 1.182 CE: 1.147 Tri: 0.035 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_baseline.log:483:Epoch[30] Iter[700/786] Loss: 1.177 CE: 1.143 Tri: 0.034 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_baseline.log:484:Epoch[30] done in 192.0s  Loss=1.170 Acc=0.990
logs/agreidv2_baseline.log:486:    [A->G] mAP=72.09  R1=80.65  R5=88.16  mINP=43.73
logs/agreidv2_baseline.log:487:    [G->A] mAP=73.32  R1=82.88  R5=89.51  mINP=36.44
logs/agreidv2_baseline.log:488:    [mean] mAP=72.71  R1=81.76
logs/agreidv2_baseline.log:489:Epoch[31] Iter[50/786] Loss: 1.207 CE: 1.178 Tri: 0.028 Acc: 0.987 LR: 2.29e-05
logs/agreidv2_baseline.log:490:Epoch[31] Iter[100/786] Loss: 1.203 CE: 1.170 Tri: 0.033 Acc: 0.987 LR: 2.29e-05
logs/agreidv2_baseline.log:491:Epoch[31] Iter[150/786] Loss: 1.198 CE: 1.165 Tri: 0.033 Acc: 0.987 LR: 2.29e-05
logs/agreidv2_baseline.log:492:Epoch[31] Iter[200/786] Loss: 1.189 CE: 1.159 Tri: 0.030 Acc: 0.987 LR: 2.29e-05
logs/agreidv2_baseline.log:493:Epoch[31] Iter[250/786] Loss: 1.184 CE: 1.154 Tri: 0.030 Acc: 0.987 LR: 2.29e-05
logs/agreidv2_baseline.log:494:Epoch[31] Iter[300/786] Loss: 1.182 CE: 1.151 Tri: 0.031 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline.log:495:Epoch[31] Iter[350/786] Loss: 1.179 CE: 1.148 Tri: 0.031 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline.log:496:Epoch[31] Iter[400/786] Loss: 1.175 CE: 1.145 Tri: 0.029 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline.log:497:Epoch[31] Iter[450/786] Loss: 1.172 CE: 1.144 Tri: 0.029 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_baseline.log:498:Epoch[31] Iter[500/786] Loss: 1.171 CE: 1.142 Tri: 0.029 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_baseline.log:499:Epoch[31] Iter[550/786] Loss: 1.168 CE: 1.140 Tri: 0.028 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_baseline.log:500:Epoch[31] Iter[600/786] Loss: 1.164 CE: 1.137 Tri: 0.027 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_baseline.log:501:Epoch[31] Iter[650/786] Loss: 1.161 CE: 1.135 Tri: 0.026 Acc: 0.990 LR: 2.29e-05
logs/agreidv2_baseline.log:502:Epoch[31] Iter[700/786] Loss: 1.157 CE: 1.132 Tri: 0.025 Acc: 0.990 LR: 2.29e-05
logs/agreidv2_baseline.log:503:Epoch[31] done in 188.3s  Loss=1.153 Acc=0.990
logs/agreidv2_baseline.log:504:Epoch[32] Iter[50/786] Loss: 1.233 CE: 1.193 Tri: 0.040 Acc: 0.982 LR: 2.19e-05
logs/agreidv2_baseline.log:505:Epoch[32] Iter[100/786] Loss: 1.209 CE: 1.171 Tri: 0.038 Acc: 0.985 LR: 2.19e-05
logs/agreidv2_baseline.log:506:Epoch[32] Iter[150/786] Loss: 1.197 CE: 1.163 Tri: 0.035 Acc: 0.987 LR: 2.19e-05
logs/agreidv2_baseline.log:507:Epoch[32] Iter[200/786] Loss: 1.188 CE: 1.156 Tri: 0.033 Acc: 0.988 LR: 2.19e-05
logs/agreidv2_baseline.log:508:Epoch[32] Iter[250/786] Loss: 1.185 CE: 1.153 Tri: 0.032 Acc: 0.988 LR: 2.19e-05
logs/agreidv2_baseline.log:509:Epoch[32] Iter[300/786] Loss: 1.181 CE: 1.150 Tri: 0.031 Acc: 0.988 LR: 2.19e-05
logs/agreidv2_baseline.log:510:Epoch[32] Iter[350/786] Loss: 1.176 CE: 1.146 Tri: 0.030 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline.log:511:Epoch[32] Iter[400/786] Loss: 1.176 CE: 1.145 Tri: 0.031 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline.log:512:Epoch[32] Iter[450/786] Loss: 1.173 CE: 1.143 Tri: 0.030 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline.log:513:Epoch[32] Iter[500/786] Loss: 1.172 CE: 1.142 Tri: 0.030 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline.log:514:Epoch[32] Iter[550/786] Loss: 1.168 CE: 1.139 Tri: 0.029 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_baseline.log:515:Epoch[32] Iter[600/786] Loss: 1.165 CE: 1.137 Tri: 0.028 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_baseline.log:516:Epoch[32] Iter[650/786] Loss: 1.162 CE: 1.135 Tri: 0.027 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_baseline.log:517:Epoch[32] Iter[700/786] Loss: 1.158 CE: 1.132 Tri: 0.026 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_baseline.log:518:Epoch[32] done in 188.6s  Loss=1.153 Acc=0.990
logs/agreidv2_baseline.log:519:Epoch[33] Iter[50/786] Loss: 1.186 CE: 1.161 Tri: 0.025 Acc: 0.985 LR: 2.08e-05
logs/agreidv2_baseline.log:520:Epoch[33] Iter[100/786] Loss: 1.178 CE: 1.151 Tri: 0.028 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_baseline.log:521:Epoch[33] Iter[150/786] Loss: 1.173 CE: 1.144 Tri: 0.029 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline.log:522:Epoch[33] Iter[200/786] Loss: 1.175 CE: 1.143 Tri: 0.031 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_baseline.log:523:Epoch[33] Iter[250/786] Loss: 1.170 CE: 1.140 Tri: 0.030 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline.log:524:Epoch[33] Iter[300/786] Loss: 1.168 CE: 1.139 Tri: 0.028 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_baseline.log:525:Epoch[33] Iter[350/786] Loss: 1.164 CE: 1.137 Tri: 0.027 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline.log:526:Epoch[33] Iter[400/786] Loss: 1.162 CE: 1.135 Tri: 0.028 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline.log:527:Epoch[33] Iter[450/786] Loss: 1.161 CE: 1.134 Tri: 0.028 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_baseline.log:528:Epoch[33] Iter[500/786] Loss: 1.158 CE: 1.132 Tri: 0.027 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_baseline.log:529:Epoch[33] Iter[550/786] Loss: 1.156 CE: 1.129 Tri: 0.026 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_baseline.log:530:Epoch[33] Iter[600/786] Loss: 1.153 CE: 1.128 Tri: 0.026 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_baseline.log:531:Epoch[33] Iter[650/786] Loss: 1.150 CE: 1.125 Tri: 0.025 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_baseline.log:532:Epoch[33] Iter[700/786] Loss: 1.147 CE: 1.123 Tri: 0.024 Acc: 0.991 LR: 2.08e-05
logs/agreidv2_baseline.log:533:Epoch[33] done in 188.4s  Loss=1.143 Acc=0.991
logs/agreidv2_baseline.log:534:Epoch[34] Iter[50/786] Loss: 1.181 CE: 1.154 Tri: 0.027 Acc: 0.987 LR: 1.97e-05
logs/agreidv2_baseline.log:535:Epoch[34] Iter[100/786] Loss: 1.169 CE: 1.145 Tri: 0.024 Acc: 0.989 LR: 1.97e-05
logs/agreidv2_baseline.log:536:Epoch[34] Iter[150/786] Loss: 1.167 CE: 1.140 Tri: 0.026 Acc: 0.989 LR: 1.97e-05
logs/agreidv2_baseline.log:537:Epoch[34] Iter[200/786] Loss: 1.160 CE: 1.136 Tri: 0.024 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_baseline.log:538:Epoch[34] Iter[250/786] Loss: 1.154 CE: 1.131 Tri: 0.023 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline.log:539:Epoch[34] Iter[300/786] Loss: 1.152 CE: 1.129 Tri: 0.024 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_baseline.log:540:Epoch[34] Iter[350/786] Loss: 1.149 CE: 1.126 Tri: 0.023 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline.log:541:Epoch[34] Iter[400/786] Loss: 1.147 CE: 1.124 Tri: 0.023 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline.log:542:Epoch[34] Iter[450/786] Loss: 1.143 CE: 1.122 Tri: 0.021 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline.log:543:Epoch[34] Iter[500/786] Loss: 1.143 CE: 1.121 Tri: 0.022 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline.log:544:Epoch[34] Iter[550/786] Loss: 1.141 CE: 1.119 Tri: 0.021 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline.log:545:Epoch[34] Iter[600/786] Loss: 1.139 CE: 1.118 Tri: 0.021 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline.log:546:Epoch[34] Iter[650/786] Loss: 1.137 CE: 1.116 Tri: 0.021 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline.log:547:Epoch[34] Iter[700/786] Loss: 1.134 CE: 1.114 Tri: 0.021 Acc: 0.992 LR: 1.97e-05
logs/agreidv2_baseline.log:548:Epoch[34] done in 189.4s  Loss=1.131 Acc=0.993
logs/agreidv2_baseline.log:549:Epoch[35] Iter[50/786] Loss: 1.171 CE: 1.147 Tri: 0.024 Acc: 0.988 LR: 1.86e-05
logs/agreidv2_baseline.log:550:Epoch[35] Iter[100/786] Loss: 1.172 CE: 1.142 Tri: 0.030 Acc: 0.988 LR: 1.86e-05
logs/agreidv2_baseline.log:551:Epoch[35] Iter[150/786] Loss: 1.158 CE: 1.134 Tri: 0.025 Acc: 0.989 LR: 1.86e-05
logs/agreidv2_baseline.log:552:Epoch[35] Iter[200/786] Loss: 1.156 CE: 1.131 Tri: 0.026 Acc: 0.989 LR: 1.86e-05
logs/agreidv2_baseline.log:553:Epoch[35] Iter[250/786] Loss: 1.150 CE: 1.127 Tri: 0.023 Acc: 0.990 LR: 1.86e-05
logs/agreidv2_baseline.log:554:Epoch[35] Iter[300/786] Loss: 1.147 CE: 1.125 Tri: 0.022 Acc: 0.990 LR: 1.86e-05
logs/agreidv2_baseline.log:555:Epoch[35] Iter[350/786] Loss: 1.144 CE: 1.123 Tri: 0.022 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline.log:556:Epoch[35] Iter[400/786] Loss: 1.144 CE: 1.121 Tri: 0.022 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline.log:557:Epoch[35] Iter[450/786] Loss: 1.143 CE: 1.121 Tri: 0.023 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline.log:558:Epoch[35] Iter[500/786] Loss: 1.142 CE: 1.119 Tri: 0.022 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline.log:559:Epoch[35] Iter[550/786] Loss: 1.140 CE: 1.118 Tri: 0.022 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline.log:560:Epoch[35] Iter[600/786] Loss: 1.138 CE: 1.116 Tri: 0.021 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline.log:561:Epoch[35] Iter[650/786] Loss: 1.135 CE: 1.114 Tri: 0.021 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_baseline.log:562:Epoch[35] Iter[700/786] Loss: 1.132 CE: 1.112 Tri: 0.020 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_baseline.log:563:Epoch[35] done in 189.8s  Loss=1.128 Acc=0.992
logs/agreidv2_baseline.log:564:Epoch[36] Iter[50/786] Loss: 1.159 CE: 1.137 Tri: 0.022 Acc: 0.985 LR: 1.75e-05
logs/agreidv2_baseline.log:565:Epoch[36] Iter[100/786] Loss: 1.152 CE: 1.131 Tri: 0.021 Acc: 0.987 LR: 1.75e-05
logs/agreidv2_baseline.log:566:Epoch[36] Iter[150/786] Loss: 1.144 CE: 1.125 Tri: 0.020 Acc: 0.988 LR: 1.75e-05
logs/agreidv2_baseline.log:567:Epoch[36] Iter[200/786] Loss: 1.144 CE: 1.121 Tri: 0.023 Acc: 0.989 LR: 1.75e-05
logs/agreidv2_baseline.log:568:Epoch[36] Iter[250/786] Loss: 1.137 CE: 1.116 Tri: 0.020 Acc: 0.990 LR: 1.75e-05
logs/agreidv2_baseline.log:569:Epoch[36] Iter[300/786] Loss: 1.136 CE: 1.115 Tri: 0.021 Acc: 0.990 LR: 1.75e-05
logs/agreidv2_baseline.log:570:Epoch[36] Iter[350/786] Loss: 1.132 CE: 1.112 Tri: 0.019 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_baseline.log:571:Epoch[36] Iter[400/786] Loss: 1.130 CE: 1.111 Tri: 0.019 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_baseline.log:572:Epoch[36] Iter[450/786] Loss: 1.129 CE: 1.109 Tri: 0.019 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_baseline.log:573:Epoch[36] Iter[500/786] Loss: 1.126 CE: 1.108 Tri: 0.018 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline.log:574:Epoch[36] Iter[550/786] Loss: 1.124 CE: 1.106 Tri: 0.018 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline.log:575:Epoch[36] Iter[600/786] Loss: 1.123 CE: 1.105 Tri: 0.018 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline.log:576:Epoch[36] Iter[650/786] Loss: 1.121 CE: 1.104 Tri: 0.017 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline.log:577:Epoch[36] Iter[700/786] Loss: 1.119 CE: 1.102 Tri: 0.017 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_baseline.log:578:Epoch[36] done in 190.6s  Loss=1.116 Acc=0.993
logs/agreidv2_baseline.log:579:Epoch[37] Iter[50/786] Loss: 1.147 CE: 1.125 Tri: 0.022 Acc: 0.990 LR: 1.64e-05
logs/agreidv2_baseline.log:580:Epoch[37] Iter[100/786] Loss: 1.138 CE: 1.118 Tri: 0.020 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_baseline.log:581:Epoch[37] Iter[150/786] Loss: 1.134 CE: 1.115 Tri: 0.019 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_baseline.log:582:Epoch[37] Iter[200/786] Loss: 1.131 CE: 1.113 Tri: 0.018 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_baseline.log:583:Epoch[37] Iter[250/786] Loss: 1.130 CE: 1.112 Tri: 0.017 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_baseline.log:584:Epoch[37] Iter[300/786] Loss: 1.129 CE: 1.111 Tri: 0.018 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_baseline.log:585:Epoch[37] Iter[350/786] Loss: 1.126 CE: 1.109 Tri: 0.017 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline.log:586:Epoch[37] Iter[400/786] Loss: 1.125 CE: 1.108 Tri: 0.017 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_baseline.log:587:Epoch[37] Iter[450/786] Loss: 1.125 CE: 1.107 Tri: 0.017 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline.log:588:Epoch[37] Iter[500/786] Loss: 1.121 CE: 1.105 Tri: 0.016 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline.log:589:Epoch[37] Iter[550/786] Loss: 1.119 CE: 1.104 Tri: 0.016 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_baseline.log:590:Epoch[37] Iter[600/786] Loss: 1.116 CE: 1.102 Tri: 0.015 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_baseline.log:591:Epoch[37] Iter[650/786] Loss: 1.115 CE: 1.100 Tri: 0.014 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_baseline.log:592:Epoch[37] Iter[700/786] Loss: 1.112 CE: 1.098 Tri: 0.014 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_baseline.log:593:Epoch[37] done in 191.3s  Loss=1.109 Acc=0.993
logs/agreidv2_baseline.log:594:Epoch[38] Iter[50/786] Loss: 1.146 CE: 1.122 Tri: 0.024 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline.log:595:Epoch[38] Iter[100/786] Loss: 1.131 CE: 1.113 Tri: 0.018 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline.log:596:Epoch[38] Iter[150/786] Loss: 1.129 CE: 1.111 Tri: 0.018 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_baseline.log:597:Epoch[38] Iter[200/786] Loss: 1.126 CE: 1.108 Tri: 0.018 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline.log:598:Epoch[38] Iter[250/786] Loss: 1.125 CE: 1.107 Tri: 0.018 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline.log:599:Epoch[38] Iter[300/786] Loss: 1.122 CE: 1.104 Tri: 0.018 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline.log:600:Epoch[38] Iter[350/786] Loss: 1.121 CE: 1.103 Tri: 0.018 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline.log:601:Epoch[38] Iter[400/786] Loss: 1.118 CE: 1.101 Tri: 0.016 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline.log:602:Epoch[38] Iter[450/786] Loss: 1.117 CE: 1.100 Tri: 0.017 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_baseline.log:603:Epoch[38] Iter[500/786] Loss: 1.115 CE: 1.099 Tri: 0.016 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline.log:604:Epoch[38] Iter[550/786] Loss: 1.114 CE: 1.098 Tri: 0.016 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline.log:605:Epoch[38] Iter[600/786] Loss: 1.112 CE: 1.097 Tri: 0.016 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline.log:606:Epoch[38] Iter[650/786] Loss: 1.110 CE: 1.095 Tri: 0.015 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_baseline.log:607:Epoch[38] Iter[700/786] Loss: 1.107 CE: 1.093 Tri: 0.014 Acc: 0.994 LR: 1.53e-05
logs/agreidv2_baseline.log:608:Epoch[38] done in 191.0s  Loss=1.104 Acc=0.994
logs/agreidv2_baseline.log:609:Epoch[39] Iter[50/786] Loss: 1.127 CE: 1.112 Tri: 0.016 Acc: 0.994 LR: 1.42e-05
logs/agreidv2_baseline.log:610:Epoch[39] Iter[100/786] Loss: 1.121 CE: 1.108 Tri: 0.013 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline.log:611:Epoch[39] Iter[150/786] Loss: 1.115 CE: 1.103 Tri: 0.013 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline.log:612:Epoch[39] Iter[200/786] Loss: 1.113 CE: 1.101 Tri: 0.012 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline.log:613:Epoch[39] Iter[250/786] Loss: 1.112 CE: 1.100 Tri: 0.012 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_baseline.log:614:Epoch[39] Iter[300/786] Loss: 1.111 CE: 1.098 Tri: 0.012 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline.log:615:Epoch[39] Iter[350/786] Loss: 1.109 CE: 1.096 Tri: 0.013 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline.log:616:Epoch[39] Iter[400/786] Loss: 1.109 CE: 1.096 Tri: 0.013 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline.log:617:Epoch[39] Iter[450/786] Loss: 1.109 CE: 1.096 Tri: 0.013 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline.log:618:Epoch[39] Iter[500/786] Loss: 1.108 CE: 1.095 Tri: 0.013 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline.log:619:Epoch[39] Iter[550/786] Loss: 1.108 CE: 1.094 Tri: 0.014 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline.log:620:Epoch[39] Iter[600/786] Loss: 1.105 CE: 1.092 Tri: 0.013 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_baseline.log:621:Epoch[39] Iter[650/786] Loss: 1.104 CE: 1.091 Tri: 0.013 Acc: 0.994 LR: 1.42e-05
logs/agreidv2_baseline.log:622:Epoch[39] Iter[700/786] Loss: 1.102 CE: 1.089 Tri: 0.013 Acc: 0.994 LR: 1.42e-05
logs/agreidv2_baseline.log:623:Epoch[39] done in 190.4s  Loss=1.099 Acc=0.994
logs/agreidv2_baseline.log:624:Epoch[40] Iter[50/786] Loss: 1.123 CE: 1.109 Tri: 0.014 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_baseline.log:625:Epoch[40] Iter[100/786] Loss: 1.122 CE: 1.105 Tri: 0.016 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_baseline.log:626:Epoch[40] Iter[150/786] Loss: 1.117 CE: 1.102 Tri: 0.016 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline.log:627:Epoch[40] Iter[200/786] Loss: 1.112 CE: 1.098 Tri: 0.014 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline.log:628:Epoch[40] Iter[250/786] Loss: 1.109 CE: 1.096 Tri: 0.014 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline.log:629:Epoch[40] Iter[300/786] Loss: 1.109 CE: 1.095 Tri: 0.014 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline.log:630:Epoch[40] Iter[350/786] Loss: 1.107 CE: 1.093 Tri: 0.014 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline.log:631:Epoch[40] Iter[400/786] Loss: 1.106 CE: 1.092 Tri: 0.014 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline.log:632:Epoch[40] Iter[450/786] Loss: 1.104 CE: 1.091 Tri: 0.013 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline.log:633:Epoch[40] Iter[500/786] Loss: 1.103 CE: 1.090 Tri: 0.013 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_baseline.log:634:Epoch[40] Iter[550/786] Loss: 1.101 CE: 1.088 Tri: 0.013 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline.log:635:Epoch[40] Iter[600/786] Loss: 1.100 CE: 1.087 Tri: 0.013 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline.log:636:Epoch[40] Iter[650/786] Loss: 1.099 CE: 1.086 Tri: 0.013 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline.log:637:Epoch[40] Iter[700/786] Loss: 1.096 CE: 1.084 Tri: 0.012 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_baseline.log:638:Epoch[40] done in 190.3s  Loss=1.094 Acc=0.994
logs/agreidv2_baseline.log:640:    [A->G] mAP=76.79  R1=84.08  R5=90.24  mINP=51.34
logs/agreidv2_baseline.log:641:    [G->A] mAP=76.84  R1=84.98  R5=90.28  mINP=41.65
logs/agreidv2_baseline.log:642:    [mean] mAP=76.82  R1=84.53
logs/agreidv2_baseline.log:643:    * new best mean mAP=76.82 (epoch 40) saved
logs/agreidv2_baseline.log:644:Epoch[41] Iter[50/786] Loss: 1.122 CE: 1.104 Tri: 0.018 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline.log:645:Epoch[41] Iter[100/786] Loss: 1.119 CE: 1.101 Tri: 0.018 Acc: 0.991 LR: 1.21e-05
logs/agreidv2_baseline.log:646:Epoch[41] Iter[150/786] Loss: 1.112 CE: 1.097 Tri: 0.015 Acc: 0.992 LR: 1.21e-05
logs/agreidv2_baseline.log:647:Epoch[41] Iter[200/786] Loss: 1.108 CE: 1.094 Tri: 0.014 Acc: 0.992 LR: 1.21e-05
logs/agreidv2_baseline.log:648:Epoch[41] Iter[250/786] Loss: 1.106 CE: 1.092 Tri: 0.013 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline.log:649:Epoch[41] Iter[300/786] Loss: 1.104 CE: 1.091 Tri: 0.013 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline.log:650:Epoch[41] Iter[350/786] Loss: 1.102 CE: 1.089 Tri: 0.013 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline.log:651:Epoch[41] Iter[400/786] Loss: 1.101 CE: 1.089 Tri: 0.012 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline.log:652:Epoch[41] Iter[450/786] Loss: 1.100 CE: 1.088 Tri: 0.013 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline.log:653:Epoch[41] Iter[500/786] Loss: 1.099 CE: 1.087 Tri: 0.013 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline.log:654:Epoch[41] Iter[550/786] Loss: 1.097 CE: 1.085 Tri: 0.012 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline.log:655:Epoch[41] Iter[600/786] Loss: 1.096 CE: 1.084 Tri: 0.012 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_baseline.log:656:Epoch[41] Iter[650/786] Loss: 1.094 CE: 1.082 Tri: 0.012 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline.log:657:Epoch[41] Iter[700/786] Loss: 1.091 CE: 1.080 Tri: 0.011 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_baseline.log:658:Epoch[41] done in 187.8s  Loss=1.089 Acc=0.994
logs/agreidv2_baseline.log:659:Epoch[42] Iter[50/786] Loss: 1.102 CE: 1.092 Tri: 0.010 Acc: 0.991 LR: 1.11e-05
logs/agreidv2_baseline.log:660:Epoch[42] Iter[100/786] Loss: 1.100 CE: 1.091 Tri: 0.010 Acc: 0.992 LR: 1.11e-05
logs/agreidv2_baseline.log:661:Epoch[42] Iter[150/786] Loss: 1.099 CE: 1.089 Tri: 0.010 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_baseline.log:662:Epoch[42] Iter[200/786] Loss: 1.096 CE: 1.086 Tri: 0.010 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline.log:663:Epoch[42] Iter[250/786] Loss: 1.095 CE: 1.085 Tri: 0.010 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline.log:664:Epoch[42] Iter[300/786] Loss: 1.094 CE: 1.084 Tri: 0.010 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline.log:665:Epoch[42] Iter[350/786] Loss: 1.094 CE: 1.083 Tri: 0.010 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline.log:666:Epoch[42] Iter[400/786] Loss: 1.092 CE: 1.082 Tri: 0.010 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline.log:667:Epoch[42] Iter[450/786] Loss: 1.092 CE: 1.081 Tri: 0.010 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline.log:668:Epoch[42] Iter[500/786] Loss: 1.092 CE: 1.081 Tri: 0.011 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline.log:669:Epoch[42] Iter[550/786] Loss: 1.090 CE: 1.080 Tri: 0.010 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline.log:670:Epoch[42] Iter[600/786] Loss: 1.089 CE: 1.079 Tri: 0.010 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_baseline.log:671:Epoch[42] Iter[650/786] Loss: 1.087 CE: 1.077 Tri: 0.010 Acc: 0.995 LR: 1.11e-05
logs/agreidv2_baseline.log:672:Epoch[42] Iter[700/786] Loss: 1.086 CE: 1.076 Tri: 0.010 Acc: 0.995 LR: 1.11e-05
logs/agreidv2_baseline.log:673:Epoch[42] done in 187.5s  Loss=1.083 Acc=0.995
logs/agreidv2_baseline.log:674:Epoch[43] Iter[50/786] Loss: 1.100 CE: 1.090 Tri: 0.010 Acc: 0.993 LR: 1.00e-05
logs/agreidv2_baseline.log:675:Epoch[43] Iter[100/786] Loss: 1.094 CE: 1.086 Tri: 0.008 Acc: 0.993 LR: 1.00e-05
logs/agreidv2_baseline.log:676:Epoch[43] Iter[150/786] Loss: 1.089 CE: 1.081 Tri: 0.007 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline.log:677:Epoch[43] Iter[200/786] Loss: 1.089 CE: 1.080 Tri: 0.009 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline.log:678:Epoch[43] Iter[250/786] Loss: 1.089 CE: 1.080 Tri: 0.009 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline.log:679:Epoch[43] Iter[300/786] Loss: 1.088 CE: 1.079 Tri: 0.009 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline.log:680:Epoch[43] Iter[350/786] Loss: 1.087 CE: 1.078 Tri: 0.009 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline.log:681:Epoch[43] Iter[400/786] Loss: 1.087 CE: 1.078 Tri: 0.009 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline.log:682:Epoch[43] Iter[450/786] Loss: 1.086 CE: 1.077 Tri: 0.009 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline.log:683:Epoch[43] Iter[500/786] Loss: 1.085 CE: 1.076 Tri: 0.009 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_baseline.log:684:Epoch[43] Iter[550/786] Loss: 1.083 CE: 1.074 Tri: 0.008 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline.log:685:Epoch[43] Iter[600/786] Loss: 1.081 CE: 1.073 Tri: 0.008 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline.log:686:Epoch[43] Iter[650/786] Loss: 1.080 CE: 1.072 Tri: 0.008 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline.log:687:Epoch[43] Iter[700/786] Loss: 1.078 CE: 1.070 Tri: 0.007 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_baseline.log:688:Epoch[43] done in 188.4s  Loss=1.076 Acc=0.995
logs/agreidv2_baseline.log:689:Epoch[44] Iter[50/786] Loss: 1.115 CE: 1.094 Tri: 0.021 Acc: 0.990 LR: 9.07e-06
logs/agreidv2_baseline.log:690:Epoch[44] Iter[100/786] Loss: 1.103 CE: 1.088 Tri: 0.015 Acc: 0.992 LR: 9.07e-06
logs/agreidv2_baseline.log:691:Epoch[44] Iter[150/786] Loss: 1.096 CE: 1.082 Tri: 0.013 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_baseline.log:692:Epoch[44] Iter[200/786] Loss: 1.097 CE: 1.081 Tri: 0.016 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_baseline.log:693:Epoch[44] Iter[250/786] Loss: 1.092 CE: 1.078 Tri: 0.015 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline.log:694:Epoch[44] Iter[300/786] Loss: 1.090 CE: 1.076 Tri: 0.013 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline.log:695:Epoch[44] Iter[350/786] Loss: 1.090 CE: 1.077 Tri: 0.014 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_baseline.log:696:Epoch[44] Iter[400/786] Loss: 1.088 CE: 1.075 Tri: 0.013 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline.log:697:Epoch[44] Iter[450/786] Loss: 1.086 CE: 1.074 Tri: 0.012 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline.log:698:Epoch[44] Iter[500/786] Loss: 1.085 CE: 1.073 Tri: 0.012 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_baseline.log:699:Epoch[44] Iter[550/786] Loss: 1.083 CE: 1.072 Tri: 0.011 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline.log:700:Epoch[44] Iter[600/786] Loss: 1.081 CE: 1.071 Tri: 0.010 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline.log:701:Epoch[44] Iter[650/786] Loss: 1.080 CE: 1.070 Tri: 0.010 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline.log:702:Epoch[44] Iter[700/786] Loss: 1.078 CE: 1.068 Tri: 0.010 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_baseline.log:703:Epoch[44] done in 190.9s  Loss=1.076 Acc=0.995
logs/agreidv2_baseline.log:704:Epoch[45] Iter[50/786] Loss: 1.089 CE: 1.083 Tri: 0.006 Acc: 0.993 LR: 8.12e-06
logs/agreidv2_baseline.log:705:Epoch[45] Iter[100/786] Loss: 1.085 CE: 1.077 Tri: 0.008 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline.log:706:Epoch[45] Iter[150/786] Loss: 1.083 CE: 1.075 Tri: 0.007 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline.log:707:Epoch[45] Iter[200/786] Loss: 1.082 CE: 1.075 Tri: 0.007 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline.log:708:Epoch[45] Iter[250/786] Loss: 1.081 CE: 1.073 Tri: 0.007 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline.log:709:Epoch[45] Iter[300/786] Loss: 1.080 CE: 1.072 Tri: 0.008 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_baseline.log:710:Epoch[45] Iter[350/786] Loss: 1.078 CE: 1.071 Tri: 0.007 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline.log:711:Epoch[45] Iter[400/786] Loss: 1.078 CE: 1.071 Tri: 0.007 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline.log:712:Epoch[45] Iter[450/786] Loss: 1.077 CE: 1.070 Tri: 0.008 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline.log:713:Epoch[45] Iter[500/786] Loss: 1.076 CE: 1.069 Tri: 0.007 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline.log:714:Epoch[45] Iter[550/786] Loss: 1.075 CE: 1.068 Tri: 0.007 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline.log:715:Epoch[45] Iter[600/786] Loss: 1.074 CE: 1.067 Tri: 0.007 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline.log:716:Epoch[45] Iter[650/786] Loss: 1.072 CE: 1.066 Tri: 0.007 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline.log:717:Epoch[45] Iter[700/786] Loss: 1.071 CE: 1.065 Tri: 0.006 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_baseline.log:718:Epoch[45] done in 190.2s  Loss=1.069 Acc=0.996
logs/agreidv2_baseline.log:719:Epoch[46] Iter[50/786] Loss: 1.098 CE: 1.082 Tri: 0.016 Acc: 0.993 LR: 7.21e-06
logs/agreidv2_baseline.log:720:Epoch[46] Iter[100/786] Loss: 1.092 CE: 1.078 Tri: 0.014 Acc: 0.993 LR: 7.21e-06
logs/agreidv2_baseline.log:721:Epoch[46] Iter[150/786] Loss: 1.091 CE: 1.077 Tri: 0.014 Acc: 0.993 LR: 7.21e-06
logs/agreidv2_baseline.log:722:Epoch[46] Iter[200/786] Loss: 1.088 CE: 1.075 Tri: 0.013 Acc: 0.993 LR: 7.21e-06
logs/agreidv2_baseline.log:723:Epoch[46] Iter[250/786] Loss: 1.085 CE: 1.073 Tri: 0.012 Acc: 0.994 LR: 7.21e-06
logs/agreidv2_baseline.log:724:Epoch[46] Iter[300/786] Loss: 1.084 CE: 1.072 Tri: 0.012 Acc: 0.994 LR: 7.21e-06
logs/agreidv2_baseline.log:725:Epoch[46] Iter[350/786] Loss: 1.081 CE: 1.071 Tri: 0.011 Acc: 0.994 LR: 7.21e-06
logs/agreidv2_baseline.log:726:Epoch[46] Iter[400/786] Loss: 1.079 CE: 1.069 Tri: 0.010 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline.log:727:Epoch[46] Iter[450/786] Loss: 1.079 CE: 1.069 Tri: 0.010 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline.log:728:Epoch[46] Iter[500/786] Loss: 1.077 CE: 1.067 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline.log:729:Epoch[46] Iter[550/786] Loss: 1.076 CE: 1.067 Tri: 0.010 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline.log:730:Epoch[46] Iter[600/786] Loss: 1.075 CE: 1.066 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline.log:731:Epoch[46] Iter[650/786] Loss: 1.073 CE: 1.065 Tri: 0.009 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline.log:732:Epoch[46] Iter[700/786] Loss: 1.072 CE: 1.063 Tri: 0.008 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_baseline.log:733:Epoch[46] done in 191.6s  Loss=1.069 Acc=0.996
logs/agreidv2_baseline.log:734:Epoch[47] Iter[50/786] Loss: 1.075 CE: 1.071 Tri: 0.005 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline.log:735:Epoch[47] Iter[100/786] Loss: 1.078 CE: 1.071 Tri: 0.007 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline.log:736:Epoch[47] Iter[150/786] Loss: 1.078 CE: 1.071 Tri: 0.007 Acc: 0.994 LR: 6.35e-06
logs/agreidv2_baseline.log:737:Epoch[47] Iter[200/786] Loss: 1.078 CE: 1.071 Tri: 0.007 Acc: 0.994 LR: 6.35e-06
logs/agreidv2_baseline.log:738:Epoch[47] Iter[250/786] Loss: 1.076 CE: 1.069 Tri: 0.007 Acc: 0.994 LR: 6.35e-06
logs/agreidv2_baseline.log:739:Epoch[47] Iter[300/786] Loss: 1.076 CE: 1.068 Tri: 0.007 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline.log:740:Epoch[47] Iter[350/786] Loss: 1.076 CE: 1.068 Tri: 0.007 Acc: 0.994 LR: 6.35e-06
logs/agreidv2_baseline.log:741:Epoch[47] Iter[400/786] Loss: 1.073 CE: 1.067 Tri: 0.007 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline.log:742:Epoch[47] Iter[450/786] Loss: 1.072 CE: 1.066 Tri: 0.007 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline.log:743:Epoch[47] Iter[500/786] Loss: 1.072 CE: 1.065 Tri: 0.007 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline.log:744:Epoch[47] Iter[550/786] Loss: 1.070 CE: 1.064 Tri: 0.007 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_baseline.log:745:Epoch[47] Iter[600/786] Loss: 1.069 CE: 1.063 Tri: 0.006 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline.log:746:Epoch[47] Iter[650/786] Loss: 1.068 CE: 1.062 Tri: 0.006 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline.log:747:Epoch[47] Iter[700/786] Loss: 1.067 CE: 1.061 Tri: 0.006 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_baseline.log:748:Epoch[47] done in 189.0s  Loss=1.065 Acc=0.996
logs/agreidv2_baseline.log:749:Epoch[48] Iter[50/786] Loss: 1.073 CE: 1.069 Tri: 0.004 Acc: 0.993 LR: 5.52e-06
logs/agreidv2_baseline.log:750:Epoch[48] Iter[100/786] Loss: 1.077 CE: 1.069 Tri: 0.008 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline.log:751:Epoch[48] Iter[150/786] Loss: 1.080 CE: 1.070 Tri: 0.010 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline.log:752:Epoch[48] Iter[200/786] Loss: 1.078 CE: 1.069 Tri: 0.009 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline.log:753:Epoch[48] Iter[250/786] Loss: 1.077 CE: 1.068 Tri: 0.009 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline.log:754:Epoch[48] Iter[300/786] Loss: 1.076 CE: 1.067 Tri: 0.009 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline.log:755:Epoch[48] Iter[350/786] Loss: 1.075 CE: 1.066 Tri: 0.009 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline.log:756:Epoch[48] Iter[400/786] Loss: 1.073 CE: 1.065 Tri: 0.008 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_baseline.log:757:Epoch[48] Iter[450/786] Loss: 1.071 CE: 1.063 Tri: 0.008 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline.log:758:Epoch[48] Iter[500/786] Loss: 1.071 CE: 1.063 Tri: 0.008 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline.log:759:Epoch[48] Iter[550/786] Loss: 1.069 CE: 1.061 Tri: 0.008 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_baseline.log:760:Epoch[48] Iter[600/786] Loss: 1.068 CE: 1.060 Tri: 0.007 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_baseline.log:761:Epoch[48] Iter[650/786] Loss: 1.066 CE: 1.059 Tri: 0.007 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_baseline.log:762:Epoch[48] Iter[700/786] Loss: 1.065 CE: 1.058 Tri: 0.007 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_baseline.log:763:Epoch[48] done in 190.0s  Loss=1.063 Acc=0.996
logs/agreidv2_baseline.log:764:Epoch[49] Iter[50/786] Loss: 1.067 CE: 1.064 Tri: 0.002 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline.log:765:Epoch[49] Iter[100/786] Loss: 1.073 CE: 1.065 Tri: 0.008 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline.log:766:Epoch[49] Iter[150/786] Loss: 1.075 CE: 1.066 Tri: 0.009 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_baseline.log:767:Epoch[49] Iter[200/786] Loss: 1.074 CE: 1.065 Tri: 0.009 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_baseline.log:768:Epoch[49] Iter[250/786] Loss: 1.074 CE: 1.066 Tri: 0.008 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_baseline.log:769:Epoch[49] Iter[300/786] Loss: 1.072 CE: 1.065 Tri: 0.008 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_baseline.log:770:Epoch[49] Iter[350/786] Loss: 1.073 CE: 1.065 Tri: 0.008 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_baseline.log:771:Epoch[49] Iter[400/786] Loss: 1.071 CE: 1.064 Tri: 0.007 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_baseline.log:772:Epoch[49] Iter[450/786] Loss: 1.070 CE: 1.063 Tri: 0.007 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline.log:773:Epoch[49] Iter[500/786] Loss: 1.069 CE: 1.062 Tri: 0.007 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline.log:774:Epoch[49] Iter[550/786] Loss: 1.067 CE: 1.061 Tri: 0.007 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_baseline.log:775:Epoch[49] Iter[600/786] Loss: 1.066 CE: 1.059 Tri: 0.007 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline.log:776:Epoch[49] Iter[650/786] Loss: 1.065 CE: 1.059 Tri: 0.006 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline.log:777:Epoch[49] Iter[700/786] Loss: 1.064 CE: 1.058 Tri: 0.006 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_baseline.log:778:Epoch[49] done in 189.7s  Loss=1.062 Acc=0.996
logs/agreidv2_baseline.log:779:Epoch[50] Iter[50/786] Loss: 1.062 CE: 1.059 Tri: 0.003 Acc: 0.998 LR: 4.02e-06
logs/agreidv2_baseline.log:780:Epoch[50] Iter[100/786] Loss: 1.064 CE: 1.060 Tri: 0.003 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_baseline.log:781:Epoch[50] Iter[150/786] Loss: 1.069 CE: 1.062 Tri: 0.007 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline.log:782:Epoch[50] Iter[200/786] Loss: 1.069 CE: 1.061 Tri: 0.008 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline.log:783:Epoch[50] Iter[250/786] Loss: 1.068 CE: 1.060 Tri: 0.007 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline.log:784:Epoch[50] Iter[300/786] Loss: 1.067 CE: 1.060 Tri: 0.008 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_baseline.log:785:Epoch[50] Iter[350/786] Loss: 1.066 CE: 1.059 Tri: 0.007 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_baseline.log:786:Epoch[50] Iter[400/786] Loss: 1.065 CE: 1.058 Tri: 0.007 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_baseline.log:787:Epoch[50] Iter[450/786] Loss: 1.064 CE: 1.057 Tri: 0.007 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_baseline.log:788:Epoch[50] Iter[500/786] Loss: 1.062 CE: 1.056 Tri: 0.006 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_baseline.log:789:Epoch[50] Iter[550/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_baseline.log:790:Epoch[50] Iter[600/786] Loss: 1.060 CE: 1.055 Tri: 0.006 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_baseline.log:791:Epoch[50] Iter[650/786] Loss: 1.060 CE: 1.054 Tri: 0.006 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_baseline.log:792:Epoch[50] Iter[700/786] Loss: 1.058 CE: 1.053 Tri: 0.006 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_baseline.log:793:Epoch[50] done in 191.0s  Loss=1.057 Acc=0.997
logs/agreidv2_baseline.log:795:    [A->G] mAP=79.14  R1=85.78  R5=92.32  mINP=55.52
logs/agreidv2_baseline.log:796:    [G->A] mAP=79.29  R1=86.97  R5=91.50  mINP=45.80
logs/agreidv2_baseline.log:797:    [mean] mAP=79.22  R1=86.37
logs/agreidv2_baseline.log:798:    * new best mean mAP=79.22 (epoch 50) saved
logs/agreidv2_baseline.log:799:Epoch[51] Iter[50/786] Loss: 1.064 CE: 1.060 Tri: 0.004 Acc: 0.995 LR: 3.34e-06
logs/agreidv2_baseline.log:800:Epoch[51] Iter[100/786] Loss: 1.067 CE: 1.060 Tri: 0.007 Acc: 0.995 LR: 3.34e-06
logs/agreidv2_baseline.log:801:Epoch[51] Iter[150/786] Loss: 1.065 CE: 1.059 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline.log:802:Epoch[51] Iter[200/786] Loss: 1.063 CE: 1.058 Tri: 0.005 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline.log:803:Epoch[51] Iter[250/786] Loss: 1.064 CE: 1.058 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline.log:804:Epoch[51] Iter[300/786] Loss: 1.063 CE: 1.058 Tri: 0.006 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline.log:805:Epoch[51] Iter[350/786] Loss: 1.062 CE: 1.057 Tri: 0.005 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline.log:806:Epoch[51] Iter[400/786] Loss: 1.062 CE: 1.057 Tri: 0.005 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline.log:807:Epoch[51] Iter[450/786] Loss: 1.061 CE: 1.056 Tri: 0.005 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline.log:808:Epoch[51] Iter[500/786] Loss: 1.061 CE: 1.056 Tri: 0.005 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline.log:809:Epoch[51] Iter[550/786] Loss: 1.060 CE: 1.055 Tri: 0.005 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline.log:810:Epoch[51] Iter[600/786] Loss: 1.059 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_baseline.log:811:Epoch[51] Iter[650/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.997 LR: 3.34e-06
logs/agreidv2_baseline.log:812:Epoch[51] Iter[700/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.997 LR: 3.34e-06
logs/agreidv2_baseline.log:813:Epoch[51] done in 188.4s  Loss=1.055 Acc=0.997
logs/agreidv2_baseline.log:814:Epoch[52] Iter[50/786] Loss: 1.068 CE: 1.061 Tri: 0.007 Acc: 0.995 LR: 2.72e-06
logs/agreidv2_baseline.log:815:Epoch[52] Iter[100/786] Loss: 1.067 CE: 1.060 Tri: 0.007 Acc: 0.995 LR: 2.72e-06
logs/agreidv2_baseline.log:816:Epoch[52] Iter[150/786] Loss: 1.066 CE: 1.058 Tri: 0.008 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline.log:817:Epoch[52] Iter[200/786] Loss: 1.065 CE: 1.057 Tri: 0.007 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_baseline.log:818:Epoch[52] Iter[250/786] Loss: 1.063 CE: 1.056 Tri: 0.007 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_baseline.log:819:Epoch[52] Iter[300/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_baseline.log:820:Epoch[52] Iter[350/786] Loss: 1.060 CE: 1.055 Tri: 0.006 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_baseline.log:821:Epoch[52] Iter[400/786] Loss: 1.060 CE: 1.055 Tri: 0.006 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_baseline.log:822:Epoch[52] Iter[450/786] Loss: 1.060 CE: 1.054 Tri: 0.006 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_baseline.log:823:Epoch[52] Iter[500/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_baseline.log:824:Epoch[52] Iter[550/786] Loss: 1.058 CE: 1.053 Tri: 0.006 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_baseline.log:825:Epoch[52] Iter[600/786] Loss: 1.058 CE: 1.052 Tri: 0.006 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_baseline.log:826:Epoch[52] Iter[650/786] Loss: 1.057 CE: 1.051 Tri: 0.006 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_baseline.log:827:Epoch[52] Iter[700/786] Loss: 1.056 CE: 1.051 Tri: 0.006 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_baseline.log:828:Epoch[52] done in 187.8s  Loss=1.055 Acc=0.997
logs/agreidv2_baseline.log:829:Epoch[53] Iter[50/786] Loss: 1.065 CE: 1.059 Tri: 0.006 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline.log:830:Epoch[53] Iter[100/786] Loss: 1.061 CE: 1.056 Tri: 0.005 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline.log:831:Epoch[53] Iter[150/786] Loss: 1.063 CE: 1.058 Tri: 0.006 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline.log:832:Epoch[53] Iter[200/786] Loss: 1.062 CE: 1.057 Tri: 0.006 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline.log:833:Epoch[53] Iter[250/786] Loss: 1.061 CE: 1.056 Tri: 0.005 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline.log:834:Epoch[53] Iter[300/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_baseline.log:835:Epoch[53] Iter[350/786] Loss: 1.060 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline.log:836:Epoch[53] Iter[400/786] Loss: 1.059 CE: 1.054 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline.log:837:Epoch[53] Iter[450/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline.log:838:Epoch[53] Iter[500/786] Loss: 1.058 CE: 1.052 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline.log:839:Epoch[53] Iter[550/786] Loss: 1.058 CE: 1.052 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_baseline.log:840:Epoch[53] Iter[600/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 2.16e-06
logs/agreidv2_baseline.log:841:Epoch[53] Iter[650/786] Loss: 1.056 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 2.16e-06
logs/agreidv2_baseline.log:842:Epoch[53] Iter[700/786] Loss: 1.054 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 2.16e-06
logs/agreidv2_baseline.log:843:Epoch[53] done in 190.0s  Loss=1.053 Acc=0.997
logs/agreidv2_baseline.log:844:Epoch[54] Iter[50/786] Loss: 1.071 CE: 1.063 Tri: 0.008 Acc: 0.993 LR: 1.67e-06
logs/agreidv2_baseline.log:845:Epoch[54] Iter[100/786] Loss: 1.068 CE: 1.059 Tri: 0.008 Acc: 0.995 LR: 1.67e-06
logs/agreidv2_baseline.log:846:Epoch[54] Iter[150/786] Loss: 1.067 CE: 1.058 Tri: 0.009 Acc: 0.995 LR: 1.67e-06
logs/agreidv2_baseline.log:847:Epoch[54] Iter[200/786] Loss: 1.064 CE: 1.057 Tri: 0.007 Acc: 0.995 LR: 1.67e-06
logs/agreidv2_baseline.log:848:Epoch[54] Iter[250/786] Loss: 1.063 CE: 1.056 Tri: 0.007 Acc: 0.995 LR: 1.67e-06
logs/agreidv2_baseline.log:849:Epoch[54] Iter[300/786] Loss: 1.061 CE: 1.055 Tri: 0.007 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline.log:850:Epoch[54] Iter[350/786] Loss: 1.060 CE: 1.054 Tri: 0.006 Acc: 0.995 LR: 1.67e-06
logs/agreidv2_baseline.log:851:Epoch[54] Iter[400/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline.log:852:Epoch[54] Iter[450/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline.log:853:Epoch[54] Iter[500/786] Loss: 1.058 CE: 1.052 Tri: 0.006 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline.log:854:Epoch[54] Iter[550/786] Loss: 1.057 CE: 1.051 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline.log:855:Epoch[54] Iter[600/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_baseline.log:856:Epoch[54] Iter[650/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_baseline.log:857:Epoch[54] Iter[700/786] Loss: 1.053 CE: 1.048 Tri: 0.005 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_baseline.log:858:Epoch[54] done in 190.4s  Loss=1.052 Acc=0.997
logs/agreidv2_baseline.log:859:Epoch[55] Iter[50/786] Loss: 1.060 CE: 1.056 Tri: 0.005 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline.log:860:Epoch[55] Iter[100/786] Loss: 1.063 CE: 1.057 Tri: 0.006 Acc: 0.995 LR: 1.23e-06
logs/agreidv2_baseline.log:861:Epoch[55] Iter[150/786] Loss: 1.060 CE: 1.055 Tri: 0.005 Acc: 0.995 LR: 1.23e-06
logs/agreidv2_baseline.log:862:Epoch[55] Iter[200/786] Loss: 1.058 CE: 1.054 Tri: 0.004 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline.log:863:Epoch[55] Iter[250/786] Loss: 1.058 CE: 1.054 Tri: 0.004 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline.log:864:Epoch[55] Iter[300/786] Loss: 1.056 CE: 1.053 Tri: 0.003 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline.log:865:Epoch[55] Iter[350/786] Loss: 1.056 CE: 1.052 Tri: 0.003 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline.log:866:Epoch[55] Iter[400/786] Loss: 1.055 CE: 1.052 Tri: 0.003 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline.log:867:Epoch[55] Iter[450/786] Loss: 1.054 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_baseline.log:868:Epoch[55] Iter[500/786] Loss: 1.053 CE: 1.050 Tri: 0.003 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline.log:869:Epoch[55] Iter[550/786] Loss: 1.052 CE: 1.049 Tri: 0.003 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline.log:870:Epoch[55] Iter[600/786] Loss: 1.052 CE: 1.049 Tri: 0.003 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline.log:871:Epoch[55] Iter[650/786] Loss: 1.050 CE: 1.047 Tri: 0.003 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline.log:872:Epoch[55] Iter[700/786] Loss: 1.050 CE: 1.047 Tri: 0.003 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_baseline.log:873:Epoch[55] done in 191.9s  Loss=1.048 Acc=0.997
logs/agreidv2_baseline.log:874:Epoch[56] Iter[50/786] Loss: 1.054 CE: 1.052 Tri: 0.002 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline.log:875:Epoch[56] Iter[100/786] Loss: 1.058 CE: 1.054 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline.log:876:Epoch[56] Iter[150/786] Loss: 1.059 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline.log:877:Epoch[56] Iter[200/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline.log:878:Epoch[56] Iter[250/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline.log:879:Epoch[56] Iter[300/786] Loss: 1.056 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_baseline.log:880:Epoch[56] Iter[350/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline.log:881:Epoch[56] Iter[400/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline.log:882:Epoch[56] Iter[450/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline.log:883:Epoch[56] Iter[500/786] Loss: 1.052 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline.log:884:Epoch[56] Iter[550/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline.log:885:Epoch[56] Iter[600/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline.log:886:Epoch[56] Iter[650/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline.log:887:Epoch[56] Iter[700/786] Loss: 1.050 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_baseline.log:888:Epoch[56] done in 189.7s  Loss=1.049 Acc=0.997
logs/agreidv2_baseline.log:889:Epoch[57] Iter[50/786] Loss: 1.074 CE: 1.055 Tri: 0.019 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline.log:890:Epoch[57] Iter[100/786] Loss: 1.066 CE: 1.054 Tri: 0.012 Acc: 0.995 LR: 5.50e-07
logs/agreidv2_baseline.log:891:Epoch[57] Iter[150/786] Loss: 1.064 CE: 1.054 Tri: 0.010 Acc: 0.995 LR: 5.50e-07
logs/agreidv2_baseline.log:892:Epoch[57] Iter[200/786] Loss: 1.062 CE: 1.053 Tri: 0.010 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline.log:893:Epoch[57] Iter[250/786] Loss: 1.061 CE: 1.052 Tri: 0.009 Acc: 0.995 LR: 5.50e-07
logs/agreidv2_baseline.log:894:Epoch[57] Iter[300/786] Loss: 1.059 CE: 1.051 Tri: 0.008 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline.log:895:Epoch[57] Iter[350/786] Loss: 1.058 CE: 1.051 Tri: 0.007 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline.log:896:Epoch[57] Iter[400/786] Loss: 1.057 CE: 1.050 Tri: 0.006 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline.log:897:Epoch[57] Iter[450/786] Loss: 1.055 CE: 1.049 Tri: 0.006 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_baseline.log:898:Epoch[57] Iter[500/786] Loss: 1.054 CE: 1.048 Tri: 0.006 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_baseline.log:899:Epoch[57] Iter[550/786] Loss: 1.054 CE: 1.048 Tri: 0.006 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_baseline.log:900:Epoch[57] Iter[600/786] Loss: 1.052 CE: 1.047 Tri: 0.005 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_baseline.log:901:Epoch[57] Iter[650/786] Loss: 1.052 CE: 1.046 Tri: 0.005 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_baseline.log:902:Epoch[57] Iter[700/786] Loss: 1.050 CE: 1.045 Tri: 0.005 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_baseline.log:903:Epoch[57] done in 190.9s  Loss=1.049 Acc=0.997
logs/agreidv2_baseline.log:904:Epoch[58] Iter[50/786] Loss: 1.058 CE: 1.055 Tri: 0.003 Acc: 0.995 LR: 3.10e-07
logs/agreidv2_baseline.log:905:Epoch[58] Iter[100/786] Loss: 1.055 CE: 1.053 Tri: 0.002 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline.log:906:Epoch[58] Iter[150/786] Loss: 1.055 CE: 1.052 Tri: 0.002 Acc: 0.997 LR: 3.10e-07
logs/agreidv2_baseline.log:907:Epoch[58] Iter[200/786] Loss: 1.055 CE: 1.052 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline.log:908:Epoch[58] Iter[250/786] Loss: 1.055 CE: 1.052 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline.log:909:Epoch[58] Iter[300/786] Loss: 1.054 CE: 1.052 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline.log:910:Epoch[58] Iter[350/786] Loss: 1.054 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline.log:911:Epoch[58] Iter[400/786] Loss: 1.054 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline.log:912:Epoch[58] Iter[450/786] Loss: 1.053 CE: 1.050 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline.log:913:Epoch[58] Iter[500/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_baseline.log:914:Epoch[58] Iter[550/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
logs/agreidv2_baseline.log:915:Epoch[58] Iter[600/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
logs/agreidv2_baseline.log:916:Epoch[58] Iter[650/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
logs/agreidv2_baseline.log:917:Epoch[58] Iter[700/786] Loss: 1.050 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
logs/agreidv2_baseline.log:918:Epoch[58] done in 189.3s  Loss=1.049 Acc=0.997
logs/agreidv2_baseline.log:919:Epoch[59] Iter[50/786] Loss: 1.066 CE: 1.056 Tri: 0.009 Acc: 0.994 LR: 1.38e-07
logs/agreidv2_baseline.log:920:Epoch[59] Iter[100/786] Loss: 1.060 CE: 1.054 Tri: 0.006 Acc: 0.995 LR: 1.38e-07
logs/agreidv2_baseline.log:921:Epoch[59] Iter[150/786] Loss: 1.058 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 1.38e-07
logs/agreidv2_baseline.log:922:Epoch[59] Iter[200/786] Loss: 1.058 CE: 1.052 Tri: 0.005 Acc: 0.996 LR: 1.38e-07
logs/agreidv2_baseline.log:923:Epoch[59] Iter[250/786] Loss: 1.056 CE: 1.052 Tri: 0.005 Acc: 0.996 LR: 1.38e-07
logs/agreidv2_baseline.log:924:Epoch[59] Iter[300/786] Loss: 1.055 CE: 1.051 Tri: 0.005 Acc: 0.996 LR: 1.38e-07
logs/agreidv2_baseline.log:925:Epoch[59] Iter[350/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline.log:926:Epoch[59] Iter[400/786] Loss: 1.054 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline.log:927:Epoch[59] Iter[450/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline.log:928:Epoch[59] Iter[500/786] Loss: 1.053 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline.log:929:Epoch[59] Iter[550/786] Loss: 1.052 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline.log:930:Epoch[59] Iter[600/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline.log:931:Epoch[59] Iter[650/786] Loss: 1.050 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline.log:932:Epoch[59] Iter[700/786] Loss: 1.049 CE: 1.045 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_baseline.log:933:Epoch[59] done in 191.8s  Loss=1.048 Acc=0.998
logs/agreidv2_baseline.log:934:Epoch[60] Iter[50/786] Loss: 1.064 CE: 1.054 Tri: 0.011 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_baseline.log:935:Epoch[60] Iter[100/786] Loss: 1.060 CE: 1.052 Tri: 0.008 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:936:Epoch[60] Iter[150/786] Loss: 1.057 CE: 1.050 Tri: 0.006 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:937:Epoch[60] Iter[200/786] Loss: 1.057 CE: 1.050 Tri: 0.007 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:938:Epoch[60] Iter[250/786] Loss: 1.056 CE: 1.050 Tri: 0.006 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:939:Epoch[60] Iter[300/786] Loss: 1.055 CE: 1.050 Tri: 0.006 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:940:Epoch[60] Iter[350/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:941:Epoch[60] Iter[400/786] Loss: 1.054 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:942:Epoch[60] Iter[450/786] Loss: 1.053 CE: 1.048 Tri: 0.005 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:943:Epoch[60] Iter[500/786] Loss: 1.052 CE: 1.048 Tri: 0.005 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:944:Epoch[60] Iter[550/786] Loss: 1.052 CE: 1.047 Tri: 0.005 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:945:Epoch[60] Iter[600/786] Loss: 1.051 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:946:Epoch[60] Iter[650/786] Loss: 1.050 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:947:Epoch[60] Iter[700/786] Loss: 1.049 CE: 1.045 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_baseline.log:948:Epoch[60] done in 190.7s  Loss=1.047 Acc=0.998
logs/agreidv2_baseline.log:950:    [A->G] mAP=79.72  R1=86.42  R5=92.28  mINP=56.04
logs/agreidv2_baseline.log:951:    [G->A] mAP=80.04  R1=87.80  R5=92.16  mINP=46.62
logs/agreidv2_baseline.log:952:    [mean] mAP=79.88  R1=87.11
logs/agreidv2_baseline.log:953:    * new best mean mAP=79.88 (epoch 60) saved
logs/agreidv2_baseline.log:955:Training finished. Best mean A<->G mAP=79.88 @ epoch 60
logs/agreidv2_baseline.log:956:Checkpoints in /root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_baseline
logs/agreidv2_airl_iso.log:2:CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
logs/agreidv2_airl_iso.log:7:  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
logs/agreidv2_airl_iso.log:8:  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
logs/agreidv2_airl_iso.log:9:  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_airl_iso.log:11:  out_dir=/root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_airl_iso
logs/agreidv2_airl_iso.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_airl_iso.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_airl_iso.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_airl_iso.log:25:  [AIRL-ISO] iso_stage=3: rec late stage (14185392 params, 28 tensors) + rec BNNeck head (620544 params, 2 tensors) in optimizer [rec late stage @ Swin LR 3.50e-05, rec BNNeck @ full LR 3.50e-04]; degradation-consistency grad isolated from shared trunk (detached degraded pass at stage-3 input); trunk_recce=0 (clean ID-CE + consistency BOTH detached = original full-isolation); eval soft-fusion cos=w*cos_rec+(1-w)*cos_full w=0.25
logs/agreidv2_airl_iso.log:26:Epoch[1] Iter[50/786] Loss: 44.803 CE: 6.693 Tri: 31.417 CE_rec: 6.693 AIRL_rec: 0.0002 Acc: 0.002 LR: 3.50e-07
logs/agreidv2_airl_iso.log:27:Epoch[1] Iter[100/786] Loss: 38.578 CE: 6.692 Tri: 25.194 CE_rec: 6.692 AIRL_rec: 0.0002 Acc: 0.002 LR: 3.50e-07
logs/agreidv2_airl_iso.log:28:Epoch[1] Iter[150/786] Loss: 35.377 CE: 6.690 Tri: 21.996 CE_rec: 6.690 AIRL_rec: 0.0002 Acc: 0.002 LR: 3.50e-07
logs/agreidv2_airl_iso.log:29:Epoch[1] Iter[200/786] Loss: 33.454 CE: 6.689 Tri: 20.078 CE_rec: 6.688 AIRL_rec: 0.0002 Acc: 0.003 LR: 3.50e-07
logs/agreidv2_airl_iso.log:30:Epoch[1] Iter[250/786] Loss: 31.992 CE: 6.687 Tri: 18.620 CE_rec: 6.686 AIRL_rec: 0.0002 Acc: 0.005 LR: 3.50e-07
logs/agreidv2_airl_iso.log:31:Epoch[1] Iter[300/786] Loss: 30.805 CE: 6.684 Tri: 17.438 CE_rec: 6.683 AIRL_rec: 0.0002 Acc: 0.007 LR: 3.50e-07
logs/agreidv2_airl_iso.log:32:Epoch[1] Iter[350/786] Loss: 29.875 CE: 6.682 Tri: 16.512 CE_rec: 6.681 AIRL_rec: 0.0002 Acc: 0.011 LR: 3.50e-07
logs/agreidv2_airl_iso.log:33:Epoch[1] Iter[400/786] Loss: 29.089 CE: 6.680 Tri: 15.730 CE_rec: 6.679 AIRL_rec: 0.0002 Acc: 0.015 LR: 3.50e-07
logs/agreidv2_airl_iso.log:34:Epoch[1] Iter[450/786] Loss: 28.341 CE: 6.677 Tri: 14.988 CE_rec: 6.676 AIRL_rec: 0.0002 Acc: 0.021 LR: 3.50e-07
logs/agreidv2_airl_iso.log:35:Epoch[1] Iter[500/786] Loss: 27.685 CE: 6.675 Tri: 14.337 CE_rec: 6.673 AIRL_rec: 0.0002 Acc: 0.028 LR: 3.50e-07
logs/agreidv2_airl_iso.log:36:Epoch[1] Iter[550/786] Loss: 27.062 CE: 6.672 Tri: 13.720 CE_rec: 6.671 AIRL_rec: 0.0002 Acc: 0.038 LR: 3.50e-07
logs/agreidv2_airl_iso.log:37:Epoch[1] Iter[600/786] Loss: 26.542 CE: 6.669 Tri: 13.206 CE_rec: 6.668 AIRL_rec: 0.0002 Acc: 0.051 LR: 3.50e-07
logs/agreidv2_airl_iso.log:38:Epoch[1] Iter[650/786] Loss: 26.088 CE: 6.666 Tri: 12.758 CE_rec: 6.664 AIRL_rec: 0.0002 Acc: 0.067 LR: 3.50e-07
logs/agreidv2_airl_iso.log:39:Epoch[1] Iter[700/786] Loss: 25.652 CE: 6.662 Tri: 12.330 CE_rec: 6.660 AIRL_rec: 0.0002 Acc: 0.089 LR: 3.50e-07
logs/agreidv2_airl_iso.log:40:Epoch[1] done in 234.6s  Loss=25.287 Acc=0.117 AIRL-ISO[lam_eff=0.100 ce_rec=6.655 consistency=0.0002 deg_scale_mean=0.625 n_ground=28771]
logs/agreidv2_airl_iso.log:41:Epoch[2] Iter[50/786] Loss: 18.702 CE: 6.602 Tri: 5.508 CE_rec: 6.592 AIRL_rec: 0.0003 Acc: 0.139 LR: 3.82e-06
logs/agreidv2_airl_iso.log:42:Epoch[2] Iter[100/786] Loss: 17.803 CE: 6.570 Tri: 4.683 CE_rec: 6.550 AIRL_rec: 0.0004 Acc: 0.184 LR: 3.82e-06
logs/agreidv2_airl_iso.log:43:Epoch[2] Iter[150/786] Loss: 17.348 CE: 6.539 Tri: 4.304 CE_rec: 6.505 AIRL_rec: 0.0006 Acc: 0.212 LR: 3.82e-06
logs/agreidv2_airl_iso.log:44:Epoch[2] Iter[200/786] Loss: 16.933 CE: 6.505 Tri: 3.972 CE_rec: 6.455 AIRL_rec: 0.0010 Acc: 0.241 LR: 3.82e-06
logs/agreidv2_airl_iso.log:45:Epoch[2] Iter[250/786] Loss: 16.605 CE: 6.472 Tri: 3.728 CE_rec: 6.405 AIRL_rec: 0.0014 Acc: 0.269 LR: 3.82e-06
logs/agreidv2_airl_iso.log:46:Epoch[2] Iter[300/786] Loss: 16.324 CE: 6.439 Tri: 3.531 CE_rec: 6.354 AIRL_rec: 0.0019 Acc: 0.292 LR: 3.82e-06
logs/agreidv2_airl_iso.log:47:Epoch[2] Iter[350/786] Loss: 16.086 CE: 6.406 Tri: 3.377 CE_rec: 6.303 AIRL_rec: 0.0025 Acc: 0.316 LR: 3.82e-06
logs/agreidv2_airl_iso.log:48:Epoch[2] Iter[400/786] Loss: 15.862 CE: 6.371 Tri: 3.241 CE_rec: 6.249 AIRL_rec: 0.0032 Acc: 0.336 LR: 3.82e-06
logs/agreidv2_airl_iso.log:49:Epoch[2] Iter[450/786] Loss: 15.651 CE: 6.337 Tri: 3.116 CE_rec: 6.197 AIRL_rec: 0.0039 Acc: 0.355 LR: 3.82e-06
logs/agreidv2_airl_iso.log:50:Epoch[2] Iter[500/786] Loss: 15.440 CE: 6.302 Tri: 2.995 CE_rec: 6.142 AIRL_rec: 0.0047 Acc: 0.374 LR: 3.82e-06
logs/agreidv2_airl_iso.log:51:Epoch[2] Iter[550/786] Loss: 15.238 CE: 6.264 Tri: 2.888 CE_rec: 6.086 AIRL_rec: 0.0054 Acc: 0.396 LR: 3.82e-06
logs/agreidv2_airl_iso.log:52:Epoch[2] Iter[600/786] Loss: 15.056 CE: 6.225 Tri: 2.802 CE_rec: 6.028 AIRL_rec: 0.0063 Acc: 0.417 LR: 3.82e-06
logs/agreidv2_airl_iso.log:53:Epoch[2] Iter[650/786] Loss: 14.856 CE: 6.180 Tri: 2.714 CE_rec: 5.960 AIRL_rec: 0.0071 Acc: 0.439 LR: 3.82e-06
logs/agreidv2_airl_iso.log:54:Epoch[2] Iter[700/786] Loss: 14.660 CE: 6.129 Tri: 2.643 CE_rec: 5.886 AIRL_rec: 0.0079 Acc: 0.459 LR: 3.82e-06
logs/agreidv2_airl_iso.log:55:Epoch[2] done in 235.3s  Loss=14.438 Acc=0.483 AIRL-ISO[lam_eff=0.200 ce_rec=5.788 consistency=0.0088 deg_scale_mean=0.624 n_ground=28774]
logs/agreidv2_airl_iso.log:56:Epoch[3] Iter[50/786] Loss: 12.261 CE: 5.580 Tri: 1.358 CE_rec: 5.318 AIRL_rec: 0.0198 Acc: 0.323 LR: 7.28e-06
logs/agreidv2_airl_iso.log:57:Epoch[3] Iter[100/786] Loss: 12.102 CE: 5.512 Tri: 1.355 CE_rec: 5.229 AIRL_rec: 0.0209 Acc: 0.347 LR: 7.28e-06
logs/agreidv2_airl_iso.log:58:Epoch[3] Iter[150/786] Loss: 11.840 CE: 5.423 Tri: 1.288 CE_rec: 5.123 AIRL_rec: 0.0221 Acc: 0.395 LR: 7.28e-06
logs/agreidv2_airl_iso.log:59:Epoch[3] Iter[200/786] Loss: 11.606 CE: 5.331 Tri: 1.255 CE_rec: 5.014 AIRL_rec: 0.0236 Acc: 0.431 LR: 7.28e-06
logs/agreidv2_airl_iso.log:60:Epoch[3] Iter[250/786] Loss: 11.366 CE: 5.244 Tri: 1.200 CE_rec: 4.916 AIRL_rec: 0.0245 Acc: 0.460 LR: 7.28e-06
logs/agreidv2_airl_iso.log:61:Epoch[3] Iter[300/786] Loss: 11.194 CE: 5.165 Tri: 1.192 CE_rec: 4.829 AIRL_rec: 0.0261 Acc: 0.484 LR: 7.28e-06
logs/agreidv2_airl_iso.log:62:Epoch[3] Iter[350/786] Loss: 10.996 CE: 5.088 Tri: 1.158 CE_rec: 4.743 AIRL_rec: 0.0272 Acc: 0.505 LR: 7.28e-06
logs/agreidv2_airl_iso.log:63:Epoch[3] Iter[400/786] Loss: 10.818 CE: 5.008 Tri: 1.141 CE_rec: 4.659 AIRL_rec: 0.0285 Acc: 0.524 LR: 7.28e-06
logs/agreidv2_airl_iso.log:64:Epoch[3] Iter[450/786] Loss: 10.614 CE: 4.923 Tri: 1.113 CE_rec: 4.569 AIRL_rec: 0.0298 Acc: 0.545 LR: 7.28e-06
logs/agreidv2_airl_iso.log:65:Epoch[3] Iter[500/786] Loss: 10.422 CE: 4.836 Tri: 1.098 CE_rec: 4.479 AIRL_rec: 0.0311 Acc: 0.562 LR: 7.28e-06
logs/agreidv2_airl_iso.log:66:Epoch[3] Iter[550/786] Loss: 10.215 CE: 4.741 Tri: 1.083 CE_rec: 4.382 AIRL_rec: 0.0326 Acc: 0.581 LR: 7.28e-06
logs/agreidv2_airl_iso.log:67:Epoch[3] Iter[600/786] Loss: 9.998 CE: 4.643 Tri: 1.063 CE_rec: 4.282 AIRL_rec: 0.0339 Acc: 0.598 LR: 7.28e-06
logs/agreidv2_airl_iso.log:68:Epoch[3] Iter[650/786] Loss: 9.773 CE: 4.540 Tri: 1.045 CE_rec: 4.177 AIRL_rec: 0.0352 Acc: 0.614 LR: 7.28e-06
logs/agreidv2_airl_iso.log:69:Epoch[3] Iter[700/786] Loss: 9.537 CE: 4.428 Tri: 1.035 CE_rec: 4.064 AIRL_rec: 0.0365 Acc: 0.631 LR: 7.28e-06
logs/agreidv2_airl_iso.log:70:Epoch[3] done in 237.1s  Loss=9.276 Acc=0.648 AIRL-ISO[lam_eff=0.300 ce_rec=3.941 consistency=0.0381 deg_scale_mean=0.627 n_ground=28670]
logs/agreidv2_airl_iso.log:71:Epoch[4] Iter[50/786] Loss: 8.192 CE: 3.776 Tri: 0.738 CE_rec: 3.654 AIRL_rec: 0.0601 Acc: 0.469 LR: 1.07e-05
logs/agreidv2_airl_iso.log:72:Epoch[4] Iter[100/786] Loss: 7.902 CE: 3.658 Tri: 0.701 CE_rec: 3.517 AIRL_rec: 0.0621 Acc: 0.509 LR: 1.07e-05
logs/agreidv2_airl_iso.log:73:Epoch[4] Iter[150/786] Loss: 7.704 CE: 3.566 Tri: 0.700 CE_rec: 3.413 AIRL_rec: 0.0634 Acc: 0.549 LR: 1.07e-05
logs/agreidv2_airl_iso.log:74:Epoch[4] Iter[200/786] Loss: 7.473 CE: 3.458 Tri: 0.691 CE_rec: 3.298 AIRL_rec: 0.0650 Acc: 0.588 LR: 1.07e-05
logs/agreidv2_airl_iso.log:75:Epoch[4] Iter[250/786] Loss: 7.236 CE: 3.351 Tri: 0.667 CE_rec: 3.190 AIRL_rec: 0.0674 Acc: 0.623 LR: 1.07e-05
logs/agreidv2_airl_iso.log:76:Epoch[4] Iter[300/786] Loss: 7.064 CE: 3.266 Tri: 0.663 CE_rec: 3.107 AIRL_rec: 0.0695 Acc: 0.646 LR: 1.07e-05
logs/agreidv2_airl_iso.log:77:Epoch[4] Iter[350/786] Loss: 6.875 CE: 3.176 Tri: 0.652 CE_rec: 3.019 AIRL_rec: 0.0715 Acc: 0.671 LR: 1.07e-05
logs/agreidv2_airl_iso.log:78:Epoch[4] Iter[400/786] Loss: 6.700 CE: 3.091 Tri: 0.645 CE_rec: 2.935 AIRL_rec: 0.0738 Acc: 0.693 LR: 1.07e-05
logs/agreidv2_airl_iso.log:79:Epoch[4] Iter[450/786] Loss: 6.534 CE: 3.009 Tri: 0.639 CE_rec: 2.856 AIRL_rec: 0.0759 Acc: 0.711 LR: 1.07e-05
logs/agreidv2_airl_iso.log:80:Epoch[4] Iter[500/786] Loss: 6.383 CE: 2.934 Tri: 0.635 CE_rec: 2.783 AIRL_rec: 0.0779 Acc: 0.727 LR: 1.07e-05
logs/agreidv2_airl_iso.log:81:Epoch[4] Iter[550/786] Loss: 6.219 CE: 2.857 Tri: 0.621 CE_rec: 2.709 AIRL_rec: 0.0799 Acc: 0.741 LR: 1.07e-05
logs/agreidv2_airl_iso.log:82:Epoch[4] Iter[600/786] Loss: 6.071 CE: 2.786 Tri: 0.612 CE_rec: 2.641 AIRL_rec: 0.0818 Acc: 0.754 LR: 1.07e-05
logs/agreidv2_airl_iso.log:83:Epoch[4] Iter[650/786] Loss: 5.924 CE: 2.715 Tri: 0.603 CE_rec: 2.573 AIRL_rec: 0.0837 Acc: 0.767 LR: 1.07e-05
logs/agreidv2_airl_iso.log:84:Epoch[4] Iter[700/786] Loss: 5.775 CE: 2.642 Tri: 0.595 CE_rec: 2.503 AIRL_rec: 0.0850 Acc: 0.778 LR: 1.07e-05
logs/agreidv2_airl_iso.log:85:Epoch[4] done in 240.9s  Loss=5.621 Acc=0.790 AIRL-ISO[lam_eff=0.400 ce_rec=2.430 consistency=0.0872 deg_scale_mean=0.625 n_ground=28860]
logs/agreidv2_airl_iso.log:86:Epoch[5] Iter[50/786] Loss: 5.403 CE: 2.443 Tri: 0.512 CE_rec: 2.390 AIRL_rec: 0.1154 Acc: 0.733 LR: 1.42e-05
logs/agreidv2_airl_iso.log:87:Epoch[5] Iter[100/786] Loss: 5.220 CE: 2.374 Tri: 0.489 CE_rec: 2.300 AIRL_rec: 0.1133 Acc: 0.758 LR: 1.42e-05
logs/agreidv2_airl_iso.log:88:Epoch[5] Iter[150/786] Loss: 5.024 CE: 2.283 Tri: 0.476 CE_rec: 2.208 AIRL_rec: 0.1151 Acc: 0.788 LR: 1.42e-05
logs/agreidv2_airl_iso.log:89:Epoch[5] Iter[200/786] Loss: 4.906 CE: 2.225 Tri: 0.471 CE_rec: 2.152 AIRL_rec: 0.1181 Acc: 0.804 LR: 1.42e-05
logs/agreidv2_airl_iso.log:90:Epoch[5] Iter[250/786] Loss: 4.763 CE: 2.160 Tri: 0.454 CE_rec: 2.089 AIRL_rec: 0.1200 Acc: 0.823 LR: 1.42e-05
logs/agreidv2_airl_iso.log:91:Epoch[5] Iter[300/786] Loss: 4.636 CE: 2.100 Tri: 0.445 CE_rec: 2.029 AIRL_rec: 0.1220 Acc: 0.837 LR: 1.42e-05
logs/agreidv2_airl_iso.log:92:Epoch[5] Iter[350/786] Loss: 4.532 CE: 2.053 Tri: 0.435 CE_rec: 1.983 AIRL_rec: 0.1236 Acc: 0.848 LR: 1.42e-05
logs/agreidv2_airl_iso.log:93:Epoch[5] Iter[400/786] Loss: 4.466 CE: 2.018 Tri: 0.437 CE_rec: 1.948 AIRL_rec: 0.1255 Acc: 0.856 LR: 1.42e-05
logs/agreidv2_airl_iso.log:94:Epoch[5] Iter[450/786] Loss: 4.387 CE: 1.983 Tri: 0.425 CE_rec: 1.915 AIRL_rec: 0.1274 Acc: 0.863 LR: 1.42e-05
logs/agreidv2_airl_iso.log:95:Epoch[5] Iter[500/786] Loss: 4.306 CE: 1.948 Tri: 0.414 CE_rec: 1.879 AIRL_rec: 0.1283 Acc: 0.870 LR: 1.42e-05
logs/agreidv2_airl_iso.log:96:Epoch[5] Iter[550/786] Loss: 4.244 CE: 1.918 Tri: 0.412 CE_rec: 1.850 AIRL_rec: 0.1297 Acc: 0.876 LR: 1.42e-05
logs/agreidv2_airl_iso.log:97:Epoch[5] Iter[600/786] Loss: 4.185 CE: 1.890 Tri: 0.408 CE_rec: 1.821 AIRL_rec: 0.1313 Acc: 0.881 LR: 1.42e-05
logs/agreidv2_airl_iso.log:98:Epoch[5] Iter[650/786] Loss: 4.121 CE: 1.863 Tri: 0.400 CE_rec: 1.792 AIRL_rec: 0.1324 Acc: 0.886 LR: 1.42e-05
logs/agreidv2_airl_iso.log:99:Epoch[5] Iter[700/786] Loss: 4.057 CE: 1.836 Tri: 0.392 CE_rec: 1.763 AIRL_rec: 0.1335 Acc: 0.890 LR: 1.42e-05
logs/agreidv2_airl_iso.log:100:Epoch[5] done in 240.6s  Loss=3.997 Acc=0.895 AIRL-ISO[lam_eff=0.500 ce_rec=1.733 consistency=0.1346 deg_scale_mean=0.625 n_ground=28856]
logs/agreidv2_airl_iso.log:101:Epoch[6] Iter[50/786] Loss: 4.065 CE: 1.844 Tri: 0.332 CE_rec: 1.816 AIRL_rec: 0.1462 Acc: 0.861 LR: 1.77e-05
logs/agreidv2_airl_iso.log:102:Epoch[6] Iter[100/786] Loss: 3.979 CE: 1.808 Tri: 0.333 CE_rec: 1.766 AIRL_rec: 0.1447 Acc: 0.874 LR: 1.77e-05
logs/agreidv2_airl_iso.log:103:Epoch[6] Iter[150/786] Loss: 3.914 CE: 1.771 Tri: 0.345 CE_rec: 1.724 AIRL_rec: 0.1472 Acc: 0.883 LR: 1.77e-05
logs/agreidv2_airl_iso.log:104:Epoch[6] Iter[200/786] Loss: 3.844 CE: 1.740 Tri: 0.337 CE_rec: 1.692 AIRL_rec: 0.1504 Acc: 0.892 LR: 1.77e-05
logs/agreidv2_airl_iso.log:105:Epoch[6] Iter[250/786] Loss: 3.780 CE: 1.713 Tri: 0.329 CE_rec: 1.662 AIRL_rec: 0.1523 Acc: 0.899 LR: 1.77e-05
logs/agreidv2_airl_iso.log:106:Epoch[6] Iter[300/786] Loss: 3.721 CE: 1.688 Tri: 0.320 CE_rec: 1.636 AIRL_rec: 0.1542 Acc: 0.905 LR: 1.77e-05
logs/agreidv2_airl_iso.log:107:Epoch[6] Iter[350/786] Loss: 3.687 CE: 1.672 Tri: 0.319 CE_rec: 1.617 AIRL_rec: 0.1562 Acc: 0.908 LR: 1.77e-05
logs/agreidv2_airl_iso.log:108:Epoch[6] Iter[400/786] Loss: 3.637 CE: 1.652 Tri: 0.311 CE_rec: 1.597 AIRL_rec: 0.1570 Acc: 0.912 LR: 1.77e-05
logs/agreidv2_airl_iso.log:109:Epoch[6] Iter[450/786] Loss: 3.605 CE: 1.637 Tri: 0.309 CE_rec: 1.580 AIRL_rec: 0.1578 Acc: 0.915 LR: 1.77e-05
logs/agreidv2_airl_iso.log:110:Epoch[6] Iter[500/786] Loss: 3.575 CE: 1.624 Tri: 0.307 CE_rec: 1.565 AIRL_rec: 0.1589 Acc: 0.918 LR: 1.77e-05
logs/agreidv2_airl_iso.log:111:Epoch[6] Iter[550/786] Loss: 3.574 CE: 1.617 Tri: 0.320 CE_rec: 1.557 AIRL_rec: 0.1610 Acc: 0.919 LR: 1.77e-05
logs/agreidv2_airl_iso.log:112:Epoch[6] Iter[600/786] Loss: 3.555 CE: 1.607 Tri: 0.322 CE_rec: 1.544 AIRL_rec: 0.1625 Acc: 0.922 LR: 1.77e-05
logs/agreidv2_airl_iso.log:113:Epoch[6] Iter[650/786] Loss: 3.535 CE: 1.596 Tri: 0.326 CE_rec: 1.532 AIRL_rec: 0.1636 Acc: 0.924 LR: 1.77e-05
logs/agreidv2_airl_iso.log:114:Epoch[6] Iter[700/786] Loss: 3.507 CE: 1.583 Tri: 0.326 CE_rec: 1.516 AIRL_rec: 0.1640 Acc: 0.927 LR: 1.77e-05
logs/agreidv2_airl_iso.log:115:Epoch[6] Iter[750/786] Loss: 3.465 CE: 1.565 Tri: 0.322 CE_rec: 1.496 AIRL_rec: 0.1647 Acc: 0.931 LR: 1.77e-05
logs/agreidv2_airl_iso.log:116:Epoch[6] done in 241.4s  Loss=3.465 Acc=0.931 AIRL-ISO[lam_eff=0.500 ce_rec=1.496 consistency=0.1647 deg_scale_mean=0.624 n_ground=28935]
logs/agreidv2_airl_iso.log:117:Epoch[7] Iter[50/786] Loss: 3.841 CE: 1.710 Tri: 0.365 CE_rec: 1.683 AIRL_rec: 0.1652 Acc: 0.892 LR: 2.11e-05
logs/agreidv2_airl_iso.log:118:Epoch[7] Iter[100/786] Loss: 3.664 CE: 1.644 Tri: 0.334 CE_rec: 1.601 AIRL_rec: 0.1690 Acc: 0.910 LR: 2.11e-05
logs/agreidv2_airl_iso.log:119:Epoch[7] Iter[150/786] Loss: 3.541 CE: 1.605 Tri: 0.298 CE_rec: 1.553 AIRL_rec: 0.1700 Acc: 0.920 LR: 2.11e-05
logs/agreidv2_airl_iso.log:120:Epoch[7] Iter[200/786] Loss: 3.476 CE: 1.582 Tri: 0.286 CE_rec: 1.523 AIRL_rec: 0.1707 Acc: 0.925 LR: 2.11e-05
logs/agreidv2_airl_iso.log:121:Epoch[7] Iter[250/786] Loss: 3.447 CE: 1.569 Tri: 0.284 CE_rec: 1.507 AIRL_rec: 0.1731 Acc: 0.928 LR: 2.11e-05
logs/agreidv2_airl_iso.log:122:Epoch[7] Iter[300/786] Loss: 3.435 CE: 1.561 Tri: 0.288 CE_rec: 1.498 AIRL_rec: 0.1750 Acc: 0.930 LR: 2.11e-05
logs/agreidv2_airl_iso.log:123:Epoch[7] Iter[350/786] Loss: 3.415 CE: 1.551 Tri: 0.290 CE_rec: 1.487 AIRL_rec: 0.1748 Acc: 0.932 LR: 2.11e-05
logs/agreidv2_airl_iso.log:124:Epoch[7] Iter[400/786] Loss: 3.393 CE: 1.542 Tri: 0.286 CE_rec: 1.476 AIRL_rec: 0.1754 Acc: 0.934 LR: 2.11e-05
logs/agreidv2_airl_iso.log:125:Epoch[7] Iter[450/786] Loss: 3.378 CE: 1.535 Tri: 0.287 CE_rec: 1.468 AIRL_rec: 0.1758 Acc: 0.936 LR: 2.11e-05
logs/agreidv2_airl_iso.log:126:Epoch[7] Iter[500/786] Loss: 3.351 CE: 1.524 Tri: 0.282 CE_rec: 1.456 AIRL_rec: 0.1762 Acc: 0.939 LR: 2.11e-05
logs/agreidv2_airl_iso.log:127:Epoch[7] Iter[550/786] Loss: 3.336 CE: 1.519 Tri: 0.280 CE_rec: 1.448 AIRL_rec: 0.1764 Acc: 0.940 LR: 2.11e-05
logs/agreidv2_airl_iso.log:128:Epoch[7] Iter[600/786] Loss: 3.317 CE: 1.511 Tri: 0.278 CE_rec: 1.440 AIRL_rec: 0.1758 Acc: 0.941 LR: 2.11e-05
logs/agreidv2_airl_iso.log:129:Epoch[7] Iter[650/786] Loss: 3.290 CE: 1.501 Tri: 0.273 CE_rec: 1.428 AIRL_rec: 0.1751 Acc: 0.943 LR: 2.11e-05
logs/agreidv2_airl_iso.log:130:Epoch[7] Iter[700/786] Loss: 3.265 CE: 1.491 Tri: 0.271 CE_rec: 1.416 AIRL_rec: 0.1739 Acc: 0.945 LR: 2.11e-05
logs/agreidv2_airl_iso.log:131:Epoch[7] done in 240.6s  Loss=3.234 Acc=0.947 AIRL-ISO[lam_eff=0.500 ce_rec=1.403 consistency=0.1735 deg_scale_mean=0.625 n_ground=28828]
logs/agreidv2_airl_iso.log:132:Epoch[8] Iter[50/786] Loss: 3.430 CE: 1.556 Tri: 0.257 CE_rec: 1.531 AIRL_rec: 0.1718 Acc: 0.921 LR: 2.46e-05
logs/agreidv2_airl_iso.log:133:Epoch[8] Iter[100/786] Loss: 3.332 CE: 1.530 Tri: 0.236 CE_rec: 1.480 AIRL_rec: 0.1727 Acc: 0.928 LR: 2.46e-05
logs/agreidv2_airl_iso.log:134:Epoch[8] Iter[150/786] Loss: 3.278 CE: 1.509 Tri: 0.230 CE_rec: 1.451 AIRL_rec: 0.1752 Acc: 0.935 LR: 2.46e-05
logs/agreidv2_airl_iso.log:135:Epoch[8] Iter[200/786] Loss: 3.242 CE: 1.497 Tri: 0.227 CE_rec: 1.430 AIRL_rec: 0.1780 Acc: 0.939 LR: 2.46e-05
logs/agreidv2_airl_iso.log:136:Epoch[8] Iter[250/786] Loss: 3.220 CE: 1.486 Tri: 0.227 CE_rec: 1.417 AIRL_rec: 0.1795 Acc: 0.942 LR: 2.46e-05
logs/agreidv2_airl_iso.log:137:Epoch[8] Iter[300/786] Loss: 3.208 CE: 1.481 Tri: 0.227 CE_rec: 1.410 AIRL_rec: 0.1801 Acc: 0.943 LR: 2.46e-05
logs/agreidv2_airl_iso.log:138:Epoch[8] Iter[350/786] Loss: 3.194 CE: 1.476 Tri: 0.225 CE_rec: 1.402 AIRL_rec: 0.1809 Acc: 0.944 LR: 2.46e-05
logs/agreidv2_airl_iso.log:139:Epoch[8] Iter[400/786] Loss: 3.182 CE: 1.469 Tri: 0.228 CE_rec: 1.394 AIRL_rec: 0.1817 Acc: 0.946 LR: 2.46e-05
logs/agreidv2_airl_iso.log:140:Epoch[8] Iter[450/786] Loss: 3.172 CE: 1.465 Tri: 0.228 CE_rec: 1.389 AIRL_rec: 0.1812 Acc: 0.947 LR: 2.46e-05
logs/agreidv2_airl_iso.log:141:Epoch[8] Iter[500/786] Loss: 3.158 CE: 1.458 Tri: 0.228 CE_rec: 1.382 AIRL_rec: 0.1807 Acc: 0.949 LR: 2.46e-05
logs/agreidv2_airl_iso.log:142:Epoch[8] Iter[550/786] Loss: 3.146 CE: 1.454 Tri: 0.225 CE_rec: 1.377 AIRL_rec: 0.1802 Acc: 0.950 LR: 2.46e-05
logs/agreidv2_airl_iso.log:143:Epoch[8] Iter[600/786] Loss: 3.127 CE: 1.446 Tri: 0.224 CE_rec: 1.368 AIRL_rec: 0.1795 Acc: 0.951 LR: 2.46e-05
logs/agreidv2_airl_iso.log:144:Epoch[8] Iter[650/786] Loss: 3.111 CE: 1.439 Tri: 0.222 CE_rec: 1.361 AIRL_rec: 0.1789 Acc: 0.953 LR: 2.46e-05
logs/agreidv2_airl_iso.log:145:Epoch[8] Iter[700/786] Loss: 3.091 CE: 1.431 Tri: 0.219 CE_rec: 1.352 AIRL_rec: 0.1779 Acc: 0.954 LR: 2.46e-05
logs/agreidv2_airl_iso.log:146:Epoch[8] done in 238.7s  Loss=3.067 Acc=0.956 AIRL-ISO[lam_eff=0.500 ce_rec=1.341 consistency=0.1775 deg_scale_mean=0.624 n_ground=28656]
logs/agreidv2_airl_iso.log:147:Epoch[9] Iter[50/786] Loss: 3.231 CE: 1.485 Tri: 0.211 CE_rec: 1.451 AIRL_rec: 0.1684 Acc: 0.934 LR: 2.81e-05
logs/agreidv2_airl_iso.log:148:Epoch[9] Iter[100/786] Loss: 3.202 CE: 1.476 Tri: 0.211 CE_rec: 1.427 AIRL_rec: 0.1764 Acc: 0.937 LR: 2.81e-05
logs/agreidv2_airl_iso.log:149:Epoch[9] Iter[150/786] Loss: 3.178 CE: 1.467 Tri: 0.213 CE_rec: 1.409 AIRL_rec: 0.1775 Acc: 0.941 LR: 2.81e-05
logs/agreidv2_airl_iso.log:150:Epoch[9] Iter[200/786] Loss: 3.170 CE: 1.462 Tri: 0.221 CE_rec: 1.398 AIRL_rec: 0.1776 Acc: 0.943 LR: 2.81e-05
logs/agreidv2_airl_iso.log:151:Epoch[9] Iter[250/786] Loss: 3.149 CE: 1.454 Tri: 0.221 CE_rec: 1.385 AIRL_rec: 0.1776 Acc: 0.945 LR: 2.81e-05
logs/agreidv2_airl_iso.log:152:Epoch[9] Iter[300/786] Loss: 3.134 CE: 1.448 Tri: 0.219 CE_rec: 1.378 AIRL_rec: 0.1784 Acc: 0.946 LR: 2.81e-05
logs/agreidv2_airl_iso.log:153:Epoch[9] Iter[350/786] Loss: 3.122 CE: 1.440 Tri: 0.222 CE_rec: 1.370 AIRL_rec: 0.1789 Acc: 0.948 LR: 2.81e-05
logs/agreidv2_airl_iso.log:154:Epoch[9] Iter[400/786] Loss: 3.110 CE: 1.437 Tri: 0.218 CE_rec: 1.365 AIRL_rec: 0.1795 Acc: 0.949 LR: 2.81e-05
logs/agreidv2_airl_iso.log:155:Epoch[9] Iter[450/786] Loss: 3.100 CE: 1.434 Tri: 0.217 CE_rec: 1.360 AIRL_rec: 0.1802 Acc: 0.950 LR: 2.81e-05
logs/agreidv2_airl_iso.log:156:Epoch[9] Iter[500/786] Loss: 3.093 CE: 1.430 Tri: 0.217 CE_rec: 1.355 AIRL_rec: 0.1805 Acc: 0.950 LR: 2.81e-05
logs/agreidv2_airl_iso.log:157:Epoch[9] Iter[550/786] Loss: 3.077 CE: 1.425 Tri: 0.214 CE_rec: 1.349 AIRL_rec: 0.1805 Acc: 0.952 LR: 2.81e-05
logs/agreidv2_airl_iso.log:158:Epoch[9] Iter[600/786] Loss: 3.064 CE: 1.419 Tri: 0.212 CE_rec: 1.342 AIRL_rec: 0.1800 Acc: 0.953 LR: 2.81e-05
logs/agreidv2_airl_iso.log:159:Epoch[9] Iter[650/786] Loss: 3.052 CE: 1.414 Tri: 0.212 CE_rec: 1.337 AIRL_rec: 0.1796 Acc: 0.953 LR: 2.81e-05
logs/agreidv2_airl_iso.log:160:Epoch[9] Iter[700/786] Loss: 3.033 CE: 1.406 Tri: 0.209 CE_rec: 1.329 AIRL_rec: 0.1788 Acc: 0.955 LR: 2.81e-05
logs/agreidv2_airl_iso.log:161:Epoch[9] done in 238.4s  Loss=3.010 Acc=0.956 AIRL-ISO[lam_eff=0.500 ce_rec=1.319 consistency=0.1772 deg_scale_mean=0.625 n_ground=28598]
logs/agreidv2_airl_iso.log:162:Epoch[10] Iter[50/786] Loss: 3.174 CE: 1.468 Tri: 0.197 CE_rec: 1.421 AIRL_rec: 0.1742 Acc: 0.939 LR: 3.15e-05
logs/agreidv2_airl_iso.log:163:Epoch[10] Iter[100/786] Loss: 3.117 CE: 1.445 Tri: 0.197 CE_rec: 1.385 AIRL_rec: 0.1773 Acc: 0.947 LR: 3.15e-05
logs/agreidv2_airl_iso.log:164:Epoch[10] Iter[150/786] Loss: 3.083 CE: 1.432 Tri: 0.196 CE_rec: 1.365 AIRL_rec: 0.1808 Acc: 0.950 LR: 3.15e-05
logs/agreidv2_airl_iso.log:165:Epoch[10] Iter[200/786] Loss: 3.058 CE: 1.423 Tri: 0.192 CE_rec: 1.353 AIRL_rec: 0.1808 Acc: 0.952 LR: 3.15e-05
logs/agreidv2_airl_iso.log:166:Epoch[10] Iter[250/786] Loss: 3.058 CE: 1.423 Tri: 0.193 CE_rec: 1.351 AIRL_rec: 0.1819 Acc: 0.952 LR: 3.15e-05
logs/agreidv2_airl_iso.log:167:Epoch[10] Iter[300/786] Loss: 3.042 CE: 1.417 Tri: 0.191 CE_rec: 1.343 AIRL_rec: 0.1835 Acc: 0.953 LR: 3.15e-05
logs/agreidv2_airl_iso.log:168:Epoch[10] Iter[350/786] Loss: 3.038 CE: 1.415 Tri: 0.189 CE_rec: 1.342 AIRL_rec: 0.1840 Acc: 0.953 LR: 3.15e-05
logs/agreidv2_airl_iso.log:169:Epoch[10] Iter[400/786] Loss: 3.024 CE: 1.410 Tri: 0.184 CE_rec: 1.337 AIRL_rec: 0.1841 Acc: 0.955 LR: 3.15e-05
logs/agreidv2_airl_iso.log:170:Epoch[10] Iter[450/786] Loss: 3.011 CE: 1.405 Tri: 0.184 CE_rec: 1.331 AIRL_rec: 0.1831 Acc: 0.956 LR: 3.15e-05
logs/agreidv2_airl_iso.log:171:Epoch[10] Iter[500/786] Loss: 3.009 CE: 1.404 Tri: 0.186 CE_rec: 1.327 AIRL_rec: 0.1840 Acc: 0.956 LR: 3.15e-05
logs/agreidv2_airl_iso.log:172:Epoch[10] Iter[550/786] Loss: 3.006 CE: 1.401 Tri: 0.188 CE_rec: 1.324 AIRL_rec: 0.1837 Acc: 0.956 LR: 3.15e-05
logs/agreidv2_airl_iso.log:173:Epoch[10] Iter[600/786] Loss: 2.992 CE: 1.396 Tri: 0.185 CE_rec: 1.319 AIRL_rec: 0.1827 Acc: 0.957 LR: 3.15e-05
logs/agreidv2_airl_iso.log:174:Epoch[10] Iter[650/786] Loss: 2.981 CE: 1.391 Tri: 0.186 CE_rec: 1.314 AIRL_rec: 0.1812 Acc: 0.958 LR: 3.15e-05
logs/agreidv2_airl_iso.log:175:Epoch[10] Iter[700/786] Loss: 2.970 CE: 1.385 Tri: 0.187 CE_rec: 1.308 AIRL_rec: 0.1794 Acc: 0.959 LR: 3.15e-05
logs/agreidv2_airl_iso.log:176:Epoch[10] done in 239.8s  Loss=2.947 Acc=0.960 AIRL-ISO[lam_eff=0.500 ce_rec=1.299 consistency=0.1776 deg_scale_mean=0.624 n_ground=28763]
logs/agreidv2_airl_iso.log:178:    [A->G] mAP=71.36  R1=79.84  R5=87.61  mINP=43.40
logs/agreidv2_airl_iso.log:179:    [G->A] mAP=71.86  R1=81.61  R5=88.07  mINP=35.13
logs/agreidv2_airl_iso.log:180:    [mean] mAP=71.61  R1=80.73
logs/agreidv2_airl_iso.log:181:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 10 ----
logs/agreidv2_airl_iso.log:182:    [A->G] full mAP=71.36 R1=79.84 | rec mAP=72.28 R1=81.20 | FUSE mAP=72.89 R1=81.20
logs/agreidv2_airl_iso.log:183:    [G->A] full mAP=71.86 R1=81.61 | rec mAP=72.38 R1=81.61 | FUSE mAP=73.53 R1=82.99
logs/agreidv2_airl_iso.log:185:    * new best mean mAP=73.21 (epoch 10) saved
logs/agreidv2_airl_iso.log:186:Epoch[11] Iter[50/786] Loss: 3.207 CE: 1.495 Tri: 0.179 CE_rec: 1.445 AIRL_rec: 0.1774 Acc: 0.932 LR: 3.50e-05
logs/agreidv2_airl_iso.log:187:Epoch[11] Iter[100/786] Loss: 3.109 CE: 1.459 Tri: 0.167 CE_rec: 1.394 AIRL_rec: 0.1780 Acc: 0.941 LR: 3.50e-05
logs/agreidv2_airl_iso.log:188:Epoch[11] Iter[150/786] Loss: 3.070 CE: 1.440 Tri: 0.167 CE_rec: 1.372 AIRL_rec: 0.1807 Acc: 0.946 LR: 3.50e-05
logs/agreidv2_airl_iso.log:189:Epoch[11] Iter[200/786] Loss: 3.066 CE: 1.435 Tri: 0.178 CE_rec: 1.363 AIRL_rec: 0.1818 Acc: 0.947 LR: 3.50e-05
logs/agreidv2_airl_iso.log:190:Epoch[11] Iter[250/786] Loss: 3.049 CE: 1.425 Tri: 0.179 CE_rec: 1.353 AIRL_rec: 0.1834 Acc: 0.949 LR: 3.50e-05
logs/agreidv2_airl_iso.log:191:Epoch[11] Iter[300/786] Loss: 3.035 CE: 1.419 Tri: 0.176 CE_rec: 1.347 AIRL_rec: 0.1845 Acc: 0.951 LR: 3.50e-05
logs/agreidv2_airl_iso.log:192:Epoch[11] Iter[350/786] Loss: 3.020 CE: 1.413 Tri: 0.174 CE_rec: 1.340 AIRL_rec: 0.1851 Acc: 0.952 LR: 3.50e-05
logs/agreidv2_airl_iso.log:193:Epoch[11] Iter[400/786] Loss: 3.008 CE: 1.408 Tri: 0.172 CE_rec: 1.336 AIRL_rec: 0.1842 Acc: 0.953 LR: 3.50e-05
logs/agreidv2_airl_iso.log:194:Epoch[11] Iter[450/786] Loss: 3.004 CE: 1.405 Tri: 0.174 CE_rec: 1.333 AIRL_rec: 0.1835 Acc: 0.953 LR: 3.50e-05
logs/agreidv2_airl_iso.log:195:Epoch[11] Iter[500/786] Loss: 2.998 CE: 1.402 Tri: 0.175 CE_rec: 1.329 AIRL_rec: 0.1834 Acc: 0.954 LR: 3.50e-05
logs/agreidv2_airl_iso.log:196:Epoch[11] Iter[550/786] Loss: 2.982 CE: 1.396 Tri: 0.173 CE_rec: 1.322 AIRL_rec: 0.1837 Acc: 0.956 LR: 3.50e-05
logs/agreidv2_airl_iso.log:197:Epoch[11] Iter[600/786] Loss: 2.971 CE: 1.390 Tri: 0.174 CE_rec: 1.316 AIRL_rec: 0.1826 Acc: 0.957 LR: 3.50e-05
logs/agreidv2_airl_iso.log:198:Epoch[11] Iter[650/786] Loss: 2.956 CE: 1.383 Tri: 0.172 CE_rec: 1.310 AIRL_rec: 0.1813 Acc: 0.958 LR: 3.50e-05
logs/agreidv2_airl_iso.log:199:Epoch[11] Iter[700/786] Loss: 2.937 CE: 1.375 Tri: 0.170 CE_rec: 1.302 AIRL_rec: 0.1796 Acc: 0.959 LR: 3.50e-05
logs/agreidv2_airl_iso.log:200:Epoch[11] done in 231.7s  Loss=2.915 Acc=0.961 AIRL-ISO[lam_eff=0.500 ce_rec=1.293 consistency=0.1781 deg_scale_mean=0.624 n_ground=28694]
logs/agreidv2_airl_iso.log:201:Epoch[12] Iter[50/786] Loss: 3.185 CE: 1.496 Tri: 0.170 CE_rec: 1.436 AIRL_rec: 0.1665 Acc: 0.931 LR: 3.50e-05
logs/agreidv2_airl_iso.log:202:Epoch[12] Iter[100/786] Loss: 3.132 CE: 1.471 Tri: 0.171 CE_rec: 1.404 AIRL_rec: 0.1724 Acc: 0.936 LR: 3.50e-05
logs/agreidv2_airl_iso.log:203:Epoch[12] Iter[150/786] Loss: 3.053 CE: 1.438 Tri: 0.162 CE_rec: 1.367 AIRL_rec: 0.1732 Acc: 0.945 LR: 3.50e-05
logs/agreidv2_airl_iso.log:204:Epoch[12] Iter[200/786] Loss: 3.018 CE: 1.422 Tri: 0.158 CE_rec: 1.350 AIRL_rec: 0.1751 Acc: 0.949 LR: 3.50e-05
logs/agreidv2_airl_iso.log:205:Epoch[12] Iter[250/786] Loss: 2.991 CE: 1.409 Tri: 0.155 CE_rec: 1.338 AIRL_rec: 0.1773 Acc: 0.951 LR: 3.50e-05
logs/agreidv2_airl_iso.log:206:Epoch[12] Iter[300/786] Loss: 2.981 CE: 1.404 Tri: 0.157 CE_rec: 1.331 AIRL_rec: 0.1780 Acc: 0.953 LR: 3.50e-05
logs/agreidv2_airl_iso.log:207:Epoch[12] Iter[350/786] Loss: 2.966 CE: 1.397 Tri: 0.156 CE_rec: 1.323 AIRL_rec: 0.1788 Acc: 0.954 LR: 3.50e-05
logs/agreidv2_airl_iso.log:208:Epoch[12] Iter[400/786] Loss: 2.957 CE: 1.393 Tri: 0.157 CE_rec: 1.317 AIRL_rec: 0.1800 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_airl_iso.log:209:Epoch[12] Iter[450/786] Loss: 2.942 CE: 1.386 Tri: 0.156 CE_rec: 1.310 AIRL_rec: 0.1799 Acc: 0.957 LR: 3.50e-05
logs/agreidv2_airl_iso.log:210:Epoch[12] Iter[500/786] Loss: 2.931 CE: 1.381 Tri: 0.154 CE_rec: 1.305 AIRL_rec: 0.1798 Acc: 0.957 LR: 3.50e-05
logs/agreidv2_airl_iso.log:211:Epoch[12] Iter[550/786] Loss: 2.918 CE: 1.375 Tri: 0.153 CE_rec: 1.299 AIRL_rec: 0.1790 Acc: 0.959 LR: 3.50e-05
logs/agreidv2_airl_iso.log:212:Epoch[12] Iter[600/786] Loss: 2.904 CE: 1.369 Tri: 0.153 CE_rec: 1.293 AIRL_rec: 0.1777 Acc: 0.960 LR: 3.50e-05
logs/agreidv2_airl_iso.log:213:Epoch[12] Iter[650/786] Loss: 2.892 CE: 1.364 Tri: 0.152 CE_rec: 1.288 AIRL_rec: 0.1756 Acc: 0.961 LR: 3.50e-05
logs/agreidv2_airl_iso.log:214:Epoch[12] Iter[700/786] Loss: 2.878 CE: 1.357 Tri: 0.151 CE_rec: 1.282 AIRL_rec: 0.1738 Acc: 0.962 LR: 3.50e-05
logs/agreidv2_airl_iso.log:215:Epoch[12] done in 233.9s  Loss=2.858 Acc=0.963 AIRL-ISO[lam_eff=0.500 ce_rec=1.274 consistency=0.1719 deg_scale_mean=0.625 n_ground=28735]
logs/agreidv2_airl_iso.log:216:Epoch[13] Iter[50/786] Loss: 3.121 CE: 1.457 Tri: 0.186 CE_rec: 1.396 AIRL_rec: 0.1647 Acc: 0.945 LR: 3.49e-05
logs/agreidv2_airl_iso.log:217:Epoch[13] Iter[100/786] Loss: 3.029 CE: 1.423 Tri: 0.167 CE_rec: 1.356 AIRL_rec: 0.1680 Acc: 0.952 LR: 3.49e-05
logs/agreidv2_airl_iso.log:218:Epoch[13] Iter[150/786] Loss: 2.962 CE: 1.399 Tri: 0.152 CE_rec: 1.327 AIRL_rec: 0.1698 Acc: 0.956 LR: 3.49e-05
logs/agreidv2_airl_iso.log:219:Epoch[13] Iter[200/786] Loss: 2.938 CE: 1.388 Tri: 0.148 CE_rec: 1.316 AIRL_rec: 0.1740 Acc: 0.957 LR: 3.49e-05
logs/agreidv2_airl_iso.log:220:Epoch[13] Iter[250/786] Loss: 2.916 CE: 1.379 Tri: 0.144 CE_rec: 1.305 AIRL_rec: 0.1753 Acc: 0.959 LR: 3.49e-05
logs/agreidv2_airl_iso.log:221:Epoch[13] Iter[300/786] Loss: 2.901 CE: 1.373 Tri: 0.141 CE_rec: 1.299 AIRL_rec: 0.1760 Acc: 0.960 LR: 3.49e-05
logs/agreidv2_airl_iso.log:222:Epoch[13] Iter[350/786] Loss: 2.887 CE: 1.367 Tri: 0.139 CE_rec: 1.293 AIRL_rec: 0.1751 Acc: 0.960 LR: 3.49e-05
logs/agreidv2_airl_iso.log:223:Epoch[13] Iter[400/786] Loss: 2.880 CE: 1.363 Tri: 0.140 CE_rec: 1.289 AIRL_rec: 0.1750 Acc: 0.961 LR: 3.49e-05
logs/agreidv2_airl_iso.log:224:Epoch[13] Iter[450/786] Loss: 2.867 CE: 1.357 Tri: 0.139 CE_rec: 1.283 AIRL_rec: 0.1743 Acc: 0.962 LR: 3.49e-05
logs/agreidv2_airl_iso.log:225:Epoch[13] Iter[500/786] Loss: 2.854 CE: 1.352 Tri: 0.137 CE_rec: 1.278 AIRL_rec: 0.1733 Acc: 0.963 LR: 3.49e-05
logs/agreidv2_airl_iso.log:226:Epoch[13] Iter[550/786] Loss: 2.842 CE: 1.347 Tri: 0.136 CE_rec: 1.274 AIRL_rec: 0.1723 Acc: 0.964 LR: 3.49e-05
logs/agreidv2_airl_iso.log:227:Epoch[13] Iter[600/786] Loss: 2.835 CE: 1.343 Tri: 0.136 CE_rec: 1.270 AIRL_rec: 0.1716 Acc: 0.964 LR: 3.49e-05
logs/agreidv2_airl_iso.log:228:Epoch[13] Iter[650/786] Loss: 2.824 CE: 1.338 Tri: 0.136 CE_rec: 1.266 AIRL_rec: 0.1702 Acc: 0.965 LR: 3.49e-05
logs/agreidv2_airl_iso.log:229:Epoch[13] Iter[700/786] Loss: 2.810 CE: 1.331 Tri: 0.135 CE_rec: 1.260 AIRL_rec: 0.1684 Acc: 0.966 LR: 3.49e-05
logs/agreidv2_airl_iso.log:230:Epoch[13] done in 234.3s  Loss=2.792 Acc=0.967 AIRL-ISO[lam_eff=0.500 ce_rec=1.252 consistency=0.1663 deg_scale_mean=0.624 n_ground=28655]
logs/agreidv2_airl_iso.log:231:Epoch[14] Iter[50/786] Loss: 2.998 CE: 1.415 Tri: 0.149 CE_rec: 1.352 AIRL_rec: 0.1644 Acc: 0.947 LR: 3.47e-05
logs/agreidv2_airl_iso.log:232:Epoch[14] Iter[100/786] Loss: 2.917 CE: 1.384 Tri: 0.131 CE_rec: 1.317 AIRL_rec: 0.1709 Acc: 0.955 LR: 3.47e-05
logs/agreidv2_airl_iso.log:233:Epoch[14] Iter[150/786] Loss: 2.878 CE: 1.368 Tri: 0.127 CE_rec: 1.298 AIRL_rec: 0.1727 Acc: 0.959 LR: 3.47e-05
logs/agreidv2_airl_iso.log:234:Epoch[14] Iter[200/786] Loss: 2.855 CE: 1.357 Tri: 0.126 CE_rec: 1.286 AIRL_rec: 0.1741 Acc: 0.961 LR: 3.47e-05
logs/agreidv2_airl_iso.log:235:Epoch[14] Iter[250/786] Loss: 2.847 CE: 1.350 Tri: 0.132 CE_rec: 1.277 AIRL_rec: 0.1737 Acc: 0.963 LR: 3.47e-05
logs/agreidv2_airl_iso.log:236:Epoch[14] Iter[300/786] Loss: 2.827 CE: 1.344 Tri: 0.127 CE_rec: 1.270 AIRL_rec: 0.1728 Acc: 0.964 LR: 3.47e-05
logs/agreidv2_airl_iso.log:237:Epoch[14] Iter[350/786] Loss: 2.815 CE: 1.339 Tri: 0.125 CE_rec: 1.266 AIRL_rec: 0.1719 Acc: 0.964 LR: 3.47e-05
logs/agreidv2_airl_iso.log:238:Epoch[14] Iter[400/786] Loss: 2.812 CE: 1.337 Tri: 0.127 CE_rec: 1.263 AIRL_rec: 0.1715 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_airl_iso.log:239:Epoch[14] Iter[450/786] Loss: 2.801 CE: 1.333 Tri: 0.124 CE_rec: 1.259 AIRL_rec: 0.1704 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_airl_iso.log:240:Epoch[14] Iter[500/786] Loss: 2.797 CE: 1.332 Tri: 0.123 CE_rec: 1.258 AIRL_rec: 0.1696 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_airl_iso.log:241:Epoch[14] Iter[550/786] Loss: 2.789 CE: 1.328 Tri: 0.123 CE_rec: 1.254 AIRL_rec: 0.1687 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_airl_iso.log:242:Epoch[14] Iter[600/786] Loss: 2.775 CE: 1.322 Tri: 0.120 CE_rec: 1.249 AIRL_rec: 0.1676 Acc: 0.966 LR: 3.47e-05
logs/agreidv2_airl_iso.log:243:Epoch[14] Iter[650/786] Loss: 2.763 CE: 1.317 Tri: 0.119 CE_rec: 1.244 AIRL_rec: 0.1659 Acc: 0.967 LR: 3.47e-05
logs/agreidv2_airl_iso.log:244:Epoch[14] Iter[700/786] Loss: 2.752 CE: 1.311 Tri: 0.119 CE_rec: 1.239 AIRL_rec: 0.1639 Acc: 0.968 LR: 3.47e-05
logs/agreidv2_airl_iso.log:245:Epoch[14] done in 236.4s  Loss=2.735 Acc=0.969 AIRL-ISO[lam_eff=0.500 ce_rec=1.233 consistency=0.1618 deg_scale_mean=0.626 n_ground=28817]
logs/agreidv2_airl_iso.log:246:Epoch[15] Iter[50/786] Loss: 3.023 CE: 1.434 Tri: 0.160 CE_rec: 1.349 AIRL_rec: 0.1613 Acc: 0.942 LR: 3.45e-05
logs/agreidv2_airl_iso.log:247:Epoch[15] Iter[100/786] Loss: 2.913 CE: 1.390 Tri: 0.135 CE_rec: 1.307 AIRL_rec: 0.1617 Acc: 0.953 LR: 3.45e-05
logs/agreidv2_airl_iso.log:248:Epoch[15] Iter[150/786] Loss: 2.869 CE: 1.369 Tri: 0.129 CE_rec: 1.289 AIRL_rec: 0.1626 Acc: 0.957 LR: 3.45e-05
logs/agreidv2_airl_iso.log:249:Epoch[15] Iter[200/786] Loss: 2.844 CE: 1.358 Tri: 0.123 CE_rec: 1.280 AIRL_rec: 0.1661 Acc: 0.959 LR: 3.45e-05
logs/agreidv2_airl_iso.log:250:Epoch[15] Iter[250/786] Loss: 2.811 CE: 1.343 Tri: 0.117 CE_rec: 1.268 AIRL_rec: 0.1666 Acc: 0.962 LR: 3.45e-05
logs/agreidv2_airl_iso.log:251:Epoch[15] Iter[300/786] Loss: 2.792 CE: 1.335 Tri: 0.112 CE_rec: 1.262 AIRL_rec: 0.1657 Acc: 0.963 LR: 3.45e-05
logs/agreidv2_airl_iso.log:252:Epoch[15] Iter[350/786] Loss: 2.788 CE: 1.333 Tri: 0.115 CE_rec: 1.258 AIRL_rec: 0.1659 Acc: 0.963 LR: 3.45e-05
logs/agreidv2_airl_iso.log:253:Epoch[15] Iter[400/786] Loss: 2.779 CE: 1.328 Tri: 0.115 CE_rec: 1.254 AIRL_rec: 0.1650 Acc: 0.964 LR: 3.45e-05
logs/agreidv2_airl_iso.log:254:Epoch[15] Iter[450/786] Loss: 2.769 CE: 1.323 Tri: 0.114 CE_rec: 1.249 AIRL_rec: 0.1636 Acc: 0.965 LR: 3.45e-05
logs/agreidv2_airl_iso.log:255:Epoch[15] Iter[500/786] Loss: 2.758 CE: 1.318 Tri: 0.114 CE_rec: 1.245 AIRL_rec: 0.1629 Acc: 0.966 LR: 3.45e-05
logs/agreidv2_airl_iso.log:256:Epoch[15] Iter[550/786] Loss: 2.749 CE: 1.314 Tri: 0.113 CE_rec: 1.242 AIRL_rec: 0.1621 Acc: 0.967 LR: 3.45e-05
logs/agreidv2_airl_iso.log:257:Epoch[15] Iter[600/786] Loss: 2.737 CE: 1.308 Tri: 0.112 CE_rec: 1.236 AIRL_rec: 0.1611 Acc: 0.968 LR: 3.45e-05
logs/agreidv2_airl_iso.log:258:Epoch[15] Iter[650/786] Loss: 2.730 CE: 1.305 Tri: 0.112 CE_rec: 1.233 AIRL_rec: 0.1595 Acc: 0.969 LR: 3.45e-05
logs/agreidv2_airl_iso.log:259:Epoch[15] Iter[700/786] Loss: 2.713 CE: 1.298 Tri: 0.109 CE_rec: 1.228 AIRL_rec: 0.1573 Acc: 0.970 LR: 3.45e-05
logs/agreidv2_airl_iso.log:260:Epoch[15] done in 238.3s  Loss=2.696 Acc=0.971 AIRL-ISO[lam_eff=0.500 ce_rec=1.221 consistency=0.1557 deg_scale_mean=0.623 n_ground=28659]
logs/agreidv2_airl_iso.log:261:Epoch[16] Iter[50/786] Loss: 2.797 CE: 1.339 Tri: 0.100 CE_rec: 1.280 AIRL_rec: 0.1556 Acc: 0.962 LR: 3.41e-05
logs/agreidv2_airl_iso.log:262:Epoch[16] Iter[100/786] Loss: 2.774 CE: 1.330 Tri: 0.103 CE_rec: 1.263 AIRL_rec: 0.1558 Acc: 0.964 LR: 3.41e-05
logs/agreidv2_airl_iso.log:263:Epoch[16] Iter[150/786] Loss: 2.750 CE: 1.321 Tri: 0.096 CE_rec: 1.254 AIRL_rec: 0.1569 Acc: 0.966 LR: 3.41e-05
logs/agreidv2_airl_iso.log:264:Epoch[16] Iter[200/786] Loss: 2.746 CE: 1.319 Tri: 0.098 CE_rec: 1.249 AIRL_rec: 0.1599 Acc: 0.966 LR: 3.41e-05
logs/agreidv2_airl_iso.log:265:Epoch[16] Iter[250/786] Loss: 2.725 CE: 1.311 Tri: 0.093 CE_rec: 1.240 AIRL_rec: 0.1607 Acc: 0.967 LR: 3.41e-05
logs/agreidv2_airl_iso.log:266:Epoch[16] Iter[300/786] Loss: 2.723 CE: 1.309 Tri: 0.095 CE_rec: 1.238 AIRL_rec: 0.1604 Acc: 0.968 LR: 3.41e-05
logs/agreidv2_airl_iso.log:267:Epoch[16] Iter[350/786] Loss: 2.727 CE: 1.309 Tri: 0.100 CE_rec: 1.238 AIRL_rec: 0.1604 Acc: 0.968 LR: 3.41e-05
logs/agreidv2_airl_iso.log:268:Epoch[16] Iter[400/786] Loss: 2.722 CE: 1.305 Tri: 0.102 CE_rec: 1.235 AIRL_rec: 0.1596 Acc: 0.968 LR: 3.41e-05
logs/agreidv2_airl_iso.log:269:Epoch[16] Iter[450/786] Loss: 2.713 CE: 1.302 Tri: 0.100 CE_rec: 1.232 AIRL_rec: 0.1587 Acc: 0.969 LR: 3.41e-05
logs/agreidv2_airl_iso.log:270:Epoch[16] Iter[500/786] Loss: 2.708 CE: 1.298 Tri: 0.101 CE_rec: 1.229 AIRL_rec: 0.1579 Acc: 0.970 LR: 3.41e-05
logs/agreidv2_airl_iso.log:271:Epoch[16] Iter[550/786] Loss: 2.704 CE: 1.296 Tri: 0.102 CE_rec: 1.227 AIRL_rec: 0.1571 Acc: 0.970 LR: 3.41e-05
logs/agreidv2_airl_iso.log:272:Epoch[16] Iter[600/786] Loss: 2.690 CE: 1.290 Tri: 0.099 CE_rec: 1.223 AIRL_rec: 0.1557 Acc: 0.971 LR: 3.41e-05
logs/agreidv2_airl_iso.log:273:Epoch[16] Iter[650/786] Loss: 2.678 CE: 1.284 Tri: 0.099 CE_rec: 1.218 AIRL_rec: 0.1546 Acc: 0.972 LR: 3.41e-05
logs/agreidv2_airl_iso.log:274:Epoch[16] Iter[700/786] Loss: 2.667 CE: 1.279 Tri: 0.098 CE_rec: 1.214 AIRL_rec: 0.1533 Acc: 0.973 LR: 3.41e-05
logs/agreidv2_airl_iso.log:275:Epoch[16] done in 238.1s  Loss=2.653 Acc=0.974 AIRL-ISO[lam_eff=0.500 ce_rec=1.208 consistency=0.1514 deg_scale_mean=0.625 n_ground=28697]
logs/agreidv2_airl_iso.log:276:Epoch[17] Iter[50/786] Loss: 2.867 CE: 1.377 Tri: 0.101 CE_rec: 1.310 AIRL_rec: 0.1578 Acc: 0.953 LR: 3.38e-05
logs/agreidv2_airl_iso.log:277:Epoch[17] Iter[100/786] Loss: 2.812 CE: 1.349 Tri: 0.099 CE_rec: 1.285 AIRL_rec: 0.1556 Acc: 0.962 LR: 3.38e-05
logs/agreidv2_airl_iso.log:278:Epoch[17] Iter[150/786] Loss: 2.774 CE: 1.333 Tri: 0.098 CE_rec: 1.265 AIRL_rec: 0.1568 Acc: 0.965 LR: 3.38e-05
logs/agreidv2_airl_iso.log:279:Epoch[17] Iter[200/786] Loss: 2.743 CE: 1.319 Tri: 0.093 CE_rec: 1.252 AIRL_rec: 0.1569 Acc: 0.968 LR: 3.38e-05
logs/agreidv2_airl_iso.log:280:Epoch[17] Iter[250/786] Loss: 2.721 CE: 1.308 Tri: 0.093 CE_rec: 1.241 AIRL_rec: 0.1556 Acc: 0.970 LR: 3.38e-05
logs/agreidv2_airl_iso.log:281:Epoch[17] Iter[300/786] Loss: 2.712 CE: 1.304 Tri: 0.093 CE_rec: 1.237 AIRL_rec: 0.1559 Acc: 0.970 LR: 3.38e-05
logs/agreidv2_airl_iso.log:282:Epoch[17] Iter[350/786] Loss: 2.698 CE: 1.298 Tri: 0.091 CE_rec: 1.231 AIRL_rec: 0.1544 Acc: 0.970 LR: 3.38e-05
logs/agreidv2_airl_iso.log:283:Epoch[17] Iter[400/786] Loss: 2.687 CE: 1.293 Tri: 0.091 CE_rec: 1.226 AIRL_rec: 0.1539 Acc: 0.972 LR: 3.38e-05
logs/agreidv2_airl_iso.log:284:Epoch[17] Iter[450/786] Loss: 2.676 CE: 1.288 Tri: 0.090 CE_rec: 1.222 AIRL_rec: 0.1530 Acc: 0.972 LR: 3.38e-05
logs/agreidv2_airl_iso.log:285:Epoch[17] Iter[500/786] Loss: 2.665 CE: 1.283 Tri: 0.088 CE_rec: 1.218 AIRL_rec: 0.1515 Acc: 0.973 LR: 3.38e-05
logs/agreidv2_airl_iso.log:286:Epoch[17] Iter[550/786] Loss: 2.657 CE: 1.280 Tri: 0.088 CE_rec: 1.214 AIRL_rec: 0.1500 Acc: 0.973 LR: 3.38e-05
logs/agreidv2_airl_iso.log:287:Epoch[17] Iter[600/786] Loss: 2.645 CE: 1.275 Tri: 0.086 CE_rec: 1.210 AIRL_rec: 0.1484 Acc: 0.974 LR: 3.38e-05
logs/agreidv2_airl_iso.log:288:Epoch[17] Iter[650/786] Loss: 2.633 CE: 1.270 Tri: 0.085 CE_rec: 1.205 AIRL_rec: 0.1473 Acc: 0.975 LR: 3.38e-05
logs/agreidv2_airl_iso.log:289:Epoch[17] Iter[700/786] Loss: 2.621 CE: 1.264 Tri: 0.083 CE_rec: 1.201 AIRL_rec: 0.1454 Acc: 0.976 LR: 3.38e-05
logs/agreidv2_airl_iso.log:290:Epoch[17] done in 239.0s  Loss=2.603 Acc=0.977 AIRL-ISO[lam_eff=0.500 ce_rec=1.194 consistency=0.1433 deg_scale_mean=0.624 n_ground=28762]
logs/agreidv2_airl_iso.log:291:Epoch[18] Iter[50/786] Loss: 2.814 CE: 1.358 Tri: 0.099 CE_rec: 1.288 AIRL_rec: 0.1395 Acc: 0.955 LR: 3.33e-05
logs/agreidv2_airl_iso.log:292:Epoch[18] Iter[100/786] Loss: 2.748 CE: 1.327 Tri: 0.093 CE_rec: 1.255 AIRL_rec: 0.1453 Acc: 0.962 LR: 3.33e-05
logs/agreidv2_airl_iso.log:293:Epoch[18] Iter[150/786] Loss: 2.719 CE: 1.313 Tri: 0.092 CE_rec: 1.241 AIRL_rec: 0.1474 Acc: 0.966 LR: 3.33e-05
logs/agreidv2_airl_iso.log:294:Epoch[18] Iter[200/786] Loss: 2.701 CE: 1.303 Tri: 0.091 CE_rec: 1.232 AIRL_rec: 0.1491 Acc: 0.968 LR: 3.33e-05
logs/agreidv2_airl_iso.log:295:Epoch[18] Iter[250/786] Loss: 2.688 CE: 1.296 Tri: 0.091 CE_rec: 1.226 AIRL_rec: 0.1499 Acc: 0.969 LR: 3.33e-05
logs/agreidv2_airl_iso.log:296:Epoch[18] Iter[300/786] Loss: 2.667 CE: 1.286 Tri: 0.087 CE_rec: 1.218 AIRL_rec: 0.1490 Acc: 0.971 LR: 3.33e-05
logs/agreidv2_airl_iso.log:297:Epoch[18] Iter[350/786] Loss: 2.661 CE: 1.283 Tri: 0.088 CE_rec: 1.215 AIRL_rec: 0.1488 Acc: 0.971 LR: 3.33e-05
logs/agreidv2_airl_iso.log:298:Epoch[18] Iter[400/786] Loss: 2.651 CE: 1.279 Tri: 0.087 CE_rec: 1.211 AIRL_rec: 0.1478 Acc: 0.972 LR: 3.33e-05
logs/agreidv2_airl_iso.log:299:Epoch[18] Iter[450/786] Loss: 2.643 CE: 1.274 Tri: 0.087 CE_rec: 1.208 AIRL_rec: 0.1474 Acc: 0.973 LR: 3.33e-05
logs/agreidv2_airl_iso.log:300:Epoch[18] Iter[500/786] Loss: 2.629 CE: 1.269 Tri: 0.083 CE_rec: 1.203 AIRL_rec: 0.1461 Acc: 0.974 LR: 3.33e-05
logs/agreidv2_airl_iso.log:301:Epoch[18] Iter[550/786] Loss: 2.618 CE: 1.264 Tri: 0.082 CE_rec: 1.199 AIRL_rec: 0.1448 Acc: 0.975 LR: 3.33e-05
logs/agreidv2_airl_iso.log:302:Epoch[18] Iter[600/786] Loss: 2.610 CE: 1.261 Tri: 0.082 CE_rec: 1.196 AIRL_rec: 0.1433 Acc: 0.975 LR: 3.33e-05
logs/agreidv2_airl_iso.log:303:Epoch[18] Iter[650/786] Loss: 2.600 CE: 1.256 Tri: 0.080 CE_rec: 1.192 AIRL_rec: 0.1418 Acc: 0.976 LR: 3.33e-05
logs/agreidv2_airl_iso.log:304:Epoch[18] Iter[700/786] Loss: 2.587 CE: 1.251 Tri: 0.079 CE_rec: 1.187 AIRL_rec: 0.1399 Acc: 0.977 LR: 3.33e-05
logs/agreidv2_airl_iso.log:305:Epoch[18] done in 239.3s  Loss=2.571 Acc=0.978 AIRL-ISO[lam_eff=0.500 ce_rec=1.182 consistency=0.1377 deg_scale_mean=0.623 n_ground=28736]
logs/agreidv2_airl_iso.log:306:Epoch[19] Iter[50/786] Loss: 2.748 CE: 1.331 Tri: 0.090 CE_rec: 1.259 AIRL_rec: 0.1348 Acc: 0.961 LR: 3.28e-05
logs/agreidv2_airl_iso.log:307:Epoch[19] Iter[100/786] Loss: 2.697 CE: 1.306 Tri: 0.084 CE_rec: 1.238 AIRL_rec: 0.1376 Acc: 0.968 LR: 3.28e-05
logs/agreidv2_airl_iso.log:308:Epoch[19] Iter[150/786] Loss: 2.656 CE: 1.288 Tri: 0.078 CE_rec: 1.221 AIRL_rec: 0.1390 Acc: 0.972 LR: 3.28e-05
logs/agreidv2_airl_iso.log:309:Epoch[19] Iter[200/786] Loss: 2.658 CE: 1.287 Tri: 0.081 CE_rec: 1.220 AIRL_rec: 0.1399 Acc: 0.970 LR: 3.28e-05
logs/agreidv2_airl_iso.log:310:Epoch[19] Iter[250/786] Loss: 2.643 CE: 1.279 Tri: 0.080 CE_rec: 1.213 AIRL_rec: 0.1410 Acc: 0.972 LR: 3.28e-05
logs/agreidv2_airl_iso.log:311:Epoch[19] Iter[300/786] Loss: 2.635 CE: 1.275 Tri: 0.080 CE_rec: 1.210 AIRL_rec: 0.1417 Acc: 0.972 LR: 3.28e-05
logs/agreidv2_airl_iso.log:312:Epoch[19] Iter[350/786] Loss: 2.623 CE: 1.270 Tri: 0.077 CE_rec: 1.206 AIRL_rec: 0.1410 Acc: 0.973 LR: 3.28e-05
logs/agreidv2_airl_iso.log:313:Epoch[19] Iter[400/786] Loss: 2.615 CE: 1.266 Tri: 0.076 CE_rec: 1.203 AIRL_rec: 0.1411 Acc: 0.974 LR: 3.28e-05
logs/agreidv2_airl_iso.log:314:Epoch[19] Iter[450/786] Loss: 2.607 CE: 1.262 Tri: 0.075 CE_rec: 1.199 AIRL_rec: 0.1407 Acc: 0.974 LR: 3.28e-05
logs/agreidv2_airl_iso.log:315:Epoch[19] Iter[500/786] Loss: 2.597 CE: 1.258 Tri: 0.074 CE_rec: 1.196 AIRL_rec: 0.1397 Acc: 0.975 LR: 3.28e-05
logs/agreidv2_airl_iso.log:316:Epoch[19] Iter[550/786] Loss: 2.589 CE: 1.254 Tri: 0.073 CE_rec: 1.192 AIRL_rec: 0.1387 Acc: 0.976 LR: 3.28e-05
logs/agreidv2_airl_iso.log:317:Epoch[19] Iter[600/786] Loss: 2.580 CE: 1.250 Tri: 0.072 CE_rec: 1.188 AIRL_rec: 0.1377 Acc: 0.977 LR: 3.28e-05
logs/agreidv2_airl_iso.log:318:Epoch[19] Iter[650/786] Loss: 2.569 CE: 1.245 Tri: 0.071 CE_rec: 1.184 AIRL_rec: 0.1364 Acc: 0.977 LR: 3.28e-05
logs/agreidv2_airl_iso.log:319:Epoch[19] Iter[700/786] Loss: 2.557 CE: 1.240 Tri: 0.069 CE_rec: 1.180 AIRL_rec: 0.1345 Acc: 0.978 LR: 3.28e-05
logs/agreidv2_airl_iso.log:320:Epoch[19] done in 240.1s  Loss=2.541 Acc=0.979 AIRL-ISO[lam_eff=0.500 ce_rec=1.174 consistency=0.1322 deg_scale_mean=0.624 n_ground=28722]
logs/agreidv2_airl_iso.log:321:Epoch[20] Iter[50/786] Loss: 2.666 CE: 1.305 Tri: 0.066 CE_rec: 1.233 AIRL_rec: 0.1236 Acc: 0.965 LR: 3.23e-05
logs/agreidv2_airl_iso.log:322:Epoch[20] Iter[100/786] Loss: 2.657 CE: 1.290 Tri: 0.076 CE_rec: 1.225 AIRL_rec: 0.1327 Acc: 0.968 LR: 3.23e-05
logs/agreidv2_airl_iso.log:323:Epoch[20] Iter[150/786] Loss: 2.635 CE: 1.279 Tri: 0.071 CE_rec: 1.219 AIRL_rec: 0.1333 Acc: 0.970 LR: 3.23e-05
logs/agreidv2_airl_iso.log:324:Epoch[20] Iter[200/786] Loss: 2.626 CE: 1.273 Tri: 0.072 CE_rec: 1.213 AIRL_rec: 0.1353 Acc: 0.972 LR: 3.23e-05
logs/agreidv2_airl_iso.log:325:Epoch[20] Iter[250/786] Loss: 2.615 CE: 1.268 Tri: 0.073 CE_rec: 1.207 AIRL_rec: 0.1357 Acc: 0.973 LR: 3.23e-05
logs/agreidv2_airl_iso.log:326:Epoch[20] Iter[300/786] Loss: 2.608 CE: 1.263 Tri: 0.074 CE_rec: 1.204 AIRL_rec: 0.1354 Acc: 0.974 LR: 3.23e-05
logs/agreidv2_airl_iso.log:327:Epoch[20] Iter[350/786] Loss: 2.599 CE: 1.259 Tri: 0.073 CE_rec: 1.199 AIRL_rec: 0.1358 Acc: 0.974 LR: 3.23e-05
logs/agreidv2_airl_iso.log:328:Epoch[20] Iter[400/786] Loss: 2.588 CE: 1.254 Tri: 0.071 CE_rec: 1.194 AIRL_rec: 0.1359 Acc: 0.975 LR: 3.23e-05
logs/agreidv2_airl_iso.log:329:Epoch[20] Iter[450/786] Loss: 2.578 CE: 1.250 Tri: 0.071 CE_rec: 1.190 AIRL_rec: 0.1353 Acc: 0.976 LR: 3.23e-05
logs/agreidv2_airl_iso.log:330:Epoch[20] Iter[500/786] Loss: 2.575 CE: 1.248 Tri: 0.071 CE_rec: 1.189 AIRL_rec: 0.1342 Acc: 0.976 LR: 3.23e-05
logs/agreidv2_airl_iso.log:331:Epoch[20] Iter[550/786] Loss: 2.569 CE: 1.245 Tri: 0.070 CE_rec: 1.186 AIRL_rec: 0.1334 Acc: 0.977 LR: 3.23e-05
logs/agreidv2_airl_iso.log:332:Epoch[20] Iter[600/786] Loss: 2.559 CE: 1.241 Tri: 0.069 CE_rec: 1.183 AIRL_rec: 0.1324 Acc: 0.977 LR: 3.23e-05
logs/agreidv2_airl_iso.log:333:Epoch[20] Iter[650/786] Loss: 2.550 CE: 1.237 Tri: 0.069 CE_rec: 1.179 AIRL_rec: 0.1308 Acc: 0.978 LR: 3.23e-05
logs/agreidv2_airl_iso.log:334:Epoch[20] Iter[700/786] Loss: 2.537 CE: 1.231 Tri: 0.068 CE_rec: 1.174 AIRL_rec: 0.1288 Acc: 0.979 LR: 3.23e-05
logs/agreidv2_airl_iso.log:335:Epoch[20] done in 240.7s  Loss=2.522 Acc=0.980 AIRL-ISO[lam_eff=0.500 ce_rec=1.168 consistency=0.1265 deg_scale_mean=0.625 n_ground=28838]
logs/agreidv2_airl_iso.log:337:    [A->G] mAP=70.65  R1=80.18  R5=87.82  mINP=40.17
logs/agreidv2_airl_iso.log:338:    [G->A] mAP=71.55  R1=82.61  R5=88.02  mINP=32.51
logs/agreidv2_airl_iso.log:339:    [mean] mAP=71.10  R1=81.39
logs/agreidv2_airl_iso.log:340:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 20 ----
logs/agreidv2_airl_iso.log:341:    [A->G] full mAP=70.65 R1=80.18 | rec mAP=72.52 R1=81.75 | FUSE mAP=72.83 R1=81.45
logs/agreidv2_airl_iso.log:342:    [G->A] full mAP=71.55 R1=82.61 | rec mAP=71.05 R1=81.50 | FUSE mAP=73.14 R1=83.38
logs/agreidv2_airl_iso.log:344:Epoch[21] Iter[50/786] Loss: 2.694 CE: 1.305 Tri: 0.082 CE_rec: 1.241 AIRL_rec: 0.1326 Acc: 0.968 LR: 3.17e-05
logs/agreidv2_airl_iso.log:345:Epoch[21] Iter[100/786] Loss: 2.631 CE: 1.279 Tri: 0.070 CE_rec: 1.215 AIRL_rec: 0.1336 Acc: 0.973 LR: 3.17e-05
logs/agreidv2_airl_iso.log:346:Epoch[21] Iter[150/786] Loss: 2.604 CE: 1.265 Tri: 0.070 CE_rec: 1.201 AIRL_rec: 0.1355 Acc: 0.974 LR: 3.17e-05
logs/agreidv2_airl_iso.log:347:Epoch[21] Iter[200/786] Loss: 2.585 CE: 1.256 Tri: 0.069 CE_rec: 1.192 AIRL_rec: 0.1357 Acc: 0.976 LR: 3.17e-05
logs/agreidv2_airl_iso.log:348:Epoch[21] Iter[250/786] Loss: 2.573 CE: 1.250 Tri: 0.068 CE_rec: 1.188 AIRL_rec: 0.1351 Acc: 0.976 LR: 3.17e-05
logs/agreidv2_airl_iso.log:349:Epoch[21] Iter[300/786] Loss: 2.571 CE: 1.249 Tri: 0.069 CE_rec: 1.186 AIRL_rec: 0.1343 Acc: 0.976 LR: 3.17e-05
logs/agreidv2_airl_iso.log:350:Epoch[21] Iter[350/786] Loss: 2.560 CE: 1.245 Tri: 0.066 CE_rec: 1.183 AIRL_rec: 0.1333 Acc: 0.977 LR: 3.17e-05
logs/agreidv2_airl_iso.log:351:Epoch[21] Iter[400/786] Loss: 2.551 CE: 1.240 Tri: 0.066 CE_rec: 1.179 AIRL_rec: 0.1325 Acc: 0.978 LR: 3.17e-05
logs/agreidv2_airl_iso.log:352:Epoch[21] Iter[450/786] Loss: 2.544 CE: 1.237 Tri: 0.065 CE_rec: 1.177 AIRL_rec: 0.1313 Acc: 0.978 LR: 3.17e-05
logs/agreidv2_airl_iso.log:353:Epoch[21] Iter[500/786] Loss: 2.538 CE: 1.234 Tri: 0.065 CE_rec: 1.174 AIRL_rec: 0.1305 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_airl_iso.log:354:Epoch[21] Iter[550/786] Loss: 2.531 CE: 1.231 Tri: 0.064 CE_rec: 1.171 AIRL_rec: 0.1291 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_airl_iso.log:355:Epoch[21] Iter[600/786] Loss: 2.523 CE: 1.227 Tri: 0.063 CE_rec: 1.169 AIRL_rec: 0.1280 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_airl_iso.log:356:Epoch[21] Iter[650/786] Loss: 2.515 CE: 1.223 Tri: 0.063 CE_rec: 1.166 AIRL_rec: 0.1266 Acc: 0.980 LR: 3.17e-05
logs/agreidv2_airl_iso.log:357:Epoch[21] Iter[700/786] Loss: 2.502 CE: 1.218 Tri: 0.061 CE_rec: 1.161 AIRL_rec: 0.1247 Acc: 0.981 LR: 3.17e-05
logs/agreidv2_airl_iso.log:358:Epoch[21] done in 231.0s  Loss=2.491 Acc=0.981 AIRL-ISO[lam_eff=0.500 ce_rec=1.156 consistency=0.1225 deg_scale_mean=0.624 n_ground=28621]
logs/agreidv2_airl_iso.log:359:Epoch[22] Iter[50/786] Loss: 2.659 CE: 1.301 Tri: 0.068 CE_rec: 1.229 AIRL_rec: 0.1197 Acc: 0.966 LR: 3.10e-05
logs/agreidv2_airl_iso.log:360:Epoch[22] Iter[100/786] Loss: 2.617 CE: 1.274 Tri: 0.069 CE_rec: 1.212 AIRL_rec: 0.1248 Acc: 0.970 LR: 3.10e-05
logs/agreidv2_airl_iso.log:361:Epoch[22] Iter[150/786] Loss: 2.587 CE: 1.259 Tri: 0.064 CE_rec: 1.200 AIRL_rec: 0.1270 Acc: 0.973 LR: 3.10e-05
logs/agreidv2_airl_iso.log:362:Epoch[22] Iter[200/786] Loss: 2.562 CE: 1.248 Tri: 0.059 CE_rec: 1.191 AIRL_rec: 0.1272 Acc: 0.975 LR: 3.10e-05
logs/agreidv2_airl_iso.log:363:Epoch[22] Iter[250/786] Loss: 2.551 CE: 1.242 Tri: 0.061 CE_rec: 1.185 AIRL_rec: 0.1269 Acc: 0.976 LR: 3.10e-05
logs/agreidv2_airl_iso.log:364:Epoch[22] Iter[300/786] Loss: 2.539 CE: 1.235 Tri: 0.062 CE_rec: 1.178 AIRL_rec: 0.1259 Acc: 0.978 LR: 3.10e-05
logs/agreidv2_airl_iso.log:365:Epoch[22] Iter[350/786] Loss: 2.528 CE: 1.231 Tri: 0.060 CE_rec: 1.174 AIRL_rec: 0.1256 Acc: 0.978 LR: 3.10e-05
logs/agreidv2_airl_iso.log:366:Epoch[22] Iter[400/786] Loss: 2.521 CE: 1.227 Tri: 0.060 CE_rec: 1.172 AIRL_rec: 0.1253 Acc: 0.979 LR: 3.10e-05
logs/agreidv2_airl_iso.log:367:Epoch[22] Iter[450/786] Loss: 2.514 CE: 1.224 Tri: 0.059 CE_rec: 1.168 AIRL_rec: 0.1245 Acc: 0.979 LR: 3.10e-05
logs/agreidv2_airl_iso.log:368:Epoch[22] Iter[500/786] Loss: 2.506 CE: 1.221 Tri: 0.058 CE_rec: 1.166 AIRL_rec: 0.1237 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_airl_iso.log:369:Epoch[22] Iter[550/786] Loss: 2.498 CE: 1.217 Tri: 0.057 CE_rec: 1.162 AIRL_rec: 0.1227 Acc: 0.981 LR: 3.10e-05
logs/agreidv2_airl_iso.log:370:Epoch[22] Iter[600/786] Loss: 2.491 CE: 1.214 Tri: 0.056 CE_rec: 1.160 AIRL_rec: 0.1216 Acc: 0.981 LR: 3.10e-05
logs/agreidv2_airl_iso.log:371:Epoch[22] Iter[650/786] Loss: 2.483 CE: 1.211 Tri: 0.055 CE_rec: 1.157 AIRL_rec: 0.1202 Acc: 0.982 LR: 3.10e-05
logs/agreidv2_airl_iso.log:372:Epoch[22] Iter[700/786] Loss: 2.475 CE: 1.207 Tri: 0.055 CE_rec: 1.153 AIRL_rec: 0.1186 Acc: 0.982 LR: 3.10e-05
logs/agreidv2_airl_iso.log:373:Epoch[22] done in 234.9s  Loss=2.462 Acc=0.983 AIRL-ISO[lam_eff=0.500 ce_rec=1.148 consistency=0.1170 deg_scale_mean=0.625 n_ground=28842]
logs/agreidv2_airl_iso.log:374:Epoch[23] Iter[50/786] Loss: 2.639 CE: 1.289 Tri: 0.071 CE_rec: 1.220 AIRL_rec: 0.1190 Acc: 0.964 LR: 3.03e-05
logs/agreidv2_airl_iso.log:375:Epoch[23] Iter[100/786] Loss: 2.606 CE: 1.266 Tri: 0.072 CE_rec: 1.206 AIRL_rec: 0.1260 Acc: 0.970 LR: 3.03e-05
logs/agreidv2_airl_iso.log:376:Epoch[23] Iter[150/786] Loss: 2.572 CE: 1.250 Tri: 0.065 CE_rec: 1.194 AIRL_rec: 0.1257 Acc: 0.973 LR: 3.03e-05
logs/agreidv2_airl_iso.log:377:Epoch[23] Iter[200/786] Loss: 2.551 CE: 1.242 Tri: 0.062 CE_rec: 1.184 AIRL_rec: 0.1249 Acc: 0.975 LR: 3.03e-05
logs/agreidv2_airl_iso.log:378:Epoch[23] Iter[250/786] Loss: 2.542 CE: 1.236 Tri: 0.065 CE_rec: 1.180 AIRL_rec: 0.1244 Acc: 0.976 LR: 3.03e-05
logs/agreidv2_airl_iso.log:379:Epoch[23] Iter[300/786] Loss: 2.531 CE: 1.231 Tri: 0.064 CE_rec: 1.175 AIRL_rec: 0.1229 Acc: 0.977 LR: 3.03e-05
logs/agreidv2_airl_iso.log:380:Epoch[23] Iter[350/786] Loss: 2.527 CE: 1.229 Tri: 0.064 CE_rec: 1.173 AIRL_rec: 0.1229 Acc: 0.978 LR: 3.03e-05
logs/agreidv2_airl_iso.log:381:Epoch[23] Iter[400/786] Loss: 2.521 CE: 1.227 Tri: 0.063 CE_rec: 1.171 AIRL_rec: 0.1216 Acc: 0.978 LR: 3.03e-05
logs/agreidv2_airl_iso.log:382:Epoch[23] Iter[450/786] Loss: 2.514 CE: 1.223 Tri: 0.063 CE_rec: 1.167 AIRL_rec: 0.1211 Acc: 0.978 LR: 3.03e-05
logs/agreidv2_airl_iso.log:383:Epoch[23] Iter[500/786] Loss: 2.506 CE: 1.220 Tri: 0.061 CE_rec: 1.165 AIRL_rec: 0.1205 Acc: 0.978 LR: 3.03e-05
logs/agreidv2_airl_iso.log:384:Epoch[23] Iter[550/786] Loss: 2.498 CE: 1.216 Tri: 0.060 CE_rec: 1.162 AIRL_rec: 0.1198 Acc: 0.979 LR: 3.03e-05
logs/agreidv2_airl_iso.log:385:Epoch[23] Iter[600/786] Loss: 2.490 CE: 1.213 Tri: 0.059 CE_rec: 1.159 AIRL_rec: 0.1187 Acc: 0.979 LR: 3.03e-05
logs/agreidv2_airl_iso.log:386:Epoch[23] Iter[650/786] Loss: 2.481 CE: 1.209 Tri: 0.057 CE_rec: 1.156 AIRL_rec: 0.1174 Acc: 0.980 LR: 3.03e-05
logs/agreidv2_airl_iso.log:387:Epoch[23] Iter[700/786] Loss: 2.470 CE: 1.205 Tri: 0.055 CE_rec: 1.152 AIRL_rec: 0.1158 Acc: 0.981 LR: 3.03e-05
logs/agreidv2_airl_iso.log:388:Epoch[23] done in 235.7s  Loss=2.457 Acc=0.981 AIRL-ISO[lam_eff=0.500 ce_rec=1.147 consistency=0.1139 deg_scale_mean=0.626 n_ground=28837]
logs/agreidv2_airl_iso.log:389:Epoch[24] Iter[50/786] Loss: 2.584 CE: 1.265 Tri: 0.060 CE_rec: 1.203 AIRL_rec: 0.1128 Acc: 0.975 LR: 2.95e-05
logs/agreidv2_airl_iso.log:390:Epoch[24] Iter[100/786] Loss: 2.541 CE: 1.242 Tri: 0.056 CE_rec: 1.185 AIRL_rec: 0.1172 Acc: 0.978 LR: 2.95e-05
logs/agreidv2_airl_iso.log:391:Epoch[24] Iter[150/786] Loss: 2.524 CE: 1.231 Tri: 0.057 CE_rec: 1.176 AIRL_rec: 0.1182 Acc: 0.979 LR: 2.95e-05
logs/agreidv2_airl_iso.log:392:Epoch[24] Iter[200/786] Loss: 2.507 CE: 1.224 Tri: 0.054 CE_rec: 1.170 AIRL_rec: 0.1181 Acc: 0.980 LR: 2.95e-05
logs/agreidv2_airl_iso.log:393:Epoch[24] Iter[250/786] Loss: 2.489 CE: 1.216 Tri: 0.051 CE_rec: 1.164 AIRL_rec: 0.1180 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_airl_iso.log:394:Epoch[24] Iter[300/786] Loss: 2.480 CE: 1.211 Tri: 0.051 CE_rec: 1.159 AIRL_rec: 0.1166 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_airl_iso.log:395:Epoch[24] Iter[350/786] Loss: 2.470 CE: 1.207 Tri: 0.050 CE_rec: 1.155 AIRL_rec: 0.1165 Acc: 0.983 LR: 2.95e-05
logs/agreidv2_airl_iso.log:396:Epoch[24] Iter[400/786] Loss: 2.461 CE: 1.203 Tri: 0.049 CE_rec: 1.151 AIRL_rec: 0.1156 Acc: 0.983 LR: 2.95e-05
logs/agreidv2_airl_iso.log:397:Epoch[24] Iter[450/786] Loss: 2.453 CE: 1.200 Tri: 0.047 CE_rec: 1.149 AIRL_rec: 0.1153 Acc: 0.984 LR: 2.95e-05
logs/agreidv2_airl_iso.log:398:Epoch[24] Iter[500/786] Loss: 2.449 CE: 1.198 Tri: 0.048 CE_rec: 1.146 AIRL_rec: 0.1147 Acc: 0.984 LR: 2.95e-05
logs/agreidv2_airl_iso.log:399:Epoch[24] Iter[550/786] Loss: 2.444 CE: 1.196 Tri: 0.048 CE_rec: 1.143 AIRL_rec: 0.1143 Acc: 0.984 LR: 2.95e-05
logs/agreidv2_airl_iso.log:400:Epoch[24] Iter[600/786] Loss: 2.441 CE: 1.194 Tri: 0.048 CE_rec: 1.142 AIRL_rec: 0.1135 Acc: 0.984 LR: 2.95e-05
logs/agreidv2_airl_iso.log:401:Epoch[24] Iter[650/786] Loss: 2.433 CE: 1.191 Tri: 0.047 CE_rec: 1.139 AIRL_rec: 0.1121 Acc: 0.985 LR: 2.95e-05
logs/agreidv2_airl_iso.log:402:Epoch[24] Iter[700/786] Loss: 2.423 CE: 1.187 Tri: 0.045 CE_rec: 1.136 AIRL_rec: 0.1103 Acc: 0.985 LR: 2.95e-05
logs/agreidv2_airl_iso.log:403:Epoch[24] done in 237.3s  Loss=2.411 Acc=0.986 AIRL-ISO[lam_eff=0.500 ce_rec=1.131 consistency=0.1082 deg_scale_mean=0.624 n_ground=28829]
logs/agreidv2_airl_iso.log:404:Epoch[25] Iter[50/786] Loss: 2.575 CE: 1.259 Tri: 0.057 CE_rec: 1.203 AIRL_rec: 0.1103 Acc: 0.976 LR: 2.87e-05
logs/agreidv2_airl_iso.log:405:Epoch[25] Iter[100/786] Loss: 2.526 CE: 1.233 Tri: 0.055 CE_rec: 1.182 AIRL_rec: 0.1101 Acc: 0.979 LR: 2.87e-05
logs/agreidv2_airl_iso.log:406:Epoch[25] Iter[150/786] Loss: 2.502 CE: 1.222 Tri: 0.053 CE_rec: 1.170 AIRL_rec: 0.1119 Acc: 0.980 LR: 2.87e-05
logs/agreidv2_airl_iso.log:407:Epoch[25] Iter[200/786] Loss: 2.486 CE: 1.213 Tri: 0.054 CE_rec: 1.162 AIRL_rec: 0.1134 Acc: 0.982 LR: 2.87e-05
logs/agreidv2_airl_iso.log:408:Epoch[25] Iter[250/786] Loss: 2.473 CE: 1.208 Tri: 0.052 CE_rec: 1.156 AIRL_rec: 0.1128 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_airl_iso.log:409:Epoch[25] Iter[300/786] Loss: 2.469 CE: 1.205 Tri: 0.053 CE_rec: 1.154 AIRL_rec: 0.1141 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_airl_iso.log:410:Epoch[25] Iter[350/786] Loss: 2.462 CE: 1.202 Tri: 0.053 CE_rec: 1.150 AIRL_rec: 0.1146 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_airl_iso.log:411:Epoch[25] Iter[400/786] Loss: 2.456 CE: 1.199 Tri: 0.052 CE_rec: 1.148 AIRL_rec: 0.1147 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_airl_iso.log:412:Epoch[25] Iter[450/786] Loss: 2.447 CE: 1.195 Tri: 0.050 CE_rec: 1.145 AIRL_rec: 0.1139 Acc: 0.984 LR: 2.87e-05
logs/agreidv2_airl_iso.log:413:Epoch[25] Iter[500/786] Loss: 2.441 CE: 1.193 Tri: 0.050 CE_rec: 1.142 AIRL_rec: 0.1125 Acc: 0.984 LR: 2.87e-05
logs/agreidv2_airl_iso.log:414:Epoch[25] Iter[550/786] Loss: 2.433 CE: 1.190 Tri: 0.049 CE_rec: 1.140 AIRL_rec: 0.1113 Acc: 0.984 LR: 2.87e-05
logs/agreidv2_airl_iso.log:415:Epoch[25] Iter[600/786] Loss: 2.428 CE: 1.187 Tri: 0.048 CE_rec: 1.137 AIRL_rec: 0.1105 Acc: 0.985 LR: 2.87e-05
logs/agreidv2_airl_iso.log:416:Epoch[25] Iter[650/786] Loss: 2.418 CE: 1.183 Tri: 0.046 CE_rec: 1.134 AIRL_rec: 0.1092 Acc: 0.985 LR: 2.87e-05
logs/agreidv2_airl_iso.log:417:Epoch[25] Iter[700/786] Loss: 2.411 CE: 1.180 Tri: 0.046 CE_rec: 1.131 AIRL_rec: 0.1079 Acc: 0.986 LR: 2.87e-05
logs/agreidv2_airl_iso.log:418:Epoch[25] done in 238.6s  Loss=2.401 Acc=0.986 AIRL-ISO[lam_eff=0.500 ce_rec=1.127 consistency=0.1061 deg_scale_mean=0.627 n_ground=28728]
logs/agreidv2_airl_iso.log:419:Epoch[26] Iter[50/786] Loss: 2.521 CE: 1.237 Tri: 0.053 CE_rec: 1.175 AIRL_rec: 0.1113 Acc: 0.979 LR: 2.78e-05
logs/agreidv2_airl_iso.log:420:Epoch[26] Iter[100/786] Loss: 2.496 CE: 1.224 Tri: 0.051 CE_rec: 1.166 AIRL_rec: 0.1103 Acc: 0.978 LR: 2.78e-05
logs/agreidv2_airl_iso.log:421:Epoch[26] Iter[150/786] Loss: 2.469 CE: 1.210 Tri: 0.047 CE_rec: 1.156 AIRL_rec: 0.1114 Acc: 0.981 LR: 2.78e-05
logs/agreidv2_airl_iso.log:422:Epoch[26] Iter[200/786] Loss: 2.457 CE: 1.204 Tri: 0.047 CE_rec: 1.150 AIRL_rec: 0.1119 Acc: 0.982 LR: 2.78e-05
logs/agreidv2_airl_iso.log:423:Epoch[26] Iter[250/786] Loss: 2.452 CE: 1.202 Tri: 0.047 CE_rec: 1.148 AIRL_rec: 0.1106 Acc: 0.982 LR: 2.78e-05
logs/agreidv2_airl_iso.log:424:Epoch[26] Iter[300/786] Loss: 2.450 CE: 1.200 Tri: 0.048 CE_rec: 1.147 AIRL_rec: 0.1111 Acc: 0.981 LR: 2.78e-05
logs/agreidv2_airl_iso.log:425:Epoch[26] Iter[350/786] Loss: 2.438 CE: 1.195 Tri: 0.045 CE_rec: 1.142 AIRL_rec: 0.1098 Acc: 0.982 LR: 2.78e-05
logs/agreidv2_airl_iso.log:426:Epoch[26] Iter[400/786] Loss: 2.427 CE: 1.191 Tri: 0.043 CE_rec: 1.138 AIRL_rec: 0.1093 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_airl_iso.log:427:Epoch[26] Iter[450/786] Loss: 2.421 CE: 1.187 Tri: 0.043 CE_rec: 1.136 AIRL_rec: 0.1086 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_airl_iso.log:428:Epoch[26] Iter[500/786] Loss: 2.419 CE: 1.186 Tri: 0.044 CE_rec: 1.135 AIRL_rec: 0.1078 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_airl_iso.log:429:Epoch[26] Iter[550/786] Loss: 2.413 CE: 1.183 Tri: 0.043 CE_rec: 1.133 AIRL_rec: 0.1067 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_airl_iso.log:430:Epoch[26] Iter[600/786] Loss: 2.407 CE: 1.181 Tri: 0.043 CE_rec: 1.131 AIRL_rec: 0.1055 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_airl_iso.log:431:Epoch[26] Iter[650/786] Loss: 2.402 CE: 1.178 Tri: 0.043 CE_rec: 1.129 AIRL_rec: 0.1042 Acc: 0.985 LR: 2.78e-05
logs/agreidv2_airl_iso.log:432:Epoch[26] Iter[700/786] Loss: 2.393 CE: 1.174 Tri: 0.042 CE_rec: 1.126 AIRL_rec: 0.1025 Acc: 0.985 LR: 2.78e-05
logs/agreidv2_airl_iso.log:433:Epoch[26] done in 239.4s  Loss=2.382 Acc=0.986 AIRL-ISO[lam_eff=0.500 ce_rec=1.122 consistency=0.1008 deg_scale_mean=0.625 n_ground=28699]
logs/agreidv2_airl_iso.log:434:Epoch[27] Iter[50/786] Loss: 2.490 CE: 1.223 Tri: 0.051 CE_rec: 1.163 AIRL_rec: 0.1067 Acc: 0.980 LR: 2.69e-05
logs/agreidv2_airl_iso.log:435:Epoch[27] Iter[100/786] Loss: 2.476 CE: 1.215 Tri: 0.049 CE_rec: 1.158 AIRL_rec: 0.1082 Acc: 0.980 LR: 2.69e-05
logs/agreidv2_airl_iso.log:436:Epoch[27] Iter[150/786] Loss: 2.458 CE: 1.204 Tri: 0.049 CE_rec: 1.151 AIRL_rec: 0.1081 Acc: 0.981 LR: 2.69e-05
logs/agreidv2_airl_iso.log:437:Epoch[27] Iter[200/786] Loss: 2.438 CE: 1.196 Tri: 0.044 CE_rec: 1.144 AIRL_rec: 0.1076 Acc: 0.983 LR: 2.69e-05
logs/agreidv2_airl_iso.log:438:Epoch[27] Iter[250/786] Loss: 2.429 CE: 1.191 Tri: 0.044 CE_rec: 1.141 AIRL_rec: 0.1069 Acc: 0.984 LR: 2.69e-05
logs/agreidv2_airl_iso.log:439:Epoch[27] Iter[300/786] Loss: 2.422 CE: 1.187 Tri: 0.044 CE_rec: 1.138 AIRL_rec: 0.1062 Acc: 0.984 LR: 2.69e-05
logs/agreidv2_airl_iso.log:440:Epoch[27] Iter[350/786] Loss: 2.413 CE: 1.182 Tri: 0.043 CE_rec: 1.134 AIRL_rec: 0.1057 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_airl_iso.log:441:Epoch[27] Iter[400/786] Loss: 2.403 CE: 1.179 Tri: 0.041 CE_rec: 1.131 AIRL_rec: 0.1046 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_airl_iso.log:442:Epoch[27] Iter[450/786] Loss: 2.396 CE: 1.176 Tri: 0.040 CE_rec: 1.128 AIRL_rec: 0.1036 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_airl_iso.log:443:Epoch[27] Iter[500/786] Loss: 2.391 CE: 1.174 Tri: 0.039 CE_rec: 1.126 AIRL_rec: 0.1029 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_airl_iso.log:444:Epoch[27] Iter[550/786] Loss: 2.391 CE: 1.173 Tri: 0.041 CE_rec: 1.126 AIRL_rec: 0.1025 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_airl_iso.log:445:Epoch[27] Iter[600/786] Loss: 2.385 CE: 1.170 Tri: 0.040 CE_rec: 1.123 AIRL_rec: 0.1020 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_airl_iso.log:446:Epoch[27] Iter[650/786] Loss: 2.378 CE: 1.167 Tri: 0.040 CE_rec: 1.120 AIRL_rec: 0.1011 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_airl_iso.log:447:Epoch[27] Iter[700/786] Loss: 2.368 CE: 1.163 Tri: 0.038 CE_rec: 1.117 AIRL_rec: 0.0997 Acc: 0.987 LR: 2.69e-05
logs/agreidv2_airl_iso.log:448:Epoch[27] done in 238.6s  Loss=2.359 Acc=0.988 AIRL-ISO[lam_eff=0.500 ce_rec=1.114 consistency=0.0982 deg_scale_mean=0.625 n_ground=28673]
logs/agreidv2_airl_iso.log:449:Epoch[28] Iter[50/786] Loss: 2.454 CE: 1.207 Tri: 0.039 CE_rec: 1.159 AIRL_rec: 0.0991 Acc: 0.981 LR: 2.59e-05
logs/agreidv2_airl_iso.log:450:Epoch[28] Iter[100/786] Loss: 2.430 CE: 1.191 Tri: 0.044 CE_rec: 1.145 AIRL_rec: 0.0996 Acc: 0.984 LR: 2.59e-05
logs/agreidv2_airl_iso.log:451:Epoch[28] Iter[150/786] Loss: 2.404 CE: 1.180 Tri: 0.039 CE_rec: 1.135 AIRL_rec: 0.1005 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_airl_iso.log:452:Epoch[28] Iter[200/786] Loss: 2.403 CE: 1.178 Tri: 0.039 CE_rec: 1.135 AIRL_rec: 0.1012 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_airl_iso.log:453:Epoch[28] Iter[250/786] Loss: 2.404 CE: 1.178 Tri: 0.040 CE_rec: 1.134 AIRL_rec: 0.1015 Acc: 0.984 LR: 2.59e-05
logs/agreidv2_airl_iso.log:454:Epoch[28] Iter[300/786] Loss: 2.396 CE: 1.176 Tri: 0.040 CE_rec: 1.131 AIRL_rec: 0.1008 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_airl_iso.log:455:Epoch[28] Iter[350/786] Loss: 2.391 CE: 1.173 Tri: 0.039 CE_rec: 1.128 AIRL_rec: 0.1003 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_airl_iso.log:456:Epoch[28] Iter[400/786] Loss: 2.387 CE: 1.171 Tri: 0.039 CE_rec: 1.127 AIRL_rec: 0.0997 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_airl_iso.log:457:Epoch[28] Iter[450/786] Loss: 2.383 CE: 1.170 Tri: 0.039 CE_rec: 1.125 AIRL_rec: 0.0991 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_airl_iso.log:458:Epoch[28] Iter[500/786] Loss: 2.378 CE: 1.168 Tri: 0.038 CE_rec: 1.123 AIRL_rec: 0.0984 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_airl_iso.log:459:Epoch[28] Iter[550/786] Loss: 2.373 CE: 1.165 Tri: 0.038 CE_rec: 1.121 AIRL_rec: 0.0976 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_airl_iso.log:460:Epoch[28] Iter[600/786] Loss: 2.369 CE: 1.163 Tri: 0.038 CE_rec: 1.119 AIRL_rec: 0.0967 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_airl_iso.log:461:Epoch[28] Iter[650/786] Loss: 2.363 CE: 1.161 Tri: 0.037 CE_rec: 1.117 AIRL_rec: 0.0958 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_airl_iso.log:462:Epoch[28] Iter[700/786] Loss: 2.355 CE: 1.157 Tri: 0.037 CE_rec: 1.114 AIRL_rec: 0.0945 Acc: 0.987 LR: 2.59e-05
logs/agreidv2_airl_iso.log:463:Epoch[28] done in 239.6s  Loss=2.346 Acc=0.987 AIRL-ISO[lam_eff=0.500 ce_rec=1.110 consistency=0.0931 deg_scale_mean=0.627 n_ground=28746]
logs/agreidv2_airl_iso.log:464:Epoch[29] Iter[50/786] Loss: 2.441 CE: 1.202 Tri: 0.036 CE_rec: 1.154 AIRL_rec: 0.0976 Acc: 0.980 LR: 2.50e-05
logs/agreidv2_airl_iso.log:465:Epoch[29] Iter[100/786] Loss: 2.414 CE: 1.188 Tri: 0.037 CE_rec: 1.141 AIRL_rec: 0.0978 Acc: 0.983 LR: 2.50e-05
logs/agreidv2_airl_iso.log:466:Epoch[29] Iter[150/786] Loss: 2.402 CE: 1.180 Tri: 0.039 CE_rec: 1.134 AIRL_rec: 0.0979 Acc: 0.985 LR: 2.50e-05
logs/agreidv2_airl_iso.log:467:Epoch[29] Iter[200/786] Loss: 2.395 CE: 1.176 Tri: 0.039 CE_rec: 1.132 AIRL_rec: 0.0984 Acc: 0.985 LR: 2.50e-05
logs/agreidv2_airl_iso.log:468:Epoch[29] Iter[250/786] Loss: 2.384 CE: 1.171 Tri: 0.037 CE_rec: 1.127 AIRL_rec: 0.0979 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_airl_iso.log:469:Epoch[29] Iter[300/786] Loss: 2.374 CE: 1.167 Tri: 0.035 CE_rec: 1.123 AIRL_rec: 0.0978 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_airl_iso.log:470:Epoch[29] Iter[350/786] Loss: 2.371 CE: 1.165 Tri: 0.035 CE_rec: 1.122 AIRL_rec: 0.0979 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_airl_iso.log:471:Epoch[29] Iter[400/786] Loss: 2.365 CE: 1.162 Tri: 0.034 CE_rec: 1.121 AIRL_rec: 0.0972 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_airl_iso.log:472:Epoch[29] Iter[450/786] Loss: 2.361 CE: 1.161 Tri: 0.033 CE_rec: 1.119 AIRL_rec: 0.0968 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_airl_iso.log:473:Epoch[29] Iter[500/786] Loss: 2.356 CE: 1.159 Tri: 0.033 CE_rec: 1.117 AIRL_rec: 0.0964 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_airl_iso.log:474:Epoch[29] Iter[550/786] Loss: 2.350 CE: 1.156 Tri: 0.032 CE_rec: 1.115 AIRL_rec: 0.0958 Acc: 0.988 LR: 2.50e-05
logs/agreidv2_airl_iso.log:475:Epoch[29] Iter[600/786] Loss: 2.344 CE: 1.154 Tri: 0.031 CE_rec: 1.112 AIRL_rec: 0.0950 Acc: 0.988 LR: 2.50e-05
logs/agreidv2_airl_iso.log:476:Epoch[29] Iter[650/786] Loss: 2.337 CE: 1.150 Tri: 0.030 CE_rec: 1.109 AIRL_rec: 0.0939 Acc: 0.988 LR: 2.50e-05
logs/agreidv2_airl_iso.log:477:Epoch[29] Iter[700/786] Loss: 2.329 CE: 1.147 Tri: 0.030 CE_rec: 1.106 AIRL_rec: 0.0923 Acc: 0.989 LR: 2.50e-05
logs/agreidv2_airl_iso.log:478:Epoch[29] done in 240.6s  Loss=2.319 Acc=0.989 AIRL-ISO[lam_eff=0.500 ce_rec=1.103 consistency=0.0903 deg_scale_mean=0.624 n_ground=28853]
logs/agreidv2_airl_iso.log:479:Epoch[30] Iter[50/786] Loss: 2.429 CE: 1.193 Tri: 0.046 CE_rec: 1.144 AIRL_rec: 0.0907 Acc: 0.984 LR: 2.39e-05
logs/agreidv2_airl_iso.log:480:Epoch[30] Iter[100/786] Loss: 2.395 CE: 1.175 Tri: 0.042 CE_rec: 1.132 AIRL_rec: 0.0930 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_airl_iso.log:481:Epoch[30] Iter[150/786] Loss: 2.376 CE: 1.168 Tri: 0.036 CE_rec: 1.125 AIRL_rec: 0.0942 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_airl_iso.log:482:Epoch[30] Iter[200/786] Loss: 2.362 CE: 1.162 Tri: 0.034 CE_rec: 1.118 AIRL_rec: 0.0941 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_airl_iso.log:483:Epoch[30] Iter[250/786] Loss: 2.355 CE: 1.159 Tri: 0.035 CE_rec: 1.114 AIRL_rec: 0.0940 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_airl_iso.log:484:Epoch[30] Iter[300/786] Loss: 2.343 CE: 1.154 Tri: 0.032 CE_rec: 1.110 AIRL_rec: 0.0932 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_airl_iso.log:485:Epoch[30] Iter[350/786] Loss: 2.337 CE: 1.151 Tri: 0.031 CE_rec: 1.109 AIRL_rec: 0.0925 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_airl_iso.log:486:Epoch[30] Iter[400/786] Loss: 2.335 CE: 1.150 Tri: 0.031 CE_rec: 1.107 AIRL_rec: 0.0921 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_airl_iso.log:487:Epoch[30] Iter[450/786] Loss: 2.334 CE: 1.150 Tri: 0.033 CE_rec: 1.106 AIRL_rec: 0.0916 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_airl_iso.log:488:Epoch[30] Iter[500/786] Loss: 2.333 CE: 1.149 Tri: 0.033 CE_rec: 1.106 AIRL_rec: 0.0912 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_airl_iso.log:489:Epoch[30] Iter[550/786] Loss: 2.330 CE: 1.147 Tri: 0.033 CE_rec: 1.105 AIRL_rec: 0.0906 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_airl_iso.log:490:Epoch[30] Iter[600/786] Loss: 2.326 CE: 1.145 Tri: 0.032 CE_rec: 1.104 AIRL_rec: 0.0898 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_airl_iso.log:491:Epoch[30] Iter[650/786] Loss: 2.320 CE: 1.143 Tri: 0.031 CE_rec: 1.102 AIRL_rec: 0.0888 Acc: 0.989 LR: 2.39e-05
logs/agreidv2_airl_iso.log:492:Epoch[30] Iter[700/786] Loss: 2.312 CE: 1.139 Tri: 0.030 CE_rec: 1.099 AIRL_rec: 0.0874 Acc: 0.990 LR: 2.39e-05
logs/agreidv2_airl_iso.log:493:Epoch[30] done in 240.9s  Loss=2.302 Acc=0.990 AIRL-ISO[lam_eff=0.500 ce_rec=1.095 consistency=0.0856 deg_scale_mean=0.625 n_ground=28873]
logs/agreidv2_airl_iso.log:495:    [A->G] mAP=73.46  R1=82.77  R5=89.60  mINP=44.99
logs/agreidv2_airl_iso.log:496:    [G->A] mAP=73.80  R1=82.50  R5=88.90  mINP=37.87
logs/agreidv2_airl_iso.log:497:    [mean] mAP=73.63  R1=82.63
logs/agreidv2_airl_iso.log:498:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 30 ----
logs/agreidv2_airl_iso.log:499:    [A->G] full mAP=73.46 R1=82.77 | rec mAP=73.34 R1=82.00 | FUSE mAP=74.80 R1=83.36
logs/agreidv2_airl_iso.log:500:    [G->A] full mAP=73.80 R1=82.50 | rec mAP=73.06 R1=82.44 | FUSE mAP=75.26 R1=83.55
logs/agreidv2_airl_iso.log:502:    * new best mean mAP=75.03 (epoch 30) saved
logs/agreidv2_airl_iso.log:503:Epoch[31] Iter[50/786] Loss: 2.378 CE: 1.180 Tri: 0.029 CE_rec: 1.124 AIRL_rec: 0.0898 Acc: 0.983 LR: 2.29e-05
logs/agreidv2_airl_iso.log:504:Epoch[31] Iter[100/786] Loss: 2.372 CE: 1.171 Tri: 0.032 CE_rec: 1.122 AIRL_rec: 0.0920 Acc: 0.984 LR: 2.29e-05
logs/agreidv2_airl_iso.log:505:Epoch[31] Iter[150/786] Loss: 2.368 CE: 1.167 Tri: 0.033 CE_rec: 1.122 AIRL_rec: 0.0916 Acc: 0.985 LR: 2.29e-05
logs/agreidv2_airl_iso.log:506:Epoch[31] Iter[200/786] Loss: 2.367 CE: 1.166 Tri: 0.035 CE_rec: 1.120 AIRL_rec: 0.0924 Acc: 0.985 LR: 2.29e-05
logs/agreidv2_airl_iso.log:507:Epoch[31] Iter[250/786] Loss: 2.358 CE: 1.161 Tri: 0.035 CE_rec: 1.116 AIRL_rec: 0.0924 Acc: 0.986 LR: 2.29e-05
logs/agreidv2_airl_iso.log:508:Epoch[31] Iter[300/786] Loss: 2.352 CE: 1.158 Tri: 0.035 CE_rec: 1.113 AIRL_rec: 0.0913 Acc: 0.986 LR: 2.29e-05
logs/agreidv2_airl_iso.log:509:Epoch[31] Iter[350/786] Loss: 2.346 CE: 1.155 Tri: 0.034 CE_rec: 1.111 AIRL_rec: 0.0913 Acc: 0.987 LR: 2.29e-05
logs/agreidv2_airl_iso.log:510:Epoch[31] Iter[400/786] Loss: 2.338 CE: 1.152 Tri: 0.033 CE_rec: 1.108 AIRL_rec: 0.0907 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_airl_iso.log:511:Epoch[31] Iter[450/786] Loss: 2.331 CE: 1.149 Tri: 0.031 CE_rec: 1.106 AIRL_rec: 0.0900 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_airl_iso.log:512:Epoch[31] Iter[500/786] Loss: 2.324 CE: 1.146 Tri: 0.031 CE_rec: 1.103 AIRL_rec: 0.0891 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_airl_iso.log:513:Epoch[31] Iter[550/786] Loss: 2.320 CE: 1.144 Tri: 0.030 CE_rec: 1.102 AIRL_rec: 0.0885 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_airl_iso.log:514:Epoch[31] Iter[600/786] Loss: 2.316 CE: 1.142 Tri: 0.030 CE_rec: 1.101 AIRL_rec: 0.0875 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_airl_iso.log:515:Epoch[31] Iter[650/786] Loss: 2.312 CE: 1.140 Tri: 0.030 CE_rec: 1.099 AIRL_rec: 0.0864 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_airl_iso.log:516:Epoch[31] Iter[700/786] Loss: 2.306 CE: 1.137 Tri: 0.029 CE_rec: 1.097 AIRL_rec: 0.0853 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_airl_iso.log:517:Epoch[31] done in 232.8s  Loss=2.297 Acc=0.989 AIRL-ISO[lam_eff=0.500 ce_rec=1.093 consistency=0.0839 deg_scale_mean=0.624 n_ground=28809]
logs/agreidv2_airl_iso.log:518:Epoch[32] Iter[50/786] Loss: 2.406 CE: 1.184 Tri: 0.039 CE_rec: 1.137 AIRL_rec: 0.0912 Acc: 0.982 LR: 2.19e-05
logs/agreidv2_airl_iso.log:519:Epoch[32] Iter[100/786] Loss: 2.374 CE: 1.168 Tri: 0.036 CE_rec: 1.124 AIRL_rec: 0.0914 Acc: 0.985 LR: 2.19e-05
logs/agreidv2_airl_iso.log:520:Epoch[32] Iter[150/786] Loss: 2.354 CE: 1.159 Tri: 0.034 CE_rec: 1.116 AIRL_rec: 0.0897 Acc: 0.987 LR: 2.19e-05
logs/agreidv2_airl_iso.log:521:Epoch[32] Iter[200/786] Loss: 2.343 CE: 1.154 Tri: 0.034 CE_rec: 1.111 AIRL_rec: 0.0889 Acc: 0.988 LR: 2.19e-05
logs/agreidv2_airl_iso.log:522:Epoch[32] Iter[250/786] Loss: 2.332 CE: 1.149 Tri: 0.031 CE_rec: 1.108 AIRL_rec: 0.0882 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_airl_iso.log:523:Epoch[32] Iter[300/786] Loss: 2.320 CE: 1.145 Tri: 0.027 CE_rec: 1.104 AIRL_rec: 0.0873 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_airl_iso.log:524:Epoch[32] Iter[350/786] Loss: 2.311 CE: 1.141 Tri: 0.026 CE_rec: 1.101 AIRL_rec: 0.0866 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_airl_iso.log:525:Epoch[32] Iter[400/786] Loss: 2.307 CE: 1.139 Tri: 0.025 CE_rec: 1.099 AIRL_rec: 0.0859 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_airl_iso.log:526:Epoch[32] Iter[450/786] Loss: 2.304 CE: 1.137 Tri: 0.026 CE_rec: 1.098 AIRL_rec: 0.0853 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_airl_iso.log:527:Epoch[32] Iter[500/786] Loss: 2.298 CE: 1.135 Tri: 0.026 CE_rec: 1.095 AIRL_rec: 0.0847 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_airl_iso.log:528:Epoch[32] Iter[550/786] Loss: 2.291 CE: 1.132 Tri: 0.025 CE_rec: 1.093 AIRL_rec: 0.0839 Acc: 0.991 LR: 2.19e-05
logs/agreidv2_airl_iso.log:529:Epoch[32] Iter[600/786] Loss: 2.287 CE: 1.130 Tri: 0.024 CE_rec: 1.091 AIRL_rec: 0.0832 Acc: 0.991 LR: 2.19e-05
logs/agreidv2_airl_iso.log:530:Epoch[32] Iter[650/786] Loss: 2.280 CE: 1.127 Tri: 0.023 CE_rec: 1.089 AIRL_rec: 0.0819 Acc: 0.991 LR: 2.19e-05
logs/agreidv2_airl_iso.log:531:Epoch[32] Iter[700/786] Loss: 2.275 CE: 1.125 Tri: 0.023 CE_rec: 1.087 AIRL_rec: 0.0808 Acc: 0.991 LR: 2.19e-05
logs/agreidv2_airl_iso.log:532:Epoch[32] done in 234.3s  Loss=2.267 Acc=0.992 AIRL-ISO[lam_eff=0.500 ce_rec=1.084 consistency=0.0797 deg_scale_mean=0.624 n_ground=28779]
logs/agreidv2_airl_iso.log:533:Epoch[33] Iter[50/786] Loss: 2.373 CE: 1.176 Tri: 0.029 CE_rec: 1.125 AIRL_rec: 0.0857 Acc: 0.985 LR: 2.08e-05
logs/agreidv2_airl_iso.log:534:Epoch[33] Iter[100/786] Loss: 2.334 CE: 1.156 Tri: 0.024 CE_rec: 1.112 AIRL_rec: 0.0850 Acc: 0.987 LR: 2.08e-05
logs/agreidv2_airl_iso.log:535:Epoch[33] Iter[150/786] Loss: 2.336 CE: 1.152 Tri: 0.029 CE_rec: 1.111 AIRL_rec: 0.0876 Acc: 0.987 LR: 2.08e-05
logs/agreidv2_airl_iso.log:536:Epoch[33] Iter[200/786] Loss: 2.325 CE: 1.146 Tri: 0.029 CE_rec: 1.106 AIRL_rec: 0.0869 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_airl_iso.log:537:Epoch[33] Iter[250/786] Loss: 2.319 CE: 1.144 Tri: 0.027 CE_rec: 1.104 AIRL_rec: 0.0875 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_airl_iso.log:538:Epoch[33] Iter[300/786] Loss: 2.312 CE: 1.141 Tri: 0.026 CE_rec: 1.101 AIRL_rec: 0.0876 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_airl_iso.log:539:Epoch[33] Iter[350/786] Loss: 2.307 CE: 1.139 Tri: 0.026 CE_rec: 1.099 AIRL_rec: 0.0871 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_airl_iso.log:540:Epoch[33] Iter[400/786] Loss: 2.302 CE: 1.136 Tri: 0.025 CE_rec: 1.097 AIRL_rec: 0.0861 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_airl_iso.log:541:Epoch[33] Iter[450/786] Loss: 2.297 CE: 1.134 Tri: 0.025 CE_rec: 1.095 AIRL_rec: 0.0854 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_airl_iso.log:542:Epoch[33] Iter[500/786] Loss: 2.292 CE: 1.132 Tri: 0.024 CE_rec: 1.093 AIRL_rec: 0.0845 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_airl_iso.log:543:Epoch[33] Iter[550/786] Loss: 2.288 CE: 1.130 Tri: 0.024 CE_rec: 1.092 AIRL_rec: 0.0834 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_airl_iso.log:544:Epoch[33] Iter[600/786] Loss: 2.285 CE: 1.128 Tri: 0.024 CE_rec: 1.091 AIRL_rec: 0.0825 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_airl_iso.log:545:Epoch[33] Iter[650/786] Loss: 2.281 CE: 1.126 Tri: 0.024 CE_rec: 1.089 AIRL_rec: 0.0813 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_airl_iso.log:546:Epoch[33] Iter[700/786] Loss: 2.274 CE: 1.123 Tri: 0.023 CE_rec: 1.087 AIRL_rec: 0.0801 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_airl_iso.log:547:Epoch[33] done in 235.0s  Loss=2.267 Acc=0.990 AIRL-ISO[lam_eff=0.500 ce_rec=1.084 consistency=0.0788 deg_scale_mean=0.628 n_ground=28718]
logs/agreidv2_airl_iso.log:548:Epoch[34] Iter[50/786] Loss: 2.342 CE: 1.160 Tri: 0.031 CE_rec: 1.109 AIRL_rec: 0.0833 Acc: 0.986 LR: 1.97e-05
logs/agreidv2_airl_iso.log:549:Epoch[34] Iter[100/786] Loss: 2.319 CE: 1.147 Tri: 0.023 CE_rec: 1.106 AIRL_rec: 0.0853 Acc: 0.988 LR: 1.97e-05
logs/agreidv2_airl_iso.log:550:Epoch[34] Iter[150/786] Loss: 2.304 CE: 1.140 Tri: 0.020 CE_rec: 1.101 AIRL_rec: 0.0860 Acc: 0.988 LR: 1.97e-05
logs/agreidv2_airl_iso.log:551:Epoch[34] Iter[200/786] Loss: 2.290 CE: 1.134 Tri: 0.018 CE_rec: 1.095 AIRL_rec: 0.0854 Acc: 0.989 LR: 1.97e-05
logs/agreidv2_airl_iso.log:552:Epoch[34] Iter[250/786] Loss: 2.281 CE: 1.130 Tri: 0.017 CE_rec: 1.092 AIRL_rec: 0.0842 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_airl_iso.log:553:Epoch[34] Iter[300/786] Loss: 2.277 CE: 1.128 Tri: 0.017 CE_rec: 1.090 AIRL_rec: 0.0837 Acc: 0.989 LR: 1.97e-05
logs/agreidv2_airl_iso.log:554:Epoch[34] Iter[350/786] Loss: 2.272 CE: 1.126 Tri: 0.017 CE_rec: 1.088 AIRL_rec: 0.0827 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_airl_iso.log:555:Epoch[34] Iter[400/786] Loss: 2.269 CE: 1.124 Tri: 0.018 CE_rec: 1.087 AIRL_rec: 0.0816 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_airl_iso.log:556:Epoch[34] Iter[450/786] Loss: 2.268 CE: 1.123 Tri: 0.019 CE_rec: 1.086 AIRL_rec: 0.0811 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_airl_iso.log:557:Epoch[34] Iter[500/786] Loss: 2.264 CE: 1.121 Tri: 0.018 CE_rec: 1.085 AIRL_rec: 0.0803 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_airl_iso.log:558:Epoch[34] Iter[550/786] Loss: 2.261 CE: 1.119 Tri: 0.018 CE_rec: 1.084 AIRL_rec: 0.0798 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_airl_iso.log:559:Epoch[34] Iter[600/786] Loss: 2.258 CE: 1.118 Tri: 0.018 CE_rec: 1.082 AIRL_rec: 0.0787 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_airl_iso.log:560:Epoch[34] Iter[650/786] Loss: 2.254 CE: 1.116 Tri: 0.018 CE_rec: 1.081 AIRL_rec: 0.0778 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_airl_iso.log:561:Epoch[34] Iter[700/786] Loss: 2.247 CE: 1.113 Tri: 0.018 CE_rec: 1.079 AIRL_rec: 0.0766 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_airl_iso.log:562:Epoch[34] done in 235.6s  Loss=2.241 Acc=0.992 AIRL-ISO[lam_eff=0.500 ce_rec=1.076 consistency=0.0752 deg_scale_mean=0.626 n_ground=28692]
logs/agreidv2_airl_iso.log:563:Epoch[35] Iter[50/786] Loss: 2.298 CE: 1.134 Tri: 0.023 CE_rec: 1.101 AIRL_rec: 0.0793 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_iso.log:564:Epoch[35] Iter[100/786] Loss: 2.283 CE: 1.128 Tri: 0.022 CE_rec: 1.092 AIRL_rec: 0.0806 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_iso.log:565:Epoch[35] Iter[150/786] Loss: 2.278 CE: 1.126 Tri: 0.022 CE_rec: 1.089 AIRL_rec: 0.0798 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_iso.log:566:Epoch[35] Iter[200/786] Loss: 2.276 CE: 1.125 Tri: 0.023 CE_rec: 1.088 AIRL_rec: 0.0807 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_iso.log:567:Epoch[35] Iter[250/786] Loss: 2.273 CE: 1.123 Tri: 0.022 CE_rec: 1.087 AIRL_rec: 0.0802 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_iso.log:568:Epoch[35] Iter[300/786] Loss: 2.268 CE: 1.121 Tri: 0.021 CE_rec: 1.086 AIRL_rec: 0.0796 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_iso.log:569:Epoch[35] Iter[350/786] Loss: 2.266 CE: 1.120 Tri: 0.022 CE_rec: 1.085 AIRL_rec: 0.0793 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_iso.log:570:Epoch[35] Iter[400/786] Loss: 2.262 CE: 1.118 Tri: 0.021 CE_rec: 1.084 AIRL_rec: 0.0789 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_iso.log:571:Epoch[35] Iter[450/786] Loss: 2.259 CE: 1.117 Tri: 0.020 CE_rec: 1.083 AIRL_rec: 0.0778 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_iso.log:572:Epoch[35] Iter[500/786] Loss: 2.256 CE: 1.116 Tri: 0.021 CE_rec: 1.082 AIRL_rec: 0.0772 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_iso.log:573:Epoch[35] Iter[550/786] Loss: 2.254 CE: 1.114 Tri: 0.021 CE_rec: 1.080 AIRL_rec: 0.0765 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_iso.log:574:Epoch[35] Iter[600/786] Loss: 2.248 CE: 1.112 Tri: 0.020 CE_rec: 1.078 AIRL_rec: 0.0753 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_iso.log:575:Epoch[35] Iter[650/786] Loss: 2.243 CE: 1.110 Tri: 0.019 CE_rec: 1.077 AIRL_rec: 0.0743 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_iso.log:576:Epoch[35] Iter[700/786] Loss: 2.239 CE: 1.108 Tri: 0.019 CE_rec: 1.075 AIRL_rec: 0.0733 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_iso.log:577:Epoch[35] done in 238.5s  Loss=2.232 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.072 consistency=0.0722 deg_scale_mean=0.626 n_ground=28770]
logs/agreidv2_airl_iso.log:578:Epoch[36] Iter[50/786] Loss: 2.292 CE: 1.134 Tri: 0.024 CE_rec: 1.096 AIRL_rec: 0.0751 Acc: 0.986 LR: 1.75e-05
logs/agreidv2_airl_iso.log:579:Epoch[36] Iter[100/786] Loss: 2.282 CE: 1.127 Tri: 0.025 CE_rec: 1.092 AIRL_rec: 0.0744 Acc: 0.989 LR: 1.75e-05
logs/agreidv2_airl_iso.log:580:Epoch[36] Iter[150/786] Loss: 2.281 CE: 1.126 Tri: 0.026 CE_rec: 1.092 AIRL_rec: 0.0757 Acc: 0.990 LR: 1.75e-05
logs/agreidv2_airl_iso.log:581:Epoch[36] Iter[200/786] Loss: 2.277 CE: 1.123 Tri: 0.028 CE_rec: 1.088 AIRL_rec: 0.0753 Acc: 0.990 LR: 1.75e-05
logs/agreidv2_airl_iso.log:582:Epoch[36] Iter[250/786] Loss: 2.267 CE: 1.119 Tri: 0.025 CE_rec: 1.086 AIRL_rec: 0.0745 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_airl_iso.log:583:Epoch[36] Iter[300/786] Loss: 2.262 CE: 1.117 Tri: 0.023 CE_rec: 1.084 AIRL_rec: 0.0744 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_airl_iso.log:584:Epoch[36] Iter[350/786] Loss: 2.257 CE: 1.115 Tri: 0.022 CE_rec: 1.082 AIRL_rec: 0.0743 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_airl_iso.log:585:Epoch[36] Iter[400/786] Loss: 2.252 CE: 1.114 Tri: 0.021 CE_rec: 1.080 AIRL_rec: 0.0740 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_airl_iso.log:586:Epoch[36] Iter[450/786] Loss: 2.247 CE: 1.112 Tri: 0.020 CE_rec: 1.079 AIRL_rec: 0.0736 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_airl_iso.log:587:Epoch[36] Iter[500/786] Loss: 2.243 CE: 1.110 Tri: 0.019 CE_rec: 1.077 AIRL_rec: 0.0734 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_airl_iso.log:588:Epoch[36] Iter[550/786] Loss: 2.238 CE: 1.108 Tri: 0.018 CE_rec: 1.076 AIRL_rec: 0.0727 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_airl_iso.log:589:Epoch[36] Iter[600/786] Loss: 2.234 CE: 1.106 Tri: 0.018 CE_rec: 1.074 AIRL_rec: 0.0719 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_airl_iso.log:590:Epoch[36] Iter[650/786] Loss: 2.231 CE: 1.104 Tri: 0.018 CE_rec: 1.073 AIRL_rec: 0.0712 Acc: 0.993 LR: 1.75e-05
logs/agreidv2_airl_iso.log:591:Epoch[36] Iter[700/786] Loss: 2.226 CE: 1.102 Tri: 0.018 CE_rec: 1.071 AIRL_rec: 0.0702 Acc: 0.993 LR: 1.75e-05
logs/agreidv2_airl_iso.log:592:Epoch[36] done in 239.3s  Loss=2.219 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.069 consistency=0.0690 deg_scale_mean=0.627 n_ground=28773]
logs/agreidv2_airl_iso.log:593:Epoch[37] Iter[50/786] Loss: 2.283 CE: 1.132 Tri: 0.020 CE_rec: 1.092 AIRL_rec: 0.0763 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_airl_iso.log:594:Epoch[37] Iter[100/786] Loss: 2.272 CE: 1.125 Tri: 0.021 CE_rec: 1.088 AIRL_rec: 0.0763 Acc: 0.990 LR: 1.64e-05
logs/agreidv2_airl_iso.log:595:Epoch[37] Iter[150/786] Loss: 2.263 CE: 1.121 Tri: 0.020 CE_rec: 1.085 AIRL_rec: 0.0753 Acc: 0.990 LR: 1.64e-05
logs/agreidv2_airl_iso.log:596:Epoch[37] Iter[200/786] Loss: 2.256 CE: 1.117 Tri: 0.020 CE_rec: 1.083 AIRL_rec: 0.0740 Acc: 0.990 LR: 1.64e-05
logs/agreidv2_airl_iso.log:597:Epoch[37] Iter[250/786] Loss: 2.253 CE: 1.115 Tri: 0.020 CE_rec: 1.081 AIRL_rec: 0.0740 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_airl_iso.log:598:Epoch[37] Iter[300/786] Loss: 2.251 CE: 1.113 Tri: 0.020 CE_rec: 1.081 AIRL_rec: 0.0737 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_airl_iso.log:599:Epoch[37] Iter[350/786] Loss: 2.246 CE: 1.111 Tri: 0.019 CE_rec: 1.079 AIRL_rec: 0.0734 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_airl_iso.log:600:Epoch[37] Iter[400/786] Loss: 2.241 CE: 1.109 Tri: 0.018 CE_rec: 1.077 AIRL_rec: 0.0730 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_airl_iso.log:601:Epoch[37] Iter[450/786] Loss: 2.239 CE: 1.108 Tri: 0.018 CE_rec: 1.076 AIRL_rec: 0.0728 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_airl_iso.log:602:Epoch[37] Iter[500/786] Loss: 2.235 CE: 1.106 Tri: 0.018 CE_rec: 1.075 AIRL_rec: 0.0727 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_airl_iso.log:603:Epoch[37] Iter[550/786] Loss: 2.233 CE: 1.105 Tri: 0.018 CE_rec: 1.074 AIRL_rec: 0.0722 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_airl_iso.log:604:Epoch[37] Iter[600/786] Loss: 2.230 CE: 1.103 Tri: 0.018 CE_rec: 1.073 AIRL_rec: 0.0715 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_airl_iso.log:605:Epoch[37] Iter[650/786] Loss: 2.225 CE: 1.101 Tri: 0.017 CE_rec: 1.071 AIRL_rec: 0.0705 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_airl_iso.log:606:Epoch[37] Iter[700/786] Loss: 2.220 CE: 1.099 Tri: 0.017 CE_rec: 1.070 AIRL_rec: 0.0695 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_airl_iso.log:607:Epoch[37] done in 240.1s  Loss=2.214 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.067 consistency=0.0682 deg_scale_mean=0.624 n_ground=28831]
logs/agreidv2_airl_iso.log:608:Epoch[38] Iter[50/786] Loss: 2.254 CE: 1.119 Tri: 0.013 CE_rec: 1.085 AIRL_rec: 0.0731 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_airl_iso.log:609:Epoch[38] Iter[100/786] Loss: 2.250 CE: 1.115 Tri: 0.016 CE_rec: 1.081 AIRL_rec: 0.0761 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_airl_iso.log:610:Epoch[38] Iter[150/786] Loss: 2.244 CE: 1.112 Tri: 0.016 CE_rec: 1.078 AIRL_rec: 0.0750 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_airl_iso.log:611:Epoch[38] Iter[200/786] Loss: 2.243 CE: 1.111 Tri: 0.016 CE_rec: 1.078 AIRL_rec: 0.0757 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_airl_iso.log:612:Epoch[38] Iter[250/786] Loss: 2.237 CE: 1.108 Tri: 0.014 CE_rec: 1.077 AIRL_rec: 0.0749 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_airl_iso.log:613:Epoch[38] Iter[300/786] Loss: 2.233 CE: 1.106 Tri: 0.015 CE_rec: 1.075 AIRL_rec: 0.0742 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_airl_iso.log:614:Epoch[38] Iter[350/786] Loss: 2.227 CE: 1.104 Tri: 0.014 CE_rec: 1.073 AIRL_rec: 0.0733 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_airl_iso.log:615:Epoch[38] Iter[400/786] Loss: 2.223 CE: 1.102 Tri: 0.014 CE_rec: 1.071 AIRL_rec: 0.0724 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_airl_iso.log:616:Epoch[38] Iter[450/786] Loss: 2.222 CE: 1.101 Tri: 0.014 CE_rec: 1.071 AIRL_rec: 0.0720 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_airl_iso.log:617:Epoch[38] Iter[500/786] Loss: 2.217 CE: 1.099 Tri: 0.014 CE_rec: 1.069 AIRL_rec: 0.0708 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_airl_iso.log:618:Epoch[38] Iter[550/786] Loss: 2.213 CE: 1.097 Tri: 0.013 CE_rec: 1.067 AIRL_rec: 0.0703 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_airl_iso.log:619:Epoch[38] Iter[600/786] Loss: 2.210 CE: 1.096 Tri: 0.013 CE_rec: 1.066 AIRL_rec: 0.0696 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_airl_iso.log:620:Epoch[38] Iter[650/786] Loss: 2.207 CE: 1.095 Tri: 0.013 CE_rec: 1.065 AIRL_rec: 0.0689 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_airl_iso.log:621:Epoch[38] Iter[700/786] Loss: 2.203 CE: 1.093 Tri: 0.013 CE_rec: 1.064 AIRL_rec: 0.0681 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_airl_iso.log:622:Epoch[38] done in 240.2s  Loss=2.198 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.062 consistency=0.0669 deg_scale_mean=0.627 n_ground=28802]
logs/agreidv2_airl_iso.log:623:Epoch[39] Iter[50/786] Loss: 2.259 CE: 1.121 Tri: 0.016 CE_rec: 1.086 AIRL_rec: 0.0706 Acc: 0.988 LR: 1.42e-05
logs/agreidv2_airl_iso.log:624:Epoch[39] Iter[100/786] Loss: 2.238 CE: 1.112 Tri: 0.014 CE_rec: 1.077 AIRL_rec: 0.0709 Acc: 0.990 LR: 1.42e-05
logs/agreidv2_airl_iso.log:625:Epoch[39] Iter[150/786] Loss: 2.232 CE: 1.107 Tri: 0.015 CE_rec: 1.074 AIRL_rec: 0.0712 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_iso.log:626:Epoch[39] Iter[200/786] Loss: 2.229 CE: 1.105 Tri: 0.015 CE_rec: 1.073 AIRL_rec: 0.0710 Acc: 0.991 LR: 1.42e-05
logs/agreidv2_airl_iso.log:627:Epoch[39] Iter[250/786] Loss: 2.227 CE: 1.104 Tri: 0.016 CE_rec: 1.072 AIRL_rec: 0.0707 Acc: 0.991 LR: 1.42e-05
logs/agreidv2_airl_iso.log:628:Epoch[39] Iter[300/786] Loss: 2.222 CE: 1.101 Tri: 0.016 CE_rec: 1.070 AIRL_rec: 0.0700 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_iso.log:629:Epoch[39] Iter[350/786] Loss: 2.218 CE: 1.100 Tri: 0.015 CE_rec: 1.068 AIRL_rec: 0.0696 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_iso.log:630:Epoch[39] Iter[400/786] Loss: 2.215 CE: 1.098 Tri: 0.015 CE_rec: 1.067 AIRL_rec: 0.0691 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_iso.log:631:Epoch[39] Iter[450/786] Loss: 2.215 CE: 1.098 Tri: 0.016 CE_rec: 1.067 AIRL_rec: 0.0690 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_iso.log:632:Epoch[39] Iter[500/786] Loss: 2.214 CE: 1.097 Tri: 0.016 CE_rec: 1.066 AIRL_rec: 0.0687 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_iso.log:633:Epoch[39] Iter[550/786] Loss: 2.211 CE: 1.096 Tri: 0.016 CE_rec: 1.066 AIRL_rec: 0.0680 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_iso.log:634:Epoch[39] Iter[600/786] Loss: 2.208 CE: 1.095 Tri: 0.015 CE_rec: 1.065 AIRL_rec: 0.0674 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_airl_iso.log:635:Epoch[39] Iter[650/786] Loss: 2.204 CE: 1.093 Tri: 0.015 CE_rec: 1.063 AIRL_rec: 0.0667 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_airl_iso.log:636:Epoch[39] Iter[700/786] Loss: 2.200 CE: 1.091 Tri: 0.015 CE_rec: 1.062 AIRL_rec: 0.0658 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_airl_iso.log:637:Epoch[39] done in 239.2s  Loss=2.194 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.060 consistency=0.0646 deg_scale_mean=0.625 n_ground=28792]
logs/agreidv2_airl_iso.log:638:Epoch[40] Iter[50/786] Loss: 2.268 CE: 1.118 Tri: 0.031 CE_rec: 1.085 AIRL_rec: 0.0692 Acc: 0.991 LR: 1.31e-05
logs/agreidv2_airl_iso.log:639:Epoch[40] Iter[100/786] Loss: 2.237 CE: 1.106 Tri: 0.022 CE_rec: 1.074 AIRL_rec: 0.0695 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_airl_iso.log:640:Epoch[40] Iter[150/786] Loss: 2.221 CE: 1.100 Tri: 0.018 CE_rec: 1.069 AIRL_rec: 0.0689 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_airl_iso.log:641:Epoch[40] Iter[200/786] Loss: 2.216 CE: 1.097 Tri: 0.017 CE_rec: 1.067 AIRL_rec: 0.0683 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_airl_iso.log:642:Epoch[40] Iter[250/786] Loss: 2.212 CE: 1.095 Tri: 0.016 CE_rec: 1.066 AIRL_rec: 0.0679 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_airl_iso.log:643:Epoch[40] Iter[300/786] Loss: 2.210 CE: 1.095 Tri: 0.016 CE_rec: 1.065 AIRL_rec: 0.0678 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_airl_iso.log:644:Epoch[40] Iter[350/786] Loss: 2.206 CE: 1.093 Tri: 0.015 CE_rec: 1.064 AIRL_rec: 0.0672 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_airl_iso.log:645:Epoch[40] Iter[400/786] Loss: 2.202 CE: 1.091 Tri: 0.015 CE_rec: 1.063 AIRL_rec: 0.0663 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_airl_iso.log:646:Epoch[40] Iter[450/786] Loss: 2.198 CE: 1.090 Tri: 0.014 CE_rec: 1.061 AIRL_rec: 0.0662 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_airl_iso.log:647:Epoch[40] Iter[500/786] Loss: 2.194 CE: 1.088 Tri: 0.013 CE_rec: 1.060 AIRL_rec: 0.0655 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_airl_iso.log:648:Epoch[40] Iter[550/786] Loss: 2.191 CE: 1.087 Tri: 0.013 CE_rec: 1.059 AIRL_rec: 0.0650 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_airl_iso.log:649:Epoch[40] Iter[600/786] Loss: 2.187 CE: 1.085 Tri: 0.012 CE_rec: 1.058 AIRL_rec: 0.0646 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_airl_iso.log:650:Epoch[40] Iter[650/786] Loss: 2.183 CE: 1.084 Tri: 0.012 CE_rec: 1.056 AIRL_rec: 0.0640 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_airl_iso.log:651:Epoch[40] Iter[700/786] Loss: 2.180 CE: 1.082 Tri: 0.011 CE_rec: 1.055 AIRL_rec: 0.0632 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_airl_iso.log:652:Epoch[40] done in 239.4s  Loss=2.175 Acc=0.994 AIRL-ISO[lam_eff=0.500 ce_rec=1.053 consistency=0.0621 deg_scale_mean=0.625 n_ground=28741]
logs/agreidv2_airl_iso.log:654:    [A->G] mAP=76.09  R1=83.87  R5=90.70  mINP=49.93
logs/agreidv2_airl_iso.log:655:    [G->A] mAP=76.69  R1=84.82  R5=90.50  mINP=42.19
logs/agreidv2_airl_iso.log:656:    [mean] mAP=76.39  R1=84.34
logs/agreidv2_airl_iso.log:657:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 40 ----
logs/agreidv2_airl_iso.log:658:    [A->G] full mAP=76.09 R1=83.87 | rec mAP=75.40 R1=83.23 | FUSE mAP=77.58 R1=85.44
logs/agreidv2_airl_iso.log:659:    [G->A] full mAP=76.69 R1=84.82 | rec mAP=75.07 R1=83.88 | FUSE mAP=77.65 R1=85.37
logs/agreidv2_airl_iso.log:661:    * new best mean mAP=77.62 (epoch 40) saved
logs/agreidv2_airl_iso.log:662:Epoch[41] Iter[50/786] Loss: 2.205 CE: 1.096 Tri: 0.009 CE_rec: 1.066 AIRL_rec: 0.0657 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_iso.log:663:Epoch[41] Iter[100/786] Loss: 2.212 CE: 1.097 Tri: 0.011 CE_rec: 1.070 AIRL_rec: 0.0675 Acc: 0.992 LR: 1.21e-05
logs/agreidv2_airl_iso.log:664:Epoch[41] Iter[150/786] Loss: 2.205 CE: 1.094 Tri: 0.011 CE_rec: 1.067 AIRL_rec: 0.0673 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_iso.log:665:Epoch[41] Iter[200/786] Loss: 2.201 CE: 1.092 Tri: 0.011 CE_rec: 1.064 AIRL_rec: 0.0670 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_iso.log:666:Epoch[41] Iter[250/786] Loss: 2.198 CE: 1.090 Tri: 0.011 CE_rec: 1.063 AIRL_rec: 0.0666 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_iso.log:667:Epoch[41] Iter[300/786] Loss: 2.196 CE: 1.089 Tri: 0.012 CE_rec: 1.062 AIRL_rec: 0.0662 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_iso.log:668:Epoch[41] Iter[350/786] Loss: 2.194 CE: 1.088 Tri: 0.011 CE_rec: 1.062 AIRL_rec: 0.0656 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_iso.log:669:Epoch[41] Iter[400/786] Loss: 2.192 CE: 1.087 Tri: 0.011 CE_rec: 1.061 AIRL_rec: 0.0651 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_iso.log:670:Epoch[41] Iter[450/786] Loss: 2.192 CE: 1.087 Tri: 0.012 CE_rec: 1.061 AIRL_rec: 0.0647 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_iso.log:671:Epoch[41] Iter[500/786] Loss: 2.189 CE: 1.086 Tri: 0.012 CE_rec: 1.059 AIRL_rec: 0.0644 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_iso.log:672:Epoch[41] Iter[550/786] Loss: 2.186 CE: 1.084 Tri: 0.011 CE_rec: 1.058 AIRL_rec: 0.0639 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_airl_iso.log:673:Epoch[41] Iter[600/786] Loss: 2.183 CE: 1.083 Tri: 0.011 CE_rec: 1.057 AIRL_rec: 0.0633 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_airl_iso.log:674:Epoch[41] Iter[650/786] Loss: 2.181 CE: 1.082 Tri: 0.011 CE_rec: 1.056 AIRL_rec: 0.0626 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_airl_iso.log:675:Epoch[41] Iter[700/786] Loss: 2.177 CE: 1.080 Tri: 0.011 CE_rec: 1.055 AIRL_rec: 0.0617 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_airl_iso.log:676:Epoch[41] done in 232.2s  Loss=2.171 Acc=0.994 AIRL-ISO[lam_eff=0.500 ce_rec=1.053 consistency=0.0606 deg_scale_mean=0.625 n_ground=28778]
logs/agreidv2_airl_iso.log:677:Epoch[42] Iter[50/786] Loss: 2.212 CE: 1.100 Tri: 0.009 CE_rec: 1.072 AIRL_rec: 0.0614 Acc: 0.991 LR: 1.11e-05
logs/agreidv2_airl_iso.log:678:Epoch[42] Iter[100/786] Loss: 2.206 CE: 1.095 Tri: 0.012 CE_rec: 1.068 AIRL_rec: 0.0633 Acc: 0.992 LR: 1.11e-05
logs/agreidv2_airl_iso.log:679:Epoch[42] Iter[150/786] Loss: 2.197 CE: 1.090 Tri: 0.011 CE_rec: 1.063 AIRL_rec: 0.0644 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_airl_iso.log:680:Epoch[42] Iter[200/786] Loss: 2.191 CE: 1.088 Tri: 0.011 CE_rec: 1.060 AIRL_rec: 0.0644 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_iso.log:681:Epoch[42] Iter[250/786] Loss: 2.195 CE: 1.089 Tri: 0.013 CE_rec: 1.061 AIRL_rec: 0.0650 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_airl_iso.log:682:Epoch[42] Iter[300/786] Loss: 2.193 CE: 1.088 Tri: 0.013 CE_rec: 1.060 AIRL_rec: 0.0644 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_airl_iso.log:683:Epoch[42] Iter[350/786] Loss: 2.191 CE: 1.087 Tri: 0.012 CE_rec: 1.059 AIRL_rec: 0.0642 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_airl_iso.log:684:Epoch[42] Iter[400/786] Loss: 2.186 CE: 1.085 Tri: 0.012 CE_rec: 1.058 AIRL_rec: 0.0636 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_iso.log:685:Epoch[42] Iter[450/786] Loss: 2.183 CE: 1.084 Tri: 0.011 CE_rec: 1.056 AIRL_rec: 0.0630 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_iso.log:686:Epoch[42] Iter[500/786] Loss: 2.183 CE: 1.084 Tri: 0.012 CE_rec: 1.057 AIRL_rec: 0.0624 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_airl_iso.log:687:Epoch[42] Iter[550/786] Loss: 2.180 CE: 1.082 Tri: 0.011 CE_rec: 1.056 AIRL_rec: 0.0618 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_iso.log:688:Epoch[42] Iter[600/786] Loss: 2.177 CE: 1.081 Tri: 0.011 CE_rec: 1.055 AIRL_rec: 0.0613 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_iso.log:689:Epoch[42] Iter[650/786] Loss: 2.173 CE: 1.079 Tri: 0.010 CE_rec: 1.053 AIRL_rec: 0.0605 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_iso.log:690:Epoch[42] Iter[700/786] Loss: 2.169 CE: 1.077 Tri: 0.010 CE_rec: 1.052 AIRL_rec: 0.0595 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_iso.log:691:Epoch[42] done in 234.3s  Loss=2.164 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.050 consistency=0.0584 deg_scale_mean=0.627 n_ground=28749]
logs/agreidv2_airl_iso.log:692:Epoch[43] Iter[50/786] Loss: 2.202 CE: 1.091 Tri: 0.014 CE_rec: 1.066 AIRL_rec: 0.0604 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_airl_iso.log:693:Epoch[43] Iter[100/786] Loss: 2.185 CE: 1.084 Tri: 0.010 CE_rec: 1.061 AIRL_rec: 0.0605 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_iso.log:694:Epoch[43] Iter[150/786] Loss: 2.182 CE: 1.083 Tri: 0.010 CE_rec: 1.059 AIRL_rec: 0.0610 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_iso.log:695:Epoch[43] Iter[200/786] Loss: 2.181 CE: 1.082 Tri: 0.012 CE_rec: 1.057 AIRL_rec: 0.0617 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_airl_iso.log:696:Epoch[43] Iter[250/786] Loss: 2.180 CE: 1.081 Tri: 0.011 CE_rec: 1.057 AIRL_rec: 0.0617 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_airl_iso.log:697:Epoch[43] Iter[300/786] Loss: 2.176 CE: 1.080 Tri: 0.010 CE_rec: 1.055 AIRL_rec: 0.0615 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_airl_iso.log:698:Epoch[43] Iter[350/786] Loss: 2.176 CE: 1.080 Tri: 0.010 CE_rec: 1.055 AIRL_rec: 0.0614 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_airl_iso.log:699:Epoch[43] Iter[400/786] Loss: 2.175 CE: 1.080 Tri: 0.010 CE_rec: 1.055 AIRL_rec: 0.0614 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_airl_iso.log:700:Epoch[43] Iter[450/786] Loss: 2.172 CE: 1.078 Tri: 0.010 CE_rec: 1.054 AIRL_rec: 0.0608 Acc: 0.994 LR: 1.00e-05
logs/agreidv2_airl_iso.log:701:Epoch[43] Iter[500/786] Loss: 2.170 CE: 1.077 Tri: 0.010 CE_rec: 1.053 AIRL_rec: 0.0606 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_iso.log:702:Epoch[43] Iter[550/786] Loss: 2.169 CE: 1.077 Tri: 0.010 CE_rec: 1.052 AIRL_rec: 0.0602 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_iso.log:703:Epoch[43] Iter[600/786] Loss: 2.168 CE: 1.076 Tri: 0.010 CE_rec: 1.052 AIRL_rec: 0.0597 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_iso.log:704:Epoch[43] Iter[650/786] Loss: 2.165 CE: 1.075 Tri: 0.010 CE_rec: 1.050 AIRL_rec: 0.0590 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_iso.log:705:Epoch[43] Iter[700/786] Loss: 2.160 CE: 1.073 Tri: 0.010 CE_rec: 1.049 AIRL_rec: 0.0580 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_iso.log:706:Epoch[43] done in 235.0s  Loss=2.156 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.047 consistency=0.0572 deg_scale_mean=0.625 n_ground=28769]
logs/agreidv2_airl_iso.log:707:Epoch[44] Iter[50/786] Loss: 2.204 CE: 1.090 Tri: 0.020 CE_rec: 1.064 AIRL_rec: 0.0601 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_airl_iso.log:708:Epoch[44] Iter[100/786] Loss: 2.188 CE: 1.085 Tri: 0.015 CE_rec: 1.058 AIRL_rec: 0.0603 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_iso.log:709:Epoch[44] Iter[150/786] Loss: 2.186 CE: 1.083 Tri: 0.015 CE_rec: 1.057 AIRL_rec: 0.0610 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_iso.log:710:Epoch[44] Iter[200/786] Loss: 2.185 CE: 1.082 Tri: 0.017 CE_rec: 1.057 AIRL_rec: 0.0608 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_airl_iso.log:711:Epoch[44] Iter[250/786] Loss: 2.181 CE: 1.080 Tri: 0.016 CE_rec: 1.055 AIRL_rec: 0.0610 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_iso.log:712:Epoch[44] Iter[300/786] Loss: 2.178 CE: 1.079 Tri: 0.014 CE_rec: 1.054 AIRL_rec: 0.0605 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_iso.log:713:Epoch[44] Iter[350/786] Loss: 2.176 CE: 1.079 Tri: 0.013 CE_rec: 1.054 AIRL_rec: 0.0605 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_iso.log:714:Epoch[44] Iter[400/786] Loss: 2.173 CE: 1.077 Tri: 0.012 CE_rec: 1.053 AIRL_rec: 0.0604 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_iso.log:715:Epoch[44] Iter[450/786] Loss: 2.170 CE: 1.076 Tri: 0.012 CE_rec: 1.052 AIRL_rec: 0.0599 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_iso.log:716:Epoch[44] Iter[500/786] Loss: 2.168 CE: 1.075 Tri: 0.011 CE_rec: 1.052 AIRL_rec: 0.0597 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_iso.log:717:Epoch[44] Iter[550/786] Loss: 2.164 CE: 1.074 Tri: 0.011 CE_rec: 1.050 AIRL_rec: 0.0590 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_iso.log:718:Epoch[44] Iter[600/786] Loss: 2.161 CE: 1.073 Tri: 0.010 CE_rec: 1.049 AIRL_rec: 0.0582 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_airl_iso.log:719:Epoch[44] Iter[650/786] Loss: 2.159 CE: 1.072 Tri: 0.010 CE_rec: 1.048 AIRL_rec: 0.0574 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_airl_iso.log:720:Epoch[44] Iter[700/786] Loss: 2.155 CE: 1.070 Tri: 0.010 CE_rec: 1.047 AIRL_rec: 0.0568 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_airl_iso.log:721:Epoch[44] done in 236.8s  Loss=2.151 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.045 consistency=0.0559 deg_scale_mean=0.625 n_ground=28880]
logs/agreidv2_airl_iso.log:722:Epoch[45] Iter[50/786] Loss: 2.187 CE: 1.086 Tri: 0.009 CE_rec: 1.061 AIRL_rec: 0.0600 Acc: 0.992 LR: 8.12e-06
logs/agreidv2_airl_iso.log:723:Epoch[45] Iter[100/786] Loss: 2.175 CE: 1.080 Tri: 0.010 CE_rec: 1.055 AIRL_rec: 0.0601 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_airl_iso.log:724:Epoch[45] Iter[150/786] Loss: 2.164 CE: 1.075 Tri: 0.008 CE_rec: 1.051 AIRL_rec: 0.0597 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_iso.log:725:Epoch[45] Iter[200/786] Loss: 2.161 CE: 1.074 Tri: 0.007 CE_rec: 1.050 AIRL_rec: 0.0597 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_iso.log:726:Epoch[45] Iter[250/786] Loss: 2.161 CE: 1.074 Tri: 0.008 CE_rec: 1.050 AIRL_rec: 0.0593 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_iso.log:727:Epoch[45] Iter[300/786] Loss: 2.162 CE: 1.074 Tri: 0.008 CE_rec: 1.051 AIRL_rec: 0.0592 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_iso.log:728:Epoch[45] Iter[350/786] Loss: 2.161 CE: 1.073 Tri: 0.008 CE_rec: 1.050 AIRL_rec: 0.0590 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_iso.log:729:Epoch[45] Iter[400/786] Loss: 2.158 CE: 1.072 Tri: 0.008 CE_rec: 1.049 AIRL_rec: 0.0587 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_iso.log:730:Epoch[45] Iter[450/786] Loss: 2.155 CE: 1.071 Tri: 0.007 CE_rec: 1.047 AIRL_rec: 0.0581 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_iso.log:731:Epoch[45] Iter[500/786] Loss: 2.155 CE: 1.071 Tri: 0.008 CE_rec: 1.048 AIRL_rec: 0.0576 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_iso.log:732:Epoch[45] Iter[550/786] Loss: 2.153 CE: 1.070 Tri: 0.008 CE_rec: 1.047 AIRL_rec: 0.0574 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_iso.log:733:Epoch[45] Iter[600/786] Loss: 2.151 CE: 1.069 Tri: 0.008 CE_rec: 1.046 AIRL_rec: 0.0569 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_iso.log:734:Epoch[45] Iter[650/786] Loss: 2.148 CE: 1.068 Tri: 0.007 CE_rec: 1.045 AIRL_rec: 0.0562 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_airl_iso.log:735:Epoch[45] Iter[700/786] Loss: 2.145 CE: 1.066 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0554 Acc: 0.996 LR: 8.12e-06
logs/agreidv2_airl_iso.log:736:Epoch[45] done in 239.5s  Loss=2.140 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.042 consistency=0.0546 deg_scale_mean=0.625 n_ground=28820]
logs/agreidv2_airl_iso.log:737:Epoch[46] Iter[50/786] Loss: 2.185 CE: 1.083 Tri: 0.011 CE_rec: 1.062 AIRL_rec: 0.0578 Acc: 0.990 LR: 7.21e-06
logs/agreidv2_airl_iso.log:738:Epoch[46] Iter[100/786] Loss: 2.168 CE: 1.076 Tri: 0.008 CE_rec: 1.055 AIRL_rec: 0.0569 Acc: 0.993 LR: 7.21e-06
logs/agreidv2_airl_iso.log:739:Epoch[46] Iter[150/786] Loss: 2.161 CE: 1.073 Tri: 0.007 CE_rec: 1.051 AIRL_rec: 0.0576 Acc: 0.994 LR: 7.21e-06
logs/agreidv2_airl_iso.log:740:Epoch[46] Iter[200/786] Loss: 2.160 CE: 1.072 Tri: 0.007 CE_rec: 1.051 AIRL_rec: 0.0582 Acc: 0.994 LR: 7.21e-06
logs/agreidv2_airl_iso.log:741:Epoch[46] Iter[250/786] Loss: 2.158 CE: 1.071 Tri: 0.008 CE_rec: 1.050 AIRL_rec: 0.0582 Acc: 0.994 LR: 7.21e-06
logs/agreidv2_airl_iso.log:742:Epoch[46] Iter[300/786] Loss: 2.154 CE: 1.070 Tri: 0.008 CE_rec: 1.048 AIRL_rec: 0.0584 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_airl_iso.log:743:Epoch[46] Iter[350/786] Loss: 2.150 CE: 1.068 Tri: 0.007 CE_rec: 1.046 AIRL_rec: 0.0579 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_airl_iso.log:744:Epoch[46] Iter[400/786] Loss: 2.149 CE: 1.067 Tri: 0.007 CE_rec: 1.046 AIRL_rec: 0.0573 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_airl_iso.log:745:Epoch[46] Iter[450/786] Loss: 2.146 CE: 1.066 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0568 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_airl_iso.log:746:Epoch[46] Iter[500/786] Loss: 2.144 CE: 1.065 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0563 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_airl_iso.log:747:Epoch[46] Iter[550/786] Loss: 2.143 CE: 1.064 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0558 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_iso.log:748:Epoch[46] Iter[600/786] Loss: 2.140 CE: 1.063 Tri: 0.007 CE_rec: 1.042 AIRL_rec: 0.0553 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_iso.log:749:Epoch[46] Iter[650/786] Loss: 2.138 CE: 1.062 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0546 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_iso.log:750:Epoch[46] Iter[700/786] Loss: 2.135 CE: 1.061 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0538 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_iso.log:751:Epoch[46] done in 241.2s  Loss=2.131 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.039 consistency=0.0528 deg_scale_mean=0.624 n_ground=28864]
logs/agreidv2_airl_iso.log:752:Epoch[47] Iter[50/786] Loss: 2.168 CE: 1.075 Tri: 0.007 CE_rec: 1.055 AIRL_rec: 0.0589 Acc: 0.994 LR: 6.35e-06
logs/agreidv2_airl_iso.log:753:Epoch[47] Iter[100/786] Loss: 2.161 CE: 1.073 Tri: 0.008 CE_rec: 1.050 AIRL_rec: 0.0584 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_iso.log:754:Epoch[47] Iter[150/786] Loss: 2.153 CE: 1.070 Tri: 0.007 CE_rec: 1.047 AIRL_rec: 0.0579 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_iso.log:755:Epoch[47] Iter[200/786] Loss: 2.151 CE: 1.069 Tri: 0.006 CE_rec: 1.047 AIRL_rec: 0.0574 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_iso.log:756:Epoch[47] Iter[250/786] Loss: 2.150 CE: 1.068 Tri: 0.007 CE_rec: 1.046 AIRL_rec: 0.0575 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_iso.log:757:Epoch[47] Iter[300/786] Loss: 2.148 CE: 1.067 Tri: 0.007 CE_rec: 1.045 AIRL_rec: 0.0572 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_iso.log:758:Epoch[47] Iter[350/786] Loss: 2.148 CE: 1.067 Tri: 0.008 CE_rec: 1.045 AIRL_rec: 0.0569 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_iso.log:759:Epoch[47] Iter[400/786] Loss: 2.145 CE: 1.066 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0564 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_iso.log:760:Epoch[47] Iter[450/786] Loss: 2.144 CE: 1.065 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0560 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_iso.log:761:Epoch[47] Iter[500/786] Loss: 2.142 CE: 1.064 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0553 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_iso.log:762:Epoch[47] Iter[550/786] Loss: 2.139 CE: 1.063 Tri: 0.007 CE_rec: 1.042 AIRL_rec: 0.0548 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_airl_iso.log:763:Epoch[47] Iter[600/786] Loss: 2.137 CE: 1.062 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0542 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_airl_iso.log:764:Epoch[47] Iter[650/786] Loss: 2.134 CE: 1.061 Tri: 0.006 CE_rec: 1.040 AIRL_rec: 0.0534 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_airl_iso.log:765:Epoch[47] Iter[700/786] Loss: 2.132 CE: 1.060 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0527 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_airl_iso.log:766:Epoch[47] done in 237.3s  Loss=2.129 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.038 consistency=0.0521 deg_scale_mean=0.624 n_ground=28543]
logs/agreidv2_airl_iso.log:767:Epoch[48] Iter[50/786] Loss: 2.150 CE: 1.069 Tri: 0.004 CE_rec: 1.049 AIRL_rec: 0.0557 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_airl_iso.log:768:Epoch[48] Iter[100/786] Loss: 2.155 CE: 1.069 Tri: 0.009 CE_rec: 1.049 AIRL_rec: 0.0574 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_airl_iso.log:769:Epoch[48] Iter[150/786] Loss: 2.153 CE: 1.069 Tri: 0.007 CE_rec: 1.048 AIRL_rec: 0.0572 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_airl_iso.log:770:Epoch[48] Iter[200/786] Loss: 2.150 CE: 1.068 Tri: 0.007 CE_rec: 1.047 AIRL_rec: 0.0567 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_airl_iso.log:771:Epoch[48] Iter[250/786] Loss: 2.148 CE: 1.067 Tri: 0.007 CE_rec: 1.046 AIRL_rec: 0.0564 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_airl_iso.log:772:Epoch[48] Iter[300/786] Loss: 2.146 CE: 1.066 Tri: 0.007 CE_rec: 1.045 AIRL_rec: 0.0562 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_airl_iso.log:773:Epoch[48] Iter[350/786] Loss: 2.145 CE: 1.066 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0559 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_airl_iso.log:774:Epoch[48] Iter[400/786] Loss: 2.142 CE: 1.065 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0559 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_iso.log:775:Epoch[48] Iter[450/786] Loss: 2.140 CE: 1.064 Tri: 0.006 CE_rec: 1.042 AIRL_rec: 0.0554 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_iso.log:776:Epoch[48] Iter[500/786] Loss: 2.138 CE: 1.063 Tri: 0.006 CE_rec: 1.041 AIRL_rec: 0.0550 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_iso.log:777:Epoch[48] Iter[550/786] Loss: 2.136 CE: 1.062 Tri: 0.006 CE_rec: 1.041 AIRL_rec: 0.0546 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_iso.log:778:Epoch[48] Iter[600/786] Loss: 2.133 CE: 1.061 Tri: 0.006 CE_rec: 1.040 AIRL_rec: 0.0538 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_airl_iso.log:779:Epoch[48] Iter[650/786] Loss: 2.131 CE: 1.060 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0533 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_airl_iso.log:780:Epoch[48] Iter[700/786] Loss: 2.128 CE: 1.058 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0526 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_airl_iso.log:781:Epoch[48] done in 239.6s  Loss=2.125 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.037 consistency=0.0518 deg_scale_mean=0.625 n_ground=28757]
logs/agreidv2_airl_iso.log:782:Epoch[49] Iter[50/786] Loss: 2.149 CE: 1.068 Tri: 0.005 CE_rec: 1.050 AIRL_rec: 0.0531 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_airl_iso.log:783:Epoch[49] Iter[100/786] Loss: 2.151 CE: 1.069 Tri: 0.007 CE_rec: 1.048 AIRL_rec: 0.0543 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_airl_iso.log:784:Epoch[49] Iter[150/786] Loss: 2.149 CE: 1.067 Tri: 0.008 CE_rec: 1.046 AIRL_rec: 0.0561 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_airl_iso.log:785:Epoch[49] Iter[200/786] Loss: 2.145 CE: 1.065 Tri: 0.008 CE_rec: 1.044 AIRL_rec: 0.0564 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_iso.log:786:Epoch[49] Iter[250/786] Loss: 2.142 CE: 1.064 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0558 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_iso.log:787:Epoch[49] Iter[300/786] Loss: 2.141 CE: 1.064 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0555 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_iso.log:788:Epoch[49] Iter[350/786] Loss: 2.142 CE: 1.064 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0554 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_iso.log:789:Epoch[49] Iter[400/786] Loss: 2.140 CE: 1.063 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0552 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_iso.log:790:Epoch[49] Iter[450/786] Loss: 2.139 CE: 1.062 Tri: 0.007 CE_rec: 1.042 AIRL_rec: 0.0548 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_iso.log:791:Epoch[49] Iter[500/786] Loss: 2.136 CE: 1.061 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0543 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_iso.log:792:Epoch[49] Iter[550/786] Loss: 2.134 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0537 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_iso.log:793:Epoch[49] Iter[600/786] Loss: 2.132 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0530 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_airl_iso.log:794:Epoch[49] Iter[650/786] Loss: 2.129 CE: 1.058 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0524 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_airl_iso.log:795:Epoch[49] Iter[700/786] Loss: 2.126 CE: 1.057 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0517 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_airl_iso.log:796:Epoch[49] done in 239.2s  Loss=2.122 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.036 consistency=0.0509 deg_scale_mean=0.625 n_ground=28643]
logs/agreidv2_airl_iso.log:797:Epoch[50] Iter[50/786] Loss: 2.137 CE: 1.064 Tri: 0.002 CE_rec: 1.044 AIRL_rec: 0.0546 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_iso.log:798:Epoch[50] Iter[100/786] Loss: 2.133 CE: 1.062 Tri: 0.003 CE_rec: 1.041 AIRL_rec: 0.0545 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_airl_iso.log:799:Epoch[50] Iter[150/786] Loss: 2.132 CE: 1.060 Tri: 0.005 CE_rec: 1.040 AIRL_rec: 0.0542 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_iso.log:800:Epoch[50] Iter[200/786] Loss: 2.133 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0544 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_airl_iso.log:801:Epoch[50] Iter[250/786] Loss: 2.136 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0551 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_iso.log:802:Epoch[50] Iter[300/786] Loss: 2.135 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0546 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_iso.log:803:Epoch[50] Iter[350/786] Loss: 2.134 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0544 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_iso.log:804:Epoch[50] Iter[400/786] Loss: 2.131 CE: 1.059 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0541 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_iso.log:805:Epoch[50] Iter[450/786] Loss: 2.129 CE: 1.058 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0537 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_iso.log:806:Epoch[50] Iter[500/786] Loss: 2.127 CE: 1.057 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0530 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_iso.log:807:Epoch[50] Iter[550/786] Loss: 2.126 CE: 1.056 Tri: 0.005 CE_rec: 1.038 AIRL_rec: 0.0525 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_iso.log:808:Epoch[50] Iter[600/786] Loss: 2.124 CE: 1.056 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0519 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_iso.log:809:Epoch[50] Iter[650/786] Loss: 2.122 CE: 1.055 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0514 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_iso.log:810:Epoch[50] Iter[700/786] Loss: 2.119 CE: 1.053 Tri: 0.006 CE_rec: 1.035 AIRL_rec: 0.0505 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_airl_iso.log:811:Epoch[50] done in 239.6s  Loss=2.116 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.034 consistency=0.0497 deg_scale_mean=0.624 n_ground=28797]
logs/agreidv2_airl_4090.log:2:CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
logs/agreidv2_airl_4090.log:7:  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
logs/agreidv2_airl_4090.log:8:  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
logs/agreidv2_airl_4090.log:9:  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_airl_4090.log:11:  out_dir=/home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090
logs/agreidv2_airl_4090.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_airl_4090.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_airl_4090.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_airl_4090.log:25:  [AIRL-ISO] iso_stage=3: rec late stage (14185392 params, 28 tensors) + rec BNNeck head (620544 params, 2 tensors) in optimizer [rec late stage @ Swin LR 3.50e-05, rec BNNeck @ full LR 3.50e-04]; degradation-consistency grad isolated from shared trunk (detached degraded pass at stage-3 input); trunk_recce=0 (clean ID-CE + consistency BOTH detached = original full-isolation); eval soft-fusion cos=w*cos_rec+(1-w)*cos_full w=0.25
logs/agreidv2_airl_4090.log:26:Epoch[1] Iter[50/786] Loss: 44.958 CE: 6.693 Tri: 31.571 CE_rec: 6.693 AIRL_rec: 0.0002 Acc: 0.001 LR: 3.50e-07
logs/agreidv2_airl_4090.log:27:Epoch[1] Iter[100/786] Loss: 38.226 CE: 6.692 Tri: 24.843 CE_rec: 6.691 AIRL_rec: 0.0002 Acc: 0.002 LR: 3.50e-07
logs/agreidv2_airl_4090.log:28:Epoch[1] Iter[150/786] Loss: 35.189 CE: 6.690 Tri: 21.809 CE_rec: 6.690 AIRL_rec: 0.0002 Acc: 0.003 LR: 3.50e-07
logs/agreidv2_airl_4090.log:29:Epoch[1] Iter[200/786] Loss: 33.301 CE: 6.689 Tri: 19.925 CE_rec: 6.688 AIRL_rec: 0.0002 Acc: 0.004 LR: 3.50e-07
logs/agreidv2_airl_4090.log:30:Epoch[1] Iter[250/786] Loss: 31.912 CE: 6.686 Tri: 18.540 CE_rec: 6.685 AIRL_rec: 0.0002 Acc: 0.006 LR: 3.50e-07
logs/agreidv2_airl_4090.log:31:Epoch[1] Iter[300/786] Loss: 30.823 CE: 6.684 Tri: 17.456 CE_rec: 6.683 AIRL_rec: 0.0002 Acc: 0.009 LR: 3.50e-07
logs/agreidv2_airl_4090.log:32:Epoch[1] Iter[350/786] Loss: 29.893 CE: 6.682 Tri: 16.530 CE_rec: 6.681 AIRL_rec: 0.0002 Acc: 0.012 LR: 3.50e-07
logs/agreidv2_airl_4090.log:33:Epoch[1] Iter[400/786] Loss: 29.132 CE: 6.680 Tri: 15.773 CE_rec: 6.679 AIRL_rec: 0.0002 Acc: 0.017 LR: 3.50e-07
logs/agreidv2_airl_4090.log:34:Epoch[1] Iter[450/786] Loss: 28.396 CE: 6.677 Tri: 15.043 CE_rec: 6.676 AIRL_rec: 0.0002 Acc: 0.022 LR: 3.50e-07
logs/agreidv2_airl_4090.log:35:Epoch[1] Iter[500/786] Loss: 27.748 CE: 6.675 Tri: 14.400 CE_rec: 6.673 AIRL_rec: 0.0002 Acc: 0.030 LR: 3.50e-07
logs/agreidv2_airl_4090.log:36:Epoch[1] Iter[550/786] Loss: 27.164 CE: 6.672 Tri: 13.822 CE_rec: 6.671 AIRL_rec: 0.0002 Acc: 0.040 LR: 3.50e-07
logs/agreidv2_airl_4090.log:37:Epoch[1] Iter[600/786] Loss: 26.662 CE: 6.669 Tri: 13.325 CE_rec: 6.667 AIRL_rec: 0.0002 Acc: 0.051 LR: 3.50e-07
logs/agreidv2_airl_4090.log:38:Epoch[1] Iter[650/786] Loss: 26.185 CE: 6.666 Tri: 12.855 CE_rec: 6.664 AIRL_rec: 0.0002 Acc: 0.066 LR: 3.50e-07
logs/agreidv2_airl_4090.log:39:Epoch[1] Iter[700/786] Loss: 25.749 CE: 6.662 Tri: 12.427 CE_rec: 6.660 AIRL_rec: 0.0002 Acc: 0.088 LR: 3.50e-07
logs/agreidv2_airl_4090.log:40:Epoch[1] done in 113.2s  Loss=25.389 Acc=0.116 AIRL-ISO[lam_eff=0.100 ce_rec=6.655 consistency=0.0002 deg_scale_mean=0.625 n_ground=28680]
logs/agreidv2_airl_4090.log:41:Epoch[2] Iter[50/786] Loss: 18.545 CE: 6.607 Tri: 5.341 CE_rec: 6.597 AIRL_rec: 0.0003 Acc: 0.129 LR: 3.82e-06
logs/agreidv2_airl_4090.log:42:Epoch[2] Iter[100/786] Loss: 17.915 CE: 6.577 Tri: 4.781 CE_rec: 6.558 AIRL_rec: 0.0004 Acc: 0.163 LR: 3.82e-06
logs/agreidv2_airl_4090.log:43:Epoch[2] Iter[150/786] Loss: 17.320 CE: 6.542 Tri: 4.266 CE_rec: 6.511 AIRL_rec: 0.0006 Acc: 0.201 LR: 3.82e-06
logs/agreidv2_airl_4090.log:44:Epoch[2] Iter[200/786] Loss: 16.966 CE: 6.509 Tri: 3.996 CE_rec: 6.462 AIRL_rec: 0.0009 Acc: 0.231 LR: 3.82e-06
logs/agreidv2_airl_4090.log:45:Epoch[2] Iter[250/786] Loss: 16.673 CE: 6.475 Tri: 3.787 CE_rec: 6.411 AIRL_rec: 0.0014 Acc: 0.264 LR: 3.82e-06
logs/agreidv2_airl_4090.log:46:Epoch[2] Iter[300/786] Loss: 16.381 CE: 6.441 Tri: 3.581 CE_rec: 6.359 AIRL_rec: 0.0020 Acc: 0.289 LR: 3.82e-06
logs/agreidv2_airl_4090.log:47:Epoch[2] Iter[350/786] Loss: 16.145 CE: 6.410 Tri: 3.425 CE_rec: 6.310 AIRL_rec: 0.0026 Acc: 0.308 LR: 3.82e-06
logs/agreidv2_airl_4090.log:48:Epoch[2] Iter[400/786] Loss: 15.907 CE: 6.376 Tri: 3.271 CE_rec: 6.258 AIRL_rec: 0.0032 Acc: 0.329 LR: 3.82e-06
logs/agreidv2_airl_4090.log:49:Epoch[2] Iter[450/786] Loss: 15.673 CE: 6.342 Tri: 3.124 CE_rec: 6.206 AIRL_rec: 0.0039 Acc: 0.349 LR: 3.82e-06
logs/agreidv2_airl_4090.log:50:Epoch[2] Iter[500/786] Loss: 15.463 CE: 6.306 Tri: 3.004 CE_rec: 6.152 AIRL_rec: 0.0046 Acc: 0.371 LR: 3.82e-06
logs/agreidv2_airl_4090.log:51:Epoch[2] Iter[550/786] Loss: 15.265 CE: 6.268 Tri: 2.901 CE_rec: 6.095 AIRL_rec: 0.0054 Acc: 0.393 LR: 3.82e-06
logs/agreidv2_airl_4090.log:52:Epoch[2] Iter[600/786] Loss: 15.062 CE: 6.227 Tri: 2.800 CE_rec: 6.034 AIRL_rec: 0.0061 Acc: 0.416 LR: 3.82e-06
logs/agreidv2_airl_4090.log:53:Epoch[2] Iter[650/786] Loss: 14.863 CE: 6.183 Tri: 2.710 CE_rec: 5.969 AIRL_rec: 0.0069 Acc: 0.438 LR: 3.82e-06
logs/agreidv2_airl_4090.log:54:Epoch[2] Iter[700/786] Loss: 14.653 CE: 6.130 Tri: 2.631 CE_rec: 5.891 AIRL_rec: 0.0078 Acc: 0.461 LR: 3.82e-06
logs/agreidv2_airl_4090.log:55:Epoch[2] done in 112.9s  Loss=14.447 Acc=0.482 AIRL-ISO[lam_eff=0.200 ce_rec=5.800 consistency=0.0086 deg_scale_mean=0.624 n_ground=28677]
logs/agreidv2_airl_4090.log:56:Epoch[3] Iter[50/786] Loss: 12.357 CE: 5.631 Tri: 1.335 CE_rec: 5.385 AIRL_rec: 0.0200 Acc: 0.304 LR: 7.28e-06
logs/agreidv2_airl_4090.log:57:Epoch[3] Iter[100/786] Loss: 12.141 CE: 5.540 Tri: 1.340 CE_rec: 5.255 AIRL_rec: 0.0212 Acc: 0.353 LR: 7.28e-06
logs/agreidv2_airl_4090.log:58:Epoch[3] Iter[150/786] Loss: 11.916 CE: 5.460 Tri: 1.293 CE_rec: 5.156 AIRL_rec: 0.0225 Acc: 0.383 LR: 7.28e-06
logs/agreidv2_airl_4090.log:59:Epoch[3] Iter[200/786] Loss: 11.682 CE: 5.368 Tri: 1.260 CE_rec: 5.046 AIRL_rec: 0.0235 Acc: 0.422 LR: 7.28e-06
logs/agreidv2_airl_4090.log:60:Epoch[3] Iter[250/786] Loss: 11.433 CE: 5.280 Tri: 1.197 CE_rec: 4.949 AIRL_rec: 0.0244 Acc: 0.452 LR: 7.28e-06
logs/agreidv2_airl_4090.log:61:Epoch[3] Iter[300/786] Loss: 11.242 CE: 5.200 Tri: 1.173 CE_rec: 4.862 AIRL_rec: 0.0259 Acc: 0.474 LR: 7.28e-06
logs/agreidv2_airl_4090.log:62:Epoch[3] Iter[350/786] Loss: 11.034 CE: 5.115 Tri: 1.140 CE_rec: 4.771 AIRL_rec: 0.0274 Acc: 0.498 LR: 7.28e-06
logs/agreidv2_airl_4090.log:63:Epoch[3] Iter[400/786] Loss: 10.817 CE: 5.024 Tri: 1.111 CE_rec: 4.673 AIRL_rec: 0.0286 Acc: 0.522 LR: 7.28e-06
logs/agreidv2_airl_4090.log:64:Epoch[3] Iter[450/786] Loss: 10.616 CE: 4.936 Tri: 1.089 CE_rec: 4.581 AIRL_rec: 0.0300 Acc: 0.542 LR: 7.28e-06
logs/agreidv2_airl_4090.log:65:Epoch[3] Iter[500/786] Loss: 10.416 CE: 4.847 Tri: 1.071 CE_rec: 4.489 AIRL_rec: 0.0314 Acc: 0.560 LR: 7.28e-06
logs/agreidv2_airl_4090.log:66:Epoch[3] Iter[550/786] Loss: 10.213 CE: 4.755 Tri: 1.055 CE_rec: 4.393 AIRL_rec: 0.0327 Acc: 0.579 LR: 7.28e-06
logs/agreidv2_airl_4090.log:67:Epoch[3] Iter[600/786] Loss: 9.997 CE: 4.658 Tri: 1.036 CE_rec: 4.293 AIRL_rec: 0.0338 Acc: 0.595 LR: 7.28e-06
logs/agreidv2_airl_4090.log:68:Epoch[3] Iter[650/786] Loss: 9.762 CE: 4.552 Tri: 1.015 CE_rec: 4.184 AIRL_rec: 0.0352 Acc: 0.612 LR: 7.28e-06
logs/agreidv2_airl_4090.log:69:Epoch[3] Iter[700/786] Loss: 9.497 CE: 4.430 Tri: 0.996 CE_rec: 4.060 AIRL_rec: 0.0365 Acc: 0.631 LR: 7.28e-06
logs/agreidv2_airl_4090.log:70:Epoch[3] done in 112.5s  Loss=9.258 Acc=0.646 AIRL-ISO[lam_eff=0.300 ce_rec=3.947 consistency=0.0379 deg_scale_mean=0.627 n_ground=28554]
logs/agreidv2_airl_4090.log:71:Epoch[4] Iter[50/786] Loss: 8.372 CE: 3.815 Tri: 0.841 CE_rec: 3.692 AIRL_rec: 0.0606 Acc: 0.461 LR: 1.07e-05
logs/agreidv2_airl_4090.log:72:Epoch[4] Iter[100/786] Loss: 8.055 CE: 3.709 Tri: 0.765 CE_rec: 3.556 AIRL_rec: 0.0620 Acc: 0.500 LR: 1.07e-05
logs/agreidv2_airl_4090.log:73:Epoch[4] Iter[150/786] Loss: 7.826 CE: 3.595 Tri: 0.780 CE_rec: 3.427 AIRL_rec: 0.0637 Acc: 0.546 LR: 1.07e-05
logs/agreidv2_airl_4090.log:74:Epoch[4] Iter[200/786] Loss: 7.603 CE: 3.489 Tri: 0.778 CE_rec: 3.309 AIRL_rec: 0.0657 Acc: 0.583 LR: 1.07e-05
logs/agreidv2_airl_4090.log:75:Epoch[4] Iter[250/786] Loss: 7.380 CE: 3.383 Tri: 0.766 CE_rec: 3.203 AIRL_rec: 0.0687 Acc: 0.615 LR: 1.07e-05
logs/agreidv2_airl_4090.log:76:Epoch[4] Iter[300/786] Loss: 7.213 CE: 3.301 Tri: 0.764 CE_rec: 3.120 AIRL_rec: 0.0708 Acc: 0.640 LR: 1.07e-05
logs/agreidv2_airl_4090.log:77:Epoch[4] Iter[350/786] Loss: 7.068 CE: 3.227 Tri: 0.764 CE_rec: 3.047 AIRL_rec: 0.0737 Acc: 0.660 LR: 1.07e-05
logs/agreidv2_airl_4090.log:78:Epoch[4] Iter[400/786] Loss: 6.905 CE: 3.147 Tri: 0.757 CE_rec: 2.970 AIRL_rec: 0.0758 Acc: 0.679 LR: 1.07e-05
logs/agreidv2_airl_4090.log:79:Epoch[4] Iter[450/786] Loss: 6.736 CE: 3.070 Tri: 0.741 CE_rec: 2.895 AIRL_rec: 0.0775 Acc: 0.697 LR: 1.07e-05
logs/agreidv2_airl_4090.log:80:Epoch[4] Iter[500/786] Loss: 6.564 CE: 2.990 Tri: 0.724 CE_rec: 2.819 AIRL_rec: 0.0791 Acc: 0.715 LR: 1.07e-05
logs/agreidv2_airl_4090.log:81:Epoch[4] Iter[550/786] Loss: 6.401 CE: 2.913 Tri: 0.708 CE_rec: 2.747 AIRL_rec: 0.0806 Acc: 0.730 LR: 1.07e-05
logs/agreidv2_airl_4090.log:82:Epoch[4] Iter[600/786] Loss: 6.237 CE: 2.838 Tri: 0.689 CE_rec: 2.676 AIRL_rec: 0.0823 Acc: 0.744 LR: 1.07e-05
logs/agreidv2_airl_4090.log:83:Epoch[4] Iter[650/786] Loss: 6.087 CE: 2.766 Tri: 0.679 CE_rec: 2.608 AIRL_rec: 0.0842 Acc: 0.757 LR: 1.07e-05
logs/agreidv2_airl_4090.log:84:Epoch[4] Iter[700/786] Loss: 5.925 CE: 2.690 Tri: 0.665 CE_rec: 2.535 AIRL_rec: 0.0855 Acc: 0.770 LR: 1.07e-05
logs/agreidv2_airl_4090.log:85:Epoch[4] done in 112.7s  Loss=5.785 Acc=0.780 AIRL-ISO[lam_eff=0.400 ce_rec=2.473 consistency=0.0869 deg_scale_mean=0.626 n_ground=28676]
logs/agreidv2_airl_4090.log:86:Epoch[5] Iter[50/786] Loss: 5.447 CE: 2.497 Tri: 0.450 CE_rec: 2.445 AIRL_rec: 0.1112 Acc: 0.722 LR: 1.42e-05
logs/agreidv2_airl_4090.log:87:Epoch[5] Iter[100/786] Loss: 5.179 CE: 2.378 Tri: 0.438 CE_rec: 2.306 AIRL_rec: 0.1127 Acc: 0.757 LR: 1.42e-05
logs/agreidv2_airl_4090.log:88:Epoch[5] Iter[150/786] Loss: 5.019 CE: 2.298 Tri: 0.437 CE_rec: 2.228 AIRL_rec: 0.1140 Acc: 0.782 LR: 1.42e-05
logs/agreidv2_airl_4090.log:89:Epoch[5] Iter[200/786] Loss: 4.886 CE: 2.227 Tri: 0.446 CE_rec: 2.155 AIRL_rec: 0.1153 Acc: 0.802 LR: 1.42e-05
logs/agreidv2_airl_4090.log:90:Epoch[5] Iter[250/786] Loss: 4.777 CE: 2.170 Tri: 0.451 CE_rec: 2.097 AIRL_rec: 0.1174 Acc: 0.819 LR: 1.42e-05
logs/agreidv2_airl_4090.log:91:Epoch[5] Iter[300/786] Loss: 4.671 CE: 2.118 Tri: 0.446 CE_rec: 2.047 AIRL_rec: 0.1193 Acc: 0.833 LR: 1.42e-05
logs/agreidv2_airl_4090.log:92:Epoch[5] Iter[350/786] Loss: 4.577 CE: 2.074 Tri: 0.438 CE_rec: 2.005 AIRL_rec: 0.1213 Acc: 0.843 LR: 1.42e-05
logs/agreidv2_airl_4090.log:93:Epoch[5] Iter[400/786] Loss: 4.497 CE: 2.034 Tri: 0.437 CE_rec: 1.964 AIRL_rec: 0.1229 Acc: 0.852 LR: 1.42e-05
logs/agreidv2_airl_4090.log:94:Epoch[5] Iter[450/786] Loss: 4.418 CE: 1.998 Tri: 0.430 CE_rec: 1.928 AIRL_rec: 0.1247 Acc: 0.860 LR: 1.42e-05
logs/agreidv2_airl_4090.log:95:Epoch[5] Iter[500/786] Loss: 4.347 CE: 1.964 Tri: 0.425 CE_rec: 1.895 AIRL_rec: 0.1263 Acc: 0.867 LR: 1.42e-05
logs/agreidv2_airl_4090.log:96:Epoch[5] Iter[550/786] Loss: 4.262 CE: 1.929 Tri: 0.410 CE_rec: 1.860 AIRL_rec: 0.1276 Acc: 0.874 LR: 1.42e-05
logs/agreidv2_airl_4090.log:97:Epoch[5] Iter[600/786] Loss: 4.202 CE: 1.900 Tri: 0.407 CE_rec: 1.830 AIRL_rec: 0.1289 Acc: 0.879 LR: 1.42e-05
logs/agreidv2_airl_4090.log:98:Epoch[5] Iter[650/786] Loss: 4.140 CE: 1.872 Tri: 0.400 CE_rec: 1.802 AIRL_rec: 0.1300 Acc: 0.884 LR: 1.42e-05
logs/agreidv2_airl_4090.log:99:Epoch[5] Iter[700/786] Loss: 4.077 CE: 1.844 Tri: 0.396 CE_rec: 1.772 AIRL_rec: 0.1316 Acc: 0.888 LR: 1.42e-05
logs/agreidv2_airl_4090.log:100:Epoch[5] done in 112.8s  Loss=4.024 Acc=0.893 AIRL-ISO[lam_eff=0.500 ce_rec=1.744 consistency=0.1331 deg_scale_mean=0.625 n_ground=28665]
logs/agreidv2_airl_4090.log:101:Epoch[6] Iter[50/786] Loss: 4.195 CE: 1.879 Tri: 0.392 CE_rec: 1.847 AIRL_rec: 0.1535 Acc: 0.852 LR: 1.77e-05
logs/agreidv2_airl_4090.log:102:Epoch[6] Iter[100/786] Loss: 4.031 CE: 1.826 Tri: 0.350 CE_rec: 1.780 AIRL_rec: 0.1512 Acc: 0.872 LR: 1.77e-05
logs/agreidv2_airl_4090.log:103:Epoch[6] Iter[150/786] Loss: 3.931 CE: 1.783 Tri: 0.342 CE_rec: 1.730 AIRL_rec: 0.1528 Acc: 0.885 LR: 1.77e-05
logs/agreidv2_airl_4090.log:104:Epoch[6] Iter[200/786] Loss: 3.862 CE: 1.748 Tri: 0.346 CE_rec: 1.691 AIRL_rec: 0.1548 Acc: 0.893 LR: 1.77e-05
logs/agreidv2_airl_4090.log:105:Epoch[6] Iter[250/786] Loss: 3.804 CE: 1.720 Tri: 0.342 CE_rec: 1.664 AIRL_rec: 0.1560 Acc: 0.900 LR: 1.77e-05
logs/agreidv2_airl_4090.log:106:Epoch[6] Iter[300/786] Loss: 3.757 CE: 1.698 Tri: 0.339 CE_rec: 1.641 AIRL_rec: 0.1569 Acc: 0.906 LR: 1.77e-05
logs/agreidv2_airl_4090.log:107:Epoch[6] Iter[350/786] Loss: 3.708 CE: 1.678 Tri: 0.331 CE_rec: 1.620 AIRL_rec: 0.1581 Acc: 0.909 LR: 1.77e-05
logs/agreidv2_airl_4090.log:108:Epoch[6] Iter[400/786] Loss: 3.664 CE: 1.661 Tri: 0.323 CE_rec: 1.601 AIRL_rec: 0.1593 Acc: 0.913 LR: 1.77e-05
logs/agreidv2_airl_4090.log:109:Epoch[6] Iter[450/786] Loss: 3.634 CE: 1.647 Tri: 0.324 CE_rec: 1.584 AIRL_rec: 0.1599 Acc: 0.916 LR: 1.77e-05
logs/agreidv2_airl_4090.log:110:Epoch[6] Iter[500/786] Loss: 3.603 CE: 1.634 Tri: 0.319 CE_rec: 1.570 AIRL_rec: 0.1605 Acc: 0.919 LR: 1.77e-05
logs/agreidv2_airl_4090.log:111:Epoch[6] Iter[550/786] Loss: 3.568 CE: 1.619 Tri: 0.315 CE_rec: 1.554 AIRL_rec: 0.1615 Acc: 0.922 LR: 1.77e-05
logs/agreidv2_airl_4090.log:112:Epoch[6] Iter[600/786] Loss: 3.532 CE: 1.604 Tri: 0.308 CE_rec: 1.539 AIRL_rec: 0.1617 Acc: 0.925 LR: 1.77e-05
logs/agreidv2_airl_4090.log:113:Epoch[6] Iter[650/786] Loss: 3.511 CE: 1.594 Tri: 0.308 CE_rec: 1.527 AIRL_rec: 0.1618 Acc: 0.926 LR: 1.77e-05
logs/agreidv2_airl_4090.log:114:Epoch[6] Iter[700/786] Loss: 3.481 CE: 1.581 Tri: 0.306 CE_rec: 1.513 AIRL_rec: 0.1618 Acc: 0.929 LR: 1.77e-05
logs/agreidv2_airl_4090.log:115:Epoch[6] done in 113.9s  Loss=3.443 Acc=0.932 AIRL-ISO[lam_eff=0.500 ce_rec=1.493 consistency=0.1621 deg_scale_mean=0.623 n_ground=28888]
logs/agreidv2_airl_4090.log:116:Epoch[7] Iter[50/786] Loss: 3.624 CE: 1.640 Tri: 0.271 CE_rec: 1.630 AIRL_rec: 0.1658 Acc: 0.907 LR: 2.11e-05
logs/agreidv2_airl_4090.log:117:Epoch[7] Iter[100/786] Loss: 3.553 CE: 1.606 Tri: 0.290 CE_rec: 1.574 AIRL_rec: 0.1665 Acc: 0.920 LR: 2.11e-05
logs/agreidv2_airl_4090.log:118:Epoch[7] Iter[150/786] Loss: 3.494 CE: 1.588 Tri: 0.275 CE_rec: 1.548 AIRL_rec: 0.1676 Acc: 0.923 LR: 2.11e-05
logs/agreidv2_airl_4090.log:119:Epoch[7] Iter[200/786] Loss: 3.446 CE: 1.571 Tri: 0.267 CE_rec: 1.524 AIRL_rec: 0.1692 Acc: 0.926 LR: 2.11e-05
logs/agreidv2_airl_4090.log:120:Epoch[7] Iter[250/786] Loss: 3.409 CE: 1.558 Tri: 0.263 CE_rec: 1.504 AIRL_rec: 0.1691 Acc: 0.929 LR: 2.11e-05
logs/agreidv2_airl_4090.log:121:Epoch[7] Iter[300/786] Loss: 3.420 CE: 1.558 Tri: 0.277 CE_rec: 1.499 AIRL_rec: 0.1711 Acc: 0.930 LR: 2.11e-05
logs/agreidv2_airl_4090.log:122:Epoch[7] Iter[350/786] Loss: 3.406 CE: 1.551 Tri: 0.282 CE_rec: 1.487 AIRL_rec: 0.1728 Acc: 0.932 LR: 2.11e-05
logs/agreidv2_airl_4090.log:123:Epoch[7] Iter[400/786] Loss: 3.404 CE: 1.546 Tri: 0.291 CE_rec: 1.480 AIRL_rec: 0.1744 Acc: 0.933 LR: 2.11e-05
logs/agreidv2_airl_4090.log:124:Epoch[7] Iter[450/786] Loss: 3.390 CE: 1.540 Tri: 0.290 CE_rec: 1.472 AIRL_rec: 0.1761 Acc: 0.934 LR: 2.11e-05
logs/agreidv2_airl_4090.log:125:Epoch[7] Iter[500/786] Loss: 3.376 CE: 1.533 Tri: 0.292 CE_rec: 1.463 AIRL_rec: 0.1760 Acc: 0.936 LR: 2.11e-05
logs/agreidv2_airl_4090.log:126:Epoch[7] Iter[550/786] Loss: 3.356 CE: 1.525 Tri: 0.289 CE_rec: 1.453 AIRL_rec: 0.1761 Acc: 0.937 LR: 2.11e-05
logs/agreidv2_airl_4090.log:127:Epoch[7] Iter[600/786] Loss: 3.333 CE: 1.516 Tri: 0.286 CE_rec: 1.443 AIRL_rec: 0.1760 Acc: 0.939 LR: 2.11e-05
logs/agreidv2_airl_4090.log:128:Epoch[7] Iter[650/786] Loss: 3.309 CE: 1.507 Tri: 0.283 CE_rec: 1.432 AIRL_rec: 0.1758 Acc: 0.941 LR: 2.11e-05
logs/agreidv2_airl_4090.log:129:Epoch[7] Iter[700/786] Loss: 3.288 CE: 1.498 Tri: 0.281 CE_rec: 1.422 AIRL_rec: 0.1749 Acc: 0.942 LR: 2.11e-05
logs/agreidv2_airl_4090.log:130:Epoch[7] done in 112.6s  Loss=3.262 Acc=0.944 AIRL-ISO[lam_eff=0.500 ce_rec=1.410 consistency=0.1741 deg_scale_mean=0.625 n_ground=28646]
logs/agreidv2_airl_4090.log:131:Epoch[8] Iter[50/786] Loss: 3.403 CE: 1.561 Tri: 0.240 CE_rec: 1.517 AIRL_rec: 0.1695 Acc: 0.924 LR: 2.46e-05
logs/agreidv2_airl_4090.log:132:Epoch[8] Iter[100/786] Loss: 3.342 CE: 1.535 Tri: 0.244 CE_rec: 1.478 AIRL_rec: 0.1701 Acc: 0.930 LR: 2.46e-05
logs/agreidv2_airl_4090.log:133:Epoch[8] Iter[150/786] Loss: 3.297 CE: 1.517 Tri: 0.240 CE_rec: 1.453 AIRL_rec: 0.1740 Acc: 0.935 LR: 2.46e-05
logs/agreidv2_airl_4090.log:134:Epoch[8] Iter[200/786] Loss: 3.281 CE: 1.507 Tri: 0.245 CE_rec: 1.440 AIRL_rec: 0.1764 Acc: 0.937 LR: 2.46e-05
logs/agreidv2_airl_4090.log:135:Epoch[8] Iter[250/786] Loss: 3.248 CE: 1.495 Tri: 0.238 CE_rec: 1.426 AIRL_rec: 0.1778 Acc: 0.940 LR: 2.46e-05
logs/agreidv2_airl_4090.log:136:Epoch[8] Iter[300/786] Loss: 3.222 CE: 1.486 Tri: 0.231 CE_rec: 1.415 AIRL_rec: 0.1785 Acc: 0.942 LR: 2.46e-05
logs/agreidv2_airl_4090.log:137:Epoch[8] Iter[350/786] Loss: 3.207 CE: 1.479 Tri: 0.230 CE_rec: 1.408 AIRL_rec: 0.1792 Acc: 0.944 LR: 2.46e-05
logs/agreidv2_airl_4090.log:138:Epoch[8] Iter[400/786] Loss: 3.193 CE: 1.473 Tri: 0.233 CE_rec: 1.398 AIRL_rec: 0.1790 Acc: 0.946 LR: 2.46e-05
logs/agreidv2_airl_4090.log:139:Epoch[8] Iter[450/786] Loss: 3.182 CE: 1.467 Tri: 0.233 CE_rec: 1.392 AIRL_rec: 0.1790 Acc: 0.946 LR: 2.46e-05
logs/agreidv2_airl_4090.log:140:Epoch[8] Iter[500/786] Loss: 3.165 CE: 1.461 Tri: 0.230 CE_rec: 1.385 AIRL_rec: 0.1782 Acc: 0.948 LR: 2.46e-05
logs/agreidv2_airl_4090.log:141:Epoch[8] Iter[550/786] Loss: 3.151 CE: 1.455 Tri: 0.229 CE_rec: 1.378 AIRL_rec: 0.1780 Acc: 0.949 LR: 2.46e-05
logs/agreidv2_airl_4090.log:142:Epoch[8] Iter[600/786] Loss: 3.136 CE: 1.449 Tri: 0.227 CE_rec: 1.371 AIRL_rec: 0.1779 Acc: 0.950 LR: 2.46e-05
logs/agreidv2_airl_4090.log:143:Epoch[8] Iter[650/786] Loss: 3.118 CE: 1.442 Tri: 0.224 CE_rec: 1.364 AIRL_rec: 0.1773 Acc: 0.951 LR: 2.46e-05
logs/agreidv2_airl_4090.log:144:Epoch[8] Iter[700/786] Loss: 3.097 CE: 1.433 Tri: 0.221 CE_rec: 1.355 AIRL_rec: 0.1760 Acc: 0.953 LR: 2.46e-05
logs/agreidv2_airl_4090.log:145:Epoch[8] done in 113.7s  Loss=3.068 Acc=0.955 AIRL-ISO[lam_eff=0.500 ce_rec=1.342 consistency=0.1751 deg_scale_mean=0.624 n_ground=28818]
logs/agreidv2_airl_4090.log:146:Epoch[9] Iter[50/786] Loss: 3.311 CE: 1.513 Tri: 0.232 CE_rec: 1.483 AIRL_rec: 0.1693 Acc: 0.937 LR: 2.81e-05
logs/agreidv2_airl_4090.log:147:Epoch[9] Iter[100/786] Loss: 3.217 CE: 1.478 Tri: 0.224 CE_rec: 1.427 AIRL_rec: 0.1754 Acc: 0.944 LR: 2.81e-05
logs/agreidv2_airl_4090.log:148:Epoch[9] Iter[150/786] Loss: 3.184 CE: 1.467 Tri: 0.222 CE_rec: 1.407 AIRL_rec: 0.1753 Acc: 0.944 LR: 2.81e-05
logs/agreidv2_airl_4090.log:149:Epoch[9] Iter[200/786] Loss: 3.173 CE: 1.462 Tri: 0.227 CE_rec: 1.397 AIRL_rec: 0.1767 Acc: 0.945 LR: 2.81e-05
logs/agreidv2_airl_4090.log:150:Epoch[9] Iter[250/786] Loss: 3.150 CE: 1.452 Tri: 0.224 CE_rec: 1.385 AIRL_rec: 0.1765 Acc: 0.947 LR: 2.81e-05
logs/agreidv2_airl_4090.log:151:Epoch[9] Iter[300/786] Loss: 3.145 CE: 1.451 Tri: 0.224 CE_rec: 1.381 AIRL_rec: 0.1771 Acc: 0.946 LR: 2.81e-05
logs/agreidv2_airl_4090.log:152:Epoch[9] Iter[350/786] Loss: 3.125 CE: 1.445 Tri: 0.219 CE_rec: 1.373 AIRL_rec: 0.1768 Acc: 0.948 LR: 2.81e-05
logs/agreidv2_airl_4090.log:153:Epoch[9] Iter[400/786] Loss: 3.114 CE: 1.441 Tri: 0.215 CE_rec: 1.369 AIRL_rec: 0.1772 Acc: 0.949 LR: 2.81e-05
logs/agreidv2_airl_4090.log:154:Epoch[9] Iter[450/786] Loss: 3.102 CE: 1.437 Tri: 0.213 CE_rec: 1.364 AIRL_rec: 0.1770 Acc: 0.950 LR: 2.81e-05
logs/agreidv2_airl_4090.log:155:Epoch[9] Iter[500/786] Loss: 3.093 CE: 1.432 Tri: 0.214 CE_rec: 1.358 AIRL_rec: 0.1774 Acc: 0.951 LR: 2.81e-05
logs/agreidv2_airl_4090.log:156:Epoch[9] Iter[550/786] Loss: 3.082 CE: 1.427 Tri: 0.215 CE_rec: 1.352 AIRL_rec: 0.1766 Acc: 0.952 LR: 2.81e-05
logs/agreidv2_airl_4090.log:157:Epoch[9] Iter[600/786] Loss: 3.069 CE: 1.422 Tri: 0.214 CE_rec: 1.345 AIRL_rec: 0.1760 Acc: 0.953 LR: 2.81e-05
logs/agreidv2_airl_4090.log:158:Epoch[9] Iter[650/786] Loss: 3.052 CE: 1.415 Tri: 0.212 CE_rec: 1.337 AIRL_rec: 0.1754 Acc: 0.954 LR: 2.81e-05
logs/agreidv2_airl_4090.log:159:Epoch[9] Iter[700/786] Loss: 3.034 CE: 1.408 Tri: 0.210 CE_rec: 1.330 AIRL_rec: 0.1746 Acc: 0.955 LR: 2.81e-05
logs/agreidv2_airl_4090.log:160:Epoch[9] done in 113.4s  Loss=3.009 Acc=0.957 AIRL-ISO[lam_eff=0.500 ce_rec=1.319 consistency=0.1735 deg_scale_mean=0.625 n_ground=28790]
logs/agreidv2_airl_4090.log:161:Epoch[10] Iter[50/786] Loss: 3.217 CE: 1.494 Tri: 0.198 CE_rec: 1.442 AIRL_rec: 0.1660 Acc: 0.935 LR: 3.15e-05
logs/agreidv2_airl_4090.log:162:Epoch[10] Iter[100/786] Loss: 3.163 CE: 1.472 Tri: 0.198 CE_rec: 1.407 AIRL_rec: 0.1713 Acc: 0.940 LR: 3.15e-05
logs/agreidv2_airl_4090.log:163:Epoch[10] Iter[150/786] Loss: 3.115 CE: 1.452 Tri: 0.193 CE_rec: 1.382 AIRL_rec: 0.1748 Acc: 0.946 LR: 3.15e-05
logs/agreidv2_airl_4090.log:164:Epoch[10] Iter[200/786] Loss: 3.101 CE: 1.442 Tri: 0.197 CE_rec: 1.372 AIRL_rec: 0.1787 Acc: 0.947 LR: 3.15e-05
logs/agreidv2_airl_4090.log:165:Epoch[10] Iter[250/786] Loss: 3.111 CE: 1.443 Tri: 0.207 CE_rec: 1.370 AIRL_rec: 0.1799 Acc: 0.946 LR: 3.15e-05
logs/agreidv2_airl_4090.log:166:Epoch[10] Iter[300/786] Loss: 3.097 CE: 1.437 Tri: 0.207 CE_rec: 1.363 AIRL_rec: 0.1808 Acc: 0.948 LR: 3.15e-05
logs/agreidv2_airl_4090.log:167:Epoch[10] Iter[350/786] Loss: 3.089 CE: 1.434 Tri: 0.207 CE_rec: 1.358 AIRL_rec: 0.1815 Acc: 0.949 LR: 3.15e-05
logs/agreidv2_airl_4090.log:168:Epoch[10] Iter[400/786] Loss: 3.081 CE: 1.430 Tri: 0.208 CE_rec: 1.352 AIRL_rec: 0.1811 Acc: 0.949 LR: 3.15e-05
logs/agreidv2_airl_4090.log:169:Epoch[10] Iter[450/786] Loss: 3.071 CE: 1.426 Tri: 0.208 CE_rec: 1.346 AIRL_rec: 0.1803 Acc: 0.950 LR: 3.15e-05
logs/agreidv2_airl_4090.log:170:Epoch[10] Iter[500/786] Loss: 3.065 CE: 1.423 Tri: 0.208 CE_rec: 1.343 AIRL_rec: 0.1808 Acc: 0.951 LR: 3.15e-05
logs/agreidv2_airl_4090.log:171:Epoch[10] Iter[550/786] Loss: 3.051 CE: 1.418 Tri: 0.204 CE_rec: 1.338 AIRL_rec: 0.1800 Acc: 0.952 LR: 3.15e-05
logs/agreidv2_airl_4090.log:172:Epoch[10] Iter[600/786] Loss: 3.039 CE: 1.413 Tri: 0.203 CE_rec: 1.333 AIRL_rec: 0.1797 Acc: 0.953 LR: 3.15e-05
logs/agreidv2_airl_4090.log:173:Epoch[10] Iter[650/786] Loss: 3.022 CE: 1.406 Tri: 0.201 CE_rec: 1.326 AIRL_rec: 0.1784 Acc: 0.954 LR: 3.15e-05
logs/agreidv2_airl_4090.log:174:Epoch[10] Iter[700/786] Loss: 3.003 CE: 1.398 Tri: 0.199 CE_rec: 1.318 AIRL_rec: 0.1769 Acc: 0.956 LR: 3.15e-05
logs/agreidv2_airl_4090.log:175:Epoch[10] done in 113.0s  Loss=2.981 Acc=0.957 AIRL-ISO[lam_eff=0.500 ce_rec=1.309 consistency=0.1752 deg_scale_mean=0.624 n_ground=28708]
logs/agreidv2_airl_4090.log:177:    [A->G] mAP=74.91  R1=83.36  R5=89.98  mINP=46.90
logs/agreidv2_airl_4090.log:178:    [G->A] mAP=74.48  R1=82.94  R5=89.45  mINP=37.71
logs/agreidv2_airl_4090.log:179:    [mean] mAP=74.70  R1=83.15
logs/agreidv2_airl_4090.log:180:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 10 ----
logs/agreidv2_airl_4090.log:181:    [A->G] full mAP=74.91 R1=83.36 | rec mAP=74.08 R1=82.98 | FUSE mAP=75.66 R1=83.74
logs/agreidv2_airl_4090.log:182:    [G->A] full mAP=74.48 R1=82.94 | rec mAP=74.61 R1=82.94 | FUSE mAP=75.57 R1=84.04
logs/agreidv2_airl_4090.log:184:    * new best mean mAP=75.61 (epoch 10) saved
logs/agreidv2_airl_4090.log:185:Epoch[11] Iter[50/786] Loss: 3.172 CE: 1.486 Tri: 0.169 CE_rec: 1.430 AIRL_rec: 0.1741 Acc: 0.932 LR: 3.50e-05
logs/agreidv2_airl_4090.log:186:Epoch[11] Iter[100/786] Loss: 3.127 CE: 1.459 Tri: 0.182 CE_rec: 1.399 AIRL_rec: 0.1745 Acc: 0.941 LR: 3.50e-05
logs/agreidv2_airl_4090.log:187:Epoch[11] Iter[150/786] Loss: 3.078 CE: 1.438 Tri: 0.180 CE_rec: 1.371 AIRL_rec: 0.1776 Acc: 0.947 LR: 3.50e-05
logs/agreidv2_airl_4090.log:188:Epoch[11] Iter[200/786] Loss: 3.061 CE: 1.429 Tri: 0.182 CE_rec: 1.360 AIRL_rec: 0.1791 Acc: 0.949 LR: 3.50e-05
logs/agreidv2_airl_4090.log:189:Epoch[11] Iter[250/786] Loss: 3.060 CE: 1.426 Tri: 0.187 CE_rec: 1.357 AIRL_rec: 0.1795 Acc: 0.949 LR: 3.50e-05
logs/agreidv2_airl_4090.log:190:Epoch[11] Iter[300/786] Loss: 3.049 CE: 1.422 Tri: 0.186 CE_rec: 1.349 AIRL_rec: 0.1822 Acc: 0.950 LR: 3.50e-05
logs/agreidv2_airl_4090.log:191:Epoch[11] Iter[350/786] Loss: 3.034 CE: 1.416 Tri: 0.184 CE_rec: 1.343 AIRL_rec: 0.1825 Acc: 0.951 LR: 3.50e-05
logs/agreidv2_airl_4090.log:192:Epoch[11] Iter[400/786] Loss: 3.016 CE: 1.409 Tri: 0.180 CE_rec: 1.335 AIRL_rec: 0.1820 Acc: 0.952 LR: 3.50e-05
logs/agreidv2_airl_4090.log:193:Epoch[11] Iter[450/786] Loss: 3.005 CE: 1.405 Tri: 0.178 CE_rec: 1.331 AIRL_rec: 0.1815 Acc: 0.953 LR: 3.50e-05
logs/agreidv2_airl_4090.log:194:Epoch[11] Iter[500/786] Loss: 2.993 CE: 1.400 Tri: 0.177 CE_rec: 1.325 AIRL_rec: 0.1814 Acc: 0.954 LR: 3.50e-05
logs/agreidv2_airl_4090.log:195:Epoch[11] Iter[550/786] Loss: 2.977 CE: 1.394 Tri: 0.174 CE_rec: 1.319 AIRL_rec: 0.1804 Acc: 0.956 LR: 3.50e-05
logs/agreidv2_airl_4090.log:196:Epoch[11] Iter[600/786] Loss: 2.962 CE: 1.387 Tri: 0.172 CE_rec: 1.313 AIRL_rec: 0.1800 Acc: 0.957 LR: 3.50e-05
logs/agreidv2_airl_4090.log:197:Epoch[11] Iter[650/786] Loss: 2.948 CE: 1.381 Tri: 0.172 CE_rec: 1.307 AIRL_rec: 0.1786 Acc: 0.959 LR: 3.50e-05
logs/agreidv2_airl_4090.log:198:Epoch[11] Iter[700/786] Loss: 2.930 CE: 1.373 Tri: 0.169 CE_rec: 1.300 AIRL_rec: 0.1767 Acc: 0.960 LR: 3.50e-05
logs/agreidv2_airl_4090.log:199:Epoch[11] done in 113.0s  Loss=2.909 Acc=0.962 AIRL-ISO[lam_eff=0.500 ce_rec=1.291 consistency=0.1750 deg_scale_mean=0.624 n_ground=28664]
logs/agreidv2_airl_4090.log:200:Epoch[12] Iter[50/786] Loss: 3.164 CE: 1.492 Tri: 0.167 CE_rec: 1.417 AIRL_rec: 0.1740 Acc: 0.934 LR: 3.50e-05
logs/agreidv2_airl_4090.log:201:Epoch[12] Iter[100/786] Loss: 3.097 CE: 1.458 Tri: 0.168 CE_rec: 1.383 AIRL_rec: 0.1780 Acc: 0.941 LR: 3.50e-05
logs/agreidv2_airl_4090.log:202:Epoch[12] Iter[150/786] Loss: 3.052 CE: 1.436 Tri: 0.165 CE_rec: 1.361 AIRL_rec: 0.1796 Acc: 0.944 LR: 3.50e-05
logs/agreidv2_airl_4090.log:203:Epoch[12] Iter[200/786] Loss: 3.014 CE: 1.419 Tri: 0.164 CE_rec: 1.342 AIRL_rec: 0.1792 Acc: 0.949 LR: 3.50e-05
logs/agreidv2_airl_4090.log:204:Epoch[12] Iter[250/786] Loss: 2.996 CE: 1.409 Tri: 0.163 CE_rec: 1.333 AIRL_rec: 0.1800 Acc: 0.951 LR: 3.50e-05
logs/agreidv2_airl_4090.log:205:Epoch[12] Iter[300/786] Loss: 2.974 CE: 1.399 Tri: 0.162 CE_rec: 1.322 AIRL_rec: 0.1816 Acc: 0.954 LR: 3.50e-05
logs/agreidv2_airl_4090.log:206:Epoch[12] Iter[350/786] Loss: 2.964 CE: 1.394 Tri: 0.161 CE_rec: 1.317 AIRL_rec: 0.1821 Acc: 0.955 LR: 3.50e-05
logs/agreidv2_airl_4090.log:207:Epoch[12] Iter[400/786] Loss: 2.959 CE: 1.390 Tri: 0.164 CE_rec: 1.313 AIRL_rec: 0.1821 Acc: 0.956 LR: 3.50e-05
logs/agreidv2_airl_4090.log:208:Epoch[12] Iter[450/786] Loss: 2.956 CE: 1.388 Tri: 0.167 CE_rec: 1.310 AIRL_rec: 0.1824 Acc: 0.956 LR: 3.50e-05
logs/agreidv2_airl_4090.log:209:Epoch[12] Iter[500/786] Loss: 2.942 CE: 1.382 Tri: 0.163 CE_rec: 1.305 AIRL_rec: 0.1817 Acc: 0.957 LR: 3.50e-05
logs/agreidv2_airl_4090.log:210:Epoch[12] Iter[550/786] Loss: 2.936 CE: 1.379 Tri: 0.164 CE_rec: 1.302 AIRL_rec: 0.1808 Acc: 0.958 LR: 3.50e-05
logs/agreidv2_airl_4090.log:211:Epoch[12] Iter[600/786] Loss: 2.923 CE: 1.374 Tri: 0.162 CE_rec: 1.297 AIRL_rec: 0.1799 Acc: 0.959 LR: 3.50e-05
logs/agreidv2_airl_4090.log:212:Epoch[12] Iter[650/786] Loss: 2.912 CE: 1.369 Tri: 0.162 CE_rec: 1.292 AIRL_rec: 0.1778 Acc: 0.960 LR: 3.50e-05
logs/agreidv2_airl_4090.log:213:Epoch[12] Iter[700/786] Loss: 2.893 CE: 1.361 Tri: 0.160 CE_rec: 1.285 AIRL_rec: 0.1756 Acc: 0.961 LR: 3.50e-05
logs/agreidv2_airl_4090.log:214:Epoch[12] done in 113.7s  Loss=2.870 Acc=0.963 AIRL-ISO[lam_eff=0.500 ce_rec=1.276 consistency=0.1736 deg_scale_mean=0.624 n_ground=28796]
logs/agreidv2_airl_4090.log:215:Epoch[13] Iter[50/786] Loss: 3.079 CE: 1.448 Tri: 0.164 CE_rec: 1.384 AIRL_rec: 0.1644 Acc: 0.940 LR: 3.49e-05
logs/agreidv2_airl_4090.log:216:Epoch[13] Iter[100/786] Loss: 3.035 CE: 1.431 Tri: 0.160 CE_rec: 1.360 AIRL_rec: 0.1677 Acc: 0.944 LR: 3.49e-05
logs/agreidv2_airl_4090.log:217:Epoch[13] Iter[150/786] Loss: 2.990 CE: 1.414 Tri: 0.149 CE_rec: 1.342 AIRL_rec: 0.1712 Acc: 0.947 LR: 3.49e-05
logs/agreidv2_airl_4090.log:218:Epoch[13] Iter[200/786] Loss: 2.957 CE: 1.399 Tri: 0.145 CE_rec: 1.327 AIRL_rec: 0.1709 Acc: 0.951 LR: 3.49e-05
logs/agreidv2_airl_4090.log:219:Epoch[13] Iter[250/786] Loss: 2.924 CE: 1.385 Tri: 0.142 CE_rec: 1.311 AIRL_rec: 0.1703 Acc: 0.955 LR: 3.49e-05
logs/agreidv2_airl_4090.log:220:Epoch[13] Iter[300/786] Loss: 2.905 CE: 1.377 Tri: 0.140 CE_rec: 1.302 AIRL_rec: 0.1714 Acc: 0.957 LR: 3.49e-05
logs/agreidv2_airl_4090.log:221:Epoch[13] Iter[350/786] Loss: 2.894 CE: 1.372 Tri: 0.139 CE_rec: 1.297 AIRL_rec: 0.1717 Acc: 0.957 LR: 3.49e-05
logs/agreidv2_airl_4090.log:222:Epoch[13] Iter[400/786] Loss: 2.886 CE: 1.367 Tri: 0.141 CE_rec: 1.292 AIRL_rec: 0.1722 Acc: 0.958 LR: 3.49e-05
logs/agreidv2_airl_4090.log:223:Epoch[13] Iter[450/786] Loss: 2.875 CE: 1.363 Tri: 0.139 CE_rec: 1.287 AIRL_rec: 0.1723 Acc: 0.959 LR: 3.49e-05
logs/agreidv2_airl_4090.log:224:Epoch[13] Iter[500/786] Loss: 2.860 CE: 1.357 Tri: 0.136 CE_rec: 1.281 AIRL_rec: 0.1711 Acc: 0.961 LR: 3.49e-05
logs/agreidv2_airl_4090.log:225:Epoch[13] Iter[550/786] Loss: 2.847 CE: 1.351 Tri: 0.134 CE_rec: 1.276 AIRL_rec: 0.1706 Acc: 0.961 LR: 3.49e-05
logs/agreidv2_airl_4090.log:226:Epoch[13] Iter[600/786] Loss: 2.842 CE: 1.348 Tri: 0.137 CE_rec: 1.273 AIRL_rec: 0.1693 Acc: 0.962 LR: 3.49e-05
logs/agreidv2_airl_4090.log:227:Epoch[13] Iter[650/786] Loss: 2.827 CE: 1.342 Tri: 0.133 CE_rec: 1.268 AIRL_rec: 0.1677 Acc: 0.963 LR: 3.49e-05
logs/agreidv2_airl_4090.log:228:Epoch[13] Iter[700/786] Loss: 2.812 CE: 1.335 Tri: 0.132 CE_rec: 1.262 AIRL_rec: 0.1662 Acc: 0.964 LR: 3.49e-05
logs/agreidv2_airl_4090.log:229:Epoch[13] done in 112.5s  Loss=2.796 Acc=0.965 AIRL-ISO[lam_eff=0.500 ce_rec=1.255 consistency=0.1648 deg_scale_mean=0.625 n_ground=28553]
logs/agreidv2_airl_4090.log:230:Epoch[14] Iter[50/786] Loss: 3.001 CE: 1.431 Tri: 0.142 CE_rec: 1.351 AIRL_rec: 0.1554 Acc: 0.945 LR: 3.47e-05
logs/agreidv2_airl_4090.log:231:Epoch[14] Iter[100/786] Loss: 2.906 CE: 1.389 Tri: 0.125 CE_rec: 1.311 AIRL_rec: 0.1619 Acc: 0.956 LR: 3.47e-05
logs/agreidv2_airl_4090.log:232:Epoch[14] Iter[150/786] Loss: 2.876 CE: 1.374 Tri: 0.125 CE_rec: 1.295 AIRL_rec: 0.1636 Acc: 0.958 LR: 3.47e-05
logs/agreidv2_airl_4090.log:233:Epoch[14] Iter[200/786] Loss: 2.862 CE: 1.366 Tri: 0.126 CE_rec: 1.288 AIRL_rec: 0.1650 Acc: 0.959 LR: 3.47e-05
logs/agreidv2_airl_4090.log:234:Epoch[14] Iter[250/786] Loss: 2.842 CE: 1.356 Tri: 0.123 CE_rec: 1.281 AIRL_rec: 0.1641 Acc: 0.961 LR: 3.47e-05
logs/agreidv2_airl_4090.log:235:Epoch[14] Iter[300/786] Loss: 2.827 CE: 1.349 Tri: 0.120 CE_rec: 1.275 AIRL_rec: 0.1649 Acc: 0.962 LR: 3.47e-05
logs/agreidv2_airl_4090.log:236:Epoch[14] Iter[350/786] Loss: 2.820 CE: 1.345 Tri: 0.121 CE_rec: 1.272 AIRL_rec: 0.1643 Acc: 0.963 LR: 3.47e-05
logs/agreidv2_airl_4090.log:237:Epoch[14] Iter[400/786] Loss: 2.812 CE: 1.341 Tri: 0.121 CE_rec: 1.268 AIRL_rec: 0.1640 Acc: 0.963 LR: 3.47e-05
logs/agreidv2_airl_4090.log:238:Epoch[14] Iter[450/786] Loss: 2.804 CE: 1.337 Tri: 0.121 CE_rec: 1.264 AIRL_rec: 0.1634 Acc: 0.964 LR: 3.47e-05
logs/agreidv2_airl_4090.log:239:Epoch[14] Iter[500/786] Loss: 2.793 CE: 1.333 Tri: 0.119 CE_rec: 1.259 AIRL_rec: 0.1628 Acc: 0.964 LR: 3.47e-05
logs/agreidv2_airl_4090.log:240:Epoch[14] Iter[550/786] Loss: 2.789 CE: 1.330 Tri: 0.122 CE_rec: 1.257 AIRL_rec: 0.1618 Acc: 0.965 LR: 3.47e-05
logs/agreidv2_airl_4090.log:241:Epoch[14] Iter[600/786] Loss: 2.781 CE: 1.325 Tri: 0.122 CE_rec: 1.252 AIRL_rec: 0.1611 Acc: 0.966 LR: 3.47e-05
logs/agreidv2_airl_4090.log:242:Epoch[14] Iter[650/786] Loss: 2.769 CE: 1.320 Tri: 0.121 CE_rec: 1.248 AIRL_rec: 0.1599 Acc: 0.967 LR: 3.47e-05
logs/agreidv2_airl_4090.log:243:Epoch[14] Iter[700/786] Loss: 2.759 CE: 1.315 Tri: 0.121 CE_rec: 1.243 AIRL_rec: 0.1589 Acc: 0.968 LR: 3.47e-05
logs/agreidv2_airl_4090.log:244:Epoch[14] done in 113.7s  Loss=2.743 Acc=0.969 AIRL-ISO[lam_eff=0.500 ce_rec=1.236 consistency=0.1572 deg_scale_mean=0.626 n_ground=28766]
logs/agreidv2_airl_4090.log:245:Epoch[15] Iter[50/786] Loss: 3.011 CE: 1.429 Tri: 0.146 CE_rec: 1.356 AIRL_rec: 0.1618 Acc: 0.947 LR: 3.45e-05
logs/agreidv2_airl_4090.log:246:Epoch[15] Iter[100/786] Loss: 2.933 CE: 1.395 Tri: 0.135 CE_rec: 1.321 AIRL_rec: 0.1654 Acc: 0.951 LR: 3.45e-05
logs/agreidv2_airl_4090.log:247:Epoch[15] Iter[150/786] Loss: 2.880 CE: 1.374 Tri: 0.125 CE_rec: 1.299 AIRL_rec: 0.1635 Acc: 0.958 LR: 3.45e-05
logs/agreidv2_airl_4090.log:248:Epoch[15] Iter[200/786] Loss: 2.839 CE: 1.357 Tri: 0.118 CE_rec: 1.282 AIRL_rec: 0.1645 Acc: 0.961 LR: 3.45e-05
logs/agreidv2_airl_4090.log:249:Epoch[15] Iter[250/786] Loss: 2.821 CE: 1.348 Tri: 0.117 CE_rec: 1.274 AIRL_rec: 0.1633 Acc: 0.963 LR: 3.45e-05
logs/agreidv2_airl_4090.log:250:Epoch[15] Iter[300/786] Loss: 2.811 CE: 1.342 Tri: 0.117 CE_rec: 1.270 AIRL_rec: 0.1645 Acc: 0.963 LR: 3.45e-05
logs/agreidv2_airl_4090.log:251:Epoch[15] Iter[350/786] Loss: 2.805 CE: 1.339 Tri: 0.116 CE_rec: 1.267 AIRL_rec: 0.1647 Acc: 0.963 LR: 3.45e-05
logs/agreidv2_airl_4090.log:252:Epoch[15] Iter[400/786] Loss: 2.792 CE: 1.334 Tri: 0.114 CE_rec: 1.262 AIRL_rec: 0.1634 Acc: 0.964 LR: 3.45e-05
logs/agreidv2_airl_4090.log:253:Epoch[15] Iter[450/786] Loss: 2.786 CE: 1.331 Tri: 0.114 CE_rec: 1.260 AIRL_rec: 0.1631 Acc: 0.964 LR: 3.45e-05
logs/agreidv2_airl_4090.log:254:Epoch[15] Iter[500/786] Loss: 2.774 CE: 1.326 Tri: 0.112 CE_rec: 1.255 AIRL_rec: 0.1626 Acc: 0.965 LR: 3.45e-05
logs/agreidv2_airl_4090.log:255:Epoch[15] Iter[550/786] Loss: 2.764 CE: 1.322 Tri: 0.111 CE_rec: 1.251 AIRL_rec: 0.1614 Acc: 0.966 LR: 3.45e-05
logs/agreidv2_airl_4090.log:256:Epoch[15] Iter[600/786] Loss: 2.753 CE: 1.317 Tri: 0.110 CE_rec: 1.246 AIRL_rec: 0.1603 Acc: 0.967 LR: 3.45e-05
logs/agreidv2_airl_4090.log:257:Epoch[15] Iter[650/786] Loss: 2.743 CE: 1.311 Tri: 0.111 CE_rec: 1.241 AIRL_rec: 0.1588 Acc: 0.968 LR: 3.45e-05
logs/agreidv2_airl_4090.log:258:Epoch[15] Iter[700/786] Loss: 2.726 CE: 1.304 Tri: 0.108 CE_rec: 1.235 AIRL_rec: 0.1567 Acc: 0.969 LR: 3.45e-05
logs/agreidv2_airl_4090.log:259:Epoch[15] done in 113.7s  Loss=2.706 Acc=0.971 AIRL-ISO[lam_eff=0.500 ce_rec=1.228 consistency=0.1542 deg_scale_mean=0.623 n_ground=28765]
logs/agreidv2_airl_4090.log:260:Epoch[16] Iter[50/786] Loss: 2.864 CE: 1.374 Tri: 0.118 CE_rec: 1.296 AIRL_rec: 0.1518 Acc: 0.956 LR: 3.41e-05
logs/agreidv2_airl_4090.log:261:Epoch[16] Iter[100/786] Loss: 2.809 CE: 1.348 Tri: 0.107 CE_rec: 1.275 AIRL_rec: 0.1587 Acc: 0.961 LR: 3.41e-05
logs/agreidv2_airl_4090.log:262:Epoch[16] Iter[150/786] Loss: 2.787 CE: 1.333 Tri: 0.111 CE_rec: 1.263 AIRL_rec: 0.1619 Acc: 0.965 LR: 3.41e-05
logs/agreidv2_airl_4090.log:263:Epoch[16] Iter[200/786] Loss: 2.773 CE: 1.328 Tri: 0.107 CE_rec: 1.256 AIRL_rec: 0.1613 Acc: 0.965 LR: 3.41e-05
logs/agreidv2_airl_4090.log:264:Epoch[16] Iter[250/786] Loss: 2.760 CE: 1.322 Tri: 0.108 CE_rec: 1.251 AIRL_rec: 0.1604 Acc: 0.966 LR: 3.41e-05
logs/agreidv2_airl_4090.log:265:Epoch[16] Iter[300/786] Loss: 2.750 CE: 1.317 Tri: 0.109 CE_rec: 1.245 AIRL_rec: 0.1592 Acc: 0.968 LR: 3.41e-05
logs/agreidv2_airl_4090.log:266:Epoch[16] Iter[350/786] Loss: 2.744 CE: 1.313 Tri: 0.110 CE_rec: 1.242 AIRL_rec: 0.1583 Acc: 0.968 LR: 3.41e-05
logs/agreidv2_airl_4090.log:267:Epoch[16] Iter[400/786] Loss: 2.736 CE: 1.310 Tri: 0.109 CE_rec: 1.239 AIRL_rec: 0.1577 Acc: 0.968 LR: 3.41e-05
logs/agreidv2_airl_4090.log:268:Epoch[16] Iter[450/786] Loss: 2.731 CE: 1.306 Tri: 0.111 CE_rec: 1.236 AIRL_rec: 0.1571 Acc: 0.969 LR: 3.41e-05
logs/agreidv2_airl_4090.log:269:Epoch[16] Iter[500/786] Loss: 2.723 CE: 1.303 Tri: 0.109 CE_rec: 1.233 AIRL_rec: 0.1567 Acc: 0.969 LR: 3.41e-05
logs/agreidv2_airl_4090.log:270:Epoch[16] Iter[550/786] Loss: 2.715 CE: 1.299 Tri: 0.109 CE_rec: 1.229 AIRL_rec: 0.1555 Acc: 0.970 LR: 3.41e-05
logs/agreidv2_airl_4090.log:271:Epoch[16] Iter[600/786] Loss: 2.700 CE: 1.293 Tri: 0.105 CE_rec: 1.224 AIRL_rec: 0.1540 Acc: 0.971 LR: 3.41e-05
logs/agreidv2_airl_4090.log:272:Epoch[16] Iter[650/786] Loss: 2.691 CE: 1.289 Tri: 0.105 CE_rec: 1.221 AIRL_rec: 0.1526 Acc: 0.971 LR: 3.41e-05
logs/agreidv2_airl_4090.log:273:Epoch[16] Iter[700/786] Loss: 2.679 CE: 1.283 Tri: 0.104 CE_rec: 1.216 AIRL_rec: 0.1509 Acc: 0.972 LR: 3.41e-05
logs/agreidv2_airl_4090.log:274:Epoch[16] done in 113.8s  Loss=2.661 Acc=0.974 AIRL-ISO[lam_eff=0.500 ce_rec=1.209 consistency=0.1487 deg_scale_mean=0.625 n_ground=28770]
logs/agreidv2_airl_4090.log:275:Epoch[17] Iter[50/786] Loss: 2.867 CE: 1.379 Tri: 0.118 CE_rec: 1.297 AIRL_rec: 0.1462 Acc: 0.953 LR: 3.38e-05
logs/agreidv2_airl_4090.log:276:Epoch[17] Iter[100/786] Loss: 2.804 CE: 1.349 Tri: 0.109 CE_rec: 1.271 AIRL_rec: 0.1492 Acc: 0.960 LR: 3.38e-05
logs/agreidv2_airl_4090.log:277:Epoch[17] Iter[150/786] Loss: 2.787 CE: 1.339 Tri: 0.109 CE_rec: 1.263 AIRL_rec: 0.1517 Acc: 0.963 LR: 3.38e-05
logs/agreidv2_airl_4090.log:278:Epoch[17] Iter[200/786] Loss: 2.783 CE: 1.334 Tri: 0.112 CE_rec: 1.260 AIRL_rec: 0.1534 Acc: 0.963 LR: 3.38e-05
logs/agreidv2_airl_4090.log:279:Epoch[17] Iter[250/786] Loss: 2.750 CE: 1.321 Tri: 0.105 CE_rec: 1.247 AIRL_rec: 0.1535 Acc: 0.966 LR: 3.38e-05
logs/agreidv2_airl_4090.log:280:Epoch[17] Iter[300/786] Loss: 2.729 CE: 1.312 Tri: 0.102 CE_rec: 1.239 AIRL_rec: 0.1543 Acc: 0.968 LR: 3.38e-05
logs/agreidv2_airl_4090.log:281:Epoch[17] Iter[350/786] Loss: 2.714 CE: 1.305 Tri: 0.100 CE_rec: 1.232 AIRL_rec: 0.1538 Acc: 0.968 LR: 3.38e-05
logs/agreidv2_airl_4090.log:282:Epoch[17] Iter[400/786] Loss: 2.705 CE: 1.301 Tri: 0.100 CE_rec: 1.227 AIRL_rec: 0.1538 Acc: 0.969 LR: 3.38e-05
logs/agreidv2_airl_4090.log:283:Epoch[17] Iter[450/786] Loss: 2.696 CE: 1.296 Tri: 0.100 CE_rec: 1.223 AIRL_rec: 0.1536 Acc: 0.970 LR: 3.38e-05
logs/agreidv2_airl_4090.log:284:Epoch[17] Iter[500/786] Loss: 2.686 CE: 1.291 Tri: 0.098 CE_rec: 1.220 AIRL_rec: 0.1531 Acc: 0.971 LR: 3.38e-05
logs/agreidv2_airl_4090.log:285:Epoch[17] Iter[550/786] Loss: 2.674 CE: 1.287 Tri: 0.096 CE_rec: 1.216 AIRL_rec: 0.1521 Acc: 0.972 LR: 3.38e-05
logs/agreidv2_airl_4090.log:286:Epoch[17] Iter[600/786] Loss: 2.665 CE: 1.283 Tri: 0.095 CE_rec: 1.212 AIRL_rec: 0.1505 Acc: 0.972 LR: 3.38e-05
logs/agreidv2_airl_4090.log:287:Epoch[17] Iter[650/786] Loss: 2.656 CE: 1.278 Tri: 0.095 CE_rec: 1.209 AIRL_rec: 0.1488 Acc: 0.973 LR: 3.38e-05
logs/agreidv2_airl_4090.log:288:Epoch[17] Iter[700/786] Loss: 2.644 CE: 1.273 Tri: 0.093 CE_rec: 1.205 AIRL_rec: 0.1466 Acc: 0.974 LR: 3.38e-05
logs/agreidv2_airl_4090.log:289:Epoch[17] done in 113.3s  Loss=2.627 Acc=0.975 AIRL-ISO[lam_eff=0.500 ce_rec=1.198 consistency=0.1444 deg_scale_mean=0.625 n_ground=28701]
logs/agreidv2_airl_4090.log:290:Epoch[18] Iter[50/786] Loss: 2.812 CE: 1.345 Tri: 0.106 CE_rec: 1.290 AIRL_rec: 0.1419 Acc: 0.960 LR: 3.33e-05
logs/agreidv2_airl_4090.log:291:Epoch[18] Iter[100/786] Loss: 2.750 CE: 1.316 Tri: 0.102 CE_rec: 1.259 AIRL_rec: 0.1457 Acc: 0.966 LR: 3.33e-05
logs/agreidv2_airl_4090.log:292:Epoch[18] Iter[150/786] Loss: 2.731 CE: 1.309 Tri: 0.103 CE_rec: 1.247 AIRL_rec: 0.1457 Acc: 0.968 LR: 3.33e-05
logs/agreidv2_airl_4090.log:293:Epoch[18] Iter[200/786] Loss: 2.703 CE: 1.297 Tri: 0.096 CE_rec: 1.235 AIRL_rec: 0.1488 Acc: 0.970 LR: 3.33e-05
logs/agreidv2_airl_4090.log:294:Epoch[18] Iter[250/786] Loss: 2.694 CE: 1.292 Tri: 0.096 CE_rec: 1.231 AIRL_rec: 0.1486 Acc: 0.970 LR: 3.33e-05
logs/agreidv2_airl_4090.log:295:Epoch[18] Iter[300/786] Loss: 2.679 CE: 1.286 Tri: 0.093 CE_rec: 1.225 AIRL_rec: 0.1491 Acc: 0.971 LR: 3.33e-05
logs/agreidv2_airl_4090.log:296:Epoch[18] Iter[350/786] Loss: 2.672 CE: 1.283 Tri: 0.092 CE_rec: 1.222 AIRL_rec: 0.1494 Acc: 0.971 LR: 3.33e-05
logs/agreidv2_airl_4090.log:297:Epoch[18] Iter[400/786] Loss: 2.656 CE: 1.277 Tri: 0.089 CE_rec: 1.216 AIRL_rec: 0.1485 Acc: 0.972 LR: 3.33e-05
logs/agreidv2_airl_4090.log:298:Epoch[18] Iter[450/786] Loss: 2.649 CE: 1.275 Tri: 0.088 CE_rec: 1.213 AIRL_rec: 0.1473 Acc: 0.972 LR: 3.33e-05
logs/agreidv2_airl_4090.log:299:Epoch[18] Iter[500/786] Loss: 2.637 CE: 1.270 Tri: 0.085 CE_rec: 1.209 AIRL_rec: 0.1465 Acc: 0.973 LR: 3.33e-05
logs/agreidv2_airl_4090.log:300:Epoch[18] Iter[550/786] Loss: 2.628 CE: 1.266 Tri: 0.085 CE_rec: 1.205 AIRL_rec: 0.1454 Acc: 0.974 LR: 3.33e-05
logs/agreidv2_airl_4090.log:301:Epoch[18] Iter[600/786] Loss: 2.621 CE: 1.262 Tri: 0.085 CE_rec: 1.201 AIRL_rec: 0.1443 Acc: 0.975 LR: 3.33e-05
logs/agreidv2_airl_4090.log:302:Epoch[18] Iter[650/786] Loss: 2.606 CE: 1.256 Tri: 0.083 CE_rec: 1.196 AIRL_rec: 0.1424 Acc: 0.976 LR: 3.33e-05
logs/agreidv2_airl_4090.log:303:Epoch[18] Iter[700/786] Loss: 2.591 CE: 1.250 Tri: 0.081 CE_rec: 1.190 AIRL_rec: 0.1402 Acc: 0.977 LR: 3.33e-05
logs/agreidv2_airl_4090.log:304:Epoch[18] done in 113.0s  Loss=2.578 Acc=0.978 AIRL-ISO[lam_eff=0.500 ce_rec=1.185 consistency=0.1383 deg_scale_mean=0.623 n_ground=28678]
logs/agreidv2_airl_4090.log:305:Epoch[19] Iter[50/786] Loss: 2.658 CE: 1.298 Tri: 0.064 CE_rec: 1.230 AIRL_rec: 0.1323 Acc: 0.969 LR: 3.28e-05
logs/agreidv2_airl_4090.log:306:Epoch[19] Iter[100/786] Loss: 2.670 CE: 1.295 Tri: 0.079 CE_rec: 1.228 AIRL_rec: 0.1351 Acc: 0.970 LR: 3.28e-05
logs/agreidv2_airl_4090.log:307:Epoch[19] Iter[150/786] Loss: 2.648 CE: 1.284 Tri: 0.079 CE_rec: 1.218 AIRL_rec: 0.1358 Acc: 0.972 LR: 3.28e-05
logs/agreidv2_airl_4090.log:308:Epoch[19] Iter[200/786] Loss: 2.646 CE: 1.281 Tri: 0.080 CE_rec: 1.216 AIRL_rec: 0.1395 Acc: 0.971 LR: 3.28e-05
logs/agreidv2_airl_4090.log:309:Epoch[19] Iter[250/786] Loss: 2.634 CE: 1.274 Tri: 0.079 CE_rec: 1.210 AIRL_rec: 0.1393 Acc: 0.973 LR: 3.28e-05
logs/agreidv2_airl_4090.log:310:Epoch[19] Iter[300/786] Loss: 2.621 CE: 1.269 Tri: 0.078 CE_rec: 1.204 AIRL_rec: 0.1400 Acc: 0.974 LR: 3.28e-05
logs/agreidv2_airl_4090.log:311:Epoch[19] Iter[350/786] Loss: 2.610 CE: 1.265 Tri: 0.075 CE_rec: 1.199 AIRL_rec: 0.1401 Acc: 0.975 LR: 3.28e-05
logs/agreidv2_airl_4090.log:312:Epoch[19] Iter[400/786] Loss: 2.599 CE: 1.260 Tri: 0.073 CE_rec: 1.196 AIRL_rec: 0.1396 Acc: 0.976 LR: 3.28e-05
logs/agreidv2_airl_4090.log:313:Epoch[19] Iter[450/786] Loss: 2.588 CE: 1.255 Tri: 0.072 CE_rec: 1.191 AIRL_rec: 0.1388 Acc: 0.977 LR: 3.28e-05
logs/agreidv2_airl_4090.log:314:Epoch[19] Iter[500/786] Loss: 2.580 CE: 1.251 Tri: 0.071 CE_rec: 1.188 AIRL_rec: 0.1377 Acc: 0.978 LR: 3.28e-05
logs/agreidv2_airl_4090.log:315:Epoch[19] Iter[550/786] Loss: 2.573 CE: 1.248 Tri: 0.070 CE_rec: 1.186 AIRL_rec: 0.1370 Acc: 0.978 LR: 3.28e-05
logs/agreidv2_airl_4090.log:316:Epoch[19] Iter[600/786] Loss: 2.567 CE: 1.245 Tri: 0.070 CE_rec: 1.184 AIRL_rec: 0.1359 Acc: 0.978 LR: 3.28e-05
logs/agreidv2_airl_4090.log:317:Epoch[19] Iter[650/786] Loss: 2.559 CE: 1.241 Tri: 0.069 CE_rec: 1.181 AIRL_rec: 0.1346 Acc: 0.979 LR: 3.28e-05
logs/agreidv2_airl_4090.log:318:Epoch[19] Iter[700/786] Loss: 2.547 CE: 1.237 Tri: 0.068 CE_rec: 1.176 AIRL_rec: 0.1329 Acc: 0.979 LR: 3.28e-05
logs/agreidv2_airl_4090.log:319:Epoch[19] done in 113.0s  Loss=2.536 Acc=0.980 AIRL-ISO[lam_eff=0.500 ce_rec=1.172 consistency=0.1314 deg_scale_mean=0.624 n_ground=28643]
logs/agreidv2_airl_4090.log:320:Epoch[20] Iter[50/786] Loss: 2.721 CE: 1.325 Tri: 0.079 CE_rec: 1.252 AIRL_rec: 0.1304 Acc: 0.963 LR: 3.23e-05
logs/agreidv2_airl_4090.log:321:Epoch[20] Iter[100/786] Loss: 2.657 CE: 1.291 Tri: 0.076 CE_rec: 1.224 AIRL_rec: 0.1325 Acc: 0.970 LR: 3.23e-05
logs/agreidv2_airl_4090.log:322:Epoch[20] Iter[150/786] Loss: 2.631 CE: 1.278 Tri: 0.076 CE_rec: 1.210 AIRL_rec: 0.1344 Acc: 0.972 LR: 3.23e-05
logs/agreidv2_airl_4090.log:323:Epoch[20] Iter[200/786] Loss: 2.627 CE: 1.274 Tri: 0.080 CE_rec: 1.206 AIRL_rec: 0.1359 Acc: 0.972 LR: 3.23e-05
logs/agreidv2_airl_4090.log:324:Epoch[20] Iter[250/786] Loss: 2.613 CE: 1.268 Tri: 0.076 CE_rec: 1.201 AIRL_rec: 0.1358 Acc: 0.973 LR: 3.23e-05
logs/agreidv2_airl_4090.log:325:Epoch[20] Iter[300/786] Loss: 2.603 CE: 1.264 Tri: 0.074 CE_rec: 1.197 AIRL_rec: 0.1361 Acc: 0.973 LR: 3.23e-05
logs/agreidv2_airl_4090.log:326:Epoch[20] Iter[350/786] Loss: 2.595 CE: 1.260 Tri: 0.074 CE_rec: 1.194 AIRL_rec: 0.1350 Acc: 0.974 LR: 3.23e-05
logs/agreidv2_airl_4090.log:327:Epoch[20] Iter[400/786] Loss: 2.590 CE: 1.257 Tri: 0.074 CE_rec: 1.192 AIRL_rec: 0.1346 Acc: 0.974 LR: 3.23e-05
logs/agreidv2_airl_4090.log:328:Epoch[20] Iter[450/786] Loss: 2.583 CE: 1.254 Tri: 0.074 CE_rec: 1.189 AIRL_rec: 0.1340 Acc: 0.975 LR: 3.23e-05
logs/agreidv2_airl_4090.log:329:Epoch[20] Iter[500/786] Loss: 2.577 CE: 1.251 Tri: 0.073 CE_rec: 1.186 AIRL_rec: 0.1335 Acc: 0.975 LR: 3.23e-05
logs/agreidv2_airl_4090.log:330:Epoch[20] Iter[550/786] Loss: 2.568 CE: 1.247 Tri: 0.072 CE_rec: 1.183 AIRL_rec: 0.1326 Acc: 0.976 LR: 3.23e-05
logs/agreidv2_airl_4090.log:331:Epoch[20] Iter[600/786] Loss: 2.559 CE: 1.243 Tri: 0.071 CE_rec: 1.180 AIRL_rec: 0.1314 Acc: 0.977 LR: 3.23e-05
logs/agreidv2_airl_4090.log:332:Epoch[20] Iter[650/786] Loss: 2.549 CE: 1.239 Tri: 0.069 CE_rec: 1.176 AIRL_rec: 0.1295 Acc: 0.977 LR: 3.23e-05
logs/agreidv2_airl_4090.log:333:Epoch[20] Iter[700/786] Loss: 2.536 CE: 1.233 Tri: 0.067 CE_rec: 1.172 AIRL_rec: 0.1274 Acc: 0.978 LR: 3.23e-05
logs/agreidv2_airl_4090.log:334:Epoch[20] done in 113.5s  Loss=2.520 Acc=0.979 AIRL-ISO[lam_eff=0.500 ce_rec=1.166 consistency=0.1255 deg_scale_mean=0.625 n_ground=28701]
logs/agreidv2_airl_4090.log:336:    [A->G] mAP=72.59  R1=83.11  R5=89.52  mINP=41.54
logs/agreidv2_airl_4090.log:337:    [G->A] mAP=73.67  R1=83.16  R5=89.45  mINP=35.62
logs/agreidv2_airl_4090.log:338:    [mean] mAP=73.13  R1=83.13
logs/agreidv2_airl_4090.log:339:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 20 ----
logs/agreidv2_airl_4090.log:340:    [A->G] full mAP=72.59 R1=83.11 | rec mAP=74.54 R1=83.62 | FUSE mAP=74.44 R1=84.30
logs/agreidv2_airl_4090.log:341:    [G->A] full mAP=73.67 R1=83.16 | rec mAP=73.89 R1=83.27 | FUSE mAP=75.19 R1=83.99
logs/agreidv2_airl_4090.log:343:Epoch[21] Iter[50/786] Loss: 2.709 CE: 1.315 Tri: 0.082 CE_rec: 1.249 AIRL_rec: 0.1267 Acc: 0.961 LR: 3.17e-05
logs/agreidv2_airl_4090.log:344:Epoch[21] Iter[100/786] Loss: 2.640 CE: 1.283 Tri: 0.070 CE_rec: 1.222 AIRL_rec: 0.1312 Acc: 0.970 LR: 3.17e-05
logs/agreidv2_airl_4090.log:345:Epoch[21] Iter[150/786] Loss: 2.612 CE: 1.269 Tri: 0.068 CE_rec: 1.209 AIRL_rec: 0.1319 Acc: 0.971 LR: 3.17e-05
logs/agreidv2_airl_4090.log:346:Epoch[21] Iter[200/786] Loss: 2.585 CE: 1.257 Tri: 0.066 CE_rec: 1.197 AIRL_rec: 0.1313 Acc: 0.975 LR: 3.17e-05
logs/agreidv2_airl_4090.log:347:Epoch[21] Iter[250/786] Loss: 2.578 CE: 1.252 Tri: 0.068 CE_rec: 1.192 AIRL_rec: 0.1305 Acc: 0.975 LR: 3.17e-05
logs/agreidv2_airl_4090.log:348:Epoch[21] Iter[300/786] Loss: 2.563 CE: 1.246 Tri: 0.066 CE_rec: 1.186 AIRL_rec: 0.1304 Acc: 0.977 LR: 3.17e-05
logs/agreidv2_airl_4090.log:349:Epoch[21] Iter[350/786] Loss: 2.556 CE: 1.242 Tri: 0.065 CE_rec: 1.183 AIRL_rec: 0.1304 Acc: 0.977 LR: 3.17e-05
logs/agreidv2_airl_4090.log:350:Epoch[21] Iter[400/786] Loss: 2.543 CE: 1.238 Tri: 0.063 CE_rec: 1.178 AIRL_rec: 0.1288 Acc: 0.978 LR: 3.17e-05
logs/agreidv2_airl_4090.log:351:Epoch[21] Iter[450/786] Loss: 2.538 CE: 1.235 Tri: 0.064 CE_rec: 1.175 AIRL_rec: 0.1278 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_airl_4090.log:352:Epoch[21] Iter[500/786] Loss: 2.534 CE: 1.232 Tri: 0.065 CE_rec: 1.173 AIRL_rec: 0.1273 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_airl_4090.log:353:Epoch[21] Iter[550/786] Loss: 2.529 CE: 1.230 Tri: 0.065 CE_rec: 1.171 AIRL_rec: 0.1270 Acc: 0.979 LR: 3.17e-05
logs/agreidv2_airl_4090.log:354:Epoch[21] Iter[600/786] Loss: 2.522 CE: 1.226 Tri: 0.065 CE_rec: 1.168 AIRL_rec: 0.1259 Acc: 0.980 LR: 3.17e-05
logs/agreidv2_airl_4090.log:355:Epoch[21] Iter[650/786] Loss: 2.514 CE: 1.222 Tri: 0.064 CE_rec: 1.165 AIRL_rec: 0.1247 Acc: 0.981 LR: 3.17e-05
logs/agreidv2_airl_4090.log:356:Epoch[21] Iter[700/786] Loss: 2.503 CE: 1.218 Tri: 0.063 CE_rec: 1.161 AIRL_rec: 0.1231 Acc: 0.981 LR: 3.17e-05
logs/agreidv2_airl_4090.log:357:Epoch[21] done in 113.0s  Loss=2.491 Acc=0.982 AIRL-ISO[lam_eff=0.500 ce_rec=1.156 consistency=0.1215 deg_scale_mean=0.625 n_ground=28669]
logs/agreidv2_airl_4090.log:358:Epoch[22] Iter[50/786] Loss: 2.624 CE: 1.277 Tri: 0.076 CE_rec: 1.209 AIRL_rec: 0.1219 Acc: 0.973 LR: 3.10e-05
logs/agreidv2_airl_4090.log:359:Epoch[22] Iter[100/786] Loss: 2.599 CE: 1.259 Tri: 0.082 CE_rec: 1.196 AIRL_rec: 0.1254 Acc: 0.975 LR: 3.10e-05
logs/agreidv2_airl_4090.log:360:Epoch[22] Iter[150/786] Loss: 2.581 CE: 1.251 Tri: 0.077 CE_rec: 1.189 AIRL_rec: 0.1273 Acc: 0.976 LR: 3.10e-05
logs/agreidv2_airl_4090.log:361:Epoch[22] Iter[200/786] Loss: 2.560 CE: 1.241 Tri: 0.074 CE_rec: 1.181 AIRL_rec: 0.1288 Acc: 0.978 LR: 3.10e-05
logs/agreidv2_airl_4090.log:362:Epoch[22] Iter[250/786] Loss: 2.543 CE: 1.234 Tri: 0.070 CE_rec: 1.175 AIRL_rec: 0.1276 Acc: 0.979 LR: 3.10e-05
logs/agreidv2_airl_4090.log:363:Epoch[22] Iter[300/786] Loss: 2.531 CE: 1.230 Tri: 0.067 CE_rec: 1.171 AIRL_rec: 0.1273 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_airl_4090.log:364:Epoch[22] Iter[350/786] Loss: 2.522 CE: 1.226 Tri: 0.065 CE_rec: 1.167 AIRL_rec: 0.1265 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_airl_4090.log:365:Epoch[22] Iter[400/786] Loss: 2.516 CE: 1.224 Tri: 0.063 CE_rec: 1.166 AIRL_rec: 0.1255 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_airl_4090.log:366:Epoch[22] Iter[450/786] Loss: 2.509 CE: 1.221 Tri: 0.062 CE_rec: 1.163 AIRL_rec: 0.1248 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_airl_4090.log:367:Epoch[22] Iter[500/786] Loss: 2.505 CE: 1.219 Tri: 0.062 CE_rec: 1.161 AIRL_rec: 0.1240 Acc: 0.980 LR: 3.10e-05
logs/agreidv2_airl_4090.log:368:Epoch[22] Iter[550/786] Loss: 2.499 CE: 1.217 Tri: 0.061 CE_rec: 1.160 AIRL_rec: 0.1228 Acc: 0.981 LR: 3.10e-05
logs/agreidv2_airl_4090.log:369:Epoch[22] Iter[600/786] Loss: 2.492 CE: 1.214 Tri: 0.061 CE_rec: 1.157 AIRL_rec: 0.1215 Acc: 0.981 LR: 3.10e-05
logs/agreidv2_airl_4090.log:370:Epoch[22] Iter[650/786] Loss: 2.486 CE: 1.211 Tri: 0.061 CE_rec: 1.155 AIRL_rec: 0.1201 Acc: 0.982 LR: 3.10e-05
logs/agreidv2_airl_4090.log:371:Epoch[22] Iter[700/786] Loss: 2.478 CE: 1.207 Tri: 0.060 CE_rec: 1.151 AIRL_rec: 0.1186 Acc: 0.982 LR: 3.10e-05
logs/agreidv2_airl_4090.log:372:Epoch[22] done in 112.7s  Loss=2.468 Acc=0.983 AIRL-ISO[lam_eff=0.500 ce_rec=1.148 consistency=0.1170 deg_scale_mean=0.625 n_ground=28572]
logs/agreidv2_airl_4090.log:373:Epoch[23] Iter[50/786] Loss: 2.654 CE: 1.297 Tri: 0.074 CE_rec: 1.223 AIRL_rec: 0.1215 Acc: 0.969 LR: 3.03e-05
logs/agreidv2_airl_4090.log:374:Epoch[23] Iter[100/786] Loss: 2.591 CE: 1.265 Tri: 0.066 CE_rec: 1.198 AIRL_rec: 0.1232 Acc: 0.973 LR: 3.03e-05
logs/agreidv2_airl_4090.log:375:Epoch[23] Iter[150/786] Loss: 2.550 CE: 1.245 Tri: 0.061 CE_rec: 1.181 AIRL_rec: 0.1234 Acc: 0.977 LR: 3.03e-05
logs/agreidv2_airl_4090.log:376:Epoch[23] Iter[200/786] Loss: 2.532 CE: 1.236 Tri: 0.059 CE_rec: 1.175 AIRL_rec: 0.1239 Acc: 0.978 LR: 3.03e-05
logs/agreidv2_airl_4090.log:377:Epoch[23] Iter[250/786] Loss: 2.527 CE: 1.234 Tri: 0.059 CE_rec: 1.172 AIRL_rec: 0.1231 Acc: 0.978 LR: 3.03e-05
logs/agreidv2_airl_4090.log:378:Epoch[23] Iter[300/786] Loss: 2.521 CE: 1.230 Tri: 0.058 CE_rec: 1.171 AIRL_rec: 0.1229 Acc: 0.978 LR: 3.03e-05
logs/agreidv2_airl_4090.log:379:Epoch[23] Iter[350/786] Loss: 2.512 CE: 1.227 Tri: 0.056 CE_rec: 1.168 AIRL_rec: 0.1217 Acc: 0.978 LR: 3.03e-05
logs/agreidv2_airl_4090.log:380:Epoch[23] Iter[400/786] Loss: 2.503 CE: 1.222 Tri: 0.056 CE_rec: 1.164 AIRL_rec: 0.1208 Acc: 0.979 LR: 3.03e-05
logs/agreidv2_airl_4090.log:381:Epoch[23] Iter[450/786] Loss: 2.496 CE: 1.219 Tri: 0.055 CE_rec: 1.161 AIRL_rec: 0.1198 Acc: 0.980 LR: 3.03e-05
logs/agreidv2_airl_4090.log:382:Epoch[23] Iter[500/786] Loss: 2.491 CE: 1.217 Tri: 0.055 CE_rec: 1.160 AIRL_rec: 0.1190 Acc: 0.980 LR: 3.03e-05
logs/agreidv2_airl_4090.log:383:Epoch[23] Iter[550/786] Loss: 2.483 CE: 1.213 Tri: 0.055 CE_rec: 1.157 AIRL_rec: 0.1178 Acc: 0.981 LR: 3.03e-05
logs/agreidv2_airl_4090.log:384:Epoch[23] Iter[600/786] Loss: 2.475 CE: 1.210 Tri: 0.053 CE_rec: 1.154 AIRL_rec: 0.1168 Acc: 0.981 LR: 3.03e-05
logs/agreidv2_airl_4090.log:385:Epoch[23] Iter[650/786] Loss: 2.467 CE: 1.206 Tri: 0.052 CE_rec: 1.150 AIRL_rec: 0.1156 Acc: 0.982 LR: 3.03e-05
logs/agreidv2_airl_4090.log:386:Epoch[23] Iter[700/786] Loss: 2.458 CE: 1.202 Tri: 0.052 CE_rec: 1.147 AIRL_rec: 0.1138 Acc: 0.982 LR: 3.03e-05
logs/agreidv2_airl_4090.log:387:Epoch[23] done in 113.7s  Loss=2.446 Acc=0.983 AIRL-ISO[lam_eff=0.500 ce_rec=1.143 consistency=0.1118 deg_scale_mean=0.625 n_ground=28797]
logs/agreidv2_airl_4090.log:388:Epoch[24] Iter[50/786] Loss: 2.589 CE: 1.262 Tri: 0.060 CE_rec: 1.209 AIRL_rec: 0.1169 Acc: 0.972 LR: 2.95e-05
logs/agreidv2_airl_4090.log:389:Epoch[24] Iter[100/786] Loss: 2.544 CE: 1.242 Tri: 0.057 CE_rec: 1.187 AIRL_rec: 0.1164 Acc: 0.976 LR: 2.95e-05
logs/agreidv2_airl_4090.log:390:Epoch[24] Iter[150/786] Loss: 2.517 CE: 1.229 Tri: 0.054 CE_rec: 1.175 AIRL_rec: 0.1161 Acc: 0.978 LR: 2.95e-05
logs/agreidv2_airl_4090.log:391:Epoch[24] Iter[200/786] Loss: 2.497 CE: 1.220 Tri: 0.052 CE_rec: 1.167 AIRL_rec: 0.1155 Acc: 0.979 LR: 2.95e-05
logs/agreidv2_airl_4090.log:392:Epoch[24] Iter[250/786] Loss: 2.491 CE: 1.216 Tri: 0.054 CE_rec: 1.164 AIRL_rec: 0.1150 Acc: 0.981 LR: 2.95e-05
logs/agreidv2_airl_4090.log:393:Epoch[24] Iter[300/786] Loss: 2.484 CE: 1.212 Tri: 0.054 CE_rec: 1.161 AIRL_rec: 0.1153 Acc: 0.981 LR: 2.95e-05
logs/agreidv2_airl_4090.log:394:Epoch[24] Iter[350/786] Loss: 2.479 CE: 1.210 Tri: 0.054 CE_rec: 1.158 AIRL_rec: 0.1149 Acc: 0.981 LR: 2.95e-05
logs/agreidv2_airl_4090.log:395:Epoch[24] Iter[400/786] Loss: 2.472 CE: 1.207 Tri: 0.052 CE_rec: 1.155 AIRL_rec: 0.1154 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_airl_4090.log:396:Epoch[24] Iter[450/786] Loss: 2.468 CE: 1.205 Tri: 0.051 CE_rec: 1.154 AIRL_rec: 0.1155 Acc: 0.981 LR: 2.95e-05
logs/agreidv2_airl_4090.log:397:Epoch[24] Iter[500/786] Loss: 2.462 CE: 1.203 Tri: 0.050 CE_rec: 1.152 AIRL_rec: 0.1151 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_airl_4090.log:398:Epoch[24] Iter[550/786] Loss: 2.456 CE: 1.200 Tri: 0.049 CE_rec: 1.150 AIRL_rec: 0.1144 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_airl_4090.log:399:Epoch[24] Iter[600/786] Loss: 2.450 CE: 1.197 Tri: 0.049 CE_rec: 1.147 AIRL_rec: 0.1139 Acc: 0.982 LR: 2.95e-05
logs/agreidv2_airl_4090.log:400:Epoch[24] Iter[650/786] Loss: 2.444 CE: 1.195 Tri: 0.048 CE_rec: 1.145 AIRL_rec: 0.1126 Acc: 0.983 LR: 2.95e-05
logs/agreidv2_airl_4090.log:401:Epoch[24] Iter[700/786] Loss: 2.434 CE: 1.190 Tri: 0.047 CE_rec: 1.141 AIRL_rec: 0.1108 Acc: 0.983 LR: 2.95e-05
logs/agreidv2_airl_4090.log:402:Epoch[24] done in 113.4s  Loss=2.423 Acc=0.984 AIRL-ISO[lam_eff=0.500 ce_rec=1.137 consistency=0.1091 deg_scale_mean=0.624 n_ground=28788]
logs/agreidv2_airl_4090.log:403:Epoch[25] Iter[50/786] Loss: 2.546 CE: 1.248 Tri: 0.055 CE_rec: 1.188 AIRL_rec: 0.1113 Acc: 0.976 LR: 2.87e-05
logs/agreidv2_airl_4090.log:404:Epoch[25] Iter[100/786] Loss: 2.506 CE: 1.227 Tri: 0.051 CE_rec: 1.170 AIRL_rec: 0.1143 Acc: 0.979 LR: 2.87e-05
logs/agreidv2_airl_4090.log:405:Epoch[25] Iter[150/786] Loss: 2.493 CE: 1.220 Tri: 0.051 CE_rec: 1.165 AIRL_rec: 0.1131 Acc: 0.980 LR: 2.87e-05
logs/agreidv2_airl_4090.log:406:Epoch[25] Iter[200/786] Loss: 2.485 CE: 1.216 Tri: 0.051 CE_rec: 1.162 AIRL_rec: 0.1146 Acc: 0.981 LR: 2.87e-05
logs/agreidv2_airl_4090.log:407:Epoch[25] Iter[250/786] Loss: 2.476 CE: 1.212 Tri: 0.050 CE_rec: 1.157 AIRL_rec: 0.1141 Acc: 0.981 LR: 2.87e-05
logs/agreidv2_airl_4090.log:408:Epoch[25] Iter[300/786] Loss: 2.469 CE: 1.208 Tri: 0.050 CE_rec: 1.154 AIRL_rec: 0.1142 Acc: 0.981 LR: 2.87e-05
logs/agreidv2_airl_4090.log:409:Epoch[25] Iter[350/786] Loss: 2.460 CE: 1.204 Tri: 0.049 CE_rec: 1.150 AIRL_rec: 0.1143 Acc: 0.982 LR: 2.87e-05
logs/agreidv2_airl_4090.log:410:Epoch[25] Iter[400/786] Loss: 2.455 CE: 1.200 Tri: 0.050 CE_rec: 1.148 AIRL_rec: 0.1137 Acc: 0.982 LR: 2.87e-05
logs/agreidv2_airl_4090.log:411:Epoch[25] Iter[450/786] Loss: 2.449 CE: 1.197 Tri: 0.050 CE_rec: 1.145 AIRL_rec: 0.1127 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_airl_4090.log:412:Epoch[25] Iter[500/786] Loss: 2.443 CE: 1.195 Tri: 0.050 CE_rec: 1.143 AIRL_rec: 0.1116 Acc: 0.983 LR: 2.87e-05
logs/agreidv2_airl_4090.log:413:Epoch[25] Iter[550/786] Loss: 2.435 CE: 1.191 Tri: 0.048 CE_rec: 1.140 AIRL_rec: 0.1105 Acc: 0.984 LR: 2.87e-05
logs/agreidv2_airl_4090.log:414:Epoch[25] Iter[600/786] Loss: 2.428 CE: 1.188 Tri: 0.047 CE_rec: 1.138 AIRL_rec: 0.1094 Acc: 0.984 LR: 2.87e-05
logs/agreidv2_airl_4090.log:415:Epoch[25] Iter[650/786] Loss: 2.419 CE: 1.185 Tri: 0.046 CE_rec: 1.135 AIRL_rec: 0.1079 Acc: 0.985 LR: 2.87e-05
logs/agreidv2_airl_4090.log:416:Epoch[25] Iter[700/786] Loss: 2.410 CE: 1.181 Tri: 0.045 CE_rec: 1.131 AIRL_rec: 0.1065 Acc: 0.985 LR: 2.87e-05
logs/agreidv2_airl_4090.log:417:Epoch[25] done in 112.8s  Loss=2.400 Acc=0.986 AIRL-ISO[lam_eff=0.500 ce_rec=1.128 consistency=0.1050 deg_scale_mean=0.628 n_ground=28670]
logs/agreidv2_airl_4090.log:418:Epoch[26] Iter[50/786] Loss: 2.477 CE: 1.227 Tri: 0.034 CE_rec: 1.164 AIRL_rec: 0.1058 Acc: 0.977 LR: 2.78e-05
logs/agreidv2_airl_4090.log:419:Epoch[26] Iter[100/786] Loss: 2.463 CE: 1.217 Tri: 0.034 CE_rec: 1.159 AIRL_rec: 0.1086 Acc: 0.978 LR: 2.78e-05
logs/agreidv2_airl_4090.log:420:Epoch[26] Iter[150/786] Loss: 2.441 CE: 1.203 Tri: 0.035 CE_rec: 1.149 AIRL_rec: 0.1086 Acc: 0.981 LR: 2.78e-05
logs/agreidv2_airl_4090.log:421:Epoch[26] Iter[200/786] Loss: 2.430 CE: 1.195 Tri: 0.037 CE_rec: 1.143 AIRL_rec: 0.1081 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_airl_4090.log:422:Epoch[26] Iter[250/786] Loss: 2.427 CE: 1.192 Tri: 0.039 CE_rec: 1.142 AIRL_rec: 0.1074 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_airl_4090.log:423:Epoch[26] Iter[300/786] Loss: 2.424 CE: 1.191 Tri: 0.038 CE_rec: 1.141 AIRL_rec: 0.1072 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_airl_4090.log:424:Epoch[26] Iter[350/786] Loss: 2.422 CE: 1.189 Tri: 0.041 CE_rec: 1.139 AIRL_rec: 0.1074 Acc: 0.983 LR: 2.78e-05
logs/agreidv2_airl_4090.log:425:Epoch[26] Iter[400/786] Loss: 2.412 CE: 1.185 Tri: 0.039 CE_rec: 1.135 AIRL_rec: 0.1065 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_airl_4090.log:426:Epoch[26] Iter[450/786] Loss: 2.409 CE: 1.183 Tri: 0.040 CE_rec: 1.133 AIRL_rec: 0.1059 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_airl_4090.log:427:Epoch[26] Iter[500/786] Loss: 2.409 CE: 1.182 Tri: 0.041 CE_rec: 1.133 AIRL_rec: 0.1059 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_airl_4090.log:428:Epoch[26] Iter[550/786] Loss: 2.405 CE: 1.180 Tri: 0.041 CE_rec: 1.131 AIRL_rec: 0.1049 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_airl_4090.log:429:Epoch[26] Iter[600/786] Loss: 2.400 CE: 1.178 Tri: 0.040 CE_rec: 1.129 AIRL_rec: 0.1035 Acc: 0.984 LR: 2.78e-05
logs/agreidv2_airl_4090.log:430:Epoch[26] Iter[650/786] Loss: 2.395 CE: 1.176 Tri: 0.041 CE_rec: 1.127 AIRL_rec: 0.1024 Acc: 0.985 LR: 2.78e-05
logs/agreidv2_airl_4090.log:431:Epoch[26] Iter[700/786] Loss: 2.389 CE: 1.172 Tri: 0.041 CE_rec: 1.125 AIRL_rec: 0.1010 Acc: 0.985 LR: 2.78e-05
logs/agreidv2_airl_4090.log:432:Epoch[26] done in 113.4s  Loss=2.379 Acc=0.986 AIRL-ISO[lam_eff=0.500 ce_rec=1.121 consistency=0.0995 deg_scale_mean=0.624 n_ground=28726]
logs/agreidv2_airl_4090.log:433:Epoch[27] Iter[50/786] Loss: 2.505 CE: 1.230 Tri: 0.051 CE_rec: 1.171 AIRL_rec: 0.1041 Acc: 0.975 LR: 2.69e-05
logs/agreidv2_airl_4090.log:434:Epoch[27] Iter[100/786] Loss: 2.476 CE: 1.213 Tri: 0.049 CE_rec: 1.160 AIRL_rec: 0.1065 Acc: 0.979 LR: 2.69e-05
logs/agreidv2_airl_4090.log:435:Epoch[27] Iter[150/786] Loss: 2.451 CE: 1.202 Tri: 0.044 CE_rec: 1.152 AIRL_rec: 0.1063 Acc: 0.981 LR: 2.69e-05
logs/agreidv2_airl_4090.log:436:Epoch[27] Iter[200/786] Loss: 2.436 CE: 1.194 Tri: 0.043 CE_rec: 1.145 AIRL_rec: 0.1064 Acc: 0.982 LR: 2.69e-05
logs/agreidv2_airl_4090.log:437:Epoch[27] Iter[250/786] Loss: 2.420 CE: 1.187 Tri: 0.041 CE_rec: 1.140 AIRL_rec: 0.1064 Acc: 0.984 LR: 2.69e-05
logs/agreidv2_airl_4090.log:438:Epoch[27] Iter[300/786] Loss: 2.412 CE: 1.183 Tri: 0.041 CE_rec: 1.135 AIRL_rec: 0.1054 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_airl_4090.log:439:Epoch[27] Iter[350/786] Loss: 2.404 CE: 1.180 Tri: 0.041 CE_rec: 1.131 AIRL_rec: 0.1047 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_airl_4090.log:440:Epoch[27] Iter[400/786] Loss: 2.399 CE: 1.177 Tri: 0.041 CE_rec: 1.129 AIRL_rec: 0.1035 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_airl_4090.log:441:Epoch[27] Iter[450/786] Loss: 2.396 CE: 1.176 Tri: 0.040 CE_rec: 1.128 AIRL_rec: 0.1029 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_airl_4090.log:442:Epoch[27] Iter[500/786] Loss: 2.393 CE: 1.175 Tri: 0.040 CE_rec: 1.127 AIRL_rec: 0.1017 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_airl_4090.log:443:Epoch[27] Iter[550/786] Loss: 2.389 CE: 1.173 Tri: 0.040 CE_rec: 1.126 AIRL_rec: 0.1009 Acc: 0.985 LR: 2.69e-05
logs/agreidv2_airl_4090.log:444:Epoch[27] Iter[600/786] Loss: 2.384 CE: 1.171 Tri: 0.039 CE_rec: 1.124 AIRL_rec: 0.1002 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_airl_4090.log:445:Epoch[27] Iter[650/786] Loss: 2.376 CE: 1.168 Tri: 0.038 CE_rec: 1.121 AIRL_rec: 0.0989 Acc: 0.986 LR: 2.69e-05
logs/agreidv2_airl_4090.log:446:Epoch[27] Iter[700/786] Loss: 2.368 CE: 1.164 Tri: 0.037 CE_rec: 1.118 AIRL_rec: 0.0975 Acc: 0.987 LR: 2.69e-05
logs/agreidv2_airl_4090.log:447:Epoch[27] done in 113.1s  Loss=2.359 Acc=0.987 AIRL-ISO[lam_eff=0.500 ce_rec=1.115 consistency=0.0961 deg_scale_mean=0.625 n_ground=28699]
logs/agreidv2_airl_4090.log:448:Epoch[28] Iter[50/786] Loss: 2.480 CE: 1.218 Tri: 0.052 CE_rec: 1.160 AIRL_rec: 0.0999 Acc: 0.980 LR: 2.59e-05
logs/agreidv2_airl_4090.log:449:Epoch[28] Iter[100/786] Loss: 2.445 CE: 1.198 Tri: 0.049 CE_rec: 1.147 AIRL_rec: 0.1025 Acc: 0.982 LR: 2.59e-05
logs/agreidv2_airl_4090.log:450:Epoch[28] Iter[150/786] Loss: 2.433 CE: 1.194 Tri: 0.045 CE_rec: 1.142 AIRL_rec: 0.1022 Acc: 0.982 LR: 2.59e-05
logs/agreidv2_airl_4090.log:451:Epoch[28] Iter[200/786] Loss: 2.422 CE: 1.188 Tri: 0.043 CE_rec: 1.138 AIRL_rec: 0.1030 Acc: 0.983 LR: 2.59e-05
logs/agreidv2_airl_4090.log:452:Epoch[28] Iter[250/786] Loss: 2.416 CE: 1.186 Tri: 0.044 CE_rec: 1.135 AIRL_rec: 0.1026 Acc: 0.983 LR: 2.59e-05
logs/agreidv2_airl_4090.log:453:Epoch[28] Iter[300/786] Loss: 2.404 CE: 1.180 Tri: 0.042 CE_rec: 1.131 AIRL_rec: 0.1019 Acc: 0.984 LR: 2.59e-05
logs/agreidv2_airl_4090.log:454:Epoch[28] Iter[350/786] Loss: 2.394 CE: 1.175 Tri: 0.041 CE_rec: 1.128 AIRL_rec: 0.1014 Acc: 0.985 LR: 2.59e-05
logs/agreidv2_airl_4090.log:455:Epoch[28] Iter[400/786] Loss: 2.385 CE: 1.170 Tri: 0.039 CE_rec: 1.125 AIRL_rec: 0.1007 Acc: 0.986 LR: 2.59e-05
logs/agreidv2_airl_4090.log:456:Epoch[28] Iter[450/786] Loss: 2.375 CE: 1.166 Tri: 0.038 CE_rec: 1.121 AIRL_rec: 0.1000 Acc: 0.987 LR: 2.59e-05
logs/agreidv2_airl_4090.log:457:Epoch[28] Iter[500/786] Loss: 2.370 CE: 1.164 Tri: 0.037 CE_rec: 1.119 AIRL_rec: 0.0993 Acc: 0.987 LR: 2.59e-05
logs/agreidv2_airl_4090.log:458:Epoch[28] Iter[550/786] Loss: 2.367 CE: 1.163 Tri: 0.037 CE_rec: 1.118 AIRL_rec: 0.0986 Acc: 0.987 LR: 2.59e-05
logs/agreidv2_airl_4090.log:459:Epoch[28] Iter[600/786] Loss: 2.360 CE: 1.160 Tri: 0.036 CE_rec: 1.115 AIRL_rec: 0.0975 Acc: 0.988 LR: 2.59e-05
logs/agreidv2_airl_4090.log:460:Epoch[28] Iter[650/786] Loss: 2.353 CE: 1.157 Tri: 0.035 CE_rec: 1.113 AIRL_rec: 0.0963 Acc: 0.988 LR: 2.59e-05
logs/agreidv2_airl_4090.log:461:Epoch[28] Iter[700/786] Loss: 2.347 CE: 1.154 Tri: 0.035 CE_rec: 1.110 AIRL_rec: 0.0949 Acc: 0.988 LR: 2.59e-05
logs/agreidv2_airl_4090.log:462:Epoch[28] done in 113.7s  Loss=2.337 Acc=0.989 AIRL-ISO[lam_eff=0.500 ce_rec=1.107 consistency=0.0934 deg_scale_mean=0.626 n_ground=28769]
logs/agreidv2_airl_4090.log:463:Epoch[29] Iter[50/786] Loss: 2.430 CE: 1.198 Tri: 0.035 CE_rec: 1.150 AIRL_rec: 0.0941 Acc: 0.982 LR: 2.50e-05
logs/agreidv2_airl_4090.log:464:Epoch[29] Iter[100/786] Loss: 2.410 CE: 1.187 Tri: 0.035 CE_rec: 1.140 AIRL_rec: 0.0972 Acc: 0.984 LR: 2.50e-05
logs/agreidv2_airl_4090.log:465:Epoch[29] Iter[150/786] Loss: 2.406 CE: 1.183 Tri: 0.038 CE_rec: 1.136 AIRL_rec: 0.0978 Acc: 0.984 LR: 2.50e-05
logs/agreidv2_airl_4090.log:466:Epoch[29] Iter[200/786] Loss: 2.398 CE: 1.177 Tri: 0.040 CE_rec: 1.131 AIRL_rec: 0.0984 Acc: 0.985 LR: 2.50e-05
logs/agreidv2_airl_4090.log:467:Epoch[29] Iter[250/786] Loss: 2.388 CE: 1.173 Tri: 0.038 CE_rec: 1.127 AIRL_rec: 0.0981 Acc: 0.985 LR: 2.50e-05
logs/agreidv2_airl_4090.log:468:Epoch[29] Iter[300/786] Loss: 2.378 CE: 1.169 Tri: 0.037 CE_rec: 1.124 AIRL_rec: 0.0973 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_airl_4090.log:469:Epoch[29] Iter[350/786] Loss: 2.375 CE: 1.167 Tri: 0.037 CE_rec: 1.123 AIRL_rec: 0.0967 Acc: 0.985 LR: 2.50e-05
logs/agreidv2_airl_4090.log:470:Epoch[29] Iter[400/786] Loss: 2.372 CE: 1.165 Tri: 0.036 CE_rec: 1.122 AIRL_rec: 0.0966 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_airl_4090.log:471:Epoch[29] Iter[450/786] Loss: 2.368 CE: 1.163 Tri: 0.037 CE_rec: 1.120 AIRL_rec: 0.0962 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_airl_4090.log:472:Epoch[29] Iter[500/786] Loss: 2.362 CE: 1.161 Tri: 0.036 CE_rec: 1.118 AIRL_rec: 0.0953 Acc: 0.986 LR: 2.50e-05
logs/agreidv2_airl_4090.log:473:Epoch[29] Iter[550/786] Loss: 2.357 CE: 1.158 Tri: 0.036 CE_rec: 1.115 AIRL_rec: 0.0945 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_airl_4090.log:474:Epoch[29] Iter[600/786] Loss: 2.353 CE: 1.156 Tri: 0.035 CE_rec: 1.114 AIRL_rec: 0.0938 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_airl_4090.log:475:Epoch[29] Iter[650/786] Loss: 2.347 CE: 1.154 Tri: 0.035 CE_rec: 1.112 AIRL_rec: 0.0927 Acc: 0.987 LR: 2.50e-05
logs/agreidv2_airl_4090.log:476:Epoch[29] Iter[700/786] Loss: 2.339 CE: 1.150 Tri: 0.033 CE_rec: 1.109 AIRL_rec: 0.0916 Acc: 0.988 LR: 2.50e-05
logs/agreidv2_airl_4090.log:477:Epoch[29] done in 113.5s  Loss=2.329 Acc=0.988 AIRL-ISO[lam_eff=0.500 ce_rec=1.106 consistency=0.0901 deg_scale_mean=0.624 n_ground=28805]
logs/agreidv2_airl_4090.log:478:Epoch[30] Iter[50/786] Loss: 2.405 CE: 1.185 Tri: 0.035 CE_rec: 1.138 AIRL_rec: 0.0947 Acc: 0.983 LR: 2.39e-05
logs/agreidv2_airl_4090.log:479:Epoch[30] Iter[100/786] Loss: 2.372 CE: 1.169 Tri: 0.032 CE_rec: 1.124 AIRL_rec: 0.0950 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_airl_4090.log:480:Epoch[30] Iter[150/786] Loss: 2.362 CE: 1.162 Tri: 0.033 CE_rec: 1.118 AIRL_rec: 0.0956 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_airl_4090.log:481:Epoch[30] Iter[200/786] Loss: 2.368 CE: 1.165 Tri: 0.035 CE_rec: 1.120 AIRL_rec: 0.0964 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_airl_4090.log:482:Epoch[30] Iter[250/786] Loss: 2.359 CE: 1.161 Tri: 0.034 CE_rec: 1.116 AIRL_rec: 0.0963 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_airl_4090.log:483:Epoch[30] Iter[300/786] Loss: 2.353 CE: 1.158 Tri: 0.033 CE_rec: 1.114 AIRL_rec: 0.0958 Acc: 0.986 LR: 2.39e-05
logs/agreidv2_airl_4090.log:484:Epoch[30] Iter[350/786] Loss: 2.349 CE: 1.157 Tri: 0.032 CE_rec: 1.113 AIRL_rec: 0.0951 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_airl_4090.log:485:Epoch[30] Iter[400/786] Loss: 2.342 CE: 1.154 Tri: 0.031 CE_rec: 1.110 AIRL_rec: 0.0946 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_airl_4090.log:486:Epoch[30] Iter[450/786] Loss: 2.339 CE: 1.152 Tri: 0.031 CE_rec: 1.109 AIRL_rec: 0.0937 Acc: 0.987 LR: 2.39e-05
logs/agreidv2_airl_4090.log:487:Epoch[30] Iter[500/786] Loss: 2.334 CE: 1.150 Tri: 0.030 CE_rec: 1.107 AIRL_rec: 0.0927 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_airl_4090.log:488:Epoch[30] Iter[550/786] Loss: 2.331 CE: 1.149 Tri: 0.030 CE_rec: 1.106 AIRL_rec: 0.0922 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_airl_4090.log:489:Epoch[30] Iter[600/786] Loss: 2.329 CE: 1.148 Tri: 0.030 CE_rec: 1.105 AIRL_rec: 0.0917 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_airl_4090.log:490:Epoch[30] Iter[650/786] Loss: 2.324 CE: 1.146 Tri: 0.029 CE_rec: 1.103 AIRL_rec: 0.0908 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_airl_4090.log:491:Epoch[30] Iter[700/786] Loss: 2.316 CE: 1.142 Tri: 0.029 CE_rec: 1.100 AIRL_rec: 0.0896 Acc: 0.988 LR: 2.39e-05
logs/agreidv2_airl_4090.log:492:Epoch[30] done in 113.4s  Loss=2.307 Acc=0.989 AIRL-ISO[lam_eff=0.500 ce_rec=1.097 consistency=0.0878 deg_scale_mean=0.625 n_ground=28734]
logs/agreidv2_airl_4090.log:494:    [A->G] mAP=72.50  R1=81.07  R5=88.41  mINP=44.80
logs/agreidv2_airl_4090.log:495:    [G->A] mAP=73.73  R1=83.27  R5=88.40  mINP=38.48
logs/agreidv2_airl_4090.log:496:    [mean] mAP=73.11  R1=82.17
logs/agreidv2_airl_4090.log:497:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 30 ----
logs/agreidv2_airl_4090.log:498:    [A->G] full mAP=72.50 R1=81.07 | rec mAP=73.20 R1=81.83 | FUSE mAP=74.31 R1=83.15
logs/agreidv2_airl_4090.log:499:    [G->A] full mAP=73.73 R1=83.27 | rec mAP=73.70 R1=82.27 | FUSE mAP=75.06 R1=83.77
logs/agreidv2_airl_4090.log:501:Epoch[31] Iter[50/786] Loss: 2.375 CE: 1.177 Tri: 0.027 CE_rec: 1.127 AIRL_rec: 0.0899 Acc: 0.984 LR: 2.29e-05
logs/agreidv2_airl_4090.log:502:Epoch[31] Iter[100/786] Loss: 2.374 CE: 1.170 Tri: 0.036 CE_rec: 1.123 AIRL_rec: 0.0924 Acc: 0.984 LR: 2.29e-05
logs/agreidv2_airl_4090.log:503:Epoch[31] Iter[150/786] Loss: 2.360 CE: 1.163 Tri: 0.031 CE_rec: 1.119 AIRL_rec: 0.0935 Acc: 0.985 LR: 2.29e-05
logs/agreidv2_airl_4090.log:504:Epoch[31] Iter[200/786] Loss: 2.351 CE: 1.159 Tri: 0.029 CE_rec: 1.117 AIRL_rec: 0.0929 Acc: 0.986 LR: 2.29e-05
logs/agreidv2_airl_4090.log:505:Epoch[31] Iter[250/786] Loss: 2.346 CE: 1.155 Tri: 0.031 CE_rec: 1.113 AIRL_rec: 0.0932 Acc: 0.986 LR: 2.29e-05
logs/agreidv2_airl_4090.log:506:Epoch[31] Iter[300/786] Loss: 2.343 CE: 1.153 Tri: 0.032 CE_rec: 1.112 AIRL_rec: 0.0922 Acc: 0.987 LR: 2.29e-05
logs/agreidv2_airl_4090.log:507:Epoch[31] Iter[350/786] Loss: 2.334 CE: 1.149 Tri: 0.031 CE_rec: 1.108 AIRL_rec: 0.0913 Acc: 0.987 LR: 2.29e-05
logs/agreidv2_airl_4090.log:508:Epoch[31] Iter[400/786] Loss: 2.327 CE: 1.146 Tri: 0.030 CE_rec: 1.106 AIRL_rec: 0.0909 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_airl_4090.log:509:Epoch[31] Iter[450/786] Loss: 2.324 CE: 1.145 Tri: 0.029 CE_rec: 1.105 AIRL_rec: 0.0904 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_airl_4090.log:510:Epoch[31] Iter[500/786] Loss: 2.323 CE: 1.144 Tri: 0.030 CE_rec: 1.104 AIRL_rec: 0.0897 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_airl_4090.log:511:Epoch[31] Iter[550/786] Loss: 2.321 CE: 1.143 Tri: 0.031 CE_rec: 1.103 AIRL_rec: 0.0893 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_airl_4090.log:512:Epoch[31] Iter[600/786] Loss: 2.317 CE: 1.141 Tri: 0.030 CE_rec: 1.101 AIRL_rec: 0.0883 Acc: 0.988 LR: 2.29e-05
logs/agreidv2_airl_4090.log:513:Epoch[31] Iter[650/786] Loss: 2.312 CE: 1.139 Tri: 0.030 CE_rec: 1.099 AIRL_rec: 0.0873 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_airl_4090.log:514:Epoch[31] Iter[700/786] Loss: 2.304 CE: 1.136 Tri: 0.029 CE_rec: 1.097 AIRL_rec: 0.0859 Acc: 0.989 LR: 2.29e-05
logs/agreidv2_airl_4090.log:515:Epoch[31] done in 112.3s  Loss=2.297 Acc=0.989 AIRL-ISO[lam_eff=0.500 ce_rec=1.094 consistency=0.0845 deg_scale_mean=0.624 n_ground=28589]
logs/agreidv2_airl_4090.log:516:Epoch[32] Iter[50/786] Loss: 2.368 CE: 1.169 Tri: 0.032 CE_rec: 1.124 AIRL_rec: 0.0876 Acc: 0.985 LR: 2.19e-05
logs/agreidv2_airl_4090.log:517:Epoch[32] Iter[100/786] Loss: 2.353 CE: 1.158 Tri: 0.035 CE_rec: 1.116 AIRL_rec: 0.0881 Acc: 0.988 LR: 2.19e-05
logs/agreidv2_airl_4090.log:518:Epoch[32] Iter[150/786] Loss: 2.347 CE: 1.154 Tri: 0.037 CE_rec: 1.111 AIRL_rec: 0.0889 Acc: 0.987 LR: 2.19e-05
logs/agreidv2_airl_4090.log:519:Epoch[32] Iter[200/786] Loss: 2.334 CE: 1.149 Tri: 0.034 CE_rec: 1.107 AIRL_rec: 0.0877 Acc: 0.988 LR: 2.19e-05
logs/agreidv2_airl_4090.log:520:Epoch[32] Iter[250/786] Loss: 2.324 CE: 1.145 Tri: 0.031 CE_rec: 1.104 AIRL_rec: 0.0869 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_airl_4090.log:521:Epoch[32] Iter[300/786] Loss: 2.321 CE: 1.143 Tri: 0.032 CE_rec: 1.103 AIRL_rec: 0.0867 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_airl_4090.log:522:Epoch[32] Iter[350/786] Loss: 2.314 CE: 1.140 Tri: 0.030 CE_rec: 1.100 AIRL_rec: 0.0864 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_airl_4090.log:523:Epoch[32] Iter[400/786] Loss: 2.310 CE: 1.139 Tri: 0.029 CE_rec: 1.099 AIRL_rec: 0.0861 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_airl_4090.log:524:Epoch[32] Iter[450/786] Loss: 2.308 CE: 1.138 Tri: 0.029 CE_rec: 1.098 AIRL_rec: 0.0856 Acc: 0.989 LR: 2.19e-05
logs/agreidv2_airl_4090.log:525:Epoch[32] Iter[500/786] Loss: 2.302 CE: 1.135 Tri: 0.028 CE_rec: 1.097 AIRL_rec: 0.0850 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_airl_4090.log:526:Epoch[32] Iter[550/786] Loss: 2.295 CE: 1.133 Tri: 0.026 CE_rec: 1.094 AIRL_rec: 0.0842 Acc: 0.990 LR: 2.19e-05
logs/agreidv2_airl_4090.log:527:Epoch[32] Iter[600/786] Loss: 2.290 CE: 1.131 Tri: 0.026 CE_rec: 1.092 AIRL_rec: 0.0836 Acc: 0.991 LR: 2.19e-05
logs/agreidv2_airl_4090.log:528:Epoch[32] Iter[650/786] Loss: 2.285 CE: 1.128 Tri: 0.025 CE_rec: 1.090 AIRL_rec: 0.0828 Acc: 0.991 LR: 2.19e-05
logs/agreidv2_airl_4090.log:529:Epoch[32] Iter[700/786] Loss: 2.280 CE: 1.126 Tri: 0.025 CE_rec: 1.088 AIRL_rec: 0.0815 Acc: 0.991 LR: 2.19e-05
logs/agreidv2_airl_4090.log:530:Epoch[32] done in 113.6s  Loss=2.272 Acc=0.991 AIRL-ISO[lam_eff=0.500 ce_rec=1.085 consistency=0.0800 deg_scale_mean=0.625 n_ground=28842]
logs/agreidv2_airl_4090.log:531:Epoch[33] Iter[50/786] Loss: 2.354 CE: 1.164 Tri: 0.031 CE_rec: 1.115 AIRL_rec: 0.0863 Acc: 0.984 LR: 2.08e-05
logs/agreidv2_airl_4090.log:532:Epoch[33] Iter[100/786] Loss: 2.344 CE: 1.157 Tri: 0.029 CE_rec: 1.114 AIRL_rec: 0.0852 Acc: 0.986 LR: 2.08e-05
logs/agreidv2_airl_4090.log:533:Epoch[33] Iter[150/786] Loss: 2.331 CE: 1.151 Tri: 0.028 CE_rec: 1.110 AIRL_rec: 0.0852 Acc: 0.987 LR: 2.08e-05
logs/agreidv2_airl_4090.log:534:Epoch[33] Iter[200/786] Loss: 2.315 CE: 1.144 Tri: 0.025 CE_rec: 1.104 AIRL_rec: 0.0851 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_airl_4090.log:535:Epoch[33] Iter[250/786] Loss: 2.313 CE: 1.142 Tri: 0.026 CE_rec: 1.103 AIRL_rec: 0.0842 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_airl_4090.log:536:Epoch[33] Iter[300/786] Loss: 2.304 CE: 1.138 Tri: 0.024 CE_rec: 1.100 AIRL_rec: 0.0839 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_airl_4090.log:537:Epoch[33] Iter[350/786] Loss: 2.302 CE: 1.137 Tri: 0.025 CE_rec: 1.099 AIRL_rec: 0.0837 Acc: 0.988 LR: 2.08e-05
logs/agreidv2_airl_4090.log:538:Epoch[33] Iter[400/786] Loss: 2.299 CE: 1.135 Tri: 0.025 CE_rec: 1.097 AIRL_rec: 0.0836 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_airl_4090.log:539:Epoch[33] Iter[450/786] Loss: 2.294 CE: 1.133 Tri: 0.024 CE_rec: 1.095 AIRL_rec: 0.0829 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_airl_4090.log:540:Epoch[33] Iter[500/786] Loss: 2.289 CE: 1.131 Tri: 0.024 CE_rec: 1.093 AIRL_rec: 0.0821 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_airl_4090.log:541:Epoch[33] Iter[550/786] Loss: 2.285 CE: 1.129 Tri: 0.023 CE_rec: 1.092 AIRL_rec: 0.0814 Acc: 0.989 LR: 2.08e-05
logs/agreidv2_airl_4090.log:542:Epoch[33] Iter[600/786] Loss: 2.282 CE: 1.128 Tri: 0.023 CE_rec: 1.091 AIRL_rec: 0.0806 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_airl_4090.log:543:Epoch[33] Iter[650/786] Loss: 2.278 CE: 1.126 Tri: 0.023 CE_rec: 1.089 AIRL_rec: 0.0799 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_airl_4090.log:544:Epoch[33] Iter[700/786] Loss: 2.271 CE: 1.123 Tri: 0.023 CE_rec: 1.087 AIRL_rec: 0.0787 Acc: 0.990 LR: 2.08e-05
logs/agreidv2_airl_4090.log:545:Epoch[33] done in 113.7s  Loss=2.264 Acc=0.991 AIRL-ISO[lam_eff=0.500 ce_rec=1.084 consistency=0.0773 deg_scale_mean=0.626 n_ground=28773]
logs/agreidv2_airl_4090.log:546:Epoch[34] Iter[50/786] Loss: 2.355 CE: 1.158 Tri: 0.030 CE_rec: 1.125 AIRL_rec: 0.0828 Acc: 0.986 LR: 1.97e-05
logs/agreidv2_airl_4090.log:547:Epoch[34] Iter[100/786] Loss: 2.328 CE: 1.145 Tri: 0.033 CE_rec: 1.108 AIRL_rec: 0.0848 Acc: 0.988 LR: 1.97e-05
logs/agreidv2_airl_4090.log:548:Epoch[34] Iter[150/786] Loss: 2.317 CE: 1.141 Tri: 0.030 CE_rec: 1.104 AIRL_rec: 0.0848 Acc: 0.988 LR: 1.97e-05
logs/agreidv2_airl_4090.log:549:Epoch[34] Iter[200/786] Loss: 2.309 CE: 1.138 Tri: 0.028 CE_rec: 1.101 AIRL_rec: 0.0850 Acc: 0.988 LR: 1.97e-05
logs/agreidv2_airl_4090.log:550:Epoch[34] Iter[250/786] Loss: 2.300 CE: 1.134 Tri: 0.026 CE_rec: 1.097 AIRL_rec: 0.0851 Acc: 0.989 LR: 1.97e-05
logs/agreidv2_airl_4090.log:551:Epoch[34] Iter[300/786] Loss: 2.292 CE: 1.131 Tri: 0.024 CE_rec: 1.095 AIRL_rec: 0.0842 Acc: 0.989 LR: 1.97e-05
logs/agreidv2_airl_4090.log:552:Epoch[34] Iter[350/786] Loss: 2.289 CE: 1.129 Tri: 0.025 CE_rec: 1.093 AIRL_rec: 0.0839 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_airl_4090.log:553:Epoch[34] Iter[400/786] Loss: 2.283 CE: 1.127 Tri: 0.024 CE_rec: 1.090 AIRL_rec: 0.0830 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_airl_4090.log:554:Epoch[34] Iter[450/786] Loss: 2.279 CE: 1.125 Tri: 0.024 CE_rec: 1.089 AIRL_rec: 0.0822 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_airl_4090.log:555:Epoch[34] Iter[500/786] Loss: 2.275 CE: 1.124 Tri: 0.024 CE_rec: 1.088 AIRL_rec: 0.0812 Acc: 0.990 LR: 1.97e-05
logs/agreidv2_airl_4090.log:556:Epoch[34] Iter[550/786] Loss: 2.273 CE: 1.122 Tri: 0.024 CE_rec: 1.086 AIRL_rec: 0.0806 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_airl_4090.log:557:Epoch[34] Iter[600/786] Loss: 2.269 CE: 1.121 Tri: 0.024 CE_rec: 1.085 AIRL_rec: 0.0800 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_airl_4090.log:558:Epoch[34] Iter[650/786] Loss: 2.264 CE: 1.119 Tri: 0.023 CE_rec: 1.083 AIRL_rec: 0.0789 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_airl_4090.log:559:Epoch[34] Iter[700/786] Loss: 2.258 CE: 1.116 Tri: 0.022 CE_rec: 1.081 AIRL_rec: 0.0776 Acc: 0.991 LR: 1.97e-05
logs/agreidv2_airl_4090.log:560:Epoch[34] done in 113.7s  Loss=2.251 Acc=0.992 AIRL-ISO[lam_eff=0.500 ce_rec=1.078 consistency=0.0763 deg_scale_mean=0.625 n_ground=28755]
logs/agreidv2_airl_4090.log:561:Epoch[35] Iter[50/786] Loss: 2.298 CE: 1.137 Tri: 0.021 CE_rec: 1.099 AIRL_rec: 0.0804 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_4090.log:562:Epoch[35] Iter[100/786] Loss: 2.284 CE: 1.129 Tri: 0.024 CE_rec: 1.091 AIRL_rec: 0.0805 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_4090.log:563:Epoch[35] Iter[150/786] Loss: 2.282 CE: 1.128 Tri: 0.023 CE_rec: 1.090 AIRL_rec: 0.0809 Acc: 0.990 LR: 1.86e-05
logs/agreidv2_airl_4090.log:564:Epoch[35] Iter[200/786] Loss: 2.274 CE: 1.124 Tri: 0.022 CE_rec: 1.088 AIRL_rec: 0.0811 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_4090.log:565:Epoch[35] Iter[250/786] Loss: 2.271 CE: 1.122 Tri: 0.022 CE_rec: 1.087 AIRL_rec: 0.0810 Acc: 0.990 LR: 1.86e-05
logs/agreidv2_airl_4090.log:566:Epoch[35] Iter[300/786] Loss: 2.268 CE: 1.121 Tri: 0.021 CE_rec: 1.086 AIRL_rec: 0.0808 Acc: 0.990 LR: 1.86e-05
logs/agreidv2_airl_4090.log:567:Epoch[35] Iter[350/786] Loss: 2.267 CE: 1.120 Tri: 0.021 CE_rec: 1.085 AIRL_rec: 0.0801 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_4090.log:568:Epoch[35] Iter[400/786] Loss: 2.263 CE: 1.118 Tri: 0.021 CE_rec: 1.084 AIRL_rec: 0.0797 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_4090.log:569:Epoch[35] Iter[450/786] Loss: 2.259 CE: 1.117 Tri: 0.020 CE_rec: 1.083 AIRL_rec: 0.0791 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_4090.log:570:Epoch[35] Iter[500/786] Loss: 2.255 CE: 1.115 Tri: 0.019 CE_rec: 1.081 AIRL_rec: 0.0782 Acc: 0.991 LR: 1.86e-05
logs/agreidv2_airl_4090.log:571:Epoch[35] Iter[550/786] Loss: 2.251 CE: 1.114 Tri: 0.019 CE_rec: 1.080 AIRL_rec: 0.0778 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_4090.log:572:Epoch[35] Iter[600/786] Loss: 2.246 CE: 1.111 Tri: 0.018 CE_rec: 1.078 AIRL_rec: 0.0766 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_4090.log:573:Epoch[35] Iter[650/786] Loss: 2.242 CE: 1.109 Tri: 0.018 CE_rec: 1.076 AIRL_rec: 0.0756 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_4090.log:574:Epoch[35] Iter[700/786] Loss: 2.237 CE: 1.107 Tri: 0.018 CE_rec: 1.075 AIRL_rec: 0.0745 Acc: 0.992 LR: 1.86e-05
logs/agreidv2_airl_4090.log:575:Epoch[35] done in 113.7s  Loss=2.229 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.072 consistency=0.0733 deg_scale_mean=0.627 n_ground=28727]
logs/agreidv2_airl_4090.log:576:Epoch[36] Iter[50/786] Loss: 2.293 CE: 1.136 Tri: 0.015 CE_rec: 1.103 AIRL_rec: 0.0794 Acc: 0.990 LR: 1.75e-05
logs/agreidv2_airl_4090.log:577:Epoch[36] Iter[100/786] Loss: 2.280 CE: 1.128 Tri: 0.018 CE_rec: 1.093 AIRL_rec: 0.0798 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_airl_4090.log:578:Epoch[36] Iter[150/786] Loss: 2.276 CE: 1.125 Tri: 0.021 CE_rec: 1.090 AIRL_rec: 0.0795 Acc: 0.990 LR: 1.75e-05
logs/agreidv2_airl_4090.log:579:Epoch[36] Iter[200/786] Loss: 2.273 CE: 1.122 Tri: 0.022 CE_rec: 1.089 AIRL_rec: 0.0794 Acc: 0.990 LR: 1.75e-05
logs/agreidv2_airl_4090.log:580:Epoch[36] Iter[250/786] Loss: 2.264 CE: 1.118 Tri: 0.021 CE_rec: 1.086 AIRL_rec: 0.0785 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_airl_4090.log:581:Epoch[36] Iter[300/786] Loss: 2.259 CE: 1.116 Tri: 0.020 CE_rec: 1.084 AIRL_rec: 0.0781 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_airl_4090.log:582:Epoch[36] Iter[350/786] Loss: 2.258 CE: 1.115 Tri: 0.020 CE_rec: 1.084 AIRL_rec: 0.0777 Acc: 0.991 LR: 1.75e-05
logs/agreidv2_airl_4090.log:583:Epoch[36] Iter[400/786] Loss: 2.252 CE: 1.113 Tri: 0.019 CE_rec: 1.081 AIRL_rec: 0.0773 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_airl_4090.log:584:Epoch[36] Iter[450/786] Loss: 2.248 CE: 1.111 Tri: 0.019 CE_rec: 1.080 AIRL_rec: 0.0769 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_airl_4090.log:585:Epoch[36] Iter[500/786] Loss: 2.246 CE: 1.110 Tri: 0.019 CE_rec: 1.079 AIRL_rec: 0.0762 Acc: 0.992 LR: 1.75e-05
logs/agreidv2_airl_4090.log:586:Epoch[36] Iter[550/786] Loss: 2.242 CE: 1.108 Tri: 0.019 CE_rec: 1.077 AIRL_rec: 0.0753 Acc: 0.993 LR: 1.75e-05
logs/agreidv2_airl_4090.log:587:Epoch[36] Iter[600/786] Loss: 2.238 CE: 1.107 Tri: 0.018 CE_rec: 1.076 AIRL_rec: 0.0747 Acc: 0.993 LR: 1.75e-05
logs/agreidv2_airl_4090.log:588:Epoch[36] Iter[650/786] Loss: 2.232 CE: 1.104 Tri: 0.018 CE_rec: 1.073 AIRL_rec: 0.0738 Acc: 0.993 LR: 1.75e-05
logs/agreidv2_airl_4090.log:589:Epoch[36] Iter[700/786] Loss: 2.227 CE: 1.102 Tri: 0.017 CE_rec: 1.072 AIRL_rec: 0.0725 Acc: 0.993 LR: 1.75e-05
logs/agreidv2_airl_4090.log:590:Epoch[36] done in 112.6s  Loss=2.222 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.070 consistency=0.0714 deg_scale_mean=0.627 n_ground=28593]
logs/agreidv2_airl_4090.log:591:Epoch[37] Iter[50/786] Loss: 2.296 CE: 1.140 Tri: 0.020 CE_rec: 1.099 AIRL_rec: 0.0749 Acc: 0.988 LR: 1.64e-05
logs/agreidv2_airl_4090.log:592:Epoch[37] Iter[100/786] Loss: 2.281 CE: 1.129 Tri: 0.021 CE_rec: 1.092 AIRL_rec: 0.0758 Acc: 0.988 LR: 1.64e-05
logs/agreidv2_airl_4090.log:593:Epoch[37] Iter[150/786] Loss: 2.266 CE: 1.122 Tri: 0.020 CE_rec: 1.086 AIRL_rec: 0.0752 Acc: 0.989 LR: 1.64e-05
logs/agreidv2_airl_4090.log:594:Epoch[37] Iter[200/786] Loss: 2.257 CE: 1.118 Tri: 0.019 CE_rec: 1.083 AIRL_rec: 0.0754 Acc: 0.990 LR: 1.64e-05
logs/agreidv2_airl_4090.log:595:Epoch[37] Iter[250/786] Loss: 2.251 CE: 1.115 Tri: 0.018 CE_rec: 1.080 AIRL_rec: 0.0748 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_airl_4090.log:596:Epoch[37] Iter[300/786] Loss: 2.248 CE: 1.113 Tri: 0.020 CE_rec: 1.079 AIRL_rec: 0.0743 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_airl_4090.log:597:Epoch[37] Iter[350/786] Loss: 2.244 CE: 1.111 Tri: 0.019 CE_rec: 1.077 AIRL_rec: 0.0743 Acc: 0.991 LR: 1.64e-05
logs/agreidv2_airl_4090.log:598:Epoch[37] Iter[400/786] Loss: 2.240 CE: 1.109 Tri: 0.018 CE_rec: 1.076 AIRL_rec: 0.0735 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_airl_4090.log:599:Epoch[37] Iter[450/786] Loss: 2.236 CE: 1.107 Tri: 0.018 CE_rec: 1.075 AIRL_rec: 0.0732 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_airl_4090.log:600:Epoch[37] Iter[500/786] Loss: 2.233 CE: 1.106 Tri: 0.017 CE_rec: 1.074 AIRL_rec: 0.0732 Acc: 0.992 LR: 1.64e-05
logs/agreidv2_airl_4090.log:601:Epoch[37] Iter[550/786] Loss: 2.229 CE: 1.104 Tri: 0.017 CE_rec: 1.072 AIRL_rec: 0.0724 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_airl_4090.log:602:Epoch[37] Iter[600/786] Loss: 2.224 CE: 1.102 Tri: 0.016 CE_rec: 1.071 AIRL_rec: 0.0713 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_airl_4090.log:603:Epoch[37] Iter[650/786] Loss: 2.220 CE: 1.100 Tri: 0.016 CE_rec: 1.069 AIRL_rec: 0.0704 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_airl_4090.log:604:Epoch[37] Iter[700/786] Loss: 2.215 CE: 1.098 Tri: 0.015 CE_rec: 1.067 AIRL_rec: 0.0693 Acc: 0.993 LR: 1.64e-05
logs/agreidv2_airl_4090.log:605:Epoch[37] done in 113.8s  Loss=2.208 Acc=0.994 AIRL-ISO[lam_eff=0.500 ce_rec=1.065 consistency=0.0680 deg_scale_mean=0.622 n_ground=28866]
logs/agreidv2_airl_4090.log:606:Epoch[38] Iter[50/786] Loss: 2.251 CE: 1.113 Tri: 0.020 CE_rec: 1.083 AIRL_rec: 0.0680 Acc: 0.990 LR: 1.53e-05
logs/agreidv2_airl_4090.log:607:Epoch[38] Iter[100/786] Loss: 2.251 CE: 1.111 Tri: 0.023 CE_rec: 1.082 AIRL_rec: 0.0707 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_airl_4090.log:608:Epoch[38] Iter[150/786] Loss: 2.249 CE: 1.111 Tri: 0.021 CE_rec: 1.080 AIRL_rec: 0.0734 Acc: 0.991 LR: 1.53e-05
logs/agreidv2_airl_4090.log:609:Epoch[38] Iter[200/786] Loss: 2.242 CE: 1.108 Tri: 0.021 CE_rec: 1.077 AIRL_rec: 0.0729 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_airl_4090.log:610:Epoch[38] Iter[250/786] Loss: 2.238 CE: 1.107 Tri: 0.019 CE_rec: 1.075 AIRL_rec: 0.0727 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_airl_4090.log:611:Epoch[38] Iter[300/786] Loss: 2.239 CE: 1.106 Tri: 0.021 CE_rec: 1.076 AIRL_rec: 0.0724 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_airl_4090.log:612:Epoch[38] Iter[350/786] Loss: 2.233 CE: 1.104 Tri: 0.019 CE_rec: 1.074 AIRL_rec: 0.0723 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_airl_4090.log:613:Epoch[38] Iter[400/786] Loss: 2.230 CE: 1.103 Tri: 0.018 CE_rec: 1.073 AIRL_rec: 0.0717 Acc: 0.992 LR: 1.53e-05
logs/agreidv2_airl_4090.log:614:Epoch[38] Iter[450/786] Loss: 2.226 CE: 1.102 Tri: 0.018 CE_rec: 1.071 AIRL_rec: 0.0713 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_airl_4090.log:615:Epoch[38] Iter[500/786] Loss: 2.224 CE: 1.100 Tri: 0.018 CE_rec: 1.071 AIRL_rec: 0.0705 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_airl_4090.log:616:Epoch[38] Iter[550/786] Loss: 2.220 CE: 1.099 Tri: 0.017 CE_rec: 1.070 AIRL_rec: 0.0697 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_airl_4090.log:617:Epoch[38] Iter[600/786] Loss: 2.217 CE: 1.098 Tri: 0.016 CE_rec: 1.068 AIRL_rec: 0.0694 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_airl_4090.log:618:Epoch[38] Iter[650/786] Loss: 2.213 CE: 1.096 Tri: 0.015 CE_rec: 1.067 AIRL_rec: 0.0687 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_airl_4090.log:619:Epoch[38] Iter[700/786] Loss: 2.208 CE: 1.094 Tri: 0.015 CE_rec: 1.065 AIRL_rec: 0.0678 Acc: 0.993 LR: 1.53e-05
logs/agreidv2_airl_4090.log:620:Epoch[38] done in 114.3s  Loss=2.201 Acc=0.994 AIRL-ISO[lam_eff=0.500 ce_rec=1.062 consistency=0.0667 deg_scale_mean=0.627 n_ground=28935]
logs/agreidv2_airl_4090.log:621:Epoch[39] Iter[50/786] Loss: 2.234 CE: 1.109 Tri: 0.014 CE_rec: 1.075 AIRL_rec: 0.0711 Acc: 0.994 LR: 1.42e-05
logs/agreidv2_airl_4090.log:622:Epoch[39] Iter[100/786] Loss: 2.233 CE: 1.105 Tri: 0.017 CE_rec: 1.074 AIRL_rec: 0.0720 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_airl_4090.log:623:Epoch[39] Iter[150/786] Loss: 2.231 CE: 1.104 Tri: 0.017 CE_rec: 1.074 AIRL_rec: 0.0723 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_airl_4090.log:624:Epoch[39] Iter[200/786] Loss: 2.226 CE: 1.101 Tri: 0.016 CE_rec: 1.072 AIRL_rec: 0.0732 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_airl_4090.log:625:Epoch[39] Iter[250/786] Loss: 2.226 CE: 1.102 Tri: 0.015 CE_rec: 1.072 AIRL_rec: 0.0728 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_4090.log:626:Epoch[39] Iter[300/786] Loss: 2.225 CE: 1.101 Tri: 0.016 CE_rec: 1.071 AIRL_rec: 0.0722 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_4090.log:627:Epoch[39] Iter[350/786] Loss: 2.223 CE: 1.100 Tri: 0.017 CE_rec: 1.070 AIRL_rec: 0.0717 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_4090.log:628:Epoch[39] Iter[400/786] Loss: 2.221 CE: 1.099 Tri: 0.017 CE_rec: 1.070 AIRL_rec: 0.0713 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_4090.log:629:Epoch[39] Iter[450/786] Loss: 2.218 CE: 1.098 Tri: 0.016 CE_rec: 1.069 AIRL_rec: 0.0709 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_4090.log:630:Epoch[39] Iter[500/786] Loss: 2.214 CE: 1.096 Tri: 0.016 CE_rec: 1.067 AIRL_rec: 0.0706 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_4090.log:631:Epoch[39] Iter[550/786] Loss: 2.211 CE: 1.095 Tri: 0.015 CE_rec: 1.066 AIRL_rec: 0.0698 Acc: 0.992 LR: 1.42e-05
logs/agreidv2_airl_4090.log:632:Epoch[39] Iter[600/786] Loss: 2.207 CE: 1.093 Tri: 0.014 CE_rec: 1.065 AIRL_rec: 0.0690 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_airl_4090.log:633:Epoch[39] Iter[650/786] Loss: 2.203 CE: 1.092 Tri: 0.014 CE_rec: 1.063 AIRL_rec: 0.0683 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_airl_4090.log:634:Epoch[39] Iter[700/786] Loss: 2.198 CE: 1.089 Tri: 0.013 CE_rec: 1.061 AIRL_rec: 0.0672 Acc: 0.993 LR: 1.42e-05
logs/agreidv2_airl_4090.log:635:Epoch[39] done in 113.6s  Loss=2.193 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.059 consistency=0.0661 deg_scale_mean=0.625 n_ground=28814]
logs/agreidv2_airl_4090.log:636:Epoch[40] Iter[50/786] Loss: 2.242 CE: 1.112 Tri: 0.013 CE_rec: 1.082 AIRL_rec: 0.0689 Acc: 0.989 LR: 1.31e-05
logs/agreidv2_airl_4090.log:637:Epoch[40] Iter[100/786] Loss: 2.221 CE: 1.100 Tri: 0.014 CE_rec: 1.071 AIRL_rec: 0.0688 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_airl_4090.log:638:Epoch[40] Iter[150/786] Loss: 2.219 CE: 1.100 Tri: 0.014 CE_rec: 1.070 AIRL_rec: 0.0689 Acc: 0.991 LR: 1.31e-05
logs/agreidv2_airl_4090.log:639:Epoch[40] Iter[200/786] Loss: 2.220 CE: 1.099 Tri: 0.016 CE_rec: 1.071 AIRL_rec: 0.0689 Acc: 0.991 LR: 1.31e-05
logs/agreidv2_airl_4090.log:640:Epoch[40] Iter[250/786] Loss: 2.214 CE: 1.097 Tri: 0.015 CE_rec: 1.068 AIRL_rec: 0.0683 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_airl_4090.log:641:Epoch[40] Iter[300/786] Loss: 2.208 CE: 1.094 Tri: 0.015 CE_rec: 1.065 AIRL_rec: 0.0676 Acc: 0.992 LR: 1.31e-05
logs/agreidv2_airl_4090.log:642:Epoch[40] Iter[350/786] Loss: 2.204 CE: 1.093 Tri: 0.013 CE_rec: 1.064 AIRL_rec: 0.0675 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_airl_4090.log:643:Epoch[40] Iter[400/786] Loss: 2.200 CE: 1.091 Tri: 0.013 CE_rec: 1.063 AIRL_rec: 0.0670 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_airl_4090.log:644:Epoch[40] Iter[450/786] Loss: 2.198 CE: 1.090 Tri: 0.013 CE_rec: 1.062 AIRL_rec: 0.0663 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_airl_4090.log:645:Epoch[40] Iter[500/786] Loss: 2.195 CE: 1.089 Tri: 0.012 CE_rec: 1.061 AIRL_rec: 0.0661 Acc: 0.993 LR: 1.31e-05
logs/agreidv2_airl_4090.log:646:Epoch[40] Iter[550/786] Loss: 2.192 CE: 1.087 Tri: 0.012 CE_rec: 1.060 AIRL_rec: 0.0655 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_airl_4090.log:647:Epoch[40] Iter[600/786] Loss: 2.187 CE: 1.085 Tri: 0.012 CE_rec: 1.058 AIRL_rec: 0.0646 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_airl_4090.log:648:Epoch[40] Iter[650/786] Loss: 2.185 CE: 1.084 Tri: 0.012 CE_rec: 1.057 AIRL_rec: 0.0639 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_airl_4090.log:649:Epoch[40] Iter[700/786] Loss: 2.181 CE: 1.083 Tri: 0.011 CE_rec: 1.056 AIRL_rec: 0.0633 Acc: 0.994 LR: 1.31e-05
logs/agreidv2_airl_4090.log:650:Epoch[40] done in 113.1s  Loss=2.176 Acc=0.994 AIRL-ISO[lam_eff=0.500 ce_rec=1.054 consistency=0.0623 deg_scale_mean=0.626 n_ground=28706]
logs/agreidv2_airl_4090.log:652:    [A->G] mAP=76.70  R1=85.10  R5=91.17  mINP=49.07
logs/agreidv2_airl_4090.log:653:    [G->A] mAP=76.49  R1=84.21  R5=89.95  mINP=42.15
logs/agreidv2_airl_4090.log:654:    [mean] mAP=76.60  R1=84.65
logs/agreidv2_airl_4090.log:655:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 40 ----
logs/agreidv2_airl_4090.log:656:    [A->G] full mAP=76.70 R1=85.10 | rec mAP=75.93 R1=83.40 | FUSE mAP=78.02 R1=85.70
logs/agreidv2_airl_4090.log:657:    [G->A] full mAP=76.49 R1=84.21 | rec mAP=75.44 R1=84.43 | FUSE mAP=77.87 R1=85.42
logs/agreidv2_airl_4090.log:659:    * new best mean mAP=77.95 (epoch 40) saved
logs/agreidv2_airl_4090.log:660:Epoch[41] Iter[50/786] Loss: 2.237 CE: 1.107 Tri: 0.018 CE_rec: 1.078 AIRL_rec: 0.0685 Acc: 0.992 LR: 1.21e-05
logs/agreidv2_airl_4090.log:661:Epoch[41] Iter[100/786] Loss: 2.223 CE: 1.101 Tri: 0.014 CE_rec: 1.073 AIRL_rec: 0.0689 Acc: 0.991 LR: 1.21e-05
logs/agreidv2_airl_4090.log:662:Epoch[41] Iter[150/786] Loss: 2.218 CE: 1.097 Tri: 0.017 CE_rec: 1.069 AIRL_rec: 0.0695 Acc: 0.991 LR: 1.21e-05
logs/agreidv2_airl_4090.log:663:Epoch[41] Iter[200/786] Loss: 2.213 CE: 1.096 Tri: 0.016 CE_rec: 1.067 AIRL_rec: 0.0690 Acc: 0.992 LR: 1.21e-05
logs/agreidv2_airl_4090.log:664:Epoch[41] Iter[250/786] Loss: 2.205 CE: 1.092 Tri: 0.014 CE_rec: 1.064 AIRL_rec: 0.0682 Acc: 0.992 LR: 1.21e-05
logs/agreidv2_airl_4090.log:665:Epoch[41] Iter[300/786] Loss: 2.203 CE: 1.091 Tri: 0.014 CE_rec: 1.064 AIRL_rec: 0.0677 Acc: 0.992 LR: 1.21e-05
logs/agreidv2_airl_4090.log:666:Epoch[41] Iter[350/786] Loss: 2.200 CE: 1.090 Tri: 0.013 CE_rec: 1.063 AIRL_rec: 0.0673 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_4090.log:667:Epoch[41] Iter[400/786] Loss: 2.195 CE: 1.088 Tri: 0.013 CE_rec: 1.062 AIRL_rec: 0.0661 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_4090.log:668:Epoch[41] Iter[450/786] Loss: 2.191 CE: 1.086 Tri: 0.012 CE_rec: 1.060 AIRL_rec: 0.0654 Acc: 0.993 LR: 1.21e-05
logs/agreidv2_airl_4090.log:669:Epoch[41] Iter[500/786] Loss: 2.187 CE: 1.084 Tri: 0.012 CE_rec: 1.058 AIRL_rec: 0.0648 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_airl_4090.log:670:Epoch[41] Iter[550/786] Loss: 2.186 CE: 1.084 Tri: 0.012 CE_rec: 1.058 AIRL_rec: 0.0642 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_airl_4090.log:671:Epoch[41] Iter[600/786] Loss: 2.183 CE: 1.082 Tri: 0.012 CE_rec: 1.057 AIRL_rec: 0.0636 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_airl_4090.log:672:Epoch[41] Iter[650/786] Loss: 2.180 CE: 1.081 Tri: 0.012 CE_rec: 1.056 AIRL_rec: 0.0629 Acc: 0.994 LR: 1.21e-05
logs/agreidv2_airl_4090.log:673:Epoch[41] Iter[700/786] Loss: 2.177 CE: 1.080 Tri: 0.012 CE_rec: 1.054 AIRL_rec: 0.0620 Acc: 0.995 LR: 1.21e-05
logs/agreidv2_airl_4090.log:674:Epoch[41] done in 114.0s  Loss=2.172 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.052 consistency=0.0609 deg_scale_mean=0.626 n_ground=28916]
logs/agreidv2_airl_4090.log:675:Epoch[42] Iter[50/786] Loss: 2.216 CE: 1.099 Tri: 0.012 CE_rec: 1.073 AIRL_rec: 0.0645 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_airl_4090.log:676:Epoch[42] Iter[100/786] Loss: 2.208 CE: 1.094 Tri: 0.016 CE_rec: 1.066 AIRL_rec: 0.0635 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_airl_4090.log:677:Epoch[42] Iter[150/786] Loss: 2.199 CE: 1.090 Tri: 0.014 CE_rec: 1.063 AIRL_rec: 0.0636 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_airl_4090.log:678:Epoch[42] Iter[200/786] Loss: 2.193 CE: 1.088 Tri: 0.012 CE_rec: 1.061 AIRL_rec: 0.0639 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_airl_4090.log:679:Epoch[42] Iter[250/786] Loss: 2.193 CE: 1.088 Tri: 0.013 CE_rec: 1.060 AIRL_rec: 0.0636 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_airl_4090.log:680:Epoch[42] Iter[300/786] Loss: 2.188 CE: 1.086 Tri: 0.012 CE_rec: 1.058 AIRL_rec: 0.0636 Acc: 0.993 LR: 1.11e-05
logs/agreidv2_airl_4090.log:681:Epoch[42] Iter[350/786] Loss: 2.185 CE: 1.084 Tri: 0.012 CE_rec: 1.057 AIRL_rec: 0.0632 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_4090.log:682:Epoch[42] Iter[400/786] Loss: 2.184 CE: 1.083 Tri: 0.012 CE_rec: 1.057 AIRL_rec: 0.0630 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_4090.log:683:Epoch[42] Iter[450/786] Loss: 2.180 CE: 1.082 Tri: 0.012 CE_rec: 1.055 AIRL_rec: 0.0629 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_4090.log:684:Epoch[42] Iter[500/786] Loss: 2.178 CE: 1.081 Tri: 0.011 CE_rec: 1.055 AIRL_rec: 0.0622 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_4090.log:685:Epoch[42] Iter[550/786] Loss: 2.176 CE: 1.080 Tri: 0.011 CE_rec: 1.054 AIRL_rec: 0.0616 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_4090.log:686:Epoch[42] Iter[600/786] Loss: 2.173 CE: 1.079 Tri: 0.011 CE_rec: 1.053 AIRL_rec: 0.0609 Acc: 0.994 LR: 1.11e-05
logs/agreidv2_airl_4090.log:687:Epoch[42] Iter[650/786] Loss: 2.170 CE: 1.078 Tri: 0.010 CE_rec: 1.052 AIRL_rec: 0.0602 Acc: 0.995 LR: 1.11e-05
logs/agreidv2_airl_4090.log:688:Epoch[42] Iter[700/786] Loss: 2.166 CE: 1.076 Tri: 0.010 CE_rec: 1.050 AIRL_rec: 0.0594 Acc: 0.995 LR: 1.11e-05
logs/agreidv2_airl_4090.log:689:Epoch[42] done in 112.8s  Loss=2.161 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.049 consistency=0.0585 deg_scale_mean=0.627 n_ground=28528]
logs/agreidv2_airl_4090.log:690:Epoch[43] Iter[50/786] Loss: 2.204 CE: 1.094 Tri: 0.010 CE_rec: 1.068 AIRL_rec: 0.0648 Acc: 0.993 LR: 1.00e-05
logs/agreidv2_airl_4090.log:691:Epoch[43] Iter[100/786] Loss: 2.192 CE: 1.087 Tri: 0.010 CE_rec: 1.061 AIRL_rec: 0.0656 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_4090.log:692:Epoch[43] Iter[150/786] Loss: 2.184 CE: 1.083 Tri: 0.010 CE_rec: 1.058 AIRL_rec: 0.0644 Acc: 0.996 LR: 1.00e-05
logs/agreidv2_airl_4090.log:693:Epoch[43] Iter[200/786] Loss: 2.180 CE: 1.081 Tri: 0.010 CE_rec: 1.057 AIRL_rec: 0.0635 Acc: 0.996 LR: 1.00e-05
logs/agreidv2_airl_4090.log:694:Epoch[43] Iter[250/786] Loss: 2.178 CE: 1.081 Tri: 0.009 CE_rec: 1.056 AIRL_rec: 0.0629 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_4090.log:695:Epoch[43] Iter[300/786] Loss: 2.175 CE: 1.080 Tri: 0.009 CE_rec: 1.055 AIRL_rec: 0.0624 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_4090.log:696:Epoch[43] Iter[350/786] Loss: 2.172 CE: 1.078 Tri: 0.009 CE_rec: 1.054 AIRL_rec: 0.0621 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_4090.log:697:Epoch[43] Iter[400/786] Loss: 2.170 CE: 1.077 Tri: 0.008 CE_rec: 1.053 AIRL_rec: 0.0616 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_4090.log:698:Epoch[43] Iter[450/786] Loss: 2.169 CE: 1.077 Tri: 0.008 CE_rec: 1.053 AIRL_rec: 0.0610 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_4090.log:699:Epoch[43] Iter[500/786] Loss: 2.168 CE: 1.077 Tri: 0.008 CE_rec: 1.053 AIRL_rec: 0.0605 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_4090.log:700:Epoch[43] Iter[550/786] Loss: 2.166 CE: 1.076 Tri: 0.008 CE_rec: 1.052 AIRL_rec: 0.0599 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_4090.log:701:Epoch[43] Iter[600/786] Loss: 2.164 CE: 1.075 Tri: 0.008 CE_rec: 1.051 AIRL_rec: 0.0594 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_4090.log:702:Epoch[43] Iter[650/786] Loss: 2.162 CE: 1.074 Tri: 0.008 CE_rec: 1.051 AIRL_rec: 0.0586 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_4090.log:703:Epoch[43] Iter[700/786] Loss: 2.160 CE: 1.073 Tri: 0.008 CE_rec: 1.049 AIRL_rec: 0.0579 Acc: 0.995 LR: 1.00e-05
logs/agreidv2_airl_4090.log:704:Epoch[43] done in 113.2s  Loss=2.155 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.048 consistency=0.0571 deg_scale_mean=0.625 n_ground=28699]
logs/agreidv2_airl_4090.log:705:Epoch[44] Iter[50/786] Loss: 2.206 CE: 1.091 Tri: 0.014 CE_rec: 1.069 AIRL_rec: 0.0638 Acc: 0.991 LR: 9.07e-06
logs/agreidv2_airl_4090.log:706:Epoch[44] Iter[100/786] Loss: 2.197 CE: 1.088 Tri: 0.015 CE_rec: 1.063 AIRL_rec: 0.0611 Acc: 0.991 LR: 9.07e-06
logs/agreidv2_airl_4090.log:707:Epoch[44] Iter[150/786] Loss: 2.189 CE: 1.085 Tri: 0.014 CE_rec: 1.060 AIRL_rec: 0.0602 Acc: 0.992 LR: 9.07e-06
logs/agreidv2_airl_4090.log:708:Epoch[44] Iter[200/786] Loss: 2.184 CE: 1.083 Tri: 0.014 CE_rec: 1.057 AIRL_rec: 0.0600 Acc: 0.992 LR: 9.07e-06
logs/agreidv2_airl_4090.log:709:Epoch[44] Iter[250/786] Loss: 2.180 CE: 1.081 Tri: 0.013 CE_rec: 1.056 AIRL_rec: 0.0599 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_airl_4090.log:710:Epoch[44] Iter[300/786] Loss: 2.176 CE: 1.079 Tri: 0.012 CE_rec: 1.055 AIRL_rec: 0.0600 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_airl_4090.log:711:Epoch[44] Iter[350/786] Loss: 2.172 CE: 1.078 Tri: 0.011 CE_rec: 1.054 AIRL_rec: 0.0597 Acc: 0.993 LR: 9.07e-06
logs/agreidv2_airl_4090.log:712:Epoch[44] Iter[400/786] Loss: 2.169 CE: 1.076 Tri: 0.011 CE_rec: 1.053 AIRL_rec: 0.0594 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_4090.log:713:Epoch[44] Iter[450/786] Loss: 2.167 CE: 1.075 Tri: 0.010 CE_rec: 1.052 AIRL_rec: 0.0590 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_4090.log:714:Epoch[44] Iter[500/786] Loss: 2.166 CE: 1.074 Tri: 0.011 CE_rec: 1.052 AIRL_rec: 0.0588 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_4090.log:715:Epoch[44] Iter[550/786] Loss: 2.164 CE: 1.074 Tri: 0.010 CE_rec: 1.051 AIRL_rec: 0.0584 Acc: 0.994 LR: 9.07e-06
logs/agreidv2_airl_4090.log:716:Epoch[44] Iter[600/786] Loss: 2.160 CE: 1.072 Tri: 0.010 CE_rec: 1.049 AIRL_rec: 0.0578 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_airl_4090.log:717:Epoch[44] Iter[650/786] Loss: 2.158 CE: 1.071 Tri: 0.010 CE_rec: 1.048 AIRL_rec: 0.0573 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_airl_4090.log:718:Epoch[44] Iter[700/786] Loss: 2.154 CE: 1.070 Tri: 0.009 CE_rec: 1.047 AIRL_rec: 0.0564 Acc: 0.995 LR: 9.07e-06
logs/agreidv2_airl_4090.log:719:Epoch[44] done in 113.2s  Loss=2.150 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.045 consistency=0.0554 deg_scale_mean=0.625 n_ground=28734]
logs/agreidv2_airl_4090.log:720:Epoch[45] Iter[50/786] Loss: 2.186 CE: 1.086 Tri: 0.012 CE_rec: 1.059 AIRL_rec: 0.0581 Acc: 0.993 LR: 8.12e-06
logs/agreidv2_airl_4090.log:721:Epoch[45] Iter[100/786] Loss: 2.177 CE: 1.081 Tri: 0.010 CE_rec: 1.056 AIRL_rec: 0.0587 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_airl_4090.log:722:Epoch[45] Iter[150/786] Loss: 2.168 CE: 1.077 Tri: 0.008 CE_rec: 1.053 AIRL_rec: 0.0597 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_airl_4090.log:723:Epoch[45] Iter[200/786] Loss: 2.168 CE: 1.077 Tri: 0.009 CE_rec: 1.052 AIRL_rec: 0.0598 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_4090.log:724:Epoch[45] Iter[250/786] Loss: 2.164 CE: 1.075 Tri: 0.009 CE_rec: 1.051 AIRL_rec: 0.0593 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_4090.log:725:Epoch[45] Iter[300/786] Loss: 2.166 CE: 1.076 Tri: 0.009 CE_rec: 1.052 AIRL_rec: 0.0586 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_airl_4090.log:726:Epoch[45] Iter[350/786] Loss: 2.166 CE: 1.076 Tri: 0.009 CE_rec: 1.052 AIRL_rec: 0.0588 Acc: 0.993 LR: 8.12e-06
logs/agreidv2_airl_4090.log:727:Epoch[45] Iter[400/786] Loss: 2.163 CE: 1.074 Tri: 0.010 CE_rec: 1.050 AIRL_rec: 0.0585 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_airl_4090.log:728:Epoch[45] Iter[450/786] Loss: 2.160 CE: 1.073 Tri: 0.009 CE_rec: 1.049 AIRL_rec: 0.0584 Acc: 0.994 LR: 8.12e-06
logs/agreidv2_airl_4090.log:729:Epoch[45] Iter[500/786] Loss: 2.157 CE: 1.072 Tri: 0.009 CE_rec: 1.048 AIRL_rec: 0.0578 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_4090.log:730:Epoch[45] Iter[550/786] Loss: 2.155 CE: 1.071 Tri: 0.009 CE_rec: 1.048 AIRL_rec: 0.0571 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_4090.log:731:Epoch[45] Iter[600/786] Loss: 2.153 CE: 1.070 Tri: 0.008 CE_rec: 1.047 AIRL_rec: 0.0566 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_4090.log:732:Epoch[45] Iter[650/786] Loss: 2.150 CE: 1.068 Tri: 0.008 CE_rec: 1.045 AIRL_rec: 0.0560 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_4090.log:733:Epoch[45] Iter[700/786] Loss: 2.146 CE: 1.067 Tri: 0.008 CE_rec: 1.044 AIRL_rec: 0.0551 Acc: 0.995 LR: 8.12e-06
logs/agreidv2_airl_4090.log:734:Epoch[45] done in 113.7s  Loss=2.142 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.043 consistency=0.0544 deg_scale_mean=0.625 n_ground=28796]
logs/agreidv2_airl_4090.log:735:Epoch[46] Iter[50/786] Loss: 2.155 CE: 1.072 Tri: 0.007 CE_rec: 1.048 AIRL_rec: 0.0569 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_4090.log:736:Epoch[46] Iter[100/786] Loss: 2.158 CE: 1.073 Tri: 0.006 CE_rec: 1.051 AIRL_rec: 0.0583 Acc: 0.994 LR: 7.21e-06
logs/agreidv2_airl_4090.log:737:Epoch[46] Iter[150/786] Loss: 2.157 CE: 1.072 Tri: 0.007 CE_rec: 1.049 AIRL_rec: 0.0592 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_airl_4090.log:738:Epoch[46] Iter[200/786] Loss: 2.155 CE: 1.070 Tri: 0.008 CE_rec: 1.048 AIRL_rec: 0.0586 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_airl_4090.log:739:Epoch[46] Iter[250/786] Loss: 2.154 CE: 1.070 Tri: 0.007 CE_rec: 1.048 AIRL_rec: 0.0581 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_airl_4090.log:740:Epoch[46] Iter[300/786] Loss: 2.153 CE: 1.070 Tri: 0.007 CE_rec: 1.047 AIRL_rec: 0.0581 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_airl_4090.log:741:Epoch[46] Iter[350/786] Loss: 2.151 CE: 1.068 Tri: 0.007 CE_rec: 1.046 AIRL_rec: 0.0578 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_4090.log:742:Epoch[46] Iter[400/786] Loss: 2.149 CE: 1.067 Tri: 0.007 CE_rec: 1.045 AIRL_rec: 0.0575 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_4090.log:743:Epoch[46] Iter[450/786] Loss: 2.146 CE: 1.066 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0568 Acc: 0.995 LR: 7.21e-06
logs/agreidv2_airl_4090.log:744:Epoch[46] Iter[500/786] Loss: 2.144 CE: 1.065 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0563 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_4090.log:745:Epoch[46] Iter[550/786] Loss: 2.142 CE: 1.064 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0558 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_4090.log:746:Epoch[46] Iter[600/786] Loss: 2.140 CE: 1.064 Tri: 0.007 CE_rec: 1.042 AIRL_rec: 0.0553 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_4090.log:747:Epoch[46] Iter[650/786] Loss: 2.138 CE: 1.063 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0549 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_4090.log:748:Epoch[46] Iter[700/786] Loss: 2.136 CE: 1.062 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0542 Acc: 0.996 LR: 7.21e-06
logs/agreidv2_airl_4090.log:749:Epoch[46] done in 112.8s  Loss=2.133 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.039 consistency=0.0535 deg_scale_mean=0.625 n_ground=28635]
logs/agreidv2_airl_4090.log:750:Epoch[47] Iter[50/786] Loss: 2.156 CE: 1.074 Tri: 0.005 CE_rec: 1.049 AIRL_rec: 0.0551 Acc: 0.994 LR: 6.35e-06
logs/agreidv2_airl_4090.log:751:Epoch[47] Iter[100/786] Loss: 2.158 CE: 1.072 Tri: 0.006 CE_rec: 1.051 AIRL_rec: 0.0566 Acc: 0.994 LR: 6.35e-06
logs/agreidv2_airl_4090.log:752:Epoch[47] Iter[150/786] Loss: 2.155 CE: 1.071 Tri: 0.006 CE_rec: 1.050 AIRL_rec: 0.0569 Acc: 0.994 LR: 6.35e-06
logs/agreidv2_airl_4090.log:753:Epoch[47] Iter[200/786] Loss: 2.153 CE: 1.070 Tri: 0.006 CE_rec: 1.048 AIRL_rec: 0.0572 Acc: 0.994 LR: 6.35e-06
logs/agreidv2_airl_4090.log:754:Epoch[47] Iter[250/786] Loss: 2.148 CE: 1.068 Tri: 0.006 CE_rec: 1.047 AIRL_rec: 0.0567 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_4090.log:755:Epoch[47] Iter[300/786] Loss: 2.146 CE: 1.067 Tri: 0.005 CE_rec: 1.045 AIRL_rec: 0.0566 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_4090.log:756:Epoch[47] Iter[350/786] Loss: 2.144 CE: 1.066 Tri: 0.006 CE_rec: 1.044 AIRL_rec: 0.0563 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_4090.log:757:Epoch[47] Iter[400/786] Loss: 2.143 CE: 1.065 Tri: 0.005 CE_rec: 1.044 AIRL_rec: 0.0562 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_4090.log:758:Epoch[47] Iter[450/786] Loss: 2.141 CE: 1.065 Tri: 0.005 CE_rec: 1.044 AIRL_rec: 0.0560 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_4090.log:759:Epoch[47] Iter[500/786] Loss: 2.140 CE: 1.064 Tri: 0.005 CE_rec: 1.043 AIRL_rec: 0.0558 Acc: 0.995 LR: 6.35e-06
logs/agreidv2_airl_4090.log:760:Epoch[47] Iter[550/786] Loss: 2.139 CE: 1.063 Tri: 0.005 CE_rec: 1.042 AIRL_rec: 0.0553 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_airl_4090.log:761:Epoch[47] Iter[600/786] Loss: 2.137 CE: 1.063 Tri: 0.005 CE_rec: 1.042 AIRL_rec: 0.0548 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_airl_4090.log:762:Epoch[47] Iter[650/786] Loss: 2.134 CE: 1.061 Tri: 0.005 CE_rec: 1.041 AIRL_rec: 0.0542 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_airl_4090.log:763:Epoch[47] Iter[700/786] Loss: 2.131 CE: 1.060 Tri: 0.005 CE_rec: 1.039 AIRL_rec: 0.0534 Acc: 0.996 LR: 6.35e-06
logs/agreidv2_airl_4090.log:764:Epoch[47] done in 113.2s  Loss=2.127 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.038 consistency=0.0525 deg_scale_mean=0.625 n_ground=28675]
logs/agreidv2_airl_4090.log:765:Epoch[48] Iter[50/786] Loss: 2.155 CE: 1.072 Tri: 0.005 CE_rec: 1.049 AIRL_rec: 0.0598 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_4090.log:766:Epoch[48] Iter[100/786] Loss: 2.151 CE: 1.068 Tri: 0.009 CE_rec: 1.046 AIRL_rec: 0.0578 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_4090.log:767:Epoch[48] Iter[150/786] Loss: 2.154 CE: 1.069 Tri: 0.009 CE_rec: 1.047 AIRL_rec: 0.0580 Acc: 0.994 LR: 5.52e-06
logs/agreidv2_airl_4090.log:768:Epoch[48] Iter[200/786] Loss: 2.150 CE: 1.067 Tri: 0.009 CE_rec: 1.045 AIRL_rec: 0.0578 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_4090.log:769:Epoch[48] Iter[250/786] Loss: 2.147 CE: 1.066 Tri: 0.008 CE_rec: 1.045 AIRL_rec: 0.0569 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_4090.log:770:Epoch[48] Iter[300/786] Loss: 2.145 CE: 1.065 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0565 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_4090.log:771:Epoch[48] Iter[350/786] Loss: 2.144 CE: 1.065 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0565 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_4090.log:772:Epoch[48] Iter[400/786] Loss: 2.143 CE: 1.064 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0562 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_4090.log:773:Epoch[48] Iter[450/786] Loss: 2.141 CE: 1.063 Tri: 0.008 CE_rec: 1.042 AIRL_rec: 0.0559 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_4090.log:774:Epoch[48] Iter[500/786] Loss: 2.140 CE: 1.063 Tri: 0.007 CE_rec: 1.042 AIRL_rec: 0.0556 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_4090.log:775:Epoch[48] Iter[550/786] Loss: 2.137 CE: 1.062 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0549 Acc: 0.995 LR: 5.52e-06
logs/agreidv2_airl_4090.log:776:Epoch[48] Iter[600/786] Loss: 2.135 CE: 1.061 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0543 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_airl_4090.log:777:Epoch[48] Iter[650/786] Loss: 2.133 CE: 1.060 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0536 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_airl_4090.log:778:Epoch[48] Iter[700/786] Loss: 2.130 CE: 1.058 Tri: 0.007 CE_rec: 1.038 AIRL_rec: 0.0530 Acc: 0.996 LR: 5.52e-06
logs/agreidv2_airl_4090.log:779:Epoch[48] done in 112.8s  Loss=2.127 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.037 consistency=0.0523 deg_scale_mean=0.626 n_ground=28637]
logs/agreidv2_airl_4090.log:780:Epoch[49] Iter[50/786] Loss: 2.161 CE: 1.073 Tri: 0.008 CE_rec: 1.053 AIRL_rec: 0.0561 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_airl_4090.log:781:Epoch[49] Iter[100/786] Loss: 2.158 CE: 1.071 Tri: 0.009 CE_rec: 1.050 AIRL_rec: 0.0558 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_airl_4090.log:782:Epoch[49] Iter[150/786] Loss: 2.150 CE: 1.067 Tri: 0.009 CE_rec: 1.046 AIRL_rec: 0.0554 Acc: 0.994 LR: 4.74e-06
logs/agreidv2_airl_4090.log:783:Epoch[49] Iter[200/786] Loss: 2.147 CE: 1.066 Tri: 0.008 CE_rec: 1.045 AIRL_rec: 0.0555 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_4090.log:784:Epoch[49] Iter[250/786] Loss: 2.148 CE: 1.065 Tri: 0.011 CE_rec: 1.044 AIRL_rec: 0.0555 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_4090.log:785:Epoch[49] Iter[300/786] Loss: 2.147 CE: 1.065 Tri: 0.010 CE_rec: 1.044 AIRL_rec: 0.0554 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_4090.log:786:Epoch[49] Iter[350/786] Loss: 2.145 CE: 1.064 Tri: 0.010 CE_rec: 1.043 AIRL_rec: 0.0550 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_4090.log:787:Epoch[49] Iter[400/786] Loss: 2.140 CE: 1.062 Tri: 0.009 CE_rec: 1.042 AIRL_rec: 0.0545 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_4090.log:788:Epoch[49] Iter[450/786] Loss: 2.137 CE: 1.061 Tri: 0.009 CE_rec: 1.041 AIRL_rec: 0.0542 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_4090.log:789:Epoch[49] Iter[500/786] Loss: 2.135 CE: 1.060 Tri: 0.008 CE_rec: 1.040 AIRL_rec: 0.0537 Acc: 0.995 LR: 4.74e-06
logs/agreidv2_airl_4090.log:790:Epoch[49] Iter[550/786] Loss: 2.133 CE: 1.059 Tri: 0.008 CE_rec: 1.039 AIRL_rec: 0.0534 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_airl_4090.log:791:Epoch[49] Iter[600/786] Loss: 2.131 CE: 1.058 Tri: 0.008 CE_rec: 1.038 AIRL_rec: 0.0531 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_airl_4090.log:792:Epoch[49] Iter[650/786] Loss: 2.129 CE: 1.058 Tri: 0.008 CE_rec: 1.038 AIRL_rec: 0.0523 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_airl_4090.log:793:Epoch[49] Iter[700/786] Loss: 2.126 CE: 1.056 Tri: 0.008 CE_rec: 1.037 AIRL_rec: 0.0516 Acc: 0.996 LR: 4.74e-06
logs/agreidv2_airl_4090.log:794:Epoch[49] done in 113.5s  Loss=2.123 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.035 consistency=0.0509 deg_scale_mean=0.623 n_ground=28779]
logs/agreidv2_airl_4090.log:795:Epoch[50] Iter[50/786] Loss: 2.136 CE: 1.061 Tri: 0.006 CE_rec: 1.043 AIRL_rec: 0.0519 Acc: 0.998 LR: 4.02e-06
logs/agreidv2_airl_4090.log:796:Epoch[50] Iter[100/786] Loss: 2.132 CE: 1.061 Tri: 0.005 CE_rec: 1.041 AIRL_rec: 0.0507 Acc: 0.995 LR: 4.02e-06
logs/agreidv2_airl_4090.log:797:Epoch[50] Iter[150/786] Loss: 2.129 CE: 1.059 Tri: 0.005 CE_rec: 1.039 AIRL_rec: 0.0514 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_4090.log:798:Epoch[50] Iter[200/786] Loss: 2.129 CE: 1.058 Tri: 0.005 CE_rec: 1.040 AIRL_rec: 0.0518 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_4090.log:799:Epoch[50] Iter[250/786] Loss: 2.131 CE: 1.059 Tri: 0.005 CE_rec: 1.040 AIRL_rec: 0.0528 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_4090.log:800:Epoch[50] Iter[300/786] Loss: 2.130 CE: 1.058 Tri: 0.006 CE_rec: 1.040 AIRL_rec: 0.0531 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_4090.log:801:Epoch[50] Iter[350/786] Loss: 2.128 CE: 1.057 Tri: 0.005 CE_rec: 1.039 AIRL_rec: 0.0527 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_4090.log:802:Epoch[50] Iter[400/786] Loss: 2.126 CE: 1.056 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0525 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_4090.log:803:Epoch[50] Iter[450/786] Loss: 2.125 CE: 1.056 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0521 Acc: 0.996 LR: 4.02e-06
logs/agreidv2_airl_4090.log:804:Epoch[50] Iter[500/786] Loss: 2.124 CE: 1.055 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0518 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_airl_4090.log:805:Epoch[50] Iter[550/786] Loss: 2.122 CE: 1.054 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0514 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_airl_4090.log:806:Epoch[50] Iter[600/786] Loss: 2.120 CE: 1.054 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0510 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_airl_4090.log:807:Epoch[50] Iter[650/786] Loss: 2.119 CE: 1.053 Tri: 0.006 CE_rec: 1.035 AIRL_rec: 0.0505 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_airl_4090.log:808:Epoch[50] Iter[700/786] Loss: 2.116 CE: 1.052 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0499 Acc: 0.997 LR: 4.02e-06
logs/agreidv2_airl_4090.log:809:Epoch[50] done in 113.6s  Loss=2.113 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.033 consistency=0.0492 deg_scale_mean=0.625 n_ground=28817]
logs/agreidv2_airl_4090.log:811:    [A->G] mAP=78.34  R1=86.29  R5=91.85  mINP=52.48
logs/agreidv2_airl_4090.log:812:    [G->A] mAP=79.19  R1=86.69  R5=90.61  mINP=46.79
logs/agreidv2_airl_4090.log:813:    [mean] mAP=78.77  R1=86.49
logs/agreidv2_airl_4090.log:814:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 50 ----
logs/agreidv2_airl_4090.log:815:    [A->G] full mAP=78.34 R1=86.29 | rec mAP=77.75 R1=85.14 | FUSE mAP=79.63 R1=87.14
logs/agreidv2_airl_4090.log:816:    [G->A] full mAP=79.19 R1=86.69 | rec mAP=77.15 R1=85.15 | FUSE mAP=80.18 R1=87.24
logs/agreidv2_airl_4090.log:818:    * new best mean mAP=79.90 (epoch 50) saved
logs/agreidv2_airl_4090.log:819:Epoch[51] Iter[50/786] Loss: 2.132 CE: 1.060 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0533 Acc: 0.998 LR: 3.34e-06
logs/agreidv2_airl_4090.log:820:Epoch[51] Iter[100/786] Loss: 2.133 CE: 1.061 Tri: 0.006 CE_rec: 1.040 AIRL_rec: 0.0537 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_airl_4090.log:821:Epoch[51] Iter[150/786] Loss: 2.132 CE: 1.059 Tri: 0.007 CE_rec: 1.038 AIRL_rec: 0.0539 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_airl_4090.log:822:Epoch[51] Iter[200/786] Loss: 2.131 CE: 1.058 Tri: 0.008 CE_rec: 1.038 AIRL_rec: 0.0531 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_airl_4090.log:823:Epoch[51] Iter[250/786] Loss: 2.130 CE: 1.058 Tri: 0.008 CE_rec: 1.038 AIRL_rec: 0.0533 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_airl_4090.log:824:Epoch[51] Iter[300/786] Loss: 2.130 CE: 1.058 Tri: 0.007 CE_rec: 1.038 AIRL_rec: 0.0530 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_airl_4090.log:825:Epoch[51] Iter[350/786] Loss: 2.129 CE: 1.057 Tri: 0.007 CE_rec: 1.038 AIRL_rec: 0.0529 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_airl_4090.log:826:Epoch[51] Iter[400/786] Loss: 2.127 CE: 1.056 Tri: 0.007 CE_rec: 1.037 AIRL_rec: 0.0526 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_airl_4090.log:827:Epoch[51] Iter[450/786] Loss: 2.124 CE: 1.056 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0524 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_airl_4090.log:828:Epoch[51] Iter[500/786] Loss: 2.123 CE: 1.055 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0520 Acc: 0.996 LR: 3.34e-06
logs/agreidv2_airl_4090.log:829:Epoch[51] Iter[550/786] Loss: 2.120 CE: 1.054 Tri: 0.006 CE_rec: 1.035 AIRL_rec: 0.0516 Acc: 0.997 LR: 3.34e-06
logs/agreidv2_airl_4090.log:830:Epoch[51] Iter[600/786] Loss: 2.118 CE: 1.053 Tri: 0.006 CE_rec: 1.034 AIRL_rec: 0.0510 Acc: 0.997 LR: 3.34e-06
logs/agreidv2_airl_4090.log:831:Epoch[51] Iter[650/786] Loss: 2.117 CE: 1.053 Tri: 0.006 CE_rec: 1.034 AIRL_rec: 0.0506 Acc: 0.997 LR: 3.34e-06
logs/agreidv2_airl_4090.log:832:Epoch[51] Iter[700/786] Loss: 2.115 CE: 1.052 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0498 Acc: 0.997 LR: 3.34e-06
logs/agreidv2_airl_4090.log:833:Epoch[51] done in 113.5s  Loss=2.112 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.032 consistency=0.0491 deg_scale_mean=0.625 n_ground=28774]
logs/agreidv2_airl_4090.log:834:Epoch[52] Iter[50/786] Loss: 2.127 CE: 1.056 Tri: 0.004 CE_rec: 1.040 AIRL_rec: 0.0536 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_airl_4090.log:835:Epoch[52] Iter[100/786] Loss: 2.125 CE: 1.056 Tri: 0.004 CE_rec: 1.038 AIRL_rec: 0.0530 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_airl_4090.log:836:Epoch[52] Iter[150/786] Loss: 2.126 CE: 1.057 Tri: 0.005 CE_rec: 1.038 AIRL_rec: 0.0539 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_airl_4090.log:837:Epoch[52] Iter[200/786] Loss: 2.125 CE: 1.056 Tri: 0.005 CE_rec: 1.038 AIRL_rec: 0.0527 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_airl_4090.log:838:Epoch[52] Iter[250/786] Loss: 2.124 CE: 1.055 Tri: 0.005 CE_rec: 1.037 AIRL_rec: 0.0523 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_airl_4090.log:839:Epoch[52] Iter[300/786] Loss: 2.123 CE: 1.055 Tri: 0.005 CE_rec: 1.037 AIRL_rec: 0.0522 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_airl_4090.log:840:Epoch[52] Iter[350/786] Loss: 2.121 CE: 1.054 Tri: 0.005 CE_rec: 1.036 AIRL_rec: 0.0516 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_airl_4090.log:841:Epoch[52] Iter[400/786] Loss: 2.120 CE: 1.054 Tri: 0.005 CE_rec: 1.035 AIRL_rec: 0.0514 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_airl_4090.log:842:Epoch[52] Iter[450/786] Loss: 2.119 CE: 1.053 Tri: 0.005 CE_rec: 1.035 AIRL_rec: 0.0514 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_airl_4090.log:843:Epoch[52] Iter[500/786] Loss: 2.117 CE: 1.053 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0509 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_airl_4090.log:844:Epoch[52] Iter[550/786] Loss: 2.117 CE: 1.052 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0504 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_airl_4090.log:845:Epoch[52] Iter[600/786] Loss: 2.114 CE: 1.052 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0498 Acc: 0.996 LR: 2.72e-06
logs/agreidv2_airl_4090.log:846:Epoch[52] Iter[650/786] Loss: 2.113 CE: 1.051 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0493 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_airl_4090.log:847:Epoch[52] Iter[700/786] Loss: 2.111 CE: 1.050 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0489 Acc: 0.997 LR: 2.72e-06
logs/agreidv2_airl_4090.log:848:Epoch[52] done in 113.9s  Loss=2.108 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.031 consistency=0.0482 deg_scale_mean=0.625 n_ground=28778]
logs/agreidv2_airl_4090.log:849:Epoch[53] Iter[50/786] Loss: 2.130 CE: 1.060 Tri: 0.005 CE_rec: 1.040 AIRL_rec: 0.0497 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_airl_4090.log:850:Epoch[53] Iter[100/786] Loss: 2.136 CE: 1.062 Tri: 0.007 CE_rec: 1.042 AIRL_rec: 0.0511 Acc: 0.992 LR: 2.16e-06
logs/agreidv2_airl_4090.log:851:Epoch[53] Iter[150/786] Loss: 2.133 CE: 1.059 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0515 Acc: 0.993 LR: 2.16e-06
logs/agreidv2_airl_4090.log:852:Epoch[53] Iter[200/786] Loss: 2.127 CE: 1.058 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0508 Acc: 0.994 LR: 2.16e-06
logs/agreidv2_airl_4090.log:853:Epoch[53] Iter[250/786] Loss: 2.124 CE: 1.056 Tri: 0.005 CE_rec: 1.037 AIRL_rec: 0.0514 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_airl_4090.log:854:Epoch[53] Iter[300/786] Loss: 2.123 CE: 1.055 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0512 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_airl_4090.log:855:Epoch[53] Iter[350/786] Loss: 2.123 CE: 1.055 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0509 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_airl_4090.log:856:Epoch[53] Iter[400/786] Loss: 2.121 CE: 1.054 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0510 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_airl_4090.log:857:Epoch[53] Iter[450/786] Loss: 2.119 CE: 1.053 Tri: 0.006 CE_rec: 1.035 AIRL_rec: 0.0505 Acc: 0.995 LR: 2.16e-06
logs/agreidv2_airl_4090.log:858:Epoch[53] Iter[500/786] Loss: 2.118 CE: 1.053 Tri: 0.006 CE_rec: 1.034 AIRL_rec: 0.0501 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_airl_4090.log:859:Epoch[53] Iter[550/786] Loss: 2.116 CE: 1.052 Tri: 0.006 CE_rec: 1.033 AIRL_rec: 0.0499 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_airl_4090.log:860:Epoch[53] Iter[600/786] Loss: 2.114 CE: 1.051 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0496 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_airl_4090.log:861:Epoch[53] Iter[650/786] Loss: 2.112 CE: 1.050 Tri: 0.005 CE_rec: 1.032 AIRL_rec: 0.0491 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_airl_4090.log:862:Epoch[53] Iter[700/786] Loss: 2.110 CE: 1.049 Tri: 0.005 CE_rec: 1.031 AIRL_rec: 0.0484 Acc: 0.996 LR: 2.16e-06
logs/agreidv2_airl_4090.log:863:Epoch[53] done in 114.0s  Loss=2.107 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.030 consistency=0.0476 deg_scale_mean=0.624 n_ground=28880]
logs/agreidv2_airl_4090.log:864:Epoch[54] Iter[50/786] Loss: 2.131 CE: 1.059 Tri: 0.005 CE_rec: 1.041 AIRL_rec: 0.0535 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_airl_4090.log:865:Epoch[54] Iter[100/786] Loss: 2.128 CE: 1.057 Tri: 0.005 CE_rec: 1.040 AIRL_rec: 0.0522 Acc: 0.995 LR: 1.67e-06
logs/agreidv2_airl_4090.log:866:Epoch[54] Iter[150/786] Loss: 2.126 CE: 1.056 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0521 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_airl_4090.log:867:Epoch[54] Iter[200/786] Loss: 2.122 CE: 1.054 Tri: 0.005 CE_rec: 1.036 AIRL_rec: 0.0529 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_airl_4090.log:868:Epoch[54] Iter[250/786] Loss: 2.122 CE: 1.054 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0530 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_airl_4090.log:869:Epoch[54] Iter[300/786] Loss: 2.120 CE: 1.054 Tri: 0.005 CE_rec: 1.035 AIRL_rec: 0.0526 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_airl_4090.log:870:Epoch[54] Iter[350/786] Loss: 2.119 CE: 1.053 Tri: 0.005 CE_rec: 1.035 AIRL_rec: 0.0522 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_airl_4090.log:871:Epoch[54] Iter[400/786] Loss: 2.118 CE: 1.052 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0517 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_airl_4090.log:872:Epoch[54] Iter[450/786] Loss: 2.116 CE: 1.052 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0514 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_airl_4090.log:873:Epoch[54] Iter[500/786] Loss: 2.114 CE: 1.051 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0506 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_airl_4090.log:874:Epoch[54] Iter[550/786] Loss: 2.114 CE: 1.051 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0502 Acc: 0.996 LR: 1.67e-06
logs/agreidv2_airl_4090.log:875:Epoch[54] Iter[600/786] Loss: 2.113 CE: 1.050 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0498 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_airl_4090.log:876:Epoch[54] Iter[650/786] Loss: 2.111 CE: 1.049 Tri: 0.005 CE_rec: 1.032 AIRL_rec: 0.0492 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_airl_4090.log:877:Epoch[54] Iter[700/786] Loss: 2.108 CE: 1.048 Tri: 0.005 CE_rec: 1.031 AIRL_rec: 0.0486 Acc: 0.997 LR: 1.67e-06
logs/agreidv2_airl_4090.log:878:Epoch[54] done in 113.7s  Loss=2.105 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.030 consistency=0.0479 deg_scale_mean=0.625 n_ground=28800]
logs/agreidv2_airl_4090.log:879:Epoch[55] Iter[50/786] Loss: 2.140 CE: 1.062 Tri: 0.008 CE_rec: 1.043 AIRL_rec: 0.0523 Acc: 0.993 LR: 1.23e-06
logs/agreidv2_airl_4090.log:880:Epoch[55] Iter[100/786] Loss: 2.128 CE: 1.058 Tri: 0.005 CE_rec: 1.039 AIRL_rec: 0.0525 Acc: 0.995 LR: 1.23e-06
logs/agreidv2_airl_4090.log:881:Epoch[55] Iter[150/786] Loss: 2.123 CE: 1.056 Tri: 0.005 CE_rec: 1.037 AIRL_rec: 0.0513 Acc: 0.995 LR: 1.23e-06
logs/agreidv2_airl_4090.log:882:Epoch[55] Iter[200/786] Loss: 2.122 CE: 1.055 Tri: 0.005 CE_rec: 1.036 AIRL_rec: 0.0513 Acc: 0.995 LR: 1.23e-06
logs/agreidv2_airl_4090.log:883:Epoch[55] Iter[250/786] Loss: 2.120 CE: 1.054 Tri: 0.005 CE_rec: 1.035 AIRL_rec: 0.0514 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_airl_4090.log:884:Epoch[55] Iter[300/786] Loss: 2.118 CE: 1.053 Tri: 0.004 CE_rec: 1.035 AIRL_rec: 0.0514 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_airl_4090.log:885:Epoch[55] Iter[350/786] Loss: 2.116 CE: 1.052 Tri: 0.004 CE_rec: 1.034 AIRL_rec: 0.0508 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_airl_4090.log:886:Epoch[55] Iter[400/786] Loss: 2.115 CE: 1.052 Tri: 0.004 CE_rec: 1.034 AIRL_rec: 0.0505 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_airl_4090.log:887:Epoch[55] Iter[450/786] Loss: 2.114 CE: 1.052 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0502 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_airl_4090.log:888:Epoch[55] Iter[500/786] Loss: 2.114 CE: 1.051 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0500 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_airl_4090.log:889:Epoch[55] Iter[550/786] Loss: 2.113 CE: 1.051 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0495 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_airl_4090.log:890:Epoch[55] Iter[600/786] Loss: 2.110 CE: 1.050 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0489 Acc: 0.996 LR: 1.23e-06
logs/agreidv2_airl_4090.log:891:Epoch[55] Iter[650/786] Loss: 2.108 CE: 1.049 Tri: 0.004 CE_rec: 1.031 AIRL_rec: 0.0484 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_airl_4090.log:892:Epoch[55] Iter[700/786] Loss: 2.107 CE: 1.048 Tri: 0.004 CE_rec: 1.030 AIRL_rec: 0.0480 Acc: 0.997 LR: 1.23e-06
logs/agreidv2_airl_4090.log:893:Epoch[55] done in 113.7s  Loss=2.103 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.029 consistency=0.0472 deg_scale_mean=0.626 n_ground=28809]
logs/agreidv2_airl_4090.log:894:Epoch[56] Iter[50/786] Loss: 2.119 CE: 1.056 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0511 Acc: 0.994 LR: 8.57e-07
logs/agreidv2_airl_4090.log:895:Epoch[56] Iter[100/786] Loss: 2.118 CE: 1.054 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0506 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_airl_4090.log:896:Epoch[56] Iter[150/786] Loss: 2.117 CE: 1.054 Tri: 0.004 CE_rec: 1.034 AIRL_rec: 0.0504 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_airl_4090.log:897:Epoch[56] Iter[200/786] Loss: 2.116 CE: 1.053 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0508 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_airl_4090.log:898:Epoch[56] Iter[250/786] Loss: 2.114 CE: 1.053 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0505 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_airl_4090.log:899:Epoch[56] Iter[300/786] Loss: 2.113 CE: 1.052 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0502 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_airl_4090.log:900:Epoch[56] Iter[350/786] Loss: 2.112 CE: 1.051 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0499 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_airl_4090.log:901:Epoch[56] Iter[400/786] Loss: 2.112 CE: 1.051 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0495 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_airl_4090.log:902:Epoch[56] Iter[450/786] Loss: 2.111 CE: 1.051 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0493 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_airl_4090.log:903:Epoch[56] Iter[500/786] Loss: 2.111 CE: 1.050 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0491 Acc: 0.996 LR: 8.57e-07
logs/agreidv2_airl_4090.log:904:Epoch[56] Iter[550/786] Loss: 2.109 CE: 1.049 Tri: 0.004 CE_rec: 1.031 AIRL_rec: 0.0486 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_airl_4090.log:905:Epoch[56] Iter[600/786] Loss: 2.108 CE: 1.049 Tri: 0.004 CE_rec: 1.031 AIRL_rec: 0.0482 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_airl_4090.log:906:Epoch[56] Iter[650/786] Loss: 2.106 CE: 1.048 Tri: 0.004 CE_rec: 1.030 AIRL_rec: 0.0477 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_airl_4090.log:907:Epoch[56] Iter[700/786] Loss: 2.104 CE: 1.047 Tri: 0.004 CE_rec: 1.030 AIRL_rec: 0.0473 Acc: 0.997 LR: 8.57e-07
logs/agreidv2_airl_4090.log:908:Epoch[56] done in 114.1s  Loss=2.101 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0466 deg_scale_mean=0.625 n_ground=28838]
logs/agreidv2_airl_4090.log:909:Epoch[57] Iter[50/786] Loss: 2.123 CE: 1.055 Tri: 0.005 CE_rec: 1.039 AIRL_rec: 0.0492 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_airl_4090.log:910:Epoch[57] Iter[100/786] Loss: 2.122 CE: 1.054 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0507 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_airl_4090.log:911:Epoch[57] Iter[150/786] Loss: 2.120 CE: 1.052 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0510 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_airl_4090.log:912:Epoch[57] Iter[200/786] Loss: 2.119 CE: 1.052 Tri: 0.007 CE_rec: 1.035 AIRL_rec: 0.0506 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_airl_4090.log:913:Epoch[57] Iter[250/786] Loss: 2.120 CE: 1.052 Tri: 0.008 CE_rec: 1.035 AIRL_rec: 0.0505 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_airl_4090.log:914:Epoch[57] Iter[300/786] Loss: 2.119 CE: 1.052 Tri: 0.007 CE_rec: 1.035 AIRL_rec: 0.0505 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_airl_4090.log:915:Epoch[57] Iter[350/786] Loss: 2.117 CE: 1.052 Tri: 0.006 CE_rec: 1.035 AIRL_rec: 0.0504 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_airl_4090.log:916:Epoch[57] Iter[400/786] Loss: 2.115 CE: 1.051 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0502 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_airl_4090.log:917:Epoch[57] Iter[450/786] Loss: 2.112 CE: 1.050 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0497 Acc: 0.996 LR: 5.50e-07
logs/agreidv2_airl_4090.log:918:Epoch[57] Iter[500/786] Loss: 2.110 CE: 1.049 Tri: 0.005 CE_rec: 1.032 AIRL_rec: 0.0491 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_airl_4090.log:919:Epoch[57] Iter[550/786] Loss: 2.109 CE: 1.049 Tri: 0.005 CE_rec: 1.031 AIRL_rec: 0.0488 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_airl_4090.log:920:Epoch[57] Iter[600/786] Loss: 2.108 CE: 1.048 Tri: 0.005 CE_rec: 1.031 AIRL_rec: 0.0485 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_airl_4090.log:921:Epoch[57] Iter[650/786] Loss: 2.107 CE: 1.048 Tri: 0.005 CE_rec: 1.030 AIRL_rec: 0.0482 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_airl_4090.log:922:Epoch[57] Iter[700/786] Loss: 2.104 CE: 1.047 Tri: 0.005 CE_rec: 1.029 AIRL_rec: 0.0475 Acc: 0.997 LR: 5.50e-07
logs/agreidv2_airl_4090.log:923:Epoch[57] done in 113.5s  Loss=2.101 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0468 deg_scale_mean=0.623 n_ground=28746]
logs/agreidv2_airl_4090.log:924:Epoch[58] Iter[50/786] Loss: 2.128 CE: 1.058 Tri: 0.003 CE_rec: 1.042 AIRL_rec: 0.0516 Acc: 0.992 LR: 3.10e-07
logs/agreidv2_airl_4090.log:925:Epoch[58] Iter[100/786] Loss: 2.128 CE: 1.057 Tri: 0.004 CE_rec: 1.041 AIRL_rec: 0.0527 Acc: 0.994 LR: 3.10e-07
logs/agreidv2_airl_4090.log:926:Epoch[58] Iter[150/786] Loss: 2.122 CE: 1.055 Tri: 0.003 CE_rec: 1.038 AIRL_rec: 0.0522 Acc: 0.994 LR: 3.10e-07
logs/agreidv2_airl_4090.log:927:Epoch[58] Iter[200/786] Loss: 2.119 CE: 1.054 Tri: 0.004 CE_rec: 1.036 AIRL_rec: 0.0511 Acc: 0.995 LR: 3.10e-07
logs/agreidv2_airl_4090.log:928:Epoch[58] Iter[250/786] Loss: 2.118 CE: 1.053 Tri: 0.004 CE_rec: 1.036 AIRL_rec: 0.0506 Acc: 0.995 LR: 3.10e-07
logs/agreidv2_airl_4090.log:929:Epoch[58] Iter[300/786] Loss: 2.118 CE: 1.053 Tri: 0.005 CE_rec: 1.035 AIRL_rec: 0.0504 Acc: 0.995 LR: 3.10e-07
logs/agreidv2_airl_4090.log:930:Epoch[58] Iter[350/786] Loss: 2.116 CE: 1.051 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0501 Acc: 0.995 LR: 3.10e-07
logs/agreidv2_airl_4090.log:931:Epoch[58] Iter[400/786] Loss: 2.113 CE: 1.051 Tri: 0.004 CE_rec: 1.034 AIRL_rec: 0.0497 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_airl_4090.log:932:Epoch[58] Iter[450/786] Loss: 2.111 CE: 1.050 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0494 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_airl_4090.log:933:Epoch[58] Iter[500/786] Loss: 2.110 CE: 1.049 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0492 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_airl_4090.log:934:Epoch[58] Iter[550/786] Loss: 2.110 CE: 1.049 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0490 Acc: 0.996 LR: 3.10e-07
logs/agreidv2_airl_4090.log:935:Epoch[58] Iter[600/786] Loss: 2.108 CE: 1.048 Tri: 0.005 CE_rec: 1.031 AIRL_rec: 0.0486 Acc: 0.997 LR: 3.10e-07
logs/agreidv2_airl_4090.log:936:Epoch[58] Iter[650/786] Loss: 2.106 CE: 1.047 Tri: 0.005 CE_rec: 1.030 AIRL_rec: 0.0482 Acc: 0.997 LR: 3.10e-07
logs/agreidv2_airl_4090.log:937:Epoch[58] Iter[700/786] Loss: 2.104 CE: 1.046 Tri: 0.005 CE_rec: 1.029 AIRL_rec: 0.0476 Acc: 0.997 LR: 3.10e-07
logs/agreidv2_airl_4090.log:938:Epoch[58] done in 113.9s  Loss=2.101 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0469 deg_scale_mean=0.628 n_ground=28817]
logs/agreidv2_airl_4090.log:939:Epoch[59] Iter[50/786] Loss: 2.116 CE: 1.051 Tri: 0.006 CE_rec: 1.034 AIRL_rec: 0.0518 Acc: 0.998 LR: 1.38e-07
logs/agreidv2_airl_4090.log:940:Epoch[59] Iter[100/786] Loss: 2.111 CE: 1.050 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0501 Acc: 0.998 LR: 1.38e-07
logs/agreidv2_airl_4090.log:941:Epoch[59] Iter[150/786] Loss: 2.113 CE: 1.051 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0505 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_airl_4090.log:942:Epoch[59] Iter[200/786] Loss: 2.113 CE: 1.050 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0505 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_airl_4090.log:943:Epoch[59] Iter[250/786] Loss: 2.112 CE: 1.050 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0504 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_airl_4090.log:944:Epoch[59] Iter[300/786] Loss: 2.111 CE: 1.050 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0499 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_airl_4090.log:945:Epoch[59] Iter[350/786] Loss: 2.112 CE: 1.050 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0498 Acc: 0.996 LR: 1.38e-07
logs/agreidv2_airl_4090.log:946:Epoch[59] Iter[400/786] Loss: 2.111 CE: 1.049 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0498 Acc: 0.996 LR: 1.38e-07
logs/agreidv2_airl_4090.log:947:Epoch[59] Iter[450/786] Loss: 2.110 CE: 1.049 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0494 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_airl_4090.log:948:Epoch[59] Iter[500/786] Loss: 2.108 CE: 1.048 Tri: 0.004 CE_rec: 1.031 AIRL_rec: 0.0490 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_airl_4090.log:949:Epoch[59] Iter[550/786] Loss: 2.107 CE: 1.048 Tri: 0.004 CE_rec: 1.031 AIRL_rec: 0.0487 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_airl_4090.log:950:Epoch[59] Iter[600/786] Loss: 2.105 CE: 1.047 Tri: 0.004 CE_rec: 1.030 AIRL_rec: 0.0483 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_airl_4090.log:951:Epoch[59] Iter[650/786] Loss: 2.103 CE: 1.046 Tri: 0.004 CE_rec: 1.029 AIRL_rec: 0.0478 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_airl_4090.log:952:Epoch[59] Iter[700/786] Loss: 2.101 CE: 1.045 Tri: 0.004 CE_rec: 1.029 AIRL_rec: 0.0472 Acc: 0.997 LR: 1.38e-07
logs/agreidv2_airl_4090.log:953:Epoch[59] done in 113.6s  Loss=2.098 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0466 deg_scale_mean=0.624 n_ground=28774]
logs/agreidv2_airl_4090.log:954:Epoch[60] Iter[50/786] Loss: 2.112 CE: 1.049 Tri: 0.004 CE_rec: 1.034 AIRL_rec: 0.0500 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_airl_4090.log:955:Epoch[60] Iter[100/786] Loss: 2.119 CE: 1.053 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0499 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_airl_4090.log:956:Epoch[60] Iter[150/786] Loss: 2.118 CE: 1.052 Tri: 0.006 CE_rec: 1.035 AIRL_rec: 0.0501 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_airl_4090.log:957:Epoch[60] Iter[200/786] Loss: 2.115 CE: 1.051 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0497 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_airl_4090.log:958:Epoch[60] Iter[250/786] Loss: 2.113 CE: 1.050 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0495 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_airl_4090.log:959:Epoch[60] Iter[300/786] Loss: 2.112 CE: 1.050 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0494 Acc: 0.996 LR: 3.45e-08
logs/agreidv2_airl_4090.log:960:Epoch[60] Iter[350/786] Loss: 2.109 CE: 1.049 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0491 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_airl_4090.log:961:Epoch[60] Iter[400/786] Loss: 2.108 CE: 1.049 Tri: 0.003 CE_rec: 1.032 AIRL_rec: 0.0487 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_airl_4090.log:962:Epoch[60] Iter[450/786] Loss: 2.108 CE: 1.048 Tri: 0.003 CE_rec: 1.032 AIRL_rec: 0.0484 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_airl_4090.log:963:Epoch[60] Iter[500/786] Loss: 2.106 CE: 1.048 Tri: 0.003 CE_rec: 1.031 AIRL_rec: 0.0482 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_airl_4090.log:964:Epoch[60] Iter[550/786] Loss: 2.105 CE: 1.047 Tri: 0.003 CE_rec: 1.030 AIRL_rec: 0.0479 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_airl_4090.log:965:Epoch[60] Iter[600/786] Loss: 2.104 CE: 1.046 Tri: 0.004 CE_rec: 1.030 AIRL_rec: 0.0474 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_airl_4090.log:966:Epoch[60] Iter[650/786] Loss: 2.102 CE: 1.046 Tri: 0.004 CE_rec: 1.029 AIRL_rec: 0.0469 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_airl_4090.log:967:Epoch[60] Iter[700/786] Loss: 2.100 CE: 1.045 Tri: 0.004 CE_rec: 1.028 AIRL_rec: 0.0465 Acc: 0.997 LR: 3.45e-08
logs/agreidv2_airl_4090.log:968:Epoch[60] done in 114.0s  Loss=2.097 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.027 consistency=0.0458 deg_scale_mean=0.624 n_ground=28813]
logs/agreidv2_airl_4090.log:970:    [A->G] mAP=79.47  R1=87.18  R5=92.57  mINP=54.19
logs/agreidv2_airl_4090.log:971:    [G->A] mAP=80.33  R1=87.47  R5=91.83  mINP=47.95
logs/agreidv2_airl_4090.log:972:    [mean] mAP=79.90  R1=87.32
logs/agreidv2_airl_4090.log:973:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 60 ----
logs/agreidv2_airl_4090.log:974:    [A->G] full mAP=79.47 R1=87.18 | rec mAP=78.70 R1=84.93 | FUSE mAP=80.67 R1=87.90
logs/agreidv2_airl_4090.log:975:    [G->A] full mAP=80.33 R1=87.47 | rec mAP=78.38 R1=86.58 | FUSE mAP=81.29 R1=88.02
logs/agreidv2_airl_4090.log:977:    * new best mean mAP=80.98 (epoch 60) saved
logs/agreidv2_airl_4090.log:979:Training finished. Best mean A<->G mAP=80.98 @ epoch 60
logs/agreidv2_airl_4090.log:980:Checkpoints in /home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090

codex
我已经定位到 AIRL 的核心路径：Swin split/iso 分支在 `afd_model.py`，训练 AMP 与 dual eval 在 `afd_train.py`。接下来我会把这些块按行号读完，再用命令从日志里抽取每次 eval 与最终 best。
exec
/bin/zsh -lc "nl -ba code/afd_model.py | sed -n '1,230p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
     1	# encoding: utf-8
     2	"""
     3	AFD-ReID model: ResNet50 backbone + GeM/avg pool + BNNeck + ID classifier.
     4	
     5	Baseline (use_afd=False) is a standard strong ReID baseline (Luo et al. BoT):
     6	    resnet50(IMAGENET1K_V1) -> last_stride=1 -> pool -> BNNeck -> linear classifier
     7	    train returns (global_feat, logits); test returns L2-normalized BN feature.
     8	
     9	AFD modules (use_afd=True) -- altitude-frequency decoupling, both switchable:
    10	  * Frequency Reliability Router (afd_router):
    11	      split a shallow feature map into low / mid / high frequency bands via 2D FFT,
    12	      predict per-band reliability weights with a light gate (optionally conditioned
    13	      on view/altitude), and recombine the bands. The recombined feature replaces the
    14	      input to the rest of the backbone. Designed to *down-weight unreliable high-freq*
    15	      content under aerial (low-resolution) views.
    16	  * Cross-View Frequency Counterfactual (afd_cvfc):
    17	      training-time regularizer (no test-time cost). Two switchable sub-mechanisms:
    18	        - high-band dropout: zero the high-freq band of the shallow feature with prob p,
    19	          producing a counterfactual view that must keep the same identity.
    20	        - low/high consistency: penalize the distance between the embedding of the full
    21	          feature and the embedding of its low-pass counterfactual, so identity relies
    22	          on the stable low/mid band shared across A<->G.
    23	      The CVFC forward exposes counterfactual features; the losses live in afd_train.py
    24	      so the mechanism stays cleanly ablatable.
    25	
    26	Every AFD piece is gated by flags so `use_afd=False` reproduces the baseline exactly
    27	(the AFD submodules are not even constructed when use_afd=False).
    28	"""
    29	import copy
    30	import math
    31	import os
    32	import sys
    33	import types
    34	
    35	import torch
    36	import torch.nn as nn
    37	import torch.nn.functional as F
    38	import torchvision
    39	
    40	
    41	# --------------------------------------------------------------------------- #
    42	# Swin-Small backbone (SOLIDER) -- optional, team asset, for SOTA push.
    43	# Lazy-built ONLY when backbone='swin_small'; the resnet50 path never touches
    44	# any of this so it stays byte-for-byte identical.
    45	# --------------------------------------------------------------------------- #
    46	# Repo root = .../SOLIDER-REID (this file lives at experiments/afd_reid/).
    47	_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    48	                                          '..', '..'))
    49	
    50	
    51	def _ensure_mmcv_stub():
    52	    """The SOLIDER swin_transformer.py does, at import time,
    53	        try:    from mmcv.runner import load_checkpoint as _load_checkpoint
    54	        except ImportError: from mmengine.runner import load_checkpoint ...
    55	    but `_load_checkpoint` is never actually called (init_weights uses
    56	    torch.load directly).  The default python on the training boxes has neither
    57	    mmcv nor mmengine, so the bare `import` would crash.  Register lightweight
    58	    stub modules (with a dummy `load_checkpoint`) so the import line succeeds.
    59	    Real mmcv/mmengine, if present, are left untouched.  Called ONLY from the
    60	    swin branch -> resnet50 import path is unaffected.
    61	    """
    62	    for pkg, sub in (('mmcv', 'mmcv.runner'), ('mmengine', 'mmengine.runner')):
    63	        try:
    64	            __import__(sub)
    65	            continue  # the real package is importable -> do not stub it
    66	        except Exception:
    67	            pass
    68	        if pkg not in sys.modules:
    69	            sys.modules[pkg] = types.ModuleType(pkg)
    70	        if sub not in sys.modules:
    71	            sys.modules[sub] = types.ModuleType(sub)
    72	        # `from X.runner import load_checkpoint` needs the attribute to exist.
    73	        sys.modules[sub].load_checkpoint = lambda *a, **k: None
    74	        setattr(sys.modules[pkg], 'runner', sys.modules[sub])
    75	
    76	
    77	class SwinBackboneReID(nn.Module):
    78	    """Thin wrapper around the SOLIDER swin_small backbone for the AFD/OVLI
    79	    trainer.
    80	
    81	    Exposes the SAME hook contract as the resnet50 AFDModel:
    82	      * `self.layer4` is an ``nn.Identity`` whose forward is fed the last-stage
    83	        spatial map, so an OVLI ``model.layer4.register_forward_hook`` captures a
    84	        ``(B, C, H, W)`` map WITHOUT detaching (gradient flows backbone->proj).
    85	      * forward returns ``(feat_map, None)`` where feat_map is that same NCHW map
    86	        (AFDModel pools it -> BNNeck), mirroring resnet's ``_forward_backbone``.
    87	
    88	    swin_small: embed_dims=96, depths=(2,2,18,2), num_features=[96,192,384,768];
    89	    last-stage channel = 768 (set as the model's in_planes).  For a 256x128 input
    90	    the last map is (B,768,8,4); for 384x128 it is (B,768,12,4).
    91	    """
    92	
    93	    OUT_DIM = 768
    94	
    95	    def __init__(self, img_size=(256, 128), pretrain_path='', semantic_weight=0.2,
    96	                 drop_path_rate=0.1, iso_branch=False, iso_stage=3,
    97	                 iso_trunk_recce=True):
    98	        super().__init__()
    99	        _ensure_mmcv_stub()
   100	        if _REPO_ROOT not in sys.path:
   101	            sys.path.insert(0, _REPO_ROOT)
   102	        # import AFTER the stub is registered and repo root is on sys.path
   103	        from model.backbones.swin_transformer import swin_small_patch4_window7_224
   104	
   105	        # img_size is (H, W); the factory takes it through to PatchEmbed.
   106	        self.swin = swin_small_patch4_window7_224(
   107	            img_size=list(img_size),
   108	            drop_path_rate=drop_path_rate,
   109	            drop_rate=0.0,
   110	            attn_drop_rate=0.0,
   111	            semantic_weight=semantic_weight,   # SOLIDER ReID default 0.2
   112	            convert_weights=False,             # teacher ckpt is already in-repo layout
   113	        )
   114	        self.out_dim = self.swin.num_features[-1]   # 768 for swin_small
   115	        if pretrain_path:
   116	            # loads the SOLIDER 'teacher' checkpoint (backbone.* keys), strict=False
   117	            self.swin.init_weights(pretrain_path)
   118	        else:
   119	            self.swin.init_weights(None)            # trunc-normal from scratch
   120	        # Identity hook point so OVLI's model.layer4 forward-hook gets the NCHW map.
   121	        self.layer4 = nn.Identity()
   122	
   123	        # ---- AIRL gradient-isolated dual-branch (f_rec independent late stage) --
   124	        # iso_branch=True forks a SECOND last-stage path (f_rec) off the shared
   125	        # residual stream at the input of stage `iso_stage`.  The rec path is an
   126	        # INDEPENDENT deep-copy of swin.stages[iso_stage:] (+ that stage's output
   127	        # norm).  Two gradient regimes on the fork-point feed, governed by
   128	        # iso_trunk_recce (the trunk-undersupervision FIX):
   129	        #
   130	        #   * DEGRADED (rec_only=True, the consistency pass): the fork feed is ALWAYS
   131	        #     detach()ed -> the AIRL degradation-consistency gradient updates ONLY the
   132	        #     rec copy + BNNeck_rec and NEVER reaches the shared trunk.  This is the
   133	        #     isolation invariant that keeps f_rec a specialised "recover expert" and
   134	        #     protects the clean trunk + f_full from being pulled toward degradation-
   135	        #     robustness.  Holds for BOTH settings of iso_trunk_recce.
   136	        #
   137	        #   * CLEAN (rec_only=False, the main forward):
   138	        #       - iso_trunk_recce=True  (default, the FIX): the fork feed is NOT
   139	        #         detached, so f_rec's CLEAN ID-CE gradient FLOWS BACK into the shared
   140	        #         trunk.  Diagnosis (codex consensus): the original full-detach iso cut
   141	        #         the trunk's extra identity supervision (f_rec's clean ID-CE only
   142	        #         updated the detached rec tail), leaving f_full WEAKER than even the
   143	        #         fully-shared dual-branch (whose trunk saw both heads' ID-CE).
   144	        #         Re-routing ONLY the clean ID-CE to the trunk restores that extra
   145	        #         identity supervision -> strengthens f_full, while the degradation-
   146	        #         consistency stays detached (above) -> f_rec stays specialised.
   147	        #       - iso_trunk_recce=False (ablation): the clean fork feed is ALSO
   148	        #         detached -> the ORIGINAL full-isolation iso (clean ID-CE + consistency
   149	        #         both severed from the trunk).  Kept for the controlled comparison
   150	        #         "does the clean-ID-CE trunk reflow help, or just any change?".
   151	        #
   152	        # OFF (iso_branch=False) -> nothing is built and the forward is byte-for-byte
   153	        # the single-map baseline.
   154	        self.iso_branch = bool(iso_branch)
   155	        self.iso_stage = int(iso_stage)
   156	        # iso_trunk_recce: whether the CLEAN rec ID-CE gradient reflows into the
   157	        # shared trunk (True, the fix) or the clean fork feed is also detached
   158	        # (False, original full-isolation ablation).  No effect when iso_branch off.
   159	        self.iso_trunk_recce = bool(iso_trunk_recce)
   160	        if self.iso_branch:
   161	            n_stages = len(self.swin.stages)
   162	            if not (1 <= self.iso_stage <= n_stages - 1):
   163	                raise ValueError(
   164	                    f"iso_stage must be in [1, {n_stages - 1}] (fork after the "
   165	                    f"shared early stages, before the last); got {self.iso_stage}")
   166	            # the rec branch re-runs stages [iso_stage .. last] on its OWN copy.
   167	            # deep-copy preserves the pretrained weights as the f_rec init (same
   168	            # starting point as f_full's stages -> divergence comes from training,
   169	            # not from a random re-init that would cripple f_rec).
   170	            self.rec_stages = nn.ModuleList(
   171	                copy.deepcopy(self.swin.stages[i]) for i in range(self.iso_stage,
   172	                                                                  n_stages))
   173	            # the last output norm (norm{last}) applied to the rec last-stage map,
   174	            # an independent copy so f_rec gets its own LayerNorm (matches the
   175	            # f_full norm recipe; reshaped exactly like swin.forward does).
   176	            last = n_stages - 1
   177	            self.rec_norm = copy.deepcopy(getattr(self.swin, f'norm{last}'))
   178	            # independent copies of the semantic-embed Linears for the rec stages
   179	            # (frozen, requires_grad=False -- same as the trunk's; deep-copy keeps
   180	            # the same frozen weights so the rec stream is modulated identically to
   181	            # the trunk at init).  swin keeps one (w,b) pair PER stage index i; the
   182	            # rec branch runs stages [iso_stage..last] so it needs those indices.
   183	            if self.swin.semantic_weight >= 0:
   184	                self.rec_semantic_embed_w = nn.ModuleList(
   185	                    copy.deepcopy(self.swin.semantic_embed_w[i])
   186	                    for i in range(self.iso_stage, n_stages))
   187	                self.rec_semantic_embed_b = nn.ModuleList(
   188	                    copy.deepcopy(self.swin.semantic_embed_b[i])
   189	                    for i in range(self.iso_stage, n_stages))
   190	                # the deep-copied Linears already carry requires_grad=False (the
   191	                # trunk froze them); re-assert defensively so the rec semantic embed
   192	                # is never trained even if a future deepcopy reset the flag.
   193	                for p in self.rec_semantic_embed_w.parameters():
   194	                    p.requires_grad = False
   195	                for p in self.rec_semantic_embed_b.parameters():
   196	                    p.requires_grad = False
   197	            # Identity hook point for the rec last-stage map (mirrors self.layer4);
   198	            # kept for parity / future hooks -- the rec map is a fresh path so OVLI's
   199	            # single layer4 hook (on the f_full map) is unaffected.
   200	            self.layer4_rec = nn.Identity()
   201	
   202	    def _run_rec_stages(self, x, hw_shape, semantic_weight):
   203	        """Run the INDEPENDENT rec copy of stages [iso_stage..last] on the residual
   204	        stream `x` (the fork feed) and return the rec last-stage NCHW map.
   205	
   206	        The caller (_forward_swin_split) decides whether `x` is detached: the DEGRADED
   207	        consistency pass always passes a detached fork (gradient isolation), while the
   208	        CLEAN pass with iso_trunk_recce=True passes a NON-detached fork so the clean
   209	        f_rec ID-CE reflows into the shared trunk.  This method itself is agnostic to
   210	        that choice -- it just runs the rec stages over whatever `x` it is given.
   211	
   212	        Replicates SwinTransformer.forward's per-stage body EXACTLY (stage -> per-
   213	        stage semantic-embed on the continuing stream -> final-stage norm + reshape)
   214	        but over self.rec_stages / self.rec_norm / self.rec_semantic_embed_*, so the
   215	        rec map is computed the same way f_full's map is -- the ONLY differences are
   216	        (a) independent weights and (b) the fork input (detached or not per above).
   217	        """
   218	        n_stages = len(self.swin.stages)
   219	        last = n_stages - 1
   220	        rec_out = None
   221	        for j, stage in enumerate(self.rec_stages):
   222	            i = self.iso_stage + j               # absolute stage index
   223	            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
   224	            if self.swin.semantic_weight >= 0:
   225	                sw = self.rec_semantic_embed_w[j](semantic_weight).unsqueeze(1)
   226	                sb = self.rec_semantic_embed_b[j](semantic_weight).unsqueeze(1)
   227	                x = x * self.swin.softplus(sw) + sb
   228	            if i == last:
   229	                out = self.rec_norm(out)
   230	                out = out.view(-1, *out_hw_shape,

exec
/bin/zsh -lc "nl -ba code/afd_model.py | sed -n '850,898p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   850	            # continue from the stage AFTER the insertion point
   851	            if stage == 'stem':
   852	                h = self.layer1(feat); h = self.layer2(h)
   853	            elif stage == 'layer1':
   854	                h = self.layer2(feat)
   855	            else:  # layer2
   856	                h = feat
   857	            h = self.layer3(h); h = self.layer4(h)
   858	            _, bn = self._embed(h)
   859	            return bn
   860	
   861	        hd = self.cvfc.high_band_dropout(shallow)
   862	        lp = self.cvfc.low_pass(shallow)
   863	        return {
   864	            'cf_highdrop_bn': finish(hd),
   865	            'cf_lowpass_bn': finish(lp),
   866	        }
   867	
   868	
   869	def build_model(num_classes, args):
   870	    """Factory from an argparse-like namespace.
   871	
   872	    backbone defaults to 'resnet50' -> the existing BoT baseline is reproduced
   873	    byte-for-byte (no new arg required of legacy callers).  backbone='swin_small'
   874	    builds the SOLIDER Swin-Small backbone instead (img_size / swin_pretrain /
   875	    swin_semantic_weight are read from args, with the SOLIDER ReID defaults).
   876	    """
   877	    return AFDModel(
   878	        num_classes=num_classes,
   879	        last_stride=getattr(args, 'last_stride', 1),
   880	        pool=getattr(args, 'pool', 'gem'),
   881	        pretrained=True,
   882	        use_afd=getattr(args, 'use_afd', False),
   883	        afd_router=getattr(args, 'afd_router', True),
   884	        afd_cvfc=getattr(args, 'afd_cvfc', True),
   885	        afd_stage=getattr(args, 'afd_stage', 'layer1'),
   886	        router_cond_view=getattr(args, 'router_cond_view', True),
   887	        low_r=getattr(args, 'low_r', 0.125),
   888	        mid_r=getattr(args, 'mid_r', 0.30),
   889	        high_drop_p=getattr(args, 'high_drop_p', 0.5),
   890	        backbone=getattr(args, 'backbone', 'resnet50'),
   891	        swin_pretrain=getattr(args, 'swin_pretrain', ''),
   892	        swin_semantic_weight=getattr(args, 'swin_semantic_weight', 0.2),
   893	        img_size=tuple(getattr(args, 'img_size', (256, 128))),
   894	        airl_dualbranch=getattr(args, 'airl_dualbranch', False),
   895	        airl_dualbranch_iso=getattr(args, 'airl_dualbranch_iso', False),
   896	        airl_iso_stage=getattr(args, 'airl_iso_stage', 3),
   897	        airl_iso_trunk_recce=getattr(args, 'airl_iso_trunk_recce', True),
   898	    )

exec
/bin/zsh -lc "nl -ba code/afd_model.py | sed -n '230,360p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
exec
/bin/zsh -lc "nl -ba code/afd_model.py | sed -n '528,850p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   230	                out = out.view(-1, *out_hw_shape,
   231	                               self.swin.num_features[i]).permute(
   232	                                   0, 3, 1, 2).contiguous()
   233	                rec_out = out
   234	        return self.layer4_rec(rec_out)
   235	
   236	    def _forward_swin_split(self, x, rec_only=False):
   237	        """Replicate SwinTransformer.forward but ALSO branch the rec path.
   238	
   239	        Returns (f_full_map, f_rec_map).  The shared patch_embed + ALL f_full stages
   240	        run FIRST, exactly and in the same order as swin.forward (so even the
   241	        training-time stochastic-depth / DropPath RNG sequence f_full sees is
   242	        identical to the single-branch path -- the rec copy runs AFTER the full loop,
   243	        not interleaved, so it cannot perturb f_full's RNG draws); the residual
   244	        stream at the input of stage `iso_stage` is captured and fed through the
   245	        independent rec stages afterward.  semantic_weight is built identically to
   246	        swin.forward.
   247	
   248	        Gradient regime on the rec fork feed (the trunk-undersupervision FIX):
   249	          * rec_only=True (degraded consistency pass): the fork feed is ALWAYS
   250	            DETACHED -> the consistency gradient cannot reach the shared trunk
   251	            (the isolation invariant, independent of iso_trunk_recce).
   252	          * rec_only=False (clean main pass): the fork feed is detached ONLY when
   253	            self.iso_trunk_recce is False (original full-isolation ablation).  When
   254	            iso_trunk_recce is True (the fix, default) the clean fork feed is NOT
   255	            detached, so f_rec's CLEAN ID-CE gradient reflows into the shared trunk
   256	            (extra identity supervision that strengthens f_full).  The
   257	            degradation-consistency still uses the rec_only=True detached path, so
   258	            it never reaches the trunk regardless of this setting.
   259	
   260	        rec_only=True: skip the f_full BNNeck-side work entirely is done by the
   261	        CALLER (it just ignores full_map); here rec_only additionally lets the
   262	        degraded consistency pass avoid keeping the f_full map's grad graph -- we
   263	        still must run the shared stages to REACH the fork point, but we do NOT need
   264	        f_full's last-stage norm/grad, so full_map is returned detached to make the
   265	        "f_full untouched by the degraded pass" contract explicit and cheap.
   266	        """
   267	        swin = self.swin
   268	        # build the per-sample semantic weight exactly like SwinTransformer.forward
   269	        if swin.semantic_weight >= 0:
   270	            w = torch.ones(x.shape[0], 1) * swin.semantic_weight
   271	            w = torch.cat([w, 1 - w], axis=-1)
   272	            semantic_weight = w.to(x.device)
   273	        else:
   274	            semantic_weight = None
   275	
   276	        x, hw_shape = swin.patch_embed(x)
   277	        if swin.use_abs_pos_embed:
   278	            x = x + swin.absolute_pos_embed
   279	        x = swin.drop_after_pos(x)
   280	
   281	        # Whether the rec fork feed is detached from the shared trunk:
   282	        #   * degraded consistency pass (rec_only) -> ALWAYS detach (isolation
   283	        #     invariant: consistency grad never reaches the trunk).
   284	        #   * clean pass -> detach ONLY when iso_trunk_recce is False (original
   285	        #     full-isolation ablation); when True (the fix) keep the graph so the
   286	        #     clean rec ID-CE reflows into the trunk (extra identity supervision).
   287	        detach_fork = bool(rec_only) or (not self.iso_trunk_recce)
   288	        fork_x = None
   289	        fork_hw = None
   290	        full_map = None
   291	        for i, stage in enumerate(swin.stages):
   292	            # the residual stream `x` HERE is the input to stage i.  When i ==
   293	            # iso_stage, snapshot this stream (the gradient-isolation boundary) to
   294	            # feed the rec branch AFTER the full loop.  Detach per detach_fork above:
   295	            # detached -> rec grad severed from trunk; non-detached -> clean rec
   296	            # ID-CE flows back into the trunk (the trunk-undersupervision fix).
   297	            if i == self.iso_stage:
   298	                fork_x = x.detach() if detach_fork else x
   299	                fork_hw = hw_shape
   300	            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
   301	            if swin.semantic_weight >= 0:
   302	                sw = swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
   303	                sb = swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
   304	                x = x * swin.softplus(sw) + sb
   305	            if i in swin.out_indices and i == len(swin.stages) - 1:
   306	                norm_layer = getattr(swin, f'norm{i}')
   307	                out = norm_layer(out)
   308	                out = out.view(-1, *out_hw_shape,
   309	                               swin.num_features[i]).permute(
   310	                                   0, 3, 1, 2).contiguous()
   311	                full_map = out
   312	        full_map = self.layer4(full_map)        # Identity passthrough (OVLI hook)
   313	        if rec_only:
   314	            # the degraded consistency pass only needs f_rec; detach f_full's map so
   315	            # no f_full grad graph is built and the contract "the degraded pass does
   316	            # not train f_full" is explicit.  (running stats of self.bottleneck are
   317	            # NOT updated for this pass because the caller never pools full_map -> no
   318	            # BatchNorm forward on it; see AFDModel.forward rec_only path.)
   319	            full_map = full_map.detach()
   320	        # rec branch: independent late stages on the fork stream.  fork_x is detached
   321	        # per detach_fork (degraded/ablation -> isolated; clean+fix -> grad reflows to
   322	        # trunk).  The semantic weight is a FROZEN constant (no params), so detaching
   323	        # it is harmless: it never blocks gradient through fork_x itself (the rec
   324	        # multiply x*softplus(sw)+sb keeps x's graph).  Run AFTER the f_full loop so
   325	        # f_full's RNG is unchanged.
   326	        rec_map = self._run_rec_stages(
   327	            fork_x, fork_hw,
   328	            None if semantic_weight is None else semantic_weight.detach())
   329	        return full_map, rec_map
   330	
   331	    def forward(self, x, return_rec=False, rec_only=False):
   332	        # Default path (return_rec=False OR iso off): SwinTransformer.forward ->
   333	        # (global_feat(B,768), outs[list of NCHW maps]); take the last spatial map
   334	        # and route it through self.layer4 so the OVLI hook fires (no detach -> grad
   335	        # flows).  Byte-for-byte the original single-map behaviour.
   336	        if not (self.iso_branch and return_rec):
   337	            _gfeat, outs = self.swin(x)
   338	            feat_map = self.layer4(outs[-1])        # (B,768,H,W), Identity passthrough
   339	            return feat_map
   340	        # iso dual-branch path: run the split forward -> (f_full map, f_rec map).
   341	        # The rec map is computed through independent late stages.  The DEGRADED pass
   342	        # (rec_only) forks off a DETACHED trunk so the consistency loss cannot perturb
   343	        # the shared trunk; the CLEAN pass forks off a NON-detached trunk when
   344	        # iso_trunk_recce=True (clean f_rec ID-CE reflows -> extra trunk supervision),
   345	        # else detached (full-isolation ablation).  See _forward_swin_split.
   346	        full_map, rec_map = self._forward_swin_split(x, rec_only=rec_only)
   347	        return full_map, rec_map
   348	
   349	
   350	# --------------------------------------------------------------------------- #
   351	# weight init (BoT style)
   352	# --------------------------------------------------------------------------- #
   353	def weights_init_kaiming(m):
   354	    classname = m.__class__.__name__
   355	    if classname.find('Linear') != -1:
   356	        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
   357	        if m.bias is not None:
   358	            nn.init.constant_(m.bias, 0.0)
   359	    elif classname.find('Conv') != -1:
   360	        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')

 succeeded in 0ms:
   528	class AFDModel(nn.Module):
   529	    def __init__(self, num_classes, last_stride=1, pool='gem',
   530	                 pretrained=True,
   531	                 use_afd=False,
   532	                 afd_router=True, afd_cvfc=True,
   533	                 afd_stage='layer1',
   534	                 router_cond_view=True,
   535	                 low_r=0.125, mid_r=0.30, high_drop_p=0.5,
   536	                 backbone='resnet50',
   537	                 swin_pretrain='', swin_semantic_weight=0.2,
   538	                 img_size=(256, 128),
   539	                 airl_dualbranch=False,
   540	                 airl_dualbranch_iso=False, airl_iso_stage=3,
   541	                 airl_iso_trunk_recce=True):
   542	        super().__init__()
   543	        self.backbone = backbone
   544	        self.use_afd = use_afd
   545	        # AIRL dual-branch: a SECOND BNNeck head (bottleneck_rec + classifier_rec)
   546	        # over the SAME shared backbone feature map.  f_full (the original head)
   547	        # keeps full-resolution identity evidence (protects G->A); f_rec (this
   548	        # second head) additionally carries the AIRL ground-degradation
   549	        # consistency at train time, so it learns identity evidence recoverable
   550	        # under a low (aerial) pixel budget (serves A->G).  At eval the two
   551	        # heads' cosine scores are SOFT-fused at the distance-matrix level
   552	        # (cos = w*cos_rec + (1-w)*cos_full) -- ONE forward yields both features.
   553	        # OFF (default) -> the second head is not even constructed and forward
   554	        # returns exactly the single-head dict/eval tensor (byte-for-byte base).
   555	        self.airl_dualbranch = bool(airl_dualbranch)
   556	        # AIRL gradient-isolated dual-branch: the SAME two-head + soft-fusion idea
   557	        # as airl_dualbranch, but f_rec is NOT a BNNeck over the shared global_feat;
   558	        # it is a BNNeck over an INDEPENDENT late Swin stage forked off a DETACHED
   559	        # trunk feature (see SwinBackboneReID.iso_branch).  This severs the f_rec
   560	        # consistency gradient from the shared trunk so the clean trunk + f_full are
   561	        # not pulled toward degradation-robustness -> the two heads re-diverge.
   562	        # swin-only (the fork lives in the Swin stage list); mutually exclusive with
   563	        # the shared airl_dualbranch (same eval/loss contract, different f_rec path).
   564	        self.airl_dualbranch_iso = bool(airl_dualbranch_iso)
   565	        self.airl_iso_stage = int(airl_iso_stage)
   566	        # airl_iso_trunk_recce: route the CLEAN f_rec ID-CE gradient back into the
   567	        # shared trunk (True, default = the trunk-undersupervision fix) vs the
   568	        # original full-isolation iso where the clean fork feed is also detached
   569	        # (False). The degradation-consistency stays trunk-isolated either way.
   570	        self.airl_iso_trunk_recce = bool(airl_iso_trunk_recce)
   571	        if self.airl_dualbranch_iso:
   572	            assert not self.airl_dualbranch, (
   573	                "airl_dualbranch_iso and airl_dualbranch are mutually exclusive "
   574	                "(shared vs gradient-isolated f_rec; pick one).")
   575	            assert backbone == 'swin_small', (
   576	                "airl_dualbranch_iso requires backbone='swin_small' (the rec branch "
   577	                "forks an independent Swin late stage).")
   578	        self.afd_router = use_afd and afd_router
   579	        self.afd_cvfc = use_afd and afd_cvfc
   580	        self.afd_stage = afd_stage
   581	
   582	        if backbone == 'resnet50':
   583	            self.in_planes = 2048
   584	
   585	            weights = 'IMAGENET1K_V1' if pretrained else None
   586	            resnet = torchvision.models.resnet50(weights=weights)
   587	            # ReID standard: last block stride 1 -> larger spatial map (16x8 for 256x128)
   588	            if last_stride == 1:
   589	                resnet.layer4[0].conv2.stride = (1, 1)
   590	                resnet.layer4[0].downsample[0].stride = (1, 1)
   591	
   592	            # split backbone so the router can be inserted after a shallow stage
   593	            self.stem = nn.Sequential(
   594	                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
   595	            self.layer1 = resnet.layer1   # 256 ch
   596	            self.layer2 = resnet.layer2   # 512 ch
   597	            self.layer3 = resnet.layer3   # 1024 ch
   598	            self.layer4 = resnet.layer4   # 2048 ch
   599	
   600	            # channel count at the chosen insertion stage
   601	            stage_ch = {'stem': 64, 'layer1': 256, 'layer2': 512}
   602	            assert afd_stage in stage_ch, f"afd_stage must be one of {list(stage_ch)}"
   603	            self.router_channels = stage_ch[afd_stage]
   604	
   605	            if self.afd_router:
   606	                self.router = FrequencyReliabilityRouter(
   607	                    self.router_channels, low_r=low_r, mid_r=mid_r,
   608	                    cond_on_view=router_cond_view)
   609	            if self.afd_cvfc:
   610	                self.cvfc = CrossViewFrequencyCounterfactual(
   611	                    low_r=low_r, mid_r=mid_r, high_drop_p=high_drop_p)
   612	
   613	            # pooling
   614	            self.pool = GeMPool() if pool == 'gem' else None  # None -> avg in forward
   615	        elif backbone == 'swin_small':
   616	            # SOLIDER Swin-Small (team asset, SOTA push).  AFD frequency modules
   617	            # insert at resnet shallow stages (stem/layer1/layer2) that do NOT
   618	            # exist in Swin -> AFD is unsupported here (OVLI is the headline and
   619	            # needs no AFD).  Enforce so a stray --use_afd cannot silently no-op.
   620	            assert not use_afd, ("backbone='swin_small' does not support the AFD "
   621	                                 "frequency modules (router/cvfc insert at resnet "
   622	                                 "shallow stages). Run swin with --use_afd off "
   623	                                 "(OVP/OVLI are independent of AFD).")
   624	            self.backbone_swin = SwinBackboneReID(
   625	                img_size=tuple(img_size), pretrain_path=swin_pretrain,
   626	                semantic_weight=swin_semantic_weight,
   627	                iso_branch=self.airl_dualbranch_iso,
   628	                iso_stage=self.airl_iso_stage,
   629	                iso_trunk_recce=self.airl_iso_trunk_recce)
   630	            self.in_planes = self.backbone_swin.out_dim   # 768
   631	            # OVLI hooks model.layer4 -> point it at the Swin Identity hook module
   632	            # so the hook captures the (B,768,H,W) last-stage map.
   633	            self.layer4 = self.backbone_swin.layer4
   634	            # Swin's last map is LayerNorm'd (signed, ~half negative); GeM's
   635	            # clamp(min=eps) would destroy the negative half -> force avg pooling
   636	            # (== SOLIDER's native avgpool head over the same map).
   637	            self.pool = None
   638	        else:
   639	            raise ValueError(f"unknown backbone '{backbone}' "
   640	                             f"(expected 'resnet50' or 'swin_small')")
   641	
   642	        # BNNeck (f_full -- the original head: full-resolution identity evidence)
   643	        self.bottleneck = nn.BatchNorm1d(self.in_planes)
   644	        self.bottleneck.bias.requires_grad_(False)
   645	        self.bottleneck.apply(weights_init_kaiming)
   646	
   647	        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
   648	        self.classifier.apply(weights_init_classifier)
   649	
   650	        # AIRL dual-branch: a SECOND independent BNNeck head (f_rec).  Same structure
   651	        # / init recipe as f_full (frozen-bias BNNeck + bias-free classifier), but its
   652	        # OWN parameters so the two heads can specialise (f_rec absorbs the
   653	        # degradation-consistency signal, f_full stays clean).
   654	        #   * airl_dualbranch     : f_rec pools the SHARED global_feat (fully shared
   655	        #                           trunk -> the gradient that collapsed the heads).
   656	        #   * airl_dualbranch_iso : f_rec pools the INDEPENDENT rec last-stage map
   657	        #                           (gradient-isolated trunk -> heads re-diverge).
   658	        # Only built when one of the two is on -> the OFF model is structurally
   659	        # identical to the single-head baseline (no extra params).
   660	        if self.airl_dualbranch or self.airl_dualbranch_iso:
   661	            self.bottleneck_rec = nn.BatchNorm1d(self.in_planes)
   662	            self.bottleneck_rec.bias.requires_grad_(False)
   663	            self.bottleneck_rec.apply(weights_init_kaiming)
   664	            self.classifier_rec = nn.Linear(self.in_planes, num_classes, bias=False)
   665	            self.classifier_rec.apply(weights_init_classifier)
   666	
   667	    # --- backbone forward with optional router insertion ------------------- #
   668	    def _forward_backbone(self, x, view_idx=None, feat_override=None,
   669	                          insert_router=False):
   670	        """Run stem->layer4. If insert_router, apply router at self.afd_stage.
   671	
   672	        feat_override: if given, a dict {stage: tensor} used to *replace* the
   673	        feature at that stage (for counterfactual passes that re-enter mid-network).
   674	        """
   675	        band_w = None
   676	        if self.backbone == 'swin_small':
   677	            # Swin wrapper runs the full backbone and routes the last spatial map
   678	            # through its Identity hook point (so the OVLI layer4 hook fires).
   679	            # No AFD router/cvfc for swin -> band_w stays None.
   680	            feat_map = self.backbone_swin(x)
   681	            return feat_map, band_w
   682	        x = self.stem(x)
   683	        if self.afd_stage == 'stem':
   684	            x = self._maybe_route(x, 'stem', view_idx, insert_router)
   685	            x, band_w = x if isinstance(x, tuple) else (x, band_w)
   686	
   687	        x = self.layer1(x)
   688	        if self.afd_stage == 'layer1':
   689	            x = self._maybe_route(x, 'layer1', view_idx, insert_router)
   690	            x, band_w = x if isinstance(x, tuple) else (x, band_w)
   691	
   692	        x = self.layer2(x)
   693	        if self.afd_stage == 'layer2':
   694	            x = self._maybe_route(x, 'layer2', view_idx, insert_router)
   695	            x, band_w = x if isinstance(x, tuple) else (x, band_w)
   696	
   697	        x = self.layer3(x)
   698	        x = self.layer4(x)
   699	        return x, band_w
   700	
   701	    def _maybe_route(self, x, stage, view_idx, insert_router):
   702	        if insert_router and self.afd_router and stage == self.afd_stage:
   703	            return self.router(x, view_idx)   # returns (feat, w)
   704	        return x
   705	
   706	    def _pool(self, x):
   707	        if self.pool is not None:
   708	            return self.pool(x)
   709	        return F.adaptive_avg_pool2d(x, 1).flatten(1)
   710	
   711	    def _embed(self, x):
   712	        """global feat -> BNNeck feat."""
   713	        g = self._pool(x)
   714	        bn = self.bottleneck(g)
   715	        return g, bn
   716	
   717	    def _embed_rec(self, x):
   718	        """rec map -> pooled rec feat -> BNNeck_rec feat (independent f_rec head).
   719	
   720	        Used only by the gradient-isolated dual-branch: the rec map already comes
   721	        from a detached trunk + independent late stage, so pooling + bottleneck_rec
   722	        here keeps the whole f_rec head isolated from the shared trunk.
   723	        """
   724	        g = self._pool(x)
   725	        bn = self.bottleneck_rec(g)
   726	        return g, bn
   727	
   728	    # --- public forward ---------------------------------------------------- #
   729	    def forward(self, x, view_idx=None, return_cvfc=False, return_dual=False,
   730	                rec_only=False):
   731	        """
   732	        Train: returns dict with global_feat, bn_feat, logits, band_w,
   733	               and (if return_cvfc & afd_cvfc) counterfactual embeddings.
   734	               When airl_dualbranch is on, the dict ALSO carries the f_rec head's
   735	               'bn_feat_rec' / 'logits_rec' (computed from the SAME pooled
   736	               global_feat through the second BNNeck) so the train loop can add
   737	               the f_rec ID-CE + degradation-consistency.
   738	        Eval : returns the L2-normalized f_full BN feature (single head); when
   739	               airl_dualbranch is on AND return_dual=True, returns the tuple
   740	               (f_full_norm, f_rec_norm) so the dual-branch eval can SOFT-fuse
   741	               the two cosine scores.  return_dual defaults to False, so the
   742	               legacy single-feature eval path (extract_features) is unchanged.
   743	
   744	               airl_dualbranch_iso: identical (f_full_norm, f_rec_norm) eval tuple
   745	               and bn_feat_rec/logits_rec train keys, but f_rec is pooled from the
   746	               INDEPENDENT rec last-stage map (gradient-isolated trunk) instead of
   747	               the shared global_feat.  return_rec on the Swin backbone yields BOTH
   748	               maps in ONE forward (split path).
   749	        """
   750	        # ---- AIRL gradient-isolated dual-branch path -------------------------- #
   751	        # When iso is on we need BOTH the f_full map and the independent rec map.
   752	        # The Swin split forward returns both in one pass; f_full pools the shared
   753	        # map (bn_feat/global_feat) and f_rec pools the rec map through bottleneck_rec.
   754	        # Fork-feed gradient regime (see _forward_swin_split): the DEGRADED pass
   755	        # (rec_only) always detaches the fork so the consistency gradient never
   756	        # reaches the trunk; the CLEAN pass detaches only when iso_trunk_recce is
   757	        # False -- with the fix (True) the clean f_rec ID-CE reflows into the trunk
   758	        # (extra identity supervision), while the consistency stays trunk-isolated.
   759	        # `or rec_only` so the rec-only degraded contract is honoured even if a
   760	        # caller invokes it under model.eval() (training=False, return_dual=False) --
   761	        # otherwise want_iso would be False and the rec_only dict request would
   762	        # silently fall through to the single f_full eval tensor.
   763	        want_iso = self.airl_dualbranch_iso and (self.training or return_dual
   764	                                                 or rec_only)
   765	        if want_iso:
   766	            full_map, rec_map = self.backbone_swin(
   767	                x, return_rec=True, rec_only=rec_only)
   768	            # rec_only (the degraded consistency pass): compute ONLY the f_rec head.
   769	            # f_full's BNNeck is NOT run on the degraded images, so self.bottleneck's
   770	            # running mean/var stay CLEAN (no degraded-ground stat leak into the
   771	            # f_full eval head) -- f_full is a true clean expert -- and the f_full
   772	            # pool+BN+classifier compute is skipped (cheaper).  The clean forward
   773	            # (rec_only=False) still produces both heads as usual.
   774	            if rec_only:
   775	                _grec, bn_feat_rec = self._embed_rec(rec_map)
   776	                return {
   777	                    'bn_feat_rec': bn_feat_rec,
   778	                    'logits_rec': self.classifier_rec(bn_feat_rec),
   779	                }
   780	            global_feat, bn_feat = self._embed(full_map)
   781	            _grec, bn_feat_rec = self._embed_rec(rec_map)
   782	            if not self.training:
   783	                # eval: ONE forward -> two L2-normalized features (f_full, f_rec).
   784	                return (F.normalize(bn_feat, dim=1),
   785	                        F.normalize(bn_feat_rec, dim=1))
   786	            out = {
   787	                'global_feat': global_feat,   # f_full triplet (shared trunk)
   788	                'bn_feat': bn_feat,
   789	                'logits': self.classifier(bn_feat),
   790	                'band_w': None,               # swin has no AFD band weights
   791	                # f_rec head over the INDEPENDENT rec map (own ID-CE + AIRL
   792	                # consistency in the train loop); pooled rec feat exposed too so the
   793	                # smoke can confirm it is NOT the shared global_feat.
   794	                'global_feat_rec': _grec,
   795	                'bn_feat_rec': bn_feat_rec,
   796	                'logits_rec': self.classifier_rec(bn_feat_rec),
   797	            }
   798	            # AFD CVFC is swin-incompatible (asserted off), so no cf block here.
   799	            return out
   800	
   801	        feat_map, band_w = self._forward_backbone(
   802	            x, view_idx=view_idx, insert_router=self.afd_router)
   803	        global_feat, bn_feat = self._embed(feat_map)
   804	
   805	        if not self.training:
   806	            f_full = F.normalize(bn_feat, dim=1)
   807	            if return_dual and self.airl_dualbranch:
   808	                # second head shares the SAME pooled global_feat -> ONE forward,
   809	                # two L2-normalized features for the distmat-level soft fusion.
   810	                bn_feat_rec = self.bottleneck_rec(global_feat)
   811	                return f_full, F.normalize(bn_feat_rec, dim=1)
   812	            return f_full
   813	
   814	        out = {
   815	            'global_feat': global_feat,   # for triplet (before BN)
   816	            'bn_feat': bn_feat,           # BN feature
   817	            'logits': self.classifier(bn_feat),
   818	            'band_w': band_w,             # (B,3) or None
   819	        }
   820	
   821	        if self.airl_dualbranch:
   822	            # f_rec head: its OWN BNNeck + classifier on the shared global_feat.
   823	            # The train loop applies f_rec ID-CE (so f_rec is a valid identity
   824	            # space) PLUS the AIRL degradation-consistency (so it is robust to a
   825	            # low pixel budget).  global_feat is shared, so the global triplet is
   826	            # NOT duplicated for f_rec (single backbone-level triplet).
   827	            bn_feat_rec = self.bottleneck_rec(global_feat)
   828	            out['bn_feat_rec'] = bn_feat_rec
   829	            out['logits_rec'] = self.classifier_rec(bn_feat_rec)
   830	
   831	        if return_cvfc and self.afd_cvfc:
   832	            # build counterfactual at the SHALLOW stage, then run remainder.
   833	            # We re-extract the shallow feature, apply CVFC, and finish forward.
   834	            cf = self._forward_counterfactual(x, view_idx)
   835	            out.update(cf)
   836	        return out
   837	
   838	    def _forward_counterfactual(self, x, view_idx):
   839	        """Run two cheap counterfactual passes (high-drop, low-pass) from the
   840	        shallow stage onward, returning their BN embeddings for consistency loss."""
   841	        # get shallow feature up to (and including) the insertion stage
   842	        stage = self.afd_stage
   843	        shallow = self.stem(x)
   844	        if stage in ('layer1', 'layer2'):
   845	            shallow = self.layer1(shallow)
   846	        if stage == 'layer2':
   847	            shallow = self.layer2(shallow)
   848	
   849	        def finish(feat):
   850	            # continue from the stage AFTER the insertion point

codex
模型侧看下来，eval 返回的两个头确实都经过 `F.normalize`；iso degrade pass 只跑 rec head，full BNNeck 不会吃 degraded 图。现在重点转到训练/评估：AMP、consistency、distmat 融合、arg 校验和日志打印。
exec
/bin/zsh -lc 'rg -n "''^def |''^class |argparse|add_argument|airl_degrade|airl_consistency_loss|extract|inference|evaluate|airl_dualbranch_eval|distmat|fuse_w|autocast|GradScaler|isfinite|nan|inf" code/afd_train.py' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
139:import argparse
163:class OVPMemory(nn.Module):
240:            logits = logits.masked_fill(~valid.view(1, -1), float('-inf'))
251:class OVLISetPool(nn.Module):
286:    autocast(enabled=False)); every reduction is NaN-safe.
415:class OVLIHead(nn.Module):
418:    Reuses the maxsim_probe token-extraction recipe (hook model.layer4 -> the
556:    # -- token extraction ---------------------------------------------------- #
569:        # (the cached map may be fp16 under autocast).
670:                                    finite floor (never -inf) so an empty row (can
765:        bias_ij  = log(w_ij)   (w_ij >= wmin > 0 -> finite, never -inf/NaN).
874:        # Use a large finite floor (not -inf) so a fully-masked row can never
875:        # yield -inf, and so (-inf)-(-inf)=nan can never appear even before the
913:def ovli_rerank_eval(model, ovli, dataset, args, device):
917:    Mirrors run_cross_view_eval but additionally extracts projected tokens via
927:    def extract(samples):
1006:        qf, qt, qp, qc = extract(q)
1007:        gf, gt, gp, gc = extract(g)
1009:            results[tag] = {'global': (float('nan'), float('nan')),
1010:                            'rerank': (float('nan'), float('nan'))}
1020:        from maxsim_probe import eval_from_distmat
1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
1029:def airl_dualbranch_eval(model, dataset, args, device):
1030:    """AIRL dual-branch eval: extract BOTH heads (f_full, f_rec) in ONE forward
1032:    (cos = w*cos_rec + (1-w)*cos_full, w = args.airl_fuse_w, fixed) for A->G and
1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
1045:    from maxsim_probe import eval_from_distmat
1051:    def extract(samples):
1071:    w = args.airl_fuse_w
1078:        q_full, q_rec, qp, qc = extract(q)
1079:        g_full, g_rec, gp, gc = extract(g)
1081:            nan2 = (float('nan'), float('nan'))
1082:            results[tag] = {'full': nan2, 'rec': nan2, 'fuse': nan2}
1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
1121:#      "if this person were shot from a UAV, how much information would survive".
1135:#   * The consistency loss runs in TRUE fp32 (autocast disabled) for KL/cosine
1140:def airl_degrade(imgs, min_scale, blur=False, generator=None):
1191:def airl_consistency_loss(logits_o, bn_o, logits_d, bn_d, mode='kl', tau=4.0):
1207:    Returns a finite non-negative scalar.  A light finite guard (nan_to_num) is
1210:    pathological forward ever produced inf/NaN logits.
1217:        return torch.nan_to_num(out)
1227:    return torch.nan_to_num(kl * (tau * tau))
1233:def main():
1234:    ap = argparse.ArgumentParser()
1235:    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
1241:    ap.add_argument('--dataset', default='cargo',
1243:    ap.add_argument('--out_dir', default='./log/cargo/cvpb_ovp')
1244:    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
1245:    ap.add_argument('--epochs', type=int, default=60)
1246:    ap.add_argument('--warmup_epochs', type=int, default=10)
1247:    ap.add_argument('--lr', type=float, default=3.5e-4)
1248:    ap.add_argument('--weight_decay', type=float, default=5e-4)
1249:    ap.add_argument('--P', type=int, default=16)
1250:    ap.add_argument('--K', type=int, default=4)
1251:    ap.add_argument('--test_batch', type=int, default=128)
1252:    ap.add_argument('--workers', type=int, default=8)
1253:    ap.add_argument('--margin', type=float, default=0.3)
1254:    ap.add_argument('--label_smooth', type=float, default=0.1)
1255:    ap.add_argument('--eval_period', type=int, default=10)
1256:    ap.add_argument('--seed', type=int, default=1)
1257:    ap.add_argument('--no_amp', action='store_true', help='disable mixed precision')
1259:    ap.add_argument('--last_stride', type=int, default=1)
1260:    ap.add_argument('--pool', default='gem', choices=['gem', 'avg'])
1267:    ap.add_argument('--backbone', default='resnet50',
1271:    ap.add_argument('--swin_pretrain', default='',
1276:    ap.add_argument('--swin_semantic_weight', type=float, default=0.2,
1280:    ap.add_argument('--swin_lr_factor', type=float, default=0.1,
1286:    ap.add_argument('--use_afd', action='store_true')
1287:    ap.add_argument('--afd_router', type=int, default=1)
1288:    ap.add_argument('--afd_cvfc', type=int, default=1)
1289:    ap.add_argument('--afd_stage', default='layer1',
1291:    ap.add_argument('--router_cond_view', type=int, default=1)
1292:    ap.add_argument('--low_r', type=float, default=0.125)
1293:    ap.add_argument('--mid_r', type=float, default=0.30)
1294:    ap.add_argument('--high_drop_p', type=float, default=0.5)
1295:    ap.add_argument('--w_cvfc', type=float, default=0.5)
1297:    ap.add_argument('--ovp', action='store_true',
1299:    ap.add_argument('--ovp_lambda', type=float, default=0.5,
1301:    ap.add_argument('--ovp_tau', type=float, default=0.05,
1303:    ap.add_argument('--ovp_momentum', type=float, default=0.2,
1305:    ap.add_argument('--ovp_warmup', type=int, default=10,
1308:    ap.add_argument('--ovli', action='store_true',
1310:    ap.add_argument('--ovli_lambda', type=float, default=0.5,
1312:    ap.add_argument('--ovli_tau', type=float, default=0.05,
1314:    ap.add_argument('--ovli_alpha', type=float, default=0.5,
1316:    ap.add_argument('--ovli_dim', type=int, default=256,
1318:    ap.add_argument('--ovli_grid', type=int, nargs=2, default=[8, 4],
1320:    ap.add_argument('--ovli_warmup', type=int, default=10,
1322:    ap.add_argument('--ovli_rerank', action='store_true',
1326:    ap.add_argument('--ovli_pool', default='mean',
1333:    ap.add_argument('--ovli_topk', type=int, default=8,
1335:    ap.add_argument('--ovli_thresh', type=float, default=0.0,
1338:    ap.add_argument('--ovli_allview', action='store_true',
1349:    ap.add_argument('--ovli_match', default='maxsim', choices=['maxsim', 'avg'],
1360:    ap.add_argument('--ovli_align', default='free', choices=['free', 'ordered'],
1377:    ap.add_argument('--ovli_setpool', default='mean',
1390:    ap.add_argument('--ovli_vlad_clusters', type=int, default=8,
1392:    ap.add_argument('--ovli_attn_heads', type=int, default=4,
1394:    ap.add_argument('--ovli_so_rank', type=int, default=32,
1403:    ap.add_argument('--ovli_setpool_residual', type=int, default=1,
1417:    ap.add_argument('--acvp', action='store_true',
1423:    ap.add_argument('--acvp_gamma', type=float, default=0.5,
1427:    ap.add_argument('--acvp_wmin', type=float, default=0.3,
1430:    ap.add_argument('--acvp_eta', type=float, default=0.05,
1432:    ap.add_argument('--acvp_margin', type=float, default=0.0,
1434:    ap.add_argument('--acvp_warmup', type=int, default=10,
1440:    # consistency = loss); the optimizer is untouched.  See airl_degrade /
1441:    # airl_consistency_loss above.  Independent of OVP/OVLI/ACVP (can co-run, but
1443:    ap.add_argument('--airl', action='store_true',
1449:    ap.add_argument('--airl_lambda', type=float, default=0.5,
1452:    ap.add_argument('--airl_min_scale', type=float, default=0.25,
1457:    ap.add_argument('--airl_consistency', default='kl', choices=['kl', 'feat'],
1461:    ap.add_argument('--airl_tau', type=float, default=4.0,
1464:    ap.add_argument('--airl_blur', action='store_true',
1467:    ap.add_argument('--airl_warmup', type=int, default=5,
1477:    #     cos = airl_fuse_w * cos(f_rec) + (1 - airl_fuse_w) * cos(f_full)
1490:    ap.add_argument('--airl_dualbranch', action='store_true',
1496:    ap.add_argument('--airl_fuse_w', type=float, default=0.25,
1498:                         '(cos = airl_fuse_w*cos_rec + (1-airl_fuse_w)*cos_full); '
1504:    ap.add_argument('--airl_dualbranch_iso', action='store_true',
1520:                         '+ --airl_fuse_w). Default OFF reproduces the baseline.')
1521:    ap.add_argument('--airl_iso_stage', type=int, default=3,
1541:    ap.add_argument('--airl_iso_trunk_recce', type=int, default=1, choices=[0, 1],
1587:    # produce inf/NaN.  Enforce wmin in (0,1], eta>0, gamma>=0 at parse time so a
1628:        if not (0.0 <= args.airl_fuse_w <= 1.0):
1629:            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
1630:                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
1636:        if args.airl_fuse_w != 0.25:
1637:            print(f"[AIRL-DUAL][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
1652:    #     warmup) and --airl_fuse_w, validated identically;
1676:        if not (0.0 <= args.airl_fuse_w <= 1.0):
1677:            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
1678:                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
1679:        if args.airl_fuse_w != 0.25:
1680:            print(f"[AIRL-ISO][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
1721:    print(f"  airl_dualbranch={args.airl_dualbranch} (fuse_w={args.airl_fuse_w} "
1730:          f"fuse_w={args.airl_fuse_w} "
1927:              f"w={args.airl_fuse_w}")
1978:              f"cos=w*cos_rec+(1-w)*cos_full w={args.airl_fuse_w}")
1980:    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)
2028:            with torch.amp.autocast('cuda', enabled=not args.no_amp):
2070:            # OVLI: compute in TRUE fp32 (autocast disabled) -- the cos/MaxSim/
2072:            # the autocast forward already cached the fp16 layer4 map) keeps the
2078:                with torch.amp.autocast('cuda', enabled=False):
2133:                    # second forward runs under the SAME autocast as the original so
2136:                        deg_imgs, deg_scales = airl_degrade(
2139:                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
2142:                    with torch.amp.autocast('cuda', enabled=False):
2146:                        loss_airl = airl_consistency_loss(
2177:                        deg_imgs, deg_scales = airl_degrade(
2180:                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
2183:                    with torch.amp.autocast('cuda', enabled=False):
2187:                        loss_airl = airl_consistency_loss(
2199:            # airl_consistency_loss.  So the consistency gradient flows ONLY through
2221:                        deg_imgs, deg_scales = airl_degrade(
2224:                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
2228:                    with torch.amp.autocast('cuda', enabled=False):
2232:                        loss_airl = airl_consistency_loss(
2346:        # (degradation not biting) / NaN/inf, OR n_ground==0 all epoch (mask wrong /
2353:            bad = (not math.isfinite(a_cons)) or (n_g == 0)
2368:            bad = (not math.isfinite(a_cons)) or (not math.isfinite(a_cerec)) \
2401:                # airl_dualbranch_eval calls model(return_dual=True) which yields
2405:                dual = airl_dualbranch_eval(model, dataset, args, device)
2408:                print(f"  ---- {_ev_tag} (fuse_w={args.airl_fuse_w}) "

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1,360p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
     1	# encoding: utf-8
     2	"""
     3	CVPB training loop on CARGO = the afd_reid baseline trainer + OVP-Mem / OVLI.
     4	
     5	This reuses the dataset / model / eval from ../afd_reid unchanged.  Two optional
     6	cross-view auxiliary mechanisms are bolted on (both default OFF -> the BoT
     7	baseline is reproduced exactly).  They are NOT mutually exclusive: --ovp,
     8	--ovli, and --ovp --ovli (both) are all valid; in `both` mode each loss keeps
     9	its own warmup / lambda / diagnostics and the two terms are simply summed into
    10	the same total (complementarity test: does prototype-memory + late-interaction
    11	beat OVP-only?).
    12	
    13	  --ovp   : OVP-Mem (Opposite-View Prototype Memory).  Per-pid per-view EMA
    14	            prototype bank + opposite-view InfoNCE.  Known prior-art overlap
    15	            (CMPC / MBCE / PDPA) -> kept only as an empirical auxiliary.
    16	
    17	  --ovli  : OVLI (Opposite-View Late-Interaction Evidence Alignment).  The
    18	            headline mechanism.  NO prototype / memory / EMA.  Instead a pure
    19	            sample-to-sample, in-batch, opposite-view *retrieval* loss whose
    20	            score is a hybrid of (a) global cosine and (b) a symmetric
    21	            token-set late-interaction (ColBERT/MaxSim-style **partial**
    22	            matching).  Framing: cross-view identity evidence is a partial
    23	            token-set matching problem, not a global prototype alignment one --
    24	            aerial<->ground has no 1-1 part correspondence, so a global
    25	            prototype penalizes missing regions whereas partial MaxSim lets the
    26	            tokens that *can* be matched carry the similarity.
    27	
    28	  --acvp  : ACVP (Ambiguity-Calibrated opposite-View negative relaxation).  An
    29	            OVLI calibration (requires --ovli).  Treats the opposite-view identity
    30	            prototype ONLY as a DETACHED ambiguity SENSOR: it softens the
    31	            unreliable NEGATIVES in the OVLI cross-view contrastive denominator and
    32	            does NOT do any prototype-positive alignment (so it stays clear of the
    33	            OVP / CMPC / PDPA prototype-contrast prior art).  Mechanism: maintain a
    34	            detached per-pid per-view EMA prototype bank (its own OVPMemory, read
    35	            detached); for an anchor i and an opposite-view negative j (different
    36	            pid) measure how close j's opposite-view identity sits to i,
    37	                delta_ij = cos(z_i, P[y_j, view_j]) - cos(z_i, P[y_i, view_j]),
    38	            map it to a weight w_ij = clamp(1 - gamma*sigmoid((delta_ij-margin)/
    39	            eta), w_min, 1) and ADD log(w_ij) to that negative's logit in the
    40	            DENOMINATOR only (positives untouched).  No learnable params, no
    41	            gradient to the encoder/proto (pure detached re-weighting).  Default
    42	            OFF => the OVLI loss is reproduced byte-for-byte.  Per-epoch
    43	            kill-switch log: relaxed_neg_frac (w<0.95 share) + mean_w (stop if
    44	            frac>0.30 or mean_w<0.75 => negatives broadly weakened = bad).
    45	
    46	OVLI details (the load-bearing design)
    47	--------------------------------------
    48	* Tokens: hook model.layer4 (the GeM-input spatial map, 16x8 for 256x128),
    49	  adaptive-avg-pool to a KxK' grid, flatten to K local tokens, then a NEW
    50	  learnable 1x1-conv/linear projection to ovli_dim (256) + per-token L2-norm.
    51	  ** The projection is a new learnable parameter set and IS added to the
    52	     optimizer ** (this is the key structural difference vs OVP, which adds no
    53	     params).  The hook does NOT detach -> gradient flows layer4 -> proj.
    54	
    55	* Opposite-view retrieval loss (supervised-contrastive, logsumexp):
    56	  within the batch, for each anchor i in view v, the positives are the same-pid
    57	  samples in the OPPOSITE view (1-v) and the negatives are the opposite-view
    58	  samples of OTHER pids.  Same-view samples are excluded as candidates entirely
    59	  (this is a *cross-view* objective).  Pairwise score:
    60	      score(i,j) = alpha * cos(g_i, g_j)
    61	                 + (1 - alpha) * sym_MaxSim(tok_i, tok_j)
    62	      sym_MaxSim = 0.5 * ( pool_u max_s <u,s> + pool_s max_u <u,s> )   # bidir
    63	  where pool_* is the --ovli_pool dustbin variant over the per-token max scores:
    64	      mean     : average over ALL token-max scores (original; NOT a true dustbin
    65	                 -- low-score non-corresponding tokens still drag the pair down).
    66	      topk     : average of the top-k highest token-max scores (--ovli_topk),
    67	                 i.e. drop the K-k worst-matching tokens -> sparse evidence /
    68	                 dustbin approximation; the headline AG-ReID design.
    69	      thresh   : average of token-max scores above theta (--ovli_thresh), with a
    70	                 single-max fallback so a fully-masked pair never NaNs.
    71	      softtopk : softmax(token-max / tau)-weighted mean (smooth, differentiable
    72	                 top-k surrogate).
    73	  Both MaxSim directions use the same pooling, so sym_MaxSim stays symmetric;
    74	  the eval rerank (--ovli_rerank) reuses the identical pooling (train/test
    75	  symmetry).  --ovli_pool mean reproduces the previous behaviour exactly.
    76	  Multi-positive InfoNCE per anchor:
    77	      L_i = -logsumexp(score(i,pos)/tau) + logsumexp(score(i,cand)/tau)
    78	  averaged over anchors that have >=1 opposite-view positive AND >=1
    79	  opposite-view negative in the batch.  No memory / EMA / prototype.
    80	
    81	* lambda warmup (--ovli_warmup, default 10): the H1 lesson from OVP -- linearly
    82	  ramp lambda over the first N epochs so the (randomly-initialised) projection
    83	  cannot inject a sharp early gradient.  Per-epoch log records
    84	  OVLI[lam_eff pos_score neg_score gap] for collapse / over-strong monitoring.
    85	
    86	* eval: OVLI is a TRAIN-time loss only; default eval is global-only (unchanged,
    87	  identical to the baseline).  --ovli_rerank additionally reports a
    88	  global + sym_MaxSim rerank at eval time (both numbers printed), so train/test
    89	  stay symmetric and the rerank is opt-in.
    90	
    91	Baseline (no flags):
    92	    resnet50(IMAGENET1K_V1) + GeM + BNNeck
    93	    loss = CE(label-smooth 0.1) + batch-hard triplet (margin 0.3)
    94	    PK sampler P=16 x K=4 (bs=64), AdamW lr 3.5e-4, 10-ep warmup + cosine, 60 ep.
    95	    eval every 10 ep: A->G and G->A cross-view mAP / R1 / mINP.
    96	
    97	--ovp (OVP-Mem):
    98	    Maintain, per train pid, two EMA prototypes (aerial / ground) of the
    99	    L2-normalized BNNeck feature in a register_buffer of shape
   100	    [num_pid, 2, feat_dim]; EMA momentum 0.2.  Each step:
   101	      1) update the prototypes of the pids/views present in the batch (EMA),
   102	      2) add an InfoNCE loss pulling each sample toward its OWN pid's
   103	         OPPOSITE-view prototype and away from all other pids' opposite-view
   104	         prototypes:  CE( cos(z, P[:, opp_view]) / tau ,  y ).
   105	    total = CE + triplet + lambda_ovp * OVP.   (batch size unchanged, bs=64.)
   106	
   107	    Cold-start handling: a sample only contributes to the OVP loss once its own
   108	    opposite-view prototype has been initialized (seen >=1 time); candidate
   109	    columns that are still uninitialized are masked out of the InfoNCE logits so
   110	    a zero prototype can never act as an easy negative/positive.
   111	
   112	Run on lab-3090:
   113	    cd /root/work/SOLIDER-REID/experiments/cargo_cvpb
   114	    # OVP-Mem (empirical auxiliary):
   115	    PYTHONUNBUFFERED=1 python3 afd_train.py \
   116	        --data_root /root/work/SOLIDER-REID/data \
   117	        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp \
   118	        --ovp \
   119	        2>&1 | tee /tmp/cvpb_ovp.log
   120	    # OVLI (headline; late-interaction opposite-view retrieval):
   121	    PYTHONUNBUFFERED=1 python3 afd_train.py \
   122	        --data_root /root/work/SOLIDER-REID/data \
   123	        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovli \
   124	        --ovli --ovli_rerank \
   125	        2>&1 | tee /tmp/cvpb_ovli.log
   126	    # both (complementarity test: OVP prototype + OVLI late-interaction):
   127	    PYTHONUNBUFFERED=1 python3 afd_train.py \
   128	        --data_root /root/work/SOLIDER-REID/data \
   129	        --out_dir   /root/work/SOLIDER-REID/log/cargo/cvpb_ovp_ovli \
   130	        --ovp --ovli --ovli_rerank \
   131	        2>&1 | tee /tmp/cvpb_ovp_ovli.log
   132	    # baseline reproduction: drop all of --ovp / --ovli
   133	"""
   134	import os
   135	import sys
   136	import time
   137	import math
   138	import random
   139	import argparse
   140	
   141	import numpy as np
   142	import torch
   143	import torch.nn as nn
   144	import torch.nn.functional as F
   145	from torch.utils.data import DataLoader
   146	
   147	# reuse afd_reid building blocks unchanged
   148	sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
   149	                                '..', 'afd_reid'))
   150	from cargo_dataset import (CARGO, CARGOImageDataset, build_transforms,  # noqa: E402
   151	                           RandomIdentitySampler, filter_by_view)
   152	from agreid_dataset import AGReIDv2  # noqa: E402  -- second benchmark (--dataset agreid)
   153	from agreid_v2_combined import AGReIDV2Combined  # noqa: E402  -- official exp1(A->G)+exp4(G->A) (--dataset agreid_v2)
   154	from afd_model import build_model, weights_init_kaiming  # noqa: E402
   155	# reuse the exact loss / schedule / eval helpers so nothing diverges
   156	from afd_train import (CrossEntropyLabelSmooth, TripletLoss, WarmupCosineLR,  # noqa: E402
   157	                       run_cross_view_eval, print_eval, set_seed)
   158	
   159	
   160	# --------------------------------------------------------------------------- #
   161	# OVP-Mem: opposite-view prototype memory
   162	# --------------------------------------------------------------------------- #
   163	class OVPMemory(nn.Module):
   164	    """Per-pid, per-view EMA prototype bank + opposite-view InfoNCE loss.
   165	
   166	    bank: register_buffer [num_pid, 2, feat_dim]  (view 0 = Aerial, 1 = Ground)
   167	    init: register_buffer [num_pid, 2] uint8 = has this (pid,view) been seen.
   168	
   169	    Prototypes are L2-normalized; features used for update/loss are the
   170	    L2-normalized BNNeck features (detached for the EMA update).
   171	    """
   172	
   173	    def __init__(self, num_pid, feat_dim, momentum=0.2, tau=0.05):
   174	        super().__init__()
   175	        self.num_pid = num_pid
   176	        self.feat_dim = feat_dim
   177	        self.momentum = momentum          # weight on the *new* batch mean
   178	        self.tau = tau
   179	        self.register_buffer('bank', torch.zeros(num_pid, 2, feat_dim))
   180	        self.register_buffer('inited', torch.zeros(num_pid, 2, dtype=torch.uint8))
   181	
   182	    @torch.no_grad()
   183	    def update(self, feats, labels, views):
   184	        """EMA-update prototypes for the (pid,view) groups present in the batch.
   185	
   186	        feats:(B,D) L2-normed BN feats (detached). labels:(B,) long.
   187	        views:(B,) long in {0,1}.  Per group, EMA toward the group mean, then
   188	        re-normalize the prototype.
   189	        """
   190	        feats = feats.detach()
   191	        for v in (0, 1):
   192	            vmask = (views == v)
   193	            if not torch.any(vmask):
   194	                continue
   195	            fv = feats[vmask]
   196	            lv = labels[vmask]
   197	            uniq = torch.unique(lv)
   198	            for pid in uniq:
   199	                pid_i = int(pid)
   200	                gmean = fv[lv == pid].mean(dim=0)
   201	                gmean = F.normalize(gmean, dim=0)
   202	                if self.inited[pid_i, v] == 0:
   203	                    self.bank[pid_i, v] = gmean
   204	                    self.inited[pid_i, v] = 1
   205	                else:
   206	                    self.bank[pid_i, v] = ((1 - self.momentum) * self.bank[pid_i, v]
   207	                                           + self.momentum * gmean)
   208	                    self.bank[pid_i, v] = F.normalize(self.bank[pid_i, v], dim=0)
   209	
   210	    def loss(self, feats, labels, views):
   211	        """InfoNCE: each sample classified against ALL pids' opposite-view
   212	        prototypes, target = its own pid.  feats:(B,D) L2-normed BN feats
   213	        (NOT detached -> gradient flows to the encoder).
   214	
   215	        Returns a scalar loss (0 if no sample has an initialized own-opp proto).
   216	        """
   217	        device = feats.device
   218	        opp = 1 - views                                   # opposite view per sample
   219	        total = feats.new_zeros(())
   220	        count = 0
   221	        for v_opp in (0, 1):
   222	            # samples whose opposite view == v_opp (i.e. samples from view 1-v_opp)
   223	            smask = (opp == v_opp)
   224	            if not torch.any(smask):
   225	                continue
   226	            protos = self.bank[:, v_opp, :]               # (num_pid, D)
   227	            valid = (self.inited[:, v_opp] == 1)          # (num_pid,) candidate cols
   228	            if valid.sum() == 0:
   229	                continue
   230	            z = feats[smask]                              # (b,D)
   231	            y = labels[smask]                             # (b,)
   232	            # only keep samples whose OWN opposite-view proto exists
   233	            own_ok = valid[y]
   234	            if own_ok.sum() == 0:
   235	                continue
   236	            z = z[own_ok]
   237	            y = y[own_ok]
   238	            logits = (z @ protos.t()) / self.tau          # (b, num_pid)
   239	            # mask out uninitialized candidate columns
   240	            logits = logits.masked_fill(~valid.view(1, -1), float('-inf'))
   241	            total = total + F.cross_entropy(logits, y)
   242	            count += 1
   243	        if count == 0:
   244	            return feats.new_zeros(())
   245	        return total / count
   246	
   247	
   248	# --------------------------------------------------------------------------- #
   249	# OVLI set pooling: learnable permutation-invariant aggregation of K tokens
   250	# --------------------------------------------------------------------------- #
   251	class OVLISetPool(nn.Module):
   252	    """Permutation-invariant pooling of K projected tokens -> 1 vector.
   253	
   254	    The headline OVLI score with --ovli_match avg --ovli_pool mean reduces
   255	    EXACTLY to <mean_k tok_i, mean_k tok_j> -- the gram of the per-sample MEAN of
   256	    the K L2-normed tokens -- and that mean-pool aggregation is the current best
   257	    single mechanism (52.37 cross-view mAP).
   258	
   259	    KEY DESIGN -- "mean + zero-init residual" (residual=True, default).  A
   260	    standalone *random-init* learnable pooling starts from noise (NetVLAD ep20
   261	    mAP collapsed to 14.66 << 52.37 << even pure global 45.14), because the
   262	    random aggregated vector is meaningless and drags the cross-view cosine down.
   263	    To start LOSSLESS from the 52.37 mean-pool and only *learn a correction*, the
   264	    forward is
   265	
   266	        pooled = mean_k(tok)  +  g * residual_module(tok)
   267	
   268	    where `g` is a scalar gate `nn.Parameter(zeros(1))` (one per module).  At
   269	    init g == 0, so `pooled == mean_k(tok)` BYTE-FOR-BYTE (== the existing
   270	    --ovli_match avg --ovli_pool mean 52.37 path); the gradient then opens the
   271	    residual.  `mean_k(tok)` is the UN-normalized mean of the K L2-normed tokens
   272	    (identical to what `tok.mean(dim=1)` feeds the avg/mean gram), so the residual
   273	    baseline matches the current best mechanism exactly.  residual=True REQUIRES
   274	    out_dim == dim (the residual is added to the D-dim mean); enforced by assert.
   275	
   276	    residual=False reverts to the ORIGINAL standalone pooling (random init -> the
   277	    aggregated vector fully REPLACES the mean) -- kept only so an ablation can
   278	    contrast standalone-vs-residual; it is the collapsing config and NOT default.
   279	
   280	    Each `*_residual()` returns a per-sample (B,out_dim) residual that is itself
   281	    permutation-invariant (the K axis is collapsed by a SUM / softmax-weighted-SUM
   282	    / mean / covariance), so `pooled` is permutation-invariant in the tokens
   283	    (verified to <1e-5 in smoke_ovli_setpool.py).  All parameters live in THIS
   284	    module -> list(ovli.parameters()) collects them -> they land in the optimizer
   285	    (the same contract as OVLIHead.proj).  Always runs in fp32 (called under
   286	    autocast(enabled=False)); every reduction is NaN-safe.
   287	
   288	      netvlad     : C learnable centers + per-token soft-assignment; residual
   289	                    aggregation -> intra-normalized (C x D) VLAD -> linear to
   290	                    out_dim.  (Arandjelovic et al., NetVLAD.)
   291	      attn        : H learnable query vectors; multi-head attention pooling over
   292	                    the K tokens (softmax weights over k) -> linear.  (Set
   293	                    Transformer PMA / learned-query attention pooling.)
   294	      gated       : a small per-token scalar reliability gate (sigmoid); gated
   295	                    convex average over the K tokens (lightweight token-wise
   296	                    reliability weighting).
   297	      secondorder : low-rank covariance pooling -- reduce tokens to r dims, take
   298	                    the order-invariant token covariance (z^T z / K), signed-sqrt
   299	                    normalize, flatten -> linear (token second-order statistics).
   300	    """
   301	
   302	    def __init__(self, mode, dim, out_dim=None, clusters=8, heads=4,
   303	                 lowrank=32, gate_hidden=None, residual=True, eps=1e-6):
   304	        super().__init__()
   305	        self.mode = str(mode)
   306	        self.dim = int(dim)
   307	        self.out_dim = int(out_dim) if out_dim else int(dim)
   308	        self.residual = bool(residual)
   309	        self.eps = float(eps)
   310	        # "mean + zero-init residual" needs the residual to be ADDED to the D-dim
   311	        # mean -> the residual must map to out_dim == dim.  (The configured call
   312	        # site always passes out_dim == proj_dim == dim, so this never trips in
   313	        # practice; the assert just makes the contract explicit.)
   314	        if self.residual:
   315	            assert self.out_dim == self.dim, (
   316	                f"--ovli_setpool_residual True needs out_dim==dim "
   317	                f"(mean+residual added in D-space), got dim={self.dim} "
   318	                f"out_dim={self.out_dim}")
   319	        D = self.dim
   320	        if self.mode == 'netvlad':
   321	            C = int(clusters)
   322	            self.clusters = C
   323	            self.assign = nn.Linear(D, C)                  # per-token soft-assign
   324	            self.centers = nn.Parameter(torch.randn(C, D) * 0.01)
   325	            self.out = nn.Linear(C * D, self.out_dim)
   326	        elif self.mode == 'attn':
   327	            H = int(heads)
   328	            assert D % H == 0, f"--ovli_attn_heads {H} must divide --ovli_dim {D}"
   329	            self.heads = H
   330	            self.head_dim = D // H
   331	            self.query = nn.Parameter(torch.randn(H, self.head_dim) * 0.02)
   332	            self.out = nn.Linear(D, self.out_dim)
   333	        elif self.mode == 'gated':
   334	            h = int(gate_hidden) if gate_hidden else max(8, D // 4)
   335	            self.gate = nn.Sequential(nn.Linear(D, h), nn.GELU(), nn.Linear(h, 1))
   336	            # identity when out_dim == D so the gate stays truly lightweight
   337	            self.out = nn.Linear(D, self.out_dim) if self.out_dim != D else nn.Identity()
   338	        elif self.mode == 'secondorder':
   339	            r = int(lowrank)
   340	            self.lowrank = r
   341	            self.reduce = nn.Linear(D, r, bias=False)
   342	            self.out = nn.Linear(r * r, self.out_dim)
   343	        else:
   344	            raise ValueError(f"unknown --ovli_setpool {self.mode}")
   345	        # kaiming init for the Linear layers (matches OVLIHead.proj convention);
   346	        # the raw Parameters (centers / query) keep their small-random init above.
   347	        self.apply(weights_init_kaiming)
   348	        # Zero-init scalar gate on the residual.  At init `g == 0` -> the residual
   349	        # contributes exactly 0 -> `forward == mean_k(tok)` byte-for-byte (== the
   350	        # 52.37 avg/mean-pool path), REGARDLESS of how `self.out` was initialized.
   351	        # Step-0 gradients: d pooled / d g = residual(tok) (generally NON-zero, so
   352	        # the gate moves off 0); d pooled / d theta = g * d residual / d theta = 0
   353	        # because g == 0.  So at step 0 ONLY the gate receives gradient -- the
   354	        # residual module's own params (centers / query / Linear / gate MLP) have
   355	        # ZERO gradient and do NOT update; they only start training from step 1
   356	        # onward, once g != 0 makes their gradient non-zero.  A scalar gate is used
   357	        # (not relying on zero-init `self.out`) because it makes the "lossless
   358	        # start" provable to bytes and independent of each branch's internal init.
   359	        self.gate_res = nn.Parameter(torch.zeros(1)) if self.residual else None
   360	

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '360,760p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   360	
   361	    # -- per-mode permutation-invariant residual (the learnable correction) ---- #
   362	    def _residual(self, tok):
   363	        """tok:(B,K,D) L2-normed fp32 -> (B,out_dim) residual, permutation-invariant
   364	        in the K (token) axis (every branch collapses K by an order-invariant op).
   365	        """
   366	        B, K, D = tok.shape
   367	        if self.mode == 'netvlad':
   368	            a = torch.softmax(self.assign(tok), dim=2)            # (B,K,C) over C
   369	            res = tok.unsqueeze(2) - self.centers.view(1, 1, -1, D)  # (B,K,C,D)
   370	            vlad = (a.unsqueeze(3) * res).sum(dim=1)              # (B,C,D) sum over K
   371	            vlad = F.normalize(vlad, dim=2)                       # intra-norm /cluster
   372	            vlad = vlad.reshape(B, -1)                            # (B, C*D)
   373	            vlad = F.normalize(vlad, dim=1)                       # global VLAD L2-norm
   374	            return self.out(vlad)                                 # (B, out_dim)
   375	        if self.mode == 'attn':
   376	            H, Dh = self.heads, self.head_dim
   377	            t = tok.reshape(B, K, H, Dh)                          # (B,K,H,Dh)
   378	            scores = (t * self.query.view(1, 1, H, Dh)).sum(-1) / math.sqrt(Dh)
   379	            a = torch.softmax(scores, dim=1)                      # (B,K,H) over K
   380	            out = (a.unsqueeze(-1) * t).sum(dim=1)                # (B,H,Dh) sum over K
   381	            return self.out(out.reshape(B, H * Dh))               # (B, out_dim)
   382	        if self.mode == 'gated':
   383	            g = torch.sigmoid(self.gate(tok))                     # (B,K,1) in (0,1)
   384	            num = (g * tok).sum(dim=1)                            # (B,D) sum over K
   385	            den = g.sum(dim=1).clamp(min=self.eps)                # (B,1) >0 -> NaN-safe
   386	            return self.out(num / den)                            # gated convex mean
   387	        # secondorder: order-invariant low-rank token covariance
   388	        z = self.reduce(tok)                                      # (B,K,r)
   389	        cov = torch.einsum('bkr,bks->brs', z, z) / K             # (B,r,r) sum over K
   390	        cov = torch.sign(cov) * torch.sqrt(cov.abs() + self.eps)  # signed-sqrt, NaN-safe
   391	        return self.out(cov.reshape(B, -1))                       # (B, r*r) -> out_dim
   392	
   393	    def forward(self, tok):
   394	        """tok:(B,K,D) L2-normed fp32 -> (B,out_dim) raw aggregated vector.
   395	
   396	        residual=True (default): `mean_k(tok) + gate_res * residual(tok)` with the
   397	        gate zero-init, so at step 0 the output == the un-normalized mean of the K
   398	        tokens.  aggregate_tokens then returns this raw vector AS-IS (no final
   399	        L2-norm), so its gram is the UN-normalized `mean @ mean.T` == the 52.37
   400	        avg/mean-pool path BYTE-FOR-BYTE.  residual=False: the original standalone
   401	        pooling REPLACES the mean (random-init), and aggregate_tokens L2-normalizes
   402	        it (cosine gram) -> the collapsing control.  Both branches are
   403	        permutation-invariant in K.
   404	        """
   405	        if not self.residual:
   406	            return self._residual(tok)                            # standalone (control)
   407	        # mean + zero-init residual: lossless start from the 52.37 mean-pool.
   408	        m = tok.mean(dim=1)                                        # (B,D) == avg/mean path
   409	        return m + self.gate_res * self._residual(tok)            # (B,out_dim==D)
   410	
   411	
   412	# --------------------------------------------------------------------------- #
   413	# OVLI: Opposite-View Late-Interaction Evidence Alignment
   414	# --------------------------------------------------------------------------- #
   415	class OVLIHead(nn.Module):
   416	    """Token projection + opposite-view late-interaction retrieval loss.
   417	
   418	    Reuses the maxsim_probe token-extraction recipe (hook model.layer4 -> the
   419	    GeM-input spatial map -> adaptive-avg-pool to a grid -> flatten to K local
   420	    tokens), but adds a NEW learnable 1x1-conv/linear projection to `proj_dim`
   421	    + per-token L2-norm.  ** The projection parameters are owned by this module
   422	    and MUST be added to the optimizer ** (this is the structural difference vs
   423	    OVP-Mem, which has no learnable params).
   424	
   425	    There is NO prototype / memory / EMA: the loss is a pure sample-to-sample,
   426	    in-batch, opposite-view supervised-contrastive retrieval objective.
   427	
   428	    Forward never holds a backbone reference; the layer4 map is captured by a
   429	    forward hook that does NOT detach, so gradient flows layer4 -> proj.
   430	    """
   431	
   432	    def __init__(self, model, in_ch=2048, proj_dim=256, grid=(8, 4),
   433	                 alpha=0.5, tau=0.05, pool='mean', topk=8, thresh=0.0,
   434	                 allview=False, match='maxsim', align='free',
   435	                 setpool='mean', vlad_clusters=8, attn_heads=4, so_rank=32,
   436	                 setpool_residual=True):
   437	        super().__init__()
   438	        self.grid = tuple(grid)               # adaptive pool grid (gh, gw)
   439	        self.alpha = float(alpha)             # weight on global cosine in score
   440	        self.tau = float(tau)
   441	        # MaxSim pooling variant over the per-token max scores (dustbin family).
   442	        #   'mean'    : average over ALL token-max scores  (original behaviour).
   443	        #   'topk'    : average only the top-k highest token-max scores (drop the
   444	        #               low-score, non-corresponding tokens -> dustbin approx).
   445	        #   'thresh'  : average only token-max scores above `thresh` (soft floor
   446	        #               fallback if none pass, so a fully-masked pair never NaNs).
   447	        #   'softtopk': logsumexp-softmax weighted mean over token-max scores
   448	        #               (smooth top-k, temperature reuses self.tau).
   449	        self.pool = str(pool)
   450	        self.topk = int(topk)
   451	        self.thresh = float(thresh)
   452	        assert self.pool in ('mean', 'topk', 'thresh', 'softtopk'), \
   453	            f"unknown --ovli_pool {self.pool}"
   454	        # Ablation: candidate view-masking mode for the contrastive loss.
   455	        #   allview=False (default): OPPOSITE-VIEW-ONLY -- per anchor the
   456	        #     candidate set is restricted to opposite-view samples (the headline
   457	        #     cross-view constraint). Reproduces the original behaviour exactly.
   458	        #   allview=True (--ovli_allview): drop the opposite-view constraint --
   459	        #     candidates = ALL other samples (incl. same view); positives =
   460	        #     same-pid any view (excl. self), negatives = other-pid any view.
   461	        #     A plain all-view token-set supervised-contrastive control isolating
   462	        #     "is the cross-view restriction what helps, or just an extra token
   463	        #     loss?". Only the view mask changes; score/MaxSim/pool/tau/proj are
   464	        #     identical across modes.
   465	        self.allview = bool(allview)
   466	        # Ablation: token-token match reduction (the late-interaction selection).
   467	        #   match='maxsim' (default): for each query token take the MAX similarity
   468	        #     over the other token set (ColBERT/MaxSim late-interaction selection).
   469	        #     Reproduces the original behaviour exactly.
   470	        #   match='avg' (--ovli_match avg): replace that per-token MAX with a MEAN
   471	        #     over the other token set -> the token-token similarities are fully
   472	        #     averaged, degenerating to a near-global soft match. Isolates "is the
   473	        #     max selection what makes late-interaction work, vs a soft average?".
   474	        #     ONLY the inner token reduction changes; the bidirectional structure,
   475	        #     the outer pool over query tokens, alpha mixing and the loss are
   476	        #     identical across modes.
   477	        self.match = str(match)
   478	        assert self.match in ('maxsim', 'avg'), \
   479	            f"unknown --ovli_match {self.match}"
   480	        # Ablation: spatial alignment of the late interaction.
   481	        #   align='free' (default): every query token may match ANY token in the
   482	        #     other set (free/global late interaction). Reproduces the original.
   483	        #   align='ordered' (--ovli_align ordered): AlignedReID-style row-ordered
   484	        #     alignment -- the K tokens form a (gh x gw) spatial grid; a query
   485	        #     token in spatial row r is restricted to match ONLY the other-set
   486	        #     tokens in the SAME row r (the simplified monotonic/diagonal cut =
   487	        #     row-correspondence), instead of a free global max. Isolates "free
   488	        #     partial set matching vs ordered body-region alignment". Needs the
   489	        #     per-token row index (flat token k -> row k // gw).
   490	        self.align = str(align)
   491	        assert self.align in ('free', 'ordered'), \
   492	            f"unknown --ovli_align {self.align}"
   493	        # Precompute a (1,K,1,K) row-equality mask used ONLY by align='ordered':
   494	        # entry (ki,kj) is True iff token ki and token kj fall in the same grid
   495	        # row.  Registered as a NON-persistent buffer so it (a) follows .to(device)
   496	        # with the module and (b) is never written into the checkpoint (it is a
   497	        # pure constant of `grid`, and keeping it out keeps old ckpts loadable).
   498	        _gh, _gw = self.grid
   499	        _K = _gh * _gw
   500	        _rows = torch.arange(_K) // _gw                     # (K,) row of each tok
   501	        _row_eq = _rows.view(_K, 1).eq(_rows.view(1, _K))    # (K,K) bool
   502	        self.register_buffer('_row_mask4', _row_eq.view(1, _K, 1, _K),
   503	                             persistent=False)
   504	        # Set-pooling of the K tokens into a single per-sample vector (the
   505	        # headline aggregation step).
   506	        #   setpool='mean' (default): NO new module -- the sym_MaxSim path below
   507	        #     runs verbatim (byte-identical to the original for EVERY match / pool
   508	        #     / align combo).  --ovli_match avg --ovli_pool mean reduces to the
   509	        #     gram of the per-sample MEAN of the K tokens (the current best mech).
   510	        #   setpool in {netvlad, attn, gated, secondorder}: a learnable, order-
   511	        #     invariant OVLISetPool REPLACES the token-set MaxSim entirely -- the
   512	        #     K tokens are aggregated into one vector and the cross-view score is
   513	        #     the gram of those aggregated vectors (residual mode: UN-normalized
   514	        #     `mean(+residual) @ .T` == the 52.37 avg/mean path; standalone mode:
   515	        #     L2-normed cosine gram).  The match / pool / align / topk / thresh
   516	        #     knobs only govern the MaxSim path and
   517	        #     are BYPASSED in this mode.  Its params live under this module, so
   518	        #     list(ovli.parameters()) -> optimizer (same contract as proj).
   519	        self.setpool = str(setpool)
   520	        assert self.setpool in ('mean', 'netvlad', 'attn', 'gated', 'secondorder'), \
   521	            f"unknown --ovli_setpool {self.setpool}"
   522	        # setpool_residual=True (default): each learnable pool is "mean + zero-init
   523	        # residual", so aggregate_tokens starts BYTE-IDENTICAL to the 52.37 mean-
   524	        # pool and only learns a correction (no random-init collapse).  =False:
   525	        # the original standalone pooling (random-init replaces the mean) -- kept
   526	        # only for the standalone-vs-residual ablation.
   527	        self.setpool_residual = bool(setpool_residual)
   528	        # 1x1 conv == per-token linear projection over channels (new params).
   529	        # Build proj BEFORE the set-pool so its weights consume the SAME slice of
   530	        # the RNG stream regardless of setpool (the set-pool draws randn for its
   531	        # centers/query + kaiming-inits its Linear layers, which would otherwise
   532	        # shift proj's init under a fixed seed).  With proj first, setpool != 'mean'
   533	        # and setpool == 'mean' get byte-identical proj weights at the same seed,
   534	        # so the residual start truly matches the 52.37 avg/mean path to the bit.
   535	        self.proj = nn.Conv2d(in_ch, proj_dim, kernel_size=1, bias=True)
   536	        self.proj.apply(weights_init_kaiming)
   537	        self.setpool_mod = None
   538	        if self.setpool != 'mean':
   539	            self.setpool_mod = OVLISetPool(self.setpool, dim=proj_dim,
   540	                                           out_dim=proj_dim, clusters=vlad_clusters,
   541	                                           heads=attn_heads, lowrank=so_rank,
   542	                                           residual=self.setpool_residual)
   543	        # hook the GeM-input map; store WITHOUT detach so grad can flow.
   544	        self._buf = {}
   545	        self._handle = model.layer4.register_forward_hook(self._hook)
   546	
   547	    def _hook(self, module, inp, out):
   548	        # keep the graph (no .detach()) -> proj/loss can backprop into layer4
   549	        self._buf['map'] = out
   550	
   551	    def remove_hook(self):
   552	        if self._handle is not None:
   553	            self._handle.remove()
   554	            self._handle = None
   555	
   556	    # -- token extraction ---------------------------------------------------- #
   557	    def tokens_from_cached_map(self):
   558	        """Project the most-recent layer4 map into L2-normed tokens (B,K,D).
   559	
   560	        Reads the map captured by the hook during the just-run forward.  The
   561	        projection runs in fp32 (numerical safety for the cos/MaxSim/logsumexp
   562	        downstream); tokens are L2-normalized per token.
   563	        """
   564	        fmap = self._buf.get('map', None)
   565	        if fmap is None:
   566	            raise RuntimeError("OVLIHead: no cached layer4 map; run model "
   567	                               "forward before tokens_from_cached_map().")
   568	        # pool to the token grid, then project. Cast to fp32 for stability
   569	        # (the cached map may be fp16 under autocast).
   570	        fmap = F.adaptive_avg_pool2d(fmap.float(), self.grid)      # (B,Dc,gh,gw)
   571	        tok = self.proj(fmap)                                       # (B,proj,gh,gw)
   572	        B, C, H, W = tok.shape
   573	        tok = tok.flatten(2).permute(0, 2, 1).contiguous()          # (B,K,proj)
   574	        tok = F.normalize(tok, dim=2)                               # per-token L2
   575	        return tok
   576	
   577	    # -- learnable set pooling of the K tokens into one vector --------------- #
   578	    def aggregate_tokens(self, tok):
   579	        """Aggregate (B,K,D) L2-normed tokens -> (B,Dout) via the chosen
   580	        permutation-invariant set pooling (OVLISetPool).
   581	
   582	        ONLY called when setpool != 'mean'.  Shared by the train loss and the eval
   583	        rerank so train and test use the IDENTICAL aggregation (train/test
   584	        symmetry).  Runs in fp32.
   585	
   586	        ** Normalization convention (load-bearing for the residual kill-switch) **
   587	        The 52.37 avg/mean path computes its cross-view gram as the UN-normalized
   588	        `mean_k(tok) @ mean_k(tok).T` (the per-sample mean of the K L2-normed
   589	        tokens is NOT re-normalized -- see --ovli_match avg --ovli_pool mean, which
   590	        reduces to exactly that gram).  So:
   591	
   592	          * setpool_residual=True (default): return the RAW aggregated vector
   593	            `mean_k(tok) + gate_res * residual(tok)` WITHOUT a final L2-norm.  At
   594	            gate_res==0 this is exactly `mean_k(tok)`, hence `a @ a.T` is the
   595	            UN-normalized mean gram == the 52.37 avg/mean-pool gram BYTE-FOR-BYTE
   596	            (verified torch.equal / <1e-6 in smoke_ovli_residual.py).  Re-norming
   597	            here would instead give `<normalize(mean), normalize(mean)>` (diag==1),
   598	            which is a DIFFERENT score and would make the kill-switch start off the
   599	            52.37 path.  The residual is what the gate learns to add.
   600	
   601	          * setpool_residual=False (standalone control): the aggregated vector
   602	            REPLACES the mean, so it IS L2-normalized -> the cross-view gram is a
   603	            cosine in [-1,1] (the original standalone convention; this branch is
   604	            the random-init collapse control, not the residual start).
   605	        """
   606	        a = self.setpool_mod(tok)              # (B, Dout) fp32, permutation-invariant
   607	        if self.setpool_residual:
   608	            # raw mean(+residual): UN-normalized gram == 52.37 avg/mean-pool path.
   609	            return a
   610	        return F.normalize(a, dim=1)           # standalone: unit vectors -> cosine gram
   611	
   612	    # -- pooling over per-token max scores (mean / topk / thresh / softtopk) -- #
   613	    @staticmethod
   614	    def pool_token_max(tmax, dim, pool='mean', topk=8, thresh=0.0, tau=0.05):
   615	        """Reduce per-token max scores `tmax` along `dim` into a pair score.
   616	
   617	        `tmax` holds, for each (pair, token) the best (max) match of that token
   618	        into the other token set; this collapses the surviving token axis `dim`
   619	        into one number per pair, using the chosen dustbin variant.
   620	
   621	        * mean     : ordinary average over all token-max scores.
   622	        * topk     : average of the top-min(k, K) highest token-max scores along
   623	                     `dim` (low-score non-corresponding tokens are dropped).
   624	        * thresh   : average of token-max scores > `thresh`; if a pair has none
   625	                     above threshold, fall back to its single max (small but
   626	                     finite) so the result is never NaN.
   627	        * softtopk : softmax(tmax/tau)-weighted mean over token-max scores along
   628	                     `dim` (a smooth, differentiable top-k surrogate).
   629	
   630	        Returns the pair-score tensor with axis `dim` removed.
   631	        """
   632	        K = tmax.size(dim)
   633	        if pool == 'mean':
   634	            return tmax.mean(dim=dim)
   635	        if pool == 'topk':
   636	            k = max(1, min(int(topk), K))
   637	            vals = tmax.topk(k, dim=dim).values                    # (...,k,...)
   638	            return vals.mean(dim=dim)
   639	        if pool == 'thresh':
   640	            mask = tmax > thresh                                    # bool, same shape
   641	            masked = torch.where(mask, tmax, tmax.new_zeros(()))
   642	            cnt = mask.sum(dim=dim)                                 # (...) ints
   643	            summed = masked.sum(dim=dim)                            # (...)
   644	            mean_above = summed / cnt.clamp(min=1).to(tmax.dtype)
   645	            # pairs with no token above threshold -> fall back to the single max
   646	            # (finite, never NaN); broadcast cnt==0 back over the reduced axis.
   647	            fallback = tmax.max(dim=dim).values
   648	            no_valid = cnt == 0
   649	            return torch.where(no_valid, fallback, mean_above)
   650	        if pool == 'softtopk':
   651	            w = torch.softmax(tmax / tau, dim=dim)                  # weights sum 1
   652	            return (w * tmax).sum(dim=dim)
   653	        raise ValueError(f"unknown pool {pool}")
   654	
   655	    # -- per-query-token reduction over the OTHER token set ------------------ #
   656	    def _reduce_other(self, sim, other_dim):
   657	        """Collapse the 'other' token axis `other_dim` of a (.,Kq,.,Kg) similarity
   658	        tensor into one score per query token, honoring the match / align modes.
   659	
   660	        `sim` layout: the two token axes are at dim 1 (query tokens) and dim 3
   661	        (other tokens); `other_dim` is the axis to reduce (3 = query->other,
   662	        1 = other->query).  The (K,K) row mask is symmetric so the same buffer
   663	        serves both directions, keeping sym_MaxSim symmetric.
   664	
   665	        * match='maxsim' (default): max over the other tokens (ColBERT selection).
   666	        * match='avg'             : mean over the other tokens (soft global match).
   667	        * align='free'   (default): reduce over ALL other tokens.
   668	        * align='ordered'         : reduce only over other tokens in the SAME grid
   669	                                    row (row-correspondence).  Masked max uses a
   670	                                    finite floor (never -inf) so an empty row (can
   671	                                    only happen for a degenerate gw=0 grid, not the
   672	                                    fixed gh x gw one) is NaN-safe; masked mean
   673	                                    clamps the per-row count to >=1.
   674	
   675	        Default (maxsim + free) returns exactly `sim.max(dim=other_dim).values`,
   676	        so the off-mode forward is byte-identical to the original.
   677	        """
   678	        if self.align == 'free':
   679	            if self.match == 'maxsim':
   680	                return sim.max(dim=other_dim).values            # original behaviour
   681	            return sim.mean(dim=other_dim)                       # match == 'avg'
   682	        # align == 'ordered': restrict each query token to its same-row others.
   683	        mask4 = self._row_mask4                                  # (1,K,1,K) bool
   684	        if self.match == 'maxsim':
   685	            floor = sim.new_full((), -1e4)                      # finite, NaN-safe
   686	            masked = torch.where(mask4, sim, floor)
   687	            return masked.max(dim=other_dim).values
   688	        # match == 'avg': masked mean over the same-row other tokens.
   689	        cnt = mask4.sum(dim=other_dim).clamp(min=1).to(sim.dtype)
   690	        summed = torch.where(mask4, sim, sim.new_zeros(())).sum(dim=other_dim)
   691	        return summed / cnt
   692	
   693	    # -- symmetric token-set MaxSim (bidirectional, full BxB) ---------------- #
   694	    def sym_maxsim_matrix(self, tok):
   695	        """Bidirectional MaxSim score for every ordered pair in the batch.
   696	
   697	        tok:(B,K,D) L2-normed.  Returns (B,B):
   698	            s(i,j) = 0.5*( pool_u max_s <u_i, s_j> + pool_s max_u <u_i, s_j> )
   699	        where pool_* is the configured dustbin pooling (mean/topk/thresh/
   700	        softtopk).  Symmetric in (i,j) by construction (both directions use the
   701	        same pooling).  Computed densely (B=64, K=32 -> a 64x64x32x32 sim tensor
   702	        ~ 4M floats; fine on-GPU).
   703	
   704	        setpool != 'mean' REPLACES the whole token-set MaxSim with the learnable
   705	        order-invariant aggregation: msim = gram of the per-sample aggregated
   706	        vectors (still symmetric).  In the default residual mode that gram is the
   707	        UN-normalized `mean(+residual) @ .T`, so at gate_res==0 it is byte-equal to
   708	        the --ovli_match avg --ovli_pool mean (52.37) gram; standalone mode uses
   709	        the L2-normed cosine gram.  setpool == 'mean' (default) leaves the original
   710	        MaxSim path below completely untouched (byte-identical for every match /
   711	        pool / align combo -> off-mode reproduction is structural).
   712	        """
   713	        if self.setpool != 'mean':
   714	            a = self.aggregate_tokens(tok)         # (B,Dout) L2-normed
   715	            return a @ a.t()                       # (B,B) cosine gram, symmetric
   716	        B, K, D = tok.shape
   717	        flat = tok.reshape(B * K, D)                                # (B*K, D)
   718	        sim = (flat @ flat.t()).reshape(B, K, B, K)                 # (B,K,B,K)
   719	        # i-token reduced over j-tokens (dim=3; max/avg, free/row) -> pool i-tokens
   720	        i2j_max = self._reduce_other(sim, other_dim=3)            # (B,K,B) over i-tok
   721	        i2j = self.pool_token_max(i2j_max, dim=1, pool=self.pool,
   722	                                  topk=self.topk, thresh=self.thresh,
   723	                                  tau=self.tau)                     # (B,B)
   724	        # j-token reduced over i-tokens (dim=1; max/avg, free/row) -> pool j-tokens
   725	        j2i_max = self._reduce_other(sim, other_dim=1)            # (B,B,K) over j-tok
   726	        j2i = self.pool_token_max(j2i_max, dim=2, pool=self.pool,
   727	                                  topk=self.topk, thresh=self.thresh,
   728	                                  tau=self.tau)                     # (B,B)
   729	        return 0.5 * (i2j + j2i)
   730	
   731	    # -- ACVP: ambiguity-calibrated opposite-view negative relaxation -------- #
   732	    @torch.no_grad()
   733	    def acvp_neg_bias(self, gfeat, labels, views, neg, proto, inited,
   734	                      gamma, wmin, eta, margin):
   735	        """Detached log-weight bias added to the NEGATIVE logits of the OVLI
   736	        contrastive denominator (ACVP).  Returns `(bias, frac, mean_w, n_soft)`:
   737	
   738	          bias:(B,B) fp32 additive term = log(w_ij) on negative (anchor i, cand j)
   739	                     entries and 0 everywhere else.  Added to `cand_logits` ONLY,
   740	                     so the numerator (positives) is untouched -- ACVP softens
   741	                     negatives only.
   742	          frac:      scalar = fraction of SOFTENABLE negative pairs with
   743	                     w_ij < 0.95 (kill-switch: frac > 0.30 => ACVP is broadly
   744	                     weakening negatives = bad).
   745	          mean_w:    scalar = mean w_ij over the SOFTENABLE negative pairs
   746	                     (kill-switch: mean_w < 0.75 => negatives broadly softened
   747	                     = bad).
   748	          n_soft:    scalar long = number of softenable negative pairs (ok.sum()),
   749	                     i.e. opposite-view negatives with BOTH prototypes initialised.
   750	                     The caller weights the per-step (frac, mean_w) stats by this
   751	                     count and SKIPS steps with n_soft==0, so cold-start batches
   752	                     (no initialised prototypes) never bias the per-epoch summary.
   753	
   754	        EVERYTHING here is under torch.no_grad() and uses the DETACHED prototype
   755	        bank, so ACVP injects no gradient into the encoder/proj/prototypes: it is a
   756	        pure, detached re-weighting of the contrastive denominator (an ambiguity
   757	        SENSOR, not a learned alignment).  No-op-safe: pairs whose prototype lookup
   758	        is uninitialised get w_ij == 1 (log 0) -> they contribute 0 bias.
   759	
   760	        delta_ij = cos(z_i, P[y_j, view_j]) - cos(z_i, P[y_i, view_j])

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '760,1120p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   760	        delta_ij = cos(z_i, P[y_j, view_j]) - cos(z_i, P[y_i, view_j])
   761	                   (how much closer negative j's opposite-view identity sits to i
   762	                    than i's own opposite-view identity -> ambiguity of j as a
   763	                    negative; z_i, P[.] L2-normed so cos == dot).
   764	        w_ij     = clamp(1 - gamma * sigmoid((delta_ij - margin)/eta), wmin, 1.0)
   765	        bias_ij  = log(w_ij)   (w_ij >= wmin > 0 -> finite, never -inf/NaN).
   766	        """
   767	        B = gfeat.size(0)
   768	        # work in fp32 for the cos/sigmoid/clamp/log numerics (gfeat may be fp32
   769	        # already; .float() is a no-op then).  proto is L2-normed in the bank.
   770	        z = gfeat.float()                                           # (B,D) L2-normed
   771	        P = proto.float()                                          # (num_pid,2,D) detached
   772	        view_j = views.view(1, B).expand(B, B)                      # (B,B) cand view
   773	        y_i = labels.view(B, 1).expand(B, B)                        # (B,B) anchor pid
   774	        y_j = labels.view(1, B).expand(B, B)                        # (B,B) cand pid
   775	        # gather prototypes P[pid, view_j] for the "i's own opp identity" and
   776	        # "j's identity in j's view" terms.  view_j indexes the SAME (cand) view
   777	        # for both, so both prototypes live in the view the negative occupies.
   778	        flatPV = P.reshape(-1, P.size(-1))                          # (num_pid*2, D)
   779	        idx_self = (y_i * 2 + view_j).reshape(-1)                   # P[y_i, view_j]
   780	        idx_neg = (y_j * 2 + view_j).reshape(-1)                   # P[y_j, view_j]
   781	        proto_self = flatPV[idx_self].reshape(B, B, -1)            # (B,B,D)
   782	        proto_neg = flatPV[idx_neg].reshape(B, B, -1)             # (B,B,D)
   783	        zb = z.view(B, 1, P.size(-1))                               # (B,1,D) anchor i
   784	        cos_self = (zb * proto_self).sum(-1)                       # (B,B) cos(z_i,P[y_i,vj])
   785	        cos_neg = (zb * proto_neg).sum(-1)                        # (B,B) cos(z_i,P[y_j,vj])
   786	        delta = cos_neg - cos_self                                 # (B,B) ambiguity
   787	        s = torch.sigmoid((delta - margin) / eta)                  # (B,B) in (0,1)
   788	        w = torch.clamp(1.0 - gamma * s, min=wmin, max=1.0)        # (B,B) in [wmin,1]
   789	        # validity: only softens a negative pair when BOTH prototype lookups are
   790	        # initialised (never-seen prototype is a zero vector -> meaningless cos);
   791	        # uninitialised -> w forced to 1 (bias 0, no softening).
   792	        flat_inited = inited.reshape(-1).bool()                    # (num_pid*2,)
   793	        valid_self = flat_inited[idx_self].reshape(B, B)
   794	        valid_neg = flat_inited[idx_neg].reshape(B, B)
   795	        ok = neg & valid_self & valid_neg                          # (B,B) softenable negs
   796	        w = torch.where(ok, w, w.new_ones(()))                     # uninit/non-neg -> w=1
   797	        bias = torch.log(w)                                        # (B,B) <=0, finite
   798	        # kill-switch stats over softenable negatives only (where ACVP can act).
   799	        # n_soft = #softenable-neg this step; the caller weights frac/mean_w by it
   800	        # and drops n_soft==0 steps so cold-start batches don't skew the summary.
   801	        n_soft = ok.sum()                                          # scalar long
   802	        if ok.any():
   803	            wv = w[ok]
   804	            frac = (wv < 0.95).float().mean()
   805	            mean_w = wv.mean()
   806	        else:
   807	            frac = w.new_zeros(())
   808	            mean_w = w.new_ones(())
   809	        return bias, frac, mean_w, n_soft
   810	
   811	    # -- loss ---------------------------------------------------------------- #
   812	    def loss(self, gfeat, tok, labels, views,
   813	             acvp_proto=None, acvp_inited=None, acvp_gamma=0.0,
   814	             acvp_wmin=0.3, acvp_eta=0.05, acvp_margin=0.0):
   815	        """Supervised-contrastive late-interaction retrieval loss.
   816	
   817	        gfeat:(B,D) L2-normed global feature (gradient flows -> encoder).
   818	        tok:(B,K,Dp) L2-normed projected tokens (gradient flows -> proj+encoder).
   819	        labels:(B,) long.  views:(B,) long in {0,1}.
   820	
   821	        Per anchor i the candidate set depends on the ablation mode:
   822	          * self.allview=False (default, headline): candidates = opposite-view
   823	            samples (view != view_i); positives = opposite-view same-pid samples.
   824	          * self.allview=True (--ovli_allview control): candidates = ALL other
   825	            samples (any view, excl. self); positives = same-pid any view.
   826	        Multi-positive InfoNCE via logsumexp over the same candidate set. Returns
   827	        (loss, pos_score_mean, neg_score_mean) where the score means are
   828	        diagnostics over the contributing (anchor, candidate) pairs.
   829	
   830	        ACVP (optional, default OFF -> byte-identical to the pre-ACVP loss):
   831	          When `acvp_proto is not None`, the NEGATIVE logits in the contrastive
   832	          DENOMINATOR get a detached ambiguity log-weight `log(w_ij)` added
   833	          (acvp_neg_bias); positives/numerator are untouched.  `acvp_proto` is the
   834	          DETACHED opposite-view prototype bank [num_pid,2,D] (no grad to encoder/
   835	          proto).  Returns the SAME 3-tuple; the per-epoch ACVP kill-switch stats
   836	          (relaxed_neg_frac, mean_w) are stashed on self._acvp_stats for the caller.
   837	          acvp_proto is None reproduces the original loss EXACTLY (the floor /
   838	          where / logsumexp lines are reached unchanged).
   839	        """
   840	        B = gfeat.size(0)
   841	        device = gfeat.device
   842	        # pairwise hybrid score in fp32 (cos in [-1,1], maxsim in [-1,1])
   843	        gsim = gfeat @ gfeat.t()                                   # (B,B)
   844	        msim = self.sym_maxsim_matrix(tok)                         # (B,B)
   845	        score = self.alpha * gsim + (1.0 - self.alpha) * msim       # (B,B)
   846	
   847	        same_view = views.view(-1, 1).eq(views.view(1, -1))         # (B,B)
   848	        same_pid = labels.view(-1, 1).eq(labels.view(1, -1))        # (B,B)
   849	        eye = torch.eye(B, dtype=torch.bool, device=device)
   850	
   851	        # Candidate set depends on the ablation mode (only this mask changes;
   852	        # everything downstream is identical across modes).
   853	        #   allview=False (default): opposite-view-only -> identical to original.
   854	        #   allview=True  (control): all-view -> candidates = every other sample.
   855	        if self.allview:
   856	            cand = ~eye                     # all-view, not self
   857	        else:
   858	            cand = (~same_view) & (~eye)    # opposite-view, not self
   859	        pos = cand & same_pid               # positives: same pid (any cand view)
   860	        neg = cand & (~same_pid)            # negatives: other pid (any cand view)
   861	
   862	        # anchors that have >=1 positive AND >=1 negative in the candidate set
   863	        valid = (pos.sum(dim=1) > 0) & (neg.sum(dim=1) > 0)         # (B,)
   864	        if valid.sum() == 0:
   865	            z = gfeat.new_zeros(())
   866	            if acvp_proto is not None:
   867	                # keep the stats slot defined even on an empty batch (no
   868	                # softenable negatives -> n_soft=0 => the caller skips this step).
   869	                self._acvp_stats = (z.detach(), gfeat.new_ones(()),
   870	                                    gfeat.new_zeros((), dtype=torch.long))
   871	            return z, z, z
   872	
   873	        logits = score / self.tau                                   # (B,B)
   874	        # Use a large finite floor (not -inf) so a fully-masked row can never
   875	        # yield -inf, and so (-inf)-(-inf)=nan can never appear even before the
   876	        # valid-row selection. logsumexp over real candidates dominates this
   877	        # floor by a wide margin (logits ~ [-20,20] at tau=0.05).
   878	        floor = logits.new_full((), -1e4)
   879	        # numerator: logsumexp over positives only; denominator: over all cands
   880	        pos_logits = torch.where(pos, logits, floor)
   881	        cand_logits = torch.where(cand, logits, floor)
   882	        # ACVP: soften the NEGATIVE logits in the denominator with a detached
   883	        # ambiguity log-weight.  acvp_proto is None -> this whole block is skipped
   884	        # and cand_logits is byte-identical to the original (off-mode reproduction).
   885	        if acvp_proto is not None:
   886	            acvp_bias, acvp_frac, acvp_mean_w, acvp_nsoft = self.acvp_neg_bias(
   887	                gfeat, labels, views, neg, acvp_proto, acvp_inited,
   888	                acvp_gamma, acvp_wmin, acvp_eta, acvp_margin)
   889	            # log(w_ij)<=0 added to negative entries ONLY (bias is 0 elsewhere and
   890	            # on uninitialised/non-neg pairs).  Detached -> no grad to proto/feat.
   891	            cand_logits = cand_logits + acvp_bias.to(cand_logits.dtype)
   892	            # stash (frac, mean_w, #softenable-neg) for the caller's per-epoch
   893	            # kill-switch summary (weighted by #softenable-neg, not batch size).
   894	            self._acvp_stats = (acvp_frac.detach(), acvp_mean_w.detach(),
   895	                                acvp_nsoft.detach())
   896	        log_num = torch.logsumexp(pos_logits, dim=1)                # (B,)
   897	        log_den = torch.logsumexp(cand_logits, dim=1)               # (B,)
   898	        per_anchor = -(log_num - log_den)                           # (B,)
   899	        # only anchors with >=1 positive AND >=1 negative in the candidate set
   900	        loss = per_anchor[valid].mean()
   901	
   902	        # diagnostics (detached): mean positive / negative pair scores
   903	        with torch.no_grad():
   904	            ps = score[pos].mean() if pos.any() else score.new_zeros(())
   905	            ns = score[neg].mean() if neg.any() else score.new_zeros(())
   906	        return loss, ps, ns
   907	
   908	
   909	# --------------------------------------------------------------------------- #
   910	# OVLI rerank: eval-time global + sym_MaxSim rerank (opt-in, symmetric w/ train)
   911	# --------------------------------------------------------------------------- #
   912	@torch.no_grad()
   913	def ovli_rerank_eval(model, ovli, dataset, args, device):
   914	    """Report A->G / G->A mAP/R1 for (a) global-only and (b) global+MaxSim
   915	    rerank, using the SAME projected tokens + sym MaxSim as the training loss.
   916	
   917	    Mirrors run_cross_view_eval but additionally extracts projected tokens via
   918	    the OVLI hook and reranks by score = alpha*cos(global) + (1-alpha)*MaxSim.
   919	    Gallery token sets can be large, so MaxSim is chunked over the gallery axis.
   920	    Returns {tag: {'global': (mAP,R1), 'rerank': (mAP,R1)}}.
   921	    """
   922	    from cargo_dataset import filter_by_view as _fbv
   923	
   924	    model.eval()
   925	
   926	    @torch.no_grad()
   927	    def extract(samples):
   928	        from afd_train import build_eval_loader as _bel
   929	        loader = _bel(samples, args)
   930	        gfs, tks, pids, cams = [], [], [], []
   931	        view_map = {'Aerial': 0, 'Ground': 1}
   932	        for batch in loader:
   933	            imgs = batch['img'].to(device, non_blocking=True)
   934	            vidx = (torch.tensor([view_map[v] for v in batch['view']],
   935	                                 device=device) if args.use_afd else None)
   936	            gf = model(imgs, view_idx=vidx)              # (b,D) L2-normed BN
   937	            tok = ovli.tokens_from_cached_map()           # (b,K,Dp) L2-normed
   938	            gfs.append(gf.cpu())
   939	            tks.append(tok.cpu())
   940	            pids.append(batch['pid'])
   941	            cams.append(batch['camid'])
   942	        if not gfs:
   943	            return (torch.empty(0), torch.empty(0),
   944	                    np.empty(0, np.int64), np.empty(0, np.int64))
   945	        return (torch.cat(gfs, 0), torch.cat(tks, 0),
   946	                torch.cat(pids, 0).numpy(), torch.cat(cams, 0).numpy())
   947	
   948	    # eval rerank uses the SAME dustbin pooling as the train loss so train/test
   949	    # stay symmetric (mean reproduces the original rerank exactly).
   950	    _pool = ovli.pool
   951	    _topk = ovli.topk
   952	    _thresh = ovli.thresh
   953	    _tau = ovli.tau
   954	
   955	    @torch.no_grad()
   956	    def maxsim_block(qt, gt):
   957	        """(Nq,Ng) bidirectional MaxSim, chunked over the gallery axis."""
   958	        # setpool != 'mean': the cross-view score is the gram of the learnable
   959	        # aggregated vectors (train/test symmetric via the SAME aggregate_tokens
   960	        # used by the train loss -> residual mode = UN-normalized mean(+residual)
   961	        # gram == 52.37 avg/mean path at gate_res==0; standalone = cosine gram),
   962	        # NOT the token-set MaxSim.  Aggregate query/gallery tokens in sample-row
   963	        # blocks (bounds the netvlad (N,K,C,D) intermediate) on-device, gram on CPU.
   964	        if ovli.setpool != 'mean':
   965	            def _agg_all(t):
   966	                outs = []
   967	                for s in range(0, t.size(0), 256):
   968	                    outs.append(ovli.aggregate_tokens(t[s:s + 256].to(device)).cpu())
   969	                return (torch.cat(outs, 0) if outs
   970	                        else torch.empty(0, ovli.setpool_mod.out_dim))
   971	            return _agg_all(qt) @ _agg_all(gt).t()
   972	        Nq, Kq, C = qt.shape
   973	        Ng, Kg, _ = gt.shape
   974	        qd = qt.to(device).reshape(Nq * Kq, C)
   975	        budget = 80_000_000
   976	        per_g = max(1, Nq * Kq * Kg)
   977	        gblk = max(1, min(Ng, budget // per_g))
   978	        out = torch.empty(Nq, Ng)
   979	        for s in range(0, Ng, gblk):
   980	            e = min(s + gblk, Ng)
   981	            gc = gt[s:e].to(device)
   982	            g = gc.size(0)
   983	            sim = (qd @ gc.reshape(g * Kg, C).t()).reshape(Nq, Kq, g, Kg)
   984	            # same match/align reduction as the train loss (train/test symmetry):
   985	            # ovli._reduce_other honors --ovli_match and --ovli_align identically.
   986	            q2g_max = ovli._reduce_other(sim, other_dim=3)  # (Nq,Kq,g) over q-tok
   987	            q2g = OVLIHead.pool_token_max(q2g_max, dim=1, pool=_pool,
   988	                                          topk=_topk, thresh=_thresh, tau=_tau)
   989	            g2q_max = ovli._reduce_other(sim, other_dim=1)  # (Nq,g,Kg) over g-tok
   990	            g2q = OVLIHead.pool_token_max(g2q_max, dim=2, pool=_pool,
   991	                                          topk=_topk, thresh=_thresh, tau=_tau)
   992	            out[:, s:e] = (0.5 * (q2g + g2q)).cpu()
   993	            del sim, q2g_max, q2g, g2q_max, g2q, gc
   994	        del qd
   995	        if device == 'cuda':
   996	            torch.cuda.empty_cache()
   997	        return out
   998	
   999	    from afd_train import eval_market
  1000	    results = {}
  1001	    splits = {
  1002	        'A->G': (_fbv(dataset.query, 'Aerial'), _fbv(dataset.gallery, 'Ground')),
  1003	        'G->A': (_fbv(dataset.query, 'Ground'), _fbv(dataset.gallery, 'Aerial')),
  1004	    }
  1005	    for tag, (q, g) in splits.items():
  1006	        qf, qt, qp, qc = extract(q)
  1007	        gf, gt, gp, gc = extract(g)
  1008	        if qf.numel() == 0 or gf.numel() == 0:
  1009	            results[tag] = {'global': (float('nan'), float('nan')),
  1010	                            'rerank': (float('nan'), float('nan'))}
  1011	            continue
  1012	        qf = F.normalize(qf, dim=1)
  1013	        gf = F.normalize(gf, dim=1)
  1014	        gsim = (qf @ gf.t()).numpy()                      # (Nq,Ng) cosine
  1015	        # global-only (rank by cosine distance == -gsim)
  1016	        gmap, gcmc, _ = eval_market(qf, qp, qc, gf, gp, gc)
  1017	        # rerank: alpha*cos + (1-alpha)*MaxSim, rank by descending hybrid
  1018	        msim = maxsim_block(qt, gt).numpy()
  1019	        hyb = args.ovli_alpha * gsim + (1.0 - args.ovli_alpha) * msim
  1020	        from maxsim_probe import eval_from_distmat
  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
  1022	        results[tag] = {
  1023	            'global': (gmap * 100, gcmc[0] * 100),
  1024	            'rerank': (rmap, rr1),
  1025	        }
  1026	    return results
  1027	
  1028	
  1029	def airl_dualbranch_eval(model, dataset, args, device):
  1030	    """AIRL dual-branch eval: extract BOTH heads (f_full, f_rec) in ONE forward
  1031	    and report f_full-only, f_rec-only, and the SOFT-FUSED cosine ranking
  1032	    (cos = w*cos_rec + (1-w)*cos_full, w = args.airl_fuse_w, fixed) for A->G and
  1033	    G->A.  This is the single-model analog of the kill-switch #3 two-model score
  1034	    fusion: cos_rec replaces the AIRL-model cosine, cos_full replaces the
  1035	    baseline-model cosine, and they share ONE backbone forward.
  1036	
  1037	    Mirrors run_cross_view_eval / ovli_rerank_eval exactly for the per-split
  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
  1039	    f_full number reproduces run_cross_view_eval's A<->G mAP bit-for-bit (same
  1040	    feature, same ranking) and the fusion is a pure distance-matrix combination.
  1041	    Returns {tag: {'full': (mAP,R1), 'rec': (mAP,R1), 'fuse': (mAP,R1)}}.
  1042	    """
  1043	    from cargo_dataset import filter_by_view as _fbv
  1044	    from afd_train import build_eval_loader as _bel
  1045	    from maxsim_probe import eval_from_distmat
  1046	
  1047	    model.eval()
  1048	    view_map = {'Aerial': 0, 'Ground': 1}
  1049	
  1050	    @torch.no_grad()
  1051	    def extract(samples):
  1052	        loader = _bel(samples, args)
  1053	        ffs, frs, pids, cams = [], [], [], []
  1054	        for batch in loader:
  1055	            imgs = batch['img'].to(device, non_blocking=True)
  1056	            vidx = (torch.tensor([view_map[v] for v in batch['view']],
  1057	                                 device=device) if args.use_afd else None)
  1058	            # ONE forward -> two L2-normalized features (f_full, f_rec).
  1059	            f_full, f_rec = model(imgs, view_idx=vidx, return_dual=True)
  1060	            ffs.append(f_full.cpu())
  1061	            frs.append(f_rec.cpu())
  1062	            pids.append(batch['pid'])
  1063	            cams.append(batch['camid'])
  1064	        if not ffs:
  1065	            empty = (torch.empty(0), torch.empty(0),
  1066	                     np.empty(0, np.int64), np.empty(0, np.int64))
  1067	            return empty
  1068	        return (torch.cat(ffs, 0), torch.cat(frs, 0),
  1069	                torch.cat(pids, 0).numpy(), torch.cat(cams, 0).numpy())
  1070	
  1071	    w = args.airl_fuse_w
  1072	    results = {}
  1073	    splits = {
  1074	        'A->G': (_fbv(dataset.query, 'Aerial'), _fbv(dataset.gallery, 'Ground')),
  1075	        'G->A': (_fbv(dataset.query, 'Ground'), _fbv(dataset.gallery, 'Aerial')),
  1076	    }
  1077	    for tag, (q, g) in splits.items():
  1078	        q_full, q_rec, qp, qc = extract(q)
  1079	        g_full, g_rec, gp, gc = extract(g)
  1080	        if q_full.numel() == 0 or g_full.numel() == 0:
  1081	            nan2 = (float('nan'), float('nan'))
  1082	            results[tag] = {'full': nan2, 'rec': nan2, 'fuse': nan2}
  1083	            continue
  1084	        # features are already L2-normalized at eval; renormalize defensively so
  1085	        # the cosine == the gram of unit vectors (matches eval_market exactly).
  1086	        q_full = F.normalize(q_full, dim=1); g_full = F.normalize(g_full, dim=1)
  1087	        q_rec = F.normalize(q_rec, dim=1);   g_rec = F.normalize(g_rec, dim=1)
  1088	        s_full = (q_full @ g_full.t()).numpy()        # (Nq,Ng) cosine, f_full
  1089	        s_rec = (q_rec @ g_rec.t()).numpy()           # (Nq,Ng) cosine, f_rec
  1090	        # soft fusion: cos = w*cos_rec + (1-w)*cos_full -> distance = 2 - 2*cos
  1091	        # (identical to kill-switch #3 GATE 5; cosine in [-1,1] -> dist in [0,4]).
  1092	        dm_full = (2.0 - 2.0 * s_full)
  1093	        dm_rec = (2.0 - 2.0 * s_rec)
  1094	        dm_fuse = (2.0 - 2.0 * (w * s_rec + (1.0 - w) * s_full))
  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
  1098	        results[tag] = {'full': (fmap, fr1), 'rec': (rmap, rr1),
  1099	                        'fuse': (zmap, zr1)}
  1100	    return results
  1101	
  1102	
  1103	# --------------------------------------------------------------------------- #
  1104	# AIRL: Aerial Identity Recoverability Learning (resolution-degradation
  1105	#       consistency).  kill-switch #2.
  1106	# --------------------------------------------------------------------------- #
  1107	# Motivation (new_angle_AIRL.md, kill-switch #1 PASS): on CARGO the aerial->ground
  1108	# error is dominated by the AERIAL crop's low PIXEL BUDGET (small bbox -> identity
  1109	# physically under-resolved), NOT only a view-alignment gap.  The zero-training
  1110	# bucketed diagnostic showed the lowest aerial-scale bucket collapses by +13~19 mAP
  1111	# vs the top bucket -- on the STRONG Swin baseline too, so it is a physical pixel
  1112	# problem, not a backbone-headroom artifact (the OVLI failure mode).
  1113	#
  1114	# AIRL turns that diagnostic into a training signal WITHOUT any cross-view
  1115	# contrastive / late-interaction / pooling / prototype machinery (those are the
  1116	# OVLI dead zone).  Mechanism = resolution-degradation CONSISTENCY:
  1117	#   1. For each GROUND image (high-res) sample a "pixel budget" = a scale ratio
  1118	#      drawn to MATCH the aerial bbox scale distribution (small aerial buckets ->
  1119	#      heavy degradation), degrade the image to that budget (bilinear down then
  1120	#      up back to the original H x W, + optional light avg-pool blur), simulating

codex
训练文件里有很多历史机制，但 AIRL 相关函数集中在 `airl_dualbranch_eval`、`airl_degrade`、`airl_consistency_loss`、arg 校验和主训练循环。融合用的是 CPU 上的 numpy distmat 组合，不是模型 forward 里做特征加权。
exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1120,1240p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
  1120	#      up back to the original H x W, + optional light avg-pool blur), simulating
  1121	#      "if this person were shot from a UAV, how much information would survive".
  1122	#   2. Both the original and the degraded image pass the SAME backbone (shared
  1123	#      weights, one extra forward); a PREDICTION-CONSISTENCY loss forces the
  1124	#      degraded view's identity prediction to agree with the original
  1125	#      (KL on logits, or cosine/MSE on the L2-normed BNNeck feature).  Intuition:
  1126	#      learn identity evidence that is STABLE under a low pixel budget; suppress
  1127	#      reliance on ground-only high-frequency detail.
  1128	#   3. total = CE + triplet + airl_lambda_eff * consistency.
  1129	#
  1130	# Design contract (hard):
  1131	#   * NO new learnable parameters -- degradation is an image-space augmentation,
  1132	#     consistency is a loss.  The optimizer / param groups are untouched.
  1133	#   * --airl OFF (default) => NO degradation, NO extra forward, NO loss term =>
  1134	#     the baseline is reproduced BYTE-FOR-BYTE (the whole AIRL block is skipped).
  1135	#   * The consistency loss runs in TRUE fp32 (autocast disabled) for KL/cosine
  1136	#     numeric safety (finite inputs: logits/features from a finite forward).
  1137	#   * AIRL is a TRAIN-time loss only; eval is unchanged (train/test symmetric).
  1138	#   * Backbone-agnostic: the degradation is purely in image space, so resnet50 and
  1139	#     swin_small are both supported (the second forward just reuses `model`).
  1140	def airl_degrade(imgs, min_scale, blur=False, generator=None):
  1141	    """Resolution-degrade a NORMALIZED image batch to a sampled pixel budget.
  1142	
  1143	    imgs:(B,C,H,W) the model-input batch (already Resize+Normalized by the
  1144	    dataloader; degradation is a linear resample in normalized space, which is a
  1145	    faithful resolution/low-pass proxy -- it only removes high-frequency detail
  1146	    and never shifts the per-channel statistics the backbone expects).
  1147	
  1148	    Per image a scale ratio s ~ U[min_scale, 1.0] is drawn (the "pixel budget":
  1149	    s=1 keeps full resolution, s=min_scale is the heaviest aerial-small-bucket
  1150	    degradation).  The image is bilinearly DOWN-sampled to (round(s*H), round(s*W))
  1151	    (>=1 px) then bilinearly UP-sampled back to (H, W), so the output keeps the
  1152	    original shape but only carries ~s*100% of the spatial detail.  Optionally a
  1153	    light 3x3 avg-pool blur (stride 1, reflect pad) is applied AFTER the up-sample
  1154	    to mimic UAV optical blur without any PIL/cv2 dependency.
  1155	
  1156	    Runs in fp32 on the input device; antialias=True for a clean low-pass on the
  1157	    down step.  Returns (degraded:(B,C,H,W), scales:(B,) the per-image s used) so
  1158	    the caller can log deg_scale_mean.  Per-image scales => per-image target sizes,
  1159	    so the resample is done one image at a time (B is small, bs<=64).
  1160	    """
  1161	    B, C, H, W = imgs.shape
  1162	    x = imgs.float()
  1163	    if generator is not None:
  1164	        s = (torch.rand(B, generator=generator, device='cpu')
  1165	             .to(x.device) * (1.0 - min_scale) + min_scale)
  1166	    else:
  1167	        s = torch.rand(B, device=x.device) * (1.0 - min_scale) + min_scale
  1168	    out = torch.empty_like(x)
  1169	    for i in range(B):
  1170	        si = float(s[i])
  1171	        th = max(1, int(round(si * H)))
  1172	        tw = max(1, int(round(si * W)))
  1173	        if th >= H and tw >= W:
  1174	            # s rounds to full size -> no spatial detail removed (still pass blur
  1175	            # below if requested so the op is uniform).
  1176	            yi = x[i:i + 1]
  1177	        else:
  1178	            down = F.interpolate(x[i:i + 1], size=(th, tw), mode='bilinear',
  1179	                                 align_corners=False, antialias=True)
  1180	            yi = F.interpolate(down, size=(H, W), mode='bilinear',
  1181	                               align_corners=False)
  1182	        out[i] = yi[0]
  1183	    if blur:
  1184	        # 3x3 average blur (stride 1) with reflect padding -> shape preserved,
  1185	        # NaN-safe (pure local mean), no learnable params.
  1186	        out = F.avg_pool2d(F.pad(out, (1, 1, 1, 1), mode='reflect'),
  1187	                           kernel_size=3, stride=1)
  1188	    return out, s
  1189	
  1190	
  1191	def airl_consistency_loss(logits_o, bn_o, logits_d, bn_d, mode='kl', tau=4.0):
  1192	    """Prediction-consistency between the ORIGINAL and DEGRADED views (fp32).
  1193	
  1194	    The degraded view must keep its identity prediction CLOSE to the original's,
  1195	    so the model learns evidence that survives a low pixel budget.  The ORIGINAL
  1196	    side is DETACHED (it is the stable target; gradient flows only through the
  1197	    degraded branch -> the model is pulled to make the degraded prediction match
  1198	    the clean one, not the reverse).
  1199	
  1200	      mode='kl' (default): symmetric KL would double-count; we use the standard
  1201	        distillation direction KL(softmax(logits_o/tau).detach || softmax(
  1202	        logits_d/tau)) * tau^2 (temperature-scaled soft-target consistency on the
  1203	        ID logits).  fp32 for numeric safety; log_softmax avoids log(0).
  1204	      mode='feat': 1 - cos(bn_o.detach, bn_d) on the L2-normed BNNeck feature
  1205	        (MSE-equivalent up to scale on unit vectors; bounded in [0,2]).
  1206	
  1207	    Returns a finite non-negative scalar.  A light finite guard (nan_to_num) is
  1208	    applied to the scalar: inputs are finite under normal training, so this is a
  1209	    no-op there, but it keeps the documented "finite scalar" contract honest if a
  1210	    pathological forward ever produced inf/NaN logits.
  1211	    """
  1212	    if mode == 'feat':
  1213	        zo = F.normalize(bn_o.float(), dim=1).detach()
  1214	        zd = F.normalize(bn_d.float(), dim=1)
  1215	        # 1 - cosine in [0,2]; mean over batch.  (== 0.5*||zo-zd||^2 on unit vecs.)
  1216	        out = (1.0 - (zo * zd).sum(dim=1)).mean()
  1217	        return torch.nan_to_num(out)
  1218	    # mode == 'kl': temperature-scaled soft-target KL (clean = detached target).
  1219	    lo = logits_o.float() / tau
  1220	    ld = logits_d.float() / tau
  1221	    p_o = F.softmax(lo, dim=1).detach()                       # stable target
  1222	    log_p_d = F.log_softmax(ld, dim=1)                        # fp32; no log(0)
  1223	    # KL(p_o || p_d) = sum p_o (log p_o - log p_d); * tau^2 keeps the gradient
  1224	    # magnitude comparable across temperatures (Hinton distillation convention).
  1225	    log_p_o = F.log_softmax(lo, dim=1).detach()
  1226	    kl = (p_o * (log_p_o - log_p_d)).sum(dim=1).mean()
  1227	    return torch.nan_to_num(kl * (tau * tau))
  1228	
  1229	
  1230	# --------------------------------------------------------------------------- #
  1231	# train
  1232	# --------------------------------------------------------------------------- #
  1233	def main():
  1234	    ap = argparse.ArgumentParser()
  1235	    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
  1236	    # 2nd benchmark. 'cargo' (default) = byte-identical CARGO path.
  1237	    # 'agreid'    = AG-ReID.v2 binary aerial<->ground merge (NON-official, legacy).
  1238	    # 'agreid_v2' = AG-ReID.v2 OFFICIAL protocols: A->G == exp1 aerial_to_cctv,
  1239	    #               G->A == exp4 cctv_to_aerial, mean of the two (the analogue of
  1240	    #               CARGO's A<->G mean). This is the cross-dataset main-table column.

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1233,1705p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
  1233	def main():
  1234	    ap = argparse.ArgumentParser()
  1235	    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
  1236	    # 2nd benchmark. 'cargo' (default) = byte-identical CARGO path.
  1237	    # 'agreid'    = AG-ReID.v2 binary aerial<->ground merge (NON-official, legacy).
  1238	    # 'agreid_v2' = AG-ReID.v2 OFFICIAL protocols: A->G == exp1 aerial_to_cctv,
  1239	    #               G->A == exp4 cctv_to_aerial, mean of the two (the analogue of
  1240	    #               CARGO's A<->G mean). This is the cross-dataset main-table column.
  1241	    ap.add_argument('--dataset', default='cargo',
  1242	                    choices=['cargo', 'agreid', 'agreid_v2'])
  1243	    ap.add_argument('--out_dir', default='./log/cargo/cvpb_ovp')
  1244	    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
  1245	    ap.add_argument('--epochs', type=int, default=60)
  1246	    ap.add_argument('--warmup_epochs', type=int, default=10)
  1247	    ap.add_argument('--lr', type=float, default=3.5e-4)
  1248	    ap.add_argument('--weight_decay', type=float, default=5e-4)
  1249	    ap.add_argument('--P', type=int, default=16)
  1250	    ap.add_argument('--K', type=int, default=4)
  1251	    ap.add_argument('--test_batch', type=int, default=128)
  1252	    ap.add_argument('--workers', type=int, default=8)
  1253	    ap.add_argument('--margin', type=float, default=0.3)
  1254	    ap.add_argument('--label_smooth', type=float, default=0.1)
  1255	    ap.add_argument('--eval_period', type=int, default=10)
  1256	    ap.add_argument('--seed', type=int, default=1)
  1257	    ap.add_argument('--no_amp', action='store_true', help='disable mixed precision')
  1258	    # model switches (keep AFD off by default -> pure BoT baseline + OVP)
  1259	    ap.add_argument('--last_stride', type=int, default=1)
  1260	    ap.add_argument('--pool', default='gem', choices=['gem', 'avg'])
  1261	    # backbone selector. 'resnet50' (default) = the existing BoT baseline
  1262	    # (IMAGENET1K_V1 + GeM + BNNeck), byte-for-byte unchanged (52.37 headline).
  1263	    # 'swin_small' = SOLIDER Swin-Small (team asset, SOTA push): SOLIDER teacher
  1264	    # pretrain + avg-pool + BNNeck, in_planes=768; AFD freq modules are NOT
  1265	    # supported on swin (OVP/OVLI are independent and DO work). OVLI hooks the
  1266	    # last-stage spatial map (B,768,8,4 @ 256x128) -> same hook contract.
  1267	    ap.add_argument('--backbone', default='resnet50',
  1268	                    choices=['resnet50', 'swin_small'],
  1269	                    help="backbone: resnet50 (default, BoT baseline, byte-identical) "
  1270	                         "or swin_small (SOLIDER Swin-Small, in_planes=768)")
  1271	    ap.add_argument('--swin_pretrain', default='',
  1272	                    help="path to the SOLIDER swin_small.pth teacher checkpoint "
  1273	                         "(e.g. <repo>/pretrained/swin_small.pth). Empty -> train "
  1274	                         "the Swin from trunc-normal init. Only used with "
  1275	                         "--backbone swin_small.")
  1276	    ap.add_argument('--swin_semantic_weight', type=float, default=0.2,
  1277	                    help="SOLIDER semantic weight for the Swin backbone "
  1278	                         "(0.2 = ReID default; <0 disables the semantic embedding). "
  1279	                         "Only used with --backbone swin_small.")
  1280	    ap.add_argument('--swin_lr_factor', type=float, default=0.1,
  1281	                    help="LR multiplier applied to the Swin BACKBONE params only "
  1282	                         "(heads/BNNeck/OVLI stay at full --lr). The resnet50-tuned "
  1283	                         "peak LR (3.5e-4) diverges the Swin transformer (collapse at "
  1284	                         "epoch ~8); 0.1 fine-tunes the backbone gently. Set 1.0 to "
  1285	                         "disable the split. Only used with --backbone swin_small.")
  1286	    ap.add_argument('--use_afd', action='store_true')
  1287	    ap.add_argument('--afd_router', type=int, default=1)
  1288	    ap.add_argument('--afd_cvfc', type=int, default=1)
  1289	    ap.add_argument('--afd_stage', default='layer1',
  1290	                    choices=['stem', 'layer1', 'layer2'])
  1291	    ap.add_argument('--router_cond_view', type=int, default=1)
  1292	    ap.add_argument('--low_r', type=float, default=0.125)
  1293	    ap.add_argument('--mid_r', type=float, default=0.30)
  1294	    ap.add_argument('--high_drop_p', type=float, default=0.5)
  1295	    ap.add_argument('--w_cvfc', type=float, default=0.5)
  1296	    # --- OVP-Mem ---
  1297	    ap.add_argument('--ovp', action='store_true',
  1298	                    help='enable OVP-Mem opposite-view prototype InfoNCE loss')
  1299	    ap.add_argument('--ovp_lambda', type=float, default=0.5,
  1300	                    help='weight of the OVP InfoNCE loss')
  1301	    ap.add_argument('--ovp_tau', type=float, default=0.05,
  1302	                    help='temperature for the OVP InfoNCE logits')
  1303	    ap.add_argument('--ovp_momentum', type=float, default=0.2,
  1304	                    help='EMA momentum = weight on the new batch mean')
  1305	    ap.add_argument('--ovp_warmup', type=int, default=10,
  1306	                    help='H1 fix: warmup OVP lambda linearly over this many epochs')
  1307	    # --- OVLI (headline: opposite-view late-interaction retrieval) ---
  1308	    ap.add_argument('--ovli', action='store_true',
  1309	                    help='enable OVLI opposite-view late-interaction retrieval loss')
  1310	    ap.add_argument('--ovli_lambda', type=float, default=0.5,
  1311	                    help='weight of the OVLI retrieval loss')
  1312	    ap.add_argument('--ovli_tau', type=float, default=0.05,
  1313	                    help='temperature for the OVLI supervised-contrastive logits')
  1314	    ap.add_argument('--ovli_alpha', type=float, default=0.5,
  1315	                    help='score = alpha*cos(global) + (1-alpha)*sym_MaxSim(tokens)')
  1316	    ap.add_argument('--ovli_dim', type=int, default=256,
  1317	                    help='token projection output dim (new learnable params)')
  1318	    ap.add_argument('--ovli_grid', type=int, nargs=2, default=[8, 4],
  1319	                    help='adaptive-pool token grid (gh gw); K = gh*gw tokens')
  1320	    ap.add_argument('--ovli_warmup', type=int, default=10,
  1321	                    help='H1 lesson: warmup OVLI lambda linearly over this many epochs')
  1322	    ap.add_argument('--ovli_rerank', action='store_true',
  1323	                    help='additionally report global+MaxSim rerank at eval time')
  1324	    # MaxSim pooling variant (dustbin / sparse evidence routing). 'mean' = the
  1325	    # original behaviour (average over ALL token-max scores, back-compatible).
  1326	    ap.add_argument('--ovli_pool', default='mean',
  1327	                    choices=['mean', 'topk', 'thresh', 'softtopk'],
  1328	                    help="MaxSim pooling over per-token max scores: mean (all "
  1329	                         "tokens, original), topk (avg of top-k highest -> drop "
  1330	                         "non-corresponding tokens = dustbin approx), thresh "
  1331	                         "(avg of token-max > theta, fall back to single max), "
  1332	                         "softtopk (softmax(tau)-weighted mean = smooth top-k)")
  1333	    ap.add_argument('--ovli_topk', type=int, default=8,
  1334	                    help='k for --ovli_pool topk (clamped to [1, K] tokens)')
  1335	    ap.add_argument('--ovli_thresh', type=float, default=0.0,
  1336	                    help='theta for --ovli_pool thresh (token-max score floor)')
  1337	    # Ablation control for the headline opposite-view-only claim.
  1338	    ap.add_argument('--ovli_allview', action='store_true',
  1339	                    help='ABLATION: drop the opposite-view-only constraint in '
  1340	                         'the OVLI loss -> candidates become ALL other samples '
  1341	                         '(positives = same-pid any view excl. self, negatives '
  1342	                         '= other-pid any view) = plain all-view token-set '
  1343	                         'supervised-contrastive. Default OFF reproduces the '
  1344	                         'headline opposite-view-only behaviour exactly. Tests '
  1345	                         'whether the cross-view restriction (not just an extra '
  1346	                         'token loss) is what helps. score/MaxSim/pool/tau/'
  1347	                         'warmup/proj are unchanged.')
  1348	    # Ablation 1: late-interaction token-match reduction (max vs avg).
  1349	    ap.add_argument('--ovli_match', default='maxsim', choices=['maxsim', 'avg'],
  1350	                    help="ABLATION: token-token match reduction. maxsim (default) "
  1351	                         "= for each query token take the MAX similarity over the "
  1352	                         "other token set (ColBERT/late-interaction selection, "
  1353	                         "original). avg = replace that max with a MEAN over the "
  1354	                         "other token set -> the token-token similarities are "
  1355	                         "fully averaged = near-global soft match. Isolates "
  1356	                         "whether the MAX selection is what makes late interaction "
  1357	                         "work. Only the inner token reduction changes; "
  1358	                         "bidirectional/pool/alpha/loss are unchanged.")
  1359	    # Ablation 2: late-interaction spatial alignment (free vs ordered/AlignedReID).
  1360	    ap.add_argument('--ovli_align', default='free', choices=['free', 'ordered'],
  1361	                    help="ABLATION: late-interaction spatial alignment. free "
  1362	                         "(default) = each query token may match ANY other token "
  1363	                         "(free/global late interaction, original). ordered = "
  1364	                         "AlignedReID-style row-ordered alignment: a query token "
  1365	                         "in grid row r may only match other-set tokens in the "
  1366	                         "SAME row r (row-correspondence / simplified monotonic "
  1367	                         "cut). Isolates free partial set matching vs ordered "
  1368	                         "body-region alignment. Only the inner token reduction "
  1369	                         "changes; bidirectional/pool/alpha/loss are unchanged.")
  1370	    # Headline aggregation: learnable permutation-invariant SET POOLING of the K
  1371	    # projected tokens into one per-sample vector (the cross-view score is then
  1372	    # the cosine gram of those vectors).  This REPLACES the fixed "mean over
  1373	    # tokens" that --ovli_match avg --ovli_pool mean reduces to (the current best
  1374	    # single mechanism, 52.37).  'mean' (default) keeps the existing token-set
  1375	    # MaxSim path verbatim (byte-identical); the four learnable modes BYPASS the
  1376	    # match/pool/align/topk/thresh MaxSim knobs.
  1377	    ap.add_argument('--ovli_setpool', default='mean',
  1378	                    choices=['mean', 'netvlad', 'attn', 'gated', 'secondorder'],
  1379	                    help="learnable permutation-invariant aggregation of the K "
  1380	                         "tokens into one vector. mean (default) = keep the "
  1381	                         "token-set MaxSim path unchanged (byte-identical; "
  1382	                         "--ovli_match avg --ovli_pool mean = gram of mean-pooled "
  1383	                         "tokens = best). netvlad = NetVLAD residual aggregation; "
  1384	                         "attn = multi-head learned-query attention pooling; "
  1385	                         "gated = per-token sigmoid reliability gate convex mean; "
  1386	                         "secondorder = low-rank token covariance pooling. The "
  1387	                         "four learnable modes replace the MaxSim entirely (match/"
  1388	                         "pool/align/topk/thresh are bypassed) and add new params "
  1389	                         "that ARE optimized (assert self-check at startup).")
  1390	    ap.add_argument('--ovli_vlad_clusters', type=int, default=8,
  1391	                    help='C learnable clusters for --ovli_setpool netvlad')
  1392	    ap.add_argument('--ovli_attn_heads', type=int, default=4,
  1393	                    help='heads for --ovli_setpool attn (must divide --ovli_dim)')
  1394	    ap.add_argument('--ovli_so_rank', type=int, default=32,
  1395	                    help='low-rank dim r for --ovli_setpool secondorder (r x r cov)')
  1396	    # "mean + zero-init residual" toggle for the learnable set pools.  Default 1
  1397	    # (True): each pool = mean_k(tok) + zero-init_gate * residual, so it starts
  1398	    # BYTE-IDENTICAL to the 52.37 mean-pool and only learns a correction (fixes
  1399	    # the random-init standalone collapse: netvlad standalone ep20 14.66 << 52.37).
  1400	    # 0 (False): the original STANDALONE pooling (random init fully replaces the
  1401	    # mean) -- kept ONLY for the standalone-vs-residual ablation, expected to
  1402	    # collapse.  Ignored when --ovli_setpool mean (no learnable pool at all).
  1403	    ap.add_argument('--ovli_setpool_residual', type=int, default=1,
  1404	                    choices=[0, 1],
  1405	                    help='1 (default): learnable set pool = mean + zero-init '
  1406	                         'residual (lossless start from the 52.37 mean-pool, only '
  1407	                         'learns a correction). 0: original standalone pooling '
  1408	                         '(random init replaces the mean -> the collapsing '
  1409	                         'control, ablation only). No effect with --ovli_setpool '
  1410	                         'mean.')
  1411	    # --- ACVP (Ambiguity-Calibrated opposite-View negative relaxation) ---
  1412	    # Detached opposite-view-prototype ambiguity SENSOR that softens unreliable
  1413	    # NEGATIVES in the OVLI cross-view contrastive denominator. NO prototype-
  1414	    # positive alignment (avoids OVP/CMPC/PDPA overlap): the prototype bank is read
  1415	    # detached only; ACVP adds no learnable param and injects no gradient.  Default
  1416	    # OFF -> the OVLI loss path is byte-identical to the current (pre-ACVP) one.
  1417	    ap.add_argument('--acvp', action='store_true',
  1418	                    help='enable ACVP: detached opposite-view-prototype ambiguity '
  1419	                         'softening of UNRELIABLE NEGATIVES in the OVLI contrastive '
  1420	                         'denominator (no prototype-positive alignment, no new '
  1421	                         'learnable params, detached). Requires --ovli. Default OFF '
  1422	                         'reproduces the OVLI loss byte-for-byte.')
  1423	    ap.add_argument('--acvp_gamma', type=float, default=0.5,
  1424	                    help='ACVP max softening strength: w_ij = clamp(1 - gamma*'
  1425	                         'sigmoid((delta-margin)/eta), wmin, 1). gamma=0 disables '
  1426	                         'softening even if --acvp is set.')
  1427	    ap.add_argument('--acvp_wmin', type=float, default=0.3,
  1428	                    help='ACVP floor on the negative weight w_ij (>0 so log(w_ij) '
  1429	                         'is finite; never fully removes a negative).')
  1430	    ap.add_argument('--acvp_eta', type=float, default=0.05,
  1431	                    help='ACVP sigmoid temperature on the ambiguity delta.')
  1432	    ap.add_argument('--acvp_margin', type=float, default=0.0,
  1433	                    help='ACVP ambiguity margin: only delta>margin softens.')
  1434	    ap.add_argument('--acvp_warmup', type=int, default=10,
  1435	                    help='ACVP linear gamma warmup over this many epochs (ramp 0 -> '
  1436	                         'acvp_gamma) so early, noisy prototypes do not mis-soften.')
  1437	    # --- AIRL (Aerial Identity Recoverability Learning) -- resolution-degradation
  1438	    # consistency.  Default OFF -> the baseline trains byte-for-byte (no degrade,
  1439	    # no extra forward, no loss).  NO learnable params (degrade = augmentation,
  1440	    # consistency = loss); the optimizer is untouched.  See airl_degrade /
  1441	    # airl_consistency_loss above.  Independent of OVP/OVLI/ACVP (can co-run, but
  1442	    # the headline AIRL run is --airl alone on the plain baseline).
  1443	    ap.add_argument('--airl', action='store_true',
  1444	                    help='enable AIRL: per-image resolution degradation (to a '
  1445	                         'sampled aerial-scale pixel budget) + original/degraded '
  1446	                         'prediction-consistency loss. NO learnable params, TRAIN-'
  1447	                         'time only, eval unchanged. Default OFF reproduces the '
  1448	                         'baseline byte-for-byte.')
  1449	    ap.add_argument('--airl_lambda', type=float, default=0.5,
  1450	                    help='weight of the AIRL consistency loss '
  1451	                         '(total = CE + triplet + airl_lambda_eff * consistency).')
  1452	    ap.add_argument('--airl_min_scale', type=float, default=0.25,
  1453	                    help='lowest degradation scale ratio (per-image s ~ U[min_scale,'
  1454	                         ' 1]); s*H x s*W is the down-sampled pixel budget before '
  1455	                         'up-sampling back. 0.25 ~ the aerial small bucket (aerial '
  1456	                         'median bbox ~1/3 ground). Must be in (0,1].')
  1457	    ap.add_argument('--airl_consistency', default='kl', choices=['kl', 'feat'],
  1458	                    help="consistency target: kl (default) = temperature-scaled "
  1459	                         "soft-target KL on the ID logits (clean detached); feat = "
  1460	                         "1 - cosine on the L2-normed BNNeck feature.")
  1461	    ap.add_argument('--airl_tau', type=float, default=4.0,
  1462	                    help='softmax temperature for --airl_consistency kl (Hinton '
  1463	                         'distillation; loss scaled by tau^2). Ignored for feat.')
  1464	    ap.add_argument('--airl_blur', action='store_true',
  1465	                    help='additionally apply a light 3x3 avg-pool blur after the '
  1466	                         'up-sample (UAV optical-blur proxy; no extra params).')
  1467	    ap.add_argument('--airl_warmup', type=int, default=5,
  1468	                    help='linear AIRL lambda warmup over this many epochs (ramp 0 -> '
  1469	                         'airl_lambda) so the consistency term opens gently.')
  1470	    # --- AIRL dual-branch (resolvability branch): the COMPLETE AIRL mechanism for
  1471	    # a single-model, single-forward score fusion.  Adds a SECOND BNNeck head
  1472	    # (f_rec) on the shared backbone: f_full keeps full-resolution identity
  1473	    # evidence (protects G->A), f_rec gets its own ID-CE PLUS the AIRL
  1474	    # ground-degradation consistency (learns low-pixel-budget recoverable
  1475	    # evidence, serves A->G).  At eval the two heads' cosine scores are
  1476	    # SOFT-fused at the distance-matrix level:
  1477	    #     cos = airl_fuse_w * cos(f_rec) + (1 - airl_fuse_w) * cos(f_full)
  1478	    # with a SINGLE FIXED global w (a prior, NOT tuned on the test set, NOT a
  1479	    # per-query gate) -> this internalises the kill-switch #3 two-model score
  1480	    # fusion (+1.46 mean @ w=0.25) into ONE forward (both heads share the
  1481	    # backbone).  Framing (pinned to avoid the RAR/MRJL resolution-adaptive /
  1482	    # query-routing collision): "observation-limited evidence ceiling under which
  1483	    # a clean (f_full) and a recover (f_rec) evidence head DIVERGE, combined by a
  1484	    # FIXED-PRIOR soft fusion".  This is deliberately NOT query-budget routing --
  1485	    # kill-switch #3 showed hard per-query routing (area / reliability) fails to
  1486	    # recover the trade-off (<=+0.41), and the win comes from the fixed-w soft
  1487	    # blend; so we claim head divergence + fixed-prior fusion, not dynamic routing.
  1488	    # Default OFF -> the second head is never built and training/eval reproduce
  1489	    # the single-head baseline byte-for-byte.
  1490	    ap.add_argument('--airl_dualbranch', action='store_true',
  1491	                    help='enable the AIRL dual-branch (resolvability branch): a '
  1492	                         'second BNNeck head f_rec (own ID-CE + AIRL degradation '
  1493	                         'consistency) alongside the clean f_full head, soft-fused '
  1494	                         'at eval (cos = w*cos_rec + (1-w)*cos_full). One forward, '
  1495	                         'two features. Default OFF reproduces the baseline.')
  1496	    ap.add_argument('--airl_fuse_w', type=float, default=0.25,
  1497	                    help='fixed global fusion weight on the f_rec cosine at eval '
  1498	                         '(cos = airl_fuse_w*cos_rec + (1-airl_fuse_w)*cos_full); '
  1499	                         '0.25 = the legal default from kill-switch #3 (plateau '
  1500	                         'w in [0.25,0.75] all >= +1.46 mean). Must be in [0,1]. '
  1501	                         'NOT tuned on test (train/test symmetric). ABLATION-ONLY: '
  1502	                         'the headline is FIXED at 0.25; non-default w is for the '
  1503	                         'w-sweep ablation only (a warning prints if changed).')
  1504	    ap.add_argument('--airl_dualbranch_iso', action='store_true',
  1505	                    help='gradient-ISOLATED AIRL dual-branch (rescue of the failed '
  1506	                         'fully-shared --airl_dualbranch): f_rec is a BNNeck over an '
  1507	                         'INDEPENDENT late Swin stage forked off the shared trunk at '
  1508	                         'iso_stage (not the shared global_feat). The degradation-'
  1509	                         'CONSISTENCY gradient updates ONLY the rec late stage + '
  1510	                         'BNNeck_rec and NEVER flows back into the shared trunk (the '
  1511	                         'degraded pass forks off a DETACHED trunk), so f_rec stays a '
  1512	                         '"recover expert" and the +0.06 collapse (shared trunk pulled '
  1513	                         'toward degradation-robustness) is avoided. The CLEAN f_rec '
  1514	                         'ID-CE routing is governed by --airl_iso_trunk_recce: default '
  1515	                         '1 (the FIX) REFLOWS it into the trunk (extra identity '
  1516	                         'supervision -> strengthens the otherwise-weak f_full); 0 = '
  1517	                         'original full-isolation (clean ID-CE also detached). '
  1518	                         'swin_small only. Same eval soft-fusion + consistency '
  1519	                         'contract as --airl_dualbranch (shares its AIRL hyperparams '
  1520	                         '+ --airl_fuse_w). Default OFF reproduces the baseline.')
  1521	    ap.add_argument('--airl_iso_stage', type=int, default=3,
  1522	                    help='Swin stage index the f_rec branch forks AFTER (the rec '
  1523	                         'branch re-runs stages [iso_stage..last] on its own deep-'
  1524	                         'copied weights fed by the DETACHED trunk stream at the '
  1525	                         'input of this stage). swin_small has 4 stages (0..3); '
  1526	                         'iso_stage=3 (default) = share stages 0-2, split ONLY the '
  1527	                         'last stage (MGN-style, cheapest); iso_stage=2 = split the '
  1528	                         'last two stages (more f_rec divergence capacity, heavier). '
  1529	                         'Must be in [1,3]. Only used with --airl_dualbranch_iso.')
  1530	    # The trunk-undersupervision FIX (codex consensus).  The original full-detach
  1531	    # iso left f_full WEAK (ep20 45.56 < baseline 48.98 < even fully-shared f_rec
  1532	    # 47.39): f_rec's clean ID-CE only updated the DETACHED rec tail, so the shared
  1533	    # trunk lost the extra identity supervision the fully-shared dual-branch's trunk
  1534	    # got from BOTH heads' ID-CE.  --airl_iso_trunk_recce 1 (default) re-routes ONLY
  1535	    # the CLEAN rec ID-CE gradient back into the shared trunk (extra identity
  1536	    # supervision -> strengthens f_full) while keeping the degradation-CONSISTENCY
  1537	    # gradient detached from the trunk (so f_rec stays a specialised recover pole --
  1538	    # the isolation that the iso variant exists for).  0 = the ORIGINAL full-isolation
  1539	    # iso (clean ID-CE also detached), kept for the controlled ablation.  Only used
  1540	    # with --airl_dualbranch_iso.
  1541	    ap.add_argument('--airl_iso_trunk_recce', type=int, default=1, choices=[0, 1],
  1542	                    help='1 (default, the FIX): route the CLEAN f_rec ID-CE gradient '
  1543	                         'back into the shared trunk (extra identity supervision -> '
  1544	                         'strengthens the weak f_full); the degradation-consistency '
  1545	                         'gradient stays DETACHED from the trunk (f_rec stays '
  1546	                         'specialised). 0: original full-isolation iso (clean ID-CE '
  1547	                         'also detached from the trunk), ablation only. No effect '
  1548	                         'without --airl_dualbranch_iso.')
  1549	    args = ap.parse_args()
  1550	    args.afd_router = bool(args.afd_router)
  1551	    args.afd_cvfc = bool(args.afd_cvfc)
  1552	    args.router_cond_view = bool(args.router_cond_view)
  1553	    args.ovli_setpool_residual = bool(args.ovli_setpool_residual)
  1554	    args.airl_iso_trunk_recce = bool(args.airl_iso_trunk_recce)
  1555	    # backbone guard: the AFD frequency modules (router/cvfc) insert at resnet
  1556	    # shallow stages that don't exist in Swin -> --use_afd is incompatible with
  1557	    # --backbone swin_small (caught in AFDModel too, but fail fast at parse time).
  1558	    if args.backbone == 'swin_small' and args.use_afd:
  1559	        ap.error("--backbone swin_small does not support --use_afd (AFD modules "
  1560	                 "insert at resnet shallow stages). OVP/OVLI work on swin; drop "
  1561	                 "--use_afd.")
  1562	    # OVP and OVLI are two distinct cross-view mechanisms (prototype-memory
  1563	    # InfoNCE vs sample-to-sample late-interaction retrieval).  All three modes
  1564	    # are supported and back-compatible:
  1565	    #   OVP-only   (--ovp)         : empirical prototype auxiliary
  1566	    #   OVLI-only  (--ovli)        : headline late-interaction retrieval
  1567	    #   both       (--ovp --ovli)  : complementarity test -- each loss keeps its
  1568	    #                                own warmup / lambda / diagnostics and is
  1569	    #                                added to the same total; OVP adds no params,
  1570	    #                                OVLI's proj is the only extra optimized set.
  1571	    # total = CE + triplet + ovp_lam_eff*OVP + ovli_lam_eff*OVLI (terms that are
  1572	    # off contribute exactly 0, so OVP-only / OVLI-only reproduce as before).
  1573	    # ACVP is a pure calibration ON TOP of the OVLI loss (softens unreliable
  1574	    # negatives in the OVLI denominator via a detached opposite-view prototype
  1575	    # ambiguity sensor); it has no loss term of its own and requires --ovli.
  1576	    if args.acvp and not args.ovli:
  1577	        ap.error("--acvp requires --ovli (it calibrates the OVLI contrastive "
  1578	                 "negatives; there is no standalone ACVP loss).")
  1579	    # ACVP is "opposite-view negative relaxation": it only makes sense when the
  1580	    # OVLI candidate set IS opposite-view-only.  Under --ovli_allview the negatives
  1581	    # include same-view pairs, which contradicts the mechanism's wording, so we
  1582	    # forbid the combination outright (cleaner than silently calibrating all-view).
  1583	    if args.acvp and args.ovli_allview:
  1584	        ap.error("--acvp is opposite-view negative relaxation and is incompatible "
  1585	                 "with --ovli_allview (which adds same-view negatives). Drop one.")
  1586	    # ACVP numeric-safety guards: bad CLI values would make w_ij / log(w_ij)
  1587	    # produce inf/NaN.  Enforce wmin in (0,1], eta>0, gamma>=0 at parse time so a
  1588	    # typo fails fast instead of corrupting the loss mid-training.
  1589	    if args.acvp:
  1590	        if not (args.acvp_wmin > 0.0 and args.acvp_wmin <= 1.0):
  1591	            ap.error("--acvp_wmin must be in (0,1] (w_ij floor; >0 so log(w) is "
  1592	                     f"finite, <=1 since w_ij<=1); got {args.acvp_wmin}.")
  1593	        if not (args.acvp_eta > 0.0):
  1594	            ap.error("--acvp_eta must be > 0 (sigmoid temperature; 0 -> div-by-0); "
  1595	                     f"got {args.acvp_eta}.")
  1596	        if not (args.acvp_gamma >= 0.0):
  1597	            ap.error("--acvp_gamma must be >= 0 (softening strength; <0 would "
  1598	                     f"AMPLIFY negatives); got {args.acvp_gamma}.")
  1599	
  1600	    # AIRL numeric-safety guard: min_scale in (0,1] so the down-sampled budget is a
  1601	    # real fraction of the input (>0) and never upscales (<=1); a typo fails fast.
  1602	    if args.airl and not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
  1603	        ap.error("--airl_min_scale must be in (0,1] (per-image scale ratio s in "
  1604	                 f"[min_scale,1]); got {args.airl_min_scale}.")
  1605	    if args.airl and not (args.airl_tau > 0.0):
  1606	        ap.error(f"--airl_tau must be > 0 (softmax temperature); got {args.airl_tau}.")
  1607	
  1608	    # AIRL dual-branch guards.  --airl (single-head consistency) and
  1609	    # --airl_dualbranch (two-head, consistency on f_rec only) are two DIFFERENT
  1610	    # AIRL instantiations of the SAME degrade+consistency primitive; running both
  1611	    # would apply consistency twice (to the single head AND to f_rec) and muddy
  1612	    # the ablation, so they are mutually exclusive.  The dual-branch shares
  1613	    # --airl_lambda / --airl_min_scale / --airl_consistency / --airl_tau /
  1614	    # --airl_blur / --airl_warmup (the consistency on f_rec is the SAME function),
  1615	    # and they are validated the same way (so a stray bad --airl_min_scale with
  1616	    # only --airl_dualbranch still fails fast).
  1617	    if args.airl_dualbranch:
  1618	        if args.airl:
  1619	            ap.error("--airl_dualbranch and --airl are mutually exclusive (both "
  1620	                     "apply the AIRL degradation-consistency; dual-branch applies "
  1621	                     "it to the f_rec head only). Pick one.")
  1622	        if not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
  1623	            ap.error("--airl_min_scale must be in (0,1] (used by --airl_dualbranch "
  1624	                     f"too); got {args.airl_min_scale}.")
  1625	        if not (args.airl_tau > 0.0):
  1626	            ap.error("--airl_tau must be > 0 (used by --airl_dualbranch too); got "
  1627	                     f"{args.airl_tau}.")
  1628	        if not (0.0 <= args.airl_fuse_w <= 1.0):
  1629	            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
  1630	                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
  1631	        # w-lock (soft): the headline fixed-prior fusion uses w=0.25.  A non-default
  1632	        # w is ABLATION-ONLY (the w-sweep), so warn rather than assert -- the sweep
  1633	        # still needs to pass other values -- but make any deviation from the
  1634	        # headline visible in the log so a stray w never silently becomes "the
  1635	        # result".
  1636	        if args.airl_fuse_w != 0.25:
  1637	            print(f"[AIRL-DUAL][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
  1638	                  "headline uses the FIXED prior w=0.25; non-default w is "
  1639	                  "ABLATION-ONLY (w-sweep), not the headline result.")
  1640	        # The dual-branch is the standalone headline AIRL mechanism; keep its
  1641	        # ablation clean by forbidding co-running the cross-view OVP/OVLI losses
  1642	        # (they target a different gap and would confound the f_rec specialisation).
  1643	        if args.ovp or args.ovli:
  1644	            ap.error("--airl_dualbranch is run standalone (headline AIRL); do not "
  1645	                     "combine with --ovp/--ovli (separate cross-view mechanisms).")
  1646	
  1647	    # AIRL gradient-isolated dual-branch guards.  This is the RESCUE variant: the
  1648	    # SAME degrade+consistency+soft-fusion contract as --airl_dualbranch, but f_rec
  1649	    # forks off a DETACHED trunk into an independent late Swin stage (so the
  1650	    # consistency gradient cannot pollute the shared trunk).  It therefore:
  1651	    #   * shares the AIRL hyperparams (--airl_lambda/min_scale/consistency/tau/blur/
  1652	    #     warmup) and --airl_fuse_w, validated identically;
  1653	    #   * is mutually exclusive with BOTH --airl (single-head) and --airl_dualbranch
  1654	    #     (fully-shared) -- three distinct AIRL instantiations, one at a time;
  1655	    #   * is swin_small-only (the fork lives in the Swin stage list);
  1656	    #   * runs standalone (no OVP/OVLI), same as --airl_dualbranch.
  1657	    if args.airl_dualbranch_iso:
  1658	        if args.airl or args.airl_dualbranch:
  1659	            ap.error("--airl_dualbranch_iso is mutually exclusive with --airl and "
  1660	                     "--airl_dualbranch (three distinct AIRL instantiations; the "
  1661	                     "iso variant forks an independent late stage off a detached "
  1662	                     "trunk). Pick one.")
  1663	        if args.backbone != 'swin_small':
  1664	            ap.error("--airl_dualbranch_iso requires --backbone swin_small (the rec "
  1665	                     "branch forks an independent Swin late stage).")
  1666	        if not (1 <= args.airl_iso_stage <= 3):
  1667	            ap.error("--airl_iso_stage must be in [1,3] (swin_small has 4 stages "
  1668	                     "0..3; fork after a shared early stage, before the last); got "
  1669	                     f"{args.airl_iso_stage}.")
  1670	        if not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
  1671	            ap.error("--airl_min_scale must be in (0,1] (used by "
  1672	                     f"--airl_dualbranch_iso too); got {args.airl_min_scale}.")
  1673	        if not (args.airl_tau > 0.0):
  1674	            ap.error("--airl_tau must be > 0 (used by --airl_dualbranch_iso too); "
  1675	                     f"got {args.airl_tau}.")
  1676	        if not (0.0 <= args.airl_fuse_w <= 1.0):
  1677	            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
  1678	                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
  1679	        if args.airl_fuse_w != 0.25:
  1680	            print(f"[AIRL-ISO][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
  1681	                  "headline uses the FIXED prior w=0.25; non-default w is "
  1682	                  "ABLATION-ONLY (w-sweep), not the headline result.")
  1683	        if args.ovp or args.ovli:
  1684	            ap.error("--airl_dualbranch_iso is run standalone (headline AIRL); do "
  1685	                     "not combine with --ovp/--ovli (separate cross-view "
  1686	                     "mechanisms).")
  1687	
  1688	    set_seed(args.seed)
  1689	    os.makedirs(args.out_dir, exist_ok=True)
  1690	    device = 'cuda'
  1691	    batch_size = args.P * args.K
  1692	
  1693	    print("=" * 70)
  1694	    print("CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)")
  1695	    if args.backbone == 'swin_small':
  1696	        print(f"  backbone=swin_small (SOLIDER, in_planes=768) "
  1697	              f"pretrain={args.swin_pretrain or 'NONE (from scratch)'} "
  1698	              f"semantic_weight={args.swin_semantic_weight}")
  1699	    else:
  1700	        print(f"  backbone=resnet50 (BoT baseline) pool={args.pool} "
  1701	              f"last_stride={args.last_stride}")
  1702	    print(f"  use_afd={args.use_afd}  ovp={args.ovp} "
  1703	          f"(lambda={args.ovp_lambda} tau={args.ovp_tau} mom={args.ovp_momentum})")
  1704	    print(f"  ovli={args.ovli} (lambda={args.ovli_lambda} tau={args.ovli_tau} "
  1705	          f"alpha={args.ovli_alpha} dim={args.ovli_dim} grid={tuple(args.ovli_grid)} "

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1990,2445p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
  1990	            ovli.train()
  1991	        t0 = time.time()
  1992	        # H1 fix: warmup OVP lambda over ovp_warmup epochs to avoid cold-start gradient spikes
  1993	        ovp_lambda_eff = (args.ovp_lambda * min(1.0, epoch / max(1, args.ovp_warmup))) if args.ovp else 0.0
  1994	        # H1 lesson: same linear warmup for OVLI (random proj -> avoid early spike)
  1995	        ovli_lambda_eff = (args.ovli_lambda * min(1.0, epoch / max(1, args.ovli_warmup))) if args.ovli else 0.0
  1996	        # ACVP: linear gamma warmup (ramp 0 -> acvp_gamma over acvp_warmup epochs)
  1997	        # so early, noisy prototypes do not aggressively soften negatives.
  1998	        acvp_gamma_eff = (args.acvp_gamma * min(1.0, epoch / max(1, args.acvp_warmup))) if args.acvp else 0.0
  1999	        # AIRL: linear lambda warmup (ramp 0 -> airl_lambda over airl_warmup epochs)
  2000	        # so the resolution-consistency term opens gently.  Shared by ALL THREE AIRL
  2001	        # instantiations (mutually exclusive): the single-head --airl, the fully-
  2002	        # shared dual-branch --airl_dualbranch, AND the gradient-isolated dual-branch
  2003	        # --airl_dualbranch_iso (same consistency function, same warmup).  MUST list
  2004	        # all three: the flags are mutually exclusive, so omitting iso here would
  2005	        # leave airl_lambda_eff==0 every epoch on an iso run and silently zero out
  2006	        # the f_rec consistency gradient (the whole mechanism being tested).
  2007	        airl_lambda_eff = (args.airl_lambda * min(1.0, epoch / max(1, args.airl_warmup))) \
  2008	            if (args.airl or args.airl_dualbranch or args.airl_dualbranch_iso) else 0.0
  2009	        meters = {'loss': 0.0, 'ce': 0.0, 'tri': 0.0, 'ovp': 0.0,
  2010	                  'ovli': 0.0, 'ovli_pos': 0.0, 'ovli_neg': 0.0, 'acc': 0.0,
  2011	                  'airl': 0.0, 'airl_scale': 0.0, 'airl_n_ground': 0.0,
  2012	                  'ce_rec': 0.0}
  2013	        # ACVP kill-switch accumulators (weighted by #softenable-neg per step;
  2014	        # steps with 0 softenable negatives are skipped, not counted).
  2015	        acvp_frac_sum = 0.0     # sum of relaxed_neg_frac * n_softenable_neg
  2016	        acvp_w_sum = 0.0        # sum of mean_w * n_softenable_neg
  2017	        acvp_steps = 0          # total #softenable-neg pairs ACVP acted on
  2018	        seen = 0
  2019	
  2020	        for it, batch in enumerate(train_loader):
  2021	            imgs = batch['img'].to(device, non_blocking=True)
  2022	            labels = batch['pid'].to(device, non_blocking=True)
  2023	            views = torch.tensor([view_map[v] for v in batch['view']],
  2024	                                 device=device)
  2025	            vidx = views if args.use_afd else None
  2026	
  2027	            optimizer.zero_grad()
  2028	            with torch.amp.autocast('cuda', enabled=not args.no_amp):
  2029	                out = model(imgs, view_idx=vidx,
  2030	                            return_cvfc=(args.use_afd and args.afd_cvfc))
  2031	                logits = out['logits']
  2032	                gfeat = out['global_feat']
  2033	                bn = out['bn_feat']
  2034	                loss_ce = ce(logits, labels)
  2035	                loss_tri = tri(gfeat, labels)
  2036	                loss = loss_ce + loss_tri
  2037	
  2038	                # AIRL dual-branch: the f_rec head needs its OWN identity grounding
  2039	                # so it is a valid discriminative space for the eval fusion (a head
  2040	                # trained on consistency alone would be unidentified).  Add f_rec's
  2041	                # ID cross-entropy (SAME label-smoothing CE as f_full); the global
  2042	                # triplet stays on the SHARED global_feat (NOT duplicated for f_rec).
  2043	                # The f_rec degradation-consistency is added below (fp32 block).
  2044	                loss_ce_rec = torch.zeros((), device=device)
  2045	                if args.airl_dualbranch or args.airl_dualbranch_iso:
  2046	                    # f_rec ID grounding.  For --airl_dualbranch f_rec reads the
  2047	                    # shared global_feat; for --airl_dualbranch_iso it reads the
  2048	                    # INDEPENDENT rec late-stage map.  Both expose logits_rec, so
  2049	                    # the CE call is identical -- only the gradient destination of
  2050	                    # this CLEAN ID-CE differs:
  2051	                    #   * --airl_dualbranch       -> the shared trunk (fully shared).
  2052	                    #   * --airl_dualbranch_iso, trunk_recce=1 (FIX) -> the shared
  2053	                    #     trunk TOO: model.forward ran the iso clean pass with a
  2054	                    #     NON-detached fork, so this clean ID-CE reflows into the
  2055	                    #     trunk (extra identity supervision -> strengthens f_full)
  2056	                    #     while the degradation-consistency below (rec_only, detached)
  2057	                    #     stays trunk-isolated.
  2058	                    #   * --airl_dualbranch_iso, trunk_recce=0 -> the isolated rec
  2059	                    #     stage only (original full-isolation: clean fork detached).
  2060	                    loss_ce_rec = ce(out['logits_rec'], labels)
  2061	                    loss = loss + loss_ce_rec
  2062	
  2063	                loss_ovp = torch.zeros((), device=device)
  2064	                if args.ovp:
  2065	                    # OVP loss in fp32 for numerical safety (cosine + softmax)
  2066	                    z = F.normalize(bn.float(), dim=1)
  2067	                    loss_ovp = ovp.loss(z, labels, views)
  2068	                    loss = loss + ovp_lambda_eff * loss_ovp
  2069	
  2070	            # OVLI: compute in TRUE fp32 (autocast disabled) -- the cos/MaxSim/
  2071	            # logsumexp at tau=0.05 want fp32, and running the proj here (after
  2072	            # the autocast forward already cached the fp16 layer4 map) keeps the
  2073	            # projection weights in fp32 while gradient still flows into layer4.
  2074	            loss_ovli = torch.zeros((), device=device)
  2075	            ovli_pos = torch.zeros((), device=device)
  2076	            ovli_neg = torch.zeros((), device=device)
  2077	            if args.ovli:
  2078	                with torch.amp.autocast('cuda', enabled=False):
  2079	                    # global feature for the score: normalized BN feat (matches
  2080	                    # the eval ranking space). gradient flows -> encoder.
  2081	                    g_ovli = F.normalize(bn.float(), dim=1)
  2082	                    tok = ovli.tokens_from_cached_map()          # (B,K,Dp) fp32
  2083	                    if args.acvp:
  2084	                        # ACVP ON: pass the DETACHED opposite-view prototype bank +
  2085	                        # the warmup-ramped gamma so the OVLI denominator softens
  2086	                        # unreliable negatives.  acvp_mem.bank/.inited are buffers
  2087	                        # (no grad); .detach() makes the no-grad contract explicit.
  2088	                        loss_ovli, ovli_pos, ovli_neg = ovli.loss(
  2089	                            g_ovli, tok, labels, views,
  2090	                            acvp_proto=acvp_mem.bank.detach(),
  2091	                            acvp_inited=acvp_mem.inited.detach(),
  2092	                            acvp_gamma=acvp_gamma_eff,
  2093	                            acvp_wmin=args.acvp_wmin,
  2094	                            acvp_eta=args.acvp_eta,
  2095	                            acvp_margin=args.acvp_margin)
  2096	                    else:
  2097	                        # ACVP OFF: byte-identical original 4-arg call -> the loss
  2098	                        # body never touches the ACVP branch (acvp_proto is None).
  2099	                        loss_ovli, ovli_pos, ovli_neg = ovli.loss(
  2100	                            g_ovli, tok, labels, views)
  2101	                loss = loss + ovli_lambda_eff * loss_ovli
  2102	
  2103	            # AIRL: resolution-degradation consistency.  ASYMMETRIC by design --
  2104	            # degrade ONLY the high-resolution GROUND view (views==1; Aerial==0) to
  2105	            # a sampled aerial-scale pixel budget, run ONE extra forward through the
  2106	            # SAME model (shared weights), and pull the degraded GROUND prediction
  2107	            # toward its own (detached) clean one.  The hypothesis is "recover
  2108	            # ground identity at an aerial pixel budget"; degrading the already
  2109	            # low-budget aerial samples would just be all-view self-degradation and
  2110	            # break that asymmetry, so aerial rows are NOT degraded.  No learnable
  2111	            # params; train-time only.  Empty-ground batch -> loss_airl=0 (no extra
  2112	            # forward).  OFF (default) -> this whole block is skipped (no degrade,
  2113	            # no extra forward, no loss) => the baseline trains byte-for-byte.
  2114	            loss_airl = torch.zeros((), device=device)
  2115	            airl_scale_mean = torch.zeros((), device=device)
  2116	            n_ground = 0
  2117	            if args.airl:
  2118	                # GROUND subset = views==1 (high-res view to degrade).  Slice the
  2119	                # clean inputs/preds to the SAME rows so consistency compares the
  2120	                # degraded ground vs its own clean ground prediction.
  2121	                g_mask = (views == 1)
  2122	                n_ground = int(g_mask.sum())
  2123	                # require >=2 ground rows: the degraded batch goes through the
  2124	                # train-mode model whose BNNeck BatchNorm1d raises "Expected more
  2125	                # than 1 value per channel" on a size-1 batch.  n_ground in {0,1}
  2126	                # -> skip AIRL this step (loss_airl stays 0, no extra forward).  The
  2127	                # ID-balanced RandomIdentitySampler makes a <2-ground batch a rare
  2128	                # cold edge, so the dropped consistency signal is negligible.
  2129	                if n_ground >= 2:
  2130	                    imgs_g = imgs[g_mask]
  2131	                    vidx_g = vidx[g_mask] if vidx is not None else None
  2132	                    # degrade in fp32 image space (resolution/low-pass proxy); the
  2133	                    # second forward runs under the SAME autocast as the original so
  2134	                    # AMP behaviour matches, while the consistency loss is fp32.
  2135	                    with torch.no_grad():
  2136	                        deg_imgs, deg_scales = airl_degrade(
  2137	                            imgs_g, args.airl_min_scale, blur=args.airl_blur)
  2138	                        airl_scale_mean = deg_scales.mean()
  2139	                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
  2140	                        out_d = model(deg_imgs, view_idx=vidx_g,
  2141	                                      return_cvfc=(args.use_afd and args.afd_cvfc))
  2142	                    with torch.amp.autocast('cuda', enabled=False):
  2143	                        # consistency forces the DEGRADED ground prediction
  2144	                        # (gradient on) toward the CLEAN ground one (detached target
  2145	                        # inside the loss).  Clean side sliced to the ground rows.
  2146	                        loss_airl = airl_consistency_loss(
  2147	                            logits[g_mask], bn[g_mask],
  2148	                            out_d['logits'], out_d['bn_feat'],
  2149	                            mode=args.airl_consistency, tau=args.airl_tau)
  2150	                    loss = loss + airl_lambda_eff * loss_airl
  2151	                # n_ground < 2 -> too few ground rows this batch: loss_airl stays 0,
  2152	                # no extra forward, nothing added to loss (does not crash; avoids
  2153	                # the size-1 BatchNorm1d error).
  2154	
  2155	            # AIRL dual-branch: the SAME ground-only degradation-consistency, but
  2156	            # applied ONLY to the f_rec head (logits_rec / bn_feat_rec).  f_full is
  2157	            # left clean in the sense that it receives ZERO consistency GRADIENT
  2158	            # (smoke D4) -> it keeps full-resolution discrimination (protects G->A);
  2159	            # f_rec is pulled toward its own clean prediction under the low pixel
  2160	            # budget (serves A->G).  NOTE: the degraded forward below is a FULL
  2161	            # model(deg_imgs) pass (the model has no rec-only path), so f_full's
  2162	            # frozen-bias BNNeck running mean/var DO see the degraded ground images
  2163	            # for stat tracking only -- exactly as in the --airl single-head path
  2164	            # above (same shared degrade+forward primitive), a deliberately accepted
  2165	            # minor exposure, NOT a gradient leak; whether it matters is settled
  2166	            # empirically by kill-switch #4, and matching --airl keeps the ablation
  2167	            # honest.  Identical degrade + >=2-ground guard + fp32 consistency as
  2168	            # --airl above; the only difference is the HEAD the consistency reads.
  2169	            # Mutually exclusive with --airl, so loss_airl is 0 unless dual-branch.
  2170	            if args.airl_dualbranch:
  2171	                g_mask = (views == 1)                      # high-res GROUND subset
  2172	                n_ground = int(g_mask.sum())
  2173	                if n_ground >= 2:
  2174	                    imgs_g = imgs[g_mask]
  2175	                    vidx_g = vidx[g_mask] if vidx is not None else None
  2176	                    with torch.no_grad():
  2177	                        deg_imgs, deg_scales = airl_degrade(
  2178	                            imgs_g, args.airl_min_scale, blur=args.airl_blur)
  2179	                        airl_scale_mean = deg_scales.mean()
  2180	                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
  2181	                        out_d = model(deg_imgs, view_idx=vidx_g,
  2182	                                      return_cvfc=(args.use_afd and args.afd_cvfc))
  2183	                    with torch.amp.autocast('cuda', enabled=False):
  2184	                        # consistency on the f_rec head ONLY: degraded f_rec
  2185	                        # prediction (grad on) -> clean f_rec one (detached target
  2186	                        # inside the loss).  Both sides sliced to the ground rows.
  2187	                        loss_airl = airl_consistency_loss(
  2188	                            out['logits_rec'][g_mask], out['bn_feat_rec'][g_mask],
  2189	                            out_d['logits_rec'], out_d['bn_feat_rec'],
  2190	                            mode=args.airl_consistency, tau=args.airl_tau)
  2191	                    loss = loss + airl_lambda_eff * loss_airl
  2192	                # n_ground < 2 -> skip (same size-1 BatchNorm1d guard as --airl).
  2193	
  2194	            # AIRL gradient-isolated dual-branch: the SAME ground-only degradation-
  2195	            # consistency on the f_rec head.  The DEGRADED side (out_d) comes from a
  2196	            # rec_only=True forward whose rec fork feed is ALWAYS detached from the
  2197	            # trunk (model.forward / _forward_swin_split), and the CLEAN side
  2198	            # (out['logits_rec'], out['bn_feat_rec']) is the DETACHED target inside
  2199	            # airl_consistency_loss.  So the consistency gradient flows ONLY through
  2200	            # out_d -> into the rec late stage + BNNeck_rec, and is severed at the
  2201	            # detach BEFORE the shared trunk -- the clean trunk + f_full receive ZERO
  2202	            # consistency gradient (smoke I4) REGARDLESS of --airl_iso_trunk_recce
  2203	            # (which only governs the CLEAN ID-CE pass, added above; the consistency's
  2204	            # clean side is detached here, so trunk_recce never opens a consistency
  2205	            # path to the trunk).  They keep full-resolution discrimination while
  2206	            # f_rec specialises as the recover pole.  The degraded forward uses
  2207	            # rec_only=True: it computes ONLY the f_rec head (the rec late stage +
  2208	            # BNNeck_rec), so f_full's BNNeck running stats are NOT updated on the
  2209	            # degraded images -> f_full stays a TRUE clean expert (no degraded-ground
  2210	            # stat leak, unlike the shared --airl_dualbranch which accepts that minor
  2211	            # exposure) and the f_full pool+classifier is skipped
  2212	            # (cheaper).  Mutually exclusive with --airl / --airl_dualbranch, so this
  2213	            # block fires only for the iso variant.
  2214	            if args.airl_dualbranch_iso:
  2215	                g_mask = (views == 1)                      # high-res GROUND subset
  2216	                n_ground = int(g_mask.sum())
  2217	                if n_ground >= 2:
  2218	                    imgs_g = imgs[g_mask]
  2219	                    vidx_g = vidx[g_mask] if vidx is not None else None
  2220	                    with torch.no_grad():
  2221	                        deg_imgs, deg_scales = airl_degrade(
  2222	                            imgs_g, args.airl_min_scale, blur=args.airl_blur)
  2223	                        airl_scale_mean = deg_scales.mean()
  2224	                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
  2225	                        # rec_only -> dict with ONLY logits_rec / bn_feat_rec (f_full
  2226	                        # BNNeck not run on degraded images).
  2227	                        out_d = model(deg_imgs, view_idx=vidx_g, rec_only=True)
  2228	                    with torch.amp.autocast('cuda', enabled=False):
  2229	                        # consistency on the ISOLATED f_rec head: degraded f_rec
  2230	                        # prediction (grad on, into the rec stage only) -> clean f_rec
  2231	                        # one (detached target).  Both sides sliced to ground rows.
  2232	                        loss_airl = airl_consistency_loss(
  2233	                            out['logits_rec'][g_mask], out['bn_feat_rec'][g_mask],
  2234	                            out_d['logits_rec'], out_d['bn_feat_rec'],
  2235	                            mode=args.airl_consistency, tau=args.airl_tau)
  2236	                    loss = loss + airl_lambda_eff * loss_airl
  2237	                # n_ground < 2 -> skip (same size-1 BatchNorm1d guard as --airl).
  2238	
  2239	            scaler.scale(loss).backward()
  2240	            scaler.step(optimizer)
  2241	            scaler.update()
  2242	
  2243	            # batch size: define right after the step and BEFORE the OVP/ACVP
  2244	            # post-step blocks and the meter accumulation below, so no later code
  2245	            # (incl. the ACVP stats / meters) can ever hit UnboundLocalError on
  2246	            # the first iteration.  (Was previously assigned AFTER the ACVP stats
  2247	            # block -> first --acvp batch crashed.)
  2248	            bs = imgs.size(0)
  2249	
  2250	            # update prototypes AFTER the optimizer step, with detached feats
  2251	            if args.ovp:
  2252	                with torch.no_grad():
  2253	                    ovp.update(F.normalize(bn.detach().float(), dim=1),
  2254	                               labels, views)
  2255	            # ACVP: EMA-update its OWN detached prototype bank from the current
  2256	            # batch (same detached BNNeck-feature recipe as OVP).  Done AFTER the
  2257	            # optimizer step so the bank tracks the just-updated encoder.  ACVP
  2258	            # only ever READS this bank (in ovli.loss above) -> no grad path.
  2259	            if args.acvp:
  2260	                with torch.no_grad():
  2261	                    acvp_mem.update(F.normalize(bn.detach().float(), dim=1),
  2262	                                    labels, views)
  2263	                    stats = getattr(ovli, '_acvp_stats', None)
  2264	                    if stats is not None:
  2265	                        # weight the per-step (frac, mean_w) by #softenable-neg
  2266	                        # (stats[2]=ok.sum()), NOT batch size, and SKIP steps with
  2267	                        # no softenable negatives so cold-start batches don't bias
  2268	                        # the per-epoch summary toward "looks safe".
  2269	                        n_soft = int(stats[2])
  2270	                        if n_soft > 0:
  2271	                            acvp_frac_sum += float(stats[0]) * n_soft
  2272	                            acvp_w_sum += float(stats[1]) * n_soft
  2273	                            acvp_steps += n_soft
  2274	
  2275	            seen += bs
  2276	            meters['loss'] += loss.item() * bs
  2277	            meters['ce'] += loss_ce.item() * bs
  2278	            meters['tri'] += loss_tri.item() * bs
  2279	            meters['ovp'] += float(loss_ovp) * bs
  2280	            meters['ovli'] += float(loss_ovli) * bs
  2281	            meters['ovli_pos'] += float(ovli_pos) * bs
  2282	            meters['ovli_neg'] += float(ovli_neg) * bs
  2283	            meters['airl'] += float(loss_airl) * bs
  2284	            meters['airl_scale'] += float(airl_scale_mean) * bs
  2285	            meters['airl_n_ground'] += n_ground
  2286	            meters['ce_rec'] += float(loss_ce_rec) * bs
  2287	            meters['acc'] += (logits.argmax(1) == labels).float().sum().item()
  2288	
  2289	            if (it + 1) % 50 == 0 or (it + 1) == n_iter_total:
  2290	                lr = optimizer.param_groups[0]['lr']
  2291	                extra = ""
  2292	                if args.ovp:
  2293	                    extra += f" OVP: {meters['ovp'] / seen:.4f}"
  2294	                if args.ovli:
  2295	                    extra += f" OVLI: {meters['ovli'] / seen:.4f}"
  2296	                if args.airl:
  2297	                    extra += f" AIRL: {meters['airl'] / seen:.4f}"
  2298	                if args.airl_dualbranch or args.airl_dualbranch_iso:
  2299	                    extra += (f" CE_rec: {meters['ce_rec'] / seen:.3f}"
  2300	                              f" AIRL_rec: {meters['airl'] / seen:.4f}")
  2301	                print(f"Epoch[{epoch}] Iter[{it + 1}/{n_iter_total}] "
  2302	                      f"Loss: {meters['loss'] / seen:.3f} "
  2303	                      f"CE: {meters['ce'] / seen:.3f} "
  2304	                      f"Tri: {meters['tri'] / seen:.3f}{extra} "
  2305	                      f"Acc: {meters['acc'] / seen:.3f} LR: {lr:.2e}")
  2306	
  2307	        scheduler.step()
  2308	        dt = time.time() - t0
  2309	        ovp_msg = ""
  2310	        if args.ovp:
  2311	            ovp_msg = (f" OVP[lam_eff={ovp_lambda_eff:.3f} "
  2312	                       f"inited={int(ovp.inited.sum())}/{2 * ovp.num_pid}]")
  2313	        ovli_msg = ""
  2314	        if args.ovli:
  2315	            pos = meters['ovli_pos'] / seen
  2316	            neg = meters['ovli_neg'] / seen
  2317	            ovli_msg = (f" OVLI[lam_eff={ovli_lambda_eff:.3f} "
  2318	                        f"loss={meters['ovli'] / seen:.4f} "
  2319	                        f"pos={pos:.3f} neg={neg:.3f} gap={pos - neg:+.3f}]")
  2320	        # ACVP kill-switch log (per-epoch): relaxed_neg_frac = fraction of
  2321	        # SOFTENABLE negatives with w_ij<0.95, mean_w = mean softenable-neg weight,
  2322	        # both #softenable-neg-weighted across steps (cold-start / no-softenable
  2323	        # steps excluded).  Watch for relaxed_neg_frac>0.30 or mean_w<0.75 (=> ACVP
  2324	        # is broadly weakening negatives = bad; stop / reduce gamma).
  2325	        acvp_msg = ""
  2326	        if args.acvp:
  2327	            inited_n = int(acvp_mem.inited.sum())
  2328	            if acvp_steps > 0:
  2329	                a_frac = acvp_frac_sum / acvp_steps
  2330	                a_mw = acvp_w_sum / acvp_steps
  2331	            else:
  2332	                a_frac, a_mw = 0.0, 1.0
  2333	            flag = " <KILL?>" if (a_frac > 0.30 or a_mw < 0.75) else ""
  2334	            acvp_msg = (f" ACVP[gamma_eff={acvp_gamma_eff:.3f} "
  2335	                        f"relaxed_neg_frac={a_frac:.3f} mean_w={a_mw:.3f} "
  2336	                        f"proto_inited={inited_n}/{2 * acvp_mem.num_pid}{flag}]")
  2337	        # AIRL per-epoch kill-switch log: lam_eff (warmup-ramped weight),
  2338	        # consistency_loss (mean degraded<->clean consistency this epoch -- watch
  2339	        # it stay finite and TREND DOWN as the model learns budget-stable evidence;
  2340	        # exploding => degradation too harsh / lambda too high), deg_scale_mean
  2341	        # (mean sampled pixel-budget ratio, ~ (1+min_scale)/2; sanity that the
  2342	        # degradation actually fires), n_ground (total GROUND samples degraded this
  2343	        # epoch -- AIRL is asymmetric: only views==1 ground rows get degraded; this
  2344	        # confirms the asymmetric mask fires and aerial rows are left intact).
  2345	        # airl_collapse flag if the consistency loss is ~0 from the start
  2346	        # (degradation not biting) / NaN/inf, OR n_ground==0 all epoch (mask wrong /
  2347	        # no ground sampled => AIRL silently did nothing).
  2348	        airl_msg = ""
  2349	        if args.airl:
  2350	            a_cons = meters['airl'] / seen
  2351	            a_scale = meters['airl_scale'] / seen
  2352	            n_g = int(meters['airl_n_ground'])
  2353	            bad = (not math.isfinite(a_cons)) or (n_g == 0)
  2354	            flag = " <KILL?>" if bad else ""
  2355	            airl_msg = (f" AIRL[lam_eff={airl_lambda_eff:.3f} "
  2356	                        f"consistency={a_cons:.4f} deg_scale_mean={a_scale:.3f} "
  2357	                        f"n_ground={n_g}{flag}]")
  2358	        # AIRL dual-branch per-epoch log: f_rec ID-CE (must converge like f_full's
  2359	        # CE -> f_rec is a valid identity space), the f_rec degradation-consistency
  2360	        # (same trend-down expectation as --airl), deg_scale_mean and n_ground.
  2361	        # Collapse flag if ce_rec is non-finite, the consistency is non-finite, or
  2362	        # n_ground==0 all epoch (asymmetric mask never fired).
  2363	        if args.airl_dualbranch or args.airl_dualbranch_iso:
  2364	            a_cons = meters['airl'] / seen
  2365	            a_scale = meters['airl_scale'] / seen
  2366	            a_cerec = meters['ce_rec'] / seen
  2367	            n_g = int(meters['airl_n_ground'])
  2368	            bad = (not math.isfinite(a_cons)) or (not math.isfinite(a_cerec)) \
  2369	                or (n_g == 0)
  2370	            flag = " <KILL?>" if bad else ""
  2371	            tag = "AIRL-ISO" if args.airl_dualbranch_iso else "AIRL-DUAL"
  2372	            airl_msg = (f" {tag}[lam_eff={airl_lambda_eff:.3f} "
  2373	                        f"ce_rec={a_cerec:.3f} consistency={a_cons:.4f} "
  2374	                        f"deg_scale_mean={a_scale:.3f} n_ground={n_g}{flag}]")
  2375	        print(f"Epoch[{epoch}] done in {dt:.1f}s  "
  2376	              f"Loss={meters['loss'] / seen:.3f} "
  2377	              f"Acc={meters['acc'] / seen:.3f}{ovp_msg}{ovli_msg}{acvp_msg}{airl_msg}")
  2378	
  2379	        if epoch % args.eval_period == 0 or epoch == args.epochs:
  2380	            results = run_cross_view_eval(model, dataset, args, device)
  2381	            mean_map = print_eval(epoch, results)
  2382	            # opt-in OVLI rerank report (global-only number above is the primary
  2383	            # eval and is unchanged; this just adds the global+MaxSim rerank).
  2384	            if args.ovli and args.ovli_rerank:
  2385	                rr = ovli_rerank_eval(model, ovli, dataset, args, device)
  2386	                print(f"  ---- OVLI rerank (alpha={args.ovli_alpha}) "
  2387	                      f"@ epoch {epoch} ----")
  2388	                for tag in ('A->G', 'G->A'):
  2389	                    gm, gr = rr[tag]['global']
  2390	                    rm, rrk = rr[tag]['rerank']
  2391	                    print(f"    [{tag}] global mAP={gm:.2f} R1={gr:.2f}  ->  "
  2392	                          f"rerank mAP={rm:.2f} R1={rrk:.2f}")
  2393	                rmean = (rr['A->G']['rerank'][0] + rr['G->A']['rerank'][0]) / 2
  2394	                print(f"    [mean] rerank mAP={rmean:.2f}")
  2395	            # AIRL dual-branch: report f_full-only, f_rec-only, and the SOFT-FUSED
  2396	            # mean (cos = w*cos_rec + (1-w)*cos_full).  The run_cross_view_eval
  2397	            # number above is the f_full-only head (model() returns f_full at eval);
  2398	            # the HEADLINE = the fused mean, which is what model-selection uses (the
  2399	            # whole point of the dual-branch is the fusion, not f_full alone).
  2400	            if args.airl_dualbranch or args.airl_dualbranch_iso:
  2401	                # airl_dualbranch_eval calls model(return_dual=True) which yields
  2402	                # (f_full, f_rec) for BOTH the shared and the iso variant (the iso
  2403	                # forward's want_iso path returns the same tuple), so the soft-fusion
  2404	                # eval is shared verbatim.
  2405	                dual = airl_dualbranch_eval(model, dataset, args, device)
  2406	                _ev_tag = ("AIRL-ISO dual-branch" if args.airl_dualbranch_iso
  2407	                           else "AIRL dual-branch")
  2408	                print(f"  ---- {_ev_tag} (fuse_w={args.airl_fuse_w}) "
  2409	                      f"@ epoch {epoch} ----")
  2410	                for tag in ('A->G', 'G->A'):
  2411	                    print(f"    [{tag}] full mAP={dual[tag]['full'][0]:.2f} "
  2412	                          f"R1={dual[tag]['full'][1]:.2f} | "
  2413	                          f"rec mAP={dual[tag]['rec'][0]:.2f} "
  2414	                          f"R1={dual[tag]['rec'][1]:.2f} | "
  2415	                          f"FUSE mAP={dual[tag]['fuse'][0]:.2f} "
  2416	                          f"R1={dual[tag]['fuse'][1]:.2f}")
  2417	                full_mean = (dual['A->G']['full'][0] + dual['G->A']['full'][0]) / 2
  2418	                rec_mean = (dual['A->G']['rec'][0] + dual['G->A']['rec'][0]) / 2
  2419	                fuse_mean = (dual['A->G']['fuse'][0] + dual['G->A']['fuse'][0]) / 2
  2420	                print(f"    [mean] full={full_mean:.2f} rec={rec_mean:.2f} "
  2421	                      f"FUSE={fuse_mean:.2f}  <- model-selection uses FUSE")
  2422	                # override model-selection metric with the fused mean
  2423	                mean_map = fuse_mean
  2424	            if mean_map > best_map:
  2425	                best_map = mean_map
  2426	                best_epoch = epoch
  2427	                torch.save(model.state_dict(),
  2428	                           os.path.join(args.out_dir, 'model_best.pth'))
  2429	                if ovli is not None:
  2430	                    torch.save(ovli.state_dict(),
  2431	                               os.path.join(args.out_dir, 'ovli_best.pth'))
  2432	                print(f"    * new best mean mAP={best_map:.2f} (epoch {epoch}) saved")
  2433	
  2434	    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_final.pth'))
  2435	    if ovli is not None:
  2436	        torch.save(ovli.state_dict(), os.path.join(args.out_dir, 'ovli_final.pth'))
  2437	        ovli.remove_hook()
  2438	    print("=" * 70)
  2439	    print(f"Training finished. Best mean A<->G mAP={best_map:.2f} @ epoch {best_epoch}")
  2440	    print(f"Checkpoints in {args.out_dir}")
  2441	    print("=" * 70)
  2442	
  2443	
  2444	if __name__ == '__main__':
  2445	    main()

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1705,1990p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
  1705	          f"alpha={args.ovli_alpha} dim={args.ovli_dim} grid={tuple(args.ovli_grid)} "
  1706	          f"warmup={args.ovli_warmup} rerank={args.ovli_rerank} "
  1707	          f"pool={args.ovli_pool} topk={args.ovli_topk} thresh={args.ovli_thresh} "
  1708	          f"cand={'allview' if args.ovli_allview else 'oppview'} "
  1709	          f"match={args.ovli_match} align={args.ovli_align} "
  1710	          f"setpool={args.ovli_setpool} "
  1711	          f"setpool_residual={args.ovli_setpool_residual})")
  1712	    print(f"  acvp={args.acvp} (gamma={args.acvp_gamma} wmin={args.acvp_wmin} "
  1713	          f"eta={args.acvp_eta} margin={args.acvp_margin} "
  1714	          f"warmup={args.acvp_warmup}) [detached neg-relaxation on OVLI; "
  1715	          f"off => OVLI byte-identical]")
  1716	    print(f"  airl={args.airl} (lambda={args.airl_lambda} "
  1717	          f"min_scale={args.airl_min_scale} consistency={args.airl_consistency} "
  1718	          f"tau={args.airl_tau} blur={args.airl_blur} warmup={args.airl_warmup}) "
  1719	          f"[resolution-degradation consistency; NO learnable params; train-only; "
  1720	          f"off => baseline byte-identical]")
  1721	    print(f"  airl_dualbranch={args.airl_dualbranch} (fuse_w={args.airl_fuse_w} "
  1722	          f"lambda={args.airl_lambda} min_scale={args.airl_min_scale} "
  1723	          f"consistency={args.airl_consistency} tau={args.airl_tau} "
  1724	          f"blur={args.airl_blur} warmup={args.airl_warmup}) "
  1725	          f"[resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL "
  1726	          f"consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; "
  1727	          f"1 forward 2 features; off => baseline byte-identical]")
  1728	    print(f"  airl_dualbranch_iso={args.airl_dualbranch_iso} "
  1729	          f"(iso_stage={args.airl_iso_stage} trunk_recce={args.airl_iso_trunk_recce} "
  1730	          f"fuse_w={args.airl_fuse_w} "
  1731	          f"lambda={args.airl_lambda} min_scale={args.airl_min_scale} "
  1732	          f"consistency={args.airl_consistency} tau={args.airl_tau} "
  1733	          f"blur={args.airl_blur} warmup={args.airl_warmup}) "
  1734	          f"[GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late "
  1735	          f"Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared "
  1736	          f"trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the "
  1737	          f"CLEAN f_rec ID-CE into the trunk (extra identity supervision -> "
  1738	          f"strengthens f_full); trunk_recce=0 = original full-isolation (clean "
  1739	          f"ID-CE also detached). f_rec learns the recover pole; same soft-fusion "
  1740	          f"eval; off => baseline byte-identical]")
  1741	    print(f"  bs={batch_size} (P={args.P} K={args.K}) lr={args.lr} "
  1742	          f"epochs={args.epochs} warmup={args.warmup_epochs} amp={not args.no_amp}")
  1743	    print(f"  out_dir={args.out_dir}")
  1744	    print("=" * 70)
  1745	
  1746	    # data
  1747	    if args.dataset == 'cargo':
  1748	        dataset = CARGO(root=args.data_root, verbose=True)
  1749	    elif args.dataset == 'agreid_v2':
  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
  1751	        # so run_cross_view_eval/print_eval report the official per-protocol mAP
  1752	        # and their mean with no change to the eval / AIRL-iso code.
  1753	        dataset = AGReIDV2Combined(root=args.data_root, verbose=True)
  1754	    else:
  1755	        dataset = AGReIDv2(root=args.data_root, verbose=True)
  1756	    train_tf = build_transforms(is_train=True, img_size=tuple(args.img_size))
  1757	    train_set = CARGOImageDataset(dataset.train, train_tf)
  1758	    sampler = RandomIdentitySampler(dataset.train, batch_size, args.K)
  1759	    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
  1760	                              num_workers=args.workers, pin_memory=True,
  1761	                              drop_last=True)
  1762	
  1763	    # model
  1764	    model = build_model(dataset.num_train_pids, args).to(device)
  1765	
  1766	    # losses
  1767	    ce = CrossEntropyLabelSmooth(dataset.num_train_pids, args.label_smooth)
  1768	    tri = TripletLoss(args.margin)
  1769	
  1770	    # OVP memory (buffers live on the model device; not optimized)
  1771	    ovp = None
  1772	    if args.ovp:
  1773	        ovp = OVPMemory(dataset.num_train_pids, model.in_planes,
  1774	                        momentum=args.ovp_momentum, tau=args.ovp_tau).to(device)
  1775	
  1776	    # OVLI head: token projection (NEW learnable params) + hook on model.layer4.
  1777	    # The projection MUST be optimized -> add ovli.parameters() to the optimizer.
  1778	    ovli = None
  1779	    if args.ovli:
  1780	        ovli = OVLIHead(model, in_ch=model.in_planes, proj_dim=args.ovli_dim,
  1781	                        grid=tuple(args.ovli_grid), alpha=args.ovli_alpha,
  1782	                        tau=args.ovli_tau, pool=args.ovli_pool,
  1783	                        topk=args.ovli_topk, thresh=args.ovli_thresh,
  1784	                        allview=args.ovli_allview,
  1785	                        match=args.ovli_match, align=args.ovli_align,
  1786	                        setpool=args.ovli_setpool,
  1787	                        vlad_clusters=args.ovli_vlad_clusters,
  1788	                        attn_heads=args.ovli_attn_heads,
  1789	                        so_rank=args.ovli_so_rank,
  1790	                        setpool_residual=args.ovli_setpool_residual).to(device)
  1791	
  1792	    # ACVP prototype bank: a DEDICATED, detached opposite-view EMA prototype bank
  1793	    # (its own OVPMemory instance, independent of --ovp so the two never share or
  1794	    # double-update a buffer).  ACVP only READS this bank (detached) to compute the
  1795	    # ambiguity weight that softens unreliable negatives in the OVLI denominator;
  1796	    # it never runs an InfoNCE on it (no prototype-positive alignment) and adds no
  1797	    # learnable param -> the bank stays out of the optimizer.  Built ONLY when
  1798	    # --acvp is set, so the no-ACVP path constructs no bank at all (off-mode is
  1799	    # structurally identical to the current code).
  1800	    acvp_mem = None
  1801	    if args.acvp:
  1802	        acvp_mem = OVPMemory(dataset.num_train_pids, model.in_planes,
  1803	                             momentum=args.ovp_momentum, tau=args.ovp_tau).to(device)
  1804	
  1805	    # optimizer -- include the OVLI projection params (model has none of them).
  1806	    # AdamW(model.parameters()) alone would silently SKIP the OVLI proj, so when
  1807	    # --ovli is on we pass list(model.parameters()) + list(ovli.parameters()).
  1808	    #
  1809	    # Swin backbone fine-tuning LR: the resnet50-tuned peak LR (3.5e-4 AdamW) is
  1810	    # SAFE for resnet50 but DIVERGES the ~50M-param SOLIDER Swin transformer --
  1811	    # cvpb_swin_ovli trained healthily for 7 epochs (Acc 0.47) then COLLAPSED at
  1812	    # epoch 8 the moment the warmup pushed LR past ~2.5e-4: the backbone fell into
  1813	    # a constant-output fixed point (last-stage map off-diag cos +0.99 -> all
  1814	    # images map to ~one vector -> cross-view mAP 0.03).  No NaN; a genuine
  1815	    # optimization collapse.  Transformer ReID is fine-tuned at a much smaller
  1816	    # backbone LR than the randomly-initialised heads (the repo's main SOLIDER
  1817	    # config likewise uses a gentle schedule).  So for backbone='swin_small' we
  1818	    # scale ONLY the Swin backbone params by swin_lr_factor (default 0.1) and keep
  1819	    # the BNNeck / classifier / OVLI proj at the full LR (they are random-init and
  1820	    # must learn fast).  resnet50 is untouched (no backbone_swin -> single group,
  1821	    # byte-identical to before).
  1822	    swin_lr_factor = getattr(args, 'swin_lr_factor', 0.1)
  1823	    if model.backbone == 'swin_small' and swin_lr_factor != 1.0:
  1824	        swin_param_ids = {id(p) for p in model.backbone_swin.parameters()}
  1825	        swin_params = [p for p in model.parameters()
  1826	                       if p.requires_grad and id(p) in swin_param_ids]
  1827	        other_params = [p for p in model.parameters()
  1828	                        if p.requires_grad and id(p) not in swin_param_ids]
  1829	        if ovli is not None:
  1830	            other_params = other_params + [p for p in ovli.parameters()
  1831	                                           if p.requires_grad]
  1832	        param_groups = [
  1833	            {'params': swin_params, 'lr': args.lr * swin_lr_factor},
  1834	            {'params': other_params, 'lr': args.lr},
  1835	        ]
  1836	        optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
  1837	                                      weight_decay=args.weight_decay)
  1838	        print(f"  [swin] backbone LR = {args.lr * swin_lr_factor:.2e} "
  1839	              f"(= base {args.lr:.2e} x {swin_lr_factor}); "
  1840	              f"heads/BNNeck/OVLI LR = {args.lr:.2e}  "
  1841	              f"[{len(swin_params)} backbone tensors, {len(other_params)} head tensors] "
  1842	              f"-- prevents the epoch-8 Swin collapse")
  1843	    else:
  1844	        opt_params = list(model.parameters())
  1845	        if ovli is not None:
  1846	            opt_params = opt_params + list(ovli.parameters())
  1847	        optimizer = torch.optim.AdamW(opt_params, lr=args.lr,
  1848	                                      weight_decay=args.weight_decay)
  1849	    # self-check: confirm the OVLI projection params actually landed in the
  1850	    # optimizer (the key structural requirement vs OVP).
  1851	    if ovli is not None:
  1852	        opt_ids = {id(p) for grp in optimizer.param_groups for p in grp['params']}
  1853	        proj_in = all(id(p) in opt_ids for p in ovli.proj.parameters())
  1854	        n_proj = sum(p.numel() for p in ovli.proj.parameters())
  1855	        assert proj_in, "OVLI proj params NOT in optimizer!"
  1856	        print(f"  [OVLI] projection params in optimizer: {proj_in} "
  1857	              f"({n_proj} params, {sum(1 for _ in ovli.proj.parameters())} tensors)")
  1858	        # self-check: the learnable set-pool params (NetVLAD centers / attn query
  1859	        # / gate MLP / covariance proj) must ALSO land in the optimizer.
  1860	        if ovli.setpool_mod is not None:
  1861	            sp_params = list(ovli.setpool_mod.parameters())
  1862	            sp_in = all(id(p) in opt_ids for p in sp_params)
  1863	            n_sp = sum(p.numel() for p in sp_params)
  1864	            assert sp_in, "OVLI setpool params NOT in optimizer!"
  1865	            # the zero-init residual gate must also be optimized (it is what lets
  1866	            # the residual turn on after the lossless mean-pool start).
  1867	            gate_in = (ovli.setpool_mod.gate_res is None
  1868	                       or id(ovli.setpool_mod.gate_res) in opt_ids)
  1869	            assert gate_in, "OVLI setpool residual gate NOT in optimizer!"
  1870	            res_msg = ("mean + zero-init residual (lossless start from 52.37 "
  1871	                       "mean-pool, gate_res zero-init)"
  1872	                       if ovli.setpool_residual else
  1873	                       "STANDALONE (random init replaces mean; ablation/collapse)")
  1874	            print(f"  [OVLI] setpool='{ovli.setpool}' params in optimizer: {sp_in} "
  1875	                  f"({n_sp} params, {len(sp_params)} tensors); "
  1876	                  f"MaxSim match/pool/align BYPASSED (set-pool aggregation); "
  1877	                  f"mode={res_msg}")
  1878	        else:
  1879	            print("  [OVLI] setpool='mean' (token-set MaxSim path; "
  1880	                  "match/pool/align active)")
  1881	        print(f"  [OVLI] candidate view-mask: "
  1882	              + ("ALL-VIEW (ablation: opposite-view constraint OFF; positives = "
  1883	                 "same-pid any view, negatives = other-pid any view)"
  1884	                 if args.ovli_allview else
  1885	                 "OPPOSITE-VIEW-ONLY (headline cross-view supervision)"))
  1886	        print(f"  [OVLI] token-match: "
  1887	              + ("MAXSIM (ColBERT/late-interaction max selection, headline)"
  1888	                 if args.ovli_match == 'maxsim' else
  1889	                 "AVG (ablation: per-token MEAN over other tokens = soft global)")
  1890	              + " | align: "
  1891	              + ("FREE (free/global late interaction, headline)"
  1892	                 if args.ovli_align == 'free' else
  1893	                 "ORDERED (ablation: AlignedReID row-correspondence)"))
  1894	        # ACVP self-check: detached prototype sensor, NO learnable params -> it
  1895	        # must NOT introduce anything into the optimizer (the structural contract:
  1896	        # ACVP is a re-weighting, not a learned alignment).
  1897	        if acvp_mem is not None:
  1898	            acvp_buf_in_opt = any(id(b) in opt_ids for b in acvp_mem.buffers())
  1899	            assert not acvp_buf_in_opt, "ACVP prototype buffers leaked into optimizer!"
  1900	            print(f"  [ACVP] ON: detached opposite-view prototype ambiguity "
  1901	                  f"softening of OVLI negatives (gamma={args.acvp_gamma} "
  1902	                  f"wmin={args.acvp_wmin} eta={args.acvp_eta} "
  1903	                  f"margin={args.acvp_margin} warmup={args.acvp_warmup}); "
  1904	                  f"NO learnable params (buffers in optimizer: {acvp_buf_in_opt}); "
  1905	                  f"no prototype-positive alignment (read-only, detached)")
  1906	    # AIRL dual-branch self-check: the SECOND BNNeck head (bottleneck_rec +
  1907	    # classifier_rec) lives inside model.parameters(), so it is in the optimizer
  1908	    # automatically -- but assert it explicitly (it is the structural requirement:
  1909	    # f_rec must actually train, with its OWN params, at the FULL head LR even on
  1910	    # Swin where the backbone is at swin_lr_factor x LR).
  1911	    if args.airl_dualbranch:
  1912	        opt_ids = {id(p) for grp in optimizer.param_groups for p in grp['params']}
  1913	        rec_params = (list(model.bottleneck_rec.parameters())
  1914	                      + list(model.classifier_rec.parameters()))
  1915	        # bottleneck_rec.bias has requires_grad_=False (frozen, like f_full's BN
  1916	        # bias) -> AdamW (a no-arg param list) still RECEIVES it but never updates
  1917	        # it (zero grad); only assert the TRAINABLE rec params are present.
  1918	        rec_trainable = [p for p in rec_params if p.requires_grad]
  1919	        rec_in = all(id(p) in opt_ids for p in rec_trainable)
  1920	        assert rec_in, "AIRL dual-branch f_rec head params NOT in optimizer!"
  1921	        # on Swin, f_rec must be at the FULL head LR (not the backbone factor):
  1922	        # both rec params are random-init heads, identical to f_full's BNNeck.
  1923	        n_rec = sum(p.numel() for p in rec_trainable)
  1924	        print(f"  [AIRL-DUAL] f_rec head (bottleneck_rec + classifier_rec) params "
  1925	              f"in optimizer: {rec_in} ({n_rec} params, {len(rec_trainable)} "
  1926	              f"trainable tensors); eval soft-fusion cos=w*cos_rec+(1-w)*cos_full "
  1927	              f"w={args.airl_fuse_w}")
  1928	    # AIRL gradient-isolated dual-branch self-check: BNNeck_rec + classifier_rec are
  1929	    # random-init heads OUTSIDE backbone_swin -> FULL-LR group; the INDEPENDENT rec
  1930	    # late stage (rec_stages/rec_norm) lives INSIDE backbone_swin -> the scaled Swin
  1931	    # LR group (pretrained weights, same as f_full's stages).  Assert both placements
  1932	    # so a future param-group refactor cannot silently freeze or mis-LR the rec path.
  1933	    if args.airl_dualbranch_iso:
  1934	        opt_ids = {id(p) for grp in optimizer.param_groups for p in grp['params']}
  1935	        bsw = model.backbone_swin
  1936	        rec_head_params = [p for p in (list(model.bottleneck_rec.parameters())
  1937	                                       + list(model.classifier_rec.parameters()))
  1938	                           if p.requires_grad]
  1939	        rec_head_in = all(id(p) in opt_ids for p in rec_head_params)
  1940	        assert rec_head_in, "AIRL-ISO f_rec head params NOT in optimizer!"
  1941	        # rec late-stage trainable params (rec_stages + rec_norm; semantic-embed is
  1942	        # frozen so excluded) must ALL be in the optimizer and trainable.
  1943	        rec_stage_params = [p for p in (list(bsw.rec_stages.parameters())
  1944	                                        + list(bsw.rec_norm.parameters()))
  1945	                            if p.requires_grad]
  1946	        rec_stage_in = all(id(p) in opt_ids for p in rec_stage_params)
  1947	        assert rec_stage_in, "AIRL-ISO rec late-stage params NOT in optimizer!"
  1948	        # the rec late stage must be on the SCALED Swin LR group (it is pretrained
  1949	        # backbone weight, byte-identical recipe to f_full's stages).  Find which
  1950	        # group each rec-stage param landed in and confirm it is the swin group when
  1951	        # the swin split is active.
  1952	        if model.backbone == 'swin_small' and swin_lr_factor != 1.0:
  1953	            swin_grp_ids = {id(p) for p in param_groups[0]['params']}
  1954	            full_grp_ids = {id(p) for p in param_groups[1]['params']}
  1955	            rec_stage_in_swin = all(id(p) in swin_grp_ids for p in rec_stage_params)
  1956	            rec_head_in_full = all(id(p) in full_grp_ids for p in rec_head_params)
  1957	            assert rec_stage_in_swin, ("AIRL-ISO rec late stage NOT in the scaled "
  1958	                                       "Swin LR group (it is pretrained backbone "
  1959	                                       "weight)!")
  1960	            assert rec_head_in_full, ("AIRL-ISO rec BNNeck head NOT in the full-LR "
  1961	                                      "group (it is a random-init head)!")
  1962	            lr_msg = (f"rec late stage @ Swin LR {args.lr * swin_lr_factor:.2e}, "
  1963	                      f"rec BNNeck @ full LR {args.lr:.2e}")
  1964	        else:
  1965	            lr_msg = f"single LR group @ {args.lr:.2e}"
  1966	        n_rh = sum(p.numel() for p in rec_head_params)
  1967	        n_rs = sum(p.numel() for p in rec_stage_params)
  1968	        recce_msg = ("trunk_recce=1 (clean f_rec ID-CE REFLOWS to trunk; degraded "
  1969	                     "consistency stays detached)" if args.airl_iso_trunk_recce
  1970	                     else "trunk_recce=0 (clean ID-CE + consistency BOTH detached = "
  1971	                          "original full-isolation)")
  1972	        print(f"  [AIRL-ISO] iso_stage={args.airl_iso_stage}: rec late stage "
  1973	              f"({n_rs} params, {len(rec_stage_params)} tensors) + rec BNNeck head "
  1974	              f"({n_rh} params, {len(rec_head_params)} tensors) in optimizer "
  1975	              f"[{lr_msg}]; degradation-consistency grad isolated from shared trunk "
  1976	              f"(detached degraded pass at stage-{args.airl_iso_stage} input); "
  1977	              f"{recce_msg}; eval soft-fusion "
  1978	              f"cos=w*cos_rec+(1-w)*cos_full w={args.airl_fuse_w}")
  1979	    scheduler = WarmupCosineLR(optimizer, args.warmup_epochs, args.epochs)
  1980	    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)
  1981	
  1982	    view_map = {'Aerial': 0, 'Ground': 1}
  1983	    best_map = -1.0
  1984	    best_epoch = -1
  1985	    n_iter_total = len(train_loader)
  1986	
  1987	    for epoch in range(1, args.epochs + 1):
  1988	        model.train()
  1989	        if ovli is not None:
  1990	            ovli.train()

codex
AIRL 训练时主 forward 和 degraded forward 都在 autocast 下，consistency loss 在禁用 autocast 的 fp32 块里算；日志里也有 `amp=True` 和每 epoch 的 `AIRL-ISO[...]` 监控。下一步我会抽表核数值结论。
exec
/bin/zsh -lc 'for f in logs/*.log; do echo "$f"; sed -n '"'1,24p' \""'$f"; echo; done' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
logs/agreidv2_airl_4090.log
======================================================================
CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/home/afr/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
  use_afd=False  ovp=False (lambda=0.5 tau=0.05 mom=0.2)
  ovli=False (lambda=0.5 tau=0.05 alpha=0.5 dim=256 grid=(8, 4) warmup=10 rerank=False pool=mean topk=8 thresh=0.0 cand=oppview match=maxsim align=free setpool=mean setpool_residual=True)
  acvp=False (gamma=0.5 wmin=0.3 eta=0.05 margin=0.0 warmup=10) [detached neg-relaxation on OVLI; off => OVLI byte-identical]
  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
  out_dir=/home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090
======================================================================
=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
  -----------------------------------------------------------
  subset   | # imgs | # pids | # cams | aerial | ground
  -----------------------------------------------------------
  train    |  51530 |    807 |      3 |  21217 |  30313
  query    |   4167 |    534 |      2 |   2356 |   1811
  gallery  |  20709 |    534 |      2 |  14362 |   6347
  -----------------------------------------------------------
unloaded parameters: <All keys matched successfully>
  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [361 backbone tensors, 4 head tensors] -- prevents the epoch-8 Swin collapse

logs/agreidv2_airl_iso.log
======================================================================
CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/root/work/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
  use_afd=False  ovp=False (lambda=0.5 tau=0.05 mom=0.2)
  ovli=False (lambda=0.5 tau=0.05 alpha=0.5 dim=256 grid=(8, 4) warmup=10 rerank=False pool=mean topk=8 thresh=0.0 cand=oppview match=maxsim align=free setpool=mean setpool_residual=True)
  acvp=False (gamma=0.5 wmin=0.3 eta=0.05 margin=0.0 warmup=10) [detached neg-relaxation on OVLI; off => OVLI byte-identical]
  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
  out_dir=/root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_airl_iso
======================================================================
=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
  -----------------------------------------------------------
  subset   | # imgs | # pids | # cams | aerial | ground
  -----------------------------------------------------------
  train    |  51530 |    807 |      3 |  21217 |  30313
  query    |   4167 |    534 |      2 |   2356 |   1811
  gallery  |  20709 |    534 |      2 |  14362 |   6347
  -----------------------------------------------------------
unloaded parameters: <All keys matched successfully>
  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [361 backbone tensors, 4 head tensors] -- prevents the epoch-8 Swin collapse

logs/agreidv2_baseline.log
======================================================================
CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/root/work/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
  use_afd=False  ovp=False (lambda=0.5 tau=0.05 mom=0.2)
  ovli=False (lambda=0.5 tau=0.05 alpha=0.5 dim=256 grid=(8, 4) warmup=10 rerank=False pool=mean topk=8 thresh=0.0 cand=oppview match=maxsim align=free setpool=mean setpool_residual=True)
  acvp=False (gamma=0.5 wmin=0.3 eta=0.05 margin=0.0 warmup=10) [detached neg-relaxation on OVLI; off => OVLI byte-identical]
  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
  out_dir=/root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_baseline
======================================================================
=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
  -----------------------------------------------------------
  subset   | # imgs | # pids | # cams | aerial | ground
  -----------------------------------------------------------
  train    |  51530 |    807 |      3 |  21217 |  30313
  query    |   4167 |    534 |      2 |   2356 |   1811
  gallery  |  20709 |    534 |      2 |  14362 |   6347
  -----------------------------------------------------------
unloaded parameters: <All keys matched successfully>
  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [333 backbone tensors, 2 head tensors] -- prevents the epoch-8 Swin collapse

logs/agreidv2_baseline_4090.log
======================================================================
CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/home/afr/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
  use_afd=False  ovp=False (lambda=0.5 tau=0.05 mom=0.2)
  ovli=False (lambda=0.5 tau=0.05 alpha=0.5 dim=256 grid=(8, 4) warmup=10 rerank=False pool=mean topk=8 thresh=0.0 cand=oppview match=maxsim align=free setpool=mean setpool_residual=True)
  acvp=False (gamma=0.5 wmin=0.3 eta=0.05 margin=0.0 warmup=10) [detached neg-relaxation on OVLI; off => OVLI byte-identical]
  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
  out_dir=/home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090
======================================================================
=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
  -----------------------------------------------------------
  subset   | # imgs | # pids | # cams | aerial | ground
  -----------------------------------------------------------
  train    |  51530 |    807 |      3 |  21217 |  30313
  query    |   4167 |    534 |      2 |   2356 |   1811
  gallery  |  20709 |    534 |      2 |  14362 |   6347
  -----------------------------------------------------------
unloaded parameters: <All keys matched successfully>
  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [333 backbone tensors, 2 head tensors] -- prevents the epoch-8 Swin collapse

logs/agreidv2_baseline_4090_s2.log
======================================================================
CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/home/afr/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
  use_afd=False  ovp=False (lambda=0.5 tau=0.05 mom=0.2)
  ovli=False (lambda=0.5 tau=0.05 alpha=0.5 dim=256 grid=(8, 4) warmup=10 rerank=False pool=mean topk=8 thresh=0.0 cand=oppview match=maxsim align=free setpool=mean setpool_residual=True)
  acvp=False (gamma=0.5 wmin=0.3 eta=0.05 margin=0.0 warmup=10) [detached neg-relaxation on OVLI; off => OVLI byte-identical]
  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
  out_dir=/home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090_s2
======================================================================
=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
  -----------------------------------------------------------
  subset   | # imgs | # pids | # cams | aerial | ground
  -----------------------------------------------------------
  train    |  51530 |    807 |      3 |  21217 |  30313
  query    |   4167 |    534 |      2 |   2356 |   1811
  gallery  |  20709 |    534 |      2 |  14362 |   6347
  -----------------------------------------------------------
unloaded parameters: <All keys matched successfully>
  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [333 backbone tensors, 2 head tensors] -- prevents the epoch-8 Swin collapse


exec
/bin/zsh -lc 'for f in logs/*.log; do echo "==== $f"; rg -n "''^    '"\\[(A->G|G->A|mean)\\]|"'^  ---- AIRL|''^Training finished|new best mean|AIRL-ISO'"\\[|AIRL-DUAL\\[\" \""'$f"; done' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
==== logs/agreidv2_airl_4090.log
40:Epoch[1] done in 113.2s  Loss=25.389 Acc=0.116 AIRL-ISO[lam_eff=0.100 ce_rec=6.655 consistency=0.0002 deg_scale_mean=0.625 n_ground=28680]
55:Epoch[2] done in 112.9s  Loss=14.447 Acc=0.482 AIRL-ISO[lam_eff=0.200 ce_rec=5.800 consistency=0.0086 deg_scale_mean=0.624 n_ground=28677]
70:Epoch[3] done in 112.5s  Loss=9.258 Acc=0.646 AIRL-ISO[lam_eff=0.300 ce_rec=3.947 consistency=0.0379 deg_scale_mean=0.627 n_ground=28554]
85:Epoch[4] done in 112.7s  Loss=5.785 Acc=0.780 AIRL-ISO[lam_eff=0.400 ce_rec=2.473 consistency=0.0869 deg_scale_mean=0.626 n_ground=28676]
100:Epoch[5] done in 112.8s  Loss=4.024 Acc=0.893 AIRL-ISO[lam_eff=0.500 ce_rec=1.744 consistency=0.1331 deg_scale_mean=0.625 n_ground=28665]
115:Epoch[6] done in 113.9s  Loss=3.443 Acc=0.932 AIRL-ISO[lam_eff=0.500 ce_rec=1.493 consistency=0.1621 deg_scale_mean=0.623 n_ground=28888]
130:Epoch[7] done in 112.6s  Loss=3.262 Acc=0.944 AIRL-ISO[lam_eff=0.500 ce_rec=1.410 consistency=0.1741 deg_scale_mean=0.625 n_ground=28646]
145:Epoch[8] done in 113.7s  Loss=3.068 Acc=0.955 AIRL-ISO[lam_eff=0.500 ce_rec=1.342 consistency=0.1751 deg_scale_mean=0.624 n_ground=28818]
160:Epoch[9] done in 113.4s  Loss=3.009 Acc=0.957 AIRL-ISO[lam_eff=0.500 ce_rec=1.319 consistency=0.1735 deg_scale_mean=0.625 n_ground=28790]
175:Epoch[10] done in 113.0s  Loss=2.981 Acc=0.957 AIRL-ISO[lam_eff=0.500 ce_rec=1.309 consistency=0.1752 deg_scale_mean=0.624 n_ground=28708]
177:    [A->G] mAP=74.91  R1=83.36  R5=89.98  mINP=46.90
178:    [G->A] mAP=74.48  R1=82.94  R5=89.45  mINP=37.71
179:    [mean] mAP=74.70  R1=83.15
180:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 10 ----
181:    [A->G] full mAP=74.91 R1=83.36 | rec mAP=74.08 R1=82.98 | FUSE mAP=75.66 R1=83.74
182:    [G->A] full mAP=74.48 R1=82.94 | rec mAP=74.61 R1=82.94 | FUSE mAP=75.57 R1=84.04
183:    [mean] full=74.70 rec=74.35 FUSE=75.61  <- model-selection uses FUSE
184:    * new best mean mAP=75.61 (epoch 10) saved
199:Epoch[11] done in 113.0s  Loss=2.909 Acc=0.962 AIRL-ISO[lam_eff=0.500 ce_rec=1.291 consistency=0.1750 deg_scale_mean=0.624 n_ground=28664]
214:Epoch[12] done in 113.7s  Loss=2.870 Acc=0.963 AIRL-ISO[lam_eff=0.500 ce_rec=1.276 consistency=0.1736 deg_scale_mean=0.624 n_ground=28796]
229:Epoch[13] done in 112.5s  Loss=2.796 Acc=0.965 AIRL-ISO[lam_eff=0.500 ce_rec=1.255 consistency=0.1648 deg_scale_mean=0.625 n_ground=28553]
244:Epoch[14] done in 113.7s  Loss=2.743 Acc=0.969 AIRL-ISO[lam_eff=0.500 ce_rec=1.236 consistency=0.1572 deg_scale_mean=0.626 n_ground=28766]
259:Epoch[15] done in 113.7s  Loss=2.706 Acc=0.971 AIRL-ISO[lam_eff=0.500 ce_rec=1.228 consistency=0.1542 deg_scale_mean=0.623 n_ground=28765]
274:Epoch[16] done in 113.8s  Loss=2.661 Acc=0.974 AIRL-ISO[lam_eff=0.500 ce_rec=1.209 consistency=0.1487 deg_scale_mean=0.625 n_ground=28770]
289:Epoch[17] done in 113.3s  Loss=2.627 Acc=0.975 AIRL-ISO[lam_eff=0.500 ce_rec=1.198 consistency=0.1444 deg_scale_mean=0.625 n_ground=28701]
304:Epoch[18] done in 113.0s  Loss=2.578 Acc=0.978 AIRL-ISO[lam_eff=0.500 ce_rec=1.185 consistency=0.1383 deg_scale_mean=0.623 n_ground=28678]
319:Epoch[19] done in 113.0s  Loss=2.536 Acc=0.980 AIRL-ISO[lam_eff=0.500 ce_rec=1.172 consistency=0.1314 deg_scale_mean=0.624 n_ground=28643]
334:Epoch[20] done in 113.5s  Loss=2.520 Acc=0.979 AIRL-ISO[lam_eff=0.500 ce_rec=1.166 consistency=0.1255 deg_scale_mean=0.625 n_ground=28701]
336:    [A->G] mAP=72.59  R1=83.11  R5=89.52  mINP=41.54
337:    [G->A] mAP=73.67  R1=83.16  R5=89.45  mINP=35.62
338:    [mean] mAP=73.13  R1=83.13
339:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 20 ----
340:    [A->G] full mAP=72.59 R1=83.11 | rec mAP=74.54 R1=83.62 | FUSE mAP=74.44 R1=84.30
341:    [G->A] full mAP=73.67 R1=83.16 | rec mAP=73.89 R1=83.27 | FUSE mAP=75.19 R1=83.99
342:    [mean] full=73.13 rec=74.21 FUSE=74.82  <- model-selection uses FUSE
357:Epoch[21] done in 113.0s  Loss=2.491 Acc=0.982 AIRL-ISO[lam_eff=0.500 ce_rec=1.156 consistency=0.1215 deg_scale_mean=0.625 n_ground=28669]
372:Epoch[22] done in 112.7s  Loss=2.468 Acc=0.983 AIRL-ISO[lam_eff=0.500 ce_rec=1.148 consistency=0.1170 deg_scale_mean=0.625 n_ground=28572]
387:Epoch[23] done in 113.7s  Loss=2.446 Acc=0.983 AIRL-ISO[lam_eff=0.500 ce_rec=1.143 consistency=0.1118 deg_scale_mean=0.625 n_ground=28797]
402:Epoch[24] done in 113.4s  Loss=2.423 Acc=0.984 AIRL-ISO[lam_eff=0.500 ce_rec=1.137 consistency=0.1091 deg_scale_mean=0.624 n_ground=28788]
417:Epoch[25] done in 112.8s  Loss=2.400 Acc=0.986 AIRL-ISO[lam_eff=0.500 ce_rec=1.128 consistency=0.1050 deg_scale_mean=0.628 n_ground=28670]
432:Epoch[26] done in 113.4s  Loss=2.379 Acc=0.986 AIRL-ISO[lam_eff=0.500 ce_rec=1.121 consistency=0.0995 deg_scale_mean=0.624 n_ground=28726]
447:Epoch[27] done in 113.1s  Loss=2.359 Acc=0.987 AIRL-ISO[lam_eff=0.500 ce_rec=1.115 consistency=0.0961 deg_scale_mean=0.625 n_ground=28699]
462:Epoch[28] done in 113.7s  Loss=2.337 Acc=0.989 AIRL-ISO[lam_eff=0.500 ce_rec=1.107 consistency=0.0934 deg_scale_mean=0.626 n_ground=28769]
477:Epoch[29] done in 113.5s  Loss=2.329 Acc=0.988 AIRL-ISO[lam_eff=0.500 ce_rec=1.106 consistency=0.0901 deg_scale_mean=0.624 n_ground=28805]
492:Epoch[30] done in 113.4s  Loss=2.307 Acc=0.989 AIRL-ISO[lam_eff=0.500 ce_rec=1.097 consistency=0.0878 deg_scale_mean=0.625 n_ground=28734]
494:    [A->G] mAP=72.50  R1=81.07  R5=88.41  mINP=44.80
495:    [G->A] mAP=73.73  R1=83.27  R5=88.40  mINP=38.48
496:    [mean] mAP=73.11  R1=82.17
497:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 30 ----
498:    [A->G] full mAP=72.50 R1=81.07 | rec mAP=73.20 R1=81.83 | FUSE mAP=74.31 R1=83.15
499:    [G->A] full mAP=73.73 R1=83.27 | rec mAP=73.70 R1=82.27 | FUSE mAP=75.06 R1=83.77
500:    [mean] full=73.11 rec=73.45 FUSE=74.69  <- model-selection uses FUSE
515:Epoch[31] done in 112.3s  Loss=2.297 Acc=0.989 AIRL-ISO[lam_eff=0.500 ce_rec=1.094 consistency=0.0845 deg_scale_mean=0.624 n_ground=28589]
530:Epoch[32] done in 113.6s  Loss=2.272 Acc=0.991 AIRL-ISO[lam_eff=0.500 ce_rec=1.085 consistency=0.0800 deg_scale_mean=0.625 n_ground=28842]
545:Epoch[33] done in 113.7s  Loss=2.264 Acc=0.991 AIRL-ISO[lam_eff=0.500 ce_rec=1.084 consistency=0.0773 deg_scale_mean=0.626 n_ground=28773]
560:Epoch[34] done in 113.7s  Loss=2.251 Acc=0.992 AIRL-ISO[lam_eff=0.500 ce_rec=1.078 consistency=0.0763 deg_scale_mean=0.625 n_ground=28755]
575:Epoch[35] done in 113.7s  Loss=2.229 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.072 consistency=0.0733 deg_scale_mean=0.627 n_ground=28727]
590:Epoch[36] done in 112.6s  Loss=2.222 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.070 consistency=0.0714 deg_scale_mean=0.627 n_ground=28593]
605:Epoch[37] done in 113.8s  Loss=2.208 Acc=0.994 AIRL-ISO[lam_eff=0.500 ce_rec=1.065 consistency=0.0680 deg_scale_mean=0.622 n_ground=28866]
620:Epoch[38] done in 114.3s  Loss=2.201 Acc=0.994 AIRL-ISO[lam_eff=0.500 ce_rec=1.062 consistency=0.0667 deg_scale_mean=0.627 n_ground=28935]
635:Epoch[39] done in 113.6s  Loss=2.193 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.059 consistency=0.0661 deg_scale_mean=0.625 n_ground=28814]
650:Epoch[40] done in 113.1s  Loss=2.176 Acc=0.994 AIRL-ISO[lam_eff=0.500 ce_rec=1.054 consistency=0.0623 deg_scale_mean=0.626 n_ground=28706]
652:    [A->G] mAP=76.70  R1=85.10  R5=91.17  mINP=49.07
653:    [G->A] mAP=76.49  R1=84.21  R5=89.95  mINP=42.15
654:    [mean] mAP=76.60  R1=84.65
655:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 40 ----
656:    [A->G] full mAP=76.70 R1=85.10 | rec mAP=75.93 R1=83.40 | FUSE mAP=78.02 R1=85.70
657:    [G->A] full mAP=76.49 R1=84.21 | rec mAP=75.44 R1=84.43 | FUSE mAP=77.87 R1=85.42
658:    [mean] full=76.60 rec=75.68 FUSE=77.95  <- model-selection uses FUSE
659:    * new best mean mAP=77.95 (epoch 40) saved
674:Epoch[41] done in 114.0s  Loss=2.172 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.052 consistency=0.0609 deg_scale_mean=0.626 n_ground=28916]
689:Epoch[42] done in 112.8s  Loss=2.161 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.049 consistency=0.0585 deg_scale_mean=0.627 n_ground=28528]
704:Epoch[43] done in 113.2s  Loss=2.155 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.048 consistency=0.0571 deg_scale_mean=0.625 n_ground=28699]
719:Epoch[44] done in 113.2s  Loss=2.150 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.045 consistency=0.0554 deg_scale_mean=0.625 n_ground=28734]
734:Epoch[45] done in 113.7s  Loss=2.142 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.043 consistency=0.0544 deg_scale_mean=0.625 n_ground=28796]
749:Epoch[46] done in 112.8s  Loss=2.133 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.039 consistency=0.0535 deg_scale_mean=0.625 n_ground=28635]
764:Epoch[47] done in 113.2s  Loss=2.127 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.038 consistency=0.0525 deg_scale_mean=0.625 n_ground=28675]
779:Epoch[48] done in 112.8s  Loss=2.127 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.037 consistency=0.0523 deg_scale_mean=0.626 n_ground=28637]
794:Epoch[49] done in 113.5s  Loss=2.123 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.035 consistency=0.0509 deg_scale_mean=0.623 n_ground=28779]
809:Epoch[50] done in 113.6s  Loss=2.113 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.033 consistency=0.0492 deg_scale_mean=0.625 n_ground=28817]
811:    [A->G] mAP=78.34  R1=86.29  R5=91.85  mINP=52.48
812:    [G->A] mAP=79.19  R1=86.69  R5=90.61  mINP=46.79
813:    [mean] mAP=78.77  R1=86.49
814:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 50 ----
815:    [A->G] full mAP=78.34 R1=86.29 | rec mAP=77.75 R1=85.14 | FUSE mAP=79.63 R1=87.14
816:    [G->A] full mAP=79.19 R1=86.69 | rec mAP=77.15 R1=85.15 | FUSE mAP=80.18 R1=87.24
817:    [mean] full=78.77 rec=77.45 FUSE=79.90  <- model-selection uses FUSE
818:    * new best mean mAP=79.90 (epoch 50) saved
833:Epoch[51] done in 113.5s  Loss=2.112 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.032 consistency=0.0491 deg_scale_mean=0.625 n_ground=28774]
848:Epoch[52] done in 113.9s  Loss=2.108 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.031 consistency=0.0482 deg_scale_mean=0.625 n_ground=28778]
863:Epoch[53] done in 114.0s  Loss=2.107 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.030 consistency=0.0476 deg_scale_mean=0.624 n_ground=28880]
878:Epoch[54] done in 113.7s  Loss=2.105 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.030 consistency=0.0479 deg_scale_mean=0.625 n_ground=28800]
893:Epoch[55] done in 113.7s  Loss=2.103 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.029 consistency=0.0472 deg_scale_mean=0.626 n_ground=28809]
908:Epoch[56] done in 114.1s  Loss=2.101 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0466 deg_scale_mean=0.625 n_ground=28838]
923:Epoch[57] done in 113.5s  Loss=2.101 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0468 deg_scale_mean=0.623 n_ground=28746]
938:Epoch[58] done in 113.9s  Loss=2.101 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0469 deg_scale_mean=0.628 n_ground=28817]
953:Epoch[59] done in 113.6s  Loss=2.098 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0466 deg_scale_mean=0.624 n_ground=28774]
968:Epoch[60] done in 114.0s  Loss=2.097 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.027 consistency=0.0458 deg_scale_mean=0.624 n_ground=28813]
970:    [A->G] mAP=79.47  R1=87.18  R5=92.57  mINP=54.19
971:    [G->A] mAP=80.33  R1=87.47  R5=91.83  mINP=47.95
972:    [mean] mAP=79.90  R1=87.32
973:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 60 ----
974:    [A->G] full mAP=79.47 R1=87.18 | rec mAP=78.70 R1=84.93 | FUSE mAP=80.67 R1=87.90
975:    [G->A] full mAP=80.33 R1=87.47 | rec mAP=78.38 R1=86.58 | FUSE mAP=81.29 R1=88.02
976:    [mean] full=79.90 rec=78.54 FUSE=80.98  <- model-selection uses FUSE
977:    * new best mean mAP=80.98 (epoch 60) saved
979:Training finished. Best mean A<->G mAP=80.98 @ epoch 60
==== logs/agreidv2_airl_iso.log
40:Epoch[1] done in 234.6s  Loss=25.287 Acc=0.117 AIRL-ISO[lam_eff=0.100 ce_rec=6.655 consistency=0.0002 deg_scale_mean=0.625 n_ground=28771]
55:Epoch[2] done in 235.3s  Loss=14.438 Acc=0.483 AIRL-ISO[lam_eff=0.200 ce_rec=5.788 consistency=0.0088 deg_scale_mean=0.624 n_ground=28774]
70:Epoch[3] done in 237.1s  Loss=9.276 Acc=0.648 AIRL-ISO[lam_eff=0.300 ce_rec=3.941 consistency=0.0381 deg_scale_mean=0.627 n_ground=28670]
85:Epoch[4] done in 240.9s  Loss=5.621 Acc=0.790 AIRL-ISO[lam_eff=0.400 ce_rec=2.430 consistency=0.0872 deg_scale_mean=0.625 n_ground=28860]
100:Epoch[5] done in 240.6s  Loss=3.997 Acc=0.895 AIRL-ISO[lam_eff=0.500 ce_rec=1.733 consistency=0.1346 deg_scale_mean=0.625 n_ground=28856]
116:Epoch[6] done in 241.4s  Loss=3.465 Acc=0.931 AIRL-ISO[lam_eff=0.500 ce_rec=1.496 consistency=0.1647 deg_scale_mean=0.624 n_ground=28935]
131:Epoch[7] done in 240.6s  Loss=3.234 Acc=0.947 AIRL-ISO[lam_eff=0.500 ce_rec=1.403 consistency=0.1735 deg_scale_mean=0.625 n_ground=28828]
146:Epoch[8] done in 238.7s  Loss=3.067 Acc=0.956 AIRL-ISO[lam_eff=0.500 ce_rec=1.341 consistency=0.1775 deg_scale_mean=0.624 n_ground=28656]
161:Epoch[9] done in 238.4s  Loss=3.010 Acc=0.956 AIRL-ISO[lam_eff=0.500 ce_rec=1.319 consistency=0.1772 deg_scale_mean=0.625 n_ground=28598]
176:Epoch[10] done in 239.8s  Loss=2.947 Acc=0.960 AIRL-ISO[lam_eff=0.500 ce_rec=1.299 consistency=0.1776 deg_scale_mean=0.624 n_ground=28763]
178:    [A->G] mAP=71.36  R1=79.84  R5=87.61  mINP=43.40
179:    [G->A] mAP=71.86  R1=81.61  R5=88.07  mINP=35.13
180:    [mean] mAP=71.61  R1=80.73
181:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 10 ----
182:    [A->G] full mAP=71.36 R1=79.84 | rec mAP=72.28 R1=81.20 | FUSE mAP=72.89 R1=81.20
183:    [G->A] full mAP=71.86 R1=81.61 | rec mAP=72.38 R1=81.61 | FUSE mAP=73.53 R1=82.99
184:    [mean] full=71.61 rec=72.33 FUSE=73.21  <- model-selection uses FUSE
185:    * new best mean mAP=73.21 (epoch 10) saved
200:Epoch[11] done in 231.7s  Loss=2.915 Acc=0.961 AIRL-ISO[lam_eff=0.500 ce_rec=1.293 consistency=0.1781 deg_scale_mean=0.624 n_ground=28694]
215:Epoch[12] done in 233.9s  Loss=2.858 Acc=0.963 AIRL-ISO[lam_eff=0.500 ce_rec=1.274 consistency=0.1719 deg_scale_mean=0.625 n_ground=28735]
230:Epoch[13] done in 234.3s  Loss=2.792 Acc=0.967 AIRL-ISO[lam_eff=0.500 ce_rec=1.252 consistency=0.1663 deg_scale_mean=0.624 n_ground=28655]
245:Epoch[14] done in 236.4s  Loss=2.735 Acc=0.969 AIRL-ISO[lam_eff=0.500 ce_rec=1.233 consistency=0.1618 deg_scale_mean=0.626 n_ground=28817]
260:Epoch[15] done in 238.3s  Loss=2.696 Acc=0.971 AIRL-ISO[lam_eff=0.500 ce_rec=1.221 consistency=0.1557 deg_scale_mean=0.623 n_ground=28659]
275:Epoch[16] done in 238.1s  Loss=2.653 Acc=0.974 AIRL-ISO[lam_eff=0.500 ce_rec=1.208 consistency=0.1514 deg_scale_mean=0.625 n_ground=28697]
290:Epoch[17] done in 239.0s  Loss=2.603 Acc=0.977 AIRL-ISO[lam_eff=0.500 ce_rec=1.194 consistency=0.1433 deg_scale_mean=0.624 n_ground=28762]
305:Epoch[18] done in 239.3s  Loss=2.571 Acc=0.978 AIRL-ISO[lam_eff=0.500 ce_rec=1.182 consistency=0.1377 deg_scale_mean=0.623 n_ground=28736]
320:Epoch[19] done in 240.1s  Loss=2.541 Acc=0.979 AIRL-ISO[lam_eff=0.500 ce_rec=1.174 consistency=0.1322 deg_scale_mean=0.624 n_ground=28722]
335:Epoch[20] done in 240.7s  Loss=2.522 Acc=0.980 AIRL-ISO[lam_eff=0.500 ce_rec=1.168 consistency=0.1265 deg_scale_mean=0.625 n_ground=28838]
337:    [A->G] mAP=70.65  R1=80.18  R5=87.82  mINP=40.17
338:    [G->A] mAP=71.55  R1=82.61  R5=88.02  mINP=32.51
339:    [mean] mAP=71.10  R1=81.39
340:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 20 ----
341:    [A->G] full mAP=70.65 R1=80.18 | rec mAP=72.52 R1=81.75 | FUSE mAP=72.83 R1=81.45
342:    [G->A] full mAP=71.55 R1=82.61 | rec mAP=71.05 R1=81.50 | FUSE mAP=73.14 R1=83.38
343:    [mean] full=71.10 rec=71.79 FUSE=72.99  <- model-selection uses FUSE
358:Epoch[21] done in 231.0s  Loss=2.491 Acc=0.981 AIRL-ISO[lam_eff=0.500 ce_rec=1.156 consistency=0.1225 deg_scale_mean=0.624 n_ground=28621]
373:Epoch[22] done in 234.9s  Loss=2.462 Acc=0.983 AIRL-ISO[lam_eff=0.500 ce_rec=1.148 consistency=0.1170 deg_scale_mean=0.625 n_ground=28842]
388:Epoch[23] done in 235.7s  Loss=2.457 Acc=0.981 AIRL-ISO[lam_eff=0.500 ce_rec=1.147 consistency=0.1139 deg_scale_mean=0.626 n_ground=28837]
403:Epoch[24] done in 237.3s  Loss=2.411 Acc=0.986 AIRL-ISO[lam_eff=0.500 ce_rec=1.131 consistency=0.1082 deg_scale_mean=0.624 n_ground=28829]
418:Epoch[25] done in 238.6s  Loss=2.401 Acc=0.986 AIRL-ISO[lam_eff=0.500 ce_rec=1.127 consistency=0.1061 deg_scale_mean=0.627 n_ground=28728]
433:Epoch[26] done in 239.4s  Loss=2.382 Acc=0.986 AIRL-ISO[lam_eff=0.500 ce_rec=1.122 consistency=0.1008 deg_scale_mean=0.625 n_ground=28699]
448:Epoch[27] done in 238.6s  Loss=2.359 Acc=0.988 AIRL-ISO[lam_eff=0.500 ce_rec=1.114 consistency=0.0982 deg_scale_mean=0.625 n_ground=28673]
463:Epoch[28] done in 239.6s  Loss=2.346 Acc=0.987 AIRL-ISO[lam_eff=0.500 ce_rec=1.110 consistency=0.0931 deg_scale_mean=0.627 n_ground=28746]
478:Epoch[29] done in 240.6s  Loss=2.319 Acc=0.989 AIRL-ISO[lam_eff=0.500 ce_rec=1.103 consistency=0.0903 deg_scale_mean=0.624 n_ground=28853]
493:Epoch[30] done in 240.9s  Loss=2.302 Acc=0.990 AIRL-ISO[lam_eff=0.500 ce_rec=1.095 consistency=0.0856 deg_scale_mean=0.625 n_ground=28873]
495:    [A->G] mAP=73.46  R1=82.77  R5=89.60  mINP=44.99
496:    [G->A] mAP=73.80  R1=82.50  R5=88.90  mINP=37.87
497:    [mean] mAP=73.63  R1=82.63
498:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 30 ----
499:    [A->G] full mAP=73.46 R1=82.77 | rec mAP=73.34 R1=82.00 | FUSE mAP=74.80 R1=83.36
500:    [G->A] full mAP=73.80 R1=82.50 | rec mAP=73.06 R1=82.44 | FUSE mAP=75.26 R1=83.55
501:    [mean] full=73.63 rec=73.20 FUSE=75.03  <- model-selection uses FUSE
502:    * new best mean mAP=75.03 (epoch 30) saved
517:Epoch[31] done in 232.8s  Loss=2.297 Acc=0.989 AIRL-ISO[lam_eff=0.500 ce_rec=1.093 consistency=0.0839 deg_scale_mean=0.624 n_ground=28809]
532:Epoch[32] done in 234.3s  Loss=2.267 Acc=0.992 AIRL-ISO[lam_eff=0.500 ce_rec=1.084 consistency=0.0797 deg_scale_mean=0.624 n_ground=28779]
547:Epoch[33] done in 235.0s  Loss=2.267 Acc=0.990 AIRL-ISO[lam_eff=0.500 ce_rec=1.084 consistency=0.0788 deg_scale_mean=0.628 n_ground=28718]
562:Epoch[34] done in 235.6s  Loss=2.241 Acc=0.992 AIRL-ISO[lam_eff=0.500 ce_rec=1.076 consistency=0.0752 deg_scale_mean=0.626 n_ground=28692]
577:Epoch[35] done in 238.5s  Loss=2.232 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.072 consistency=0.0722 deg_scale_mean=0.626 n_ground=28770]
592:Epoch[36] done in 239.3s  Loss=2.219 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.069 consistency=0.0690 deg_scale_mean=0.627 n_ground=28773]
607:Epoch[37] done in 240.1s  Loss=2.214 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.067 consistency=0.0682 deg_scale_mean=0.624 n_ground=28831]
622:Epoch[38] done in 240.2s  Loss=2.198 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.062 consistency=0.0669 deg_scale_mean=0.627 n_ground=28802]
637:Epoch[39] done in 239.2s  Loss=2.194 Acc=0.993 AIRL-ISO[lam_eff=0.500 ce_rec=1.060 consistency=0.0646 deg_scale_mean=0.625 n_ground=28792]
652:Epoch[40] done in 239.4s  Loss=2.175 Acc=0.994 AIRL-ISO[lam_eff=0.500 ce_rec=1.053 consistency=0.0621 deg_scale_mean=0.625 n_ground=28741]
654:    [A->G] mAP=76.09  R1=83.87  R5=90.70  mINP=49.93
655:    [G->A] mAP=76.69  R1=84.82  R5=90.50  mINP=42.19
656:    [mean] mAP=76.39  R1=84.34
657:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 40 ----
658:    [A->G] full mAP=76.09 R1=83.87 | rec mAP=75.40 R1=83.23 | FUSE mAP=77.58 R1=85.44
659:    [G->A] full mAP=76.69 R1=84.82 | rec mAP=75.07 R1=83.88 | FUSE mAP=77.65 R1=85.37
660:    [mean] full=76.39 rec=75.23 FUSE=77.62  <- model-selection uses FUSE
661:    * new best mean mAP=77.62 (epoch 40) saved
676:Epoch[41] done in 232.2s  Loss=2.171 Acc=0.994 AIRL-ISO[lam_eff=0.500 ce_rec=1.053 consistency=0.0606 deg_scale_mean=0.625 n_ground=28778]
691:Epoch[42] done in 234.3s  Loss=2.164 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.050 consistency=0.0584 deg_scale_mean=0.627 n_ground=28749]
706:Epoch[43] done in 235.0s  Loss=2.156 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.047 consistency=0.0572 deg_scale_mean=0.625 n_ground=28769]
721:Epoch[44] done in 236.8s  Loss=2.151 Acc=0.995 AIRL-ISO[lam_eff=0.500 ce_rec=1.045 consistency=0.0559 deg_scale_mean=0.625 n_ground=28880]
736:Epoch[45] done in 239.5s  Loss=2.140 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.042 consistency=0.0546 deg_scale_mean=0.625 n_ground=28820]
751:Epoch[46] done in 241.2s  Loss=2.131 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.039 consistency=0.0528 deg_scale_mean=0.624 n_ground=28864]
766:Epoch[47] done in 237.3s  Loss=2.129 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.038 consistency=0.0521 deg_scale_mean=0.624 n_ground=28543]
781:Epoch[48] done in 239.6s  Loss=2.125 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.037 consistency=0.0518 deg_scale_mean=0.625 n_ground=28757]
796:Epoch[49] done in 239.2s  Loss=2.122 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.036 consistency=0.0509 deg_scale_mean=0.625 n_ground=28643]
811:Epoch[50] done in 239.6s  Loss=2.116 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.034 consistency=0.0497 deg_scale_mean=0.624 n_ground=28797]
==== logs/agreidv2_baseline.log
177:    [A->G] mAP=73.39  R1=82.60  R5=89.47  mINP=45.05
178:    [G->A] mAP=73.97  R1=82.99  R5=89.51  mINP=35.92
179:    [mean] mAP=73.68  R1=82.80
180:    * new best mean mAP=73.68 (epoch 10) saved
332:    [A->G] mAP=71.25  R1=80.22  R5=87.69  mINP=41.91
333:    [G->A] mAP=71.19  R1=80.73  R5=87.85  mINP=34.72
334:    [mean] mAP=71.22  R1=80.47
486:    [A->G] mAP=72.09  R1=80.65  R5=88.16  mINP=43.73
487:    [G->A] mAP=73.32  R1=82.88  R5=89.51  mINP=36.44
488:    [mean] mAP=72.71  R1=81.76
640:    [A->G] mAP=76.79  R1=84.08  R5=90.24  mINP=51.34
641:    [G->A] mAP=76.84  R1=84.98  R5=90.28  mINP=41.65
642:    [mean] mAP=76.82  R1=84.53
643:    * new best mean mAP=76.82 (epoch 40) saved
795:    [A->G] mAP=79.14  R1=85.78  R5=92.32  mINP=55.52
796:    [G->A] mAP=79.29  R1=86.97  R5=91.50  mINP=45.80
797:    [mean] mAP=79.22  R1=86.37
798:    * new best mean mAP=79.22 (epoch 50) saved
950:    [A->G] mAP=79.72  R1=86.42  R5=92.28  mINP=56.04
951:    [G->A] mAP=80.04  R1=87.80  R5=92.16  mINP=46.62
952:    [mean] mAP=79.88  R1=87.11
953:    * new best mean mAP=79.88 (epoch 60) saved
955:Training finished. Best mean A<->G mAP=79.88 @ epoch 60
==== logs/agreidv2_baseline_4090.log
176:    [A->G] mAP=73.40  R1=82.85  R5=89.26  mINP=44.92
177:    [G->A] mAP=72.69  R1=83.43  R5=89.45  mINP=35.03
178:    [mean] mAP=73.04  R1=83.14
179:    * new best mean mAP=73.04 (epoch 10) saved
331:    [A->G] mAP=72.37  R1=81.58  R5=89.05  mINP=43.54
332:    [G->A] mAP=73.14  R1=82.83  R5=88.85  mINP=36.99
333:    [mean] mAP=72.75  R1=82.20
485:    [A->G] mAP=71.72  R1=80.60  R5=87.90  mINP=43.02
486:    [G->A] mAP=73.24  R1=83.16  R5=88.96  mINP=35.74
487:    [mean] mAP=72.48  R1=81.88
639:    [A->G] mAP=76.21  R1=83.83  R5=91.00  mINP=48.95
640:    [G->A] mAP=77.13  R1=85.42  R5=91.17  mINP=42.14
641:    [mean] mAP=76.67  R1=84.63
642:    * new best mean mAP=76.67 (epoch 40) saved
794:    [A->G] mAP=79.75  R1=86.59  R5=92.44  mINP=55.00
795:    [G->A] mAP=80.22  R1=88.51  R5=92.21  mINP=46.92
796:    [mean] mAP=79.98  R1=87.55
797:    * new best mean mAP=79.98 (epoch 50) saved
949:    [A->G] mAP=80.73  R1=87.35  R5=93.00  mINP=56.91
950:    [G->A] mAP=81.42  R1=89.07  R5=93.21  mINP=48.32
951:    [mean] mAP=81.08  R1=88.21
952:    * new best mean mAP=81.08 (epoch 60) saved
954:Training finished. Best mean A<->G mAP=81.08 @ epoch 60
==== logs/agreidv2_baseline_4090_s2.log
176:    [A->G] mAP=74.43  R1=83.49  R5=89.90  mINP=45.26
177:    [G->A] mAP=74.11  R1=83.66  R5=90.50  mINP=36.04
178:    [mean] mAP=74.27  R1=83.57
179:    * new best mean mAP=74.27 (epoch 10) saved
331:    [A->G] mAP=70.89  R1=80.73  R5=88.07  mINP=40.20
332:    [G->A] mAP=70.82  R1=81.23  R5=87.36  mINP=32.87
333:    [mean] mAP=70.85  R1=80.98
485:    [A->G] mAP=72.48  R1=81.49  R5=89.05  mINP=44.27
486:    [G->A] mAP=73.37  R1=82.44  R5=89.12  mINP=36.52
487:    [mean] mAP=72.93  R1=81.97
639:    [A->G] mAP=76.85  R1=84.76  R5=91.26  mINP=49.81
640:    [G->A] mAP=76.72  R1=85.04  R5=90.17  mINP=42.94
641:    [mean] mAP=76.79  R1=84.90
642:    * new best mean mAP=76.79 (epoch 40) saved
794:    [A->G] mAP=79.31  R1=86.67  R5=91.68  mINP=54.35
795:    [G->A] mAP=79.77  R1=87.69  R5=92.10  mINP=46.88
796:    [mean] mAP=79.54  R1=87.18
797:    * new best mean mAP=79.54 (epoch 50) saved

exec
/bin/zsh -lc 'for f in logs/*.log; do echo "==== $f tail"; tail -n 80 "$f"; done' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
==== logs/agreidv2_airl_4090.log tail
Epoch[56] Iter[450/786] Loss: 2.111 CE: 1.051 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0493 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[500/786] Loss: 2.111 CE: 1.050 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0491 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[550/786] Loss: 2.109 CE: 1.049 Tri: 0.004 CE_rec: 1.031 AIRL_rec: 0.0486 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[600/786] Loss: 2.108 CE: 1.049 Tri: 0.004 CE_rec: 1.031 AIRL_rec: 0.0482 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[650/786] Loss: 2.106 CE: 1.048 Tri: 0.004 CE_rec: 1.030 AIRL_rec: 0.0477 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[700/786] Loss: 2.104 CE: 1.047 Tri: 0.004 CE_rec: 1.030 AIRL_rec: 0.0473 Acc: 0.997 LR: 8.57e-07
Epoch[56] done in 114.1s  Loss=2.101 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0466 deg_scale_mean=0.625 n_ground=28838]
Epoch[57] Iter[50/786] Loss: 2.123 CE: 1.055 Tri: 0.005 CE_rec: 1.039 AIRL_rec: 0.0492 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[100/786] Loss: 2.122 CE: 1.054 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0507 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[150/786] Loss: 2.120 CE: 1.052 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0510 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[200/786] Loss: 2.119 CE: 1.052 Tri: 0.007 CE_rec: 1.035 AIRL_rec: 0.0506 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[250/786] Loss: 2.120 CE: 1.052 Tri: 0.008 CE_rec: 1.035 AIRL_rec: 0.0505 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[300/786] Loss: 2.119 CE: 1.052 Tri: 0.007 CE_rec: 1.035 AIRL_rec: 0.0505 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[350/786] Loss: 2.117 CE: 1.052 Tri: 0.006 CE_rec: 1.035 AIRL_rec: 0.0504 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[400/786] Loss: 2.115 CE: 1.051 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0502 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[450/786] Loss: 2.112 CE: 1.050 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0497 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[500/786] Loss: 2.110 CE: 1.049 Tri: 0.005 CE_rec: 1.032 AIRL_rec: 0.0491 Acc: 0.997 LR: 5.50e-07
Epoch[57] Iter[550/786] Loss: 2.109 CE: 1.049 Tri: 0.005 CE_rec: 1.031 AIRL_rec: 0.0488 Acc: 0.997 LR: 5.50e-07
Epoch[57] Iter[600/786] Loss: 2.108 CE: 1.048 Tri: 0.005 CE_rec: 1.031 AIRL_rec: 0.0485 Acc: 0.997 LR: 5.50e-07
Epoch[57] Iter[650/786] Loss: 2.107 CE: 1.048 Tri: 0.005 CE_rec: 1.030 AIRL_rec: 0.0482 Acc: 0.997 LR: 5.50e-07
Epoch[57] Iter[700/786] Loss: 2.104 CE: 1.047 Tri: 0.005 CE_rec: 1.029 AIRL_rec: 0.0475 Acc: 0.997 LR: 5.50e-07
Epoch[57] done in 113.5s  Loss=2.101 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0468 deg_scale_mean=0.623 n_ground=28746]
Epoch[58] Iter[50/786] Loss: 2.128 CE: 1.058 Tri: 0.003 CE_rec: 1.042 AIRL_rec: 0.0516 Acc: 0.992 LR: 3.10e-07
Epoch[58] Iter[100/786] Loss: 2.128 CE: 1.057 Tri: 0.004 CE_rec: 1.041 AIRL_rec: 0.0527 Acc: 0.994 LR: 3.10e-07
Epoch[58] Iter[150/786] Loss: 2.122 CE: 1.055 Tri: 0.003 CE_rec: 1.038 AIRL_rec: 0.0522 Acc: 0.994 LR: 3.10e-07
Epoch[58] Iter[200/786] Loss: 2.119 CE: 1.054 Tri: 0.004 CE_rec: 1.036 AIRL_rec: 0.0511 Acc: 0.995 LR: 3.10e-07
Epoch[58] Iter[250/786] Loss: 2.118 CE: 1.053 Tri: 0.004 CE_rec: 1.036 AIRL_rec: 0.0506 Acc: 0.995 LR: 3.10e-07
Epoch[58] Iter[300/786] Loss: 2.118 CE: 1.053 Tri: 0.005 CE_rec: 1.035 AIRL_rec: 0.0504 Acc: 0.995 LR: 3.10e-07
Epoch[58] Iter[350/786] Loss: 2.116 CE: 1.051 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0501 Acc: 0.995 LR: 3.10e-07
Epoch[58] Iter[400/786] Loss: 2.113 CE: 1.051 Tri: 0.004 CE_rec: 1.034 AIRL_rec: 0.0497 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[450/786] Loss: 2.111 CE: 1.050 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0494 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[500/786] Loss: 2.110 CE: 1.049 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0492 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[550/786] Loss: 2.110 CE: 1.049 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0490 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[600/786] Loss: 2.108 CE: 1.048 Tri: 0.005 CE_rec: 1.031 AIRL_rec: 0.0486 Acc: 0.997 LR: 3.10e-07
Epoch[58] Iter[650/786] Loss: 2.106 CE: 1.047 Tri: 0.005 CE_rec: 1.030 AIRL_rec: 0.0482 Acc: 0.997 LR: 3.10e-07
Epoch[58] Iter[700/786] Loss: 2.104 CE: 1.046 Tri: 0.005 CE_rec: 1.029 AIRL_rec: 0.0476 Acc: 0.997 LR: 3.10e-07
Epoch[58] done in 113.9s  Loss=2.101 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0469 deg_scale_mean=0.628 n_ground=28817]
Epoch[59] Iter[50/786] Loss: 2.116 CE: 1.051 Tri: 0.006 CE_rec: 1.034 AIRL_rec: 0.0518 Acc: 0.998 LR: 1.38e-07
Epoch[59] Iter[100/786] Loss: 2.111 CE: 1.050 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0501 Acc: 0.998 LR: 1.38e-07
Epoch[59] Iter[150/786] Loss: 2.113 CE: 1.051 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0505 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[200/786] Loss: 2.113 CE: 1.050 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0505 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[250/786] Loss: 2.112 CE: 1.050 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0504 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[300/786] Loss: 2.111 CE: 1.050 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0499 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[350/786] Loss: 2.112 CE: 1.050 Tri: 0.005 CE_rec: 1.033 AIRL_rec: 0.0498 Acc: 0.996 LR: 1.38e-07
Epoch[59] Iter[400/786] Loss: 2.111 CE: 1.049 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0498 Acc: 0.996 LR: 1.38e-07
Epoch[59] Iter[450/786] Loss: 2.110 CE: 1.049 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0494 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[500/786] Loss: 2.108 CE: 1.048 Tri: 0.004 CE_rec: 1.031 AIRL_rec: 0.0490 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[550/786] Loss: 2.107 CE: 1.048 Tri: 0.004 CE_rec: 1.031 AIRL_rec: 0.0487 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[600/786] Loss: 2.105 CE: 1.047 Tri: 0.004 CE_rec: 1.030 AIRL_rec: 0.0483 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[650/786] Loss: 2.103 CE: 1.046 Tri: 0.004 CE_rec: 1.029 AIRL_rec: 0.0478 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[700/786] Loss: 2.101 CE: 1.045 Tri: 0.004 CE_rec: 1.029 AIRL_rec: 0.0472 Acc: 0.997 LR: 1.38e-07
Epoch[59] done in 113.6s  Loss=2.098 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.028 consistency=0.0466 deg_scale_mean=0.624 n_ground=28774]
Epoch[60] Iter[50/786] Loss: 2.112 CE: 1.049 Tri: 0.004 CE_rec: 1.034 AIRL_rec: 0.0500 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[100/786] Loss: 2.119 CE: 1.053 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0499 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[150/786] Loss: 2.118 CE: 1.052 Tri: 0.006 CE_rec: 1.035 AIRL_rec: 0.0501 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[200/786] Loss: 2.115 CE: 1.051 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0497 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[250/786] Loss: 2.113 CE: 1.050 Tri: 0.005 CE_rec: 1.034 AIRL_rec: 0.0495 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[300/786] Loss: 2.112 CE: 1.050 Tri: 0.004 CE_rec: 1.033 AIRL_rec: 0.0494 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[350/786] Loss: 2.109 CE: 1.049 Tri: 0.004 CE_rec: 1.032 AIRL_rec: 0.0491 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[400/786] Loss: 2.108 CE: 1.049 Tri: 0.003 CE_rec: 1.032 AIRL_rec: 0.0487 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[450/786] Loss: 2.108 CE: 1.048 Tri: 0.003 CE_rec: 1.032 AIRL_rec: 0.0484 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[500/786] Loss: 2.106 CE: 1.048 Tri: 0.003 CE_rec: 1.031 AIRL_rec: 0.0482 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[550/786] Loss: 2.105 CE: 1.047 Tri: 0.003 CE_rec: 1.030 AIRL_rec: 0.0479 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[600/786] Loss: 2.104 CE: 1.046 Tri: 0.004 CE_rec: 1.030 AIRL_rec: 0.0474 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[650/786] Loss: 2.102 CE: 1.046 Tri: 0.004 CE_rec: 1.029 AIRL_rec: 0.0469 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[700/786] Loss: 2.100 CE: 1.045 Tri: 0.004 CE_rec: 1.028 AIRL_rec: 0.0465 Acc: 0.997 LR: 3.45e-08
Epoch[60] done in 114.0s  Loss=2.097 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.027 consistency=0.0458 deg_scale_mean=0.624 n_ground=28813]
  ---- A<->G cross-view eval @ epoch 60 ----
    [A->G] mAP=79.47  R1=87.18  R5=92.57  mINP=54.19
    [G->A] mAP=80.33  R1=87.47  R5=91.83  mINP=47.95
    [mean] mAP=79.90  R1=87.32
  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 60 ----
    [A->G] full mAP=79.47 R1=87.18 | rec mAP=78.70 R1=84.93 | FUSE mAP=80.67 R1=87.90
    [G->A] full mAP=80.33 R1=87.47 | rec mAP=78.38 R1=86.58 | FUSE mAP=81.29 R1=88.02
    [mean] full=79.90 rec=78.54 FUSE=80.98  <- model-selection uses FUSE
    * new best mean mAP=80.98 (epoch 60) saved
======================================================================
Training finished. Best mean A<->G mAP=80.98 @ epoch 60
Checkpoints in /home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090
======================================================================
==== logs/agreidv2_airl_iso.log tail
Epoch[45] Iter[550/786] Loss: 2.153 CE: 1.070 Tri: 0.008 CE_rec: 1.047 AIRL_rec: 0.0574 Acc: 0.995 LR: 8.12e-06
Epoch[45] Iter[600/786] Loss: 2.151 CE: 1.069 Tri: 0.008 CE_rec: 1.046 AIRL_rec: 0.0569 Acc: 0.995 LR: 8.12e-06
Epoch[45] Iter[650/786] Loss: 2.148 CE: 1.068 Tri: 0.007 CE_rec: 1.045 AIRL_rec: 0.0562 Acc: 0.996 LR: 8.12e-06
Epoch[45] Iter[700/786] Loss: 2.145 CE: 1.066 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0554 Acc: 0.996 LR: 8.12e-06
Epoch[45] done in 239.5s  Loss=2.140 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.042 consistency=0.0546 deg_scale_mean=0.625 n_ground=28820]
Epoch[46] Iter[50/786] Loss: 2.185 CE: 1.083 Tri: 0.011 CE_rec: 1.062 AIRL_rec: 0.0578 Acc: 0.990 LR: 7.21e-06
Epoch[46] Iter[100/786] Loss: 2.168 CE: 1.076 Tri: 0.008 CE_rec: 1.055 AIRL_rec: 0.0569 Acc: 0.993 LR: 7.21e-06
Epoch[46] Iter[150/786] Loss: 2.161 CE: 1.073 Tri: 0.007 CE_rec: 1.051 AIRL_rec: 0.0576 Acc: 0.994 LR: 7.21e-06
Epoch[46] Iter[200/786] Loss: 2.160 CE: 1.072 Tri: 0.007 CE_rec: 1.051 AIRL_rec: 0.0582 Acc: 0.994 LR: 7.21e-06
Epoch[46] Iter[250/786] Loss: 2.158 CE: 1.071 Tri: 0.008 CE_rec: 1.050 AIRL_rec: 0.0582 Acc: 0.994 LR: 7.21e-06
Epoch[46] Iter[300/786] Loss: 2.154 CE: 1.070 Tri: 0.008 CE_rec: 1.048 AIRL_rec: 0.0584 Acc: 0.995 LR: 7.21e-06
Epoch[46] Iter[350/786] Loss: 2.150 CE: 1.068 Tri: 0.007 CE_rec: 1.046 AIRL_rec: 0.0579 Acc: 0.995 LR: 7.21e-06
Epoch[46] Iter[400/786] Loss: 2.149 CE: 1.067 Tri: 0.007 CE_rec: 1.046 AIRL_rec: 0.0573 Acc: 0.995 LR: 7.21e-06
Epoch[46] Iter[450/786] Loss: 2.146 CE: 1.066 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0568 Acc: 0.995 LR: 7.21e-06
Epoch[46] Iter[500/786] Loss: 2.144 CE: 1.065 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0563 Acc: 0.995 LR: 7.21e-06
Epoch[46] Iter[550/786] Loss: 2.143 CE: 1.064 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0558 Acc: 0.996 LR: 7.21e-06
Epoch[46] Iter[600/786] Loss: 2.140 CE: 1.063 Tri: 0.007 CE_rec: 1.042 AIRL_rec: 0.0553 Acc: 0.996 LR: 7.21e-06
Epoch[46] Iter[650/786] Loss: 2.138 CE: 1.062 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0546 Acc: 0.996 LR: 7.21e-06
Epoch[46] Iter[700/786] Loss: 2.135 CE: 1.061 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0538 Acc: 0.996 LR: 7.21e-06
Epoch[46] done in 241.2s  Loss=2.131 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.039 consistency=0.0528 deg_scale_mean=0.624 n_ground=28864]
Epoch[47] Iter[50/786] Loss: 2.168 CE: 1.075 Tri: 0.007 CE_rec: 1.055 AIRL_rec: 0.0589 Acc: 0.994 LR: 6.35e-06
Epoch[47] Iter[100/786] Loss: 2.161 CE: 1.073 Tri: 0.008 CE_rec: 1.050 AIRL_rec: 0.0584 Acc: 0.995 LR: 6.35e-06
Epoch[47] Iter[150/786] Loss: 2.153 CE: 1.070 Tri: 0.007 CE_rec: 1.047 AIRL_rec: 0.0579 Acc: 0.995 LR: 6.35e-06
Epoch[47] Iter[200/786] Loss: 2.151 CE: 1.069 Tri: 0.006 CE_rec: 1.047 AIRL_rec: 0.0574 Acc: 0.995 LR: 6.35e-06
Epoch[47] Iter[250/786] Loss: 2.150 CE: 1.068 Tri: 0.007 CE_rec: 1.046 AIRL_rec: 0.0575 Acc: 0.995 LR: 6.35e-06
Epoch[47] Iter[300/786] Loss: 2.148 CE: 1.067 Tri: 0.007 CE_rec: 1.045 AIRL_rec: 0.0572 Acc: 0.995 LR: 6.35e-06
Epoch[47] Iter[350/786] Loss: 2.148 CE: 1.067 Tri: 0.008 CE_rec: 1.045 AIRL_rec: 0.0569 Acc: 0.995 LR: 6.35e-06
Epoch[47] Iter[400/786] Loss: 2.145 CE: 1.066 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0564 Acc: 0.995 LR: 6.35e-06
Epoch[47] Iter[450/786] Loss: 2.144 CE: 1.065 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0560 Acc: 0.995 LR: 6.35e-06
Epoch[47] Iter[500/786] Loss: 2.142 CE: 1.064 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0553 Acc: 0.995 LR: 6.35e-06
Epoch[47] Iter[550/786] Loss: 2.139 CE: 1.063 Tri: 0.007 CE_rec: 1.042 AIRL_rec: 0.0548 Acc: 0.996 LR: 6.35e-06
Epoch[47] Iter[600/786] Loss: 2.137 CE: 1.062 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0542 Acc: 0.996 LR: 6.35e-06
Epoch[47] Iter[650/786] Loss: 2.134 CE: 1.061 Tri: 0.006 CE_rec: 1.040 AIRL_rec: 0.0534 Acc: 0.996 LR: 6.35e-06
Epoch[47] Iter[700/786] Loss: 2.132 CE: 1.060 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0527 Acc: 0.996 LR: 6.35e-06
Epoch[47] done in 237.3s  Loss=2.129 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.038 consistency=0.0521 deg_scale_mean=0.624 n_ground=28543]
Epoch[48] Iter[50/786] Loss: 2.150 CE: 1.069 Tri: 0.004 CE_rec: 1.049 AIRL_rec: 0.0557 Acc: 0.994 LR: 5.52e-06
Epoch[48] Iter[100/786] Loss: 2.155 CE: 1.069 Tri: 0.009 CE_rec: 1.049 AIRL_rec: 0.0574 Acc: 0.994 LR: 5.52e-06
Epoch[48] Iter[150/786] Loss: 2.153 CE: 1.069 Tri: 0.007 CE_rec: 1.048 AIRL_rec: 0.0572 Acc: 0.994 LR: 5.52e-06
Epoch[48] Iter[200/786] Loss: 2.150 CE: 1.068 Tri: 0.007 CE_rec: 1.047 AIRL_rec: 0.0567 Acc: 0.994 LR: 5.52e-06
Epoch[48] Iter[250/786] Loss: 2.148 CE: 1.067 Tri: 0.007 CE_rec: 1.046 AIRL_rec: 0.0564 Acc: 0.994 LR: 5.52e-06
Epoch[48] Iter[300/786] Loss: 2.146 CE: 1.066 Tri: 0.007 CE_rec: 1.045 AIRL_rec: 0.0562 Acc: 0.994 LR: 5.52e-06
Epoch[48] Iter[350/786] Loss: 2.145 CE: 1.066 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0559 Acc: 0.994 LR: 5.52e-06
Epoch[48] Iter[400/786] Loss: 2.142 CE: 1.065 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0559 Acc: 0.995 LR: 5.52e-06
Epoch[48] Iter[450/786] Loss: 2.140 CE: 1.064 Tri: 0.006 CE_rec: 1.042 AIRL_rec: 0.0554 Acc: 0.995 LR: 5.52e-06
Epoch[48] Iter[500/786] Loss: 2.138 CE: 1.063 Tri: 0.006 CE_rec: 1.041 AIRL_rec: 0.0550 Acc: 0.995 LR: 5.52e-06
Epoch[48] Iter[550/786] Loss: 2.136 CE: 1.062 Tri: 0.006 CE_rec: 1.041 AIRL_rec: 0.0546 Acc: 0.995 LR: 5.52e-06
Epoch[48] Iter[600/786] Loss: 2.133 CE: 1.061 Tri: 0.006 CE_rec: 1.040 AIRL_rec: 0.0538 Acc: 0.996 LR: 5.52e-06
Epoch[48] Iter[650/786] Loss: 2.131 CE: 1.060 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0533 Acc: 0.996 LR: 5.52e-06
Epoch[48] Iter[700/786] Loss: 2.128 CE: 1.058 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0526 Acc: 0.996 LR: 5.52e-06
Epoch[48] done in 239.6s  Loss=2.125 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.037 consistency=0.0518 deg_scale_mean=0.625 n_ground=28757]
Epoch[49] Iter[50/786] Loss: 2.149 CE: 1.068 Tri: 0.005 CE_rec: 1.050 AIRL_rec: 0.0531 Acc: 0.996 LR: 4.74e-06
Epoch[49] Iter[100/786] Loss: 2.151 CE: 1.069 Tri: 0.007 CE_rec: 1.048 AIRL_rec: 0.0543 Acc: 0.994 LR: 4.74e-06
Epoch[49] Iter[150/786] Loss: 2.149 CE: 1.067 Tri: 0.008 CE_rec: 1.046 AIRL_rec: 0.0561 Acc: 0.994 LR: 4.74e-06
Epoch[49] Iter[200/786] Loss: 2.145 CE: 1.065 Tri: 0.008 CE_rec: 1.044 AIRL_rec: 0.0564 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[250/786] Loss: 2.142 CE: 1.064 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0558 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[300/786] Loss: 2.141 CE: 1.064 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0555 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[350/786] Loss: 2.142 CE: 1.064 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0554 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[400/786] Loss: 2.140 CE: 1.063 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0552 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[450/786] Loss: 2.139 CE: 1.062 Tri: 0.007 CE_rec: 1.042 AIRL_rec: 0.0548 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[500/786] Loss: 2.136 CE: 1.061 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0543 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[550/786] Loss: 2.134 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0537 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[600/786] Loss: 2.132 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0530 Acc: 0.996 LR: 4.74e-06
Epoch[49] Iter[650/786] Loss: 2.129 CE: 1.058 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0524 Acc: 0.996 LR: 4.74e-06
Epoch[49] Iter[700/786] Loss: 2.126 CE: 1.057 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0517 Acc: 0.996 LR: 4.74e-06
Epoch[49] done in 239.2s  Loss=2.122 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.036 consistency=0.0509 deg_scale_mean=0.625 n_ground=28643]
Epoch[50] Iter[50/786] Loss: 2.137 CE: 1.064 Tri: 0.002 CE_rec: 1.044 AIRL_rec: 0.0546 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[100/786] Loss: 2.133 CE: 1.062 Tri: 0.003 CE_rec: 1.041 AIRL_rec: 0.0545 Acc: 0.997 LR: 4.02e-06
Epoch[50] Iter[150/786] Loss: 2.132 CE: 1.060 Tri: 0.005 CE_rec: 1.040 AIRL_rec: 0.0542 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[200/786] Loss: 2.133 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0544 Acc: 0.997 LR: 4.02e-06
Epoch[50] Iter[250/786] Loss: 2.136 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0551 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[300/786] Loss: 2.135 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0546 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[350/786] Loss: 2.134 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0544 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[400/786] Loss: 2.131 CE: 1.059 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0541 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[450/786] Loss: 2.129 CE: 1.058 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0537 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[500/786] Loss: 2.127 CE: 1.057 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0530 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[550/786] Loss: 2.126 CE: 1.056 Tri: 0.005 CE_rec: 1.038 AIRL_rec: 0.0525 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[600/786] Loss: 2.124 CE: 1.056 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0519 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[650/786] Loss: 2.122 CE: 1.055 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0514 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[700/786] Loss: 2.119 CE: 1.053 Tri: 0.006 CE_rec: 1.035 AIRL_rec: 0.0505 Acc: 0.997 LR: 4.02e-06
Epoch[50] done in 239.6s  Loss=2.116 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.034 consistency=0.0497 deg_scale_mean=0.624 n_ground=28797]
==== logs/agreidv2_baseline.log tail
Epoch[56] Iter[250/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[300/786] Loss: 1.056 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[350/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[400/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[450/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[500/786] Loss: 1.052 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[550/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[600/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[650/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[700/786] Loss: 1.050 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] done in 189.7s  Loss=1.049 Acc=0.997
Epoch[57] Iter[50/786] Loss: 1.074 CE: 1.055 Tri: 0.019 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[100/786] Loss: 1.066 CE: 1.054 Tri: 0.012 Acc: 0.995 LR: 5.50e-07
Epoch[57] Iter[150/786] Loss: 1.064 CE: 1.054 Tri: 0.010 Acc: 0.995 LR: 5.50e-07
Epoch[57] Iter[200/786] Loss: 1.062 CE: 1.053 Tri: 0.010 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[250/786] Loss: 1.061 CE: 1.052 Tri: 0.009 Acc: 0.995 LR: 5.50e-07
Epoch[57] Iter[300/786] Loss: 1.059 CE: 1.051 Tri: 0.008 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[350/786] Loss: 1.058 CE: 1.051 Tri: 0.007 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[400/786] Loss: 1.057 CE: 1.050 Tri: 0.006 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[450/786] Loss: 1.055 CE: 1.049 Tri: 0.006 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[500/786] Loss: 1.054 CE: 1.048 Tri: 0.006 Acc: 0.997 LR: 5.50e-07
Epoch[57] Iter[550/786] Loss: 1.054 CE: 1.048 Tri: 0.006 Acc: 0.997 LR: 5.50e-07
Epoch[57] Iter[600/786] Loss: 1.052 CE: 1.047 Tri: 0.005 Acc: 0.997 LR: 5.50e-07
Epoch[57] Iter[650/786] Loss: 1.052 CE: 1.046 Tri: 0.005 Acc: 0.997 LR: 5.50e-07
Epoch[57] Iter[700/786] Loss: 1.050 CE: 1.045 Tri: 0.005 Acc: 0.997 LR: 5.50e-07
Epoch[57] done in 190.9s  Loss=1.049 Acc=0.997
Epoch[58] Iter[50/786] Loss: 1.058 CE: 1.055 Tri: 0.003 Acc: 0.995 LR: 3.10e-07
Epoch[58] Iter[100/786] Loss: 1.055 CE: 1.053 Tri: 0.002 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[150/786] Loss: 1.055 CE: 1.052 Tri: 0.002 Acc: 0.997 LR: 3.10e-07
Epoch[58] Iter[200/786] Loss: 1.055 CE: 1.052 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[250/786] Loss: 1.055 CE: 1.052 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[300/786] Loss: 1.054 CE: 1.052 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[350/786] Loss: 1.054 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[400/786] Loss: 1.054 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[450/786] Loss: 1.053 CE: 1.050 Tri: 0.003 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[500/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[550/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
Epoch[58] Iter[600/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
Epoch[58] Iter[650/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
Epoch[58] Iter[700/786] Loss: 1.050 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
Epoch[58] done in 189.3s  Loss=1.049 Acc=0.997
Epoch[59] Iter[50/786] Loss: 1.066 CE: 1.056 Tri: 0.009 Acc: 0.994 LR: 1.38e-07
Epoch[59] Iter[100/786] Loss: 1.060 CE: 1.054 Tri: 0.006 Acc: 0.995 LR: 1.38e-07
Epoch[59] Iter[150/786] Loss: 1.058 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 1.38e-07
Epoch[59] Iter[200/786] Loss: 1.058 CE: 1.052 Tri: 0.005 Acc: 0.996 LR: 1.38e-07
Epoch[59] Iter[250/786] Loss: 1.056 CE: 1.052 Tri: 0.005 Acc: 0.996 LR: 1.38e-07
Epoch[59] Iter[300/786] Loss: 1.055 CE: 1.051 Tri: 0.005 Acc: 0.996 LR: 1.38e-07
Epoch[59] Iter[350/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[400/786] Loss: 1.054 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[450/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[500/786] Loss: 1.053 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[550/786] Loss: 1.052 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[600/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[650/786] Loss: 1.050 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[700/786] Loss: 1.049 CE: 1.045 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] done in 191.8s  Loss=1.048 Acc=0.998
Epoch[60] Iter[50/786] Loss: 1.064 CE: 1.054 Tri: 0.011 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[100/786] Loss: 1.060 CE: 1.052 Tri: 0.008 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[150/786] Loss: 1.057 CE: 1.050 Tri: 0.006 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[200/786] Loss: 1.057 CE: 1.050 Tri: 0.007 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[250/786] Loss: 1.056 CE: 1.050 Tri: 0.006 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[300/786] Loss: 1.055 CE: 1.050 Tri: 0.006 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[350/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[400/786] Loss: 1.054 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[450/786] Loss: 1.053 CE: 1.048 Tri: 0.005 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[500/786] Loss: 1.052 CE: 1.048 Tri: 0.005 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[550/786] Loss: 1.052 CE: 1.047 Tri: 0.005 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[600/786] Loss: 1.051 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[650/786] Loss: 1.050 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[700/786] Loss: 1.049 CE: 1.045 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
Epoch[60] done in 190.7s  Loss=1.047 Acc=0.998
  ---- A<->G cross-view eval @ epoch 60 ----
    [A->G] mAP=79.72  R1=86.42  R5=92.28  mINP=56.04
    [G->A] mAP=80.04  R1=87.80  R5=92.16  mINP=46.62
    [mean] mAP=79.88  R1=87.11
    * new best mean mAP=79.88 (epoch 60) saved
======================================================================
Training finished. Best mean A<->G mAP=79.88 @ epoch 60
Checkpoints in /root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_baseline
======================================================================
==== logs/agreidv2_baseline_4090.log tail
Epoch[56] Iter[250/786] Loss: 1.056 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[300/786] Loss: 1.054 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[350/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[400/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[450/786] Loss: 1.052 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[500/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[550/786] Loss: 1.051 CE: 1.048 Tri: 0.003 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[600/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[650/786] Loss: 1.050 CE: 1.046 Tri: 0.003 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[700/786] Loss: 1.049 CE: 1.045 Tri: 0.003 Acc: 0.997 LR: 8.57e-07
Epoch[56] done in 92.0s  Loss=1.047 Acc=0.997
Epoch[57] Iter[50/786] Loss: 1.060 CE: 1.058 Tri: 0.002 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[100/786] Loss: 1.058 CE: 1.054 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[150/786] Loss: 1.056 CE: 1.053 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[200/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[250/786] Loss: 1.059 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[300/786] Loss: 1.059 CE: 1.053 Tri: 0.005 Acc: 0.995 LR: 5.50e-07
Epoch[57] Iter[350/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.995 LR: 5.50e-07
Epoch[57] Iter[400/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[450/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[500/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[550/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[600/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[650/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 5.50e-07
Epoch[57] Iter[700/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 5.50e-07
Epoch[57] done in 91.5s  Loss=1.049 Acc=0.997
Epoch[58] Iter[50/786] Loss: 1.060 CE: 1.056 Tri: 0.004 Acc: 0.993 LR: 3.10e-07
Epoch[58] Iter[100/786] Loss: 1.058 CE: 1.054 Tri: 0.004 Acc: 0.994 LR: 3.10e-07
Epoch[58] Iter[150/786] Loss: 1.058 CE: 1.054 Tri: 0.004 Acc: 0.994 LR: 3.10e-07
Epoch[58] Iter[200/786] Loss: 1.058 CE: 1.053 Tri: 0.004 Acc: 0.995 LR: 3.10e-07
Epoch[58] Iter[250/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.995 LR: 3.10e-07
Epoch[58] Iter[300/786] Loss: 1.057 CE: 1.052 Tri: 0.004 Acc: 0.995 LR: 3.10e-07
Epoch[58] Iter[350/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.995 LR: 3.10e-07
Epoch[58] Iter[400/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[450/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[500/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[550/786] Loss: 1.053 CE: 1.048 Tri: 0.004 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[600/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.996 LR: 3.10e-07
Epoch[58] Iter[650/786] Loss: 1.050 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
Epoch[58] Iter[700/786] Loss: 1.049 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 3.10e-07
Epoch[58] done in 91.9s  Loss=1.048 Acc=0.997
Epoch[59] Iter[50/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.998 LR: 1.38e-07
Epoch[59] Iter[100/786] Loss: 1.054 CE: 1.051 Tri: 0.003 Acc: 0.998 LR: 1.38e-07
Epoch[59] Iter[150/786] Loss: 1.054 CE: 1.051 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[200/786] Loss: 1.054 CE: 1.051 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[250/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 1.38e-07
Epoch[59] Iter[300/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.996 LR: 1.38e-07
Epoch[59] Iter[350/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 1.38e-07
Epoch[59] Iter[400/786] Loss: 1.054 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 1.38e-07
Epoch[59] Iter[450/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 1.38e-07
Epoch[59] Iter[500/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[550/786] Loss: 1.052 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[600/786] Loss: 1.051 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[650/786] Loss: 1.050 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] Iter[700/786] Loss: 1.049 CE: 1.045 Tri: 0.004 Acc: 0.997 LR: 1.38e-07
Epoch[59] done in 91.5s  Loss=1.047 Acc=0.998
Epoch[60] Iter[50/786] Loss: 1.057 CE: 1.054 Tri: 0.003 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[100/786] Loss: 1.056 CE: 1.053 Tri: 0.003 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[150/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[200/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[250/786] Loss: 1.056 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[300/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[350/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[400/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.996 LR: 3.45e-08
Epoch[60] Iter[450/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[500/786] Loss: 1.051 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[550/786] Loss: 1.050 CE: 1.047 Tri: 0.003 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[600/786] Loss: 1.049 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[650/786] Loss: 1.049 CE: 1.045 Tri: 0.003 Acc: 0.997 LR: 3.45e-08
Epoch[60] Iter[700/786] Loss: 1.048 CE: 1.045 Tri: 0.003 Acc: 0.997 LR: 3.45e-08
Epoch[60] done in 91.9s  Loss=1.047 Acc=0.997
  ---- A<->G cross-view eval @ epoch 60 ----
    [A->G] mAP=80.73  R1=87.35  R5=93.00  mINP=56.91
    [G->A] mAP=81.42  R1=89.07  R5=93.21  mINP=48.32
    [mean] mAP=81.08  R1=88.21
    * new best mean mAP=81.08 (epoch 60) saved
======================================================================
Training finished. Best mean A<->G mAP=81.08 @ epoch 60
Checkpoints in /home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090
======================================================================
==== logs/agreidv2_baseline_4090_s2.log tail
Epoch[52] Iter[250/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
Epoch[52] Iter[300/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.996 LR: 2.72e-06
Epoch[52] Iter[350/786] Loss: 1.060 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 2.72e-06
Epoch[52] Iter[400/786] Loss: 1.059 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 2.72e-06
Epoch[52] Iter[450/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.996 LR: 2.72e-06
Epoch[52] Iter[500/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.996 LR: 2.72e-06
Epoch[52] Iter[550/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.996 LR: 2.72e-06
Epoch[52] Iter[600/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.996 LR: 2.72e-06
Epoch[52] Iter[650/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 2.72e-06
Epoch[52] Iter[700/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.997 LR: 2.72e-06
Epoch[52] done in 91.7s  Loss=1.053 Acc=0.997
Epoch[53] Iter[50/786] Loss: 1.071 CE: 1.061 Tri: 0.009 Acc: 0.995 LR: 2.16e-06
Epoch[53] Iter[100/786] Loss: 1.068 CE: 1.060 Tri: 0.008 Acc: 0.995 LR: 2.16e-06
Epoch[53] Iter[150/786] Loss: 1.067 CE: 1.060 Tri: 0.007 Acc: 0.995 LR: 2.16e-06
Epoch[53] Iter[200/786] Loss: 1.067 CE: 1.059 Tri: 0.008 Acc: 0.995 LR: 2.16e-06
Epoch[53] Iter[250/786] Loss: 1.066 CE: 1.058 Tri: 0.008 Acc: 0.995 LR: 2.16e-06
Epoch[53] Iter[300/786] Loss: 1.064 CE: 1.057 Tri: 0.007 Acc: 0.995 LR: 2.16e-06
Epoch[53] Iter[350/786] Loss: 1.063 CE: 1.056 Tri: 0.007 Acc: 0.995 LR: 2.16e-06
Epoch[53] Iter[400/786] Loss: 1.062 CE: 1.056 Tri: 0.006 Acc: 0.995 LR: 2.16e-06
Epoch[53] Iter[450/786] Loss: 1.061 CE: 1.055 Tri: 0.006 Acc: 0.995 LR: 2.16e-06
Epoch[53] Iter[500/786] Loss: 1.060 CE: 1.054 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
Epoch[53] Iter[550/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
Epoch[53] Iter[600/786] Loss: 1.058 CE: 1.052 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
Epoch[53] Iter[650/786] Loss: 1.057 CE: 1.051 Tri: 0.006 Acc: 0.996 LR: 2.16e-06
Epoch[53] Iter[700/786] Loss: 1.056 CE: 1.050 Tri: 0.005 Acc: 0.996 LR: 2.16e-06
Epoch[53] done in 91.2s  Loss=1.054 Acc=0.997
Epoch[54] Iter[50/786] Loss: 1.067 CE: 1.062 Tri: 0.005 Acc: 0.995 LR: 1.67e-06
Epoch[54] Iter[100/786] Loss: 1.063 CE: 1.059 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
Epoch[54] Iter[150/786] Loss: 1.060 CE: 1.057 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
Epoch[54] Iter[200/786] Loss: 1.060 CE: 1.056 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
Epoch[54] Iter[250/786] Loss: 1.060 CE: 1.056 Tri: 0.004 Acc: 0.995 LR: 1.67e-06
Epoch[54] Iter[300/786] Loss: 1.060 CE: 1.056 Tri: 0.004 Acc: 0.995 LR: 1.67e-06
Epoch[54] Iter[350/786] Loss: 1.059 CE: 1.055 Tri: 0.004 Acc: 0.995 LR: 1.67e-06
Epoch[54] Iter[400/786] Loss: 1.058 CE: 1.054 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
Epoch[54] Iter[450/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
Epoch[54] Iter[500/786] Loss: 1.058 CE: 1.053 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
Epoch[54] Iter[550/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
Epoch[54] Iter[600/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.996 LR: 1.67e-06
Epoch[54] Iter[650/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 1.67e-06
Epoch[54] Iter[700/786] Loss: 1.053 CE: 1.049 Tri: 0.004 Acc: 0.997 LR: 1.67e-06
Epoch[54] done in 90.8s  Loss=1.052 Acc=0.997
Epoch[55] Iter[50/786] Loss: 1.063 CE: 1.058 Tri: 0.005 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[100/786] Loss: 1.062 CE: 1.057 Tri: 0.006 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[150/786] Loss: 1.064 CE: 1.056 Tri: 0.008 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[200/786] Loss: 1.061 CE: 1.054 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[250/786] Loss: 1.062 CE: 1.055 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[300/786] Loss: 1.061 CE: 1.054 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[350/786] Loss: 1.060 CE: 1.053 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[400/786] Loss: 1.059 CE: 1.052 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[450/786] Loss: 1.059 CE: 1.052 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[500/786] Loss: 1.058 CE: 1.051 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[550/786] Loss: 1.057 CE: 1.050 Tri: 0.007 Acc: 0.997 LR: 1.23e-06
Epoch[55] Iter[600/786] Loss: 1.055 CE: 1.049 Tri: 0.006 Acc: 0.997 LR: 1.23e-06
Epoch[55] Iter[650/786] Loss: 1.054 CE: 1.048 Tri: 0.006 Acc: 0.997 LR: 1.23e-06
Epoch[55] Iter[700/786] Loss: 1.053 CE: 1.048 Tri: 0.006 Acc: 0.997 LR: 1.23e-06
Epoch[55] done in 90.8s  Loss=1.052 Acc=0.997
Epoch[56] Iter[50/786] Loss: 1.059 CE: 1.052 Tri: 0.007 Acc: 0.998 LR: 8.57e-07
Epoch[56] Iter[100/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[150/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[200/786] Loss: 1.057 CE: 1.053 Tri: 0.005 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[250/786] Loss: 1.058 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[300/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[350/786] Loss: 1.058 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[400/786] Loss: 1.057 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[450/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[500/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[550/786] Loss: 1.054 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[600/786] Loss: 1.053 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[650/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[700/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] done in 91.8s  Loss=1.049 Acc=0.997
Epoch[57] Iter[50/786] Loss: 1.053 CE: 1.050 Tri: 0.003 Acc: 0.997 LR: 5.50e-07
Epoch[57] Iter[100/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.995 LR: 5.50e-07
Epoch[57] Iter[150/786] Loss: 1.055 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[200/786] Loss: 1.054 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[250/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[300/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[350/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[400/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[450/786] Loss: 1.053 CE: 1.050 Tri: 0.003 Acc: 0.996 LR: 5.50e-07

exec
/bin/zsh -lc 'for f in logs/*.log; do echo "==== $f anomalies"; rg -n -i "nan|inf|kill|error|traceback|oom|collapse|warn|diverg" "$f" || true; done' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
==== logs/agreidv2_airl_4090.log anomalies
24:  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [361 backbone tensors, 4 head tensors] -- prevents the epoch-8 Swin collapse
==== logs/agreidv2_airl_iso.log anomalies
24:  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [361 backbone tensors, 4 head tensors] -- prevents the epoch-8 Swin collapse
==== logs/agreidv2_baseline.log anomalies
24:  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [333 backbone tensors, 2 head tensors] -- prevents the epoch-8 Swin collapse
==== logs/agreidv2_baseline_4090.log anomalies
24:  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [333 backbone tensors, 2 head tensors] -- prevents the epoch-8 Swin collapse
==== logs/agreidv2_baseline_4090_s2.log anomalies
24:  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [333 backbone tensors, 2 head tensors] -- prevents the epoch-8 Swin collapse

codex
AG-ReID.v2 的 4090 AIRL final 是完整 60 epoch；3090 `agreidv2_airl_iso.log` 只到 epoch 50 后的 tail，没有 epoch 60 结果。baseline 有 3090 seed1、4090 seed1、4090 seed2，其中 4090 seed2 当前日志也像是截到 epoch 57，还没有最终 eval。
exec
/bin/zsh -lc 'tail -n 40 logs/agreidv2_baseline_4090_s2.log' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
Epoch[54] done in 90.8s  Loss=1.052 Acc=0.997
Epoch[55] Iter[50/786] Loss: 1.063 CE: 1.058 Tri: 0.005 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[100/786] Loss: 1.062 CE: 1.057 Tri: 0.006 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[150/786] Loss: 1.064 CE: 1.056 Tri: 0.008 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[200/786] Loss: 1.061 CE: 1.054 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[250/786] Loss: 1.062 CE: 1.055 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[300/786] Loss: 1.061 CE: 1.054 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[350/786] Loss: 1.060 CE: 1.053 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[400/786] Loss: 1.059 CE: 1.052 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[450/786] Loss: 1.059 CE: 1.052 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[500/786] Loss: 1.058 CE: 1.051 Tri: 0.007 Acc: 0.996 LR: 1.23e-06
Epoch[55] Iter[550/786] Loss: 1.057 CE: 1.050 Tri: 0.007 Acc: 0.997 LR: 1.23e-06
Epoch[55] Iter[600/786] Loss: 1.055 CE: 1.049 Tri: 0.006 Acc: 0.997 LR: 1.23e-06
Epoch[55] Iter[650/786] Loss: 1.054 CE: 1.048 Tri: 0.006 Acc: 0.997 LR: 1.23e-06
Epoch[55] Iter[700/786] Loss: 1.053 CE: 1.048 Tri: 0.006 Acc: 0.997 LR: 1.23e-06
Epoch[55] done in 90.8s  Loss=1.052 Acc=0.997
Epoch[56] Iter[50/786] Loss: 1.059 CE: 1.052 Tri: 0.007 Acc: 0.998 LR: 8.57e-07
Epoch[56] Iter[100/786] Loss: 1.057 CE: 1.052 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[150/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[200/786] Loss: 1.057 CE: 1.053 Tri: 0.005 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[250/786] Loss: 1.058 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[300/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[350/786] Loss: 1.058 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 8.57e-07
Epoch[56] Iter[400/786] Loss: 1.057 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[450/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[500/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[550/786] Loss: 1.054 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[600/786] Loss: 1.053 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[650/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] Iter[700/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
Epoch[56] done in 91.8s  Loss=1.049 Acc=0.997
Epoch[57] Iter[50/786] Loss: 1.053 CE: 1.050 Tri: 0.003 Acc: 0.997 LR: 5.50e-07
Epoch[57] Iter[100/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.995 LR: 5.50e-07
Epoch[57] Iter[150/786] Loss: 1.055 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[200/786] Loss: 1.054 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[250/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[300/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[350/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[400/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
Epoch[57] Iter[450/786] Loss: 1.053 CE: 1.050 Tri: 0.003 Acc: 0.996 LR: 5.50e-07

exec
/bin/zsh -lc 'tail -n 25 logs/agreidv2_airl_iso.log' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
Epoch[49] Iter[300/786] Loss: 2.141 CE: 1.064 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0555 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[350/786] Loss: 2.142 CE: 1.064 Tri: 0.007 CE_rec: 1.044 AIRL_rec: 0.0554 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[400/786] Loss: 2.140 CE: 1.063 Tri: 0.007 CE_rec: 1.043 AIRL_rec: 0.0552 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[450/786] Loss: 2.139 CE: 1.062 Tri: 0.007 CE_rec: 1.042 AIRL_rec: 0.0548 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[500/786] Loss: 2.136 CE: 1.061 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0543 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[550/786] Loss: 2.134 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0537 Acc: 0.995 LR: 4.74e-06
Epoch[49] Iter[600/786] Loss: 2.132 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0530 Acc: 0.996 LR: 4.74e-06
Epoch[49] Iter[650/786] Loss: 2.129 CE: 1.058 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0524 Acc: 0.996 LR: 4.74e-06
Epoch[49] Iter[700/786] Loss: 2.126 CE: 1.057 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0517 Acc: 0.996 LR: 4.74e-06
Epoch[49] done in 239.2s  Loss=2.122 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.036 consistency=0.0509 deg_scale_mean=0.625 n_ground=28643]
Epoch[50] Iter[50/786] Loss: 2.137 CE: 1.064 Tri: 0.002 CE_rec: 1.044 AIRL_rec: 0.0546 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[100/786] Loss: 2.133 CE: 1.062 Tri: 0.003 CE_rec: 1.041 AIRL_rec: 0.0545 Acc: 0.997 LR: 4.02e-06
Epoch[50] Iter[150/786] Loss: 2.132 CE: 1.060 Tri: 0.005 CE_rec: 1.040 AIRL_rec: 0.0542 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[200/786] Loss: 2.133 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0544 Acc: 0.997 LR: 4.02e-06
Epoch[50] Iter[250/786] Loss: 2.136 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0551 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[300/786] Loss: 2.135 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0546 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[350/786] Loss: 2.134 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0544 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[400/786] Loss: 2.131 CE: 1.059 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0541 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[450/786] Loss: 2.129 CE: 1.058 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0537 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[500/786] Loss: 2.127 CE: 1.057 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0530 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[550/786] Loss: 2.126 CE: 1.056 Tri: 0.005 CE_rec: 1.038 AIRL_rec: 0.0525 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[600/786] Loss: 2.124 CE: 1.056 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0519 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[650/786] Loss: 2.122 CE: 1.055 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0514 Acc: 0.996 LR: 4.02e-06
Epoch[50] Iter[700/786] Loss: 2.119 CE: 1.053 Tri: 0.006 CE_rec: 1.035 AIRL_rec: 0.0505 Acc: 0.997 LR: 4.02e-06
Epoch[50] done in 239.6s  Loss=2.116 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.034 consistency=0.0497 deg_scale_mean=0.624 n_ground=28797]

exec
/bin/zsh -lc 'for f in logs/*.log; do echo "==== $f"; rg -n "seed|manual_seed|Random|torch|cuda|cudnn|determin" "$f" || true; done' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
exec
/bin/zsh -lc "for f in logs/*.log; do printf '%s: ' \""'$f"; rg -n "Epoch'"\\[60\\] done|Training finished|@ epoch 60|\\[mean\\]\" \""'$f" | tail -n 12; done' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
==== logs/agreidv2_airl_4090.log
==== logs/agreidv2_airl_iso.log
==== logs/agreidv2_baseline.log
==== logs/agreidv2_baseline_4090.log
==== logs/agreidv2_baseline_4090_s2.log

 succeeded in 0ms:
logs/agreidv2_airl_4090.log: 496:    [mean] mAP=73.11  R1=82.17
500:    [mean] full=73.11 rec=73.45 FUSE=74.69  <- model-selection uses FUSE
654:    [mean] mAP=76.60  R1=84.65
658:    [mean] full=76.60 rec=75.68 FUSE=77.95  <- model-selection uses FUSE
813:    [mean] mAP=78.77  R1=86.49
817:    [mean] full=78.77 rec=77.45 FUSE=79.90  <- model-selection uses FUSE
968:Epoch[60] done in 114.0s  Loss=2.097 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.027 consistency=0.0458 deg_scale_mean=0.624 n_ground=28813]
969:  ---- A<->G cross-view eval @ epoch 60 ----
972:    [mean] mAP=79.90  R1=87.32
973:  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 60 ----
976:    [mean] full=79.90 rec=78.54 FUSE=80.98  <- model-selection uses FUSE
979:Training finished. Best mean A<->G mAP=80.98 @ epoch 60
logs/agreidv2_airl_iso.log: 180:    [mean] mAP=71.61  R1=80.73
184:    [mean] full=71.61 rec=72.33 FUSE=73.21  <- model-selection uses FUSE
339:    [mean] mAP=71.10  R1=81.39
343:    [mean] full=71.10 rec=71.79 FUSE=72.99  <- model-selection uses FUSE
497:    [mean] mAP=73.63  R1=82.63
501:    [mean] full=73.63 rec=73.20 FUSE=75.03  <- model-selection uses FUSE
656:    [mean] mAP=76.39  R1=84.34
660:    [mean] full=76.39 rec=75.23 FUSE=77.62  <- model-selection uses FUSE
logs/agreidv2_baseline.log: 179:    [mean] mAP=73.68  R1=82.80
334:    [mean] mAP=71.22  R1=80.47
488:    [mean] mAP=72.71  R1=81.76
642:    [mean] mAP=76.82  R1=84.53
797:    [mean] mAP=79.22  R1=86.37
948:Epoch[60] done in 190.7s  Loss=1.047 Acc=0.998
949:  ---- A<->G cross-view eval @ epoch 60 ----
952:    [mean] mAP=79.88  R1=87.11
955:Training finished. Best mean A<->G mAP=79.88 @ epoch 60
logs/agreidv2_baseline_4090.log: 178:    [mean] mAP=73.04  R1=83.14
333:    [mean] mAP=72.75  R1=82.20
487:    [mean] mAP=72.48  R1=81.88
641:    [mean] mAP=76.67  R1=84.63
796:    [mean] mAP=79.98  R1=87.55
947:Epoch[60] done in 91.9s  Loss=1.047 Acc=0.997
948:  ---- A<->G cross-view eval @ epoch 60 ----
951:    [mean] mAP=81.08  R1=88.21
954:Training finished. Best mean A<->G mAP=81.08 @ epoch 60
logs/agreidv2_baseline_4090_s2.log: 178:    [mean] mAP=74.27  R1=83.57
333:    [mean] mAP=70.85  R1=80.98
487:    [mean] mAP=72.93  R1=81.97
641:    [mean] mAP=76.79  R1=84.90
796:    [mean] mAP=79.54  R1=87.18

web search: 
web search: aerial ground person re-identification resolution degradation consistency dual branch recovery head
web search: 
web search: cross-resolution person re-identification resolution degradation consistency loss KL logits low resolution high resolution
web search: 
web search: site:openaccess.thecvf.com person re-identification cross resolution feature distillation degradation consistency
web search: 
web search: AG-ReID.v2 GitHub aerial ground person re-identification
web search: 
web search: CARGO aerial ground person re-identification benchmark paper
web search: 
web search: https://arxiv.org/abs/2401.02634
web search: 
web search: https://arxiv.org/abs/2109.07871
web search: 
web search: https://arxiv.org/abs/2605.18192
codex
数值上已经出现一个关键点：AG-ReID.v2 4090 final 里 AIRL FUSE=80.98，baseline_4090 final=81.08，所以“相对 4090 baseline final 有净增益”不成立；但 FUSE 相对 AIRL full 有 +1.08。这个差异会进最终 verdict。
exec
/bin/zsh -lc 'nl -ba logs/agreidv2_airl_iso.log | tail -n 20' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   792	Epoch[49] Iter[550/786] Loss: 2.134 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0537 Acc: 0.995 LR: 4.74e-06
   793	Epoch[49] Iter[600/786] Loss: 2.132 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0530 Acc: 0.996 LR: 4.74e-06
   794	Epoch[49] Iter[650/786] Loss: 2.129 CE: 1.058 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0524 Acc: 0.996 LR: 4.74e-06
   795	Epoch[49] Iter[700/786] Loss: 2.126 CE: 1.057 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0517 Acc: 0.996 LR: 4.74e-06
   796	Epoch[49] done in 239.2s  Loss=2.122 Acc=0.996 AIRL-ISO[lam_eff=0.500 ce_rec=1.036 consistency=0.0509 deg_scale_mean=0.625 n_ground=28643]
   797	Epoch[50] Iter[50/786] Loss: 2.137 CE: 1.064 Tri: 0.002 CE_rec: 1.044 AIRL_rec: 0.0546 Acc: 0.996 LR: 4.02e-06
   798	Epoch[50] Iter[100/786] Loss: 2.133 CE: 1.062 Tri: 0.003 CE_rec: 1.041 AIRL_rec: 0.0545 Acc: 0.997 LR: 4.02e-06
   799	Epoch[50] Iter[150/786] Loss: 2.132 CE: 1.060 Tri: 0.005 CE_rec: 1.040 AIRL_rec: 0.0542 Acc: 0.996 LR: 4.02e-06
   800	Epoch[50] Iter[200/786] Loss: 2.133 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0544 Acc: 0.997 LR: 4.02e-06
   801	Epoch[50] Iter[250/786] Loss: 2.136 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0551 Acc: 0.996 LR: 4.02e-06
   802	Epoch[50] Iter[300/786] Loss: 2.135 CE: 1.060 Tri: 0.007 CE_rec: 1.041 AIRL_rec: 0.0546 Acc: 0.996 LR: 4.02e-06
   803	Epoch[50] Iter[350/786] Loss: 2.134 CE: 1.059 Tri: 0.007 CE_rec: 1.040 AIRL_rec: 0.0544 Acc: 0.996 LR: 4.02e-06
   804	Epoch[50] Iter[400/786] Loss: 2.131 CE: 1.059 Tri: 0.006 CE_rec: 1.039 AIRL_rec: 0.0541 Acc: 0.996 LR: 4.02e-06
   805	Epoch[50] Iter[450/786] Loss: 2.129 CE: 1.058 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0537 Acc: 0.996 LR: 4.02e-06
   806	Epoch[50] Iter[500/786] Loss: 2.127 CE: 1.057 Tri: 0.006 CE_rec: 1.038 AIRL_rec: 0.0530 Acc: 0.996 LR: 4.02e-06
   807	Epoch[50] Iter[550/786] Loss: 2.126 CE: 1.056 Tri: 0.005 CE_rec: 1.038 AIRL_rec: 0.0525 Acc: 0.996 LR: 4.02e-06
   808	Epoch[50] Iter[600/786] Loss: 2.124 CE: 1.056 Tri: 0.006 CE_rec: 1.037 AIRL_rec: 0.0519 Acc: 0.996 LR: 4.02e-06
   809	Epoch[50] Iter[650/786] Loss: 2.122 CE: 1.055 Tri: 0.006 CE_rec: 1.036 AIRL_rec: 0.0514 Acc: 0.996 LR: 4.02e-06
   810	Epoch[50] Iter[700/786] Loss: 2.119 CE: 1.053 Tri: 0.006 CE_rec: 1.035 AIRL_rec: 0.0505 Acc: 0.997 LR: 4.02e-06
   811	Epoch[50] done in 239.6s  Loss=2.116 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.034 consistency=0.0497 deg_scale_mean=0.624 n_ground=28797]

exec
/bin/zsh -lc "nl -ba logs/agreidv2_baseline_4090.log | sed -n '1,30p;948,955p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
     1	======================================================================
     2	CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
     3	  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/home/afr/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
     4	  use_afd=False  ovp=False (lambda=0.5 tau=0.05 mom=0.2)
     5	  ovli=False (lambda=0.5 tau=0.05 alpha=0.5 dim=256 grid=(8, 4) warmup=10 rerank=False pool=mean topk=8 thresh=0.0 cand=oppview match=maxsim align=free setpool=mean setpool_residual=True)
     6	  acvp=False (gamma=0.5 wmin=0.3 eta=0.05 margin=0.0 warmup=10) [detached neg-relaxation on OVLI; off => OVLI byte-identical]
     7	  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
     8	  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
     9	  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
    10	  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
    11	  out_dir=/home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090
    12	======================================================================
    13	=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
    14	   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
    15	   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
    16	  -----------------------------------------------------------
    17	  subset   | # imgs | # pids | # cams | aerial | ground
    18	  -----------------------------------------------------------
    19	  train    |  51530 |    807 |      3 |  21217 |  30313
    20	  query    |   4167 |    534 |      2 |   2356 |   1811
    21	  gallery  |  20709 |    534 |      2 |  14362 |   6347
    22	  -----------------------------------------------------------
    23	unloaded parameters: <All keys matched successfully>
    24	  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [333 backbone tensors, 2 head tensors] -- prevents the epoch-8 Swin collapse
    25	Epoch[1] Iter[50/786] Loss: 38.559 CE: 6.693 Tri: 31.866 Acc: 0.001 LR: 3.50e-07
    26	Epoch[1] Iter[100/786] Loss: 31.717 CE: 6.692 Tri: 25.025 Acc: 0.002 LR: 3.50e-07
    27	Epoch[1] Iter[150/786] Loss: 28.642 CE: 6.690 Tri: 21.952 Acc: 0.002 LR: 3.50e-07
    28	Epoch[1] Iter[200/786] Loss: 26.615 CE: 6.689 Tri: 19.926 Acc: 0.004 LR: 3.50e-07
    29	Epoch[1] Iter[250/786] Loss: 25.297 CE: 6.686 Tri: 18.611 Acc: 0.005 LR: 3.50e-07
    30	Epoch[1] Iter[300/786] Loss: 24.205 CE: 6.684 Tri: 17.521 Acc: 0.008 LR: 3.50e-07
   948	  ---- A<->G cross-view eval @ epoch 60 ----
   949	    [A->G] mAP=80.73  R1=87.35  R5=93.00  mINP=56.91
   950	    [G->A] mAP=81.42  R1=89.07  R5=93.21  mINP=48.32
   951	    [mean] mAP=81.08  R1=88.21
   952	    * new best mean mAP=81.08 (epoch 60) saved
   953	======================================================================
   954	Training finished. Best mean A<->G mAP=81.08 @ epoch 60
   955	Checkpoints in /home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090

exec
/bin/zsh -lc "nl -ba logs/agreidv2_airl_4090.log | sed -n '1,30p;969,980p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
     1	======================================================================
     2	CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
     3	  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/home/afr/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
     4	  use_afd=False  ovp=False (lambda=0.5 tau=0.05 mom=0.2)
     5	  ovli=False (lambda=0.5 tau=0.05 alpha=0.5 dim=256 grid=(8, 4) warmup=10 rerank=False pool=mean topk=8 thresh=0.0 cand=oppview match=maxsim align=free setpool=mean setpool_residual=True)
     6	  acvp=False (gamma=0.5 wmin=0.3 eta=0.05 margin=0.0 warmup=10) [detached neg-relaxation on OVLI; off => OVLI byte-identical]
     7	  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
     8	  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
     9	  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
    10	  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
    11	  out_dir=/home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090
    12	======================================================================
    13	=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
    14	   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
    15	   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
    16	  -----------------------------------------------------------
    17	  subset   | # imgs | # pids | # cams | aerial | ground
    18	  -----------------------------------------------------------
    19	  train    |  51530 |    807 |      3 |  21217 |  30313
    20	  query    |   4167 |    534 |      2 |   2356 |   1811
    21	  gallery  |  20709 |    534 |      2 |  14362 |   6347
    22	  -----------------------------------------------------------
    23	unloaded parameters: <All keys matched successfully>
    24	  [swin] backbone LR = 3.50e-05 (= base 3.50e-04 x 0.1); heads/BNNeck/OVLI LR = 3.50e-04  [361 backbone tensors, 4 head tensors] -- prevents the epoch-8 Swin collapse
    25	  [AIRL-ISO] iso_stage=3: rec late stage (14185392 params, 28 tensors) + rec BNNeck head (620544 params, 2 tensors) in optimizer [rec late stage @ Swin LR 3.50e-05, rec BNNeck @ full LR 3.50e-04]; degradation-consistency grad isolated from shared trunk (detached degraded pass at stage-3 input); trunk_recce=0 (clean ID-CE + consistency BOTH detached = original full-isolation); eval soft-fusion cos=w*cos_rec+(1-w)*cos_full w=0.25
    26	Epoch[1] Iter[50/786] Loss: 44.958 CE: 6.693 Tri: 31.571 CE_rec: 6.693 AIRL_rec: 0.0002 Acc: 0.001 LR: 3.50e-07
    27	Epoch[1] Iter[100/786] Loss: 38.226 CE: 6.692 Tri: 24.843 CE_rec: 6.691 AIRL_rec: 0.0002 Acc: 0.002 LR: 3.50e-07
    28	Epoch[1] Iter[150/786] Loss: 35.189 CE: 6.690 Tri: 21.809 CE_rec: 6.690 AIRL_rec: 0.0002 Acc: 0.003 LR: 3.50e-07
    29	Epoch[1] Iter[200/786] Loss: 33.301 CE: 6.689 Tri: 19.925 CE_rec: 6.688 AIRL_rec: 0.0002 Acc: 0.004 LR: 3.50e-07
    30	Epoch[1] Iter[250/786] Loss: 31.912 CE: 6.686 Tri: 18.540 CE_rec: 6.685 AIRL_rec: 0.0002 Acc: 0.006 LR: 3.50e-07
   969	  ---- A<->G cross-view eval @ epoch 60 ----
   970	    [A->G] mAP=79.47  R1=87.18  R5=92.57  mINP=54.19
   971	    [G->A] mAP=80.33  R1=87.47  R5=91.83  mINP=47.95
   972	    [mean] mAP=79.90  R1=87.32
   973	  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 60 ----
   974	    [A->G] full mAP=79.47 R1=87.18 | rec mAP=78.70 R1=84.93 | FUSE mAP=80.67 R1=87.90
   975	    [G->A] full mAP=80.33 R1=87.47 | rec mAP=78.38 R1=86.58 | FUSE mAP=81.29 R1=88.02
   976	    [mean] full=79.90 rec=78.54 FUSE=80.98  <- model-selection uses FUSE
   977	    * new best mean mAP=80.98 (epoch 60) saved
   978	======================================================================
   979	Training finished. Best mean A<->G mAP=80.98 @ epoch 60
   980	Checkpoints in /home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090

exec
/bin/zsh -lc 'nl -ba logs/agreidv2_baseline_4090_s2.log | tail -n 20' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   877	Epoch[56] Iter[250/786] Loss: 1.058 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 8.57e-07
   878	Epoch[56] Iter[300/786] Loss: 1.059 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 8.57e-07
   879	Epoch[56] Iter[350/786] Loss: 1.058 CE: 1.053 Tri: 0.006 Acc: 0.996 LR: 8.57e-07
   880	Epoch[56] Iter[400/786] Loss: 1.057 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
   881	Epoch[56] Iter[450/786] Loss: 1.056 CE: 1.051 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
   882	Epoch[56] Iter[500/786] Loss: 1.055 CE: 1.050 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
   883	Epoch[56] Iter[550/786] Loss: 1.054 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
   884	Epoch[56] Iter[600/786] Loss: 1.053 CE: 1.049 Tri: 0.005 Acc: 0.997 LR: 8.57e-07
   885	Epoch[56] Iter[650/786] Loss: 1.052 CE: 1.048 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
   886	Epoch[56] Iter[700/786] Loss: 1.051 CE: 1.047 Tri: 0.004 Acc: 0.997 LR: 8.57e-07
   887	Epoch[56] done in 91.8s  Loss=1.049 Acc=0.997
   888	Epoch[57] Iter[50/786] Loss: 1.053 CE: 1.050 Tri: 0.003 Acc: 0.997 LR: 5.50e-07
   889	Epoch[57] Iter[100/786] Loss: 1.057 CE: 1.053 Tri: 0.004 Acc: 0.995 LR: 5.50e-07
   890	Epoch[57] Iter[150/786] Loss: 1.055 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
   891	Epoch[57] Iter[200/786] Loss: 1.054 CE: 1.051 Tri: 0.003 Acc: 0.996 LR: 5.50e-07
   892	Epoch[57] Iter[250/786] Loss: 1.056 CE: 1.052 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
   893	Epoch[57] Iter[300/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
   894	Epoch[57] Iter[350/786] Loss: 1.055 CE: 1.051 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
   895	Epoch[57] Iter[400/786] Loss: 1.054 CE: 1.050 Tri: 0.004 Acc: 0.996 LR: 5.50e-07
   896	Epoch[57] Iter[450/786] Loss: 1.053 CE: 1.050 Tri: 0.003 Acc: 0.996 LR: 5.50e-07

codex
外部先例检索也做完了：AG-ReID/跨分辨率 ReID/蒸馏式分辨率一致性都有相近问题空间；AIRL 的“固定权重 full/rec 双头距离融合”可以讲，但不能把“分辨率降质一致性”本身说成完全无人做过。
exec
/bin/zsh -lc "nl -ba logs/agreidv2_baseline.log | sed -n '949,956p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   949	  ---- A<->G cross-view eval @ epoch 60 ----
   950	    [A->G] mAP=79.72  R1=86.42  R5=92.28  mINP=56.04
   951	    [G->A] mAP=80.04  R1=87.80  R5=92.16  mINP=46.62
   952	    [mean] mAP=79.88  R1=87.11
   953	    * new best mean mAP=79.88 (epoch 60) saved
   954	======================================================================
   955	Training finished. Best mean A<->G mAP=79.88 @ epoch 60
   956	Checkpoints in /root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_baseline

exec
/bin/zsh -lc "nl -ba logs/agreidv2_baseline_4090_s2.log | sed -n '790,798p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   790	Epoch[50] Iter[650/786] Loss: 1.060 CE: 1.055 Tri: 0.005 Acc: 0.996 LR: 4.02e-06
   791	Epoch[50] Iter[700/786] Loss: 1.059 CE: 1.054 Tri: 0.005 Acc: 0.996 LR: 4.02e-06
   792	Epoch[50] done in 91.7s  Loss=1.057 Acc=0.997
   793	  ---- A<->G cross-view eval @ epoch 50 ----
   794	    [A->G] mAP=79.31  R1=86.67  R5=91.68  mINP=54.35
   795	    [G->A] mAP=79.77  R1=87.69  R5=92.10  mINP=46.88
   796	    [mean] mAP=79.54  R1=87.18
   797	    * new best mean mAP=79.54 (epoch 50) saved
   798	Epoch[51] Iter[50/786] Loss: 1.066 CE: 1.060 Tri: 0.006 Acc: 0.996 LR: 3.34e-06

exec
/bin/zsh -lc "nl -ba logs/agreidv2_airl_4090.log | sed -n '811,818p;969,980p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   811	    [A->G] mAP=78.34  R1=86.29  R5=91.85  mINP=52.48
   812	    [G->A] mAP=79.19  R1=86.69  R5=90.61  mINP=46.79
   813	    [mean] mAP=78.77  R1=86.49
   814	  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 50 ----
   815	    [A->G] full mAP=78.34 R1=86.29 | rec mAP=77.75 R1=85.14 | FUSE mAP=79.63 R1=87.14
   816	    [G->A] full mAP=79.19 R1=86.69 | rec mAP=77.15 R1=85.15 | FUSE mAP=80.18 R1=87.24
   817	    [mean] full=78.77 rec=77.45 FUSE=79.90  <- model-selection uses FUSE
   818	    * new best mean mAP=79.90 (epoch 50) saved
   969	  ---- A<->G cross-view eval @ epoch 60 ----
   970	    [A->G] mAP=79.47  R1=87.18  R5=92.57  mINP=54.19
   971	    [G->A] mAP=80.33  R1=87.47  R5=91.83  mINP=47.95
   972	    [mean] mAP=79.90  R1=87.32
   973	  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 60 ----
   974	    [A->G] full mAP=79.47 R1=87.18 | rec mAP=78.70 R1=84.93 | FUSE mAP=80.67 R1=87.90
   975	    [G->A] full mAP=80.33 R1=87.47 | rec mAP=78.38 R1=86.58 | FUSE mAP=81.29 R1=88.02
   976	    [mean] full=79.90 rec=78.54 FUSE=80.98  <- model-selection uses FUSE
   977	    * new best mean mAP=80.98 (epoch 60) saved
   978	======================================================================
   979	Training finished. Best mean A<->G mAP=80.98 @ epoch 60
   980	Checkpoints in /home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1029,1100p;1191,1228p;1980,1980p;2028,2028p;2224,2236p;2363,2374p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
  1029	def airl_dualbranch_eval(model, dataset, args, device):
  1030	    """AIRL dual-branch eval: extract BOTH heads (f_full, f_rec) in ONE forward
  1031	    and report f_full-only, f_rec-only, and the SOFT-FUSED cosine ranking
  1032	    (cos = w*cos_rec + (1-w)*cos_full, w = args.airl_fuse_w, fixed) for A->G and
  1033	    G->A.  This is the single-model analog of the kill-switch #3 two-model score
  1034	    fusion: cos_rec replaces the AIRL-model cosine, cos_full replaces the
  1035	    baseline-model cosine, and they share ONE backbone forward.
  1036	
  1037	    Mirrors run_cross_view_eval / ovli_rerank_eval exactly for the per-split
  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
  1039	    f_full number reproduces run_cross_view_eval's A<->G mAP bit-for-bit (same
  1040	    feature, same ranking) and the fusion is a pure distance-matrix combination.
  1041	    Returns {tag: {'full': (mAP,R1), 'rec': (mAP,R1), 'fuse': (mAP,R1)}}.
  1042	    """
  1043	    from cargo_dataset import filter_by_view as _fbv
  1044	    from afd_train import build_eval_loader as _bel
  1045	    from maxsim_probe import eval_from_distmat
  1046	
  1047	    model.eval()
  1048	    view_map = {'Aerial': 0, 'Ground': 1}
  1049	
  1050	    @torch.no_grad()
  1051	    def extract(samples):
  1052	        loader = _bel(samples, args)
  1053	        ffs, frs, pids, cams = [], [], [], []
  1054	        for batch in loader:
  1055	            imgs = batch['img'].to(device, non_blocking=True)
  1056	            vidx = (torch.tensor([view_map[v] for v in batch['view']],
  1057	                                 device=device) if args.use_afd else None)
  1058	            # ONE forward -> two L2-normalized features (f_full, f_rec).
  1059	            f_full, f_rec = model(imgs, view_idx=vidx, return_dual=True)
  1060	            ffs.append(f_full.cpu())
  1061	            frs.append(f_rec.cpu())
  1062	            pids.append(batch['pid'])
  1063	            cams.append(batch['camid'])
  1064	        if not ffs:
  1065	            empty = (torch.empty(0), torch.empty(0),
  1066	                     np.empty(0, np.int64), np.empty(0, np.int64))
  1067	            return empty
  1068	        return (torch.cat(ffs, 0), torch.cat(frs, 0),
  1069	                torch.cat(pids, 0).numpy(), torch.cat(cams, 0).numpy())
  1070	
  1071	    w = args.airl_fuse_w
  1072	    results = {}
  1073	    splits = {
  1074	        'A->G': (_fbv(dataset.query, 'Aerial'), _fbv(dataset.gallery, 'Ground')),
  1075	        'G->A': (_fbv(dataset.query, 'Ground'), _fbv(dataset.gallery, 'Aerial')),
  1076	    }
  1077	    for tag, (q, g) in splits.items():
  1078	        q_full, q_rec, qp, qc = extract(q)
  1079	        g_full, g_rec, gp, gc = extract(g)
  1080	        if q_full.numel() == 0 or g_full.numel() == 0:
  1081	            nan2 = (float('nan'), float('nan'))
  1082	            results[tag] = {'full': nan2, 'rec': nan2, 'fuse': nan2}
  1083	            continue
  1084	        # features are already L2-normalized at eval; renormalize defensively so
  1085	        # the cosine == the gram of unit vectors (matches eval_market exactly).
  1086	        q_full = F.normalize(q_full, dim=1); g_full = F.normalize(g_full, dim=1)
  1087	        q_rec = F.normalize(q_rec, dim=1);   g_rec = F.normalize(g_rec, dim=1)
  1088	        s_full = (q_full @ g_full.t()).numpy()        # (Nq,Ng) cosine, f_full
  1089	        s_rec = (q_rec @ g_rec.t()).numpy()           # (Nq,Ng) cosine, f_rec
  1090	        # soft fusion: cos = w*cos_rec + (1-w)*cos_full -> distance = 2 - 2*cos
  1091	        # (identical to kill-switch #3 GATE 5; cosine in [-1,1] -> dist in [0,4]).
  1092	        dm_full = (2.0 - 2.0 * s_full)
  1093	        dm_rec = (2.0 - 2.0 * s_rec)
  1094	        dm_fuse = (2.0 - 2.0 * (w * s_rec + (1.0 - w) * s_full))
  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
  1098	        results[tag] = {'full': (fmap, fr1), 'rec': (rmap, rr1),
  1099	                        'fuse': (zmap, zr1)}
  1100	    return results
  1191	def airl_consistency_loss(logits_o, bn_o, logits_d, bn_d, mode='kl', tau=4.0):
  1192	    """Prediction-consistency between the ORIGINAL and DEGRADED views (fp32).
  1193	
  1194	    The degraded view must keep its identity prediction CLOSE to the original's,
  1195	    so the model learns evidence that survives a low pixel budget.  The ORIGINAL
  1196	    side is DETACHED (it is the stable target; gradient flows only through the
  1197	    degraded branch -> the model is pulled to make the degraded prediction match
  1198	    the clean one, not the reverse).
  1199	
  1200	      mode='kl' (default): symmetric KL would double-count; we use the standard
  1201	        distillation direction KL(softmax(logits_o/tau).detach || softmax(
  1202	        logits_d/tau)) * tau^2 (temperature-scaled soft-target consistency on the
  1203	        ID logits).  fp32 for numeric safety; log_softmax avoids log(0).
  1204	      mode='feat': 1 - cos(bn_o.detach, bn_d) on the L2-normed BNNeck feature
  1205	        (MSE-equivalent up to scale on unit vectors; bounded in [0,2]).
  1206	
  1207	    Returns a finite non-negative scalar.  A light finite guard (nan_to_num) is
  1208	    applied to the scalar: inputs are finite under normal training, so this is a
  1209	    no-op there, but it keeps the documented "finite scalar" contract honest if a
  1210	    pathological forward ever produced inf/NaN logits.
  1211	    """
  1212	    if mode == 'feat':
  1213	        zo = F.normalize(bn_o.float(), dim=1).detach()
  1214	        zd = F.normalize(bn_d.float(), dim=1)
  1215	        # 1 - cosine in [0,2]; mean over batch.  (== 0.5*||zo-zd||^2 on unit vecs.)
  1216	        out = (1.0 - (zo * zd).sum(dim=1)).mean()
  1217	        return torch.nan_to_num(out)
  1218	    # mode == 'kl': temperature-scaled soft-target KL (clean = detached target).
  1219	    lo = logits_o.float() / tau
  1220	    ld = logits_d.float() / tau
  1221	    p_o = F.softmax(lo, dim=1).detach()                       # stable target
  1222	    log_p_d = F.log_softmax(ld, dim=1)                        # fp32; no log(0)
  1223	    # KL(p_o || p_d) = sum p_o (log p_o - log p_d); * tau^2 keeps the gradient
  1224	    # magnitude comparable across temperatures (Hinton distillation convention).
  1225	    log_p_o = F.log_softmax(lo, dim=1).detach()
  1226	    kl = (p_o * (log_p_o - log_p_d)).sum(dim=1).mean()
  1227	    return torch.nan_to_num(kl * (tau * tau))
  1228	
  1980	    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)
  2028	            with torch.amp.autocast('cuda', enabled=not args.no_amp):
  2224	                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
  2225	                        # rec_only -> dict with ONLY logits_rec / bn_feat_rec (f_full
  2226	                        # BNNeck not run on degraded images).
  2227	                        out_d = model(deg_imgs, view_idx=vidx_g, rec_only=True)
  2228	                    with torch.amp.autocast('cuda', enabled=False):
  2229	                        # consistency on the ISOLATED f_rec head: degraded f_rec
  2230	                        # prediction (grad on, into the rec stage only) -> clean f_rec
  2231	                        # one (detached target).  Both sides sliced to ground rows.
  2232	                        loss_airl = airl_consistency_loss(
  2233	                            out['logits_rec'][g_mask], out['bn_feat_rec'][g_mask],
  2234	                            out_d['logits_rec'], out_d['bn_feat_rec'],
  2235	                            mode=args.airl_consistency, tau=args.airl_tau)
  2236	                    loss = loss + airl_lambda_eff * loss_airl
  2363	        if args.airl_dualbranch or args.airl_dualbranch_iso:
  2364	            a_cons = meters['airl'] / seen
  2365	            a_scale = meters['airl_scale'] / seen
  2366	            a_cerec = meters['ce_rec'] / seen
  2367	            n_g = int(meters['airl_n_ground'])
  2368	            bad = (not math.isfinite(a_cons)) or (not math.isfinite(a_cerec)) \
  2369	                or (n_g == 0)
  2370	            flag = " <KILL?>" if bad else ""
  2371	            tag = "AIRL-ISO" if args.airl_dualbranch_iso else "AIRL-DUAL"
  2372	            airl_msg = (f" {tag}[lam_eff={airl_lambda_eff:.3f} "
  2373	                        f"ce_rec={a_cerec:.3f} consistency={a_cons:.4f} "
  2374	                        f"deg_scale_mean={a_scale:.3f} n_ground={n_g}{flag}]")

exec
/bin/zsh -lc "nl -ba code/afd_model.py | sed -n '236,329p;750,812p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   236	    def _forward_swin_split(self, x, rec_only=False):
   237	        """Replicate SwinTransformer.forward but ALSO branch the rec path.
   238	
   239	        Returns (f_full_map, f_rec_map).  The shared patch_embed + ALL f_full stages
   240	        run FIRST, exactly and in the same order as swin.forward (so even the
   241	        training-time stochastic-depth / DropPath RNG sequence f_full sees is
   242	        identical to the single-branch path -- the rec copy runs AFTER the full loop,
   243	        not interleaved, so it cannot perturb f_full's RNG draws); the residual
   244	        stream at the input of stage `iso_stage` is captured and fed through the
   245	        independent rec stages afterward.  semantic_weight is built identically to
   246	        swin.forward.
   247	
   248	        Gradient regime on the rec fork feed (the trunk-undersupervision FIX):
   249	          * rec_only=True (degraded consistency pass): the fork feed is ALWAYS
   250	            DETACHED -> the consistency gradient cannot reach the shared trunk
   251	            (the isolation invariant, independent of iso_trunk_recce).
   252	          * rec_only=False (clean main pass): the fork feed is detached ONLY when
   253	            self.iso_trunk_recce is False (original full-isolation ablation).  When
   254	            iso_trunk_recce is True (the fix, default) the clean fork feed is NOT
   255	            detached, so f_rec's CLEAN ID-CE gradient reflows into the shared trunk
   256	            (extra identity supervision that strengthens f_full).  The
   257	            degradation-consistency still uses the rec_only=True detached path, so
   258	            it never reaches the trunk regardless of this setting.
   259	
   260	        rec_only=True: skip the f_full BNNeck-side work entirely is done by the
   261	        CALLER (it just ignores full_map); here rec_only additionally lets the
   262	        degraded consistency pass avoid keeping the f_full map's grad graph -- we
   263	        still must run the shared stages to REACH the fork point, but we do NOT need
   264	        f_full's last-stage norm/grad, so full_map is returned detached to make the
   265	        "f_full untouched by the degraded pass" contract explicit and cheap.
   266	        """
   267	        swin = self.swin
   268	        # build the per-sample semantic weight exactly like SwinTransformer.forward
   269	        if swin.semantic_weight >= 0:
   270	            w = torch.ones(x.shape[0], 1) * swin.semantic_weight
   271	            w = torch.cat([w, 1 - w], axis=-1)
   272	            semantic_weight = w.to(x.device)
   273	        else:
   274	            semantic_weight = None
   275	
   276	        x, hw_shape = swin.patch_embed(x)
   277	        if swin.use_abs_pos_embed:
   278	            x = x + swin.absolute_pos_embed
   279	        x = swin.drop_after_pos(x)
   280	
   281	        # Whether the rec fork feed is detached from the shared trunk:
   282	        #   * degraded consistency pass (rec_only) -> ALWAYS detach (isolation
   283	        #     invariant: consistency grad never reaches the trunk).
   284	        #   * clean pass -> detach ONLY when iso_trunk_recce is False (original
   285	        #     full-isolation ablation); when True (the fix) keep the graph so the
   286	        #     clean rec ID-CE reflows into the trunk (extra identity supervision).
   287	        detach_fork = bool(rec_only) or (not self.iso_trunk_recce)
   288	        fork_x = None
   289	        fork_hw = None
   290	        full_map = None
   291	        for i, stage in enumerate(swin.stages):
   292	            # the residual stream `x` HERE is the input to stage i.  When i ==
   293	            # iso_stage, snapshot this stream (the gradient-isolation boundary) to
   294	            # feed the rec branch AFTER the full loop.  Detach per detach_fork above:
   295	            # detached -> rec grad severed from trunk; non-detached -> clean rec
   296	            # ID-CE flows back into the trunk (the trunk-undersupervision fix).
   297	            if i == self.iso_stage:
   298	                fork_x = x.detach() if detach_fork else x
   299	                fork_hw = hw_shape
   300	            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
   301	            if swin.semantic_weight >= 0:
   302	                sw = swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
   303	                sb = swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
   304	                x = x * swin.softplus(sw) + sb
   305	            if i in swin.out_indices and i == len(swin.stages) - 1:
   306	                norm_layer = getattr(swin, f'norm{i}')
   307	                out = norm_layer(out)
   308	                out = out.view(-1, *out_hw_shape,
   309	                               swin.num_features[i]).permute(
   310	                                   0, 3, 1, 2).contiguous()
   311	                full_map = out
   312	        full_map = self.layer4(full_map)        # Identity passthrough (OVLI hook)
   313	        if rec_only:
   314	            # the degraded consistency pass only needs f_rec; detach f_full's map so
   315	            # no f_full grad graph is built and the contract "the degraded pass does
   316	            # not train f_full" is explicit.  (running stats of self.bottleneck are
   317	            # NOT updated for this pass because the caller never pools full_map -> no
   318	            # BatchNorm forward on it; see AFDModel.forward rec_only path.)
   319	            full_map = full_map.detach()
   320	        # rec branch: independent late stages on the fork stream.  fork_x is detached
   321	        # per detach_fork (degraded/ablation -> isolated; clean+fix -> grad reflows to
   322	        # trunk).  The semantic weight is a FROZEN constant (no params), so detaching
   323	        # it is harmless: it never blocks gradient through fork_x itself (the rec
   324	        # multiply x*softplus(sw)+sb keeps x's graph).  Run AFTER the f_full loop so
   325	        # f_full's RNG is unchanged.
   326	        rec_map = self._run_rec_stages(
   327	            fork_x, fork_hw,
   328	            None if semantic_weight is None else semantic_weight.detach())
   329	        return full_map, rec_map
   750	        # ---- AIRL gradient-isolated dual-branch path -------------------------- #
   751	        # When iso is on we need BOTH the f_full map and the independent rec map.
   752	        # The Swin split forward returns both in one pass; f_full pools the shared
   753	        # map (bn_feat/global_feat) and f_rec pools the rec map through bottleneck_rec.
   754	        # Fork-feed gradient regime (see _forward_swin_split): the DEGRADED pass
   755	        # (rec_only) always detaches the fork so the consistency gradient never
   756	        # reaches the trunk; the CLEAN pass detaches only when iso_trunk_recce is
   757	        # False -- with the fix (True) the clean f_rec ID-CE reflows into the trunk
   758	        # (extra identity supervision), while the consistency stays trunk-isolated.
   759	        # `or rec_only` so the rec-only degraded contract is honoured even if a
   760	        # caller invokes it under model.eval() (training=False, return_dual=False) --
   761	        # otherwise want_iso would be False and the rec_only dict request would
   762	        # silently fall through to the single f_full eval tensor.
   763	        want_iso = self.airl_dualbranch_iso and (self.training or return_dual
   764	                                                 or rec_only)
   765	        if want_iso:
   766	            full_map, rec_map = self.backbone_swin(
   767	                x, return_rec=True, rec_only=rec_only)
   768	            # rec_only (the degraded consistency pass): compute ONLY the f_rec head.
   769	            # f_full's BNNeck is NOT run on the degraded images, so self.bottleneck's
   770	            # running mean/var stay CLEAN (no degraded-ground stat leak into the
   771	            # f_full eval head) -- f_full is a true clean expert -- and the f_full
   772	            # pool+BN+classifier compute is skipped (cheaper).  The clean forward
   773	            # (rec_only=False) still produces both heads as usual.
   774	            if rec_only:
   775	                _grec, bn_feat_rec = self._embed_rec(rec_map)
   776	                return {
   777	                    'bn_feat_rec': bn_feat_rec,
   778	                    'logits_rec': self.classifier_rec(bn_feat_rec),
   779	                }
   780	            global_feat, bn_feat = self._embed(full_map)
   781	            _grec, bn_feat_rec = self._embed_rec(rec_map)
   782	            if not self.training:
   783	                # eval: ONE forward -> two L2-normalized features (f_full, f_rec).
   784	                return (F.normalize(bn_feat, dim=1),
   785	                        F.normalize(bn_feat_rec, dim=1))
   786	            out = {
   787	                'global_feat': global_feat,   # f_full triplet (shared trunk)
   788	                'bn_feat': bn_feat,
   789	                'logits': self.classifier(bn_feat),
   790	                'band_w': None,               # swin has no AFD band weights
   791	                # f_rec head over the INDEPENDENT rec map (own ID-CE + AIRL
   792	                # consistency in the train loop); pooled rec feat exposed too so the
   793	                # smoke can confirm it is NOT the shared global_feat.
   794	                'global_feat_rec': _grec,
   795	                'bn_feat_rec': bn_feat_rec,
   796	                'logits_rec': self.classifier_rec(bn_feat_rec),
   797	            }
   798	            # AFD CVFC is swin-incompatible (asserted off), so no cf block here.
   799	            return out
   800	
   801	        feat_map, band_w = self._forward_backbone(
   802	            x, view_idx=view_idx, insert_router=self.afd_router)
   803	        global_feat, bn_feat = self._embed(feat_map)
   804	
   805	        if not self.training:
   806	            f_full = F.normalize(bn_feat, dim=1)
   807	            if return_dual and self.airl_dualbranch:
   808	                # second head shares the SAME pooled global_feat -> ONE forward,
   809	                # two L2-normalized features for the distmat-level soft fusion.
   810	                bn_feat_rec = self.bottleneck_rec(global_feat)
   811	                return f_full, F.normalize(bn_feat_rec, dim=1)
   812	            return f_full

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1600,1686p;2135,2150p;2176,2191p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
  1600	    # AIRL numeric-safety guard: min_scale in (0,1] so the down-sampled budget is a
  1601	    # real fraction of the input (>0) and never upscales (<=1); a typo fails fast.
  1602	    if args.airl and not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
  1603	        ap.error("--airl_min_scale must be in (0,1] (per-image scale ratio s in "
  1604	                 f"[min_scale,1]); got {args.airl_min_scale}.")
  1605	    if args.airl and not (args.airl_tau > 0.0):
  1606	        ap.error(f"--airl_tau must be > 0 (softmax temperature); got {args.airl_tau}.")
  1607	
  1608	    # AIRL dual-branch guards.  --airl (single-head consistency) and
  1609	    # --airl_dualbranch (two-head, consistency on f_rec only) are two DIFFERENT
  1610	    # AIRL instantiations of the SAME degrade+consistency primitive; running both
  1611	    # would apply consistency twice (to the single head AND to f_rec) and muddy
  1612	    # the ablation, so they are mutually exclusive.  The dual-branch shares
  1613	    # --airl_lambda / --airl_min_scale / --airl_consistency / --airl_tau /
  1614	    # --airl_blur / --airl_warmup (the consistency on f_rec is the SAME function),
  1615	    # and they are validated the same way (so a stray bad --airl_min_scale with
  1616	    # only --airl_dualbranch still fails fast).
  1617	    if args.airl_dualbranch:
  1618	        if args.airl:
  1619	            ap.error("--airl_dualbranch and --airl are mutually exclusive (both "
  1620	                     "apply the AIRL degradation-consistency; dual-branch applies "
  1621	                     "it to the f_rec head only). Pick one.")
  1622	        if not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
  1623	            ap.error("--airl_min_scale must be in (0,1] (used by --airl_dualbranch "
  1624	                     f"too); got {args.airl_min_scale}.")
  1625	        if not (args.airl_tau > 0.0):
  1626	            ap.error("--airl_tau must be > 0 (used by --airl_dualbranch too); got "
  1627	                     f"{args.airl_tau}.")
  1628	        if not (0.0 <= args.airl_fuse_w <= 1.0):
  1629	            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
  1630	                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
  1631	        # w-lock (soft): the headline fixed-prior fusion uses w=0.25.  A non-default
  1632	        # w is ABLATION-ONLY (the w-sweep), so warn rather than assert -- the sweep
  1633	        # still needs to pass other values -- but make any deviation from the
  1634	        # headline visible in the log so a stray w never silently becomes "the
  1635	        # result".
  1636	        if args.airl_fuse_w != 0.25:
  1637	            print(f"[AIRL-DUAL][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
  1638	                  "headline uses the FIXED prior w=0.25; non-default w is "
  1639	                  "ABLATION-ONLY (w-sweep), not the headline result.")
  1640	        # The dual-branch is the standalone headline AIRL mechanism; keep its
  1641	        # ablation clean by forbidding co-running the cross-view OVP/OVLI losses
  1642	        # (they target a different gap and would confound the f_rec specialisation).
  1643	        if args.ovp or args.ovli:
  1644	            ap.error("--airl_dualbranch is run standalone (headline AIRL); do not "
  1645	                     "combine with --ovp/--ovli (separate cross-view mechanisms).")
  1646	
  1647	    # AIRL gradient-isolated dual-branch guards.  This is the RESCUE variant: the
  1648	    # SAME degrade+consistency+soft-fusion contract as --airl_dualbranch, but f_rec
  1649	    # forks off a DETACHED trunk into an independent late Swin stage (so the
  1650	    # consistency gradient cannot pollute the shared trunk).  It therefore:
  1651	    #   * shares the AIRL hyperparams (--airl_lambda/min_scale/consistency/tau/blur/
  1652	    #     warmup) and --airl_fuse_w, validated identically;
  1653	    #   * is mutually exclusive with BOTH --airl (single-head) and --airl_dualbranch
  1654	    #     (fully-shared) -- three distinct AIRL instantiations, one at a time;
  1655	    #   * is swin_small-only (the fork lives in the Swin stage list);
  1656	    #   * runs standalone (no OVP/OVLI), same as --airl_dualbranch.
  1657	    if args.airl_dualbranch_iso:
  1658	        if args.airl or args.airl_dualbranch:
  1659	            ap.error("--airl_dualbranch_iso is mutually exclusive with --airl and "
  1660	                     "--airl_dualbranch (three distinct AIRL instantiations; the "
  1661	                     "iso variant forks an independent late stage off a detached "
  1662	                     "trunk). Pick one.")
  1663	        if args.backbone != 'swin_small':
  1664	            ap.error("--airl_dualbranch_iso requires --backbone swin_small (the rec "
  1665	                     "branch forks an independent Swin late stage).")
  1666	        if not (1 <= args.airl_iso_stage <= 3):
  1667	            ap.error("--airl_iso_stage must be in [1,3] (swin_small has 4 stages "
  1668	                     "0..3; fork after a shared early stage, before the last); got "
  1669	                     f"{args.airl_iso_stage}.")
  1670	        if not (args.airl_min_scale > 0.0 and args.airl_min_scale <= 1.0):
  1671	            ap.error("--airl_min_scale must be in (0,1] (used by "
  1672	                     f"--airl_dualbranch_iso too); got {args.airl_min_scale}.")
  1673	        if not (args.airl_tau > 0.0):
  1674	            ap.error("--airl_tau must be > 0 (used by --airl_dualbranch_iso too); "
  1675	                     f"got {args.airl_tau}.")
  1676	        if not (0.0 <= args.airl_fuse_w <= 1.0):
  1677	            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
  1678	                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
  1679	        if args.airl_fuse_w != 0.25:
  1680	            print(f"[AIRL-ISO][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
  1681	                  "headline uses the FIXED prior w=0.25; non-default w is "
  1682	                  "ABLATION-ONLY (w-sweep), not the headline result.")
  1683	        if args.ovp or args.ovli:
  1684	            ap.error("--airl_dualbranch_iso is run standalone (headline AIRL); do "
  1685	                     "not combine with --ovp/--ovli (separate cross-view "
  1686	                     "mechanisms).")
  2135	                    with torch.no_grad():
  2136	                        deg_imgs, deg_scales = airl_degrade(
  2137	                            imgs_g, args.airl_min_scale, blur=args.airl_blur)
  2138	                        airl_scale_mean = deg_scales.mean()
  2139	                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
  2140	                        out_d = model(deg_imgs, view_idx=vidx_g,
  2141	                                      return_cvfc=(args.use_afd and args.afd_cvfc))
  2142	                    with torch.amp.autocast('cuda', enabled=False):
  2143	                        # consistency forces the DEGRADED ground prediction
  2144	                        # (gradient on) toward the CLEAN ground one (detached target
  2145	                        # inside the loss).  Clean side sliced to the ground rows.
  2146	                        loss_airl = airl_consistency_loss(
  2147	                            logits[g_mask], bn[g_mask],
  2148	                            out_d['logits'], out_d['bn_feat'],
  2149	                            mode=args.airl_consistency, tau=args.airl_tau)
  2150	                    loss = loss + airl_lambda_eff * loss_airl
  2176	                    with torch.no_grad():
  2177	                        deg_imgs, deg_scales = airl_degrade(
  2178	                            imgs_g, args.airl_min_scale, blur=args.airl_blur)
  2179	                        airl_scale_mean = deg_scales.mean()
  2180	                    with torch.amp.autocast('cuda', enabled=not args.no_amp):
  2181	                        out_d = model(deg_imgs, view_idx=vidx_g,
  2182	                                      return_cvfc=(args.use_afd and args.afd_cvfc))
  2183	                    with torch.amp.autocast('cuda', enabled=False):
  2184	                        # consistency on the f_rec head ONLY: degraded f_rec
  2185	                        # prediction (grad on) -> clean f_rec one (detached target
  2186	                        # inside the loss).  Both sides sliced to the ground rows.
  2187	                        loss_airl = airl_consistency_loss(
  2188	                            out['logits_rec'][g_mask], out['bn_feat_rec'][g_mask],
  2189	                            out_d['logits_rec'], out_d['bn_feat_rec'],
  2190	                            mode=args.airl_consistency, tau=args.airl_tau)
  2191	                    loss = loss + airl_lambda_eff * loss_airl

web search: 
web search: resolution adaptive re-identification RAR person re-id resolution adaptive representation learning
web search: 
web search: "AG-ReID.v2" "aerial-ground" person re-identification
web search: 
web search: "CARGO" "Aerial-Ground Person Re-Identification"
codex
1. Verdict: needs-attention

代码/AMP/融合实现基本可信；主要问题在结果解释。AG-ReID.v2 4090 final 不支持“AIRL 相对 baseline final 有净增益”，只能支持“FUSE 相对 AIRL full 有融合增益”。

2. Findings

- High - [logs/agreidv2_airl_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_4090.log:973) + [logs/agreidv2_baseline_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_baseline_4090.log:948): AG-ReID.v2 4090 final 是 AIRL FUSE 80.98 vs baseline 81.08，净值为 -0.10，不是正增益。FUSE-full 是 +1.08，这是融合内部收益，不等于相对 baseline 净收益。
- Medium - [logs/agreidv2_airl_iso.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_iso.log:811) + [logs/agreidv2_baseline_4090_s2.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_baseline_4090_s2.log:896): 两个 log 不完整，分别停在 AIRL epoch 50、baseline_s2 epoch 57 中途，不能作为 final 或完整 seed 统计。
- Medium - [code/afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1217): `airl_consistency_loss` 用 `torch.nan_to_num` 把异常标量吞掉；后面的 kill-switch 只能看到处理后的 finite 值。当前 logs 没有 NaN/Inf 证据，但这是潜在诊断盲点。
- Low - [logs/agreidv2_airl_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_4090.log:9): 这次 headline run 是 `trunk_recce=False`，即原 full-isolation，不是代码注释里默认的 trunk-recce fix。不能把该 run 的 full-head 行为归因于 trunk-recce=1 fix。

3. 三条待验结论

1. “AIRL net gain ≈ 融合增益 FUSE-full ~+1.5，双数据集一致”  
反对 / log 数据不支持。AG-ReID.v2 4090 final：FUSE-full = 80.98 - 79.90 = +1.08；FUSE-baseline_4090 = 80.98 - 81.08 = -0.10。若只和 3090 baseline final 79.88 比是 +1.10，但这混了机器/seed。bundle 里没有 CARGO log，不能验证“双数据集一致”。

2. “f_full 头 ≈ baseline ± seed 噪声；AG 4090 full 背运 -1.2 → net≈0”  
基本同意 AG-ReID.v2 这部分，但不能外推 CARGO。AG 4090：AIRL full 79.90 vs baseline_4090 81.08 = -1.18；AG 3090 baseline 79.88 vs AIRL full 79.90 = +0.02。baseline 完整两条 final spread 正好约 1.20，所以“full 头像 baseline、4090 背运”是合理解释。但 AIRL FUSE 对 4090 baseline 不是正净增益，是约 0。

3. “方向特化是 CARGO-specific，第二数据集方向翻号/消失”  
AG-ReID.v2 支持“方向特化消失”，但 CARGO-specific 需要 CARGO log。AG final 里 rec 比 full 两个方向都低：A->G 78.70 vs 79.47，G->A 78.38 vs 80.33；FUSE 两方向都涨：A->G +1.20，G->A +0.96。没有稳定的 rec/full 方向专门化。

4. 本轮 lens 专项结论

AMP 有：`GradScaler` 和主/degraded forward autocast 都启用，log 里 `amp=True`。consistency loss 在 `autocast(enabled=False)` 下 fp32 计算。

归一化有：iso eval 返回的 `f_full/f_rec` 都 `F.normalize`，dual eval 里又 defensive re-normalize。cos 融合是 CPU numpy 上的 `0.25*s_rec + 0.75*s_full`，`fuse_w` 是 Python float，不存在 device mismatch；CLI 也校验 `[0,1]`。

数值稳定性整体可接受：日志未见 NaN/Inf/KILL，AIRL consistency 从高位回落到 0.0458。唯一需要注意的是 `nan_to_num` 会掩盖异常源，建议后续加显式 `torch.isfinite` 计数/报警。

Novelty/先例：AG-ReID.v2、CARGO/VDT、ViSA 已经覆盖 aerial-ground ReID 的视角/语义/双分支方向；跨分辨率 ReID 也有 Recover-and-Identify、RFD 这类 degradation / recovery / distillation 先例。AIRL 的可讲新意应收窄为“aerial-ground 下的 ground-to-aerial pixel-budget degradation + isolated recover head + fixed-prior score fusion”，不要泛称“分辨率降质一致性”本身是新问题。

参考来源：AG-ReID.v2 https://arxiv.org/abs/2401.02634 ，VDT/CARGO https://arxiv.org/abs/2403.14513 ，ViSA https://arxiv.org/abs/2605.18192 ，Recover-and-Identify https://arxiv.org/abs/1908.06052 ，RFD https://arxiv.org/abs/2109.07871
tokens used
208,490
1. Verdict: needs-attention

代码/AMP/融合实现基本可信；主要问题在结果解释。AG-ReID.v2 4090 final 不支持“AIRL 相对 baseline final 有净增益”，只能支持“FUSE 相对 AIRL full 有融合增益”。

2. Findings

- High - [logs/agreidv2_airl_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_4090.log:973) + [logs/agreidv2_baseline_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_baseline_4090.log:948): AG-ReID.v2 4090 final 是 AIRL FUSE 80.98 vs baseline 81.08，净值为 -0.10，不是正增益。FUSE-full 是 +1.08，这是融合内部收益，不等于相对 baseline 净收益。
- Medium - [logs/agreidv2_airl_iso.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_iso.log:811) + [logs/agreidv2_baseline_4090_s2.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_baseline_4090_s2.log:896): 两个 log 不完整，分别停在 AIRL epoch 50、baseline_s2 epoch 57 中途，不能作为 final 或完整 seed 统计。
- Medium - [code/afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1217): `airl_consistency_loss` 用 `torch.nan_to_num` 把异常标量吞掉；后面的 kill-switch 只能看到处理后的 finite 值。当前 logs 没有 NaN/Inf 证据，但这是潜在诊断盲点。
- Low - [logs/agreidv2_airl_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_4090.log:9): 这次 headline run 是 `trunk_recce=False`，即原 full-isolation，不是代码注释里默认的 trunk-recce fix。不能把该 run 的 full-head 行为归因于 trunk-recce=1 fix。

3. 三条待验结论

1. “AIRL net gain ≈ 融合增益 FUSE-full ~+1.5，双数据集一致”  
反对 / log 数据不支持。AG-ReID.v2 4090 final：FUSE-full = 80.98 - 79.90 = +1.08；FUSE-baseline_4090 = 80.98 - 81.08 = -0.10。若只和 3090 baseline final 79.88 比是 +1.10，但这混了机器/seed。bundle 里没有 CARGO log，不能验证“双数据集一致”。

2. “f_full 头 ≈ baseline ± seed 噪声；AG 4090 full 背运 -1.2 → net≈0”  
基本同意 AG-ReID.v2 这部分，但不能外推 CARGO。AG 4090：AIRL full 79.90 vs baseline_4090 81.08 = -1.18；AG 3090 baseline 79.88 vs AIRL full 79.90 = +0.02。baseline 完整两条 final spread 正好约 1.20，所以“full 头像 baseline、4090 背运”是合理解释。但 AIRL FUSE 对 4090 baseline 不是正净增益，是约 0。

3. “方向特化是 CARGO-specific，第二数据集方向翻号/消失”  
AG-ReID.v2 支持“方向特化消失”，但 CARGO-specific 需要 CARGO log。AG final 里 rec 比 full 两个方向都低：A->G 78.70 vs 79.47，G->A 78.38 vs 80.33；FUSE 两方向都涨：A->G +1.20，G->A +0.96。没有稳定的 rec/full 方向专门化。

4. 本轮 lens 专项结论

AMP 有：`GradScaler` 和主/degraded forward autocast 都启用，log 里 `amp=True`。consistency loss 在 `autocast(enabled=False)` 下 fp32 计算。

归一化有：iso eval 返回的 `f_full/f_rec` 都 `F.normalize`，dual eval 里又 defensive re-normalize。cos 融合是 CPU numpy 上的 `0.25*s_rec + 0.75*s_full`，`fuse_w` 是 Python float，不存在 device mismatch；CLI 也校验 `[0,1]`。

数值稳定性整体可接受：日志未见 NaN/Inf/KILL，AIRL consistency 从高位回落到 0.0458。唯一需要注意的是 `nan_to_num` 会掩盖异常源，建议后续加显式 `torch.isfinite` 计数/报警。

Novelty/先例：AG-ReID.v2、CARGO/VDT、ViSA 已经覆盖 aerial-ground ReID 的视角/语义/双分支方向；跨分辨率 ReID 也有 Recover-and-Identify、RFD 这类 degradation / recovery / distillation 先例。AIRL 的可讲新意应收窄为“aerial-ground 下的 ground-to-aerial pixel-budget degradation + isolated recover head + fixed-prior score fusion”，不要泛称“分辨率降质一致性”本身是新问题。

参考来源：AG-ReID.v2 https://arxiv.org/abs/2401.02634 ，VDT/CARGO https://arxiv.org/abs/2403.14513 ，ViSA https://arxiv.org/abs/2605.18192 ，Recover-and-Identify https://arxiv.org/abs/1908.06052 ，RFD https://arxiv.org/abs/2109.07871
