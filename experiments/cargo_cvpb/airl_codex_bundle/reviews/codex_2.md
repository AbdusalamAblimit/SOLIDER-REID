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
session id: 019ef6ad-dd0c-76b0-9a8c-76d05e65952c
--------
user
你是 10 名独立审查员中的第 2 名,对 AIRL 做全量代码 + log 审查(不是只看你的 lens,要全量,但本轮额外深挖下面这个重点)。
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
融合机制是否 trivial: soft-fuse w=0.25 固定合理吗?FUSE−full 的 +1.5 是真互补证据还是两个相关 head 的平庸 ensemble?换成两个独立随机 init 单 head 平均会不会也涨这么多(即增益是否来自 degradation 机制还是单纯双头平均)?

== 要求 ==
逐行读 code/afd_model.py 和 code/afd_train.py 的相关部分,并对照 logs/ 里的真实数字。
输出格式:
1. Verdict: approve(代码与结果可信)/ needs-attention(有问题)
2. Findings: 每条给 severity(Critical/High/Medium/Low)+ 文件:行号 + 具体问题 + 为什么
3. 对主控 3 条待验结论的独立判断: 逐条 同意 / 反对 / log数据不支持,给依据
4. 本轮 lens 的专项结论
用 web search 查相关 novelty/先例。中文输出,务实不客套。
codex
我先按审查流程把两处代码和所有相关 log 的可解析指标拉出来，再对 soft-fuse 机制和文献先例单独核一遍。当前环境是只读，所以本轮只做审查和结论，不改文件。
exec
/bin/zsh -lc 'wc -l code/afd_model.py code/afd_train.py logs/agreidv2_airl_4090.log logs/agreidv2_airl_iso.log logs/agreidv2_baseline_4090.log logs/agreidv2_baseline_4090_s2.log logs/agreidv2_baseline.log' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
     898 code/afd_model.py
    2445 code/afd_train.py
     981 logs/agreidv2_airl_4090.log
     811 logs/agreidv2_airl_iso.log
     956 logs/agreidv2_baseline_4090.log
     896 logs/agreidv2_baseline_4090_s2.log
     957 logs/agreidv2_baseline.log
    7944 total

exec
/bin/zsh -lc "pwd && rg --files | rg '("'^|/)(afd_model'"\\.py|afd_train\\.py|logs/.*\\.log)"'$'"'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
code/afd_train.py
code/afd_model.py
logs/agreidv2_baseline_4090_s2.log
logs/agreidv2_airl_iso.log
logs/agreidv2_baseline_4090.log
logs/agreidv2_baseline.log
logs/agreidv2_airl_4090.log

exec
/bin/zsh -lc 'git status --short' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-ogYrRNUa' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-RsX55Oj0' (errno=Operation not permitted)
 M ../../decisions.md
 M ../../../model/backbones/swin_transformer.py
?? ../../afd_reid/
?? ../
?? ../../cbcl_t2i/
?? ../../cross_view_cargo/
?? ../../library_167_synthesis.md
?? ../../publishable_angles_synthesis.md
?? ../../smpl_anchor_vireid/

codex
我已经确认目标文件和 log 都在当前 bundle 里。接下来会先定位关键实现路径：模型分支拆分、训练 loss、eval 特征收集、FUSE 计算和日志输出。
exec
/bin/zsh -lc 'rg -n "class AFDModel|_forward_swin_split|airl_dualbranch_iso|airl_iso|trunk_recce|f_full|f_rec|fuse|FUSE|do_inference|cos|normalize|rec" code/afd_model.py' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
7:    train returns (global_feat, logits); test returns L2-normalized BN feature.
13:      on view/altitude), and recombine the bands. The recombined feature replaces the
17:      training-time regularizer (no test-time cost). Two switchable sub-mechanisms:
56:    torch.load directly).  The default python on the training boxes has neither
97:                 iso_trunk_recce=True):
123:        # ---- AIRL gradient-isolated dual-branch (f_rec independent late stage) --
124:        # iso_branch=True forks a SECOND last-stage path (f_rec) off the shared
125:        # residual stream at the input of stage `iso_stage`.  The rec path is an
128:        # iso_trunk_recce (the trunk-undersupervision FIX):
130:        #   * DEGRADED (rec_only=True, the consistency pass): the fork feed is ALWAYS
132:        #     rec copy + BNNeck_rec and NEVER reaches the shared trunk.  This is the
133:        #     isolation invariant that keeps f_rec a specialised "recover expert" and
134:        #     protects the clean trunk + f_full from being pulled toward degradation-
135:        #     robustness.  Holds for BOTH settings of iso_trunk_recce.
137:        #   * CLEAN (rec_only=False, the main forward):
138:        #       - iso_trunk_recce=True  (default, the FIX): the fork feed is NOT
139:        #         detached, so f_rec's CLEAN ID-CE gradient FLOWS BACK into the shared
141:        #         the trunk's extra identity supervision (f_rec's clean ID-CE only
142:        #         updated the detached rec tail), leaving f_full WEAKER than even the
145:        #         identity supervision -> strengthens f_full, while the degradation-
146:        #         consistency stays detached (above) -> f_rec stays specialised.
147:        #       - iso_trunk_recce=False (ablation): the clean fork feed is ALSO
156:        # iso_trunk_recce: whether the CLEAN rec ID-CE gradient reflows into the
159:        self.iso_trunk_recce = bool(iso_trunk_recce)
166:            # the rec branch re-runs stages [iso_stage .. last] on its OWN copy.
167:            # deep-copy preserves the pretrained weights as the f_rec init (same
168:            # starting point as f_full's stages -> divergence comes from training,
169:            # not from a random re-init that would cripple f_rec).
170:            self.rec_stages = nn.ModuleList(
173:            # the last output norm (norm{last}) applied to the rec last-stage map,
174:            # an independent copy so f_rec gets its own LayerNorm (matches the
175:            # f_full norm recipe; reshaped exactly like swin.forward does).
177:            self.rec_norm = copy.deepcopy(getattr(self.swin, f'norm{last}'))
178:            # independent copies of the semantic-embed Linears for the rec stages
180:            # the same frozen weights so the rec stream is modulated identically to
182:            # rec branch runs stages [iso_stage..last] so it needs those indices.
184:                self.rec_semantic_embed_w = nn.ModuleList(
187:                self.rec_semantic_embed_b = nn.ModuleList(
191:                # trunk froze them); re-assert defensively so the rec semantic embed
193:                for p in self.rec_semantic_embed_w.parameters():
195:                for p in self.rec_semantic_embed_b.parameters():
197:            # Identity hook point for the rec last-stage map (mirrors self.layer4);
198:            # kept for parity / future hooks -- the rec map is a fresh path so OVLI's
199:            # single layer4 hook (on the f_full map) is unaffected.
200:            self.layer4_rec = nn.Identity()
202:    def _run_rec_stages(self, x, hw_shape, semantic_weight):
203:        """Run the INDEPENDENT rec copy of stages [iso_stage..last] on the residual
204:        stream `x` (the fork feed) and return the rec last-stage NCHW map.
206:        The caller (_forward_swin_split) decides whether `x` is detached: the DEGRADED
208:        CLEAN pass with iso_trunk_recce=True passes a NON-detached fork so the clean
209:        f_rec ID-CE reflows into the shared trunk.  This method itself is agnostic to
210:        that choice -- it just runs the rec stages over whatever `x` it is given.
214:        but over self.rec_stages / self.rec_norm / self.rec_semantic_embed_*, so the
215:        rec map is computed the same way f_full's map is -- the ONLY differences are
220:        rec_out = None
221:        for j, stage in enumerate(self.rec_stages):
225:                sw = self.rec_semantic_embed_w[j](semantic_weight).unsqueeze(1)
226:                sb = self.rec_semantic_embed_b[j](semantic_weight).unsqueeze(1)
229:                out = self.rec_norm(out)
233:                rec_out = out
234:        return self.layer4_rec(rec_out)
236:    def _forward_swin_split(self, x, rec_only=False):
237:        """Replicate SwinTransformer.forward but ALSO branch the rec path.
239:        Returns (f_full_map, f_rec_map).  The shared patch_embed + ALL f_full stages
241:        training-time stochastic-depth / DropPath RNG sequence f_full sees is
242:        identical to the single-branch path -- the rec copy runs AFTER the full loop,
243:        not interleaved, so it cannot perturb f_full's RNG draws); the residual
245:        independent rec stages afterward.  semantic_weight is built identically to
248:        Gradient regime on the rec fork feed (the trunk-undersupervision FIX):
249:          * rec_only=True (degraded consistency pass): the fork feed is ALWAYS
251:            (the isolation invariant, independent of iso_trunk_recce).
252:          * rec_only=False (clean main pass): the fork feed is detached ONLY when
253:            self.iso_trunk_recce is False (original full-isolation ablation).  When
254:            iso_trunk_recce is True (the fix, default) the clean fork feed is NOT
255:            detached, so f_rec's CLEAN ID-CE gradient reflows into the shared trunk
256:            (extra identity supervision that strengthens f_full).  The
257:            degradation-consistency still uses the rec_only=True detached path, so
260:        rec_only=True: skip the f_full BNNeck-side work entirely is done by the
261:        CALLER (it just ignores full_map); here rec_only additionally lets the
262:        degraded consistency pass avoid keeping the f_full map's grad graph -- we
264:        f_full's last-stage norm/grad, so full_map is returned detached to make the
265:        "f_full untouched by the degraded pass" contract explicit and cheap.
281:        # Whether the rec fork feed is detached from the shared trunk:
282:        #   * degraded consistency pass (rec_only) -> ALWAYS detach (isolation
284:        #   * clean pass -> detach ONLY when iso_trunk_recce is False (original
286:        #     clean rec ID-CE reflows into the trunk (extra identity supervision).
287:        detach_fork = bool(rec_only) or (not self.iso_trunk_recce)
294:            # feed the rec branch AFTER the full loop.  Detach per detach_fork above:
295:            # detached -> rec grad severed from trunk; non-detached -> clean rec
313:        if rec_only:
314:            # the degraded consistency pass only needs f_rec; detach f_full's map so
315:            # no f_full grad graph is built and the contract "the degraded pass does
316:            # not train f_full" is explicit.  (running stats of self.bottleneck are
318:            # BatchNorm forward on it; see AFDModel.forward rec_only path.)
320:        # rec branch: independent late stages on the fork stream.  fork_x is detached
323:        # it is harmless: it never blocks gradient through fork_x itself (the rec
324:        # multiply x*softplus(sw)+sb keeps x's graph).  Run AFTER the f_full loop so
325:        # f_full's RNG is unchanged.
326:        rec_map = self._run_rec_stages(
329:        return full_map, rec_map
331:    def forward(self, x, return_rec=False, rec_only=False):
332:        # Default path (return_rec=False OR iso off): SwinTransformer.forward ->
336:        if not (self.iso_branch and return_rec):
340:        # iso dual-branch path: run the split forward -> (f_full map, f_rec map).
341:        # The rec map is computed through independent late stages.  The DEGRADED pass
342:        # (rec_only) forks off a DETACHED trunk so the consistency loss cannot perturb
344:        # iso_trunk_recce=True (clean f_rec ID-CE reflows -> extra trunk supervision),
345:        # else detached (full-isolation ablation).  See _forward_swin_split.
346:        full_map, rec_map = self._forward_swin_split(x, rec_only=rec_only)
347:        return full_map, rec_map
396:    """Build (low, mid, high) centered rectangular FFT-shifted masks on an HxW grid.
472:        Returns (recombined feature, band_weights(B,3)).
487:        recomb = b(wl, low) + b(wm, mid) + b(wh, high)
489:        recomb = recomb * 3.0
491:            recomb = 0.5 * x + 0.5 * recomb
492:        return recomb, w
528:class AFDModel(nn.Module):
540:                 airl_dualbranch_iso=False, airl_iso_stage=3,
541:                 airl_iso_trunk_recce=True):
545:        # AIRL dual-branch: a SECOND BNNeck head (bottleneck_rec + classifier_rec)
546:        # over the SAME shared backbone feature map.  f_full (the original head)
547:        # keeps full-resolution identity evidence (protects G->A); f_rec (this
549:        # consistency at train time, so it learns identity evidence recoverable
551:        # heads' cosine scores are SOFT-fused at the distance-matrix level
552:        # (cos = w*cos_rec + (1-w)*cos_full) -- ONE forward yields both features.
557:        # as airl_dualbranch, but f_rec is NOT a BNNeck over the shared global_feat;
559:        # trunk feature (see SwinBackboneReID.iso_branch).  This severs the f_rec
560:        # consistency gradient from the shared trunk so the clean trunk + f_full are
563:        # the shared airl_dualbranch (same eval/loss contract, different f_rec path).
564:        self.airl_dualbranch_iso = bool(airl_dualbranch_iso)
565:        self.airl_iso_stage = int(airl_iso_stage)
566:        # airl_iso_trunk_recce: route the CLEAN f_rec ID-CE gradient back into the
570:        self.airl_iso_trunk_recce = bool(airl_iso_trunk_recce)
571:        if self.airl_dualbranch_iso:
573:                "airl_dualbranch_iso and airl_dualbranch are mutually exclusive "
574:                "(shared vs gradient-isolated f_rec; pick one).")
576:                "airl_dualbranch_iso requires backbone='swin_small' (the rec branch "
627:                iso_branch=self.airl_dualbranch_iso,
628:                iso_stage=self.airl_iso_stage,
629:                iso_trunk_recce=self.airl_iso_trunk_recce)
642:        # BNNeck (f_full -- the original head: full-resolution identity evidence)
650:        # AIRL dual-branch: a SECOND independent BNNeck head (f_rec).  Same structure
651:        # / init recipe as f_full (frozen-bias BNNeck + bias-free classifier), but its
652:        # OWN parameters so the two heads can specialise (f_rec absorbs the
653:        # degradation-consistency signal, f_full stays clean).
654:        #   * airl_dualbranch     : f_rec pools the SHARED global_feat (fully shared
656:        #   * airl_dualbranch_iso : f_rec pools the INDEPENDENT rec last-stage map
660:        if self.airl_dualbranch or self.airl_dualbranch_iso:
661:            self.bottleneck_rec = nn.BatchNorm1d(self.in_planes)
662:            self.bottleneck_rec.bias.requires_grad_(False)
663:            self.bottleneck_rec.apply(weights_init_kaiming)
664:            self.classifier_rec = nn.Linear(self.in_planes, num_classes, bias=False)
665:            self.classifier_rec.apply(weights_init_classifier)
717:    def _embed_rec(self, x):
718:        """rec map -> pooled rec feat -> BNNeck_rec feat (independent f_rec head).
720:        Used only by the gradient-isolated dual-branch: the rec map already comes
721:        from a detached trunk + independent late stage, so pooling + bottleneck_rec
722:        here keeps the whole f_rec head isolated from the shared trunk.
725:        bn = self.bottleneck_rec(g)
730:                rec_only=False):
734:               When airl_dualbranch is on, the dict ALSO carries the f_rec head's
735:               'bn_feat_rec' / 'logits_rec' (computed from the SAME pooled
737:               the f_rec ID-CE + degradation-consistency.
738:        Eval : returns the L2-normalized f_full BN feature (single head); when
740:               (f_full_norm, f_rec_norm) so the dual-branch eval can SOFT-fuse
741:               the two cosine scores.  return_dual defaults to False, so the
744:               airl_dualbranch_iso: identical (f_full_norm, f_rec_norm) eval tuple
745:               and bn_feat_rec/logits_rec train keys, but f_rec is pooled from the
746:               INDEPENDENT rec last-stage map (gradient-isolated trunk) instead of
747:               the shared global_feat.  return_rec on the Swin backbone yields BOTH
751:        # When iso is on we need BOTH the f_full map and the independent rec map.
752:        # The Swin split forward returns both in one pass; f_full pools the shared
753:        # map (bn_feat/global_feat) and f_rec pools the rec map through bottleneck_rec.
754:        # Fork-feed gradient regime (see _forward_swin_split): the DEGRADED pass
755:        # (rec_only) always detaches the fork so the consistency gradient never
756:        # reaches the trunk; the CLEAN pass detaches only when iso_trunk_recce is
757:        # False -- with the fix (True) the clean f_rec ID-CE reflows into the trunk
759:        # `or rec_only` so the rec-only degraded contract is honoured even if a
761:        # otherwise want_iso would be False and the rec_only dict request would
762:        # silently fall through to the single f_full eval tensor.
763:        want_iso = self.airl_dualbranch_iso and (self.training or return_dual
764:                                                 or rec_only)
766:            full_map, rec_map = self.backbone_swin(
767:                x, return_rec=True, rec_only=rec_only)
768:            # rec_only (the degraded consistency pass): compute ONLY the f_rec head.
769:            # f_full's BNNeck is NOT run on the degraded images, so self.bottleneck's
771:            # f_full eval head) -- f_full is a true clean expert -- and the f_full
773:            # (rec_only=False) still produces both heads as usual.
774:            if rec_only:
775:                _grec, bn_feat_rec = self._embed_rec(rec_map)
777:                    'bn_feat_rec': bn_feat_rec,
778:                    'logits_rec': self.classifier_rec(bn_feat_rec),
781:            _grec, bn_feat_rec = self._embed_rec(rec_map)
783:                # eval: ONE forward -> two L2-normalized features (f_full, f_rec).
784:                return (F.normalize(bn_feat, dim=1),
785:                        F.normalize(bn_feat_rec, dim=1))
787:                'global_feat': global_feat,   # f_full triplet (shared trunk)
791:                # f_rec head over the INDEPENDENT rec map (own ID-CE + AIRL
792:                # consistency in the train loop); pooled rec feat exposed too so the
794:                'global_feat_rec': _grec,
795:                'bn_feat_rec': bn_feat_rec,
796:                'logits_rec': self.classifier_rec(bn_feat_rec),
806:            f_full = F.normalize(bn_feat, dim=1)
809:                # two L2-normalized features for the distmat-level soft fusion.
810:                bn_feat_rec = self.bottleneck_rec(global_feat)
811:                return f_full, F.normalize(bn_feat_rec, dim=1)
812:            return f_full
822:            # f_rec head: its OWN BNNeck + classifier on the shared global_feat.
823:            # The train loop applies f_rec ID-CE (so f_rec is a valid identity
826:            # NOT duplicated for f_rec (single backbone-level triplet).
827:            bn_feat_rec = self.bottleneck_rec(global_feat)
828:            out['bn_feat_rec'] = bn_feat_rec
829:            out['logits_rec'] = self.classifier_rec(bn_feat_rec)
895:        airl_dualbranch_iso=getattr(args, 'airl_dualbranch_iso', False),
896:        airl_iso_stage=getattr(args, 'airl_iso_stage', 3),
897:        airl_iso_trunk_recce=getattr(args, 'airl_iso_trunk_recce', True),

exec
/bin/zsh -lc 'rg -n "do_inference|airl_dualbranch_iso|airl_fuse_w|FUSE|A->G|G->A|full mAP|rec mAP|baseline|mAP|cos|normalize|extract|feature|query|gallery|evaluate" code/afd_train.py' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
3:CVPB training loop on CARGO = the afd_reid baseline trainer + OVP-Mem / OVLI.
7:baseline is reproduced exactly).  They are NOT mutually exclusive: --ovp,
20:            score is a hybrid of (a) global cosine and (b) a symmetric
37:                delta_ij = cos(z_i, P[y_j, view_j]) - cos(z_i, P[y_i, view_j]),
60:      score(i,j) = alpha * cos(g_i, g_j)
87:  identical to the baseline).  --ovli_rerank additionally reports a
94:    PK sampler P=16 x K=4 (bs=64), AdamW lr 3.5e-4, 10-ep warmup + cosine, 60 ep.
95:    eval every 10 ep: A->G and G->A cross-view mAP / R1 / mINP.
99:    L2-normalized BNNeck feature in a register_buffer of shape
104:         prototypes:  CE( cos(z, P[:, opp_view]) / tau ,  y ).
132:    # baseline reproduction: drop all of --ovp / --ovli
153:from agreid_v2_combined import AGReIDV2Combined  # noqa: E402  -- official exp1(A->G)+exp4(G->A) (--dataset agreid_v2)
169:    Prototypes are L2-normalized; features used for update/loss are the
170:    L2-normalized BNNeck features (detached for the EMA update).
188:        re-normalize the prototype.
201:                gmean = F.normalize(gmean, dim=0)
208:                    self.bank[pid_i, v] = F.normalize(self.bank[pid_i, v], dim=0)
257:    single mechanism (52.37 cross-view mAP).
261:    mAP collapsed to 14.66 << 52.37 << even pure global 45.14), because the
262:    random aggregated vector is meaningless and drags the cross-view cosine down.
271:    residual.  `mean_k(tok)` is the UN-normalized mean of the K L2-normed tokens
273:    baseline matches the current best mechanism exactly.  residual=True REQUIRES
289:                    aggregation -> intra-normalized (C x D) VLAD -> linear to
291:      attn        : H learnable query vectors; multi-head attention pooling over
293:                    Transformer PMA / learned-query attention pooling.)
299:                    normalize, flatten -> linear (token second-order statistics).
331:            self.query = nn.Parameter(torch.randn(H, self.head_dim) * 0.02)
346:        # the raw Parameters (centers / query) keep their small-random init above.
354:        # residual module's own params (centers / query / Linear / gate MLP) have
371:            vlad = F.normalize(vlad, dim=2)                       # intra-norm /cluster
373:            vlad = F.normalize(vlad, dim=1)                       # global VLAD L2-norm
378:            scores = (t * self.query.view(1, 1, H, Dh)).sum(-1) / math.sqrt(Dh)
397:        gate zero-init, so at step 0 the output == the un-normalized mean of the K
399:        L2-norm), so its gram is the UN-normalized `mean @ mean.T` == the 52.37
401:        pooling REPLACES the mean (random-init), and aggregate_tokens L2-normalizes
402:        it (cosine gram) -> the collapsing control.  Both branches are
418:    Reuses the maxsim_probe token-extraction recipe (hook model.layer4 -> the
439:        self.alpha = float(alpha)             # weight on global cosine in score
467:        #   match='maxsim' (default): for each query token take the MAX similarity
475:        #     the outer pool over query tokens, alpha mixing and the loss are
481:        #   align='free' (default): every query token may match ANY token in the
484:        #     alignment -- the K tokens form a (gh x gw) spatial grid; a query
513:        #     the gram of those aggregated vectors (residual mode: UN-normalized
515:        #     L2-normed cosine gram).  The match / pool / align / topk / thresh
531:        # centers/query + kaiming-inits its Linear layers, which would otherwise
556:    # -- token extraction ---------------------------------------------------- #
561:        projection runs in fp32 (numerical safety for the cos/MaxSim/logsumexp
562:        downstream); tokens are L2-normalized per token.
574:        tok = F.normalize(tok, dim=2)                               # per-token L2
587:        The 52.37 avg/mean path computes its cross-view gram as the UN-normalized
589:        tokens is NOT re-normalized -- see --ovli_match avg --ovli_pool mean, which
595:            UN-normalized mean gram == the 52.37 avg/mean-pool gram BYTE-FOR-BYTE
597:            here would instead give `<normalize(mean), normalize(mean)>` (diag==1),
602:            REPLACES the mean, so it IS L2-normalized -> the cross-view gram is a
603:            cosine in [-1,1] (the original standalone convention; this branch is
608:            # raw mean(+residual): UN-normalized gram == 52.37 avg/mean-pool path.
610:        return F.normalize(a, dim=1)           # standalone: unit vectors -> cosine gram
655:    # -- per-query-token reduction over the OTHER token set ------------------ #
658:        tensor into one score per query token, honoring the match / align modes.
660:        `sim` layout: the two token axes are at dim 1 (query tokens) and dim 3
661:        (other tokens); `other_dim` is the axis to reduce (3 = query->other,
662:        1 = other->query).  The (K,K) row mask is symmetric so the same buffer
682:        # align == 'ordered': restrict each query token to its same-row others.
707:        UN-normalized `mean(+residual) @ .T`, so at gate_res==0 it is byte-equal to
709:        the L2-normed cosine gram.  setpool == 'mean' (default) leaves the original
715:            return a @ a.t()                       # (B,B) cosine gram, symmetric
760:        delta_ij = cos(z_i, P[y_j, view_j]) - cos(z_i, P[y_i, view_j])
763:                    negative; z_i, P[.] L2-normed so cos == dot).
768:        # work in fp32 for the cos/sigmoid/clamp/log numerics (gfeat may be fp32
784:        cos_self = (zb * proto_self).sum(-1)                       # (B,B) cos(z_i,P[y_i,vj])
785:        cos_neg = (zb * proto_neg).sum(-1)                        # (B,B) cos(z_i,P[y_j,vj])
786:        delta = cos_neg - cos_self                                 # (B,B) ambiguity
790:        # initialised (never-seen prototype is a zero vector -> meaningless cos);
817:        gfeat:(B,D) L2-normed global feature (gradient flows -> encoder).
842:        # pairwise hybrid score in fp32 (cos in [-1,1], maxsim in [-1,1])
914:    """Report A->G / G->A mAP/R1 for (a) global-only and (b) global+MaxSim
917:    Mirrors run_cross_view_eval but additionally extracts projected tokens via
918:    the OVLI hook and reranks by score = alpha*cos(global) + (1-alpha)*MaxSim.
919:    Gallery token sets can be large, so MaxSim is chunked over the gallery axis.
920:    Returns {tag: {'global': (mAP,R1), 'rerank': (mAP,R1)}}.
927:    def extract(samples):
957:        """(Nq,Ng) bidirectional MaxSim, chunked over the gallery axis."""
960:        # used by the train loss -> residual mode = UN-normalized mean(+residual)
961:        # gram == 52.37 avg/mean path at gate_res==0; standalone = cosine gram),
962:        # NOT the token-set MaxSim.  Aggregate query/gallery tokens in sample-row
1002:        'A->G': (_fbv(dataset.query, 'Aerial'), _fbv(dataset.gallery, 'Ground')),
1003:        'G->A': (_fbv(dataset.query, 'Ground'), _fbv(dataset.gallery, 'Aerial')),
1006:        qf, qt, qp, qc = extract(q)
1007:        gf, gt, gp, gc = extract(g)
1012:        qf = F.normalize(qf, dim=1)
1013:        gf = F.normalize(gf, dim=1)
1014:        gsim = (qf @ gf.t()).numpy()                      # (Nq,Ng) cosine
1015:        # global-only (rank by cosine distance == -gsim)
1017:        # rerank: alpha*cos + (1-alpha)*MaxSim, rank by descending hybrid
1030:    """AIRL dual-branch eval: extract BOTH heads (f_full, f_rec) in ONE forward
1031:    and report f_full-only, f_rec-only, and the SOFT-FUSED cosine ranking
1032:    (cos = w*cos_rec + (1-w)*cos_full, w = args.airl_fuse_w, fixed) for A->G and
1033:    G->A.  This is the single-model analog of the kill-switch #3 two-model score
1034:    fusion: cos_rec replaces the AIRL-model cosine, cos_full replaces the
1035:    baseline-model cosine, and they share ONE backbone forward.
1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
1039:    f_full number reproduces run_cross_view_eval's A<->G mAP bit-for-bit (same
1040:    feature, same ranking) and the fusion is a pure distance-matrix combination.
1041:    Returns {tag: {'full': (mAP,R1), 'rec': (mAP,R1), 'fuse': (mAP,R1)}}.
1051:    def extract(samples):
1058:            # ONE forward -> two L2-normalized features (f_full, f_rec).
1071:    w = args.airl_fuse_w
1074:        'A->G': (_fbv(dataset.query, 'Aerial'), _fbv(dataset.gallery, 'Ground')),
1075:        'G->A': (_fbv(dataset.query, 'Ground'), _fbv(dataset.gallery, 'Aerial')),
1078:        q_full, q_rec, qp, qc = extract(q)
1079:        g_full, g_rec, gp, gc = extract(g)
1084:        # features are already L2-normalized at eval; renormalize defensively so
1085:        # the cosine == the gram of unit vectors (matches eval_market exactly).
1086:        q_full = F.normalize(q_full, dim=1); g_full = F.normalize(g_full, dim=1)
1087:        q_rec = F.normalize(q_rec, dim=1);   g_rec = F.normalize(g_rec, dim=1)
1088:        s_full = (q_full @ g_full.t()).numpy()        # (Nq,Ng) cosine, f_full
1089:        s_rec = (q_rec @ g_rec.t()).numpy()           # (Nq,Ng) cosine, f_rec
1090:        # soft fusion: cos = w*cos_rec + (1-w)*cos_full -> distance = 2 - 2*cos
1091:        # (identical to kill-switch #3 GATE 5; cosine in [-1,1] -> dist in [0,4]).
1110:# bucketed diagnostic showed the lowest aerial-scale bucket collapses by +13~19 mAP
1111:# vs the top bucket -- on the STRONG Swin baseline too, so it is a physical pixel
1125:#      (KL on logits, or cosine/MSE on the L2-normed BNNeck feature).  Intuition:
1134:#     the baseline is reproduced BYTE-FOR-BYTE (the whole AIRL block is skipped).
1135:#   * The consistency loss runs in TRUE fp32 (autocast disabled) for KL/cosine
1136:#     numeric safety (finite inputs: logits/features from a finite forward).
1144:    dataloader; degradation is a linear resample in normalized space, which is a
1204:      mode='feat': 1 - cos(bn_o.detach, bn_d) on the L2-normed BNNeck feature
1213:        zo = F.normalize(bn_o.float(), dim=1).detach()
1214:        zd = F.normalize(bn_d.float(), dim=1)
1215:        # 1 - cosine in [0,2]; mean over batch.  (== 0.5*||zo-zd||^2 on unit vecs.)
1238:    # 'agreid_v2' = AG-ReID.v2 OFFICIAL protocols: A->G == exp1 aerial_to_cctv,
1239:    #               G->A == exp4 cctv_to_aerial, mean of the two (the analogue of
1258:    # model switches (keep AFD off by default -> pure BoT baseline + OVP)
1261:    # backbone selector. 'resnet50' (default) = the existing BoT baseline
1269:                    help="backbone: resnet50 (default, BoT baseline, byte-identical) "
1315:                    help='score = alpha*cos(global) + (1-alpha)*sym_MaxSim(tokens)')
1351:                         "= for each query token take the MAX similarity over the "
1362:                         "(default) = each query token may match ANY other token "
1364:                         "AlignedReID-style row-ordered alignment: a query token "
1372:    # the cosine gram of those vectors).  This REPLACES the fixed "mean over
1384:                         "attn = multi-head learned-query attention pooling; "
1438:    # consistency.  Default OFF -> the baseline trains byte-for-byte (no degrade,
1442:    # the headline AIRL run is --airl alone on the plain baseline).
1448:                         'baseline byte-for-byte.')
1460:                         "1 - cosine on the L2-normed BNNeck feature.")
1473:    # evidence (protects G->A), f_rec gets its own ID-CE PLUS the AIRL
1475:    # evidence, serves A->G).  At eval the two heads' cosine scores are
1477:    #     cos = airl_fuse_w * cos(f_rec) + (1 - airl_fuse_w) * cos(f_full)
1479:    # per-query gate) -> this internalises the kill-switch #3 two-model score
1482:    # query-routing collision): "observation-limited evidence ceiling under which
1484:    # FIXED-PRIOR soft fusion".  This is deliberately NOT query-budget routing --
1485:    # kill-switch #3 showed hard per-query routing (area / reliability) fails to
1489:    # the single-head baseline byte-for-byte.
1494:                         'at eval (cos = w*cos_rec + (1-w)*cos_full). One forward, '
1495:                         'two features. Default OFF reproduces the baseline.')
1496:    ap.add_argument('--airl_fuse_w', type=float, default=0.25,
1497:                    help='fixed global fusion weight on the f_rec cosine at eval '
1498:                         '(cos = airl_fuse_w*cos_rec + (1-airl_fuse_w)*cos_full); '
1504:    ap.add_argument('--airl_dualbranch_iso', action='store_true',
1520:                         '+ --airl_fuse_w). Default OFF reproduces the baseline.')
1529:                         'Must be in [1,3]. Only used with --airl_dualbranch_iso.')
1531:    # iso left f_full WEAK (ep20 45.56 < baseline 48.98 < even fully-shared f_rec
1540:    # with --airl_dualbranch_iso.
1548:                         'without --airl_dualbranch_iso.')
1628:        if not (0.0 <= args.airl_fuse_w <= 1.0):
1629:            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
1630:                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
1636:        if args.airl_fuse_w != 0.25:
1637:            print(f"[AIRL-DUAL][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
1652:    #     warmup) and --airl_fuse_w, validated identically;
1657:    if args.airl_dualbranch_iso:
1659:            ap.error("--airl_dualbranch_iso is mutually exclusive with --airl and "
1664:            ap.error("--airl_dualbranch_iso requires --backbone swin_small (the rec "
1672:                     f"--airl_dualbranch_iso too); got {args.airl_min_scale}.")
1674:            ap.error("--airl_tau must be > 0 (used by --airl_dualbranch_iso too); "
1676:        if not (0.0 <= args.airl_fuse_w <= 1.0):
1677:            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
1678:                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
1679:        if args.airl_fuse_w != 0.25:
1680:            print(f"[AIRL-ISO][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
1684:            ap.error("--airl_dualbranch_iso is run standalone (headline AIRL); do "
1694:    print("CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)")
1700:        print(f"  backbone=resnet50 (BoT baseline) pool={args.pool} "
1720:          f"off => baseline byte-identical]")
1721:    print(f"  airl_dualbranch={args.airl_dualbranch} (fuse_w={args.airl_fuse_w} "
1726:          f"consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; "
1727:          f"1 forward 2 features; off => baseline byte-identical]")
1728:    print(f"  airl_dualbranch_iso={args.airl_dualbranch_iso} "
1730:          f"fuse_w={args.airl_fuse_w} "
1740:          f"eval; off => baseline byte-identical]")
1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
1751:        # so run_cross_view_eval/print_eval report the official per-protocol mAP
1813:    # a constant-output fixed point (last-stage map off-diag cos +0.99 -> all
1814:    # images map to ~one vector -> cross-view mAP 0.03).  No NaN; a genuine
1858:        # self-check: the learnable set-pool params (NetVLAD centers / attn query
1926:              f"trainable tensors); eval soft-fusion cos=w*cos_rec+(1-w)*cos_full "
1927:              f"w={args.airl_fuse_w}")
1933:    if args.airl_dualbranch_iso:
1978:              f"cos=w*cos_rec+(1-w)*cos_full w={args.airl_fuse_w}")
2003:        # --airl_dualbranch_iso (same consistency function, same warmup).  MUST list
2008:            if (args.airl or args.airl_dualbranch or args.airl_dualbranch_iso) else 0.0
2045:                if args.airl_dualbranch or args.airl_dualbranch_iso:
2047:                    # shared global_feat; for --airl_dualbranch_iso it reads the
2052:                    #   * --airl_dualbranch_iso, trunk_recce=1 (FIX) -> the shared
2058:                    #   * --airl_dualbranch_iso, trunk_recce=0 -> the isolated rec
2065:                    # OVP loss in fp32 for numerical safety (cosine + softmax)
2066:                    z = F.normalize(bn.float(), dim=1)
2070:            # OVLI: compute in TRUE fp32 (autocast disabled) -- the cos/MaxSim/
2079:                    # global feature for the score: normalized BN feat (matches
2081:                    g_ovli = F.normalize(bn.float(), dim=1)
2113:            # no extra forward, no loss) => the baseline trains byte-for-byte.
2158:            # (smoke D4) -> it keeps full-resolution discrimination (protects G->A);
2160:            # budget (serves A->G).  NOTE: the degraded forward below is a FULL
2214:            if args.airl_dualbranch_iso:
2253:                    ovp.update(F.normalize(bn.detach().float(), dim=1),
2256:            # batch (same detached BNNeck-feature recipe as OVP).  Done AFTER the
2261:                    acvp_mem.update(F.normalize(bn.detach().float(), dim=1),
2298:                if args.airl_dualbranch or args.airl_dualbranch_iso:
2363:        if args.airl_dualbranch or args.airl_dualbranch_iso:
2371:            tag = "AIRL-ISO" if args.airl_dualbranch_iso else "AIRL-DUAL"
2388:                for tag in ('A->G', 'G->A'):
2391:                    print(f"    [{tag}] global mAP={gm:.2f} R1={gr:.2f}  ->  "
2392:                          f"rerank mAP={rm:.2f} R1={rrk:.2f}")
2393:                rmean = (rr['A->G']['rerank'][0] + rr['G->A']['rerank'][0]) / 2
2394:                print(f"    [mean] rerank mAP={rmean:.2f}")
2395:            # AIRL dual-branch: report f_full-only, f_rec-only, and the SOFT-FUSED
2396:            # mean (cos = w*cos_rec + (1-w)*cos_full).  The run_cross_view_eval
2400:            if args.airl_dualbranch or args.airl_dualbranch_iso:
2406:                _ev_tag = ("AIRL-ISO dual-branch" if args.airl_dualbranch_iso
2408:                print(f"  ---- {_ev_tag} (fuse_w={args.airl_fuse_w}) "
2410:                for tag in ('A->G', 'G->A'):
2411:                    print(f"    [{tag}] full mAP={dual[tag]['full'][0]:.2f} "
2413:                          f"rec mAP={dual[tag]['rec'][0]:.2f} "
2415:                          f"FUSE mAP={dual[tag]['fuse'][0]:.2f} "
2417:                full_mean = (dual['A->G']['full'][0] + dual['G->A']['full'][0]) / 2
2418:                rec_mean = (dual['A->G']['rec'][0] + dual['G->A']['rec'][0]) / 2
2419:                fuse_mean = (dual['A->G']['fuse'][0] + dual['G->A']['fuse'][0]) / 2
2421:                      f"FUSE={fuse_mean:.2f}  <- model-selection uses FUSE")
2432:                print(f"    * new best mean mAP={best_map:.2f} (epoch {epoch}) saved")
2439:    print(f"Training finished. Best mean A<->G mAP={best_map:.2f} @ epoch {best_epoch}")

exec
/bin/zsh -lc 'rg -n "A->G|G->A|mAP|R1|FUSE|full mAP|rec mAP|Namespace|airl_dualbranch_iso|airl_fuse_w|airl_iso_stage|airl_iso_trunk_recce|seed|Final|Best|Best mAP" logs/agreidv2_airl_4090.log logs/agreidv2_airl_iso.log logs/agreidv2_baseline_4090.log logs/agreidv2_baseline_4090_s2.log logs/agreidv2_baseline.log' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
logs/agreidv2_baseline.log:9:  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_baseline.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_baseline.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_baseline.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_baseline.log:177:    [A->G] mAP=73.39  R1=82.60  R5=89.47  mINP=45.05
logs/agreidv2_baseline.log:178:    [G->A] mAP=73.97  R1=82.99  R5=89.51  mINP=35.92
logs/agreidv2_baseline.log:179:    [mean] mAP=73.68  R1=82.80
logs/agreidv2_baseline.log:180:    * new best mean mAP=73.68 (epoch 10) saved
logs/agreidv2_baseline.log:332:    [A->G] mAP=71.25  R1=80.22  R5=87.69  mINP=41.91
logs/agreidv2_baseline.log:333:    [G->A] mAP=71.19  R1=80.73  R5=87.85  mINP=34.72
logs/agreidv2_baseline.log:334:    [mean] mAP=71.22  R1=80.47
logs/agreidv2_baseline.log:486:    [A->G] mAP=72.09  R1=80.65  R5=88.16  mINP=43.73
logs/agreidv2_baseline.log:487:    [G->A] mAP=73.32  R1=82.88  R5=89.51  mINP=36.44
logs/agreidv2_baseline.log:488:    [mean] mAP=72.71  R1=81.76
logs/agreidv2_baseline.log:640:    [A->G] mAP=76.79  R1=84.08  R5=90.24  mINP=51.34
logs/agreidv2_baseline.log:641:    [G->A] mAP=76.84  R1=84.98  R5=90.28  mINP=41.65
logs/agreidv2_baseline.log:642:    [mean] mAP=76.82  R1=84.53
logs/agreidv2_baseline.log:643:    * new best mean mAP=76.82 (epoch 40) saved
logs/agreidv2_baseline.log:795:    [A->G] mAP=79.14  R1=85.78  R5=92.32  mINP=55.52
logs/agreidv2_baseline.log:796:    [G->A] mAP=79.29  R1=86.97  R5=91.50  mINP=45.80
logs/agreidv2_baseline.log:797:    [mean] mAP=79.22  R1=86.37
logs/agreidv2_baseline.log:798:    * new best mean mAP=79.22 (epoch 50) saved
logs/agreidv2_baseline.log:950:    [A->G] mAP=79.72  R1=86.42  R5=92.28  mINP=56.04
logs/agreidv2_baseline.log:951:    [G->A] mAP=80.04  R1=87.80  R5=92.16  mINP=46.62
logs/agreidv2_baseline.log:952:    [mean] mAP=79.88  R1=87.11
logs/agreidv2_baseline.log:953:    * new best mean mAP=79.88 (epoch 60) saved
logs/agreidv2_baseline.log:955:Training finished. Best mean A<->G mAP=79.88 @ epoch 60
logs/agreidv2_airl_iso.log:9:  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_airl_iso.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_airl_iso.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_airl_iso.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_airl_iso.log:178:    [A->G] mAP=71.36  R1=79.84  R5=87.61  mINP=43.40
logs/agreidv2_airl_iso.log:179:    [G->A] mAP=71.86  R1=81.61  R5=88.07  mINP=35.13
logs/agreidv2_airl_iso.log:180:    [mean] mAP=71.61  R1=80.73
logs/agreidv2_airl_iso.log:182:    [A->G] full mAP=71.36 R1=79.84 | rec mAP=72.28 R1=81.20 | FUSE mAP=72.89 R1=81.20
logs/agreidv2_airl_iso.log:183:    [G->A] full mAP=71.86 R1=81.61 | rec mAP=72.38 R1=81.61 | FUSE mAP=73.53 R1=82.99
logs/agreidv2_airl_iso.log:184:    [mean] full=71.61 rec=72.33 FUSE=73.21  <- model-selection uses FUSE
logs/agreidv2_airl_iso.log:185:    * new best mean mAP=73.21 (epoch 10) saved
logs/agreidv2_airl_iso.log:337:    [A->G] mAP=70.65  R1=80.18  R5=87.82  mINP=40.17
logs/agreidv2_airl_iso.log:338:    [G->A] mAP=71.55  R1=82.61  R5=88.02  mINP=32.51
logs/agreidv2_airl_iso.log:339:    [mean] mAP=71.10  R1=81.39
logs/agreidv2_airl_iso.log:341:    [A->G] full mAP=70.65 R1=80.18 | rec mAP=72.52 R1=81.75 | FUSE mAP=72.83 R1=81.45
logs/agreidv2_airl_iso.log:342:    [G->A] full mAP=71.55 R1=82.61 | rec mAP=71.05 R1=81.50 | FUSE mAP=73.14 R1=83.38
logs/agreidv2_airl_iso.log:343:    [mean] full=71.10 rec=71.79 FUSE=72.99  <- model-selection uses FUSE
logs/agreidv2_airl_iso.log:495:    [A->G] mAP=73.46  R1=82.77  R5=89.60  mINP=44.99
logs/agreidv2_airl_iso.log:496:    [G->A] mAP=73.80  R1=82.50  R5=88.90  mINP=37.87
logs/agreidv2_airl_iso.log:497:    [mean] mAP=73.63  R1=82.63
logs/agreidv2_airl_iso.log:499:    [A->G] full mAP=73.46 R1=82.77 | rec mAP=73.34 R1=82.00 | FUSE mAP=74.80 R1=83.36
logs/agreidv2_airl_iso.log:500:    [G->A] full mAP=73.80 R1=82.50 | rec mAP=73.06 R1=82.44 | FUSE mAP=75.26 R1=83.55
logs/agreidv2_airl_iso.log:501:    [mean] full=73.63 rec=73.20 FUSE=75.03  <- model-selection uses FUSE
logs/agreidv2_airl_iso.log:502:    * new best mean mAP=75.03 (epoch 30) saved
logs/agreidv2_airl_iso.log:654:    [A->G] mAP=76.09  R1=83.87  R5=90.70  mINP=49.93
logs/agreidv2_airl_iso.log:655:    [G->A] mAP=76.69  R1=84.82  R5=90.50  mINP=42.19
logs/agreidv2_airl_iso.log:656:    [mean] mAP=76.39  R1=84.34
logs/agreidv2_airl_iso.log:658:    [A->G] full mAP=76.09 R1=83.87 | rec mAP=75.40 R1=83.23 | FUSE mAP=77.58 R1=85.44
logs/agreidv2_airl_iso.log:659:    [G->A] full mAP=76.69 R1=84.82 | rec mAP=75.07 R1=83.88 | FUSE mAP=77.65 R1=85.37
logs/agreidv2_airl_iso.log:660:    [mean] full=76.39 rec=75.23 FUSE=77.62  <- model-selection uses FUSE
logs/agreidv2_airl_iso.log:661:    * new best mean mAP=77.62 (epoch 40) saved
logs/agreidv2_airl_4090.log:9:  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_airl_4090.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_airl_4090.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_airl_4090.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_airl_4090.log:177:    [A->G] mAP=74.91  R1=83.36  R5=89.98  mINP=46.90
logs/agreidv2_airl_4090.log:178:    [G->A] mAP=74.48  R1=82.94  R5=89.45  mINP=37.71
logs/agreidv2_airl_4090.log:179:    [mean] mAP=74.70  R1=83.15
logs/agreidv2_airl_4090.log:181:    [A->G] full mAP=74.91 R1=83.36 | rec mAP=74.08 R1=82.98 | FUSE mAP=75.66 R1=83.74
logs/agreidv2_airl_4090.log:182:    [G->A] full mAP=74.48 R1=82.94 | rec mAP=74.61 R1=82.94 | FUSE mAP=75.57 R1=84.04
logs/agreidv2_airl_4090.log:183:    [mean] full=74.70 rec=74.35 FUSE=75.61  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:184:    * new best mean mAP=75.61 (epoch 10) saved
logs/agreidv2_airl_4090.log:336:    [A->G] mAP=72.59  R1=83.11  R5=89.52  mINP=41.54
logs/agreidv2_airl_4090.log:337:    [G->A] mAP=73.67  R1=83.16  R5=89.45  mINP=35.62
logs/agreidv2_airl_4090.log:338:    [mean] mAP=73.13  R1=83.13
logs/agreidv2_airl_4090.log:340:    [A->G] full mAP=72.59 R1=83.11 | rec mAP=74.54 R1=83.62 | FUSE mAP=74.44 R1=84.30
logs/agreidv2_airl_4090.log:341:    [G->A] full mAP=73.67 R1=83.16 | rec mAP=73.89 R1=83.27 | FUSE mAP=75.19 R1=83.99
logs/agreidv2_airl_4090.log:342:    [mean] full=73.13 rec=74.21 FUSE=74.82  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:494:    [A->G] mAP=72.50  R1=81.07  R5=88.41  mINP=44.80
logs/agreidv2_airl_4090.log:495:    [G->A] mAP=73.73  R1=83.27  R5=88.40  mINP=38.48
logs/agreidv2_airl_4090.log:496:    [mean] mAP=73.11  R1=82.17
logs/agreidv2_airl_4090.log:498:    [A->G] full mAP=72.50 R1=81.07 | rec mAP=73.20 R1=81.83 | FUSE mAP=74.31 R1=83.15
logs/agreidv2_airl_4090.log:499:    [G->A] full mAP=73.73 R1=83.27 | rec mAP=73.70 R1=82.27 | FUSE mAP=75.06 R1=83.77
logs/agreidv2_airl_4090.log:500:    [mean] full=73.11 rec=73.45 FUSE=74.69  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:652:    [A->G] mAP=76.70  R1=85.10  R5=91.17  mINP=49.07
logs/agreidv2_airl_4090.log:653:    [G->A] mAP=76.49  R1=84.21  R5=89.95  mINP=42.15
logs/agreidv2_airl_4090.log:654:    [mean] mAP=76.60  R1=84.65
logs/agreidv2_airl_4090.log:656:    [A->G] full mAP=76.70 R1=85.10 | rec mAP=75.93 R1=83.40 | FUSE mAP=78.02 R1=85.70
logs/agreidv2_airl_4090.log:657:    [G->A] full mAP=76.49 R1=84.21 | rec mAP=75.44 R1=84.43 | FUSE mAP=77.87 R1=85.42
logs/agreidv2_airl_4090.log:658:    [mean] full=76.60 rec=75.68 FUSE=77.95  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:659:    * new best mean mAP=77.95 (epoch 40) saved
logs/agreidv2_airl_4090.log:811:    [A->G] mAP=78.34  R1=86.29  R5=91.85  mINP=52.48
logs/agreidv2_airl_4090.log:812:    [G->A] mAP=79.19  R1=86.69  R5=90.61  mINP=46.79
logs/agreidv2_airl_4090.log:813:    [mean] mAP=78.77  R1=86.49
logs/agreidv2_airl_4090.log:815:    [A->G] full mAP=78.34 R1=86.29 | rec mAP=77.75 R1=85.14 | FUSE mAP=79.63 R1=87.14
logs/agreidv2_airl_4090.log:816:    [G->A] full mAP=79.19 R1=86.69 | rec mAP=77.15 R1=85.15 | FUSE mAP=80.18 R1=87.24
logs/agreidv2_airl_4090.log:817:    [mean] full=78.77 rec=77.45 FUSE=79.90  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:818:    * new best mean mAP=79.90 (epoch 50) saved
logs/agreidv2_airl_4090.log:970:    [A->G] mAP=79.47  R1=87.18  R5=92.57  mINP=54.19
logs/agreidv2_airl_4090.log:971:    [G->A] mAP=80.33  R1=87.47  R5=91.83  mINP=47.95
logs/agreidv2_airl_4090.log:972:    [mean] mAP=79.90  R1=87.32
logs/agreidv2_airl_4090.log:974:    [A->G] full mAP=79.47 R1=87.18 | rec mAP=78.70 R1=84.93 | FUSE mAP=80.67 R1=87.90
logs/agreidv2_airl_4090.log:975:    [G->A] full mAP=80.33 R1=87.47 | rec mAP=78.38 R1=86.58 | FUSE mAP=81.29 R1=88.02
logs/agreidv2_airl_4090.log:976:    [mean] full=79.90 rec=78.54 FUSE=80.98  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:977:    * new best mean mAP=80.98 (epoch 60) saved
logs/agreidv2_airl_4090.log:979:Training finished. Best mean A<->G mAP=80.98 @ epoch 60
logs/agreidv2_baseline_4090.log:9:  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_baseline_4090.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_baseline_4090.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_baseline_4090.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_baseline_4090.log:176:    [A->G] mAP=73.40  R1=82.85  R5=89.26  mINP=44.92
logs/agreidv2_baseline_4090.log:177:    [G->A] mAP=72.69  R1=83.43  R5=89.45  mINP=35.03
logs/agreidv2_baseline_4090.log:178:    [mean] mAP=73.04  R1=83.14
logs/agreidv2_baseline_4090.log:179:    * new best mean mAP=73.04 (epoch 10) saved
logs/agreidv2_baseline_4090.log:331:    [A->G] mAP=72.37  R1=81.58  R5=89.05  mINP=43.54
logs/agreidv2_baseline_4090.log:332:    [G->A] mAP=73.14  R1=82.83  R5=88.85  mINP=36.99
logs/agreidv2_baseline_4090.log:333:    [mean] mAP=72.75  R1=82.20
logs/agreidv2_baseline_4090.log:485:    [A->G] mAP=71.72  R1=80.60  R5=87.90  mINP=43.02
logs/agreidv2_baseline_4090.log:486:    [G->A] mAP=73.24  R1=83.16  R5=88.96  mINP=35.74
logs/agreidv2_baseline_4090.log:487:    [mean] mAP=72.48  R1=81.88
logs/agreidv2_baseline_4090.log:639:    [A->G] mAP=76.21  R1=83.83  R5=91.00  mINP=48.95
logs/agreidv2_baseline_4090.log:640:    [G->A] mAP=77.13  R1=85.42  R5=91.17  mINP=42.14
logs/agreidv2_baseline_4090.log:641:    [mean] mAP=76.67  R1=84.63
logs/agreidv2_baseline_4090.log:642:    * new best mean mAP=76.67 (epoch 40) saved
logs/agreidv2_baseline_4090.log:794:    [A->G] mAP=79.75  R1=86.59  R5=92.44  mINP=55.00
logs/agreidv2_baseline_4090.log:795:    [G->A] mAP=80.22  R1=88.51  R5=92.21  mINP=46.92
logs/agreidv2_baseline_4090.log:796:    [mean] mAP=79.98  R1=87.55
logs/agreidv2_baseline_4090.log:797:    * new best mean mAP=79.98 (epoch 50) saved
logs/agreidv2_baseline_4090.log:949:    [A->G] mAP=80.73  R1=87.35  R5=93.00  mINP=56.91
logs/agreidv2_baseline_4090.log:950:    [G->A] mAP=81.42  R1=89.07  R5=93.21  mINP=48.32
logs/agreidv2_baseline_4090.log:951:    [mean] mAP=81.08  R1=88.21
logs/agreidv2_baseline_4090.log:952:    * new best mean mAP=81.08 (epoch 60) saved
logs/agreidv2_baseline_4090.log:954:Training finished. Best mean A<->G mAP=81.08 @ epoch 60
logs/agreidv2_baseline_4090_s2.log:9:  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_baseline_4090_s2.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_baseline_4090_s2.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_baseline_4090_s2.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_baseline_4090_s2.log:176:    [A->G] mAP=74.43  R1=83.49  R5=89.90  mINP=45.26
logs/agreidv2_baseline_4090_s2.log:177:    [G->A] mAP=74.11  R1=83.66  R5=90.50  mINP=36.04
logs/agreidv2_baseline_4090_s2.log:178:    [mean] mAP=74.27  R1=83.57
logs/agreidv2_baseline_4090_s2.log:179:    * new best mean mAP=74.27 (epoch 10) saved
logs/agreidv2_baseline_4090_s2.log:331:    [A->G] mAP=70.89  R1=80.73  R5=88.07  mINP=40.20
logs/agreidv2_baseline_4090_s2.log:332:    [G->A] mAP=70.82  R1=81.23  R5=87.36  mINP=32.87
logs/agreidv2_baseline_4090_s2.log:333:    [mean] mAP=70.85  R1=80.98
logs/agreidv2_baseline_4090_s2.log:485:    [A->G] mAP=72.48  R1=81.49  R5=89.05  mINP=44.27
logs/agreidv2_baseline_4090_s2.log:486:    [G->A] mAP=73.37  R1=82.44  R5=89.12  mINP=36.52
logs/agreidv2_baseline_4090_s2.log:487:    [mean] mAP=72.93  R1=81.97
logs/agreidv2_baseline_4090_s2.log:639:    [A->G] mAP=76.85  R1=84.76  R5=91.26  mINP=49.81
logs/agreidv2_baseline_4090_s2.log:640:    [G->A] mAP=76.72  R1=85.04  R5=90.17  mINP=42.94
logs/agreidv2_baseline_4090_s2.log:641:    [mean] mAP=76.79  R1=84.90
logs/agreidv2_baseline_4090_s2.log:642:    * new best mean mAP=76.79 (epoch 40) saved
logs/agreidv2_baseline_4090_s2.log:794:    [A->G] mAP=79.31  R1=86.67  R5=91.68  mINP=54.35
logs/agreidv2_baseline_4090_s2.log:795:    [G->A] mAP=79.77  R1=87.69  R5=92.10  mINP=46.88
logs/agreidv2_baseline_4090_s2.log:796:    [mean] mAP=79.54  R1=87.18
logs/agreidv2_baseline_4090_s2.log:797:    * new best mean mAP=79.54 (epoch 50) saved

codex
第一轮定位显示一个重要事实：AG-ReID.v2 的 AIRL final log 本身是 `trunk_recce=False`，不是题干 config 里的 `--airl_iso_trunk_recce 0` 以外的“fix”版本；baseline 有 lab-4090 两个 seed 和 lab-3090 一个 seed。接下来我会逐段读实现，尤其看 degraded pass 是否真的只训练 rec、FUSE 是否只在 score 矩阵上固定加权。
exec
/bin/zsh -lc "nl -ba code/afd_model.py | sed -n '80,360p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1020,1105p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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

exec
/bin/zsh -lc "nl -ba code/afd_model.py | sed -n '528,835p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
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

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1430,1690p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '2350,2445p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1910,2245p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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

codex
代码主路径基本闭合：iso 分支确实 deep-copy 了最后阶段，degraded consistency 通过 `rec_only=True` 只跑 rec 头，eval 的 FUSE 是固定 `0.25*rec + 0.75*full` 的距离矩阵融合。现在我会抽取日志成表，避免手算遗漏，再看是否有 seed/硬件/epoch 混用。
exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1700,1905p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1103,1225p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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

 succeeded in 0ms:
  1700	        print(f"  backbone=resnet50 (BoT baseline) pool={args.pool} "
  1701	              f"last_stride={args.last_stride}")
  1702	    print(f"  use_afd={args.use_afd}  ovp={args.ovp} "
  1703	          f"(lambda={args.ovp_lambda} tau={args.ovp_tau} mom={args.ovp_momentum})")
  1704	    print(f"  ovli={args.ovli} (lambda={args.ovli_lambda} tau={args.ovli_tau} "
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

exec
/bin/zsh -lc 'rg -n "seed|epochs=|eval_period|out_dir|backbone=|swin_lr_factor|torch|CUDA|Python|Command|Namespace|args" logs/agreidv2_airl_4090.log logs/agreidv2_airl_iso.log logs/agreidv2_baseline_4090.log logs/agreidv2_baseline_4090_s2.log logs/agreidv2_baseline.log' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
logs/agreidv2_baseline.log:3:  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/root/work/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
logs/agreidv2_baseline.log:10:  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
logs/agreidv2_baseline.log:11:  out_dir=/root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_baseline
logs/agreidv2_baseline_4090.log:3:  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/home/afr/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
logs/agreidv2_baseline_4090.log:10:  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
logs/agreidv2_baseline_4090.log:11:  out_dir=/home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090
logs/agreidv2_airl_iso.log:3:  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/root/work/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
logs/agreidv2_airl_iso.log:10:  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
logs/agreidv2_airl_iso.log:11:  out_dir=/root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_airl_iso
logs/agreidv2_baseline_4090_s2.log:3:  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/home/afr/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
logs/agreidv2_baseline_4090_s2.log:10:  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
logs/agreidv2_baseline_4090_s2.log:11:  out_dir=/home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090_s2
logs/agreidv2_airl_4090.log:3:  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/home/afr/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
logs/agreidv2_airl_4090.log:10:  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
logs/agreidv2_airl_4090.log:11:  out_dir=/home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090

exec
/bin/zsh -lc "rg -n \"Epoch\\[[0-9]+\\].*AIRL|Epoch\\[[0-9]+\\].*Loss|Training finished|new best|Checkpoints|airl_dualbranch_iso|AIRL-ISO\\]\" logs/agreidv2_airl_4090.log logs/agreidv2_airl_iso.log" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
logs/agreidv2_airl_iso.log:9:  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
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
logs/agreidv2_airl_4090.log:9:  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
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
logs/agreidv2_airl_4090.log:977:    * new best mean mAP=80.98 (epoch 60) saved
logs/agreidv2_airl_4090.log:979:Training finished. Best mean A<->G mAP=80.98 @ epoch 60
logs/agreidv2_airl_4090.log:980:Checkpoints in /home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090

codex
我看到 logs 里的 AIRL 两份都是 `trunk_recce=0` 原始 full-isolation，而不是默认 `trunk_recce=1` fix；这会影响“full 头≈baseline”的解释。现在用脚本把每个 eval epoch 的 full/rec/FUSE 和 baseline final/best 汇总出来。
exec
/bin/zsh -lc "sed -n '160,190p' logs/agreidv2_airl_4090.log" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
Epoch[9] done in 113.4s  Loss=3.009 Acc=0.957 AIRL-ISO[lam_eff=0.500 ce_rec=1.319 consistency=0.1735 deg_scale_mean=0.625 n_ground=28790]
Epoch[10] Iter[50/786] Loss: 3.217 CE: 1.494 Tri: 0.198 CE_rec: 1.442 AIRL_rec: 0.1660 Acc: 0.935 LR: 3.15e-05
Epoch[10] Iter[100/786] Loss: 3.163 CE: 1.472 Tri: 0.198 CE_rec: 1.407 AIRL_rec: 0.1713 Acc: 0.940 LR: 3.15e-05
Epoch[10] Iter[150/786] Loss: 3.115 CE: 1.452 Tri: 0.193 CE_rec: 1.382 AIRL_rec: 0.1748 Acc: 0.946 LR: 3.15e-05
Epoch[10] Iter[200/786] Loss: 3.101 CE: 1.442 Tri: 0.197 CE_rec: 1.372 AIRL_rec: 0.1787 Acc: 0.947 LR: 3.15e-05
Epoch[10] Iter[250/786] Loss: 3.111 CE: 1.443 Tri: 0.207 CE_rec: 1.370 AIRL_rec: 0.1799 Acc: 0.946 LR: 3.15e-05
Epoch[10] Iter[300/786] Loss: 3.097 CE: 1.437 Tri: 0.207 CE_rec: 1.363 AIRL_rec: 0.1808 Acc: 0.948 LR: 3.15e-05
Epoch[10] Iter[350/786] Loss: 3.089 CE: 1.434 Tri: 0.207 CE_rec: 1.358 AIRL_rec: 0.1815 Acc: 0.949 LR: 3.15e-05
Epoch[10] Iter[400/786] Loss: 3.081 CE: 1.430 Tri: 0.208 CE_rec: 1.352 AIRL_rec: 0.1811 Acc: 0.949 LR: 3.15e-05
Epoch[10] Iter[450/786] Loss: 3.071 CE: 1.426 Tri: 0.208 CE_rec: 1.346 AIRL_rec: 0.1803 Acc: 0.950 LR: 3.15e-05
Epoch[10] Iter[500/786] Loss: 3.065 CE: 1.423 Tri: 0.208 CE_rec: 1.343 AIRL_rec: 0.1808 Acc: 0.951 LR: 3.15e-05
Epoch[10] Iter[550/786] Loss: 3.051 CE: 1.418 Tri: 0.204 CE_rec: 1.338 AIRL_rec: 0.1800 Acc: 0.952 LR: 3.15e-05
Epoch[10] Iter[600/786] Loss: 3.039 CE: 1.413 Tri: 0.203 CE_rec: 1.333 AIRL_rec: 0.1797 Acc: 0.953 LR: 3.15e-05
Epoch[10] Iter[650/786] Loss: 3.022 CE: 1.406 Tri: 0.201 CE_rec: 1.326 AIRL_rec: 0.1784 Acc: 0.954 LR: 3.15e-05
Epoch[10] Iter[700/786] Loss: 3.003 CE: 1.398 Tri: 0.199 CE_rec: 1.318 AIRL_rec: 0.1769 Acc: 0.956 LR: 3.15e-05
Epoch[10] done in 113.0s  Loss=2.981 Acc=0.957 AIRL-ISO[lam_eff=0.500 ce_rec=1.309 consistency=0.1752 deg_scale_mean=0.624 n_ground=28708]
  ---- A<->G cross-view eval @ epoch 10 ----
    [A->G] mAP=74.91  R1=83.36  R5=89.98  mINP=46.90
    [G->A] mAP=74.48  R1=82.94  R5=89.45  mINP=37.71
    [mean] mAP=74.70  R1=83.15
  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 10 ----
    [A->G] full mAP=74.91 R1=83.36 | rec mAP=74.08 R1=82.98 | FUSE mAP=75.66 R1=83.74
    [G->A] full mAP=74.48 R1=82.94 | rec mAP=74.61 R1=82.94 | FUSE mAP=75.57 R1=84.04
    [mean] full=74.70 rec=74.35 FUSE=75.61  <- model-selection uses FUSE
    * new best mean mAP=75.61 (epoch 10) saved
Epoch[11] Iter[50/786] Loss: 3.172 CE: 1.486 Tri: 0.169 CE_rec: 1.430 AIRL_rec: 0.1741 Acc: 0.932 LR: 3.50e-05
Epoch[11] Iter[100/786] Loss: 3.127 CE: 1.459 Tri: 0.182 CE_rec: 1.399 AIRL_rec: 0.1745 Acc: 0.941 LR: 3.50e-05
Epoch[11] Iter[150/786] Loss: 3.078 CE: 1.438 Tri: 0.180 CE_rec: 1.371 AIRL_rec: 0.1776 Acc: 0.947 LR: 3.50e-05
Epoch[11] Iter[200/786] Loss: 3.061 CE: 1.429 Tri: 0.182 CE_rec: 1.360 AIRL_rec: 0.1791 Acc: 0.949 LR: 3.50e-05
Epoch[11] Iter[250/786] Loss: 3.060 CE: 1.426 Tri: 0.187 CE_rec: 1.357 AIRL_rec: 0.1795 Acc: 0.949 LR: 3.50e-05
Epoch[11] Iter[300/786] Loss: 3.049 CE: 1.422 Tri: 0.186 CE_rec: 1.349 AIRL_rec: 0.1822 Acc: 0.950 LR: 3.50e-05

exec
/bin/zsh -lc "sed -n '165,185p' logs/agreidv2_baseline_4090.log" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
Epoch[10] Iter[300/786] Loss: 1.630 CE: 1.432 Tri: 0.199 Acc: 0.950 LR: 3.15e-05
Epoch[10] Iter[350/786] Loss: 1.626 CE: 1.428 Tri: 0.197 Acc: 0.951 LR: 3.15e-05
Epoch[10] Iter[400/786] Loss: 1.623 CE: 1.425 Tri: 0.199 Acc: 0.952 LR: 3.15e-05
Epoch[10] Iter[450/786] Loss: 1.618 CE: 1.419 Tri: 0.199 Acc: 0.953 LR: 3.15e-05
Epoch[10] Iter[500/786] Loss: 1.613 CE: 1.416 Tri: 0.198 Acc: 0.953 LR: 3.15e-05
Epoch[10] Iter[550/786] Loss: 1.606 CE: 1.410 Tri: 0.196 Acc: 0.954 LR: 3.15e-05
Epoch[10] Iter[600/786] Loss: 1.600 CE: 1.406 Tri: 0.194 Acc: 0.954 LR: 3.15e-05
Epoch[10] Iter[650/786] Loss: 1.595 CE: 1.401 Tri: 0.194 Acc: 0.955 LR: 3.15e-05
Epoch[10] Iter[700/786] Loss: 1.584 CE: 1.392 Tri: 0.192 Acc: 0.957 LR: 3.15e-05
Epoch[10] done in 91.1s  Loss=1.571 Acc=0.958
  ---- A<->G cross-view eval @ epoch 10 ----
    [A->G] mAP=73.40  R1=82.85  R5=89.26  mINP=44.92
    [G->A] mAP=72.69  R1=83.43  R5=89.45  mINP=35.03
    [mean] mAP=73.04  R1=83.14
    * new best mean mAP=73.04 (epoch 10) saved
Epoch[11] Iter[50/786] Loss: 1.728 CE: 1.512 Tri: 0.216 Acc: 0.927 LR: 3.50e-05
Epoch[11] Iter[100/786] Loss: 1.648 CE: 1.468 Tri: 0.179 Acc: 0.942 LR: 3.50e-05
Epoch[11] Iter[150/786] Loss: 1.617 CE: 1.440 Tri: 0.177 Acc: 0.948 LR: 3.50e-05
Epoch[11] Iter[200/786] Loss: 1.611 CE: 1.432 Tri: 0.179 Acc: 0.949 LR: 3.50e-05
Epoch[11] Iter[250/786] Loss: 1.616 CE: 1.431 Tri: 0.185 Acc: 0.949 LR: 3.50e-05
Epoch[11] Iter[300/786] Loss: 1.606 CE: 1.425 Tri: 0.180 Acc: 0.950 LR: 3.50e-05

exec
/bin/zsh -lc "perl -ne 'BEGIN{print \"file,epoch,tag,full,rec,fuse,base_map,base_r1\\n\"} if(/Epoch\\[(\\d+)\\] done/){"'$e=$1} if(/'"\\[(A->G|G->A)\\] full mAP=([0-9.]+).*rec mAP=([0-9.]+).*FUSE mAP=([0-9.]+)/){print \""'$ARGV,$e,$1,$2,$3,$4,,'"\\n\"} if(/\\[(A->G|G->A)\\] mAP=([0-9.]+)\\s+R1=([0-9.]+)/ && "'!/full mAP/){print "$ARGV,$e,$1,,,,,$2,$3'"\\n\"} if(/\\[mean\\] full=([0-9.]+) rec=([0-9.]+) FUSE=([0-9.]+)/){print \""'$ARGV,$e,mean,$1,$2,$3,,'"\\n\"} if(/\\[mean\\] mAP=([0-9.]+)\\s+R1=([0-9.]+)/){print \""'$ARGV,$e,mean,,,,,$1,$2'"\\n\"}' logs/agreidv2_airl_4090.log logs/agreidv2_airl_iso.log logs/agreidv2_baseline_4090.log logs/agreidv2_baseline_4090_s2.log logs/agreidv2_baseline.log" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
file,epoch,tag,full,rec,fuse,base_map,base_r1
logs/agreidv2_airl_4090.log,10,A->G,,,,,74.91,83.36
logs/agreidv2_airl_4090.log,10,G->A,,,,,74.48,82.94
logs/agreidv2_airl_4090.log,10,mean,,,,,74.70,83.15
logs/agreidv2_airl_4090.log,10,A->G,74.91,74.08,75.66,,
logs/agreidv2_airl_4090.log,10,G->A,74.48,74.61,75.57,,
logs/agreidv2_airl_4090.log,10,mean,74.70,74.35,75.61,,
logs/agreidv2_airl_4090.log,20,A->G,,,,,72.59,83.11
logs/agreidv2_airl_4090.log,20,G->A,,,,,73.67,83.16
logs/agreidv2_airl_4090.log,20,mean,,,,,73.13,83.13
logs/agreidv2_airl_4090.log,20,A->G,72.59,74.54,74.44,,
logs/agreidv2_airl_4090.log,20,G->A,73.67,73.89,75.19,,
logs/agreidv2_airl_4090.log,20,mean,73.13,74.21,74.82,,
logs/agreidv2_airl_4090.log,30,A->G,,,,,72.50,81.07
logs/agreidv2_airl_4090.log,30,G->A,,,,,73.73,83.27
logs/agreidv2_airl_4090.log,30,mean,,,,,73.11,82.17
logs/agreidv2_airl_4090.log,30,A->G,72.50,73.20,74.31,,
logs/agreidv2_airl_4090.log,30,G->A,73.73,73.70,75.06,,
logs/agreidv2_airl_4090.log,30,mean,73.11,73.45,74.69,,
logs/agreidv2_airl_4090.log,40,A->G,,,,,76.70,85.10
logs/agreidv2_airl_4090.log,40,G->A,,,,,76.49,84.21
logs/agreidv2_airl_4090.log,40,mean,,,,,76.60,84.65
logs/agreidv2_airl_4090.log,40,A->G,76.70,75.93,78.02,,
logs/agreidv2_airl_4090.log,40,G->A,76.49,75.44,77.87,,
logs/agreidv2_airl_4090.log,40,mean,76.60,75.68,77.95,,
logs/agreidv2_airl_4090.log,50,A->G,,,,,78.34,86.29
logs/agreidv2_airl_4090.log,50,G->A,,,,,79.19,86.69
logs/agreidv2_airl_4090.log,50,mean,,,,,78.77,86.49
logs/agreidv2_airl_4090.log,50,A->G,78.34,77.75,79.63,,
logs/agreidv2_airl_4090.log,50,G->A,79.19,77.15,80.18,,
logs/agreidv2_airl_4090.log,50,mean,78.77,77.45,79.90,,
logs/agreidv2_airl_4090.log,60,A->G,,,,,79.47,87.18
logs/agreidv2_airl_4090.log,60,G->A,,,,,80.33,87.47
logs/agreidv2_airl_4090.log,60,mean,,,,,79.90,87.32
logs/agreidv2_airl_4090.log,60,A->G,79.47,78.70,80.67,,
logs/agreidv2_airl_4090.log,60,G->A,80.33,78.38,81.29,,
logs/agreidv2_airl_4090.log,60,mean,79.90,78.54,80.98,,
logs/agreidv2_airl_iso.log,10,A->G,,,,,71.36,79.84
logs/agreidv2_airl_iso.log,10,G->A,,,,,71.86,81.61
logs/agreidv2_airl_iso.log,10,mean,,,,,71.61,80.73
logs/agreidv2_airl_iso.log,10,A->G,71.36,72.28,72.89,,
logs/agreidv2_airl_iso.log,10,G->A,71.86,72.38,73.53,,
logs/agreidv2_airl_iso.log,10,mean,71.61,72.33,73.21,,
logs/agreidv2_airl_iso.log,20,A->G,,,,,70.65,80.18
logs/agreidv2_airl_iso.log,20,G->A,,,,,71.55,82.61
logs/agreidv2_airl_iso.log,20,mean,,,,,71.10,81.39
logs/agreidv2_airl_iso.log,20,A->G,70.65,72.52,72.83,,
logs/agreidv2_airl_iso.log,20,G->A,71.55,71.05,73.14,,
logs/agreidv2_airl_iso.log,20,mean,71.10,71.79,72.99,,
logs/agreidv2_airl_iso.log,30,A->G,,,,,73.46,82.77
logs/agreidv2_airl_iso.log,30,G->A,,,,,73.80,82.50
logs/agreidv2_airl_iso.log,30,mean,,,,,73.63,82.63
logs/agreidv2_airl_iso.log,30,A->G,73.46,73.34,74.80,,
logs/agreidv2_airl_iso.log,30,G->A,73.80,73.06,75.26,,
logs/agreidv2_airl_iso.log,30,mean,73.63,73.20,75.03,,
logs/agreidv2_airl_iso.log,40,A->G,,,,,76.09,83.87
logs/agreidv2_airl_iso.log,40,G->A,,,,,76.69,84.82
logs/agreidv2_airl_iso.log,40,mean,,,,,76.39,84.34
logs/agreidv2_airl_iso.log,40,A->G,76.09,75.40,77.58,,
logs/agreidv2_airl_iso.log,40,G->A,76.69,75.07,77.65,,
logs/agreidv2_airl_iso.log,40,mean,76.39,75.23,77.62,,
logs/agreidv2_baseline_4090.log,10,A->G,,,,,73.40,82.85
logs/agreidv2_baseline_4090.log,10,G->A,,,,,72.69,83.43
logs/agreidv2_baseline_4090.log,10,mean,,,,,73.04,83.14
logs/agreidv2_baseline_4090.log,20,A->G,,,,,72.37,81.58
logs/agreidv2_baseline_4090.log,20,G->A,,,,,73.14,82.83
logs/agreidv2_baseline_4090.log,20,mean,,,,,72.75,82.20
logs/agreidv2_baseline_4090.log,30,A->G,,,,,71.72,80.60
logs/agreidv2_baseline_4090.log,30,G->A,,,,,73.24,83.16
logs/agreidv2_baseline_4090.log,30,mean,,,,,72.48,81.88
logs/agreidv2_baseline_4090.log,40,A->G,,,,,76.21,83.83
logs/agreidv2_baseline_4090.log,40,G->A,,,,,77.13,85.42
logs/agreidv2_baseline_4090.log,40,mean,,,,,76.67,84.63
logs/agreidv2_baseline_4090.log,50,A->G,,,,,79.75,86.59
logs/agreidv2_baseline_4090.log,50,G->A,,,,,80.22,88.51
logs/agreidv2_baseline_4090.log,50,mean,,,,,79.98,87.55
logs/agreidv2_baseline_4090.log,60,A->G,,,,,80.73,87.35
logs/agreidv2_baseline_4090.log,60,G->A,,,,,81.42,89.07
logs/agreidv2_baseline_4090.log,60,mean,,,,,81.08,88.21
logs/agreidv2_baseline_4090_s2.log,10,A->G,,,,,74.43,83.49
logs/agreidv2_baseline_4090_s2.log,10,G->A,,,,,74.11,83.66
logs/agreidv2_baseline_4090_s2.log,10,mean,,,,,74.27,83.57
logs/agreidv2_baseline_4090_s2.log,20,A->G,,,,,70.89,80.73
logs/agreidv2_baseline_4090_s2.log,20,G->A,,,,,70.82,81.23
logs/agreidv2_baseline_4090_s2.log,20,mean,,,,,70.85,80.98
logs/agreidv2_baseline_4090_s2.log,30,A->G,,,,,72.48,81.49
logs/agreidv2_baseline_4090_s2.log,30,G->A,,,,,73.37,82.44
logs/agreidv2_baseline_4090_s2.log,30,mean,,,,,72.93,81.97
logs/agreidv2_baseline_4090_s2.log,40,A->G,,,,,76.85,84.76
logs/agreidv2_baseline_4090_s2.log,40,G->A,,,,,76.72,85.04
logs/agreidv2_baseline_4090_s2.log,40,mean,,,,,76.79,84.90
logs/agreidv2_baseline_4090_s2.log,50,A->G,,,,,79.31,86.67
logs/agreidv2_baseline_4090_s2.log,50,G->A,,,,,79.77,87.69
logs/agreidv2_baseline_4090_s2.log,50,mean,,,,,79.54,87.18
logs/agreidv2_baseline.log,10,A->G,,,,,73.39,82.60
logs/agreidv2_baseline.log,10,G->A,,,,,73.97,82.99
logs/agreidv2_baseline.log,10,mean,,,,,73.68,82.80
logs/agreidv2_baseline.log,20,A->G,,,,,71.25,80.22
logs/agreidv2_baseline.log,20,G->A,,,,,71.19,80.73
logs/agreidv2_baseline.log,20,mean,,,,,71.22,80.47
logs/agreidv2_baseline.log,30,A->G,,,,,72.09,80.65
logs/agreidv2_baseline.log,30,G->A,,,,,73.32,82.88
logs/agreidv2_baseline.log,30,mean,,,,,72.71,81.76
logs/agreidv2_baseline.log,40,A->G,,,,,76.79,84.08
logs/agreidv2_baseline.log,40,G->A,,,,,76.84,84.98
logs/agreidv2_baseline.log,40,mean,,,,,76.82,84.53
logs/agreidv2_baseline.log,50,A->G,,,,,79.14,85.78
logs/agreidv2_baseline.log,50,G->A,,,,,79.29,86.97
logs/agreidv2_baseline.log,50,mean,,,,,79.22,86.37
logs/agreidv2_baseline.log,60,A->G,,,,,79.72,86.42
logs/agreidv2_baseline.log,60,G->A,,,,,80.04,87.80
logs/agreidv2_baseline.log,60,mean,,,,,79.88,87.11

exec
/bin/zsh -lc "perl -ne 'if(/Training finished\\. Best mean A<->G mAP=([0-9.]+) @ epoch (\\d+)/){print \""'$ARGV best=$1 epoch=$2'"\\n\"}' logs/agreidv2_airl_4090.log logs/agreidv2_airl_iso.log logs/agreidv2_baseline_4090.log logs/agreidv2_baseline_4090_s2.log logs/agreidv2_baseline.log" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
logs/agreidv2_airl_4090.log best=80.98 epoch=60
logs/agreidv2_baseline_4090.log best=81.08 epoch=60
logs/agreidv2_baseline.log best=79.88 epoch=60

codex
日志汇总里有一个完整性问题：`agreidv2_baseline_4090_s2.log` 没有训练完成行，当前只到 epoch 50 eval 后的部分；不能把它当 final seed 使用。AIRL 4090 final FUSE 是 80.98，baseline 4090 seed1 final 是 81.08。
exec
/bin/zsh -lc 'tail -n 80 logs/agreidv2_baseline_4090_s2.log' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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
/bin/zsh -lc 'tail -n 60 logs/agreidv2_airl_iso.log' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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

exec
/bin/zsh -lc "perl -ne 'if(/\\[mean\\] full=([0-9.]+) rec=([0-9.]+) FUSE=([0-9.]+)/){printf \"%s epoch=%s FUSE-full=%.2f FUSE-rec=%.2f rec-full=%.2f\\n\","'$ARGV,$e,$3-$1,$3-$2,$2-$1} if(/Epoch'"\\[(\\d+)\\] done/){"'$e=$1}'"' logs/agreidv2_airl_4090.log logs/agreidv2_airl_iso.log" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
exec
/bin/zsh -lc "perl -ne 'if(/\\[A->G\\] full mAP=([0-9.]+).*rec mAP=([0-9.]+).*FUSE mAP=([0-9.]+)/){printf \"%s epoch=%s A->G FUSE-full=%.2f rec-full=%.2f\\n\","'$ARGV,$e,$3-$1,$2-$1} if(/'"\\[G->A\\] full mAP=([0-9.]+).*rec mAP=([0-9.]+).*FUSE mAP=([0-9.]+)/){printf \"%s epoch=%s G->A FUSE-full=%.2f rec-full=%.2f\\n\","'$ARGV,$e,$3-$1,$2-$1} if(/Epoch'"\\[(\\d+)\\] done/){"'$e=$1}'"' logs/agreidv2_airl_4090.log logs/agreidv2_airl_iso.log" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
logs/agreidv2_airl_4090.log epoch=10 FUSE-full=0.91 FUSE-rec=1.26 rec-full=-0.35
logs/agreidv2_airl_4090.log epoch=20 FUSE-full=1.69 FUSE-rec=0.61 rec-full=1.08
logs/agreidv2_airl_4090.log epoch=30 FUSE-full=1.58 FUSE-rec=1.24 rec-full=0.34
logs/agreidv2_airl_4090.log epoch=40 FUSE-full=1.35 FUSE-rec=2.27 rec-full=-0.92
logs/agreidv2_airl_4090.log epoch=50 FUSE-full=1.13 FUSE-rec=2.45 rec-full=-1.32
logs/agreidv2_airl_4090.log epoch=60 FUSE-full=1.08 FUSE-rec=2.44 rec-full=-1.36
logs/agreidv2_airl_iso.log epoch=10 FUSE-full=1.60 FUSE-rec=0.88 rec-full=0.72
logs/agreidv2_airl_iso.log epoch=20 FUSE-full=1.89 FUSE-rec=1.20 rec-full=0.69
logs/agreidv2_airl_iso.log epoch=30 FUSE-full=1.40 FUSE-rec=1.83 rec-full=-0.43
logs/agreidv2_airl_iso.log epoch=40 FUSE-full=1.23 FUSE-rec=2.39 rec-full=-1.16

 succeeded in 0ms:
logs/agreidv2_airl_4090.log epoch=10 A->G FUSE-full=0.75 rec-full=-0.83
logs/agreidv2_airl_4090.log epoch=10 G->A FUSE-full=1.09 rec-full=0.13
logs/agreidv2_airl_4090.log epoch=20 A->G FUSE-full=1.85 rec-full=1.95
logs/agreidv2_airl_4090.log epoch=20 G->A FUSE-full=1.52 rec-full=0.22
logs/agreidv2_airl_4090.log epoch=30 A->G FUSE-full=1.81 rec-full=0.70
logs/agreidv2_airl_4090.log epoch=30 G->A FUSE-full=1.33 rec-full=-0.03
logs/agreidv2_airl_4090.log epoch=40 A->G FUSE-full=1.32 rec-full=-0.77
logs/agreidv2_airl_4090.log epoch=40 G->A FUSE-full=1.38 rec-full=-1.05
logs/agreidv2_airl_4090.log epoch=50 A->G FUSE-full=1.29 rec-full=-0.59
logs/agreidv2_airl_4090.log epoch=50 G->A FUSE-full=0.99 rec-full=-2.04
logs/agreidv2_airl_4090.log epoch=60 A->G FUSE-full=1.20 rec-full=-0.77
logs/agreidv2_airl_4090.log epoch=60 G->A FUSE-full=0.96 rec-full=-1.95
logs/agreidv2_airl_iso.log epoch=10 A->G FUSE-full=1.53 rec-full=0.92
logs/agreidv2_airl_iso.log epoch=10 G->A FUSE-full=1.67 rec-full=0.52
logs/agreidv2_airl_iso.log epoch=20 A->G FUSE-full=2.18 rec-full=1.87
logs/agreidv2_airl_iso.log epoch=20 G->A FUSE-full=1.59 rec-full=-0.50
logs/agreidv2_airl_iso.log epoch=30 A->G FUSE-full=1.34 rec-full=-0.12
logs/agreidv2_airl_iso.log epoch=30 G->A FUSE-full=1.46 rec-full=-0.74
logs/agreidv2_airl_iso.log epoch=40 A->G FUSE-full=1.49 rec-full=-0.69
logs/agreidv2_airl_iso.log epoch=40 G->A FUSE-full=0.96 rec-full=-1.62

codex
AG-ReID.v2 的 FUSE−full 稳定为约 +1.1 到 +1.9，但 final 只有 +1.08；不是题干里的 +1.5。更关键的是 net gain 对 lab-4090 baseline final 是 -0.10，而不是正增益。现在我会查 bundle 内是否还有 CARGO AIRL/baseline log，否则“双数据集一致”和 CARGO 的数不能从本轮给定 logs 验证。
exec
/bin/zsh -lc 'find logs -maxdepth 1 -type f -print' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
logs/agreidv2_airl_4090.log
logs/agreidv2_baseline.log
logs/agreidv2_baseline_4090.log
logs/agreidv2_airl_iso.log
logs/agreidv2_baseline_4090_s2.log

exec
/bin/zsh -lc 'rg -n "CARGO|cargo|FUSE|full mAP|airl_dualbranch_iso|Training finished|Best mean|A->G|G->A" logs' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
logs/agreidv2_baseline_4090_s2.log:9:  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_baseline_4090_s2.log:11:  out_dir=/home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090_s2
logs/agreidv2_baseline_4090_s2.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_baseline_4090_s2.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_baseline_4090_s2.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_baseline_4090_s2.log:176:    [A->G] mAP=74.43  R1=83.49  R5=89.90  mINP=45.26
logs/agreidv2_baseline_4090_s2.log:177:    [G->A] mAP=74.11  R1=83.66  R5=90.50  mINP=36.04
logs/agreidv2_baseline_4090_s2.log:331:    [A->G] mAP=70.89  R1=80.73  R5=88.07  mINP=40.20
logs/agreidv2_baseline_4090_s2.log:332:    [G->A] mAP=70.82  R1=81.23  R5=87.36  mINP=32.87
logs/agreidv2_baseline_4090_s2.log:485:    [A->G] mAP=72.48  R1=81.49  R5=89.05  mINP=44.27
logs/agreidv2_baseline_4090_s2.log:486:    [G->A] mAP=73.37  R1=82.44  R5=89.12  mINP=36.52
logs/agreidv2_baseline_4090_s2.log:639:    [A->G] mAP=76.85  R1=84.76  R5=91.26  mINP=49.81
logs/agreidv2_baseline_4090_s2.log:640:    [G->A] mAP=76.72  R1=85.04  R5=90.17  mINP=42.94
logs/agreidv2_baseline_4090_s2.log:794:    [A->G] mAP=79.31  R1=86.67  R5=91.68  mINP=54.35
logs/agreidv2_baseline_4090_s2.log:795:    [G->A] mAP=79.77  R1=87.69  R5=92.10  mINP=46.88
logs/agreidv2_airl_iso.log:9:  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_airl_iso.log:11:  out_dir=/root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_airl_iso
logs/agreidv2_airl_iso.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_airl_iso.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_airl_iso.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_airl_iso.log:178:    [A->G] mAP=71.36  R1=79.84  R5=87.61  mINP=43.40
logs/agreidv2_airl_iso.log:179:    [G->A] mAP=71.86  R1=81.61  R5=88.07  mINP=35.13
logs/agreidv2_airl_iso.log:182:    [A->G] full mAP=71.36 R1=79.84 | rec mAP=72.28 R1=81.20 | FUSE mAP=72.89 R1=81.20
logs/agreidv2_airl_iso.log:183:    [G->A] full mAP=71.86 R1=81.61 | rec mAP=72.38 R1=81.61 | FUSE mAP=73.53 R1=82.99
logs/agreidv2_airl_iso.log:184:    [mean] full=71.61 rec=72.33 FUSE=73.21  <- model-selection uses FUSE
logs/agreidv2_airl_iso.log:337:    [A->G] mAP=70.65  R1=80.18  R5=87.82  mINP=40.17
logs/agreidv2_airl_iso.log:338:    [G->A] mAP=71.55  R1=82.61  R5=88.02  mINP=32.51
logs/agreidv2_airl_iso.log:341:    [A->G] full mAP=70.65 R1=80.18 | rec mAP=72.52 R1=81.75 | FUSE mAP=72.83 R1=81.45
logs/agreidv2_airl_iso.log:342:    [G->A] full mAP=71.55 R1=82.61 | rec mAP=71.05 R1=81.50 | FUSE mAP=73.14 R1=83.38
logs/agreidv2_airl_iso.log:343:    [mean] full=71.10 rec=71.79 FUSE=72.99  <- model-selection uses FUSE
logs/agreidv2_airl_iso.log:495:    [A->G] mAP=73.46  R1=82.77  R5=89.60  mINP=44.99
logs/agreidv2_airl_iso.log:496:    [G->A] mAP=73.80  R1=82.50  R5=88.90  mINP=37.87
logs/agreidv2_airl_iso.log:499:    [A->G] full mAP=73.46 R1=82.77 | rec mAP=73.34 R1=82.00 | FUSE mAP=74.80 R1=83.36
logs/agreidv2_airl_iso.log:500:    [G->A] full mAP=73.80 R1=82.50 | rec mAP=73.06 R1=82.44 | FUSE mAP=75.26 R1=83.55
logs/agreidv2_airl_iso.log:501:    [mean] full=73.63 rec=73.20 FUSE=75.03  <- model-selection uses FUSE
logs/agreidv2_airl_iso.log:654:    [A->G] mAP=76.09  R1=83.87  R5=90.70  mINP=49.93
logs/agreidv2_airl_iso.log:655:    [G->A] mAP=76.69  R1=84.82  R5=90.50  mINP=42.19
logs/agreidv2_airl_iso.log:658:    [A->G] full mAP=76.09 R1=83.87 | rec mAP=75.40 R1=83.23 | FUSE mAP=77.58 R1=85.44
logs/agreidv2_airl_iso.log:659:    [G->A] full mAP=76.69 R1=84.82 | rec mAP=75.07 R1=83.88 | FUSE mAP=77.65 R1=85.37
logs/agreidv2_airl_iso.log:660:    [mean] full=76.39 rec=75.23 FUSE=77.62  <- model-selection uses FUSE
logs/agreidv2_baseline_4090.log:9:  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_baseline_4090.log:11:  out_dir=/home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090
logs/agreidv2_baseline_4090.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_baseline_4090.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_baseline_4090.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_baseline_4090.log:176:    [A->G] mAP=73.40  R1=82.85  R5=89.26  mINP=44.92
logs/agreidv2_baseline_4090.log:177:    [G->A] mAP=72.69  R1=83.43  R5=89.45  mINP=35.03
logs/agreidv2_baseline_4090.log:331:    [A->G] mAP=72.37  R1=81.58  R5=89.05  mINP=43.54
logs/agreidv2_baseline_4090.log:332:    [G->A] mAP=73.14  R1=82.83  R5=88.85  mINP=36.99
logs/agreidv2_baseline_4090.log:485:    [A->G] mAP=71.72  R1=80.60  R5=87.90  mINP=43.02
logs/agreidv2_baseline_4090.log:486:    [G->A] mAP=73.24  R1=83.16  R5=88.96  mINP=35.74
logs/agreidv2_baseline_4090.log:639:    [A->G] mAP=76.21  R1=83.83  R5=91.00  mINP=48.95
logs/agreidv2_baseline_4090.log:640:    [G->A] mAP=77.13  R1=85.42  R5=91.17  mINP=42.14
logs/agreidv2_baseline_4090.log:794:    [A->G] mAP=79.75  R1=86.59  R5=92.44  mINP=55.00
logs/agreidv2_baseline_4090.log:795:    [G->A] mAP=80.22  R1=88.51  R5=92.21  mINP=46.92
logs/agreidv2_baseline_4090.log:949:    [A->G] mAP=80.73  R1=87.35  R5=93.00  mINP=56.91
logs/agreidv2_baseline_4090.log:950:    [G->A] mAP=81.42  R1=89.07  R5=93.21  mINP=48.32
logs/agreidv2_baseline_4090.log:954:Training finished. Best mean A<->G mAP=81.08 @ epoch 60
logs/agreidv2_baseline_4090.log:955:Checkpoints in /home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090
logs/agreidv2_baseline.log:9:  airl_dualbranch_iso=False (iso_stage=3 trunk_recce=True fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_baseline.log:11:  out_dir=/root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_baseline
logs/agreidv2_baseline.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_baseline.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_baseline.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_baseline.log:177:    [A->G] mAP=73.39  R1=82.60  R5=89.47  mINP=45.05
logs/agreidv2_baseline.log:178:    [G->A] mAP=73.97  R1=82.99  R5=89.51  mINP=35.92
logs/agreidv2_baseline.log:332:    [A->G] mAP=71.25  R1=80.22  R5=87.69  mINP=41.91
logs/agreidv2_baseline.log:333:    [G->A] mAP=71.19  R1=80.73  R5=87.85  mINP=34.72
logs/agreidv2_baseline.log:486:    [A->G] mAP=72.09  R1=80.65  R5=88.16  mINP=43.73
logs/agreidv2_baseline.log:487:    [G->A] mAP=73.32  R1=82.88  R5=89.51  mINP=36.44
logs/agreidv2_baseline.log:640:    [A->G] mAP=76.79  R1=84.08  R5=90.24  mINP=51.34
logs/agreidv2_baseline.log:641:    [G->A] mAP=76.84  R1=84.98  R5=90.28  mINP=41.65
logs/agreidv2_baseline.log:795:    [A->G] mAP=79.14  R1=85.78  R5=92.32  mINP=55.52
logs/agreidv2_baseline.log:796:    [G->A] mAP=79.29  R1=86.97  R5=91.50  mINP=45.80
logs/agreidv2_baseline.log:950:    [A->G] mAP=79.72  R1=86.42  R5=92.28  mINP=56.04
logs/agreidv2_baseline.log:951:    [G->A] mAP=80.04  R1=87.80  R5=92.16  mINP=46.62
logs/agreidv2_baseline.log:955:Training finished. Best mean A<->G mAP=79.88 @ epoch 60
logs/agreidv2_baseline.log:956:Checkpoints in /root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_baseline
logs/agreidv2_airl_4090.log:9:  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
logs/agreidv2_airl_4090.log:11:  out_dir=/home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090
logs/agreidv2_airl_4090.log:13:=> AG-ReID.v2 loaded (OFFICIAL exp1 A->G + exp4 G->A, CARGO-aligned)
logs/agreidv2_airl_4090.log:14:   A->G = exp1 aerial_to_cctv (2356 q / 6347 g)
logs/agreidv2_airl_4090.log:15:   G->A = exp4 cctv_to_aerial (1811 q / 14362 g)
logs/agreidv2_airl_4090.log:177:    [A->G] mAP=74.91  R1=83.36  R5=89.98  mINP=46.90
logs/agreidv2_airl_4090.log:178:    [G->A] mAP=74.48  R1=82.94  R5=89.45  mINP=37.71
logs/agreidv2_airl_4090.log:181:    [A->G] full mAP=74.91 R1=83.36 | rec mAP=74.08 R1=82.98 | FUSE mAP=75.66 R1=83.74
logs/agreidv2_airl_4090.log:182:    [G->A] full mAP=74.48 R1=82.94 | rec mAP=74.61 R1=82.94 | FUSE mAP=75.57 R1=84.04
logs/agreidv2_airl_4090.log:183:    [mean] full=74.70 rec=74.35 FUSE=75.61  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:336:    [A->G] mAP=72.59  R1=83.11  R5=89.52  mINP=41.54
logs/agreidv2_airl_4090.log:337:    [G->A] mAP=73.67  R1=83.16  R5=89.45  mINP=35.62
logs/agreidv2_airl_4090.log:340:    [A->G] full mAP=72.59 R1=83.11 | rec mAP=74.54 R1=83.62 | FUSE mAP=74.44 R1=84.30
logs/agreidv2_airl_4090.log:341:    [G->A] full mAP=73.67 R1=83.16 | rec mAP=73.89 R1=83.27 | FUSE mAP=75.19 R1=83.99
logs/agreidv2_airl_4090.log:342:    [mean] full=73.13 rec=74.21 FUSE=74.82  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:494:    [A->G] mAP=72.50  R1=81.07  R5=88.41  mINP=44.80
logs/agreidv2_airl_4090.log:495:    [G->A] mAP=73.73  R1=83.27  R5=88.40  mINP=38.48
logs/agreidv2_airl_4090.log:498:    [A->G] full mAP=72.50 R1=81.07 | rec mAP=73.20 R1=81.83 | FUSE mAP=74.31 R1=83.15
logs/agreidv2_airl_4090.log:499:    [G->A] full mAP=73.73 R1=83.27 | rec mAP=73.70 R1=82.27 | FUSE mAP=75.06 R1=83.77
logs/agreidv2_airl_4090.log:500:    [mean] full=73.11 rec=73.45 FUSE=74.69  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:652:    [A->G] mAP=76.70  R1=85.10  R5=91.17  mINP=49.07
logs/agreidv2_airl_4090.log:653:    [G->A] mAP=76.49  R1=84.21  R5=89.95  mINP=42.15
logs/agreidv2_airl_4090.log:656:    [A->G] full mAP=76.70 R1=85.10 | rec mAP=75.93 R1=83.40 | FUSE mAP=78.02 R1=85.70
logs/agreidv2_airl_4090.log:657:    [G->A] full mAP=76.49 R1=84.21 | rec mAP=75.44 R1=84.43 | FUSE mAP=77.87 R1=85.42
logs/agreidv2_airl_4090.log:658:    [mean] full=76.60 rec=75.68 FUSE=77.95  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:811:    [A->G] mAP=78.34  R1=86.29  R5=91.85  mINP=52.48
logs/agreidv2_airl_4090.log:812:    [G->A] mAP=79.19  R1=86.69  R5=90.61  mINP=46.79
logs/agreidv2_airl_4090.log:815:    [A->G] full mAP=78.34 R1=86.29 | rec mAP=77.75 R1=85.14 | FUSE mAP=79.63 R1=87.14
logs/agreidv2_airl_4090.log:816:    [G->A] full mAP=79.19 R1=86.69 | rec mAP=77.15 R1=85.15 | FUSE mAP=80.18 R1=87.24
logs/agreidv2_airl_4090.log:817:    [mean] full=78.77 rec=77.45 FUSE=79.90  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:970:    [A->G] mAP=79.47  R1=87.18  R5=92.57  mINP=54.19
logs/agreidv2_airl_4090.log:971:    [G->A] mAP=80.33  R1=87.47  R5=91.83  mINP=47.95
logs/agreidv2_airl_4090.log:974:    [A->G] full mAP=79.47 R1=87.18 | rec mAP=78.70 R1=84.93 | FUSE mAP=80.67 R1=87.90
logs/agreidv2_airl_4090.log:975:    [G->A] full mAP=80.33 R1=87.47 | rec mAP=78.38 R1=86.58 | FUSE mAP=81.29 R1=88.02
logs/agreidv2_airl_4090.log:976:    [mean] full=79.90 rec=78.54 FUSE=80.98  <- model-selection uses FUSE
logs/agreidv2_airl_4090.log:979:Training finished. Best mean A<->G mAP=80.98 @ epoch 60
logs/agreidv2_airl_4090.log:980:Checkpoints in /home/afr/SOLIDER-REID/log/cargo/airl_iso_agreidv2_4090

exec
/bin/zsh -lc "rg --files /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle | rg '(cargo|airl|baseline).*\\.log"'$'"'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_baseline_4090_s2.log
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_iso.log
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_baseline_4090.log
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_baseline.log
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_4090.log

codex
bundle 里只有 AG-ReID.v2 日志，没有 CARGO 日志；CARGO 相关数字只能视为题干外部材料，不能由本轮 log 复核。下一步我查文献/先例，重点是“双分支/双特征 score fusion 是否已是常见 ensemble 风险”，然后给出审查结论。
web search: 
web search: person re-identification multi branch ensemble feature fusion score fusion cosine distance mAP
web search: 
web search: MGN multiple granularity network person re-identification multi branch feature fusion paper
web search: 
web search: low-resolution person re-identification resolution degradation consistency knowledge distillation paper ReID
web search: 
web search: person re-identification consistency regularization degraded image augmentation low resolution self distillation
codex
文献先例方向已经明确：低分辨/退化不变性、cross-resolution ReID、双流/多分支 ReID 和 score/feature 融合都有较多先例；AIRL 的证据链需要证明“退化一致性导致互补”，不能只拿 FUSE 提升当创新证据。现在补读构建模型、普通 eval、参数传入和日志完整性边界。
exec
/bin/zsh -lc "nl -ba code/afd_model.py | sed -n '835,898p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '880,1020p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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

 succeeded in 0ms:
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

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1225,1435p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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

exec
/bin/zsh -lc "rg -n \"add_argument\\('--seed|seed\\)|set_seed|def set_seed|manual_seed|cudnn|determin\" code/afd_train.py code/afd_model.py" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
code/afd_train.py:157:                       run_cross_view_eval, print_eval, set_seed)
code/afd_train.py:532:        # shift proj's init under a fixed seed).  With proj first, setpool != 'mean'
code/afd_train.py:1256:    ap.add_argument('--seed', type=int, default=1)
code/afd_train.py:1688:    set_seed(args.seed)

exec
/bin/zsh -lc "rg -n \"def run_cross_view_eval|def print_eval|def build_eval_loader|def extract_features|return_dual|model\\.eval|torch\\.no_grad\" code/afd_train.py" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
182:    @torch.no_grad()
732:    @torch.no_grad()
754:        EVERYTHING here is under torch.no_grad() and uses the DETACHED prototype
903:        with torch.no_grad():
912:@torch.no_grad()
924:    model.eval()
926:    @torch.no_grad()
955:    @torch.no_grad()
1047:    model.eval()
1050:    @torch.no_grad()
1059:            f_full, f_rec = model(imgs, view_idx=vidx, return_dual=True)
2135:                    with torch.no_grad():
2176:                    with torch.no_grad():
2220:                    with torch.no_grad():
2252:                with torch.no_grad():
2260:                with torch.no_grad():
2401:                # airl_dualbranch_eval calls model(return_dual=True) which yields

exec
/bin/zsh -lc "rg -n \"def build_model|AFDModel\\(|airl_dualbranch_iso|airl_iso_stage|airl_iso_trunk_recce|airl_fuse_w\" code/afd_model.py code/afd_train.py" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
code/afd_model.py:528:class AFDModel(nn.Module):
code/afd_model.py:540:                 airl_dualbranch_iso=False, airl_iso_stage=3,
code/afd_model.py:541:                 airl_iso_trunk_recce=True):
code/afd_model.py:564:        self.airl_dualbranch_iso = bool(airl_dualbranch_iso)
code/afd_model.py:565:        self.airl_iso_stage = int(airl_iso_stage)
code/afd_model.py:566:        # airl_iso_trunk_recce: route the CLEAN f_rec ID-CE gradient back into the
code/afd_model.py:570:        self.airl_iso_trunk_recce = bool(airl_iso_trunk_recce)
code/afd_model.py:571:        if self.airl_dualbranch_iso:
code/afd_model.py:573:                "airl_dualbranch_iso and airl_dualbranch are mutually exclusive "
code/afd_model.py:576:                "airl_dualbranch_iso requires backbone='swin_small' (the rec branch "
code/afd_model.py:627:                iso_branch=self.airl_dualbranch_iso,
code/afd_model.py:628:                iso_stage=self.airl_iso_stage,
code/afd_model.py:629:                iso_trunk_recce=self.airl_iso_trunk_recce)
code/afd_model.py:656:        #   * airl_dualbranch_iso : f_rec pools the INDEPENDENT rec last-stage map
code/afd_model.py:660:        if self.airl_dualbranch or self.airl_dualbranch_iso:
code/afd_model.py:744:               airl_dualbranch_iso: identical (f_full_norm, f_rec_norm) eval tuple
code/afd_model.py:763:        want_iso = self.airl_dualbranch_iso and (self.training or return_dual
code/afd_model.py:869:def build_model(num_classes, args):
code/afd_model.py:877:    return AFDModel(
code/afd_model.py:895:        airl_dualbranch_iso=getattr(args, 'airl_dualbranch_iso', False),
code/afd_model.py:896:        airl_iso_stage=getattr(args, 'airl_iso_stage', 3),
code/afd_model.py:897:        airl_iso_trunk_recce=getattr(args, 'airl_iso_trunk_recce', True),
code/afd_train.py:1032:    (cos = w*cos_rec + (1-w)*cos_full, w = args.airl_fuse_w, fixed) for A->G and
code/afd_train.py:1071:    w = args.airl_fuse_w
code/afd_train.py:1477:    #     cos = airl_fuse_w * cos(f_rec) + (1 - airl_fuse_w) * cos(f_full)
code/afd_train.py:1496:    ap.add_argument('--airl_fuse_w', type=float, default=0.25,
code/afd_train.py:1498:                         '(cos = airl_fuse_w*cos_rec + (1-airl_fuse_w)*cos_full); '
code/afd_train.py:1504:    ap.add_argument('--airl_dualbranch_iso', action='store_true',
code/afd_train.py:1514:                         'ID-CE routing is governed by --airl_iso_trunk_recce: default '
code/afd_train.py:1520:                         '+ --airl_fuse_w). Default OFF reproduces the baseline.')
code/afd_train.py:1521:    ap.add_argument('--airl_iso_stage', type=int, default=3,
code/afd_train.py:1529:                         'Must be in [1,3]. Only used with --airl_dualbranch_iso.')
code/afd_train.py:1534:    # got from BOTH heads' ID-CE.  --airl_iso_trunk_recce 1 (default) re-routes ONLY
code/afd_train.py:1540:    # with --airl_dualbranch_iso.
code/afd_train.py:1541:    ap.add_argument('--airl_iso_trunk_recce', type=int, default=1, choices=[0, 1],
code/afd_train.py:1548:                         'without --airl_dualbranch_iso.')
code/afd_train.py:1554:    args.airl_iso_trunk_recce = bool(args.airl_iso_trunk_recce)
code/afd_train.py:1628:        if not (0.0 <= args.airl_fuse_w <= 1.0):
code/afd_train.py:1629:            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
code/afd_train.py:1630:                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
code/afd_train.py:1636:        if args.airl_fuse_w != 0.25:
code/afd_train.py:1637:            print(f"[AIRL-DUAL][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
code/afd_train.py:1652:    #     warmup) and --airl_fuse_w, validated identically;
code/afd_train.py:1657:    if args.airl_dualbranch_iso:
code/afd_train.py:1659:            ap.error("--airl_dualbranch_iso is mutually exclusive with --airl and "
code/afd_train.py:1664:            ap.error("--airl_dualbranch_iso requires --backbone swin_small (the rec "
code/afd_train.py:1666:        if not (1 <= args.airl_iso_stage <= 3):
code/afd_train.py:1667:            ap.error("--airl_iso_stage must be in [1,3] (swin_small has 4 stages "
code/afd_train.py:1669:                     f"{args.airl_iso_stage}.")
code/afd_train.py:1672:                     f"--airl_dualbranch_iso too); got {args.airl_min_scale}.")
code/afd_train.py:1674:            ap.error("--airl_tau must be > 0 (used by --airl_dualbranch_iso too); "
code/afd_train.py:1676:        if not (0.0 <= args.airl_fuse_w <= 1.0):
code/afd_train.py:1677:            ap.error("--airl_fuse_w must be in [0,1] (eval fusion weight cos = "
code/afd_train.py:1678:                     f"w*cos_rec + (1-w)*cos_full); got {args.airl_fuse_w}.")
code/afd_train.py:1679:        if args.airl_fuse_w != 0.25:
code/afd_train.py:1680:            print(f"[AIRL-ISO][WARN] --airl_fuse_w={args.airl_fuse_w} != 0.25: the "
code/afd_train.py:1684:            ap.error("--airl_dualbranch_iso is run standalone (headline AIRL); do "
code/afd_train.py:1721:    print(f"  airl_dualbranch={args.airl_dualbranch} (fuse_w={args.airl_fuse_w} "
code/afd_train.py:1728:    print(f"  airl_dualbranch_iso={args.airl_dualbranch_iso} "
code/afd_train.py:1729:          f"(iso_stage={args.airl_iso_stage} trunk_recce={args.airl_iso_trunk_recce} "
code/afd_train.py:1730:          f"fuse_w={args.airl_fuse_w} "
code/afd_train.py:1927:              f"w={args.airl_fuse_w}")
code/afd_train.py:1933:    if args.airl_dualbranch_iso:
code/afd_train.py:1969:                     "consistency stays detached)" if args.airl_iso_trunk_recce
code/afd_train.py:1972:        print(f"  [AIRL-ISO] iso_stage={args.airl_iso_stage}: rec late stage "
code/afd_train.py:1976:              f"(detached degraded pass at stage-{args.airl_iso_stage} input); "
code/afd_train.py:1978:              f"cos=w*cos_rec+(1-w)*cos_full w={args.airl_fuse_w}")
code/afd_train.py:2003:        # --airl_dualbranch_iso (same consistency function, same warmup).  MUST list
code/afd_train.py:2008:            if (args.airl or args.airl_dualbranch or args.airl_dualbranch_iso) else 0.0
code/afd_train.py:2045:                if args.airl_dualbranch or args.airl_dualbranch_iso:
code/afd_train.py:2047:                    # shared global_feat; for --airl_dualbranch_iso it reads the
code/afd_train.py:2052:                    #   * --airl_dualbranch_iso, trunk_recce=1 (FIX) -> the shared
code/afd_train.py:2058:                    #   * --airl_dualbranch_iso, trunk_recce=0 -> the isolated rec
code/afd_train.py:2202:            # consistency gradient (smoke I4) REGARDLESS of --airl_iso_trunk_recce
code/afd_train.py:2214:            if args.airl_dualbranch_iso:
code/afd_train.py:2298:                if args.airl_dualbranch or args.airl_dualbranch_iso:
code/afd_train.py:2363:        if args.airl_dualbranch or args.airl_dualbranch_iso:
code/afd_train.py:2371:            tag = "AIRL-ISO" if args.airl_dualbranch_iso else "AIRL-DUAL"
code/afd_train.py:2400:            if args.airl_dualbranch or args.airl_dualbranch_iso:
code/afd_train.py:2406:                _ev_tag = ("AIRL-ISO dual-branch" if args.airl_dualbranch_iso
code/afd_train.py:2408:                print(f"  ---- {_ev_tag} (fuse_w={args.airl_fuse_w}) "

codex
补读后没有看到 eval 公式本身的实现错误：普通 cross-view eval 是 f_full，dual eval 额外复算 full/rec/FUSE，并用 FUSE 覆盖 model-selection 指标。真正的风险集中在结果解释和对照缺失，而不是这段 FUSE 代码本身。
exec
/bin/zsh -lc "rg --files | rg 'afd_reid|baseline|cargo_dataset|maxsim_probe|agreid'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
logs/agreidv2_baseline_4090_s2.log
logs/agreidv2_airl_iso.log
logs/agreidv2_baseline_4090.log
logs/agreidv2_baseline.log
logs/agreidv2_airl_4090.log

exec
/bin/zsh -lc 'rg -n "def set_seed|def eval_market|def run_cross_view_eval|def print_eval|def build_eval_loader|filter_by_view|eval_from_distmat" -S .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
exec
/bin/zsh -lc "find . -maxdepth 3 -type f -name '*.md' -print | sort | head -200" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
./reviews/codex_6.md:492:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_6.md:493:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_6.md:666:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_6.md:5915:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_6.md:5948:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_6.md:5953:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_6.md:5955:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_6.md:6005:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_6.md:6006:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_6.md:6007:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_6.md:6432:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_6.md:6865:/bin/zsh -lc "rg -n \"def set_seed|class RandomIdentitySampler|torch.initial_seed|worker_init_fn|manual_seed|random.seed|np.random.seed|torch.backends|cudnn|DataLoader|build_eval_loader\" afd_reid code -g '*.py'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
./reviews/codex_6.md:6908:/bin/zsh -lc "rg -n \"def set_seed|class RandomIdentitySampler|def __iter__|np.random|random\\.seed|torch.manual_seed|cudnn|worker_init_fn|build_eval_loader|DataLoader\" /Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/afd_train.py /Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/cargo_dataset.py /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
./reviews/codex_6.md:6914:/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/afd_train.py:196:def build_eval_loader(samples, args):
./reviews/codex_6.md:6918:/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/afd_train.py:240:def set_seed(seed):
./reviews/codex_6.md:6968:    38	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_6.md:7378:   196	def build_eval_loader(samples, args):
./reviews/codex_6.md:7385:   203	def run_cross_view_eval(model, dataset, args, device):
./reviews/codex_6.md:7387:   205	    q_aerial = filter_by_view(dataset.query, 'Aerial')
./reviews/codex_6.md:7388:   206	    q_ground = filter_by_view(dataset.query, 'Ground')
./reviews/codex_6.md:7389:   240	def set_seed(seed):
./reviews/codex_2.md:406:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_2.md:496:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_2.md:993:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_2.md:994:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_2.md:1011:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_2.md:1016:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_2.md:1018:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_2.md:1068:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_2.md:1069:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_2.md:1070:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_2.md:2279:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_2.md:4760:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_2.md:4858:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_2.md:5076:/bin/zsh -lc "rg -n \"add_argument\\('--seed|seed\\)|set_seed|def set_seed|manual_seed|cudnn|determin\" code/afd_train.py code/afd_model.py" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
./reviews/codex_2.md:5084:/bin/zsh -lc "rg -n \"def run_cross_view_eval|def print_eval|def build_eval_loader|def extract_features|return_dual|model\\.eval|torch\\.no_grad\" code/afd_train.py" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
./reviews/codex_3.md:5055:1020:        from maxsim_probe import eval_from_distmat
./reviews/codex_3.md:5056:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_3.md:5065:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_3.md:5069:1045:    from maxsim_probe import eval_from_distmat
./reviews/codex_3.md:5092:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_3.md:5093:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_3.md:5094:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_3.md:5331:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_3.md:6208:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_3.md:6213:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_3.md:6215:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_3.md:6265:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_3.md:6266:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_3.md:6267:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:28:指标计算: mAP/Rank/eval_from_distmat 实现对吗?cos→dist 转换(2-2cos?)对吗?A->G / G->A 的 query/gallery 划分对吗?pid 解析有无错(folder name vs P-prefix 之类的坑)?camid 过滤对吗?
./reviews/codex_7.md:491:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_7.md:1061:code/afd_train.py:1020:        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:1062:code/afd_train.py:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:1072:code/afd_train.py:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:1077:code/afd_train.py:1045:    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:1105:code/afd_train.py:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:1106:code/afd_train.py:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:1107:code/afd_train.py:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:1310:code/afd_train.py:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:2861:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:2959:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:2960:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:2977:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:2982:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:2984:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:3034:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:3035:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:3036:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:3325:AIRL 双分支 soft-fuse 代码本身已定位：它先算两个头的 cosine，再用 `2 - 2*cos` 送 `eval_from_distmat`。我还需要确认 PID/cam/query-gallery 的来源，因为这些是从外部 dataset/eval helper 导入的，当前 bundle 可能缺文件。
./reviews/codex_7.md:3819:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:4544:/bin/zsh -lc 'rg -n "eval_from_distmat|eval_market|build_eval_loader|filter_by_view|class AGReID|class CARGO|parse|pid|camid|query|gallery" .' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
./reviews/codex_7.md:4571:./reviews/codex_6.md:492:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:4572:./reviews/codex_6.md:493:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:4573:./reviews/codex_6.md:666:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:4577:./reviews/codex_6.md:5915:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_7.md:4581:./reviews/codex_6.md:5948:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:4582:./reviews/codex_6.md:5953:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:4584:./reviews/codex_6.md:5955:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:4592:./reviews/codex_6.md:6005:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:4593:./reviews/codex_6.md:6006:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:4594:./reviews/codex_6.md:6007:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:4596:./reviews/codex_6.md:6432:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:4605:./launch_10codex.sh:26:'指标计算: mAP/Rank/eval_from_distmat 实现对吗?cos→dist 转换(2-2cos?)对吗?A->G / G->A 的 query/gallery 划分对吗?pid 解析有无错(folder name vs P-prefix 之类的坑)?camid 过滤对吗?'
./reviews/codex_7.md:4618:./code/afd_train.py:151:                           RandomIdentitySampler, filter_by_view)
./reviews/codex_7.md:4671:./code/afd_train.py:922:    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:4683:./code/afd_train.py:1020:        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:4684:./code/afd_train.py:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:4685:./code/afd_train.py:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:4686:./code/afd_train.py:1043:    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:4688:./code/afd_train.py:1045:    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:4696:./code/afd_train.py:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:4697:./code/afd_train.py:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:4698:./code/afd_train.py:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:4714:./code/afd_train.py:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:4797:./reviews/codex_1.md:568:1020:        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:4798:./reviews/codex_1.md:569:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:4799:./reviews/codex_1.md:576:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:4800:./reviews/codex_1.md:579:1045:    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:4808:./reviews/codex_1.md:605:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:4809:./reviews/codex_1.md:606:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:4810:./reviews/codex_1.md:607:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:4821:./reviews/codex_1.md:811:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:4832:./reviews/codex_1.md:3603:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:4844:./reviews/codex_1.md:3701:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:4845:./reviews/codex_1.md:3702:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:4846:./reviews/codex_1.md:3719:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:4847:./reviews/codex_1.md:3724:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:4849:./reviews/codex_1.md:3726:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:4857:./reviews/codex_1.md:3776:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:4858:./reviews/codex_1.md:3777:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:4859:./reviews/codex_1.md:3778:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:4872:./reviews/codex_5.md:284:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:4873:./reviews/codex_5.md:349:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:4893:./reviews/codex_5.md:6634:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:4905:./reviews/codex_5.md:6732:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:4906:./reviews/codex_5.md:6733:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:4907:./reviews/codex_5.md:6750:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:4908:./reviews/codex_5.md:6755:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:4910:./reviews/codex_5.md:6757:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:4918:./reviews/codex_5.md:6807:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:4919:./reviews/codex_5.md:6808:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:4920:./reviews/codex_5.md:6809:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:4923:./reviews/codex_8.md:295:code/afd_train.py:151:                           RandomIdentitySampler, filter_by_view)
./reviews/codex_7.md:4930:./reviews/codex_8.md:406:code/afd_train.py:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:4936:./reviews/codex_8.md:6039:1020:        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:4937:./reviews/codex_8.md:6040:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:4938:./reviews/codex_8.md:6044:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:4939:./reviews/codex_8.md:6045:1045:    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:4940:./reviews/codex_8.md:6052:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:4941:./reviews/codex_8.md:6053:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:4942:./reviews/codex_8.md:6054:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:4957:./reviews/codex_8.md:6339:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_7.md:5010:./reviews/codex_8.md:7120:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:5022:./reviews/codex_8.md:7218:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5023:./reviews/codex_8.md:7219:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:5024:./reviews/codex_8.md:7236:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:5025:./reviews/codex_8.md:7241:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:5027:./reviews/codex_8.md:7243:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5035:./reviews/codex_8.md:7293:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:5036:./reviews/codex_8.md:7294:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:5037:./reviews/codex_8.md:7295:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:5057:./reviews/codex_8.md:8432:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:5066:./reviews/codex_9.md:431:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:5067:./reviews/codex_9.md:432:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:5068:./reviews/codex_9.md:433:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:5070:./reviews/codex_9.md:640:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:5086:./reviews/codex_7.md:28:指标计算: mAP/Rank/eval_from_distmat 实现对吗?cos→dist 转换(2-2cos?)对吗?A->G / G->A 的 query/gallery 划分对吗?pid 解析有无错(folder name vs P-prefix 之类的坑)?camid 过滤对吗?
./reviews/codex_7.md:5099:./reviews/codex_7.md:491:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_7.md:5195:./reviews/codex_7.md:1061:code/afd_train.py:1020:        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5196:./reviews/codex_7.md:1062:code/afd_train.py:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:5197:./reviews/codex_7.md:1072:code/afd_train.py:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:5199:./reviews/codex_7.md:1077:code/afd_train.py:1045:    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5207:./reviews/codex_7.md:1105:code/afd_train.py:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:5208:./reviews/codex_7.md:1106:code/afd_train.py:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:5209:./reviews/codex_7.md:1107:code/afd_train.py:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:5220:./reviews/codex_7.md:1310:code/afd_train.py:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:5262:./reviews/codex_7.md:2861:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:5274:./reviews/codex_7.md:2959:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5275:./reviews/codex_7.md:2960:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:5276:./reviews/codex_7.md:2977:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:5277:./reviews/codex_7.md:2982:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:5279:./reviews/codex_7.md:2984:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5287:./reviews/codex_7.md:3034:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:5288:./reviews/codex_7.md:3035:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:5289:./reviews/codex_7.md:3036:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:5291:./reviews/codex_7.md:3325:AIRL 双分支 soft-fuse 代码本身已定位：它先算两个头的 cosine，再用 `2 - 2*cos` 送 `eval_from_distmat`。我还需要确认 PID/cam/query-gallery 的来源，因为这些是从外部 dataset/eval helper 导入的，当前 bundle 可能缺文件。
./reviews/codex_7.md:5306:./reviews/codex_7.md:3819:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:5328:./reviews/codex_10.md:499:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_7.md:5354:./reviews/codex_10.md:907:1020:        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5355:./reviews/codex_10.md:908:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:5356:./reviews/codex_10.md:916:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:5358:./reviews/codex_10.md:920:1045:    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5360:./reviews/codex_10.md:939:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:5361:./reviews/codex_10.md:940:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:5362:./reviews/codex_10.md:941:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:5371:./reviews/codex_3.md:5055:1020:        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5372:./reviews/codex_3.md:5056:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:5373:./reviews/codex_3.md:5065:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:5375:./reviews/codex_3.md:5069:1045:    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5379:./reviews/codex_3.md:5092:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:5380:./reviews/codex_3.md:5093:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:5381:./reviews/codex_3.md:5094:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:5382:./reviews/codex_3.md:5331:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:5386:./reviews/codex_3.md:6208:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:5387:./reviews/codex_3.md:6213:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:5389:./reviews/codex_3.md:6215:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5397:./reviews/codex_3.md:6265:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:5398:./reviews/codex_3.md:6266:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:5399:./reviews/codex_3.md:6267:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:5419:./reviews/codex_4.md:1141:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_7.md:5472:./reviews/codex_4.md:1912:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:5484:./reviews/codex_4.md:2010:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5485:./reviews/codex_4.md:2011:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:5486:./reviews/codex_4.md:2028:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:5487:./reviews/codex_4.md:2033:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:5489:./reviews/codex_4.md:2035:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5497:./reviews/codex_4.md:2085:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:5498:./reviews/codex_4.md:2086:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:5499:./reviews/codex_4.md:2087:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:5515:./reviews/codex_4.md:2740:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:5525:./reviews/codex_4.md:8041:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:5551:./reviews/codex_2.md:406:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:5563:./reviews/codex_2.md:496:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:5565:./reviews/codex_2.md:993:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5566:./reviews/codex_2.md:994:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_7.md:5567:./reviews/codex_2.md:1011:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_7.md:5568:./reviews/codex_2.md:1016:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_7.md:5570:./reviews/codex_2.md:1018:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_7.md:5578:./reviews/codex_2.md:1068:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_7.md:5579:./reviews/codex_2.md:1069:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_7.md:5580:./reviews/codex_2.md:1070:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_7.md:5589:./reviews/codex_2.md:2279:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_7.md:6217:关键风险也已显露：`eval_from_distmat`、`eval_market`、`build_eval_loader`、`filter_by_view`、AG-ReID.v2 的 PID/cam 解析都不是当前 bundle 的源文件，只能从调用点和日志交叉验证，不能做到逐行核验这些 helper 的内部实现。接下来我会尝试在允许范围内查找原文件；若读不到，会把它列为审查限制/风险。
./reviews/codex_7.md:6465:我能读到上一级真实源码。现在转去核 `eval_from_distmat`、AG-ReID.v2 combined split、PID/cam 解析和 baseline eval helper，这部分会决定本轮 lens 的结论。
./reviews/codex_7.md:6506:    38	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_7.md:6607:   139	def eval_market(qf, q_pids, q_camids, gf, g_pids, g_camids, max_rank=50):
./reviews/codex_7.md:6664:   196	def build_eval_loader(samples, args):
./reviews/codex_7.md:6671:   203	def run_cross_view_eval(model, dataset, args, device):
./reviews/codex_7.md:6673:   205	    q_aerial = filter_by_view(dataset.query, 'Aerial')
./reviews/codex_7.md:6674:   206	    q_ground = filter_by_view(dataset.query, 'Ground')
./reviews/codex_7.md:6675:   207	    g_aerial = filter_by_view(dataset.gallery, 'Aerial')
./reviews/codex_7.md:6676:   208	    g_ground = filter_by_view(dataset.gallery, 'Ground')
./reviews/codex_7.md:6693:   225	def print_eval(epoch, results):
./reviews/codex_7.md:6708:   240	def set_seed(seed):
./reviews/codex_7.md:6993:   261	def filter_by_view(samples, view):
./reviews/codex_7.md:7006:   274	    print("aerial query:", len(filter_by_view(ds.query, 'Aerial')),
./reviews/codex_7.md:7007:   275	          "ground query:", len(filter_by_view(ds.query, 'Ground')))
./reviews/codex_7.md:7029:    18	cross-view directions with `filter_by_view(dataset.query|gallery, 'Aerial'|'Ground')`.
./reviews/codex_7.md:7037:    26	    filter_by_view(query,   'Aerial') == exp1 query  (UAV)    \\  A->G == official exp1
./reviews/codex_7.md:7038:    27	    filter_by_view(gallery, 'Ground') == exp1 gallery (CCTV)  /
./reviews/codex_7.md:7039:    28	    filter_by_view(query,   'Ground') == exp4 query  (CCTV)   \\  G->A == official exp4
./reviews/codex_7.md:7040:    29	    filter_by_view(gallery, 'Aerial') == exp4 gallery (UAV)   /
./reviews/codex_7.md:7120:   109	        # Concatenate so filter_by_view recovers each official direction.
./reviews/codex_7.md:7142:   131	        A->G/G->A layout contract that filter_by_view depends on).
./reviews/codex_7.md:7217:    14	                              RandomIdentitySampler, filter_by_view
./reviews/codex_7.md:7246:    43	    CARGO cross-view helpers (filter_by_view, eval_market junk removal) keep
./reviews/codex_7.md:7526:    60	                           filter_by_view)
./reviews/codex_7.md:7632:   166	def eval_from_distmat(distmat, q_pids, q_camids, g_pids, g_camids, max_rank=50):
./reviews/codex_7.md:7729:我确认到了真实 `eval_market` 和 `eval_from_distmat`：两者都做 same pid & same camid junk removal，`eval_from_distmat` 返回百分制。还有一个需要核实的点：训练脚本的 `sys.path` 让它导入 `../afd_reid/afd_model.py`，所以我要确认 bundle 里的 `code/afd_model.py` 是否就是实际被训练用的那份。
./reviews/codex_7.md:7909:/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/afd_train.py:240:def set_seed(seed):
./launch_10codex.sh:26:'指标计算: mAP/Rank/eval_from_distmat 实现对吗?cos→dist 转换(2-2cos?)对吗?A->G / G->A 的 query/gallery 划分对吗?pid 解析有无错(folder name vs P-prefix 之类的坑)?camid 过滤对吗?'
./code/afd_train.py:151:                           RandomIdentitySampler, filter_by_view)
./code/afd_train.py:922:    from cargo_dataset import filter_by_view as _fbv
./code/afd_train.py:1020:        from maxsim_probe import eval_from_distmat
./code/afd_train.py:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./code/afd_train.py:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./code/afd_train.py:1043:    from cargo_dataset import filter_by_view as _fbv
./code/afd_train.py:1045:    from maxsim_probe import eval_from_distmat
./code/afd_train.py:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./code/afd_train.py:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./code/afd_train.py:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./code/afd_train.py:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_5.md:284:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_5.md:349:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_5.md:6634:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_5.md:6732:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_5.md:6733:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_5.md:6750:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_5.md:6755:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_5.md:6757:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_5.md:6807:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_5.md:6808:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_5.md:6809:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_8.md:295:code/afd_train.py:151:                           RandomIdentitySampler, filter_by_view)
./reviews/codex_8.md:406:code/afd_train.py:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_8.md:6039:1020:        from maxsim_probe import eval_from_distmat
./reviews/codex_8.md:6040:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_8.md:6044:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_8.md:6045:1045:    from maxsim_probe import eval_from_distmat
./reviews/codex_8.md:6052:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_8.md:6053:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_8.md:6054:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_8.md:6339:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_8.md:7120:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_8.md:7218:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_8.md:7219:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_8.md:7236:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_8.md:7241:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_8.md:7243:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_8.md:7293:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_8.md:7294:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_8.md:7295:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_8.md:8432:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_10.md:499:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_10.md:907:1020:        from maxsim_probe import eval_from_distmat
./reviews/codex_10.md:908:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_10.md:916:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_10.md:920:1045:    from maxsim_probe import eval_from_distmat
./reviews/codex_10.md:939:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_10.md:940:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_10.md:941:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_10.md:7375:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_10.md:7376:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_10.md:7393:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_10.md:7398:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_10.md:7400:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_10.md:7450:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_10.md:7451:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_10.md:7452:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_10.md:7815:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_10.md:8880:922:    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_10.md:8895:1043:    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_1.md:568:1020:        from maxsim_probe import eval_from_distmat
./reviews/codex_1.md:569:1021:        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_1.md:576:1038:    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_1.md:579:1045:    from maxsim_probe import eval_from_distmat
./reviews/codex_1.md:605:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_1.md:606:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_1.md:607:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_1.md:811:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_1.md:3603:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_1.md:3701:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_1.md:3702:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_1.md:3719:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_1.md:3724:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_1.md:3726:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_1.md:3776:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_1.md:3777:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_1.md:3778:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_1.md:4448:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_1.md:5163:/bin/zsh -lc 'rg -n "def build_eval_loader|def eval_market|def run_cross_view_eval|def print_eval|filter_by_view|AGReIDV2Combined|CARGOImageDataset" code/afd_train.py' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
./reviews/codex_1.md:5166:151:                           RandomIdentitySampler, filter_by_view)
./reviews/codex_1.md:5168:922:    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_1.md:5169:1043:    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_1.md:5170:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_1.md:5352:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_1.md:5374:有一个审查边界需要明示：这个 bundle 的 `code/afd_train.py` 复用了外部 `../afd_reid/afd_train.py` 里的 `run_cross_view_eval/print_eval/eval_market`，以及 `maxsim_probe.eval_from_distmat`。我会尝试读取这些依赖；如果沙箱不允许，就只能把这部分作为未完全覆盖的外部依赖风险记录。
./reviews/codex_1.md:5465:外部依赖可以读到，路径在 bundle 上一级。现在补查原始 eval helper 和 `eval_from_distmat`，这会决定是否存在 label/camid 泄漏或融合矩阵方向错误。
./reviews/codex_1.md:5467:/bin/zsh -lc 'rg -n "def build_eval_loader|def eval_market|def run_cross_view_eval|def print_eval|def extract_features|filter_by_view|CARGOImageDataset" ../afd_train.py ../maxsim_probe.py ../agreid_dataset.py ../agreid_v2_combined.py' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
./reviews/codex_1.md:5471:../maxsim_probe.py:60:                           filter_by_view)
./reviews/codex_1.md:5473:../maxsim_probe.py:278:    q_aerial = filter_by_view(dataset.query, 'Aerial')
./reviews/codex_1.md:5474:../maxsim_probe.py:279:    q_ground = filter_by_view(dataset.query, 'Ground')
./reviews/codex_1.md:5475:../maxsim_probe.py:280:    g_aerial = filter_by_view(dataset.gallery, 'Aerial')
./reviews/codex_1.md:5476:../maxsim_probe.py:281:    g_ground = filter_by_view(dataset.gallery, 'Ground')
./reviews/codex_1.md:5478:../agreid_dataset.py:9:RandomIdentitySampler / filter_by_view / eval_market / run_cross_view_eval).
./reviews/codex_1.md:5479:../agreid_dataset.py:63:CARGO, and the downstream `filter_by_view` does the A->G / G->A directions:
./reviews/codex_1.md:5480:../agreid_dataset.py:283:def filter_by_view(samples, view):
./reviews/codex_1.md:5484:../agreid_dataset.py:310:    print("A->G  q_aerial:", len(filter_by_view(ds.query, 'Aerial')),
./reviews/codex_1.md:5485:../agreid_dataset.py:311:          " g_ground:", len(filter_by_view(ds.gallery, 'Ground')))
./reviews/codex_1.md:5486:../agreid_dataset.py:312:    print("G->A  q_ground:", len(filter_by_view(ds.query, 'Ground')),
./reviews/codex_1.md:5487:../agreid_dataset.py:313:          " g_aerial:", len(filter_by_view(ds.gallery, 'Aerial')))
./reviews/codex_1.md:5489:../afd_train.py:151:                           RandomIdentitySampler, filter_by_view)
./reviews/codex_1.md:5490:../afd_train.py:922:    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_1.md:5491:../afd_train.py:1043:    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_1.md:5492:../afd_train.py:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_1.md:5650:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_1.md:5821:    60	                           filter_by_view)
./reviews/codex_1.md:5927:   166	def eval_from_distmat(distmat, q_pids, q_camids, g_pids, g_camids, max_rank=50):
./reviews/codex_1.md:5984:`eval_from_distmat` 是标准 Market-style：只用 pid/camid 计算指标和同 pid 同 cam 的 junk remove，不参与排序权重。融合矩阵传入的是距离矩阵，`dm_fuse=2-2*(0.25 rec + 0.75 full)`，排序方向正确。
./reviews/codex_1.md:6003:/bin/zsh -lc 'rg -n "def filter_by_view|class AGReID|query|gallery|camid|pid|view|exp1|exp4" ../agreid_dataset.py' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
./reviews/codex_1.md:6005:9:RandomIdentitySampler / filter_by_view / eval_market / run_cross_view_eval).
./reviews/codex_1.md:6026:63:CARGO, and the downstream `filter_by_view` does the A->G / G->A directions:
./reviews/codex_1.md:6100:283:def filter_by_view(samples, view):
./reviews/codex_1.md:6105:310:    print("A->G  q_aerial:", len(filter_by_view(ds.query, 'Aerial')),
./reviews/codex_1.md:6106:311:          " g_ground:", len(filter_by_view(ds.gallery, 'Ground')))
./reviews/codex_1.md:6107:312:    print("G->A  q_ground:", len(filter_by_view(ds.query, 'Ground')),
./reviews/codex_1.md:6108:313:          " g_aerial:", len(filter_by_view(ds.gallery, 'Aerial')))
./reviews/codex_1.md:6146:   283	def filter_by_view(samples, view):
./reviews/codex_1.md:6173:   310	    print("A->G  q_aerial:", len(filter_by_view(ds.query, 'Aerial')),
./reviews/codex_1.md:6174:   311	          " g_ground:", len(filter_by_view(ds.gallery, 'Ground')))
./reviews/codex_1.md:6175:   312	    print("G->A  q_ground:", len(filter_by_view(ds.query, 'Ground')),
./reviews/codex_1.md:6176:   313	          " g_aerial:", len(filter_by_view(ds.gallery, 'Aerial')))
./reviews/codex_9.md:431:1095:        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_9.md:432:1096:        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_9.md:433:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_9.md:640:1750:        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_9.md:6235:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_9.md:6240:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_9.md:6242:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_9.md:6292:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_9.md:6293:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_9.md:6294:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_9.md:9087:./reviews/codex_6.md:5953:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_9.md:9291:./reviews/codex_6.md:6865:/bin/zsh -lc "rg -n \"def set_seed|class RandomIdentitySampler|torch.initial_seed|worker_init_fn|manual_seed|random.seed|np.random.seed|torch.backends|cudnn|DataLoader|build_eval_loader\" afd_reid code -g '*.py'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
./reviews/codex_9.md:9301:./reviews/codex_6.md:6908:/bin/zsh -lc "rg -n \"def set_seed|class RandomIdentitySampler|def __iter__|np.random|random\\.seed|torch.manual_seed|cudnn|worker_init_fn|build_eval_loader|DataLoader\" /Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/afd_train.py /Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/cargo_dataset.py /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
./reviews/codex_9.md:9454:./code/afd_train.py:922:    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_9.md:9462:./code/afd_train.py:1043:    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_9.md:9930:./reviews/codex_2.md:1016:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_9.md:13166:./reviews/codex_5.md:6634:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_9.md:13174:./reviews/codex_5.md:6755:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_4.md:1141:   151	                           RandomIdentitySampler, filter_by_view)
./reviews/codex_4.md:1912:   922	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_4.md:2010:  1020	        from maxsim_probe import eval_from_distmat
./reviews/codex_4.md:2011:  1021	        rmap, rr1, _ = eval_from_distmat(-hyb, qp, qc, gp, gc)
./reviews/codex_4.md:2028:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_4.md:2033:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_4.md:2035:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_4.md:2085:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_4.md:2086:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_4.md:2087:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_4.md:2740:  1750	        # official exp1(A->G) + exp4(G->A); filter_by_view recovers each direction
./reviews/codex_4.md:8041:1097:        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)
./reviews/codex_4.md:9390:  1038	    feature extraction and the eval_market / eval_from_distmat ranking, so the
./reviews/codex_4.md:9395:  1043	    from cargo_dataset import filter_by_view as _fbv
./reviews/codex_4.md:9397:  1045	    from maxsim_probe import eval_from_distmat
./reviews/codex_4.md:9447:  1095	        fmap, fr1, _ = eval_from_distmat(dm_full, qp, qc, gp, gc)
./reviews/codex_4.md:9448:  1096	        rmap, rr1, _ = eval_from_distmat(dm_rec, qp, qc, gp, gc)
./reviews/codex_4.md:9449:  1097	        zmap, zr1, _ = eval_from_distmat(dm_fuse, qp, qc, gp, gc)

 succeeded in 0ms:
./reviews/codex_1.md
./reviews/codex_10.md
./reviews/codex_2.md
./reviews/codex_3.md
./reviews/codex_4.md
./reviews/codex_5.md
./reviews/codex_6.md
./reviews/codex_7.md
./reviews/codex_8.md
./reviews/codex_9.md

codex
有一点审查边界：`code/afd_train.py` 从外部 `afd_reid`/`cargo_dataset`/`maxsim_probe` 导入 eval 和数据 helper，bundle 内本身不包含这些源文件，只能从现有日志与当前代码片段判断 AIRL 主路径；我不会把外部 helper 视作已逐行复核。
exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1,170p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
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

exec
/bin/zsh -lc "nl -ba code/afd_model.py | sed -n '1,80p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
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

exec
/bin/zsh -lc "find /Users/abdslm/Desktop/SOLIDER-REID/experiments -maxdepth 2 -type f \\( -name 'afd_train.py' -o -name 'cargo_dataset.py' -o -name 'maxsim_probe.py' -o -name 'agreid_v2_combined.py' \\) -print" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py
/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/maxsim_probe.py
/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/agreid_v2_combined.py
/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/afd_train.py
/Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/cargo_dataset.py

codex
还有一个实际代码路径风险：这份 bundle 的 `code/afd_train.py` 是拷贝版，顶部仍按原仓库相对路径插入 `../afd_reid`。在 bundle 内该目录不存在；真实训练日志很可能来自原始 `experiments/cargo_cvpb/afd_train.py` + `experiments/afd_reid/afd_model.py` 组合，而不是直接从 bundle 运行。我要对比一下拷贝与原始文件是否一致。
exec
/bin/zsh -lc "cmp -s code/afd_train.py /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py; printf 'afd_train_cmp=%s\\n' "'$?' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
afd_train_cmp=0

exec
/bin/zsh -lc "cmp -s code/afd_model.py /Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/afd_model.py; printf 'afd_model_cmp=%s\\n' "'$?' in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
afd_model_cmp=0

exec
/bin/zsh -lc "diff -u --label bundle/code/afd_model.py --label original/afd_reid/afd_model.py code/afd_model.py /Users/abdslm/Desktop/SOLIDER-REID/experiments/afd_reid/afd_model.py | sed -n '1,220p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
exec
/bin/zsh -lc "diff -u --label bundle/code/afd_train.py --label original/cargo_cvpb/afd_train.py code/afd_train.py /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/afd_train.py | sed -n '1,220p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
 succeeded in 0ms:
exec
/bin/zsh -lc "test -d afd_reid; printf 'bundle_afd_reid_dir=%s\\n' "'$?; ls -la | sed -n '"'1,80p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
bundle_afd_reid_dir=1
total 16
drwxr-xr-x@  6 abdslm  staff   192 Jun 24 06:50 .
drwxr-xr-x@ 60 abdslm  staff  1920 Jun 24 06:47 ..
drwxr-xr-x@  4 abdslm  staff   128 Jun 24 06:43 code
-rw-r--r--@  1 abdslm  staff  5421 Jun 24 06:47 launch_10codex.sh
drwxr-xr-x@  7 abdslm  staff   224 Jun 24 06:43 logs
drwxr-xr-x@ 12 abdslm  staff   384 Jun 24 06:51 reviews

web search: 
web search: arXiv MGN Multiple Granularity Network person re-identification multi branch feature fusion
web search: 
web search: Multiple Granularity Network person re-identification multi branch feature fusion arxiv
web search: 
web search: site:arxiv.org/abs MGN Multiple Granularity Network person re-identification
codex
我准备收口了。当前主要结论是：代码路径能产出日志里的 full/rec/FUSE，FUSE 算法没有明显方向/距离错误；但“净增益”和“机制归因”证据不足，尤其缺少 no-degradation 双头/独立 baseline ensemble 对照。
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

exec
/bin/zsh -lc "nl -ba logs/agreidv2_airl_4090.log | sed -n '966,980p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   966	Epoch[60] Iter[650/786] Loss: 2.102 CE: 1.046 Tri: 0.004 CE_rec: 1.029 AIRL_rec: 0.0469 Acc: 0.997 LR: 3.45e-08
   967	Epoch[60] Iter[700/786] Loss: 2.100 CE: 1.045 Tri: 0.004 CE_rec: 1.028 AIRL_rec: 0.0465 Acc: 0.997 LR: 3.45e-08
   968	Epoch[60] done in 114.0s  Loss=2.097 Acc=0.997 AIRL-ISO[lam_eff=0.500 ce_rec=1.027 consistency=0.0458 deg_scale_mean=0.624 n_ground=28813]
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
/bin/zsh -lc "nl -ba logs/agreidv2_baseline_4090.log | sed -n '945,955p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   945	Epoch[60] Iter[650/786] Loss: 1.049 CE: 1.045 Tri: 0.003 Acc: 0.997 LR: 3.45e-08
   946	Epoch[60] Iter[700/786] Loss: 1.048 CE: 1.045 Tri: 0.003 Acc: 0.997 LR: 3.45e-08
   947	Epoch[60] done in 91.9s  Loss=1.047 Acc=0.997
   948	  ---- A<->G cross-view eval @ epoch 60 ----
   949	    [A->G] mAP=80.73  R1=87.35  R5=93.00  mINP=56.91
   950	    [G->A] mAP=81.42  R1=89.07  R5=93.21  mINP=48.32
   951	    [mean] mAP=81.08  R1=88.21
   952	    * new best mean mAP=81.08 (epoch 60) saved
   953	======================================================================
   954	Training finished. Best mean A<->G mAP=81.08 @ epoch 60
   955	Checkpoints in /home/afr/SOLIDER-REID/log/cargo/baseline_agreidv2_4090

exec
/bin/zsh -lc "nl -ba logs/agreidv2_baseline.log | sed -n '946,956p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   946	Epoch[60] Iter[650/786] Loss: 1.050 CE: 1.046 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
   947	Epoch[60] Iter[700/786] Loss: 1.049 CE: 1.045 Tri: 0.004 Acc: 0.997 LR: 3.45e-08
   948	Epoch[60] done in 190.7s  Loss=1.047 Acc=0.998
   949	  ---- A<->G cross-view eval @ epoch 60 ----
   950	    [A->G] mAP=79.72  R1=86.42  R5=92.28  mINP=56.04
   951	    [G->A] mAP=80.04  R1=87.80  R5=92.16  mINP=46.62
   952	    [mean] mAP=79.88  R1=87.11
   953	    * new best mean mAP=79.88 (epoch 60) saved
   954	======================================================================
   955	Training finished. Best mean A<->G mAP=79.88 @ epoch 60
   956	Checkpoints in /root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_baseline

exec
/bin/zsh -lc "nl -ba logs/agreidv2_airl_iso.log | sed -n '654,661p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
   654	    [A->G] mAP=76.09  R1=83.87  R5=90.70  mINP=49.93
   655	    [G->A] mAP=76.69  R1=84.82  R5=90.50  mINP=42.19
   656	    [mean] mAP=76.39  R1=84.34
   657	  ---- AIRL-ISO dual-branch (fuse_w=0.25) @ epoch 40 ----
   658	    [A->G] full mAP=76.09 R1=83.87 | rec mAP=75.40 R1=83.23 | FUSE mAP=77.58 R1=85.44
   659	    [G->A] full mAP=76.69 R1=84.82 | rec mAP=75.07 R1=83.88 | FUSE mAP=77.65 R1=85.37
   660	    [mean] full=76.39 rec=75.23 FUSE=77.62  <- model-selection uses FUSE
   661	    * new best mean mAP=77.62 (epoch 40) saved

exec
/bin/zsh -lc "nl -ba logs/agreidv2_airl_4090.log | sed -n '1,26p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
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

exec
/bin/zsh -lc "nl -ba logs/agreidv2_airl_iso.log | sed -n '1,26p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
     1	======================================================================
     2	CVPB-ReID training (afd_reid baseline + OVP-Mem / OVLI)
     3	  backbone=swin_small (SOLIDER, in_planes=768) pretrain=/root/work/SOLIDER-REID/pretrained/swin_small.pth semantic_weight=0.2
     4	  use_afd=False  ovp=False (lambda=0.5 tau=0.05 mom=0.2)
     5	  ovli=False (lambda=0.5 tau=0.05 alpha=0.5 dim=256 grid=(8, 4) warmup=10 rerank=False pool=mean topk=8 thresh=0.0 cand=oppview match=maxsim align=free setpool=mean setpool_residual=True)
     6	  acvp=False (gamma=0.5 wmin=0.3 eta=0.05 margin=0.0 warmup=10) [detached neg-relaxation on OVLI; off => OVLI byte-identical]
     7	  airl=False (lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolution-degradation consistency; NO learnable params; train-only; off => baseline byte-identical]
     8	  airl_dualbranch=False (fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [resolvability branch: 2nd BNNeck head f_rec (own ID-CE + AIRL consistency) + clean f_full, soft-fused cos=w*cos_rec+(1-w)*cos_full; 1 forward 2 features; off => baseline byte-identical]
     9	  airl_dualbranch_iso=True (iso_stage=3 trunk_recce=False fuse_w=0.25 lambda=0.5 min_scale=0.25 consistency=kl tau=4.0 blur=False warmup=5) [GRADIENT-ISOLATED rescue: f_rec = BNNeck over an independent late Swin stage; the degradation-CONSISTENCY grad NEVER reaches the shared trunk (detached degraded pass). trunk_recce=1 (FIX) also reflows the CLEAN f_rec ID-CE into the trunk (extra identity supervision -> strengthens f_full); trunk_recce=0 = original full-isolation (clean ID-CE also detached). f_rec learns the recover pole; same soft-fusion eval; off => baseline byte-identical]
    10	  bs=64 (P=16 K=4) lr=0.00035 epochs=60 warmup=10 amp=True
    11	  out_dir=/root/work/SOLIDER-REID/log/cargo/cvpb_agreidv2_airl_iso
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
    26	Epoch[1] Iter[50/786] Loss: 44.803 CE: 6.693 Tri: 31.417 CE_rec: 6.693 AIRL_rec: 0.0002 Acc: 0.002 LR: 3.50e-07

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1470,1505p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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

exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1530,1556p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
exec
/bin/zsh -lc "nl -ba code/afd_train.py | sed -n '1071,1097p'" in /Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle
 succeeded in 0ms:
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

 succeeded in 0ms:
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

web search: 
web search: https://arxiv.org/abs/1804.01438
web search: 
web search: AG-ReID.v2 aerial ground person re-identification arxiv
web search: 
web search: cross resolution person re-identification low resolution high resolution degradation consistency arxiv
web search: 
web search: person re-identification ensemble score fusion multiple models arxiv
web search: 
web search: https://arxiv.org/abs/2401.02634
web search: 
web search: https://arxiv.org/abs/1901.05798
codex
**1. Verdict**

needs-attention。代码主路径基本能解释日志里的 full/rec/FUSE 数字，但主控 3 条结论里至少第 (1) 不能按当前 logs 通过；本轮 lens 也不能 approve 机制归因。

**2. Findings**

- High | [logs/agreidv2_airl_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_4090.log:976) + [logs/agreidv2_baseline_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_baseline_4090.log:951)  
  AG-ReID.v2 lab-4090 final：AIRL FUSE=80.98，baseline=81.08，net=-0.10，不是正增益。把这组日志说成 “net gain≈+1.5/+2.37” 不成立。

- High | [code/afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1071)  
  FUSE 是固定 `w=0.25` 的纯 score blend：`0.25*cos_rec + 0.75*cos_full`。当前 logs 没有 no-AIRL 双头、独立 baseline ensemble、或 w-sweep 对照；所以 `FUSE-full` 只能证明两个头排序有互补，不能证明增益来自 degradation-consistency。

- Medium | [logs/agreidv2_airl_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_4090.log:9) + [code/afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1541)  
  AIRL 两份 AG-ReID.v2 日志实际都是 `trunk_recce=False`，而代码注释把 `trunk_recce=0` 定义为 original full-isolation / ablation only。若把这组作为 headline，需要明说它不是 `trunk_recce=1` fix 版本；f_full 变弱也不能只归为 seed 噪声。

- Medium | [logs/agreidv2_baseline_4090_s2.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_baseline_4090_s2.log:887)  
  baseline seed2 日志不完整，停在 epoch57 训练中，没有 epoch60 eval / `Training finished`。不能纳入 final mean 或 final seed-noise 估计。

- Low | [code/afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:148)  
  bundle 不是自包含运行包：脚本仍按原仓库相对路径插入 `../afd_reid`。我对比过，bundle 的 `code/afd_train.py` 与原始 `experiments/cargo_cvpb/afd_train.py` 一致，`code/afd_model.py` 与原始 `experiments/afd_reid/afd_model.py` 一致；但单独在 bundle 内直接跑会依赖外部路径。

**3. 对 3 条待验结论的判断**

1. 反对。AG-ReID.v2 4090 final：full=79.90，rec=78.54，FUSE=80.98，`FUSE-full=+1.08`；baseline 4090 final=81.08，所以 net=-0.10。3090 AIRL log 只有到 ep40 eval，FUSE=77.62，不能当 final。bundle 内没有 CARGO log，无法验证“双数据集一致”。

2. 部分同意，但证据不足。AG-ReID.v2 4090 full=79.90 vs baseline=81.08，确实约 -1.18；这符合“背运/seed noise”的幅度说法，但 seed2 final 缺失，不能证明只是噪声。CARGO 的 +0.6 / net +2.37 不在本轮 logs 里，无法核。

3. 基本同意“AG-ReID.v2 没有稳定方向特化”，但不能证明 “CARGO-specific”。AG final rec 比 full 两个方向都差：A->G 78.70 vs 79.47，G->A 78.38 vs 80.33；早期 ep20 rec 在 A->G 强一些，但后期消失。CARGO-specific 需要 CARGO log 支撑。

**4. Lens 结论**

soft-fuse 不是实现 bug，但机制归因目前不够。`w=0.25` 在 AG-ReID.v2 上确实让 FUSE 高于 full：4090 各 eval 的 `FUSE-full` 约 +0.91 到 +1.69，final +1.08；这是真实排序互补信号。但它也完全可能是“两个相关 head 的普通 ensemble 收益”。

文献上，多分支/多特征融合在 ReID 已是常见路线：MGN 明确是 global + local 多分支；EnsembleNet 直接把单网络分成多个端到端分支做 ensemble feature；OSNet 也用多流多尺度动态融合。低分辨/跨分辨 ReID 也已有 resolution degradation / resolution-invariant / feature distillation 先例。参考：[MGN](https://arxiv.org/abs/1804.01438), [Ensemble Feature for ReID](https://arxiv.org/abs/1901.05798), [OSNet](https://arxiv.org/abs/1905.00953), [Cross-Resolution DAL](https://arxiv.org/abs/1810.10221), [RFD](https://arxiv.org/abs/2109.07871), [AG-ReID.v2](https://arxiv.org/abs/2401.02634)。

要把 AIRL 讲成机制贡献，最低还缺：`dualbranch_iso no-degrade`、两独立 baseline score average、固定预注册 w-sweep、head score/rank correlation、以及 per-direction error complementarity。当前只能说 “FUSE 有收益”，不能说 “degradation-consistency 造成了非平凡互补”。
tokens used
205,018
**1. Verdict**

needs-attention。代码主路径基本能解释日志里的 full/rec/FUSE 数字，但主控 3 条结论里至少第 (1) 不能按当前 logs 通过；本轮 lens 也不能 approve 机制归因。

**2. Findings**

- High | [logs/agreidv2_airl_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_4090.log:976) + [logs/agreidv2_baseline_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_baseline_4090.log:951)  
  AG-ReID.v2 lab-4090 final：AIRL FUSE=80.98，baseline=81.08，net=-0.10，不是正增益。把这组日志说成 “net gain≈+1.5/+2.37” 不成立。

- High | [code/afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1071)  
  FUSE 是固定 `w=0.25` 的纯 score blend：`0.25*cos_rec + 0.75*cos_full`。当前 logs 没有 no-AIRL 双头、独立 baseline ensemble、或 w-sweep 对照；所以 `FUSE-full` 只能证明两个头排序有互补，不能证明增益来自 degradation-consistency。

- Medium | [logs/agreidv2_airl_4090.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_airl_4090.log:9) + [code/afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:1541)  
  AIRL 两份 AG-ReID.v2 日志实际都是 `trunk_recce=False`，而代码注释把 `trunk_recce=0` 定义为 original full-isolation / ablation only。若把这组作为 headline，需要明说它不是 `trunk_recce=1` fix 版本；f_full 变弱也不能只归为 seed 噪声。

- Medium | [logs/agreidv2_baseline_4090_s2.log](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/logs/agreidv2_baseline_4090_s2.log:887)  
  baseline seed2 日志不完整，停在 epoch57 训练中，没有 epoch60 eval / `Training finished`。不能纳入 final mean 或 final seed-noise 估计。

- Low | [code/afd_train.py](/Users/abdslm/Desktop/SOLIDER-REID/experiments/cargo_cvpb/airl_codex_bundle/code/afd_train.py:148)  
  bundle 不是自包含运行包：脚本仍按原仓库相对路径插入 `../afd_reid`。我对比过，bundle 的 `code/afd_train.py` 与原始 `experiments/cargo_cvpb/afd_train.py` 一致，`code/afd_model.py` 与原始 `experiments/afd_reid/afd_model.py` 一致；但单独在 bundle 内直接跑会依赖外部路径。

**3. 对 3 条待验结论的判断**

1. 反对。AG-ReID.v2 4090 final：full=79.90，rec=78.54，FUSE=80.98，`FUSE-full=+1.08`；baseline 4090 final=81.08，所以 net=-0.10。3090 AIRL log 只有到 ep40 eval，FUSE=77.62，不能当 final。bundle 内没有 CARGO log，无法验证“双数据集一致”。

2. 部分同意，但证据不足。AG-ReID.v2 4090 full=79.90 vs baseline=81.08，确实约 -1.18；这符合“背运/seed noise”的幅度说法，但 seed2 final 缺失，不能证明只是噪声。CARGO 的 +0.6 / net +2.37 不在本轮 logs 里，无法核。

3. 基本同意“AG-ReID.v2 没有稳定方向特化”，但不能证明 “CARGO-specific”。AG final rec 比 full 两个方向都差：A->G 78.70 vs 79.47，G->A 78.38 vs 80.33；早期 ep20 rec 在 A->G 强一些，但后期消失。CARGO-specific 需要 CARGO log 支撑。

**4. Lens 结论**

soft-fuse 不是实现 bug，但机制归因目前不够。`w=0.25` 在 AG-ReID.v2 上确实让 FUSE 高于 full：4090 各 eval 的 `FUSE-full` 约 +0.91 到 +1.69，final +1.08；这是真实排序互补信号。但它也完全可能是“两个相关 head 的普通 ensemble 收益”。

文献上，多分支/多特征融合在 ReID 已是常见路线：MGN 明确是 global + local 多分支；EnsembleNet 直接把单网络分成多个端到端分支做 ensemble feature；OSNet 也用多流多尺度动态融合。低分辨/跨分辨 ReID 也已有 resolution degradation / resolution-invariant / feature distillation 先例。参考：[MGN](https://arxiv.org/abs/1804.01438), [Ensemble Feature for ReID](https://arxiv.org/abs/1901.05798), [OSNet](https://arxiv.org/abs/1905.00953), [Cross-Resolution DAL](https://arxiv.org/abs/1810.10221), [RFD](https://arxiv.org/abs/2109.07871), [AG-ReID.v2](https://arxiv.org/abs/2401.02634)。

要把 AIRL 讲成机制贡献，最低还缺：`dualbranch_iso no-degrade`、两独立 baseline score average、固定预注册 w-sweep、head score/rank correlation、以及 per-direction error complementarity。当前只能说 “FUSE 有收益”，不能说 “degradation-consistency 造成了非平凡互补”。
