# Codex Review — exp290_target_s_op_s42

**Verdict**: approve (with 1 pre-launch logistics check outside the diff)
**Date**: 2026-04-22 11:58 CST
**Review round**: 1
**Reviewer**: independent-second-pass (codex binary unavailable; general-purpose agent acting as
independent reviewer — no access to `claude_review.md` conclusions while forming findings)

## Summary
The diff is a 2-line guarded swap plus a `getattr`-backed flag. Code is clean, backward-compat
airtight, and train/test/flip/OA-SD paths all remain symmetric. I found **one non-diff concern**
worth flagging (lab4090 OP pose_data availability, per user memory), and two **minor wording /
sanity-log** nits. No new code defects beyond what the first reviewer surfaced.

## Findings

### Low: lab4090 does NOT yet have OP pose_data (pre-launch logistics, NOT in the diff)
Location: design.md:89 (`机器: lab4090`) vs MEMORY.md snapshot
Description: `~/.claude/.../project_lab4090_pose_data.md` explicitly states
"其他数据集(Market-1501, Occluded-ReID, Occluded-PoseTrack-ReID)的 pose_data 尚未同步到
lab4090,仅 Occluded-Duke 就绪". The design doc pins the experiment to lab4090 (idle). If the
OP pose_data has not been rsynced since 2026-04-20, training will fail at first `__getitem__`
when `PoseImageDataset.__init__` tries to load `data/occluded_posetrack_reid/pose_data/train/index.json`.
Fix: before launch, either (a) verify via `ssh lab4090 "ls -la /mnt1/afrdata/.../occluded_posetrack_reid/pose_data/train/index.json"`
and confirm `target_person_idx` present in the index (run `python -c "import json;
print(list(json.load(open(...)).values())[0].keys())"`), or (b) move the run to lab3090 / srvA
where OP pose_data already lives. Not a code bug — a deployment prerequisite.

### Low: _prepare_pose recomputes target_heatmaps unconditionally (marginal, no action)
Location: pose_backbone_model.py:908-938
Description: The 4-tuple return always materializes `target_heatmaps` and `diff_heatmaps` tensors,
even when `use_target_heatmap=False`. This is pre-existing at HEAD (not introduced by exp290)
but it means the flag-off path already pays the (B,17,H,W) tensor allocation cost. The exp290
change does NOT worsen this. Noting for completeness — reducing this would require a separate
refactor (e.g., `_prepare_pose` returning `target=None` when not needed). No action for exp290.

### Low: Comment at line 467-471 names VCSR/PPA/STR/FSDC, all of which are inactive for this YAML
Location: pose_backbone_model.py:467-471
Description: The comment "all downstream pose-aware modules (PSG/LGPA/VCSR/PPA/STR/FSDC/etc.)"
is technically accurate for the *class* in general, but for `prcv_best_small.yml` the actually
active modules are PSG (2-stage) + LGPA + GCN. Only PSG and LGPA consume `scene_heatmaps` in
this config; GCN uses `pose_dict['keypoints'][:, 0]` directly. The comment correctly covers the
general case — no change needed, but be aware the exp290 training run will exercise only the
PSG + LGPA swap paths.

### Low (informational): PSG gate is target-agnostic by design — swap is safe
Location: model/modules/pose_spatial_gate.py:53-84
Description: PSG's encoder is a fixed pose_channels=17 input → hidden → feat_channels conv.
It interpolates `scene_heatmaps` to (H,W) and applies `sigmoid → conv → residual gate`. There
is NO assumption about "multi-person max" semantics — the input is just a (B, 17, H, W) tensor
whose channels represent 17 keypoint activations. After the swap, the same tensor shape with
target-only activations still satisfies the module contract. Zero-init of the last conv guarantees
the network starts from identity gate regardless of scene vs target.

### Low (informational): LGPA docstring already labels arg `target_heatmaps` (naming bug is
pre-existing, exp290 aligns semantics)
Location: model/modules/clip_part_head.py:133, 222, 226
Description: `_compute_pose_bias(heatmaps, ...)` docstring says "heatmaps: (B, 17, H, W) — TARGET
person heatmaps" at line 133, and `forward(self, feat_map, target_heatmaps, ...)` uses the
parameter name `target_heatmaps` at line 222, 226. Pre-exp290, the caller was passing
`scene_heatmaps` (max-merged scene) into a parameter explicitly documented as TARGET. The
exp290 swap corrects this **runtime** semantic mismatch — the module was always designed to
receive target-only heatmaps. First reviewer also noted this; I confirm.

## Gradient / data flow analysis

1. **Graph integrity**: `target_heatmaps = heatmaps[:, 0] * person_mask[:, 0].view(-1,1,1,1)` at
   line 929 does not break gradient flow because pose tensors (`heatmaps`, `scores`, `person_mask`)
   are loaded from .npz files via `torch.from_numpy` in `pose_dataset.py:286-298`. None of these
   have `requires_grad=True`. The swap at line 472-473 reassigns a Python reference, not a
   gradient-breaking op. The downstream `sigmoid → conv → residual gate` autograd graph
   terminates at PSG's conv parameters (trainable) and the (non-grad) heatmap tensor. **Safe.**

2. **Downstream scene_heatmaps consumers** (all post-swap, line-inspected):
   - PSG `_run_backbone_with_psg` line 482 → `pose_spatial_gate.py:53` (no scene assumption)
   - PAA `paa_modules_dict[key]` line 449 (inactive in this YAML; adapter.py:52-70 is shape-only)
   - VCSR line 500, STR line 514, FSDC line 672 (all inactive in this YAML)
   - LGPA `clip_part_head` line 586 (active; `_compute_pose_bias` max-pools heatmap channels per
     part group — works identically on target-only)
   - PPA `part_assignment_head.forward` line 608 (inactive; CE supervision `gt_labels` derived
     from argmax of heatmap channels — works identically on target-only)
   - Test-time mirrors at lines 770-867 (LGPA branch at 779 is the active one)

   **All consumers are semantics-agnostic w.r.t. scene-vs-target** — they only depend on (B, 17,
   H, W) shape + non-negative activations. No consumer calls `merge_person_heatmaps` or reaches
   into `pose_dict['person_mask']` internally (verified via grep on `model/modules/`).

3. **Pose dropout ordering** (pose_backbone_model.py:475-479): the swap precedes the dropout.
   SPD zeros out per-sample with probability p — applied to target heatmaps, this is an augmentation
   against the target signal, which is semantically valid. For `prcv_best_small.yml`,
   `POSE_DROPOUT_P` is not set → defaults to 0.0 → dropout dormant. **No asymmetric drop concern.**

## Boundary / edge cases (independent verification)

1. **`target_heatmaps.shape != scene_heatmaps.shape` under hypothetical refactor**: both come
   from the same `_prepare_pose` helper, both are `(B, 17, H, W)` float32 on same device. As
   long as that helper is the only producer, shapes match by construction. A future refactor
   that changes the return shape would break the swap AND every existing consumer — the swap
   adds no new fragility.

2. **`person_mask[:, 0] == 0` (target not detected)**: `target_heatmaps` becomes zero tensor.
   PSG's `sigmoid(0) = 0.5` constant fed to zero-init conv → near-zero gate → passthrough
   `x * 1.0`. LGPA's `_compute_pose_bias` line 147-156 produces `body_max=0`, part_activations
   all zero except background `(1.0 - 0).clamp(min=0)` = 1.0 → heavy attention toward background
   class. Graceful degradation, no NaN/Inf. Dataset-level expectation: OP guarantees annotated
   target per sample (defining property of the benchmark), so this case should be ~0% of batch
   mass. Worth a one-line sanity log at epoch 1, but not blocking.

3. **`pose_dict=None`**: both scene and target remain None (lines 461-462). Swap guard
   `target_heatmaps is not None` short-circuits. **Safe.**

4. **Single-person samples (OD/Market baseline)**: `scene = max(person_0, person_0*0, ...) =
   heatmaps[:, 0]`; `target = heatmaps[:, 0] * 1 = heatmaps[:, 0]`. Swap is a **strict numerical
   no-op**. This mathematically guarantees zero regression on OD/Market should this flag ever
   be flipped there — the first reviewer's finding confirmed.

5. **Flip-test symmetry** (utils/flip_test.py and scripts/eval_fliptest_maxsim.py): both flip
   `pose_dict['heatmaps']` along W and swap COCO L/R channel pairs BEFORE calling `model.forward`.
   Crucially, neither flip helper touches the **person dimension** — person_0 stays at index 0
   post-flip. Therefore `heatmaps[:, 0]` in `_prepare_pose` still extracts the flipped target,
   and L/R channel swap is applied consistently whether we feed scene or target downstream.
   **Train-test + flip-test symmetry preserved.**

6. **OA-SD teacher**: `processor.py:478` does `ema_teacher = copy.deepcopy(base_model)` —
   `self.use_target_heatmap` is a plain Python bool, deepcopied. Teacher calls its own
   `forward()` at line 796 / 897, which applies the same swap guard. Teacher pose comes from
   `pose_dict.get('teacher_pose', pose_dict)` (line 795), built via
   `copy.deepcopy(persons)` in dataset.py:175 which preserves the target-at-0 reordering done
   by `_load_persons`. Teacher-student distillation therefore compares target-only features on
   both sides. **Symmetric.**

7. **Config precedence via yacs CLI**: `getattr(cfg.MODEL, 'POSE_USE_TARGET_HEATMAP', False)`
   combined with `_C.MODEL.POSE_USE_TARGET_HEATMAP = False` in defaults means:
   - YAML override `MODEL: { POSE_USE_TARGET_HEATMAP: True }` works.
   - CLI `python train.py --config_file ... MODEL.POSE_USE_TARGET_HEATMAP True` works
     (yacs merges list-of-strings into _C).
   - Old YAMLs without the key → getattr returns False → existing behavior.
   - Loading old checkpoints with no `use_target_heatmap` in state_dict → attribute purely from
     cfg, no state_dict coupling.
   **Rock-solid.**

8. **Target annotation availability on eval set**: `datasets/pose_dataset.py:352-362` unconditionally
   reorders persons by `entry.get('target_person_idx', 0)` — same code path for train/query/gallery
   since `_load_persons` is called from `__getitem__` regardless of `is_train`. `scripts/compute_target_assignment.py:176`
   iterates over `['train', 'query', 'gallery']` by default, populating `target_person_idx` for
   ALL splits. Eval-time target is annotation-driven, not test-time manual. **Train/test symmetric.**

9. **No config key collision**: grep for `POSE_USE_` in config/ returns only the new
   `POSE_USE_TARGET_HEATMAP`. No conflict with any existing key (checked `POSE_USE_` prefix and
   `USE_TARGET` substring across defaults.py).

## Residual concerns & recommendations

Concur with first reviewer's Medium finding on zero-target graceful degradation (boundary #2).
One additional pre-launch item the first reviewer did not flag:

- **lab4090 OP pose_data**: per user memory 2026-04-20, OP pose_data is not yet on lab4090.
  Verify by running `ssh lab4090 "ls /mnt1/afrdata/occluded_posetrack_reid/pose_data/train/index.json
  && python -c 'import json,sys; d=json.load(open(sys.argv[1])); k=list(d.keys())[0];
  print(k, list(d[k].keys()))' /mnt1/afrdata/occluded_posetrack_reid/pose_data/train/index.json"`.
  If missing, either rsync from srvA/srvB/lab3090 OR switch the run to a machine that has OP
  pose_data (e.g., lab3090 where exp266b_3090 just finished). This is outside the diff but
  blocks the launch cited in the design doc.

All other first-reviewer findings I concur with. No new code defects found.

## Conclusion

**verdict**: approve
codex 审查通过

The diff itself is production-ready. Backward compatibility is byte-identical under flag=False
(verified by inspection of the forward path). Train/test/flip/OA-SD symmetry all hold. The
only pre-launch item is a deployment check (OP pose_data on lab4090), not a code defect. The
design doc's "如果持平 exp265 (无改善)" contingency is an honest null hypothesis, and the
flag-gated minimal-invasive nature of the change means a null result carries zero cleanup cost
for other branches.

Cleared for training launch once OP pose_data presence on the chosen machine is confirmed.
