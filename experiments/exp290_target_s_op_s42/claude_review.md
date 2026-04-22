# Claude Broad Review — exp290_target_s_op_s42

**Review round**: 1
**Date**: 2026-04-22 12:05 CST
**Reviewer**: claude-opus-4-7 (general-purpose agent)

## Scope

Reviewed files (LOC):

- `config/defaults.py` (448 lines; diff: +7 lines block around line 104-110)
- `model/pose_backbone_model.py` (938 lines; diff: +17 lines total — `__init__` at 133-139, `forward()` at 466-473)
- `experiments/exp290_target_s_op_s42/design.md` (134 lines)

Cross-referenced for data flow / flag semantics:

- `model/modules/pose_utils.py` — `merge_person_heatmaps` (unchanged)
- `model/modules/pose_spatial_gate.py` — PSG gate math (zero-init last conv)
- `model/modules/clip_part_head.py` — LGPA (docstring says `target_heatmaps`, aligns with intent)
- `model/modules/skeleton_gcn.py` — GCN uses `pose_dict['keypoints'][:, 0]` (already target-only)
- `datasets/pose_dataset.py` — `_load_persons` reorders target to index 0 via `target_person_idx`
- `configs/occluded_posetrack/prcv_best_small.yml` — active OP Small config
- `processor/processor.py` — OA-SD EMA teacher branch (`copy.deepcopy` preserves flag)
- `model/make_model.py` — model factory (selects `PoseBackboneModel` when `POSE_BACKBONE_PSG=True`)
- `model/pose_psg_part_model.py` — subclass that overrides forward (not active in OP configs)
- `model/pose_model.py`, `model/pose_dual_stream_model.py` — not activated by OP small config
- `utils/flip_test.py` — flip preserves person indexing
- `test.py` — uses same `do_inference` / `forward()` path

## Findings

### Critical

None. The swap is a 2-line guarded addition behind a new default-False flag; all downstream reads of `scene_heatmaps` are textually below the swap point (line 473), so nothing escapes it.

### High

None.

### Medium

1. **Zero-target downstream behavior (behavioral, in-design-scope)** — `model/pose_backbone_model.py:929` computes `target_heatmaps = heatmaps[:, 0] * person_mask[:, 0]`. If a sample's `person_mask[:, 0] == 0` (target detection missing but distractor persons present), `target_heatmaps` is a zero tensor. With flag ON, LGPA's `_compute_pose_bias` then produces uniform attention favoring the background token (body_max=0 → bg=1), so the part-features for that sample collapse to background. PSG's gate with a zero heatmap after `sigmoid(0)=0.5` still produces only a small learned constant offset (last-layer weight/bias are zero-init and likely stay small), so PSG degrades gracefully. For Occ-PoseTrack the dataset tools guarantee a valid target (the dataset is defined by having an annotated target per clip), so this is unlikely to fire in practice — the design doc already calls this out (Risk #2). **Non-blocking**; worth a sanity assertion during training start-up (e.g. log the fraction of `person_mask[:,0]==0` over the first epoch) but not required for exp290 launch.

2. **Docstring nit in `config/defaults.py:107`** — the inline comment says "pose_dataset.py:_load_pose_data" but the function is actually named `_load_persons`. Not a correctness issue; fix opportunistically.

### Low

3. **Naming symmetry** — `model/modules/clip_part_head.py:222` defines `forward(self, feat_map, target_heatmaps, return_cls=True)` and the body-of-docstring at line 133 already calls the argument `TARGET person heatmaps`. Pre-change, the caller was passing `scene_heatmaps` into a param literally named `target_heatmaps` — a latent naming mismatch. The exp290 swap actually *aligns* runtime semantics with the pre-existing parameter name. No action needed; just noting that the module was always *intended* to receive target-only heatmaps.

4. **Minor internal inconsistency now resolved by flag ON** — `skeleton_gcn.py` has always sampled keypoints at `[:, 0]` (target only) while PSG/LGPA were seeing the max-merged scene heatmap. With flag ON, the full pose pipeline becomes consistently target-centric, improving internal coherence. This is a design strength of exp290, not a bug.

5. **Potential log improvement** — `pose_backbone_model.py:136-139` prints a static string; consider also logging `person_mask[:,0].mean()` over one batch at init to confirm the target signal is actually dense. Non-blocking.

## Backward compatibility (user-stressed)

Detailed flag=False path walkthrough, proving byte-identical behavior vs. HEAD:

**`__init__`** (line 133-139 diff):

```
self.use_target_heatmap = getattr(cfg.MODEL, 'POSE_USE_TARGET_HEATMAP', False)
if self.use_target_heatmap:
    print(...)
```

- `getattr` with `False` default → harmless for older YAMLs / older checkpoints missing the key.
- When flag is False: `self.use_target_heatmap = False` (attribute added, harmless); `if False:` → no print, no side effect.
- No module creation, no parameter registration, no buffer registration. Optimizer param groups, EMA teacher state-dict keys, saved checkpoint structure — all unchanged.

**`forward`** (line 466-473 diff):

```
if self.use_target_heatmap and target_heatmaps is not None:
    scene_heatmaps = target_heatmaps
```

- `self.use_target_heatmap == False` short-circuits the `and` → `target_heatmaps` branch not taken.
- `scene_heatmaps` retains its pre-swap value from `_prepare_pose`, which is the existing `merge_person_heatmaps(heatmaps, person_mask)` output — identical to HEAD.
- Line 462 `target_heatmaps = None` initialization and line 464 4-tuple unpacking `scene_heatmaps, _, target_heatmaps, _ = self._prepare_pose(pose_dict)` both **pre-exist** the exp290 change (verified via `git show HEAD:model/pose_backbone_model.py`). `_prepare_pose` already computed `target_heatmaps` and `diff_heatmaps` unconditionally at HEAD lines 454-456 (old line numbers). So no extra tensor allocation or compute is introduced on the flag-off path.
- Subsequent `scene_heatmaps` consumers (pose dropout 476-479, `_run_backbone_with_psg(x, scene_heatmaps)` 482, VCSR 500, STR 514, LGPA 586, PPA 608, GCN duals 646, FSDC 672, dual-branch viz 726, all test-time uses 770-867) receive the unmodified scene heatmap.

Conclusion: Flag=False path is **byte-identical** to HEAD. Existing experiments in flight (exp266b_3090, exp285b, exp287, exp288) are not affected; previously saved checkpoints load without issue (the new attribute is not in their state_dict and is initialized purely from cfg).

## Data flow verification

`_prepare_pose` (lines 908-938) returns:

- `scene_heatmaps`: `(B, 17, H, W)`, float32, on `heatmaps.device` — from `merge_person_heatmaps(heatmaps, person_mask)` (max over persons after masking).
- `target_heatmaps`: `(B, 17, H, W)`, float32, on `heatmaps.device` — `heatmaps[:, 0] * person_mask[:, 0].view(-1, 1, 1, 1)`.

Shape / dtype / device are **identical** between scene and target. After the swap, `scene_heatmaps` remains a (B, 17, H, W) float32 tensor on the correct device — no downstream interpolation / matmul / sigmoid will see a shape or dtype surprise.

**Downstream users of `scene_heatmaps` (post-swap, in order of first appearance)**:

| # | Location | Use |
|---|----------|-----|
| 1 | pose_backbone_model.py:477-479 | Pose Dropout zero-masking (OP small config has `POSE_DROPOUT_P=0.0`, so dormant) |
| 2 | pose_backbone_model.py:482 | `_run_backbone_with_psg(x, scene_heatmaps)` — routes to PSG, PAPE, PosePrompt, PAA via stage iteration |
| 3 | pose_backbone_model.py:497-500 | VCSR (not active) |
| 4 | pose_backbone_model.py:505-547 | STR / STD-PR (not active) |
| 5 | pose_backbone_model.py:582-586 | LGPA cross-attention head (ACTIVE, detached per `POSE_LGPA_DETACH=True`) |
| 6 | pose_backbone_model.py:605-608 | PPA (not active) |
| 7 | pose_backbone_model.py:639-672 | Dual STR+GCN / FSDC (not active) |
| 8 | pose_backbone_model.py:726 | `part_visibility` recompute for Dual branch (not active) |
| 9 | pose_backbone_model.py:770-867 | Test-time VCSR / LGPA / PPA / STR / FSDC paths (mirror training; LGPA test path active) |

All post-swap uses correctly receive the swapped tensor. There is **no shadowing** — `scene_heatmaps` is not re-assigned between the swap and any consumer except the in-place dropout multiply at line 479, which preserves the swap semantics (zeroing some samples' target heatmaps is still legitimate SPD augmentation).

`_run_backbone_with_psg` passes `scene_heatmaps` by value to `_run_stage_with_psg`, which in turn passes it to `self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)`. The PSG `forward` at `pose_spatial_gate.py:53` interpolates to the feature's spatial size, sigmoids, and runs a zero-initialized conv — all shape/dtype-agnostic w.r.t. whether the heatmap is scene-merged or target-only.

Skeleton GCN at `skeleton_gcn.py:745` takes `pose_dict` (not `scene_heatmaps`) and internally indexes `[:, 0]` at lines 498, 572, 653. So GCN's behavior is **unchanged by the flag** — GCN has always been target-only. The exp290 swap brings PSG/LGPA into alignment with GCN.

## Boundary conditions

1. **`pose_dict=None`**: Both `scene_heatmaps` and `target_heatmaps` are initialized to `None` at lines 461-462 and remain `None` because `_prepare_pose` is only invoked inside `if pose_dict is not None:`. The swap guard `self.use_target_heatmap and target_heatmaps is not None` short-circuits correctly — `scene_heatmaps` stays `None`. Downstream `if scene_heatmaps is not None` guards at lines 476, 497, 505, 582, 605, 639, 770, 777, 792, 806, 849, 860 all handle this correctly. **Safe.**

2. **`person_mask[:, 0] == 0` (target not detected)**: `target_heatmaps = heatmaps[:, 0] * 0 = zeros(B, 17, H, W)`. The swap produces a zero tensor. PSG receives `sigmoid(0)=0.5` constant input → zero-init conv → near-zero gate → `x * (1 + ~0) ≈ x`, i.e. passthrough (no harm done). LGPA receives zero heatmap → uniform attention bias toward background token → part features become essentially background-only. This is a graceful degradation, not a crash. For Occ-PoseTrack this case is rare (dataset has annotated target per clip, `target_person_idx` guarantees valid person 0). Flagged as Medium risk in findings; **not blocking**.

3. **Single-person image (nominal for OD / Market; fallback check for OP)**: `person_mask = [1, 0, 0, ...]`. `merge_person_heatmaps` computes `max(heatmaps * mask)` — only person 0 contributes → `scene_heatmap == heatmaps[:, 0]`. `target_heatmaps = heatmaps[:, 0] * 1 = heatmaps[:, 0]`. Therefore **scene_heatmap and target_heatmap are numerically identical** for single-person samples. Exp290 is a **strict no-op** on single-person data. This mathematically guarantees zero regression risk on OD / Market; only OP's multi-person samples see a behavioral change. **Safe.**

4. **Multi-person with target at index 0 (nominal OP case)**: `scene_heatmap = max(target, distractor_1, ..., distractor_N)` mixes target + distractor keypoints. `target_heatmap = heatmaps[:, 0]` keeps only target. Swap produces the intended target-only signal. **Correct.**

5. **`_prepare_pose` only invoked under `if pose_dict is not None:`**: the swap is additionally guarded by `target_heatmaps is not None`, which is redundant-but-safe belt-and-suspenders. The redundancy is harmless and increases robustness against future refactors that might call `_prepare_pose` defensively elsewhere.

## Pose dropout interaction

`POSE_DROPOUT_P` is 0.0 in `prcv_best_small.yml` (not set; default is `0.0`), so SPD is dormant for exp290. If it were enabled, the dropout at line 476-479 runs AFTER the swap, so it zeroes out per-sample target heatmaps during training — an SPD-style augmentation against the target signal. This is semantically valid (still encourages the backbone not to over-rely on pose) and shape/dtype-safe (keep_mask has matching device / broadcasts correctly). **Non-issue for this experiment**; forward-compatible with future configs that enable SPD.

## OA-SD teacher/student symmetry

`processor/processor.py:478` constructs `ema_teacher = copy.deepcopy(base_model)`. `deepcopy` copies `self.use_target_heatmap` (a plain Python bool attribute) from the student. Both teacher and student enter the same `forward()` method, so both apply the swap consistently when flag is on. The teacher's forward consumes `pose_dict.get('teacher_pose', pose_dict)` (line 795), where `teacher_pose` has the same target-at-index-0 ordering (built from `persons_clean_for_oa_sd = copy.deepcopy(persons)` at dataset.py:175, preserving the `_load_persons` target-reordering). **Teacher-student symmetry is preserved** — distillation loss compares target-only features on both sides, not mixed signals. OA-SD remains semantically intact.

## Test-time consistency

`test.py` → `do_inference` → `_extract_feat_flip` → `model(img, ..., pose_dict=pose_dict)` → same `forward()` → same swap. Flip-test (`utils/flip_test.py:6-40`) flips `heatmaps` along W and swaps L/R channels while preserving person indexing at `[:, 0]`, so the flipped forward also correctly extracts person-0 after the swap. **Train-test semantics match. No train-test mismatch.**

## Config reading robustness

`getattr(cfg.MODEL, 'POSE_USE_TARGET_HEATMAP', False)` at pose_backbone_model.py:135 handles: (a) old YAML configs without the key, (b) older trained checkpoints whose cfg is reconstructed from file, (c) hot-swaps where the attribute was absent. Combined with the default `_C.MODEL.POSE_USE_TARGET_HEATMAP = False` in `config/defaults.py:110`, the flag is rock-solid. YAML override like `MODEL.POSE_USE_TARGET_HEATMAP: True` or command-line `MODEL.POSE_USE_TARGET_HEATMAP True` both work with yacs merge.

## Model factory routing

`model/make_model.py:467-469` selects `PoseBackboneModel` only when `POSE_BACKBONE_PSG=True`, which is set in `prcv_best_small.yml:18`. `POSE_PSG_PART` and `POSE_DUAL_STREAM` are both False (not set in this YAML, default False). So `PoseBackboneModel.forward()` — the ONLY forward() modified for exp290 — is indeed the active code path. `pose_model.py` / `pose_dual_stream_model.py` / `pose_psg_part_model.py` are not active; their absence of the swap is irrelevant.

## Design doc sanity

`design.md` (134 lines) is honest and thorough:
- Motivation correctly cites KPR w/prompt 82.3/92.3 vs current 78.4/86.2 (3.8 mAP / 6.1 R1 gap).
- Identifies the specific code line (`pose_utils.py:21-39`) causing scene-mixing.
- Backward-compat section (lines 64-79) explicitly walks through default-off behavior.
- Risk assessment (lines 112-119) covers train/test mismatch (none), target annotation quality (pre-existing dataset issue), OD/Market regression (out of scope, single-person no-op), Full Scaffold compatibility, pose dropout.
- Expected results (lines 94-104) give a reasonable range (80-82 mAP) and three scenarios (SOTA, partial, null). No over-claiming; the "如果持平 exp265" fallback narrative is appropriately humble.

The single doc drawback: no explicit mention that the change is semantically a no-op on single-person data (which would be a strong backward-compat reassurance for OD / Market). Fix optional; the Medium-severity boundary-condition section of this review covers it.

## Scope creep / premature abstraction

The change is **minimally invasive**:
- 1 new cfg key
- 1 new instance attribute + 1 conditional print
- 1 2-line swap in `forward()`

No new modules, no new loss terms, no new tensor paths. The flag is perfectly decoupled: if the experiment fails, zero cleanup is required for other branches. No premature abstraction (e.g. no "heatmap source selector enum" — just a boolean). **Clean.**

## Risk assessment

- **Blast radius if experiment fails**: zero. Flag defaults off, no other experiment impacted. Existing in-flight jobs (exp266b_3090, exp285b, exp287, exp288, etc.) continue with unchanged behavior.
- **Implementation risk**: minimal. 2-line guarded swap with redundant `is not None` check.
- **Scientific risk**: moderate but well-bounded. The hypothesis (target-only heatmap closes 4 mAP gap vs KPR-with-prompt) is plausible given the structural reasoning (PSG/LGPA saw mixed signals on multi-person OP images), but not guaranteed. The `diff_heatmaps` fallback (`target - distractor_hm`) is already computed in `_prepare_pose` at line 936, leaving room for a follow-up exp if the pure-swap result is only a marginal improvement.

## Innovation quality check (per CLAUDE.md experiment_protocol)

"如果实验只改了配置参数或几行代码，审查必须质疑这是否只是小调参？"

- **This is NOT a parameter-tuning exp**: it changes the semantic content of the pose signal fed to the network, not a hyperparameter value.
- **This is NOT a microscopic tweak hiding as innovation**: the motivation (KPR uses test-time manual prompt; we use annotation-embedded target) is a legitimate architectural distinction. If SOTA is reached, the narrative "training + test target disambiguation via annotation replaces test-time keypoint prompt" is an honest paper-worthy story.
- **If result is <+1 mAP**: the design doc correctly pre-commits to an honest null / positive-but-marginal conclusion rather than p-hacking.

The review does not find this change to be a disguised ablation or parameter sweep.

## Verdict

审查通过

- No Critical or High findings.
- Two Medium findings (zero-target degradation edge case; minor docstring naming) are non-blocking, in-scope of design, and either covered by the design doc's risk section or cosmetic.
- Backward compatibility is airtight: flag=False is byte-identical to HEAD.
- Single-person datasets (OD / Market) are a **mathematically strict no-op** for this change.
- OA-SD teacher-student symmetry preserved via `copy.deepcopy`.
- Train-test consistency preserved (same `forward()` path).
- Design doc is well-scoped, honestly framed, and does not over-claim.

Cleared for Codex review and subsequent training launch.
