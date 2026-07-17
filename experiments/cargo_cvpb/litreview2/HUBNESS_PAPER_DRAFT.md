# Gallery Negative In-Degree: A Topological Diagnosis of Residual Failures in Strong Person Re-Identification

> Analysis-short 初稿（2026-06-24）。诊断贡献, 非训练方法。数字均来自零训练冻结实验（exp255/exp260b），可直接进表。用户可重写措辞/调框架。

## Abstract
Modern person ReID backbones (Swin/SOLIDER) reach 73–95 mAP, yet a non-trivial residual of queries still fail. We ask **where** these failures concentrate, and show they are not well explained by per-query pairwise difficulty. Instead, we identify a **topological** hidden variable: a small set of gallery images act as **negative attraction hubs**—each is pulled into the top-k of *many queries of different identities*. We formalize this as the gallery **negative in-degree** `H_k(g)` and the per-query **hub mass** `M(q)`, and show on Occluded-Duke and Market-1501 that (i) `M(q)` explains per-query AP error with Spearman ρ up to **+0.65** (perm-p<0.001), surviving partial-correlation control against feature-norm, top-1 margin, camera, and #positives; (ii) negative in-degree is **near-orthogonal** to generic sample popularity (all in-degree); (iii) high-hub galleries cluster in feature space across *24 distinct identities* (mean pairwise cosine 0.166 vs 0.025 control; 27× hub-enrichment in their kNN) because the model **over-encodes a non-identity scene factor**. We are explicit that the *remedy* for this failure is already covered by k-reciprocal re-ranking and camera-aware corrections; our contribution is a **clean, falsifiable diagnosis** of the residual-failure structure of strong ReID, not a new retrieval method.

## 1. Introduction
Strong supervised ReID is often treated as "solved" on Market (94+ mAP). Yet a residual of queries fail even with state-of-the-art backbones. The dominant framing treats each failure as an isolated **pairwise** problem—this query's feature is not close enough to its gallery match. We argue the residual failure is better understood as a **directed kNN-graph** property of the gallery: **a few gallery images become attraction hubs for many queries of unrelated identities** (many-to-one). The hidden variable is the gallery's **negative in-degree**, distinct from hard-negative distance (hard = close to *one* anchor; hub = close to *many different identities*).

Contributions:
- We define gallery negative in-degree `H_k(g)` and query hub mass `M(q)` and show `M(q)` is the dominant explanatory variable for residual AP error (ρ+0.60 on Occluded-Duke), beyond cheap difficulty proxies.
- Destructive controls (permutation, partial-correlation, negative-vs-all in-degree) establish the variable is real, independent, and specifically *negative*.
- A failure-case characterization shows the mechanism: hubs arise from **non-identity scene over-encoding**, not identity confusion.
- We honestly bound the contribution: the remedy lies within the space already covered by k-reciprocal/camera-aware re-ranking, so this is a **diagnosis**, not a method.

## 2. Related Work
**Re-ranking / neighbor topology.** k-reciprocal re-ranking and CA-Jaccard exploit gallery-gallery neighbor structure at test time; they *fix* part of the topology we diagnose, but do not *define* negative in-degree as the failure variable nor isolate it from camera bias. **Hubness in retrieval.** Radović et al. define hubness in high-dim kNN; CSLS/Mutual-Proximity are post-hoc hub corrections in cross-modal retrieval; HAL/NeighborRetr add training-time hubness-aware losses for image-text. None target person-ReID gallery negative in-degree as a residual-failure diagnostic. **Difficulty-aware ReID.** Hard-negative mining and difficulty-adaptive inference use per-image difficulty; our variable is relation-level (many-to-one absorption), not per-image hardness.

## 3. The Hidden Variable
- Gallery negative in-degree: `H_k(g) = #{ q : g ∈ top-k(q) ∧ y_g ≠ y_q }`.
- Query hub mass: `M(q) = Σ_{g∈topk(q), y_g≠y_q} H_k(g)`.
Both are computed once from frozen embeddings; no training. We use a strong frozen ckpt per dataset (Occluded-Duke exp255 mAP 73.05; Market exp260b mAP 94.61).

## 4. Experiments
**4.1 Hub mass explains residual AP error (Table 1).**
| | Market | Occluded-Duke |
|---|---|---|
| ρ(AP-err, M(q)) | +0.28 | **+0.65** (perm-p<0.001) |
| partial ρ (M \| norm+margin+camera+#pos) | +0.33 | **+0.60** |
| top-1% hub share of false-top1 (k=10) | 22.2% | 26.7% (k=5: 30.7%) |
Harder/un-saturated benchmarks show a *larger* hub disease (2.3× explanatory power, 5× the zero-training intervention gain).

**4.2 Destructive controls.**
- D1 (permute `H_k`): intervention gain → +0.002, i.e. signal is real.
- D3 (partial vs 4 cheap proxies): ρ stays +0.60 on Occluded-Duke → not a difficulty proxy.
- D4 (negative vs all in-degree): per-query rho +0.65 (negative) vs +0.02 (all); intervention +1.51 (neg) / +0.00 (all). The failure variable is specifically *cross-identity* absorption, not generic popularity.

**4.3 Mechanism: non-identity scene over-encoding (Fig. 1).**
Top-30 hub galleries vs camera-matched controls: hubs are brighter (140 vs 119), more colorful (18 vs 11), and a bright high-contrast scene (a parked orange car + brick plaza) is over-represented (~7–8/30 vs 0–1/30 base rate). Critically, the 30 hubs span **24 distinct identities** yet cluster with mean pairwise cosine **0.166** (control 0.025); each hub's 10-NN contains 26.7% other top-1% hubs (27× the 1% base rate). The model groups unrelated identities by a **shared non-identity factor**—the scene—producing the cross-ID attraction. (All replicated within camera-0 to exclude camera confound.)

## 5. Discussion: Diagnosis, not Remedy
We tested the obvious remedies and report them honestly. A zero-training hub penalty `score' = cos − λ·log(1+H_k)` gives only +0.31 (Market) / +1.51 (Occluded-Duke) mAP, and is **dominated** by same-camera down-weighting (+0.67 / +3.13) and k-reciprocal (+1.26 / **+10.98**)—the gap *widens* on the harder set. A training-side anti-hub embedding sits in the same space already covered by re-ranking. The mechanism (scene over-encoding) points to background/region suppression, which is non-generalizable here (a dataset-specific scene) and overlaps prior pose-masked suppression. We therefore present negative in-degree as a **diagnostic**: it tells you *where* strong ReID fails and *why* (gallery topology / non-identity factor), while the *fix* remains the province of established test-time tools.

## 6. Conclusion
Residual failures of strong person ReID have a clean topological structure: a few gallery images become cross-identity negative-in-degree hubs, driven by non-identity scene over-encoding, and `M(q)` explains per-query AP error (ρ+0.60, controlled) far better than pairwise difficulty. This reframes "where strong ReID fails" from pairwise similarity to directed gallery topology—a diagnosis we hope motivates remedies beyond the re-ranking space that currently subsumes the obvious ones.

---
*Figures*: Fig.1 = `hub_failure_grid_FINAL.png` (hub / control / random base-rate). *Repro*: `cvpb_hubness_killswitch.py`, `hub_failure_characterize.py`. *Logs*: `hubness_logs/hub_full_{market,occluded_duke}.log`.
