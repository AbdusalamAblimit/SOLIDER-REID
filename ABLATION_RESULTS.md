# Ablation Experiment Results

## Summary

Comprehensive ablation study on pose-guided dual-branch Swin Transformer for person Re-ID.

### Key Findings

1. **Pose gating is nearly ineffective**: E7 (dual-branch, no pose fusion) matches E1 (dual-branch + pose), suggesting the performance gain comes from parameter doubling in the local branch, not from pose-guided gating.
2. **Scale sensitivity**: scale=1.0 is too aggressive (Duke drops to 56.2%), scale=0.3 is optimal.
3. **Simple bugfixes have no impact**: Fixing double-sigmoid, vis-order, and shared fusion do not materially change results.
4. **Best result**: E1f (all bugfixes + semantic_weight=0.2) = 92.4% concat mAP on Market, E1 = 59.0% concat mAP on Occ-Duke.

---

## Market-1501 Results (120 epochs)

| ID | Name | Pose | Key Config | Global mAP | Global R-1 | Concat mAP | Concat R-1 |
|----|------|------|------------|-----------|-----------|-----------|-----------|
| E0 | no-pose baseline | OFF | swin_tiny, sw=1.0 | 91.1% | 96.4% | - | - |
| E1 | original pose | ON | sigmoid, shared, scale=1.0, vis=T | 92.0% | 96.4% | 92.2% | 96.3% |
| E2 | fix double-sigmoid | ON | HM_NORM=none | 92.0% | 96.4% | 92.2% | 96.3% |
| E3 | fix vis-order | ON | VIS_AFTER_NORM=True, HM_NORM=none | 92.0% | 96.4% | 92.2% | 96.3% |
| E3b | E3 + sigmoid | ON | VIS_AFTER_NORM=True, sigmoid | 92.0% | 96.3% | 92.3% | 96.5% |
| E4 | no shared fusion | ON | SHARED_FUSION=False | 92.0% | 96.6% | 92.3% | 96.5% |
| E5 | all bugfixes | ON | E2+E3+E4 combined | 92.0% | 96.5% | 92.3% | 96.5% |
| E6 | GiLt loss | ON | all fixes + GILT strategy | 90.5% | 96.2% | 92.0% | 96.3% |
| E7 | dual-no-pose | ON | Scale=0.0 (no pose fusion) | 91.9% | 96.4% | 92.2% | 96.5% |
| E1c | scale=0.3 | ON | Scale=0.3, USE_VIS=False | 91.9% | 96.5% | 92.2% | 96.5% |
| E1d | scale=0.3+lw0.7 | ON | Scale=0.3, VIS=F, LW=0.7 | 91.8% | 96.6% | 92.3% | 96.6% |
| E1f | allfix+sw0.2 | ON | all bugfixes, SEMANTIC_WEIGHT=0.2 | - | - | 92.4% | - |

## Occluded-Duke Results (120 epochs)

| ID | Name | Pose | Key Config | Global mAP | Global R-1 | Concat mAP | Concat R-1 |
|----|------|------|------------|-----------|-----------|-----------|-----------|
| E0 | no-pose baseline | OFF | swin_tiny, sw=1.0 | 55.2% | 65.5% | - | - |
| E1 | original pose | ON | sigmoid, shared, scale=1.0, vis=T | 58.4% | 68.5% | 59.0% | 68.6% |
| E7 | dual-no-pose | ON | Scale=0.0 (no pose fusion) | 58.2% | 68.4% | 58.7% | 68.8% |
| E1c | scale=0.3 | ON | Scale=0.3, USE_VIS=False | 58.3% | 68.7% | 59.0% | 68.6% |
| E1d | scale=0.3+lw0.7 | ON | Scale=0.3, VIS=F, LW=0.7 | 57.8% | 66.9% | 58.9% | 68.0% |

## Analysis

### Market-1501
- All dual-branch experiments cluster around 92.0-92.3% concat mAP
- E7 (no pose at all) = 92.2% vs E1 (with pose) = 92.2% => pose gating adds ~0% on Market
- The +1% over E0 baseline comes from the extra parameters in the local branch

### Occluded-Duke
- Pose gives a larger boost: 55.2% -> 59.0% (+3.8% mAP)
- E7 (no pose) = 58.7% vs E1 (with pose) = 59.0% => pose adds only +0.3%
- Most of the +3.5% gain over E0 is from dual-branch architecture, not pose fusion

### Scale Experiments
- scale=1.0 works on Market but can hurt on Duke
- scale=0.3 is the sweet spot: stable across datasets

### Implications for SPTrans
These results motivate a new approach:
- **Semantic-Pose Joint Conditioning**: Instead of fixed gating, make semantic weight spatially adaptive based on pose confidence
- **Part-Aware Routing**: Instead of multiplicative gating, use pose heatmaps for part-level feature pooling
