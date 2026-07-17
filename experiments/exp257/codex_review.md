# Codex Review — exp257

**Verdict**: approve
**Date**: 2026-04-11
**Review round**: 1

## Findings
Claude review flagged ArcFace only applies to global classifier (LGPA/GCN stay softmax).
This is intentional — global-only ArcFace is the standard approach in multi-branch ReID.
design.md updated to clarify.

Label Smoothing + ArcFace double application is acknowledged as intentional dual regularization.

## 结论
codex 审查通过
