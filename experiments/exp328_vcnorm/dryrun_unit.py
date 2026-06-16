#!/usr/bin/env python
"""VC-Norm dry-run unit checks (no training, no data needed).

Verifies the load-bearing correctness claims before any real training:
  1. VCN module at zero-init is an EXACT identity (out == in).
  2. vcnorm_align_loss runs, returns finite scalar + stats, gradient flows to
     student only (teacher detached), AND is ZERO when student==teacher.
  3. All-occluded-teacher -> no valid keypoints -> zero loss, finite grad.
  4. AMP (float16) safety: VCN forward + align loss under autocast.
  5. *** High-1 fix verification ***: the OCCLUDED student cohort
     (s_sc<thr & t_sc>=thr) actually enters the statistic and RECEIVES gradient,
     while the both-visible (non-occluded) student tokens get ~zero gradient.
     This is the whole point of the fix — without it VCA aligns the wrong tokens.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from model.modules.vcnorm import VisibilityConditionedNorm
from loss.vcnorm_loss import vcnorm_align_loss

torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
B, K, C = 8, 17, 1024  # Swin-Base feat dim

print(f'[dryrun] device={dev}, B={B} K={K} C={C}')

# ---- 1. VCN zero-init identity ----
vcn = VisibilityConditionedNorm(feat_dim=C, hidden=64).to(dev)
feats = torch.randn(B, K, C, device=dev)
scores = torch.rand(B, K, device=dev)
out, stats = vcn(feats, scores)
max_abs_diff = (out - feats).abs().max().item()
print(f'[1] zero-init identity: max|out-in|={max_abs_diff:.3e}  '
      f'gain_abs={stats["vcn_gain_abs"]:.3e} shift_abs={stats["vcn_shift_abs"]:.3e}')
assert max_abs_diff < 1e-5, f'VCN not identity at init! diff={max_abs_diff}'
print('    PASS: VCN zero-init == identity (VCNORM_MODULE=True untrained == baseline)')

# ---- 2. align loss: finite, teacher-detached, zero when equal ----
# Make some keypoints student-occluded so the occluded cohort is non-empty.
student = torch.randn(B, K, C, device=dev, requires_grad=True)
teacher = torch.randn(B, K, C, device=dev)
s_sc = torch.rand(B, K, device=dev)
s_sc[:, :6] = 0.0                        # first 6 keypoints occluded in student
t_sc = torch.rand(B, K, device=dev) * 0.5 + 0.5  # teacher mostly visible
loss, lstats = vcnorm_align_loss(student, teacher, s_sc, t_sc, vis_thr=0.3)
print(f'[2] align loss={loss.item():.4f} stats={ {k: round(v,3) for k,v in lstats.items()} }')
assert torch.isfinite(loss), 'align loss not finite'
assert lstats['vca_valid_k'] > 0, 'no valid keypoint despite occluded cohort present'
loss.backward()
assert student.grad is not None and torch.isfinite(student.grad).all(), 'no/NaN grad to student'
assert teacher.grad is None, 'teacher received grad (should be detached)'
print('    PASS: finite loss, valid_k>0, grad to student only, teacher detached')

# zero when student==teacher AND occluded cohort is just-as-occluded.
# Here we feed identical features with the SAME occlusion pattern on both sides
# for the occluded cohort; the occluded-student moment equals the teacher moment
# (same tokens), so the alignment loss must be ~0.
x = torch.randn(B, K, C, device=dev)
sc_occ = torch.rand(B, K, device=dev) * 0.5 + 0.5
sc_occ[:, :6] = 0.0                       # student-occluded keypoints
tc_vis = torch.rand(B, K, device=dev) * 0.5 + 0.5  # teacher all-visible
loss_eq, eqs = vcnorm_align_loss(x.clone(), x.clone(), sc_occ, tc_vis, vis_thr=0.3)
print(f'    student==teacher align loss={loss_eq.item():.3e} valid_k={eqs["vca_valid_k"]}')
assert loss_eq.item() < 1e-3, f'align loss not ~0 when equal: {loss_eq.item()}'
print('    PASS: align loss ~0 when student==teacher on the occluded cohort')

# ---- 3. all-occluded teacher -> no valid keypoints -> zero loss, finite grad ----
s2 = torch.randn(B, K, C, device=dev, requires_grad=True)
t2 = torch.randn(B, K, C, device=dev)
t_sc_low = torch.zeros(B, K, device=dev)  # teacher sees nothing
loss0, st0 = vcnorm_align_loss(s2, t2, s_sc, t_sc_low, vis_thr=0.3)
loss0.backward()
print(f'[3] all-occluded-teacher loss={loss0.item():.3e} valid_k={st0["vca_valid_k"]} '
      f'grad_finite={torch.isfinite(s2.grad).all().item()}')
assert loss0.item() == 0.0 and st0['vca_valid_k'] == 0.0
print('    PASS: graceful zero-loss + finite grad when no valid teacher keypoint')

# ---- 4. AMP float16 safety ----
if dev == 'cuda':
    vcn2 = VisibilityConditionedNorm(feat_dim=C, hidden=64).to(dev)
    with torch.cuda.amp.autocast(enabled=True):
        f16 = torch.randn(B, K, C, device=dev)
        sc16 = torch.rand(B, K, device=dev)
        o16, _ = vcn2(f16, sc16)
        st16 = torch.randn(B, K, C, device=dev, requires_grad=True)
        tt16 = torch.randn(B, K, C, device=dev)
        s_sc16 = torch.rand(B, K, device=dev); s_sc16[:, :6] = 0.0
        t_sc16 = torch.rand(B, K, device=dev) * 0.5 + 0.5
        l16, _ = vcnorm_align_loss(st16, tt16, s_sc16, t_sc16, vis_thr=0.3)
    print(f'[4] AMP: vcn out dtype={o16.dtype}, align loss dtype={l16.dtype}, '
          f'finite={torch.isfinite(l16).item()}')
    assert o16.dtype == f16.dtype, 'VCN changed dtype under AMP'
    assert torch.isfinite(l16), 'AMP align loss not finite'
    print('    PASS: AMP-safe (dtype preserved, finite loss)')
else:
    print('[4] SKIP AMP (no CUDA)')

# ---- 5. High-1 fix: OCCLUDED student tokens get gradient, visible ones do NOT ----
# Build a clean separation: keypoints 0..5 occluded in student (s_sc=0) but
# visible in teacher (t_sc high); keypoints 6..16 visible in BOTH. Per the fix,
# the student moment is taken ONLY over the occluded&teacher-visible cohort, so:
#   - occluded student tokens (k<6) must receive non-zero gradient,
#   - both-visible student tokens (k>=6) must receive ~zero gradient
#     (they are excluded from the student moment -> no alignment signal).
s3 = torch.randn(B, K, C, device=dev, requires_grad=True)
t3 = torch.randn(B, K, C, device=dev)
s_sc3 = torch.full((B, K), 0.9, device=dev)
s_sc3[:, :6] = 0.0                        # occluded in student
t_sc3 = torch.full((B, K), 0.9, device=dev)  # all visible in teacher
loss3, st3 = vcnorm_align_loss(s3, t3, s_sc3, t_sc3, vis_thr=0.3)
loss3.backward()
gpk = s3.grad.abs().sum(dim=(0, 2))       # (K,) total grad magnitude per keypoint
occ_grad = gpk[:6].sum().item()
vis_grad = gpk[6:].sum().item()
print(f'[5] occluded-token grad sum={occ_grad:.4e}  visible-token grad sum={vis_grad:.4e}')
print(f'    valid_k={st3["vca_valid_k"]}  occ_ratio={st3["vca_occ_ratio"]:.3f}')
assert st3['vca_valid_k'] == 6.0, f'expected 6 valid (occluded) keypoints, got {st3["vca_valid_k"]}'
assert occ_grad > 1e-6, 'OCCLUDED student tokens received NO gradient (High-1 bug NOT fixed!)'
assert vis_grad < 1e-8, f'both-visible tokens leaked gradient ({vis_grad:.2e}); cohort selection wrong'
print('    PASS: occluded student tokens get gradient, both-visible tokens do NOT (High-1 fixed)')

print('\n[dryrun] ALL UNIT CHECKS PASSED')
