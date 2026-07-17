#!/usr/bin/env python3
"""exp360 Intruder stage-0 probe — verify the MECHANISM PREMISE before any training.

H1: a donor person pasted onto a target leaks donor-ID into the target embedding
    (donor-ID linear probe on f(mix) >> chance).
H2: the amount of donor leakage correlates with target retrieval damage
    (leak = cos(f_mix,f_donor) - cos(f_clean,f_donor)  vs per-query AP drop),
    AND survives the trivial #false-in-topk control (memory iron rule: any per-query
    explanatory variable must beat #false-in-topk partial correlation).
Control: random gray patch (same area/position, non-person) should leak far less than a real donor.

Frozen strong baseline (no training here). PASS => mechanism real => go stage-1 (adversarial suppression).
PASS: H1 donor-probe acc >= 3x chance  AND  H2 partial-spearman(leak, APdrop | #false) >= +0.15
      AND  person-leak >> rand-leak. Otherwise the donor-suppression premise is wrong, revisit synth/mechanism."""
import sys, os, argparse, numpy as np
from collections import defaultdict
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='log/market1501/exp359_abl_noLMloss/transformer_40.pth')
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--occ_frac', type=float, default=0.45)   # donor occupies bottom occ_frac of target
ap.add_argument('--n_donor_id', type=int, default=50)      # K donor PIDs for H1 probe
ap.add_argument('--per_donor', type=int, default=20)       # M targets per donor PID
ap.add_argument('--n_query', type=int, default=700)        # H2 queries
ap.add_argument('--n_gallery', type=int, default=4000)
ap.add_argument('--probe_epochs', type=int, default=60)
cli = ap.parse_args()
sys.argv = ['ks', '--ckpt', cli.ckpt, '--config', cli.config, '--data_root', 'data']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpb_lattice_killswitch as ks
import torch, torch.nn.functional as F
from datasets.bases import read_image
from PIL import Image
RNG = np.random.RandomState(42)
ext = ks.FrozenExtractor(); REPO = ks._repo; DEV = 'cuda'


def items(split):
    return ks.list_split(os.path.join(REPO, 'data', 'market1501', split))


def paste_donor(target_pil, donor_pil, frac, mode='person'):
    """Paste donor onto bottom `frac` of target (person occlusion常下半身). mode='rand' = gray patch control."""
    t = ks._to_target_aspect(target_pil).copy(); W, H = t.size
    oh = max(1, int(H * frac))
    if mode == 'person':
        d = ks._to_target_aspect(donor_pil).resize((W, oh))
        t.paste(d, (0, H - oh))
    else:
        t.paste(Image.new('RGB', (W, oh), (127, 127, 127)), (0, H - oh))
    return t


def feats(pils):
    return ext.feats_from_pil(pils)   # [N,D] L2-normalized


# ---------- H1: donor-ID leaks into f(mix) ----------
print(f"[H1] build {cli.n_donor_id} donor IDs x {cli.per_donor} targets ...", flush=True)
train_its = items('bounding_box_train')
by_pid = defaultdict(list)
for it in train_its:
    by_pid[it[1]].append(it[0])
pids = [p for p in by_pid if len(by_pid[p]) >= 2]
donor_pids = list(RNG.choice(pids, cli.n_donor_id, replace=False))
other_pids = [p for p in pids if p not in set(donor_pids)]
mix_pils, donor_label = [], []
for di, dp in enumerate(donor_pids):
    donor_path = by_pid[dp][0]
    donor_im = read_image(donor_path)
    tgt_pids = RNG.choice(other_pids, cli.per_donor, replace=False)
    for tp in tgt_pids:
        tgt_im = read_image(by_pid[tp][RNG.randint(len(by_pid[tp]))])
        mix_pils.append(paste_donor(tgt_im, donor_im, cli.occ_frac, 'person'))
        donor_label.append(di)
Fmix = feats(mix_pils); ylab = np.array(donor_label)
# linear probe donor-ID on f(mix), 80/20 split
N = len(ylab); idx = RNG.permutation(N); ntr = int(N * 0.8)
tr, te = idx[:ntr], idx[ntr:]
Xtr = torch.tensor(Fmix[tr], device=DEV); ytr = torch.tensor(ylab[tr], device=DEV)
Xte = torch.tensor(Fmix[te], device=DEV); yte = torch.tensor(ylab[te], device=DEV)
clf = torch.nn.Linear(Fmix.shape[1], cli.n_donor_id).to(DEV)
opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
for ep in range(cli.probe_epochs):
    opt.zero_grad(); loss = F.cross_entropy(clf(Xtr), ytr); loss.backward(); opt.step()
with torch.no_grad():
    acc = (clf(Xte).argmax(1) == yte).float().mean().item()
chance = 1.0 / cli.n_donor_id
print(f"[H1] donor-ID probe acc={acc:.4f}  chance={chance:.4f}  ratio={acc/chance:.2f}x", flush=True)

# ---------- H2: leak vs AP drop (+ #false control) + rand control ----------
print(f"[H2] {cli.n_query} queries vs {cli.n_gallery} gallery ...", flush=True)
q_its = items('query')[:cli.n_query]
g_its = items('bounding_box_test')[:cli.n_gallery]
q_pid = np.array([x[1] for x in q_its]); q_cam = np.array([x[2] for x in q_its])
g_pid = np.array([x[1] for x in g_its]); g_cam = np.array([x[2] for x in g_its])
# donor per query: a random different-PID train crop
donor_for_q = []
for qp in q_pid:
    dp = RNG.choice(other_pids)
    donor_for_q.append(read_image(by_pid[dp][0]))
clean_pils = [ks._to_target_aspect(read_image(x[0])) for x in q_its]
mixp_pils = [paste_donor(read_image(q_its[i][0]), donor_for_q[i], cli.occ_frac, 'person') for i in range(len(q_its))]
mixr_pils = [paste_donor(read_image(q_its[i][0]), donor_for_q[i], cli.occ_frac, 'rand') for i in range(len(q_its))]
Fclean = feats(clean_pils); Fmixp = feats(mixp_pils); Fmixr = feats(mixr_pils)
Fdonor = feats([ks._to_target_aspect(d) for d in donor_for_q])
Fg = feats([ks._to_target_aspect(read_image(x[0])) for x in g_its])
leak_p = np.sum(Fmixp * Fdonor, 1) - np.sum(Fclean * Fdonor, 1)   # per-query donor leak (person)
leak_r = np.sum(Fmixr * Fdonor, 1) - np.sum(Fclean * Fdonor, 1)   # control (rand patch)
ap_clean = ks.per_query_ap(-(Fclean @ Fg.T), q_pid, q_cam, g_pid, g_cam)   # funcs want DIST not sim
ap_mix = ks.per_query_ap(-(Fmixp @ Fg.T), q_pid, q_cam, g_pid, g_cam)
ap_drop = ap_clean - ap_mix
nfalse = ks.n_false_in_topk(-(Fmixp @ Fg.T), q_pid, q_cam, g_pid, g_cam, k=10)
sp_raw = ks.spearman(leak_p, ap_drop)[0]          # spearman returns (corr, n)
sp_part = ks.partial_spearman(leak_p, ap_drop, nfalse[:, None])[0]
print(f"[H2] leak-person mean={leak_p.mean():.4f}  leak-rand mean={leak_r.mean():.4f}  "
      f"(person/rand={leak_p.mean()/ (leak_r.mean()+1e-9):.2f}x)")
print(f"[H2] AP drop mean={ap_drop.mean():.4f}  (clean {ap_clean.mean():.3f} -> mix {ap_mix.mean():.3f})")
print(f"[H2] spearman(leak,APdrop) raw={sp_raw:+.3f}  partial|#false={sp_part:+.3f}")
h1_ok = acc / chance >= 3.0
h2_ok = sp_part >= 0.15
ctrl_ok = leak_p.mean() >= 2.0 * leak_r.mean()
print(f"\n[VERDICT-H] H1(donor-leak可测)={'PASS' if h1_ok else 'FAIL'}  "
      f"H2(leak↔APdrop,控#false)={'PASS' if h2_ok else 'FAIL'}  person>>rand={'PASS' if ctrl_ok else 'FAIL'}")

# ---------- Stage0.5: donor-null projection CAUSAL test (codex 7/10) ----------
# Frozen upper-bound proxy for GRL: rank-r remove donor-discriminative directions, see if mix AP recovers.
# If projecting out donor-ID does NOT recover AP, GRL training won't help -> revisit/reselect.
print("\n[Stage0.5] donor-null projection causal test (does removing donor-ID rescue retrieval?)", flush=True)
Wd = clf.weight.detach().cpu().numpy().astype(np.float32)         # [K,D] donor directions
_, _, Vt = np.linalg.svd(Wd, full_matrices=False)                 # Vt[:r] = top-r donor subspace
Dd = Fmix.shape[1]
base_mixAP, base_cleanAP, base_nf = ap_mix.mean(), ap_clean.mean(), nfalse.mean()
def proj_out(Fm, r):
    Ur = Vt[:r]
    Fp = Fm - (Fm @ Ur.T) @ Ur
    return Fp / (np.linalg.norm(Fp, axis=1, keepdims=True) + 1e-12)
best = None
for r in [5, 10, 20, 40]:
    Xtr_p = torch.tensor(proj_out(Fmix[tr], r), device=DEV); Xte_p = torch.tensor(proj_out(Fmix[te], r), device=DEV)
    c2 = torch.nn.Linear(Dd, cli.n_donor_id).to(DEV); o2 = torch.optim.Adam(c2.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(cli.probe_epochs):
        o2.zero_grad(); F.cross_entropy(c2(Xtr_p), ytr).backward(); o2.step()
    acc_r = (c2(Xte_p).argmax(1) == yte).float().mean().item()
    Fc_p, Fm_p, Fg_p = proj_out(Fclean, r), proj_out(Fmixp, r), proj_out(Fg, r)
    apm = ks.per_query_ap(-(Fm_p @ Fg_p.T), q_pid, q_cam, g_pid, g_cam).mean()
    apc = ks.per_query_ap(-(Fc_p @ Fg_p.T), q_pid, q_cam, g_pid, g_cam).mean()
    nfp = ks.n_false_in_topk(-(Fm_p @ Fg_p.T), q_pid, q_cam, g_pid, g_cam, k=10).mean()
    print(f"[S0.5 r={r:2d}] donorAcc {acc:.3f}->{acc_r:.3f}  mixAP {base_mixAP:.3f}->{apm:.3f} (Δ{apm-base_mixAP:+.3f})  "
          f"cleanAP {base_cleanAP:.3f}->{apc:.3f} (Δ{apc-base_cleanAP:+.3f})  #false {base_nf:.2f}->{nfp:.2f}", flush=True)
    if acc_r <= 0.15 and (best is None or apm - base_mixAP > best[1]):
        best = (r, apm - base_mixAP, apc - base_cleanAP)
s05_ok = best is not None and best[1] >= 0.03 and best[2] >= -0.01
if best is None:
    msg = ">>> 无 r 能把 donor acc 压到 <=15% (donor 信息高度分布式) — frozen 投影抠不掉, 转 B PSC-JEPA"
elif s05_ok:
    msg = f">>> CAUSAL PASS: donor-null 救 AP Δ{best[1]:+.3f} @r={best[0]} (clean Δ{best[2]:+.3f}) — go small Stage1 GRL"
else:
    msg = f">>> CAUSAL FAIL: 压到 acc<=15% 但 mixAP 只 Δ{best[1]:+.3f} (<+0.03) — 压泄漏不救排序, 转 B PSC-JEPA"
print(f"[Stage0.5 VERDICT] {msg}")
print("[done] intruder stage-0 + stage0.5 complete.")
