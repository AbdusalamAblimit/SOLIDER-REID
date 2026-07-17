# encoding: utf-8
"""
AIRL fusion -- ZERO-TRAINING oracle kill-switch (codex red-team design).

Question: AIRL (degrade-ground-only regularizer) gives A->G +3.15 / G->A -3.18 on
CARGO Swin (mean ~tie, 60.83 vs baseline 60.84). The red team notes a DIRECTIONAL
upper bound: route A->G to AIRL, G->A to baseline -> mean = (61.90+62.93)/2 = 62.42
= baseline +1.58. This script asks whether a LEGAL FIXED gate (NOT test-tuned)
can approach that bound. If yes (>=+1.0) -> build the dual-branch resolvability
mechanism. If a legal fixed gate cannot recover the trade-off (<+0.5) -> KILL.

NO TRAINING. Only the two existing checkpoints + their CARGO eval features.
    baseline-Swin: cvpb_swin_baseline256/model_best.pth   (mean 60.84)
    AIRL-Swin    : cvpb_airl_swin2/model_best.pth          (mean 60.83)
Both are eval-architecture-identical (AIRL is a training-time-only ground
degradation regularizer; "无新可学参数, eval 路径与 baseline 逐键相同"), so each
loads into the plain baseline build_model(use_afd=False) and we use the standard
L2-normalized BNNeck global feature -- the exact afd_train.eval_market feature.

Gates evaluated (final metric is always mean = (A->G mAP + G->A mAP)/2):
  - view/direction gate (legal upper bound): A->G uses AIRL, G->A uses baseline.
    Query view is known at test time, so this is a legal gate, not an oracle.
  - area-threshold gate (single-model-approximable): route each query by its
    NATIVE bbox area; low area -> AIRL branch, high area -> baseline branch.
    Threshold = CARGO TRAIN-split area quantile (per direction's query view),
    NOT a test-tuned threshold.
  - reliability gate: route each query by baseline top-1 cosine confidence; low
    confidence -> AIRL. Threshold = TRAIN-derived confidence quantile (computed
    from train images scored against the test gallery of the opposite view;
    train identities are disjoint from test, so the gallery's similarity scale is
    a legal calibration source, never the test queries' labels). We ALSO report
    the test-optimal threshold purely as an oracle ceiling for context.
  - per-query oracle (theoretical upper bound): per query pick the better branch.
  - score fusion (soft, no hard routing): cos = w*cos_AIRL + (1-w)*cos_base,
    sweep w; same w for both directions -> mean.

Run on lab-4090:
    cd /home/afr/SOLIDER-REID/experiments/cargo_cvpb
    PYTHONUNBUFFERED=1 /home/afr/vireid/.venv/bin/python airl_gate_oracle.py \
        --base_ckpt /home/afr/SOLIDER-REID/log/cargo/cvpb_swin_baseline256/model_best.pth \
        --airl_ckpt /home/afr/SOLIDER-REID/log/cargo/cvpb_airl_swin2/model_best.pth \
        --swin_pretrain /home/afr/SOLIDER-REID/pretrained/swin_small.pth \
        --data_root /home/afr/SOLIDER-REID/data \
        2>&1 | tee /tmp/airl_gate_oracle.log
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_AFD_REID = os.path.join(_HERE, '..', 'afd_reid')
sys.path.insert(0, _AFD_REID)

from cargo_dataset import (CARGO, CARGOImageDataset, build_transforms,  # noqa: E402
                           filter_by_view)
from afd_model import build_model  # noqa: E402


# --------------------------------------------------------------------------- #
# model config (baseline eval path; identical for both checkpoints)
# --------------------------------------------------------------------------- #
def _args(backbone, swin_pretrain, img_size, semantic_weight=0.2):
    ns = argparse.Namespace()
    ns.last_stride = 1
    ns.pool = 'gem'
    ns.use_afd = False
    ns.afd_router = False
    ns.afd_cvfc = False
    ns.afd_stage = 'layer1'
    ns.router_cond_view = False
    ns.low_r, ns.mid_r, ns.high_drop_p = 0.125, 0.30, 0.5
    ns.backbone = backbone
    ns.swin_pretrain = swin_pretrain
    ns.swin_semantic_weight = semantic_weight
    ns.img_size = img_size
    return ns


@torch.no_grad()
def extract(model, loader, device):
    """Return feats[N,D] (float cpu, L2-normalized BN feat), pids, camids, paths."""
    model.eval()
    feats, pids, camids, paths = [], [], [], []
    for batch in loader:
        imgs = batch['img'].to(device, non_blocking=True)
        f = model(imgs)                       # eval -> L2-normalized BN feature
        feats.append(f.float().cpu())
        pids.append(batch['pid'])
        camids.append(batch['camid'])
        paths.extend(batch['img_path'])
    feats = torch.cat(feats, 0)
    pids = torch.cat(pids, 0).numpy().astype(np.int64)
    camids = torch.cat(camids, 0).numpy().astype(np.int64)
    return feats, pids, camids, paths


def load_model(ckpt, backbone, swin_pretrain, img_size, num_classes, device,
               tag=''):
    model = build_model(num_classes=num_classes,
                        args=_args(backbone, swin_pretrain, img_size)).to(device)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    res = model.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys]
    unexp = [k for k in res.unexpected_keys]
    backbone_miss = [k for k in miss if not k.startswith('classifier')]
    print(f'  [load {tag}] missing={len(miss)} unexpected={len(unexp)}  '
          f'backbone_missing={len(backbone_miss)}')
    if backbone_miss:
        print(f'    [WARN backbone missing] {backbone_miss[:8]}')
    if unexp:
        print(f'    [unexpected sample] {unexp[:6]}')
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# mAP / CMC from a precomputed distmat  (mirrors afd_train.eval_market exactly;
# refactored to accept any distmat so we can route/fuse per query)
# --------------------------------------------------------------------------- #
def mAP_from_distmat(distmat, q_pids, q_camids, g_pids, g_camids, max_rank=50):
    """Return (mAP, cmc[max_rank], mINP, per_query_AP_full[num_q]).

    per_query_AP_full is NaN for queries with no valid gallery match (so the
    oracle min-over-branches is well defined and consistent with eval_market's
    'skip queries with no GT' rule)."""
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP, all_INP = [], [], []
    per_q_ap = np.full(num_q, np.nan, dtype=np.float64)
    num_valid_q = 0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue
        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)[0]
        max_pos_idx = pos_idx[-1]
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)
        cmc_clip = cmc.copy()
        cmc_clip[cmc_clip > 1] = 1
        all_cmc.append(cmc_clip[:max_rank])
        num_valid_q += 1
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        per_q_ap[q_idx] = AP
    if num_valid_q == 0:
        return (float('nan'),
                np.full(max_rank, float('nan'), dtype=np.float32),
                float('nan'), per_q_ap)
    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    mAP = float(np.mean(all_AP))
    mINP = float(np.mean(all_INP))
    return mAP, all_cmc, mINP, per_q_ap


def cosdist(qf, gf):
    """Cosine distance matrix 2-2*cos, matching eval_market (feats already
    L2-normalized at eval, but renormalize defensively)."""
    qfn = F.normalize(qf, dim=1)
    gfn = F.normalize(gf, dim=1)
    return (2 - 2 * qfn @ gfn.t()).numpy()


def cossim(qf, gf):
    qfn = F.normalize(qf, dim=1)
    gfn = F.normalize(gf, dim=1)
    return (qfn @ gfn.t()).numpy()


def per_query_top1_conf(sim, q_pids, q_camids, g_pids, g_camids):
    """top-1 cosine confidence after eval_market junk removal + rank-1 correct."""
    conf, correct = [], []
    for i in range(sim.shape[0]):
        s = sim[i].copy()
        junk = (g_pids == q_pids[i]) & (g_camids == q_camids[i])
        s[junk] = -1e9
        j = int(np.argmax(s))
        conf.append(float(s[j]))
        correct.append(int(g_pids[j] == q_pids[i]))
    return np.array(conf), np.array(correct)


def native_area_hw(path):
    try:
        with Image.open(path) as im:
            w, h = im.size
        return h * w, h, w
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# data loading helpers
# --------------------------------------------------------------------------- #
def make_loader(samples, tf, batch, workers):
    return DataLoader(CARGOImageDataset(samples, tf), batch_size=batch,
                      shuffle=False, num_workers=workers, pin_memory=True)


def areas_for(paths):
    out = []
    for p in paths:
        r = native_area_hw(p)
        out.append(np.nan if r is None else r[0])
    return np.array(out, float)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--airl_ckpt', required=True)
    ap.add_argument('--backbone', default='swin_small')
    ap.add_argument('--swin_pretrain', default='')
    ap.add_argument('--data_root', default='/home/afr/SOLIDER-REID/data')
    ap.add_argument('--img_size', type=int, nargs=2, default=[256, 128])
    ap.add_argument('--num_classes', type=int, default=2500)
    ap.add_argument('--test_batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    assert torch.cuda.is_available(), 'need CUDA'
    device = 'cuda'
    img_size = tuple(args.img_size)
    np.set_printoptions(precision=3, suppress=True)

    print('=' * 80)
    print('AIRL fusion -- ZERO-TRAINING oracle kill-switch')
    print(f'  baseline ckpt = {args.base_ckpt}')
    print(f'  AIRL     ckpt = {args.airl_ckpt}')
    print('=' * 80)

    # ---- data splits ----
    dataset = CARGO(root=args.data_root, verbose=True)
    q_aerial = filter_by_view(dataset.query, 'Aerial')
    q_ground = filter_by_view(dataset.query, 'Ground')
    g_aerial = filter_by_view(dataset.gallery, 'Aerial')
    g_ground = filter_by_view(dataset.gallery, 'Ground')
    tr_aerial = filter_by_view(dataset.train, 'Aerial')
    tr_ground = filter_by_view(dataset.train, 'Ground')
    print(f'  q_aerial={len(q_aerial)} q_ground={len(q_ground)} '
          f'g_aerial={len(g_aerial)} g_ground={len(g_ground)} '
          f'tr_aerial={len(tr_aerial)} tr_ground={len(tr_ground)}')

    tf = build_transforms(is_train=False, img_size=img_size)

    def L(s):
        return make_loader(s, tf, args.test_batch, args.workers)

    # ---- two models ----
    base = load_model(args.base_ckpt, args.backbone, args.swin_pretrain,
                      img_size, args.num_classes, device, tag='baseline')
    airl = load_model(args.airl_ckpt, args.backbone, args.swin_pretrain,
                      img_size, args.num_classes, device, tag='AIRL')

    # ---- extract eval features for every (model, split) ----
    feats = {}
    meta = {}
    splits = {'qa': q_aerial, 'qg': q_ground, 'ga': g_aerial, 'gg': g_ground,
              'tra': tr_aerial, 'trg': tr_ground}
    for name, model in (('base', base), ('airl', airl)):
        for sk, samples in splits.items():
            f, p, c, paths = extract(model, L(samples), device)
            feats[(name, sk)] = f
            if name == 'base':
                meta[sk] = (p, c, paths)
        print(f'  [extracted all splits for {name}]')

    # sanity: pids/camids identical across models (same loader order)
    # (we only stored meta from base; loaders are shuffle=False so order matches)

    # native areas (model-independent)
    area = {sk: areas_for(meta[sk][2]) for sk in ('qa', 'qg')}

    # ============================================================= #
    # FULL per-direction mAP for each model (sanity vs doc numbers)
    # ============================================================= #
    # direction A->G : query=aerial, gallery=ground
    # direction G->A : query=ground, gallery=aerial
    DIR = {
        'A->G': dict(qk='qa', gk='gg'),
        'G->A': dict(qk='qg', gk='ga'),
    }

    print('\n' + '=' * 80)
    print('FULL per-direction mAP per model (sanity: should match doc)')
    print('  doc: base A->G 58.75 / G->A 62.93 ;  AIRL A->G 61.90 / G->A 59.75')
    print('=' * 80)
    full = {}   # full[(model,dir)] = (mAP, distmat, per_q_ap, qpid, qcam, gpid, gcam)
    for d, cfg in DIR.items():
        qk, gk = cfg['qk'], cfg['gk']
        qp, qc, _ = meta[qk]
        gp, gc, _ = meta[gk]
        for m in ('base', 'airl'):
            dm = cosdist(feats[(m, qk)], feats[(m, gk)])
            mp, cmc, minp, pap = mAP_from_distmat(dm, qp, qc, gp, gc)
            full[(m, d)] = dict(mAP=mp * 100, distmat=dm, per_q_ap=pap,
                                qp=qp, qc=qc, gp=gp, gc=gc, qk=qk, gk=gk)
            print(f'  {m:5s} {d}: mAP={mp*100:6.2f}  R1={cmc[0]*100:6.2f}  '
                  f'mINP={minp*100:6.2f}  (Nq={len(qp)} Ng={len(gp)})')

    def mean_of(ag, ga):
        return 0.5 * (ag + ga)

    base_mean = mean_of(full[('base', 'A->G')]['mAP'], full[('base', 'G->A')]['mAP'])
    airl_mean = mean_of(full[('airl', 'A->G')]['mAP'], full[('airl', 'G->A')]['mAP'])
    print('-' * 80)
    print(f'  baseline mean = {base_mean:.2f}   AIRL mean = {airl_mean:.2f}')
    print(f'  (baseline is the reference for all Delta below)')
    BASE = base_mean

    # ============================================================= #
    # GATE 1: view / direction gate (legal upper bound)
    #   A->G uses AIRL, G->A uses baseline
    # ============================================================= #
    print('\n' + '=' * 80)
    print('GATE 1 -- view/direction gate (legal upper bound)')
    print('  A->G -> AIRL branch ; G->A -> baseline branch (query view is known)')
    print('=' * 80)
    view_gate_mean = mean_of(full[('airl', 'A->G')]['mAP'],
                             full[('base', 'G->A')]['mAP'])
    print(f'  mean = {view_gate_mean:.2f}   Delta vs baseline = '
          f'{view_gate_mean - BASE:+.2f}')
    # also the complementary (wrong) routing, for completeness
    view_gate_alt = mean_of(full[('base', 'A->G')]['mAP'],
                            full[('airl', 'G->A')]['mAP'])
    print(f'  [complementary routing mean = {view_gate_alt:.2f} '
          f'({view_gate_alt - BASE:+.2f})]')

    # ============================================================= #
    # helper: per-query ROW-routed distmat mAP for a direction.
    #   route_mask[i]=True -> use AIRL row for query i, else baseline row.
    # ============================================================= #
    def routed_mAP(d, route_to_airl):
        info_b = full[('base', d)]
        info_a = full[('airl', d)]
        dm = info_b['distmat'].copy()
        dm[route_to_airl] = info_a['distmat'][route_to_airl]
        mp, cmc, minp, pap = mAP_from_distmat(
            dm, info_b['qp'], info_b['qc'], info_b['gp'], info_b['gc'])
        return mp * 100, pap

    # ============================================================= #
    # GATE 2: area-threshold gate (single-model-approximable)
    #   route each query by NATIVE bbox area; low area -> AIRL.
    #   threshold = TRAIN-split area quantile of that direction's query view.
    # ============================================================= #
    print('\n' + '=' * 80)
    print('GATE 2 -- area-threshold gate (low area -> AIRL ; threshold = TRAIN '
          'quantile, NOT test-tuned)')
    print('=' * 80)
    tr_area = {'A->G': areas_for([d['img_path'] for d in tr_aerial]),
               'G->A': areas_for([d['img_path'] for d in tr_ground])}
    for d in ('A->G',):  # area gate is meaningful when query=aerial
        print(f'  [train aerial area: min={np.nanmin(tr_area[d]):.0f} '
              f'med={np.nanmedian(tr_area[d]):.0f} max={np.nanmax(tr_area[d]):.0f}]')
    for d in ('G->A',):
        print(f'  [train ground area: min={np.nanmin(tr_area[d]):.0f} '
              f'med={np.nanmedian(tr_area[d]):.0f} max={np.nanmax(tr_area[d]):.0f}]')

    for frac in (0.25, 0.5, 0.75):
        ag_mp, _ = None, None
        means = {}
        for d in ('A->G', 'G->A'):
            qk = DIR[d]['qk']
            thr = np.nanquantile(tr_area[d], frac)
            qa = area[qk]
            route = qa <= thr            # low area -> AIRL
            route = np.where(np.isnan(qa), False, route)  # unreadable -> baseline
            mp, _ = routed_mAP(d, route)
            means[d] = mp
            n_air = int(route.sum())
            print(f'    frac={frac:.2f} {d}: thr_area={thr:8.0f} '
                  f'#->AIRL={n_air:3d}/{len(qa)}  mAP={mp:6.2f}')
        m = mean_of(means['A->G'], means['G->A'])
        print(f'    frac={frac:.2f} MEAN={m:.2f}  Delta={m - BASE:+.2f}')

    # area-gate oracle ceiling: best single area threshold on test (context only)
    print('  [context only -- TEST-tuned area threshold ceiling (upper, not legal)]')
    best_area_mean, best_frac = -1, None
    for frac in np.linspace(0.05, 0.95, 19):
        means = {}
        for d in ('A->G', 'G->A'):
            qk = DIR[d]['qk']
            thr = np.nanquantile(area[qk], frac)   # test quantile (peeking)
            qa = area[qk]
            route = np.where(np.isnan(qa), False, qa <= thr)
            mp, _ = routed_mAP(d, route)
            means[d] = mp
        m = mean_of(means['A->G'], means['G->A'])
        if m > best_area_mean:
            best_area_mean, best_frac = m, frac
    print(f'    best test-tuned area frac={best_frac:.2f} -> MEAN={best_area_mean:.2f} '
          f'(Delta={best_area_mean - BASE:+.2f})  [CEILING, not a legal gate]')

    # ============================================================= #
    # GATE 3: reliability gate (low baseline confidence -> AIRL)
    #   threshold = TRAIN confidence quantile (train images of the query view
    #   scored vs the TEST gallery of the opposite view; train ids disjoint from
    #   test ids -> gallery scale is a legal calibration source, no test labels).
    # ============================================================= #
    print('\n' + '=' * 80)
    print('GATE 3 -- reliability gate (low baseline top-1 cos -> AIRL ; threshold '
          '= TRAIN-confidence quantile)')
    print('=' * 80)
    # test-query confidences under baseline (used for routing value)
    conf_test = {}
    for d in ('A->G', 'G->A'):
        qk, gk = DIR[d]['qk'], DIR[d]['gk']
        qp, qc, _ = meta[qk]
        gp, gc, _ = meta[gk]
        sim = cossim(feats[('base', qk)], feats[('base', gk)])
        cf, corr = per_query_top1_conf(sim, qp, qc, gp, gc)
        conf_test[d] = cf
        print(f'  [{d}] baseline test top1-conf: '
              f'min={cf.min():.3f} med={np.median(cf):.3f} max={cf.max():.3f}  '
              f'AUROC(conf->R1)={_auroc(cf, corr):.3f}')
    # train confidences (calibration): train query-view images vs test gallery
    conf_train = {}
    for d in ('A->G', 'G->A'):
        gk = DIR[d]['gk']
        trk = 'tra' if DIR[d]['qk'] == 'qa' else 'trg'
        trp, trc, _ = (meta[trk] if trk in meta else (None, None, None))
        gp, gc, _ = meta[gk]
        sim = cossim(feats[('base', trk)], feats[('base', gk)])
        # train ids are disjoint from gallery(test) ids -> every match is a
        # non-self distractor; we only use the *distribution of top-1 cos* as a
        # scale, not correctness. junk removal by (pid,camid) won't fire across
        # disjoint-id sets, which is fine.
        cf, _ = per_query_top1_conf(sim, trp, trc, gp, gc)
        conf_train[d] = cf
        print(f'  [{d}] TRAIN top1-conf (calibration): '
              f'min={cf.min():.3f} med={np.median(cf):.3f} max={cf.max():.3f}')

    for frac in (0.25, 0.5, 0.75):
        means = {}
        for d in ('A->G', 'G->A'):
            thr = np.quantile(conf_train[d], frac)
            route = conf_test[d] <= thr     # low confidence -> AIRL
            mp, _ = routed_mAP(d, route)
            means[d] = mp
            print(f'    frac={frac:.2f} {d}: thr_conf={thr:.3f} '
                  f'#->AIRL={int(route.sum()):3d}/{len(route)}  mAP={mp:6.2f}')
        m = mean_of(means['A->G'], means['G->A'])
        print(f'    frac={frac:.2f} MEAN={m:.2f}  Delta={m - BASE:+.2f}')

    # reliability-gate test-tuned ceiling (context only)
    print('  [context only -- TEST-tuned confidence threshold ceiling]')
    best_rel_mean, best_rfrac = -1, None
    for frac in np.linspace(0.05, 0.95, 19):
        means = {}
        for d in ('A->G', 'G->A'):
            thr = np.quantile(conf_test[d], frac)   # test quantile (peeking)
            route = conf_test[d] <= thr
            mp, _ = routed_mAP(d, route)
            means[d] = mp
        m = mean_of(means['A->G'], means['G->A'])
        if m > best_rel_mean:
            best_rel_mean, best_rfrac = m, frac
    print(f'    best test-tuned conf frac={best_rfrac:.2f} -> MEAN={best_rel_mean:.2f} '
          f'(Delta={best_rel_mean - BASE:+.2f})  [CEILING, not a legal gate]')

    # ---- GATE 3b: confidence-DIFFERENCE routing (fully legal: no labels, no
    #      threshold). route query i to AIRL iff AIRL's top-1 cos > baseline's.
    #      Both models exist at test time; confidence needs no labels. ----
    print('  [GATE 3b -- legal: route to AIRL iff conf_AIRL > conf_base '
          '(no label, no threshold)]')
    confdiff_means = {}
    for d in ('A->G', 'G->A'):
        qk, gk = DIR[d]['qk'], DIR[d]['gk']
        qp, qc, _ = meta[qk]
        gp, gc, _ = meta[gk]
        sim_b = cossim(feats[('base', qk)], feats[('base', gk)])
        sim_a = cossim(feats[('airl', qk)], feats[('airl', gk)])
        cf_b, _ = per_query_top1_conf(sim_b, qp, qc, gp, gc)
        cf_a, _ = per_query_top1_conf(sim_a, qp, qc, gp, gc)
        route = cf_a > cf_b
        mp, _ = routed_mAP(d, route)
        confdiff_means[d] = mp
        print(f'    {d}: #->AIRL(conf_a>conf_b)={int(route.sum()):3d}/{len(route)} '
              f' mAP={mp:6.2f}')
    confdiff_mean = mean_of(confdiff_means['A->G'], confdiff_means['G->A'])
    print(f'    GATE 3b MEAN={confdiff_mean:.2f}  Delta={confdiff_mean - BASE:+.2f}'
          f'  [LEGAL fixed gate]')

    # ============================================================= #
    # GATE 4: per-query oracle (theoretical upper bound)
    #   per query take the better branch's AP -> mean
    # ============================================================= #
    print('\n' + '=' * 80)
    print('GATE 4 -- per-query ORACLE (theoretical upper bound; pick better branch '
          'per query)')
    print('=' * 80)
    oracle_means = {}
    for d in ('A->G', 'G->A'):
        ap_b = full[('base', d)]['per_q_ap']
        ap_a = full[('airl', d)]['per_q_ap']
        # only queries valid in eval (non-nan in both share the same valid set,
        # since valid set is determined by gallery GT presence, model-independent)
        valid = ~np.isnan(ap_b) & ~np.isnan(ap_a)
        best = np.where(ap_a[valid] > ap_b[valid], ap_a[valid], ap_b[valid])
        # mAP over valid queries (eval_market averages AP over valid queries)
        oracle_means[d] = float(best.mean()) * 100
        # report how often AIRL wins / ties / loses per query
        win = int((ap_a[valid] > ap_b[valid] + 1e-9).sum())
        tie = int((np.abs(ap_a[valid] - ap_b[valid]) <= 1e-9).sum())
        lose = int((ap_a[valid] < ap_b[valid] - 1e-9).sum())
        print(f'  {d}: oracle mAP={oracle_means[d]:6.2f}  '
              f'(AIRL win/tie/lose per-q = {win}/{tie}/{lose} of {int(valid.sum())})')
    oracle_mean = mean_of(oracle_means['A->G'], oracle_means['G->A'])
    print(f'  per-query ORACLE MEAN = {oracle_mean:.2f}  Delta={oracle_mean - BASE:+.2f}')

    # ============================================================= #
    # GATE 5: score fusion (soft): cos = w*cos_AIRL + (1-w)*cos_base
    # ============================================================= #
    print('\n' + '=' * 80)
    print('GATE 5 -- score fusion (soft): cos = w*cos_AIRL + (1-w)*cos_base, '
          'same w both directions')
    print('=' * 80)
    best_fuse_mean, best_w = -1, None
    for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        means = {}
        for d in ('A->G', 'G->A'):
            qk, gk = DIR[d]['qk'], DIR[d]['gk']
            qp, qc, _ = meta[qk]
            gp, gc, _ = meta[gk]
            s_a = cossim(feats[('airl', qk)], feats[('airl', gk)])
            s_b = cossim(feats[('base', qk)], feats[('base', gk)])
            dm = 2 - 2 * (w * s_a + (1 - w) * s_b)
            mp, _, _, _ = mAP_from_distmat(dm, qp, qc, gp, gc)
            means[d] = mp * 100
        m = mean_of(means['A->G'], means['G->A'])
        tag = ''
        if m > best_fuse_mean:
            best_fuse_mean, best_w = m, w
        print(f'  w={w:.2f}: A->G={means["A->G"]:6.2f} G->A={means["G->A"]:6.2f}  '
              f'MEAN={m:.2f}  Delta={m - BASE:+.2f}')
    print(f'  best score-fusion w={best_w:.2f} MEAN={best_fuse_mean:.2f} '
          f'(Delta={best_fuse_mean - BASE:+.2f})  [w swept on test = mild ceiling]')

    # ============================================================= #
    # VERDICT
    # ============================================================= #
    print('\n' + '=' * 80)
    print('SUMMARY (Delta vs baseline mean = %.2f)' % BASE)
    print('=' * 80)
    print(f'  view/direction gate (legal UB)      : {view_gate_mean - BASE:+.2f}')
    print(f'  per-query oracle (theoretical UB)   : {oracle_mean - BASE:+.2f}')
    print(f'  best score-fusion (w on test)       : {best_fuse_mean - BASE:+.2f}')
    print(f'  area-gate ceiling (thr on test)     : {best_area_mean - BASE:+.2f}')
    print(f'  reliability-gate ceiling (thr on test): {best_rel_mean - BASE:+.2f}')
    print(f'  LEGAL conf-diff gate (3b, no thr)   : {confdiff_mean - BASE:+.2f}')
    print('  (other legal fixed-gate Deltas are the train-quantile rows above)')
    print('=' * 80)


def _auroc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    avg = {}
    start = 0
    for i, c in enumerate(cnt):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


if __name__ == '__main__':
    main()
