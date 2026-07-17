# encoding: utf-8
"""
Verify the AG-ReID.v2 adapter loads correctly (NO training).

Checks, per protocol:
  - #query / #gallery images
  - #unique query/gallery pid (must use FULL folder name, not P prefix)
  - query/gallery overlap (every query id must have a match in gallery)
  - platform (C-field) purity per side
  - CARGO-compat: filter_by_view + eval_market shape sanity on random feats

Official spec to match (from the paper / protocol txt):
  exp1 aerial_to_cctv : 2356 query / 6347 gallery
  exp4 cctv_to_aerial : 1811 query / 14362 gallery
  test identities (full-folder pid): 808 total across the test set.

Run on lab-3090:
    cd /root/work/SOLIDER-REID/experiments/afd_reid
    python verify_agreid_v2.py --data_root /root/work/SOLIDER-REID/data
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agreid_v2_dataset import AGReIDV2
from cargo_dataset import filter_by_view
from afd_train import eval_market


# official (#query, #gallery) per protocol
OFFICIAL = {
    'exp1': (2356, 6347),
    'exp2': (2209, 12912),
    'exp4': (1811, 14362),
    'exp5': (2340, 12568),
}


def _pid_set(split):
    return {d['pid'] for d in split}


def _cfields(split):
    # map view back to a coarse platform label for printing
    return sorted({(d['camid'], d['view']) for d in split})


def check_protocol(root, proto):
    print("\n" + "=" * 72)
    print(f"PROTOCOL {proto}")
    print("=" * 72)
    ds = AGReIDV2(root=root, protocol=proto, verbose=True)

    nq, ng = len(ds.query), len(ds.gallery)
    qp, gp = _pid_set(ds.query), _pid_set(ds.gallery)
    overlap = qp & gp

    print(f"\n  #query images   : {nq}")
    print(f"  #gallery images : {ng}")
    print(f"  #unique q pid   : {len(qp)}")
    print(f"  #unique g pid   : {len(gp)}")
    print(f"  q∩g pids        : {len(overlap)}  "
          f"(queries with >=1 same-id gallery match)")
    print(f"  query platforms : {_cfields(ds.query)}")
    print(f"  gallery platforms: {_cfields(ds.gallery)}")

    ok = True
    if proto in OFFICIAL:
        eq, eg = OFFICIAL[proto]
        q_ok = (nq == eq)
        g_ok = (ng == eg)
        print(f"\n  [official check] query {nq} vs {eq}  -> {'OK' if q_ok else 'MISMATCH'}")
        print(f"  [official check] gallery {ng} vs {eg}  -> {'OK' if g_ok else 'MISMATCH'}")
        ok = ok and q_ok and g_ok

    # pid sanity: full-folder pids look like 'P####T####A#'
    sample_pid = next(iter(qp))
    looks_full = (isinstance(sample_pid, str)
                  and sample_pid.startswith('P')
                  and 'T' in sample_pid and sample_pid[-2] == 'A')
    print(f"  [pid check] sample query pid = {sample_pid!r} "
          f"-> {'full folder name (correct)' if looks_full else 'NOT full folder (WRONG)'}")
    ok = ok and looks_full

    # every query id must be findable in gallery (else mAP undefined for it)
    unmatched = qp - gp
    print(f"  [match check] query pids with NO gallery match: {len(unmatched)} "
          f"-> {'OK' if not unmatched else 'WARN'}")

    # platform purity (one platform per side for these directions)
    q_plat = {c for c, _ in _cfields(ds.query)}
    g_plat = {c for c, _ in _cfields(ds.gallery)}
    pure = (len(q_plat) == 1 and len(g_plat) == 1 and not (q_plat & g_plat))
    print(f"  [platform check] query={sorted(q_plat)} gallery={sorted(g_plat)} "
          f"-> {'cross-platform, disjoint (OK)' if pure else 'WARN'}")

    return ds, ok


def cargo_compat_smoke(ds):
    """Confirm filter_by_view + eval_market run on the adapter's lists (random feats)."""
    print("\n  ---- CARGO-compat smoke (random features, sanity only) ----")
    q_aer = filter_by_view(ds.query, 'Aerial')
    q_grd = filter_by_view(ds.query, 'Ground')
    g_aer = filter_by_view(ds.gallery, 'Aerial')
    g_grd = filter_by_view(ds.gallery, 'Ground')
    print(f"    filter_by_view -> q_aerial={len(q_aer)} q_ground={len(q_grd)} "
          f"g_aerial={len(g_aer)} g_ground={len(g_grd)}")

    # pick the non-empty direction (A->G uses aerial query + ground gallery, etc.)
    if q_aer and g_grd:
        q, g, tag = q_aer, g_grd, 'A->G'
    else:
        q, g, tag = q_grd, g_aer, 'G->A'

    rng = np.random.default_rng(0)
    # build int pid arrays consistent across q and g via a shared mapping
    all_pids = sorted({d['pid'] for d in q} | {d['pid'] for d in g})
    pmap = {p: i for i, p in enumerate(all_pids)}
    qp = np.array([pmap[d['pid']] for d in q], dtype=np.int64)
    gp = np.array([pmap[d['pid']] for d in g], dtype=np.int64)
    qc = np.array([d['camid'] for d in q], dtype=np.int64)
    gc = np.array([d['camid'] for d in g], dtype=np.int64)
    D = 64
    qf = torch.tensor(rng.standard_normal((len(q), D)), dtype=torch.float32)
    gf = torch.tensor(rng.standard_normal((len(g), D)), dtype=torch.float32)
    mAP, cmc, mINP = eval_market(qf, qp, qc, gf, gp, gc)
    print(f"    eval_market[{tag}] ran: random mAP={mAP*100:.2f} R1={cmc[0]*100:.2f} "
          f"(random baseline, just proves the pipeline runs end-to-end)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/root/work/SOLIDER-REID/data')
    ap.add_argument('--protocols', nargs='+', default=['exp1', 'exp4'],
                    help='which protocols to verify (main A<->G = exp1 exp4)')
    args = ap.parse_args()

    all_ok = True
    first_ds = None
    union_test_pids = set()
    for proto in args.protocols:
        ds, ok = check_protocol(args.data_root, proto)
        union_test_pids |= _pid_set(ds.query) | _pid_set(ds.gallery)
        all_ok = all_ok and ok
        if first_ds is None:
            first_ds = ds

    cargo_compat_smoke(first_ds)

    print("\n" + "=" * 72)
    print(f"UNION test identities across {args.protocols}: {len(union_test_pids)} "
          f"(official AG-ReID.v2 test = 808)")
    print(f"train identities (exp1 build): {first_ds.num_train_pids} "
          f"(official train = 807)")
    print("=" * 72)
    print("RESULT:", "ALL OFFICIAL CHECKS PASS" if all_ok else "SOME CHECKS FAILED")


if __name__ == '__main__':
    main()
