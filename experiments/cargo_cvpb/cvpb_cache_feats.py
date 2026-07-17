#!/usr/bin/env python3
"""One-time cache of K-lattice-variant features (frozen no-LM-loss backbone) for train/query +
HR gallery, so the train-side probes (LCRS/LRFD/DeepSets/LS-MRT) reuse them instead of each
re-extracting (the CPU lattice generation is the bottleneck). Saves to one .npz."""
import sys, os, numpy as np, argparse, time
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='log/market1501/exp359_abl_noLMloss/transformer_40.pth')
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--h', type=int, default=16)
ap.add_argument('--K', type=int, default=9)
ap.add_argument('--out', default='/tmp/lmreid_feats_h16.npz')
cli = ap.parse_args()
sys.argv = ['ks', '--ckpt', cli.ckpt, '--config', cli.config, '--data_root', 'data', '--K', str(cli.K)]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpb_lattice_killswitch as ks
from datasets.bases import read_image
RNG = np.random.RandomState(42); H, K = cli.h, cli.K
ext = ks.FrozenExtractor(); REPO = ks._repo


def items(s):
    return ks.list_split(os.path.join(REPO, 'data', 'market1501', s))


def vfeats(its):
    flat = []
    for it in its:
        flat.extend(ks.make_lattice_variants(ks._to_target_aspect(read_image(it[0])), H, K, RNG))
    return ext.feats_from_pil(flat).reshape(len(its), K, -1).astype(np.float32)


t0 = time.time()
tr = items('bounding_box_train'); q = items('query'); g = items('bounding_box_test')
ft = vfeats(tr); print(f'[cache] train {ft.shape} ({time.time()-t0:.0f}s)', flush=True)
fq = vfeats(q); print(f'[cache] query {fq.shape} ({time.time()-t0:.0f}s)', flush=True)
gf = ext.feats_from_pil([ks._to_target_aspect(read_image(it[0])) for it in g])
np.savez(cli.out,
         ft=ft, ytr=np.array([it[1] for it in tr]),
         fq=fq, q_pid=np.array([it[1] for it in q]), q_cam=np.array([it[2] for it in q]),
         gf=gf, g_pid=np.array([it[1] for it in g]), g_cam=np.array([it[2] for it in g]))
print(f'[done] cached feats to {cli.out}  ({time.time()-t0:.0f}s)', flush=True)
