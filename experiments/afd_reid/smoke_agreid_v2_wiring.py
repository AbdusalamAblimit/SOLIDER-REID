# encoding: utf-8
"""
CPU-only wiring smoke for the --dataset agreid_v2 hookup.

Does NOT import cargo_cvpb/afd_train.py (that module is import-safe ONLY when run
as the main script from cargo_cvpb/, due to its `from afd_train import ...` shim).
The end-to-end integration test is the real training launch itself (its first
eval must print the official 2356/6347 & 1811/14362 directions). Here we cheaply
verify, WITHOUT touching the GPU (so it runs alongside a live training job):

  1. AGReIDV2Combined builds and filter_by_view recovers the official per-direction
     subsets exactly (the contract the cvpb selection branch relies on).
  2. The selection logic (replicated verbatim from afd_train.main) routes
     cargo/agreid/agreid_v2 to CARGO / AGReIDv2 / AGReIDV2Combined.
  3. A batch collates with INT pids so extract_features' torch.cat(pids).numpy()
     cannot blow up on string pids.
  4. eval_market (the shared afd_reid helper the cvpb eval reuses) is internally
     consistent on a tiny CPU tensor.

Run (data must be present):
    python3 <repo>/experiments/afd_reid/smoke_agreid_v2_wiring.py /root/work/SOLIDER-REID/data
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # afd_reid only -- no cargo_cvpb on the path

ROOT = sys.argv[1] if len(sys.argv) > 1 else '/root/work/SOLIDER-REID/data'


def section(t):
    print('\n' + '=' * 64 + f'\n{t}\n' + '=' * 64)


# imports (afd_reid side only) ------------------------------------------------ #
section('imports (afd_reid dataset adapters + eval helper)')
from cargo_dataset import CARGO, filter_by_view  # noqa: E402
from agreid_v2_combined import AGReIDV2Combined  # noqa: E402
import afd_train  # noqa: E402  (afd_reid's self-contained helper module; safe here)
# legacy binary-merge AGReIDv2 lives in cargo_cvpb/ (off the afd_reid path); the
# 'agreid' (legacy) branch is unchanged and not under test here.
sys.path.insert(0, os.path.join(HERE, '..', 'cargo_cvpb'))
from agreid_dataset import AGReIDv2  # noqa: E402
print('imported CARGO / AGReIDv2 / AGReIDV2Combined / afd_train(helpers) OK')


# 1+2. selection branch routes correctly ------------------------------------- #
section('1+2. --dataset selection branch (verbatim from afd_train.main)')


def select(dataset_name):
    if dataset_name == 'cargo':
        return CARGO(root=ROOT, verbose=False)
    elif dataset_name == 'agreid_v2':
        return AGReIDV2Combined(root=ROOT, verbose=False)
    else:
        return AGReIDv2(root=ROOT, verbose=False)


ds_v2 = select('agreid_v2')
assert isinstance(ds_v2, AGReIDV2Combined), 'agreid_v2 did not select Combined'
print('  agreid_v2 -> AGReIDV2Combined  OK')
ds_cargo = select('cargo')
assert isinstance(ds_cargo, CARGO), 'cargo default broke'
print('  cargo     -> CARGO            OK (default path intact)')
print(f'  combined num_train_pids={ds_v2.num_train_pids} '
      f'num_test_pids={len(ds_v2.test_pid2label)}')


# 3. official per-direction subset recovery ---------------------------------- #
section('3. official per-direction subset recovery (filter_by_view)')
qa = filter_by_view(ds_v2.query, 'Aerial')
gg = filter_by_view(ds_v2.gallery, 'Ground')
qg = filter_by_view(ds_v2.query, 'Ground')
ga = filter_by_view(ds_v2.gallery, 'Aerial')
print(f'  A->G  q_aerial={len(qa)}  g_ground={len(gg)}   (expect 2356 / 6347)')
print(f'  G->A  q_ground={len(qg)}  g_aerial={len(ga)}   (expect 1811 / 14362)')
assert (len(qa), len(gg)) == (2356, 6347), 'A->G (exp1) subset wrong'
assert (len(qg), len(ga)) == (1811, 14362), 'G->A (exp4) subset wrong'
assert {d['camid'] for d in qa} == {0}, 'A->G query not all UAV(0)'
assert {d['camid'] for d in gg} == {3}, 'A->G gallery not all CCTV(3)'
assert {d['camid'] for d in qg} == {3}, 'G->A query not all CCTV(3)'
assert {d['camid'] for d in ga} == {0}, 'G->A gallery not all UAV(0)'
print('  per-direction camids single-platform (0 vs 3) -> junk-removal no-op OK')
for nm, sp in (('query', ds_v2.query), ('gallery', ds_v2.gallery),
               ('train', ds_v2.train)):
    bad = [d for d in sp if not isinstance(d['pid'], int)]
    assert not bad, f'{nm} has non-int pid e.g. {bad[0]["pid"]!r}'
print('  all train/query/gallery pids are int                OK')
qa_pids = {d['pid'] for d in qa}
gg_pids = {d['pid'] for d in gg}
print(f'  A->G shared pids query/gallery: {len(qa_pids & gg_pids)} / '
      f'{len(qa_pids)} query ids')
assert len(qa_pids & gg_pids) > 0, 'A->G zero matchable identities -> bad pid map'
qg_pids = {d['pid'] for d in qg}
ga_pids = {d['pid'] for d in ga}
assert len(qg_pids & ga_pids) > 0, 'G->A zero matchable identities -> bad pid map'


# 4. collate INT pids + eval_market sanity (CPU) ----------------------------- #
section('4. collate INT pids + eval_market sanity (CPU, no GPU)')
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from cargo_dataset import CARGOImageDataset, build_transforms  # noqa: E402

tf = build_transforms(is_train=False, img_size=(256, 128))
loader = DataLoader(CARGOImageDataset(qa[:6], tf), batch_size=4, shuffle=False,
                    num_workers=0)
batch = next(iter(loader))
assert torch.is_tensor(batch['pid']), 'pid did not collate to a tensor'
assert batch['pid'].dtype in (torch.int64, torch.int32), \
    f'pid tensor dtype {batch["pid"].dtype} not int'
_ = torch.cat([batch['pid'], batch['pid']], 0).numpy()
print('  batch["pid"] collates to int tensor; torch.cat(...).numpy() OK')

import numpy as np  # noqa: E402
qf = torch.eye(3)
gf = torch.cat([torch.eye(3), torch.eye(3)], 0)
q_pids = np.array([0, 1, 2])
g_pids = np.array([0, 1, 2, 0, 1, 2])
q_cam = np.array([0, 0, 0])
g_cam = np.array([3, 3, 3, 3, 3, 3])
mAP, cmc, mINP = afd_train.eval_market(qf, q_pids, q_cam, gf, g_pids, g_cam)
print(f'  eval_market synthetic: mAP={mAP*100:.1f} R1={cmc[0]*100:.1f} '
      f'mINP={mINP*100:.1f}  (expect ~100 / 100 / 100)')
assert mAP > 0.99 and cmc[0] > 0.99, 'eval_market broke on separable case'

print('\nALL WIRING SMOKE CHECKS PASSED')
