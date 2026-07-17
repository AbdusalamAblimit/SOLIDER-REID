"""Smoke test for the AG-ReID.v2 loader (cargo_cvpb/agreid_dataset.py).

Verifies the CARGO-compatible meta-dataset loads, parses identities/cameras/views
correctly, and that the binary aerial<->ground cross-view splits + the official
per-camera protocol are both populated. Also reads one real image (PIL + torch)
to prove the file path / decode pipeline works.

Run (system python lacks deps, use an ephemeral uv env):
    cd experiments/cargo_cvpb
    uv run --no-project --with numpy --with torch --with pillow python agreid_smoke.py
Optionally pass a data root (default /tmp/agreidv2):
    ... python agreid_smoke.py /path/to/datroot
"""
import os
import sys

import numpy as np
from PIL import Image
import torch

# make `import agreid_dataset` work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agreid_dataset import AGReIDv2, filter_by_view  # noqa: E402


def show_samples(name, split, k=3):
    print(f"  [{name}] {len(split)} imgs; first {k} samples:")
    for s in split[:k]:
        print(f"    pid={s['pid']:>4d} camid={s['camid']} view={s['view']:6s} "
              f"path=.../{os.path.basename(os.path.dirname(s['img_path']))}/"
              f"{os.path.basename(s['img_path'])}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '/tmp/agreidv2'
    print(f"=== AG-ReID.v2 smoke test (root={root}) ===\n")

    ds = AGReIDv2(root=root, verbose=True)

    # ----- top-line counts -------------------------------------------------- #
    print("\n--- counts ---")
    print(f"  #train_pids = {ds.num_train_pids}   #train_imgs = {ds.num_train_imgs}"
          f"   #train_cams = {ds.num_train_cams}")
    print(f"  #query_imgs = {len(ds.query)}   #query_pids = {ds.num_query_pids}")
    print(f"  #gallery_imgs = {len(ds.gallery)}   #gallery_pids = {ds.num_gallery_pids}")

    # ----- aerial / ground split per subset --------------------------------- #
    print("\n--- aerial vs ground per split ---")
    for nm, split in (('train', ds.train), ('query', ds.query),
                      ('gallery', ds.gallery)):
        a = len(filter_by_view(split, 'Aerial'))
        g = len(filter_by_view(split, 'Ground'))
        print(f"  {nm:8s}: aerial={a:6d}  ground={g:6d}")

    # ----- CARGO-style binary cross-view directions ------------------------- #
    qa = filter_by_view(ds.query, 'Aerial')
    qg = filter_by_view(ds.query, 'Ground')
    ga = filter_by_view(ds.gallery, 'Aerial')
    gg = filter_by_view(ds.gallery, 'Ground')
    print("\n--- binary cross-view eval sets (what afd_train run_cross_view_eval uses) ---")
    print(f"  A->G : query(aerial)={len(qa):5d}  gallery(ground)={len(gg):6d}")
    print(f"  G->A : query(ground)={len(qg):5d}  gallery(aerial)={len(ga):6d}")
    # cross-view feasibility: query ids that actually have a match in target view
    def ids(split):
        return {s['pid'] for s in split}
    ag_matchable = len(ids(qa) & ids(gg))
    ga_matchable = len(ids(qg) & ids(ga))
    print(f"  A->G matchable query ids (have a ground gallery): {ag_matchable}/{len(ids(qa))}")
    print(f"  G->A matchable query ids (have an aerial gallery): {ga_matchable}/{len(ids(qg))}")

    # ----- sample dumps ----------------------------------------------------- #
    print("\n--- sample parses ---")
    show_samples('train', ds.train)
    show_samples('query', ds.query)
    show_samples('gallery', ds.gallery)

    # ----- official per-camera protocol (optional path) --------------------- #
    print("\n--- official per-camera protocol (exp txt) ---")
    for exp in ('exp1_aerial_to_cctv.txt', 'exp4_cctv_to_aerial.txt'):
        try:
            q, g = ds.official_query_gallery(exp)
            qc = sorted({s['camid'] for s in q})
            gc = sorted({s['camid'] for s in g})
            print(f"  {exp:26s}: query={len(q):5d} (cams {qc})  "
                  f"gallery={len(g):6d} (cams {gc})")
        except FileNotFoundError as e:
            print(f"  {exp:26s}: (not found: {e})")

    # ----- prove image decode works (PIL + torch, no torchvision) ----------- #
    print("\n--- image decode check ---")
    s0 = ds.query[0]
    img = Image.open(s0['img_path']).convert('RGB')
    arr = np.asarray(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    print(f"  decoded {os.path.basename(s0['img_path'])} -> PIL size={img.size} "
          f"tensor shape={tuple(t.shape)} dtype={t.dtype} "
          f"min={t.min():.3f} max={t.max():.3f}")

    # ----- sanity asserts --------------------------------------------------- #
    assert ds.num_train_pids == 807, f"expected 807 train pids, got {ds.num_train_pids}"
    assert ds.num_query_pids == 808, f"expected 808 test pids, got {ds.num_query_pids}"
    assert len(qa) > 0 and len(gg) > 0, "A->G split is empty"
    assert len(qg) > 0 and len(ga) > 0, "G->A split is empty"
    assert all(s['view'] in ('Aerial', 'Ground') for s in ds.query), "bad view label"
    assert t.shape[0] == 3, "decoded tensor is not 3-channel"
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == '__main__':
    main()
