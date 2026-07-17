# encoding: utf-8
"""
AG-ReID.v2 (aerial-ground person ReID) meta-dataset — CARGO-compatible interface.

This is the SECOND benchmark for the OVLI method paper (cross-dataset main table).
It deliberately mirrors `cargo_dataset.CARGO` so that the OVLI training loop
(`cargo_cvpb/afd_train.py`) can switch datasets with a single `--dataset agreid`
flag and reuse EVERYTHING ELSE (CARGOImageDataset / build_transforms /
RandomIdentitySampler / filter_by_view / eval_market / run_cross_view_eval).

--------------------------------------------------------------------------- #
RESEARCH FINDINGS (verified against the data + the official README/exp protocol)
--------------------------------------------------------------------------- #
Source data layout (after unzip):
    <root>/AG-ReID.v2/train_all/<ID>/<img>.jpg     807 identities, 51530 imgs
    <root>/AG-ReID.v2/query/<ID>/<img>.jpg         808 identities,  8499 imgs
    <root>/AG-ReID.v2/gallery/<ID>/<img>.jpg       808 identities, 40473 imgs
    807 + 808 = 1615 identities total  (matches the paper's "1,615 identities")

Filename format (README example "P0006T0214A0C0F1831.jpg"):
    P<id> T<MMDD><sess> A<alt> C<cam> F<frame>.jpg
    e.g. P0087T04041A0 C3 F13441.jpg
      P0087        person index
      T04041       capture session  (T + MMDD=0404 + session 1)
      A0           UAV altitude band (0 low / 1 mid / 2 high)
      C3           CAMERA TYPE  <-- the platform tag we key 'view' off of
      F13441       frame number

  *** IDENTITY = the FULL folder name `P####T####A#`, NOT just the `P####`. ***
  The `P####` prefix is NOT unique: 649 distinct prefixes over the 808 test
  folders (the same physical person at a different time/altitude is a SEPARATE
  identity in AG-ReID.v2's protocol). The folder counts (807 train + 808 test =
  1615) match the paper exactly, so the folder name is the canonical id unit.
  query/ and gallery/ contain the SAME 808 test identity folders; each identity's
  images are split across the two dirs (cross-camera images live in BOTH).

Camera-type -> platform mapping (README + cross-checked against the exp*.txt
protocol files, where the query camera of each named direction matches):
    C0 = UAV / drone            -> AERIAL  (exp1/exp2 queries are all C0)
    C2 = wearable / smartglass  -> GROUND  (exp5 wearable_to_aerial query = C2)
    C3 = stationary CCTV        -> GROUND  (exp4 cctv_to_aerial   query = C3)
  (there is NO camera "C1" in the data; codes present are exactly C0/C2/C3.)

Official AG-ReID.v2 evaluation protocol (the `exp{N}_*.txt` files):
  Each txt holds the QUERY lines ("query/...") AND the GALLERY lines
  ("gallery/...") for one camera-pair direction, market-style (mAP + CMC,
  same-(pid,camid) junk removal). Verified line composition:
    exp1 aerial_to_cctv    : 2356 query C0  + 6347 gallery C3
    exp2 aerial_to_wearable: query C0        + gallery C2
    exp4 cctv_to_aerial    : 1811 query C3  + 14362 gallery C0
    exp5 wearable_to_aerial: query C2        + gallery C0
  Only the aerial-query side of each identity that HAS a target-camera match is
  kept, which is why exp1 uses 2356 of the 4348 C0 query images.

--------------------------------------------------------------------------- #
WHAT THIS LOADER DOES (default = CARGO-aligned binary aerial<->ground)
--------------------------------------------------------------------------- #
To align with CARGO's A<->G `is_aerial` binary (so the OVLI cross-view eval is
identical across both benchmarks), we MERGE the two ground platforms:
    view = 'Aerial' if camid in {0}      (UAV)
           'Ground' otherwise            (C2 wearable + C3 cctv)
We load query/ and gallery/ in FULL (all cameras, view-tagged) exactly like
CARGO, and the downstream `filter_by_view` does the A->G / G->A directions:
    A->G : query view==Aerial , gallery view==Ground
    G->A : query view==Ground , gallery view==Aerial
NOTE: this binary merge is intentionally DIFFERENT from the official per-camera
protocol, so numbers here are NOT directly comparable to published AG-ReID.v2
tables. For the official directions use `official_query_gallery(exp_file)`.

Field schema per sample (identical to cargo_dataset):
    {'img_path': str, 'pid': int, 'camid': int, 'view': 'Aerial'|'Ground'}
  - train pids relabeled to a contiguous [0, num_train_pids).
  - query/gallery pids share ONE folder-name->int map (disjoint from train),
    so a query identity and its gallery matches carry the same int pid.
  - camid kept as the raw camera code (0 / 2 / 3); only used by eval_market for
    same-(pid,camid) junk removal. In a cross-view direction query and gallery
    never share a camid, so nothing is wrongly removed.
"""
import os
import glob
import re


# camera-type code -> platform.  Override via AGReIDv2(aerial_cams=..., ...).
AERIAL_CAMS = (0,)          # C0 = UAV
GROUND_CAMS = (2, 3)        # C2 = wearable, C3 = CCTV

# P<digits> T<digits> A<digits>  C<cam>  F<frame>[suffix].jpg
# A few wearable (C2) frames carry a trailing letter flag, e.g. ...F8671Z.jpg
# (32 such 'Z' files, all real & referenced by the official exp2/exp5 protocol),
# so allow an optional alpha suffix after the frame digits.
_NAME_RE = re.compile(r'^(?P<pid>P\d+T\d+A\d+)C(?P<cam>\d+)F\d+[A-Za-z]*\.jpg$')


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_name(img_path, aerial_cams):
    """Return (pid_str, camid, view) parsed from an AG-ReID.v2 filename.

    pid_str is the full identity token `P####T####A#` (== the folder name).
    Returns None if the basename is not a parseable image (e.g. dotfiles,
    macOS AppleDouble `._*` resource forks, stray non-jpg).
    """
    fname = os.path.basename(img_path)
    if fname.startswith('.'):
        return None
    m = _NAME_RE.match(fname)
    if m is None:
        return None
    pid_str = m.group('pid')
    camid = int(m.group('cam'))
    view = 'Aerial' if camid in aerial_cams else 'Ground'
    return pid_str, camid, view


def _scan_split(split_dir, aerial_cams):
    """Glob <split_dir>/<ID>/*.jpg -> (list[(path,pid_str,camid,view)], n_skipped)."""
    img_paths = sorted(glob.glob(os.path.join(split_dir, '*', '*.jpg')))
    data, skipped = [], 0
    for p in img_paths:
        parsed = _parse_name(p, aerial_cams)
        if parsed is None:
            skipped += 1
            continue
        pid_str, camid, view = parsed
        data.append((p, pid_str, camid, view))
    return data, skipped


# --------------------------------------------------------------------------- #
# Meta-dataset
# --------------------------------------------------------------------------- #
class AGReIDv2(object):
    """AG-ReID.v2 meta-dataset, CARGO-compatible (binary aerial<->ground).

    Attributes (each a list of dicts {img_path, pid, camid, view}):
        self.train, self.query, self.gallery
    plus:
        self.num_train_pids / num_train_imgs / num_train_cams
        self.num_query_pids / num_gallery_pids  (= 808 test identities)
        self.pid2label       (train  folder-name -> [0, num_train_pids))
        self.test_pid2label  (test   folder-name -> [0, num_query_pids))
    """

    def __init__(self, root='/root/work/SOLIDER-REID/data', verbose=True,
                 aerial_cams=AERIAL_CAMS, ground_cams=GROUND_CAMS,
                 dataset_subdir='AG-ReID.v2'):
        self.aerial_cams = tuple(aerial_cams)
        self.ground_cams = tuple(ground_cams)

        self.dataset_dir = os.path.join(root, dataset_subdir)
        self.train_dir = os.path.join(self.dataset_dir, 'train_all')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'gallery')

        for d in (self.train_dir, self.query_dir, self.gallery_dir):
            if not os.path.isdir(d):
                raise RuntimeError(f"AG-ReID.v2 split dir not found: {d}")

        train_raw, sk_tr = _scan_split(self.train_dir, self.aerial_cams)
        query_raw, sk_q = _scan_split(self.query_dir, self.aerial_cams)
        gallery_raw, sk_g = _scan_split(self.gallery_dir, self.aerial_cams)
        self._n_skipped = sk_tr + sk_q + sk_g

        if not train_raw or not query_raw or not gallery_raw:
            raise RuntimeError(
                "AG-ReID.v2 produced an empty split "
                f"(train={len(train_raw)}, query={len(query_raw)}, "
                f"gallery={len(gallery_raw)}); check the data layout under "
                f"{self.dataset_dir}")

        # train: relabel folder-name identities to a contiguous range.
        train_names = sorted({pid for _, pid, _, _ in train_raw})
        self.pid2label = {name: idx for idx, name in enumerate(train_names)}

        # test: query & gallery share ONE id map (disjoint from train ids).
        test_names = sorted({pid for _, pid, _, _ in (query_raw + gallery_raw)})
        self.test_pid2label = {name: idx for idx, name in enumerate(test_names)}

        self.train = self._pack(train_raw, self.pid2label)
        self.query = self._pack(query_raw, self.test_pid2label)
        self.gallery = self._pack(gallery_raw, self.test_pid2label)

        self.num_train_pids = len(train_names)
        self.num_train_imgs = len(self.train)
        self.num_train_cams = len({d['camid'] for d in self.train})
        self.num_query_pids = len({d['pid'] for d in self.query})
        self.num_gallery_pids = len({d['pid'] for d in self.gallery})

        if verbose:
            self._print_stats()

    def _pack(self, raw, name2label):
        out = []
        for path, pid_str, camid, view in raw:
            out.append({
                'img_path': path,
                'pid': name2label[pid_str],
                'camid': camid,             # raw camera code (0 / 2 / 3)
                'view': view,
            })
        return out

    # ----------------------------------------------------------------------- #
    # Official per-camera protocol (optional; for reproducing published tables)
    # ----------------------------------------------------------------------- #
    def official_query_gallery(self, exp_file):
        """Load an official camera-pair direction from an `exp{N}_*.txt` file.

        The txt lists both 'query/...' and 'gallery/...' relative paths. Returns
        (query_list, gallery_list) in the same dict schema, using test_pid2label.
        Reproduces e.g. aerial->cctv (exp1) or cctv->aerial (exp4) exactly.

        `exp_file` may be an absolute path or a bare filename; bare names are
        resolved against the dataset dir, its parent, and root.
        """
        path = self._resolve_protocol_file(exp_file)
        query, gallery = [], []
        with open(path) as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                full = os.path.join(self.dataset_dir, rel)
                parsed = _parse_name(full, self.aerial_cams)
                if parsed is None:
                    continue
                pid_str, camid, view = parsed
                item = {
                    'img_path': full,
                    'pid': self.test_pid2label.get(pid_str, -1),
                    'camid': camid,
                    'view': view,
                }
                if rel.startswith('query/'):
                    query.append(item)
                elif rel.startswith('gallery/'):
                    gallery.append(item)
        return query, gallery

    def _resolve_protocol_file(self, exp_file):
        if os.path.isabs(exp_file) and os.path.isfile(exp_file):
            return exp_file
        cand_dirs = [
            self.dataset_dir,
            os.path.dirname(self.dataset_dir),       # <root>
            os.path.dirname(os.path.dirname(self.dataset_dir)),
        ]
        for d in cand_dirs:
            p = os.path.join(d, exp_file)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(
            f"protocol file '{exp_file}' not found near {self.dataset_dir}")

    # ----------------------------------------------------------------------- #
    def _print_stats(self):
        def cnt(split):
            pids = len({d['pid'] for d in split})
            cams = len({d['camid'] for d in split})
            a = sum(d['view'] == 'Aerial' for d in split)
            g = sum(d['view'] == 'Ground' for d in split)
            return len(split), pids, cams, a, g
        print("=> AG-ReID.v2 loaded (CARGO-aligned binary aerial<->ground)")
        print(f"   aerial cams={self.aerial_cams}  ground cams={self.ground_cams}"
              f"  (C0=UAV, C2=wearable, C3=CCTV)")
        if self._n_skipped:
            print(f"   (skipped {self._n_skipped} non-image / unparseable files)")
        print("  -----------------------------------------------------------")
        print("  subset   | # imgs | # pids | # cams | aerial | ground")
        print("  -----------------------------------------------------------")
        for name, split in (('train', self.train), ('query', self.query),
                            ('gallery', self.gallery)):
            n, p, c, a, g = cnt(split)
            print(f"  {name:8s} | {n:6d} | {p:6d} | {c:6d} | {a:6d} | {g:6d}")
        print("  -----------------------------------------------------------")


# --------------------------------------------------------------------------- #
# View filtering (native, torchvision-free) — same semantics as cargo_dataset.
# --------------------------------------------------------------------------- #
def filter_by_view(samples, view):
    """Return subset whose 'view' == view ('Aerial' or 'Ground')."""
    return [s for s in samples if s['view'] == view]


# --------------------------------------------------------------------------- #
# Convenience re-exports of the dataset-AGNOSTIC helpers from cargo_dataset.
# Guarded so this module still imports in a torch/torchvision-free smoke env
# (the AGReIDv2 meta-dataset itself depends on nothing beyond the stdlib).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - only succeeds in the full training env
    from cargo_dataset import (build_transforms, CARGOImageDataset,
                               RandomIdentitySampler)
    ImageDataset = CARGOImageDataset           # readable alias for AG-ReID use
except Exception:  # torch / torchvision absent (light smoke env)
    build_transforms = None
    CARGOImageDataset = None
    RandomIdentitySampler = None
    ImageDataset = None


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else '/tmp/agreidv2'
    ds = AGReIDv2(root=root, verbose=True)
    print("num_train_pids:", ds.num_train_pids,
          "num_query_pids:", ds.num_query_pids)
    print("A->G  q_aerial:", len(filter_by_view(ds.query, 'Aerial')),
          " g_ground:", len(filter_by_view(ds.gallery, 'Ground')))
    print("G->A  q_ground:", len(filter_by_view(ds.query, 'Ground')),
          " g_aerial:", len(filter_by_view(ds.gallery, 'Aerial')))
