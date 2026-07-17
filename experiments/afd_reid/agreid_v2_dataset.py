# encoding: utf-8
"""
AG-ReID.v2 (aerial / ground person ReID) meta-dataset adapter.

Exposes the SAME interface as cargo_dataset.CARGO so the existing CARGO-style
eval / training code (afd_train.py, band_analysis.py) can consume it unchanged:

    ds = AGReIDV2(root='/root/work/SOLIDER-REID/data', protocol='exp1')
    ds.train / ds.query / ds.gallery   # list[dict(img_path, pid, camid, view)]
    ds.num_train_pids / num_train_imgs / num_train_cams

Re-use the CARGO torch wrappers directly:
    from cargo_dataset import CARGOImageDataset, build_transforms,
                              RandomIdentitySampler, filter_by_view


Data layout (lab-3090, /root/work/SOLIDER-REID/data/AG-ReID.v2/):
    train_all/<FOLDER>/<FOLDER><C-field>F<frame>.jpg   (807 folders)
    query/<FOLDER>/...                                 (808 folders)
    gallery/<FOLDER>/...                               (808 folders)
    exp1_aerial_to_cctv.txt   exp2_aerial_to_wearable.txt
    exp4_cctv_to_aerial.txt   exp5_wearable_to_aerial.txt

Folder / filename encoding:
    FOLDER   = P{seq}T{date}A{altitude}      e.g. P0313T02220A0
    filename = {FOLDER}C{platform}F{frame}.jpg   e.g. P0313T02220A0C0F11011.jpg
        P = capture-sequence index (NOT a usable identity on its own)
        T = date / tracklet field
        A = altitude  (0 low / 1 mid / 2 high)
        C = platform  (0 = UAV/aerial, 2 = wearable, 3 = CCTV)
        F = frame number  (may carry a trailing letter, e.g. ...F8671Z.jpg)

★ Identity (pid):
    The pid is the FULL FOLDER NAME (P####T####A#), i.e. each P-T-A track is one
    identity -- NOT the bare P prefix.  Empirically this yields 808 unique test
    identities, matching the official AG-ReID.v2 spec (807 train / 808 test).
    Using only the P prefix collapses tracks (-> 649 test / 1044 across splits),
    which is wrong.

★ Platform / view (camid + coarse view):
    platform comes from the C field of the *filename* (a single folder mixes
    platforms).  We keep a stable per-platform camid and a coarse view so the
    CARGO cross-view helpers (filter_by_view, eval_market junk removal) keep
    working:
        C0 (UAV)      -> camid 0, view 'Aerial'
        C2 (wearable) -> camid 2, view 'Ground'   (ground-side modality)
        C3 (CCTV)     -> camid 3, view 'Ground'   (ground-side modality)
    'Aerial' vs 'Ground' mirrors CARGO's A<->G split: aerial == UAV(C0),
    ground == the non-UAV platform of the protocol (CCTV or wearable).

★ Protocol subset selection:
    query / gallery are NOT the whole on-disk folders -- each protocol txt lists
    the exact image paths (lines prefixed 'query/' or 'gallery/') for that
    cross-platform direction.  We parse the txt and select exactly those images.
        exp1  aerial_to_cctv      query C0 (aerial)  -> gallery C3 (CCTV)   [A->G]
        exp2  aerial_to_wearable  query C0 (aerial)  -> gallery C2 (wear)   [A->G]
        exp4  cctv_to_aerial      query C3 (CCTV)    -> gallery C0 (aerial) [G->A]
        exp5  wearable_to_aerial  query C2 (wear)    -> gallery C0 (aerial) [G->A]
    Main protocol (paired with CARGO A<->G): exp1 (A->G) + exp4 (G->A).
"""
import os
import re
import glob


# --------------------------------------------------------------------------- #
# field parsing
# --------------------------------------------------------------------------- #
# platform (C field) -> (camid, coarse view)
#   aerial == UAV(C0); ground == wearable(C2) / CCTV(C3)
_PLATFORM = {
    0: (0, 'Aerial'),   # UAV - RGB
    2: (2, 'Ground'),   # wearable - RGB
    3: (3, 'Ground'),   # CCTV - RGB
}

# tolerant: capture the C-platform digit; frame may carry a trailing letter (…F8671Z.jpg)
_CFIELD_RE = re.compile(r'C(\d)F\d+[A-Za-z]*\.jpg$', re.IGNORECASE)

PROTOCOLS = {
    'exp1': 'exp1_aerial_to_cctv.txt',
    'exp2': 'exp2_aerial_to_wearable.txt',
    'exp4': 'exp4_cctv_to_aerial.txt',
    'exp5': 'exp5_wearable_to_aerial.txt',
}
# alias the descriptive direction names too
PROTOCOLS.update({
    'aerial_to_cctv': 'exp1_aerial_to_cctv.txt',
    'aerial_to_wearable': 'exp2_aerial_to_wearable.txt',
    'cctv_to_aerial': 'exp4_cctv_to_aerial.txt',
    'wearable_to_aerial': 'exp5_wearable_to_aerial.txt',
})


def _platform_of(filename):
    """Return (camid, view) parsed from the C field of a filename, or None."""
    m = _CFIELD_RE.search(os.path.basename(filename))
    if m is None:
        return None
    c = int(m.group(1))
    return _PLATFORM.get(c)   # None for an unexpected platform digit


def _pid_from_path(rel_or_abs_path):
    """pid = the FOLDER name (P####T####A#), the component just above the file."""
    return os.path.basename(os.path.dirname(rel_or_abs_path))


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class AGReIDV2(object):
    """
    AG-ReID.v2 meta-dataset, CARGO-compatible.

    Args:
        root      : parent dir containing 'AG-ReID.v2/'
        protocol  : which query/gallery direction to build for eval
                    ('exp1'|'exp2'|'exp4'|'exp5' or the descriptive names).
        relabel_train : relabel train pids to a contiguous [0, N) range (default True).

    Attributes (lists of dict: img_path, pid, camid, view):
        self.train, self.query, self.gallery
        self.num_train_pids / num_train_imgs / num_train_cams
        self.pid2label  (train folder-name -> contiguous label)
    Train pids are relabeled; query/gallery keep their original folder-name pid
    (test ids are disjoint from train and only need to be consistent q vs g).
    """

    def __init__(self, root='/root/work/SOLIDER-REID/data',
                 protocol='exp1', relabel_train=True, verbose=True):
        self.dataset_dir = os.path.join(root, 'AG-ReID.v2')
        self.train_dir = os.path.join(self.dataset_dir, 'train_all')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'gallery')

        if protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol {protocol!r}; "
                             f"choose from {sorted(PROTOCOLS)}")
        self.protocol = protocol
        self.protocol_file = os.path.join(self.dataset_dir, PROTOCOLS[protocol])

        for d in (self.train_dir, self.query_dir, self.gallery_dir):
            if not os.path.isdir(d):
                raise RuntimeError(f"AG-ReID.v2 split dir not found: {d}")
        if not os.path.isfile(self.protocol_file):
            raise RuntimeError(f"protocol file not found: {self.protocol_file}")

        # ---- train: every image under train_all, pid = folder name ----------
        train_raw = self._scan_train()
        train_pids = sorted({pid for _, pid, _, _ in train_raw})
        self.pid2label = {pid: i for i, pid in enumerate(train_pids)}
        self.train = self._pack(train_raw,
                                relabel=relabel_train, label_map=self.pid2label)

        # ---- query / gallery: exactly the images listed in the protocol -----
        q_raw, g_raw = self._scan_protocol()
        self.query = self._pack(q_raw, relabel=False)
        self.gallery = self._pack(g_raw, relabel=False)

        self.num_train_pids = len(train_pids)
        self.num_train_imgs = len(self.train)
        self.num_train_cams = len({d['camid'] for d in self.train})

        if verbose:
            self._print_stats()

    # ----- scanners -------------------------------------------------------- #
    def _scan_train(self):
        """All train images; pid = folder name, camid/view from C field."""
        out = []
        skipped = 0
        for path in glob.glob(os.path.join(self.train_dir, '*', '*.jpg')):
            plat = _platform_of(path)
            if plat is None:
                skipped += 1
                continue
            camid, view = plat
            out.append((path, _pid_from_path(path), camid, view))
        if skipped:
            print(f"[AGReIDV2] train: skipped {skipped} files with unparsable platform")
        return out

    def _scan_protocol(self):
        """Read protocol txt; return (query_raw, gallery_raw).

        Each line: '<split>/<folder>/<file>.jpg', split in {'query','gallery'}.
        img_path is resolved against dataset_dir; pid = folder, camid/view = C field.
        """
        q_raw, g_raw = [], []
        missing, bad = 0, 0
        with open(self.protocol_file, 'r') as fh:
            for line in fh:
                rel = line.strip()
                if not rel:
                    continue
                rel = rel.replace('\\', '/')
                split = rel.split('/', 1)[0]
                if split not in ('query', 'gallery'):
                    bad += 1
                    continue
                abs_path = os.path.join(self.dataset_dir, rel)
                if not os.path.isfile(abs_path):
                    missing += 1
                    continue
                plat = _platform_of(rel)
                if plat is None:
                    bad += 1
                    continue
                camid, view = plat
                rec = (abs_path, _pid_from_path(rel), camid, view)
                (q_raw if split == 'query' else g_raw).append(rec)
        if missing:
            print(f"[AGReIDV2] protocol {self.protocol}: {missing} listed files "
                  f"missing on disk (skipped)")
        if bad:
            print(f"[AGReIDV2] protocol {self.protocol}: {bad} unparsable lines (skipped)")
        return q_raw, g_raw

    # ----- packing --------------------------------------------------------- #
    @staticmethod
    def _pack(raw, relabel, label_map=None):
        out = []
        for path, pid, camid, view in raw:
            label = label_map[pid] if relabel else pid
            out.append({
                'img_path': path,
                'pid': label,        # int (train) or folder-name str (query/gallery)
                'camid': camid,      # 0 UAV / 2 wearable / 3 CCTV
                'view': view,        # 'Aerial' (UAV) / 'Ground' (wearable|CCTV)
            })
        return out

    # ----- stats ----------------------------------------------------------- #
    def _print_stats(self):
        def cnt(split):
            pids = len({d['pid'] for d in split})
            cams = len({d['camid'] for d in split})
            a = sum(d['view'] == 'Aerial' for d in split)
            g = sum(d['view'] == 'Ground' for d in split)
            return len(split), pids, cams, a, g
        print(f"=> AG-ReID.v2 loaded (protocol={self.protocol})")
        print("  -----------------------------------------------------------")
        print("  subset   | # imgs | # pids | # cams | aerial | ground")
        print("  -----------------------------------------------------------")
        for name, split in (('train', self.train), ('query', self.query),
                            ('gallery', self.gallery)):
            n, p, c, a, g = cnt(split)
            print(f"  {name:8s} | {n:6d} | {p:6d} | {c:6d} | {a:6d} | {g:6d}")
        print("  -----------------------------------------------------------")


if __name__ == '__main__':
    # quick smoke test (run on lab-3090)
    ds = AGReIDV2(root='/root/work/SOLIDER-REID/data', protocol='exp1', verbose=True)
    print("num_train_pids:", ds.num_train_pids)
    s = ds.query[0]
    print("query[0]:", {k: s[k] for k in ('pid', 'camid', 'view')},
          os.path.basename(s['img_path']))
