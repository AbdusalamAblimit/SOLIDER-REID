# encoding: utf-8
"""
CARGO (aerial-ground person ReID) torch Dataset + standard ReID transforms.

Data layout (lab-3090):
    /root/work/SOLIDER-REID/data/CARGO/{train,query,gallery}/Cam{1..13}/*.jpg
Filename: Cam<N>_<time>_<pid>_<idx>.jpg   e.g. Cam13_day_376_26154.jpg
    camid = int(basename.split('_')[0][3:])     # 1..13
    pid   = int(basename.split('_')[2])
    view  = 'Aerial' if camid <= 5 else 'Ground'   # cam1-5 aerial, cam6-13 ground

Protocol (follows the official fast-reid CARGO parser in cargo.py, protocol-1 ALL):
    - train: relabel pids to a contiguous [0, num_train_pids) range.
    - query / gallery: keep ORIGINAL pids (test identities are disjoint from train).
    - camid is kept 0-indexed (camid-1) to match fast-reid convention.

The cross-view A<->G evaluation (the core confound test) is built downstream in
afd_train.py / band_analysis.py by filtering query/gallery on `view`:
    Aerial-as-query (A->G): query view==Aerial, gallery view==Ground
    Ground-as-query (G->A): query view==Ground, gallery view==Aerial
"""
import os
import glob
import random
import math

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# --------------------------------------------------------------------------- #
# Dataset parsing
# --------------------------------------------------------------------------- #
def _parse_name(img_path):
    """Return (pid, camid_1based, view) parsed from a CARGO filename."""
    name = os.path.basename(img_path)
    parts = name.split('_')
    # parts = ['Cam13', 'day', '376', '26154.jpg']
    camid = int(parts[0][3:])           # strip 'Cam' prefix -> 13
    pid = int(parts[2])
    view = 'Aerial' if camid <= 5 else 'Ground'
    return pid, camid, view


def _scan_split(split_dir):
    """Glob Cam1..Cam13 under split_dir, return list of (path, pid, camid_1based, view)."""
    img_paths = []
    for cam_index in range(13):
        img_paths += glob.glob(os.path.join(split_dir, f'Cam{cam_index + 1}', '*.jpg'))
    data = []
    for p in img_paths:
        pid, camid, view = _parse_name(p)
        data.append((p, pid, camid, view))
    return data


class CARGO(object):
    """
    Lightweight CARGO meta-dataset (fast-reid style, protocol-1 ALL).

    Attributes (each a list of dicts with keys: img_path, pid, camid, view):
        self.train, self.query, self.gallery
    self.num_train_pids / num_train_imgs / num_train_cams available after build.
    Train pids are relabeled to [0, num_train_pids); test pids keep original values.
    """

    def __init__(self, root='/root/work/SOLIDER-REID/data', verbose=True):
        self.dataset_dir = os.path.join(root, 'CARGO')
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'gallery')

        for d in (self.train_dir, self.query_dir, self.gallery_dir):
            if not os.path.isdir(d):
                raise RuntimeError(f"CARGO split dir not found: {d}")

        train_raw = _scan_split(self.train_dir)
        query_raw = _scan_split(self.query_dir)
        gallery_raw = _scan_split(self.gallery_dir)

        # relabel train pids to a contiguous range; keep test pids original.
        train_pids = sorted({pid for _, pid, _, _ in train_raw})
        self.pid2label = {pid: idx for idx, pid in enumerate(train_pids)}

        self.train = self._pack(train_raw, relabel=True)
        self.query = self._pack(query_raw, relabel=False)
        self.gallery = self._pack(gallery_raw, relabel=False)

        self.num_train_pids = len(train_pids)
        self.num_train_imgs = len(self.train)
        self.num_train_cams = len({d['camid'] for d in self.train})

        if verbose:
            self._print_stats()

    def _pack(self, raw, relabel):
        out = []
        for path, pid, camid, view in raw:
            label = self.pid2label[pid] if relabel else pid
            out.append({
                'img_path': path,
                'pid': label,
                'camid': camid - 1,   # 0-indexed, fast-reid convention
                'view': view,
            })
        return out

    def _print_stats(self):
        def cnt(split):
            pids = len({d['pid'] for d in split})
            cams = len({d['camid'] for d in split})
            a = sum(d['view'] == 'Aerial' for d in split)
            g = sum(d['view'] == 'Ground' for d in split)
            return len(split), pids, cams, a, g
        print("=> CARGO loaded (protocol-1 ALL)")
        print("  -----------------------------------------------------------")
        print("  subset   | # imgs | # pids | # cams | aerial | ground")
        print("  -----------------------------------------------------------")
        for name, split in (('train', self.train), ('query', self.query),
                            ('gallery', self.gallery)):
            n, p, c, a, g = cnt(split)
            print(f"  {name:8s} | {n:6d} | {p:6d} | {c:6d} | {a:6d} | {g:6d}")
        print("  -----------------------------------------------------------")


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(is_train, img_size=(256, 128), padding=10,
                     re_prob=0.5, mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)):
    """Standard ReID transforms.

    Train: Resize -> HFlip -> Pad+RandomCrop -> ToTensor -> Normalize -> RandomErasing
    Test : Resize -> ToTensor -> Normalize
    """
    normalize = T.Normalize(mean=mean, std=std)
    if is_train:
        return T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(padding),
            T.RandomCrop(img_size),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=re_prob, value=mean),
        ])
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalize,
    ])


# --------------------------------------------------------------------------- #
# torch Dataset
# --------------------------------------------------------------------------- #
class CARGOImageDataset(Dataset):
    """Wraps a list of sample dicts into an (img, pid, camid, view, path) Dataset."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _read_image(self, path):
        # robust loader (some synthetic frames can be flaky); never crash a worker.
        got = None
        for _ in range(5):
            try:
                got = Image.open(path).convert('RGB')
                break
            except (IOError, OSError):
                continue
        if got is None:
            # last-resort black image so a single bad file doesn't kill training
            got = Image.new('RGB', (128, 256))
        return got

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self._read_image(s['img_path'])
        if self.transform is not None:
            img = self.transform(img)
        return {
            'img': img,
            'pid': s['pid'],
            'camid': s['camid'],
            'view': s['view'],
            'img_path': s['img_path'],
        }


# --------------------------------------------------------------------------- #
# PK sampler (P identities x K instances) for triplet mining
# --------------------------------------------------------------------------- #
class RandomIdentitySampler(torch.utils.data.Sampler):
    """Sample batches of P identities, K instances each (default P=16, K=4 -> bs=64)."""

    def __init__(self, samples, batch_size, num_instances):
        self.samples = samples
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        assert batch_size % num_instances == 0, \
            "batch_size must be divisible by num_instances"

        self.index_by_pid = {}
        for idx, s in enumerate(samples):
            self.index_by_pid.setdefault(s['pid'], []).append(idx)
        self.pids = list(self.index_by_pid.keys())

        # estimate length: total usable images rounded to batch
        self.length = 0
        for pid in self.pids:
            n = len(self.index_by_pid[pid])
            n = max(n, self.num_instances)
            self.length += n - n % self.num_instances

    def __iter__(self):
        batch_idxs_dict = {}
        for pid in self.pids:
            idxs = list(self.index_by_pid[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances,
                                        replace=True).tolist()
            random.shuffle(idxs)
            batches = []
            batch = []
            for i in idxs:
                batch.append(i)
                if len(batch) == self.num_instances:
                    batches.append(batch)
                    batch = []
            batch_idxs_dict[pid] = batches

        avail_pids = [pid for pid in self.pids if batch_idxs_dict[pid]]
        final = []
        while len(avail_pids) >= self.num_pids_per_batch:
            selected = random.sample(avail_pids, self.num_pids_per_batch)
            for pid in selected:
                final.extend(batch_idxs_dict[pid].pop(0))
                if not batch_idxs_dict[pid]:
                    avail_pids.remove(pid)
        self.length = len(final)
        return iter(final)

    def __len__(self):
        return self.length


# --------------------------------------------------------------------------- #
# Eval-set view filtering helpers (cross-view A<->G splits)
# --------------------------------------------------------------------------- #
def filter_by_view(samples, view):
    """Return subset whose 'view' == view ('Aerial' or 'Ground')."""
    return [s for s in samples if s['view'] == view]


if __name__ == '__main__':
    # quick smoke test (run on lab-3090)
    ds = CARGO(root='/root/work/SOLIDER-REID/data', verbose=True)
    tf = build_transforms(is_train=True)
    train_set = CARGOImageDataset(ds.train, tf)
    sample = train_set[0]
    print("sample img shape:", sample['img'].shape,
          "pid:", sample['pid'], "camid:", sample['camid'], "view:", sample['view'])
    print("aerial query:", len(filter_by_view(ds.query, 'Aerial')),
          "ground query:", len(filter_by_view(ds.query, 'Ground')))
