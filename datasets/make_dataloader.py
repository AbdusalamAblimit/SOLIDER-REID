import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from .pose_dataset import PoseImageDataset, pose_train_collate_fn, pose_val_collate_fn
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_IdUniform
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .mm import MM
from .occluded_duke import OccludedDukeMTMC
from .occluded_posetrack import OccludedPoseTrack
from .occluded_reid import OccludedREID

__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'mm': MM,
    'occluded_duke': OccludedDukeMTMC,
    'occluded_posetrack': OccludedPoseTrack,
    'occluded_reid': OccludedREID,
}


def train_collate_fn(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel',
                      max_count=1, device='cpu'),
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES == 'ourapi':
        dataset = OURAPI(root_train=cfg.DATASETS.ROOT_TRAIN_DIR,
                         root_val=cfg.DATASETS.ROOT_VAL_DIR, config=cfg)
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    use_pose = cfg.MODEL.POSE_ENABLED

    if use_pose:
        pose_base = os.path.join(cfg.MODEL.POSE_DATA_DIR, 'pose_data')
        hm_size = None
        if hasattr(cfg.MODEL, 'POSE_HEATMAP_SIZE'):
            hm_size = tuple(cfg.MODEL.POSE_HEATMAP_SIZE)

        # Load ROA occluders if enabled
        occluders = None
        if getattr(cfg.MODEL, 'POSE_ROA', False):
            from .occlusion_augmentation import load_occluders
            roa_path = getattr(cfg.MODEL, 'POSE_ROA_PATH', 'data/VOCdevkit/VOC2012')
            print(f'Loading ROA occluders from {roa_path}...')
            occluders = load_occluders(roa_path)
            print(f'Loaded {len(occluders)} occluder patches')

        pose_kwargs = dict(
            img_size=tuple(cfg.INPUT.SIZE_TRAIN),
            flip_prob=cfg.INPUT.PROB,
            pad=cfg.INPUT.PADDING,
            re_prob=cfg.INPUT.RE_PROB,
            pixel_mean=cfg.INPUT.PIXEL_MEAN,
            pixel_std=cfg.INPUT.PIXEL_STD,
            heatmap_size=hm_size,
            pose_guided_erasing=getattr(cfg.MODEL, 'POSE_GUIDED_ERASING', False),
            occluders=occluders,
            roa_prob=getattr(cfg.MODEL, 'POSE_ROA_PROB', 0.5),
        )

        train_set = PoseImageDataset(
            dataset.train,
            pose_dir=os.path.join(pose_base, 'train'),
            is_train=True, **pose_kwargs)
        # Set pose-aware ROA flag
        if getattr(cfg.MODEL, 'POSE_ROA_POSE_AWARE', False):
            train_set.pose_aware_roa = True
        if getattr(cfg.MODEL, 'POSE_PCVT', False):
            train_set.pcvt = True
            train_set.pcvt_resp_thr = float(getattr(cfg.MODEL, 'POSE_PCVT_RESP_THR', 0.10))
            train_set.pcvt_active_thr = float(getattr(cfg.MODEL, 'POSE_PCVT_ACT_THR', 0.30))
            train_set.pcvt_min_parts = int(getattr(cfg.MODEL, 'POSE_PCVT_MIN_PARTS', 2))
            train_set.pcvt_fill_value = float(getattr(cfg.MODEL, 'POSE_PCVT_FILL', 0.0))
        # Set parallel augmentation flag (3-view training)
        if getattr(cfg.MODEL, 'POSE_PARALLEL_AUG', False) or getattr(cfg.MODEL, 'POSE_PCVT', False):
            train_set.parallel_aug = True

        train_set_normal = PoseImageDataset(
            dataset.train,
            pose_dir=os.path.join(pose_base, 'train'),
            is_train=False,
            img_size=tuple(cfg.INPUT.SIZE_TEST),
            pixel_mean=cfg.INPUT.PIXEL_MEAN,
            pixel_std=cfg.INPUT.PIXEL_STD,
            heatmap_size=hm_size)

        # Val: query + gallery images, each with their own pose dir
        # We combine by pointing to both dirs via a merged index
        val_set = _make_pose_val_set(
            dataset, pose_base, cfg, hm_size)

        collate_train = pose_train_collate_fn
        collate_val = pose_val_collate_fn
    else:
        train_set = ImageDataset(dataset.train, train_transforms)
        train_set_normal = ImageDataset(dataset.train, val_transforms)
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
        collate_train = train_collate_fn
        collate_val = val_collate_fn

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'img_triplet']:
        print('using img_triplet sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(
                dataset.train, cfg.SOLVER.IMS_PER_BATCH,
                cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                data_sampler, mini_batch_size, True)
            train_loader = DataLoader(
                train_set, num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=collate_train, pin_memory=True)
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(
                    dataset.train, cfg.SOLVER.IMS_PER_BATCH,
                    cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=collate_train)
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True, num_workers=num_workers,
            collate_fn=collate_train)
    elif cfg.DATALOADER.SAMPLER in ['id_triplet', 'id']:
        print('using ID sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler_IdUniform(
                dataset.train, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=collate_train,
            drop_last=True)
    else:
        print('unsupported sampler! expected softmax or triplet '
              'but got {}'.format(cfg.SAMPLER))

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False, num_workers=num_workers,
        collate_fn=collate_val)
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False, num_workers=num_workers,
        collate_fn=collate_val)
    return (train_loader, train_loader_normal, val_loader,
            len(dataset.query), num_classes, cam_num, view_num)


def _make_pose_val_set(dataset, pose_base, cfg, hm_size):
    """Create a PoseImageDataset for validation (query + gallery combined).

    Since query and gallery have separate pose_data dirs, we merge their
    index.json files into a temporary combined directory.
    """
    import json
    import tempfile

    query_dir = os.path.join(pose_base, 'query')
    gallery_dir = os.path.join(pose_base, 'gallery')

    # Merge indices: filenames are unique across splits (different dirs)
    merged_index = {}
    merged_npz_dirs = []

    for split_dir in [query_dir, gallery_dir]:
        idx_path = os.path.join(split_dir, 'index.json')
        if os.path.exists(idx_path):
            with open(idx_path, 'r') as f:
                idx = json.load(f)
            # Prefix npz paths with absolute split directory
            abs_split_dir = os.path.abspath(split_dir)
            for fname, entry in idx.items():
                entry_copy = dict(entry)
                entry_copy['persons'] = [
                    os.path.join(abs_split_dir, p) for p in entry['persons']]
                merged_index[fname] = entry_copy
            merged_npz_dirs.append(split_dir)

    # Create a merged dir with symlinked index
    merged_dir = os.path.join(pose_base, '_val_merged')
    os.makedirs(merged_dir, exist_ok=True)
    merged_idx_path = os.path.join(merged_dir, 'index.json')
    with open(merged_idx_path, 'w') as f:
        json.dump(merged_index, f)

    # The PoseImageDataset will load npz files using absolute paths
    # stored in the merged index, so pose_dir just needs the index.json
    return PoseImageDataset(
        dataset.query + dataset.gallery,
        pose_dir=merged_dir,
        is_train=False,
        img_size=tuple(cfg.INPUT.SIZE_TEST),
        pixel_mean=cfg.INPUT.PIXEL_MEAN,
        pixel_std=cfg.INPUT.PIXEL_STD,
        heatmap_size=hm_size)
