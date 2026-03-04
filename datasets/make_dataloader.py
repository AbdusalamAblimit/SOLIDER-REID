import json
import os.path as osp

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset, PoseImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_IdUniform
from .market1501 import Market1501
from .msmt17 import MSMT17
from .occluded_duke import OccludedDukeMTMC
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .mm import MM
__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'occluded_duke': OccludedDukeMTMC,
    'mm': MM,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def pose_train_collate_fn(batch):
    """Collate with pose keypoints and visibility."""
    imgs, pids, camids, viewids, _, kpts, vis = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return (torch.stack(imgs, dim=0), pids, camids, viewids,
            torch.stack(kpts, dim=0), torch.stack(vis, dim=0))

def pose_val_collate_fn(batch):
    """Collate with pose keypoints and visibility for validation."""
    imgs, pids, camids, viewids, img_paths, kpts, vis = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return (torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths,
            torch.stack(kpts, dim=0), torch.stack(vis, dim=0))


def _load_pose_data(cfg):
    """Load pre-extracted pose JSON files for all splits."""
    pose_dir = osp.join(cfg.DATASETS.ROOT_DIR,
                        cfg.DATASETS.NAMES if isinstance(cfg.DATASETS.NAMES, str) else cfg.DATASETS.NAMES,
                        'pose')
    pose_data = {}
    for split in ['train', 'query', 'gallery']:
        path = osp.join(pose_dir, f'{split}_keypoints.json')
        if osp.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            pose_data.update(data)
            print(f"Loaded {len(data)} pose entries from {path}")
        else:
            print(f"Warning: pose file not found: {path}")
    return pose_data

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES == 'ourapi':
        dataset = OURAPI(root_train=cfg.DATASETS.ROOT_TRAIN_DIR, root_val=cfg.DATASETS.ROOT_VAL_DIR, config=cfg)
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    # Determine if we need pose data
    use_pose = getattr(cfg.MODEL, 'POSE', None) and cfg.MODEL.POSE.ENABLE
    if not use_pose:
        use_pose = getattr(cfg.MODEL, 'PAMS', None) and cfg.MODEL.PAMS.ENABLE

    if use_pose:
        pose_data = _load_pose_data(cfg)
        train_set = PoseImageDataset(dataset.train, train_transforms, pose_data)
        train_set_normal = PoseImageDataset(dataset.train, val_transforms, pose_data)
        _train_collate = pose_train_collate_fn
        _val_collate = pose_val_collate_fn
    else:
        train_set = ImageDataset(dataset.train, train_transforms)
        train_set_normal = ImageDataset(dataset.train, val_transforms)
        _train_collate = train_collate_fn
        _val_collate = val_collate_fn

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'img_triplet']:
        print('using img_triplet sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=_train_collate,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=_train_collate
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=_train_collate
        )
    elif cfg.DATALOADER.SAMPLER in ['id_triplet', 'id']:
        print('using ID sampler')
        train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler_IdUniform(dataset.train, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=_train_collate, drop_last = True,
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    if use_pose:
        val_set = PoseImageDataset(dataset.query + dataset.gallery, val_transforms, pose_data)
    else:
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=_val_collate
    )
    if use_pose:
        train_set_normal_for_loader = PoseImageDataset(dataset.train, val_transforms, pose_data)
    else:
        train_set_normal_for_loader = train_set_normal
    train_loader_normal = DataLoader(
        train_set_normal_for_loader, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=_val_collate
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
