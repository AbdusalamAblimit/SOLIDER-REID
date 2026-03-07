import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from .pose_dataset import PoseImageDataset, load_pose_data
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_IdUniform
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .mm import MM
from .occluded_duke import OccludedDukeMTMC
__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'mm': MM,
    'occluded_duke': OccludedDukeMTMC,
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


def _collate_pose_dicts(pose_dicts):
    """Stack a list of pose_dict into batched tensors."""
    batched = {}
    for key in pose_dicts[0]:
        if key == 'num_persons':
            batched[key] = torch.tensor([d[key] for d in pose_dicts], dtype=torch.int64)
        else:
            batched[key] = torch.stack([d[key] for d in pose_dicts], dim=0)
    return batched

def pose_train_collate_fn(batch):
    """Collate function for pose-aware training."""
    imgs, pids, camids, viewids, _, pose_dicts = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return (torch.stack(imgs, dim=0), pids, camids, viewids,
            _collate_pose_dicts(pose_dicts))

def pose_val_collate_fn(batch):
    """Collate function for pose-aware validation."""
    imgs, pids, camids, viewids, img_paths, pose_dicts = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return (torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths,
            _collate_pose_dicts(pose_dicts))

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

    # For pose mode: pixel-only transforms (geometric handled in dataset)
    pose_train_pixel_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    # For pose val: pixel-only (resize handled in dataset)
    pose_val_pixel_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES == 'ourapi':
        dataset = OURAPI(root_train=cfg.DATASETS.ROOT_TRAIN_DIR, root_val=cfg.DATASETS.ROOT_VAL_DIR, config=cfg)
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    # Pose data loading
    use_pose = cfg.MODEL.POSE_ENABLED
    if use_pose:
        pose_dir = cfg.MODEL.POSE_DATA_DIR
        train_pose = load_pose_data(pose_dir, 'train')
        gallery_pose = load_pose_data(pose_dir, 'gallery')
        query_pose = load_pose_data(pose_dir, 'query')
        # Merge query + gallery pose data for validation
        if query_pose is not None and gallery_pose is not None:
            if isinstance(query_pose, list) and isinstance(gallery_pose, list):
                # New PKL format: just concatenate lists
                val_pose = query_pose + gallery_pose
            elif isinstance(query_pose, dict) and isinstance(gallery_pose, dict):
                # Legacy NPZ format
                import numpy as np
                val_pose = {
                    'filenames': np.concatenate([query_pose['filenames'], gallery_pose['filenames']]),
                    'keypoints': np.concatenate([query_pose['keypoints'], gallery_pose['keypoints']]),
                    'scores': np.concatenate([query_pose['scores'], gallery_pose['scores']]),
                }
            else:
                val_pose = None
        else:
            val_pose = None
        # Heatmap directories
        heatmap_base = os.path.join(pose_dir, 'heatmaps')
        train_heatmap_dir = os.path.join(heatmap_base, 'train')
        # For val, we need separate dirs per split - handled in dataset
        query_heatmap_dir = os.path.join(heatmap_base, 'query')
        gallery_heatmap_dir = os.path.join(heatmap_base, 'gallery')
        DatasetClass = PoseImageDataset
        collate_train = pose_train_collate_fn
        collate_val = pose_val_collate_fn
    else:
        train_pose = None
        val_pose = None
        DatasetClass = ImageDataset
        collate_train = train_collate_fn
        collate_val = val_collate_fn

    if use_pose:
        train_set = DatasetClass(
            dataset.train, pose_train_pixel_transforms, train_pose,
            heatmap_dir=train_heatmap_dir,
            is_train=True,
            flip_prob=cfg.INPUT.PROB,
            pad=cfg.INPUT.PADDING,
            crop_size=cfg.INPUT.SIZE_TRAIN,
        )
        train_set_normal = DatasetClass(
            dataset.train, pose_val_pixel_transforms, train_pose,
            heatmap_dir=train_heatmap_dir,
            is_train=False,
            crop_size=cfg.INPUT.SIZE_TRAIN,
        )
    else:
        train_set = DatasetClass(dataset.train, train_transforms)
        train_set_normal = DatasetClass(dataset.train, val_transforms)
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
                collate_fn=collate_train,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=collate_train
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=collate_train
        )
    elif cfg.DATALOADER.SAMPLER in ['id_triplet', 'id']:
        print('using ID sampler')
        train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler_IdUniform(dataset.train, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=collate_train, drop_last = True,
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    if use_pose:
        val_set = DatasetClass(
            dataset.query + dataset.gallery, pose_val_pixel_transforms, val_pose,
            heatmap_dir=[query_heatmap_dir, gallery_heatmap_dir],
            is_train=False,
            crop_size=cfg.INPUT.SIZE_TEST,
        )
    else:
        val_set = DatasetClass(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_val
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_val
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
