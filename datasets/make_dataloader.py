import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
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

def train_collate_fn_pose(batch):
    """Collate function for PoseImageDataset (includes keypoints + visibility)."""
    imgs, pids, camids, viewids, _, kpts, vis = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    kpts = torch.stack(kpts, dim=0)  # [B, 17, 2]
    vis = torch.stack(vis, dim=0)    # [B, 17]
    return torch.stack(imgs, dim=0), pids, camids, viewids, kpts, vis

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def val_collate_fn_pose(batch):
    """Collate function for PoseValDataset (includes keypoints + visibility)."""
    imgs, pids, camids, viewids, img_paths, kpts, vis = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    kpts = torch.stack(kpts, dim=0)
    vis = torch.stack(vis, dim=0)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths, kpts, vis


def _load_pose_data(data_root, dataset_name):
    """Load pre-extracted pose data from .npz files."""
    pose_dir = os.path.join(data_root, dataset_name)

    result = {}
    for split in ['train', 'query', 'gallery']:
        npz_path = os.path.join(pose_dir, f'pose_{split}.npz')
        if os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            result[split] = {
                'filenames': data['filenames'],
                'keypoints': data['keypoints'],
                'visibility': data['visibility'],
            }
            print(f'Loaded pose data: {split} ({len(data["filenames"])} images)')
        else:
            print(f'WARNING: pose data not found: {npz_path}')
            result[split] = None

    return result


def make_dataloader(cfg):
    # Check if pose-aware loading is needed
    use_pose = (getattr(cfg.MODEL, 'POSE_PART', None) and cfg.MODEL.POSE_PART.ENABLE) or \
               (getattr(cfg.MODEL, 'PCFC', None) and cfg.MODEL.PCFC.ENABLE)

    train_transforms_base = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])

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

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if use_pose:
        from .pose_dataset import PoseImageDataset, PoseValDataset
        pose_data = _load_pose_data(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.NAMES)

        if pose_data.get('train'):
            bpre_prob = getattr(cfg.MODEL.PCFC, 'BPRE_PROB', 0.0) if hasattr(cfg.MODEL, 'PCFC') else 0.0
            bpre_max = getattr(cfg.MODEL.PCFC, 'BPRE_MAX_PARTS', 1) if hasattr(cfg.MODEL, 'PCFC') else 1
            train_set = PoseImageDataset(
                dataset.train, train_transforms_base, pose_data['train'],
                re_prob=cfg.INPUT.RE_PROB,
                img_size=cfg.INPUT.SIZE_TRAIN,
                bpre_prob=bpre_prob,
                bpre_max_parts=bpre_max,
            )
        else:
            train_set = ImageDataset(dataset.train, train_transforms)

        collate_train = train_collate_fn_pose if pose_data.get('train') else train_collate_fn
        collate_val = val_collate_fn_pose

        # Merge query + gallery pose data for val
        if pose_data.get('query') and pose_data.get('gallery'):
            merged_pose = {
                'filenames': np.concatenate([pose_data['query']['filenames'],
                                              pose_data['gallery']['filenames']]),
                'keypoints': np.concatenate([pose_data['query']['keypoints'],
                                              pose_data['gallery']['keypoints']]),
                'visibility': np.concatenate([pose_data['query']['visibility'],
                                               pose_data['gallery']['visibility']]),
            }
            val_set = PoseValDataset(dataset.query + dataset.gallery, val_transforms, merged_pose)
        else:
            val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
            collate_val = val_collate_fn

        train_set_normal = ImageDataset(dataset.train, val_transforms)
    else:
        train_set = ImageDataset(dataset.train, train_transforms)
        train_set_normal = ImageDataset(dataset.train, val_transforms)
        collate_train = train_collate_fn
        collate_val = val_collate_fn
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

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

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_val
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
