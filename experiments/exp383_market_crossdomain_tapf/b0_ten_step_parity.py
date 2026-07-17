"""Two independent production-like ten-step parity runs for exp383 B0.

The gate uses the real Market sampler/augmentations, batch64, pretrained
Swin-T, SGD, AMP, and eight workers. It never enters the epoch runner, creates
an output directory, evaluates, or writes a checkpoint.
"""

import gc
import hashlib
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import cfg as root_cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from solver import make_optimizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def update_hash(digest, value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            digest.update(str(key).encode())
            update_hash(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            update_hash(digest, item)
    else:
        digest.update(repr(value).encode())


def object_sha(value):
    digest = hashlib.sha256()
    update_hash(digest, value)
    return digest.hexdigest()


def load_config():
    config = root_cfg.clone()
    config.defrost()
    config.merge_from_file(str(
        ROOT / 'configs' / 'market' / 'exp383_b0.yml'))
    config.freeze()
    if config.MODEL.POSE_ENABLED:
        raise RuntimeError('B0 parity must stay RGB-only')
    if (config.SOLVER.IMS_PER_BATCH != 64
            or config.DATALOADER.NUM_WORKERS != 8
            or config.SOLVER.SEED != 1234
            or config.INPUT.RE_PROB != 0.5):
        raise RuntimeError('B0 production recipe contract failed')
    return config


def run_once(config, run_index):
    seed = int(config.SOLVER.SEED)
    set_seed(seed)
    loaders = make_dataloader(config)
    train_loader, _, _, _, num_classes, camera_num, view_num = loaders
    model = make_model(
        config, num_class=num_classes, camera_num=camera_num,
        view_num=view_num, semantic_weight=config.MODEL.SEMANTIC_WEIGHT,
    ).cuda().train()
    loss_func, center_criterion = make_loss(config, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(
        config, model, center_criterion)
    scaler = amp.GradScaler(
        enabled=True, init_scale=float(config.SOLVER.AMP_INIT_SCALE))
    iterator = iter(train_loader)
    losses = []
    batches = []
    start = time.time()

    for step in range(10):
        batch = next(iterator)
        image, target, camera, view = batch
        if image.shape[0] != 64:
            raise RuntimeError('parity step is not batch64')
        batch_digest = hashlib.sha256()
        update_hash(batch_digest, image)
        update_hash(batch_digest, target)
        update_hash(batch_digest, camera)
        update_hash(batch_digest, view)
        batches.append(batch_digest.hexdigest())

        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        camera = camera.cuda(non_blocking=True)
        view = view.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        optimizer_center.zero_grad(set_to_none=True)
        with amp.autocast(enabled=True):
            score, feature, _ = model(
                image, label=target, cam_label=camera, view_label=view)
            loss = loss_func(score, feature, target, camera)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('non-finite B0 loss at step %d' % (step + 1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))

    model_state = {name: value.detach().cpu()
                   for name, value in model.state_dict().items()}
    result = {
        'losses': tuple(losses),
        'batches': tuple(batches),
        'model_sha': object_sha(model_state),
        'optimizer_sha': object_sha(optimizer.state_dict()),
        'cpu_rng_sha': object_sha(torch.get_rng_state()),
        'cuda_rng_sha': object_sha(torch.cuda.get_rng_state_all()),
        'scale': float(scaler.get_scale()),
        'seconds': time.time() - start,
    }
    print('run%d losses=%s model=%s optimizer=%s scale=%.1f seconds=%.2f'
          % (run_index, ','.join('%.8f' % value for value in losses),
             result['model_sha'], result['optimizer_sha'], result['scale'],
             result['seconds']))

    del iterator, train_loader, loaders, model, loss_func, center_criterion
    del optimizer, optimizer_center, scaler, model_state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    config = load_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.MODEL.DEVICE_ID)
    first = run_once(config, 1)
    second = run_once(config, 2)
    for key in (
            'losses', 'batches', 'model_sha', 'optimizer_sha',
            'cpu_rng_sha', 'cuda_rng_sha', 'scale'):
        if first[key] != second[key]:
            raise RuntimeError('B0 ten-step parity failed for ' + key)
    if not all(math.isfinite(value) for value in first['losses']):
        raise RuntimeError('B0 parity retained a non-finite loss')
    print('EXP383_B0_TEN_STEP_PARITY_PASS')
    print('batch64 steps=10x2 workers=8 re_prob=0.5')
    print('model_sha=%s optimizer_sha=%s'
          % (first['model_sha'], first['optimizer_sha']))


if __name__ == '__main__':
    main()
