from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import time
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    #  logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            #  logger.info(config_str)

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    logger.info("Running with config:\n{}".format(cfg))


    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        logger.info('===========using cosine learning rate=======')
        scheduler = create_scheduler(cfg, optimizer)
    else:
        logger.info('===========using normal learning rate=======')
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    tb_dir = os.path.join(cfg.OUTPUT_DIR, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    tb_dir_abs = os.path.abspath(tb_dir)
    logger.info(f"[TB] logdir = {tb_dir_abs}")

    # 只有主进程创建writer；非主进程传 None，避免多进程同时写
    is_main = (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0
    writer = SummaryWriter(log_dir=tb_dir_abs) if is_main else None

    # 写一条“引导”记录，确保立刻落盘有事件文件
    if writer is not None:
        writer.add_text("meta/start", time.strftime("%Y-%m-%d %H:%M:%S"), 0)
        writer.add_scalar("meta/heartbeat", 0.0, 0)
        writer.flush()

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query,
        args.local_rank,
        tb_writer=writer,     # 传给训练过程
    )

    writer.close()
