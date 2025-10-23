import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num, semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT != '':
        model.load_param(cfg.TEST.WEIGHT)

    if cfg.DATASETS.NAMES == 'VehicleID':
        all_rank_1 = {}
        all_rank_5 = {}
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            results = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if not isinstance(results, dict):
                results = {'global': results}
            for name, (rank_1, rank5) in results.items():
                logger.info("rank_1({}):{:.1%}, rank_5({}) {:.1%} : trial : {}".format(name, rank_1, name, rank5, trial))
                if trial == 0:
                    all_rank_1[name] = rank_1
                    all_rank_5[name] = rank5
                else:
                    all_rank_1[name] += rank_1
                    all_rank_5[name] += rank5
        for name in all_rank_1.keys():
            logger.info("sum_rank_1({}):{:.1%}, sum_rank_5({}) {:.1%}".format(name, all_rank_1[name]/10.0, name, all_rank_5[name]/10.0))
    else:
        results = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
        if not isinstance(results, dict):
            results = {'global': results}
        for name, (rank_1, rank5) in results.items():
            logger.info("rank_1({}):{:.1%}, rank_5({}) {:.1%}".format(name, rank_1, name, rank5))

