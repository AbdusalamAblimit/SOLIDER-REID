"""Frozen full/early/late consumer ablation for the sealed exp389 checkpoint."""

import argparse
import json
import types
from contextlib import contextmanager
from pathlib import Path

import torch

from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.metrics import R1_mAP_eval


def identity_forward(self, tokens, hw_shape, field):
    del self, hw_shape, field
    return tokens, torch.zeros_like(tokens)


@contextmanager
def bypass_banks(banks):
    originals = [bank.forward for bank in banks]
    for bank in banks:
        bank.forward = types.MethodType(identity_forward, bank)
    try:
        yield
    finally:
        for bank, original in zip(banks, originals):
            bank.forward = original


def evaluate(model, val_loader, num_query):
    evaluator = R1_mAP_eval(
        num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM
    )
    evaluator.reset()
    model.eval()
    with torch.no_grad():
        for img, vid, camid, camids, target_view, _ in val_loader:
            feat, _ = model(
                img.cuda(non_blocking=True),
                cam_label=camids.cuda(non_blocking=True),
                view_label=target_view.cuda(non_blocking=True),
            )
            evaluator.update((feat, vid, camid))
    cmc, mean_ap, _, _, _, _, _ = evaluator.compute()
    return {
        "mAP": float(mean_ap * 100.0),
        "R1": float(cmc[0] * 100.0),
        "R5": float(cmc[4] * 100.0),
        "R10": float(cmc[9] * 100.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.freeze()
    loaders = make_dataloader(cfg)
    _, _, val_loader, num_query, num_classes, camera_num, view_num = loaders
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint load failed")
    model = model.cuda().eval()

    early = list(model.base.tapf.early_psg_bank)
    late = list(model.base.tapf.psg_bank)
    result = {"full": evaluate(model, val_loader, num_query)}
    with bypass_banks(early):
        result["early_bypass"] = evaluate(model, val_loader, num_query)
    with bypass_banks(late):
        result["late_bypass"] = evaluate(model, val_loader, num_query)
    with bypass_banks(early + late):
        result["all_bypass"] = evaluate(model, val_loader, num_query)
    result["full_repeat"] = evaluate(model, val_loader, num_query)
    if result["full_repeat"] != result["full"]:
        raise RuntimeError("Full evaluation did not reproduce after hooks")

    full = result["full"]
    for label in ("early_bypass", "late_bypass", "all_bypass"):
        result[label + "_minus_full"] = {
            metric: result[label][metric] - full[metric]
            for metric in ("mAP", "R1", "R5", "R10")
        }
    result["status"] = "EXP389_FROZEN_LEVEL_ABLATION_PASS"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
