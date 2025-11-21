import argparse
import os
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg
from model import make_model
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose-based visualization inference")
    parser.add_argument("image_dir", type=str, help="Directory containing input images")
    parser.add_argument("weight", type=str, help="Path to the model weight (.pth)")
    parser.add_argument(
        "--config_file", default="", type=str, help="Path to config file for the model"
    )
    parser.add_argument(
        "--output_dir", default="./pose_vis", type=str, help="Directory to save visualizations"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for inference without dataloader"
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging using tb_dump_pose"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def build_transform(cfg_file: str, opts: Optional[Sequence[str]], weight_path: str) -> None:
    if cfg_file:
        cfg.merge_from_file(cfg_file)
    if opts:
        cfg.merge_from_list(list(opts))
    cfg.defrost()
    cfg.MODEL.POSE.ENABLE = True
    cfg.MODEL.POSE.SAVE_VIS = True
    cfg.TEST.WEIGHT = weight_path
    cfg.freeze()


def get_transform() -> T.Compose:
    size = cfg.INPUT.SIZE_TRAIN
    normalize = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        normalize,
    ])


def load_model(weight_path: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    device = torch.device(cfg.MODEL.DEVICE if torch.cuda.is_available() else "cpu")
    model = make_model(cfg, num_class=1, camera_num=1, view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(weight_path)
    model.to(device)
    model.eval()
    return model, device


def list_images(image_dir: str) -> List[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = [p for p in pathlib.Path(image_dir).rglob("*") if p.suffix.lower() in exts]
    paths.sort()
    return paths


def _to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


def _save_map_image(map_2d: np.ndarray, save_path: pathlib.Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), map_2d)


def _normalize_tensor(t: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    if reduce == "mean":
        t = t.mean(dim=1, keepdim=True)
    elif reduce == "sum":
        t = t.sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported reduce method: {reduce}")
    t_min = t.amin(dim=(-2, -1), keepdim=True)
    t_max = t.amax(dim=(-2, -1), keepdim=True)
    return (t - t_min) / (t_max - t_min + 1e-6)


def visualize_heatmaps(
    hm: torch.Tensor,
    orig_hw: Tuple[int, int],
    image_bgr: np.ndarray,
    stem: str,
    out_dir: pathlib.Path,
) -> None:
    hm_up = F.interpolate(hm, size=orig_hw, mode="bilinear", align_corners=False)
    agg = hm_up.sum(dim=1, keepdim=True)
    agg = _normalize_tensor(agg, reduce="sum").squeeze(0).squeeze(0).cpu().numpy()
    heatmap_color = cv2.applyColorMap(_to_uint8(agg), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_color, 0.4, 0)
    _save_map_image(heatmap_color, out_dir / f"{stem}_heatmap.png")
    _save_map_image(overlay, out_dir / f"{stem}_overlay.png")


def visualize_feature_maps(
    feat_maps: Optional[List[torch.Tensor]],
    orig_hw: Tuple[int, int],
    image_bgr: np.ndarray,
    stem: str,
    prefix: str,
    out_dir: pathlib.Path,
) -> None:
    if not feat_maps:
        return
    for idx, fmap in enumerate(feat_maps):
        fmap_up = F.interpolate(fmap, size=orig_hw, mode="bilinear", align_corners=False)
        fmap_norm = _normalize_tensor(fmap_up, reduce="mean").squeeze(0).squeeze(0).cpu().numpy()
        fmap_img = cv2.applyColorMap(_to_uint8(fmap_norm), cv2.COLORMAP_VIRIDIS)
        overlay = cv2.addWeighted(image_bgr, 0.6, fmap_img, 0.4, 0)
        _save_map_image(fmap_img, out_dir / f"{stem}_{prefix}_stage{idx}.png")
        _save_map_image(overlay, out_dir / f"{stem}_{prefix}_stage{idx}_overlay.png")


def visualize_tb_cache(
    cache: Dict[str, torch.Tensor],
    stem: str,
    out_dir: pathlib.Path,
    sample_idx: int,
) -> None:
    if not cache:
        return
    for name in ("in_feat", "fused_feat", "hm", "hm_proj"):
        tensor = cache.get(name)
        if tensor is None or sample_idx >= tensor.size(0):
            continue
        reduce_type = "sum" if name == "hm" else "mean"
        reduced = _normalize_tensor(tensor[sample_idx : sample_idx + 1], reduce=reduce_type)
        img = reduced.squeeze(0).squeeze(0).cpu().numpy()
        cmap = cv2.COLORMAP_JET if name.startswith("hm") else cv2.COLORMAP_VIRIDIS
        mapped = cv2.applyColorMap(_to_uint8(img), cmap)
        _save_map_image(mapped, out_dir / f"{stem}_tb_{name}.png")
        if name == "hm_proj":
            gate = torch.sigmoid(tensor[sample_idx : sample_idx + 1])
            gate_img = _normalize_tensor(gate, reduce="mean").squeeze(0).squeeze(0).cpu().numpy()
            gate_mapped = cv2.applyColorMap(_to_uint8(gate_img), cv2.COLORMAP_PLASMA)
            _save_map_image(gate_mapped, out_dir / f"{stem}_tb_gate.png")


def prepare_batch(
    paths: List[pathlib.Path],
    transform: T.Compose,
) -> Tuple[torch.Tensor, List[np.ndarray], List[str]]:
    tensors: List[torch.Tensor] = []
    originals: List[np.ndarray] = []
    stems: List[str] = []
    for p in paths:
        with Image.open(p).convert("RGB") as img:
            original = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        tensor = transform(Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)))
        tensors.append(tensor)
        originals.append(original)
        stems.append(p.stem)
    batch = torch.stack(tensors, dim=0)
    return batch, originals, stems


def main():
    args = parse_args()
    build_transform(args.config_file, args.opts, args.weight)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("pose_vis", output_dir)
    logger.info(args)
    logger.info("Running with config:\n%s", cfg)

    transform = get_transform()
    model, device = load_model(args.weight)
    tb_writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=output_dir / "tb_logs")

    images = list_images(args.image_dir)
    if not images:
        logger.error("No images found in %s", args.image_dir)
        return

    logger.info("Found %d images, processing...", len(images))
    for start in tqdm(range(0, len(images), args.batch_size)):
        end = start + args.batch_size
        batch_paths = images[start:end]
        batch_tensor, originals, stems = prepare_batch(batch_paths, transform)
        batch_tensor = batch_tensor.to(device)

        base = getattr(model, "base", None)
        if base is not None and hasattr(base, "reset_pose_debug_epoch"):
            base.reset_pose_debug_epoch()

        with torch.no_grad():
            outputs = model(batch_tensor)

        hm_fullres = getattr(getattr(model, "base", model), "_hm_fullres", None)
        global_maps = outputs.get("global_maps") if isinstance(outputs, dict) else None
        local_maps = outputs.get("local_maps") if isinstance(outputs, dict) else None
        tb_cache = getattr(getattr(model, "base", model), "_tb_cache", {})

        for idx, (orig, stem) in enumerate(zip(originals, stems)):
            orig_hw = (orig.shape[0], orig.shape[1])
            if hm_fullres is not None:
                visualize_heatmaps(hm_fullres[idx : idx + 1], orig_hw, orig, stem, output_dir)
            if global_maps:
                visualize_feature_maps([m[idx : idx + 1] for m in global_maps], orig_hw, orig, stem, "global", output_dir)
            if local_maps:
                visualize_feature_maps([m[idx : idx + 1] for m in local_maps], orig_hw, orig, stem, "local", output_dir)
            if tb_cache:
                visualize_tb_cache(tb_cache, stem, output_dir, idx)

        if tb_writer is not None and base is not None and hasattr(base, "tb_dump_pose"):
            base.tb_dump_pose(tb_writer, step=start // args.batch_size + 1, tag_prefix="pose_vis")

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
