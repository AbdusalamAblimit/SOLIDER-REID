"""Extract per-person pose keypoints + heatmaps using RTMDet + ViTPose-Huge.

Top-down pipeline:
1. RTMDet detects person bboxes
2. ViTPose extracts per-person heatmaps and keypoints from each bbox crop

Saves per image per person:
  - heatmaps/<split>/<filename>_p<i>.npy: (17, 64, 48) float32 heatmaps
And per split:
  - pose_data_<split>.pkl: all persons' info (bboxes, keypoints, scores, heatmap paths)
"""
import os
import sys
import pickle
import numpy as np
import cv2

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

import torch
from mmengine.registry import DefaultScope

import mmdet
from mmdet.apis import inference_detector, init_detector
import mmpose
from mmpose.apis import inference_topdown, init_model


def extract_heatmap_crop(pose_model, img_bgr, bbox):
    """Extract raw heatmaps for a specific person bbox crop.

    Args:
        pose_model: ViTPose model
        img_bgr: full image in BGR (numpy array)
        bbox: [x1, y1, x2, y2] in pixel coords

    Returns:
        heatmaps: (17, 64, 48) float32 in bbox-local coordinate space
    """
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((17, 64, 48), dtype=np.float32)

    crop_resized = cv2.resize(crop, (192, 256))  # ViTPose input: 192x256 (WxH)
    crop_tensor = (torch.from_numpy(crop_resized).permute(2, 0, 1)
                   .float().unsqueeze(0).cuda() / 255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    crop_tensor = (crop_tensor - mean) / std

    with torch.no_grad():
        feats = pose_model.extract_feat(crop_tensor)
        heatmaps = pose_model.head.forward(feats)  # (1, 17, 64, 48)

    return heatmaps[0].cpu().numpy().astype(np.float32)


def extract_split(pose_model, det_model, img_dir, output_path,
                  heatmap_dir, det_score_thr=0.3):
    """Extract per-person keypoints + heatmaps for all images."""
    filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    n_total = len(filenames)
    print(f"Processing {n_total} images from {img_dir}", flush=True)

    os.makedirs(heatmap_dir, exist_ok=True)

    # Per-image data: list of dicts
    all_data = []

    for idx, fname in enumerate(filenames):
        if idx % 500 == 0:
            print(f"  [{idx}/{n_total}] ({100*idx/n_total:.1f}%)", flush=True)

        img_path = os.path.join(img_dir, fname)
        img_bgr = cv2.imread(img_path)
        h, w = img_bgr.shape[:2]

        # 1. Detect all persons
        with DefaultScope.overwrite_default_scope('mmdet'):
            det_result = inference_detector(det_model, img_path)
        pred_instances = det_result.pred_instances
        person_mask = pred_instances.labels == 0
        score_mask = pred_instances.scores >= det_score_thr
        mask = person_mask & score_mask
        bboxes = pred_instances.bboxes[mask].cpu().numpy()
        bbox_scores = pred_instances.scores[mask].cpu().numpy()

        # If no person detected, use full image as fallback
        if len(bboxes) == 0:
            bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
            bbox_scores = np.array([1.0], dtype=np.float32)

        # Sort by bbox area (largest first = most likely the target person)
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        sort_idx = np.argsort(-areas)
        bboxes = bboxes[sort_idx]
        bbox_scores = bbox_scores[sort_idx]

        # 2. Per-person pose estimation (keypoints via inference_topdown)
        with DefaultScope.overwrite_default_scope('mmpose'):
            pose_results = inference_topdown(pose_model, img_path, bboxes=bboxes)

        persons = []
        for j in range(len(bboxes)):
            # Keypoints from inference_topdown
            if j < len(pose_results):
                kpts = pose_results[j].pred_instances.keypoints[0]
                kp_scores = pose_results[j].pred_instances.keypoint_scores[0]
                kpts_norm = kpts.copy()
                kpts_norm[:, 0] /= w
                kpts_norm[:, 1] /= h
            else:
                kpts_norm = np.zeros((17, 2), dtype=np.float32)
                kp_scores = np.zeros(17, dtype=np.float32)

            # Raw heatmap from bbox crop
            heatmaps = extract_heatmap_crop(pose_model, img_bgr, bboxes[j])
            hm_filename = fname.replace('.jpg', f'_p{j}.npy')
            hm_path = os.path.join(heatmap_dir, hm_filename)
            np.save(hm_path, heatmaps)

            persons.append({
                'bbox': bboxes[j].astype(np.float32),
                'bbox_score': float(bbox_scores[j]),
                'keypoints': kpts_norm.astype(np.float32),
                'kp_scores': kp_scores.astype(np.float32),
                'heatmap_file': hm_filename,
            })

        all_data.append({
            'filename': fname,
            'num_persons': len(persons),
            'persons': persons,
        })

    # Save all data as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(all_data, f)

    # Also save primary person (p0) as backward-compatible NPZ
    primary_kpts = np.array([d['persons'][0]['keypoints'] for d in all_data],
                            dtype=np.float32)
    primary_scores = np.array([d['persons'][0]['kp_scores'] for d in all_data],
                              dtype=np.float32)
    primary_filenames = np.array([d['filename'] for d in all_data])
    npz_path = output_path.replace('.pkl', '.npz')
    np.savez_compressed(npz_path,
                        filenames=primary_filenames,
                        keypoints=primary_kpts,
                        scores=primary_scores)

    # Stats
    n_multi = sum(1 for d in all_data if d['num_persons'] > 1)
    n_persons_total = sum(d['num_persons'] for d in all_data)
    print(f"Saved {n_total} images to {output_path}", flush=True)
    print(f"  Primary keypoints: {primary_kpts.shape}", flush=True)
    print(f"  Score stats: mean={primary_scores.mean():.3f}, "
          f"min={primary_scores.min():.3f}, max={primary_scores.max():.3f}",
          flush=True)
    print(f"  Images with >1 person: {n_multi}/{n_total}", flush=True)
    print(f"  Total persons: {n_persons_total} "
          f"(avg {n_persons_total/n_total:.1f}/image)", flush=True)


def main():
    pose_config = 'pretrained/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py'
    pose_ckpt = ('pretrained/td-hm_ViTPose-huge_8xb64-210e_coco-'
                 '256x192-e32adcd4_20230314.pth')
    det_config = 'pretrained/rtmdet_s_8xb32-300e_coco.py'
    det_ckpt = ('pretrained/rtmdet_s_8xb32-300e_coco_20220905_161602-'
                '387a891e.pth')

    print("Loading ViTPose-Huge (heatmap) + RTMDet-s...", flush=True)
    with DefaultScope.overwrite_default_scope('mmpose'):
        pose_model = init_model(pose_config, pose_ckpt, device='cuda:0')
        pose_model.eval()
    with DefaultScope.overwrite_default_scope('mmdet'):
        det_model = init_detector(det_config, det_ckpt, device='cuda:0')
    print("Models loaded.", flush=True)

    data_root = 'data/occluded_duke'
    heatmap_base = os.path.join(data_root, 'heatmaps')
    splits = {
        'train': 'bounding_box_train',
        'gallery': 'bounding_box_test',
        'query': 'query',
    }

    for split_name, subdir in splits.items():
        img_dir = os.path.join(data_root, subdir)
        output_path = os.path.join(data_root, f'pose_data_{split_name}.pkl')
        heatmap_dir = os.path.join(heatmap_base, split_name)
        extract_split(pose_model, det_model, img_dir, output_path, heatmap_dir)

    print("\nDone! All splits extracted.", flush=True)


if __name__ == '__main__':
    main()
