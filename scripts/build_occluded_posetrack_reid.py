#!/usr/bin/env python
"""Build Occluded-PoseTrack-ReID dataset from PoseTrack21 + KPR sampling annotations.

This standalone script converts PoseTrack21 tracking data into a ReID dataset
compatible with our SOLIDER-REID framework (same format as Occluded-Duke).

Usage:
    python scripts/build_occluded_posetrack_reid.py \
        --posetrack-root ~/work/data/PoseTrack21 \
        --output-dir ~/work/data/occluded_posetrack_reid

Output structure:
    occluded_posetrack_reid/
        bounding_box_train/     # Training person crops
        query/                  # Query images (most occluded per ID)
        bounding_box_test/      # Gallery images
        train.list
        query.list
        gallery.list
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_posetrack21_annotations(anns_path):
    """Load PoseTrack21 annotation JSONs into DataFrames."""
    anns_files = sorted(Path(anns_path).glob("*.json"))
    assert len(anns_files) > 0, f"No annotation files in {anns_path}"

    all_videos = []
    all_images = []
    all_detections = []

    for json_file in tqdm(anns_files, desc="Loading annotations"):
        with open(json_file) as f:
            data = json.load(f)

        video_name = json_file.stem  # e.g. "000001_bonn_train"

        # Parse images
        for img_info in data.get("images", []):
            img_info["_video_name"] = video_name
            all_images.append(img_info)

        # Parse annotations (detections)
        for ann in data.get("annotations", []):
            ann["_video_name"] = video_name
            all_detections.append(ann)

    return all_images, all_detections


def build_dataframes(images_raw, detections_raw, dataset_path, split_name):
    """Convert raw annotation lists to pandas DataFrames with proper formatting."""

    # Build image metadata
    image_records = []
    for img in images_raw:
        vid_id = img.get("vid_id", img["_video_name"])
        file_name = img["file_name"]
        # Make absolute path
        file_path = str(Path(dataset_path) / file_name)
        # Extract frame number from filename
        fname = Path(file_name).stem
        frame_match = re.search(r"(\d+)$", fname)
        frame = int(frame_match.group(1)) if frame_match else 0

        image_records.append({
            "id": img["id"],
            "file_path": file_path,
            "video_id": str(vid_id),
            "frame": frame,
            "_video_name": img["_video_name"],
        })

    images_df = pd.DataFrame(image_records)
    if len(images_df) > 0:
        images_df.set_index("id", drop=False, inplace=True)

    # Build detection metadata
    det_records = []
    for ann in detections_raw:
        kps_flat = ann.get("keypoints", [0] * 51)
        kps = np.array(kps_flat, dtype=np.float32).reshape(17, 3)

        bbox = np.array(ann["bbox"], dtype=np.float32)  # [l, t, w, h]

        # person_id is the cross-video identity (NOT track_id which resets per video)
        person_id = ann.get("person_id", ann.get("track_id", ann["id"]))

        visibility = float(kps[:, 2].mean())

        det_records.append({
            "id": ann["id"],
            "image_id": ann["image_id"],
            "person_id": person_id,
            "keypoints_xyc": kps,
            "bbox_ltwh": bbox,
            "visibility": visibility,
            "bbox_conf": ann.get("score", 1.0),
            "_video_name": ann["_video_name"],
        })

    dets_df = pd.DataFrame(det_records)
    if len(dets_df) > 0:
        dets_df.set_index("id", drop=False, inplace=True)

        # Add video_id from image metadata
        img_vid_map = images_df[["id", "video_id"]].set_index("id")["video_id"]
        dets_df["video_id"] = dets_df["image_id"].map(img_vid_map)

    return images_df, dets_df


def load_dataset_sampling(dets_df, sampling_path):
    """Load pre-computed dataset sampling from KPR's JSON file.

    The JSON format is: {"id": {"det_id_str": det_id_int, ...},
                         "split": {"det_id_str": "train"|"none"|"query"|"gallery", ...}}
    """
    if not os.path.exists(sampling_path):
        raise FileNotFoundError(f"Dataset sampling file not found: {sampling_path}")

    with open(sampling_path) as f:
        sampling_data = json.load(f)

    # Build a mapping: int detection_id -> split
    split_map = {}
    for det_id_str, split_val in sampling_data["split"].items():
        det_id_int = int(det_id_str)
        split_map[det_id_int] = split_val

    # Apply to dets_df (indexed by integer detection ID)
    dets_df["split"] = dets_df.index.map(lambda x: split_map.get(x, "none"))


def compute_negative_keypoints(dets_df):
    """Compute keypoints of other persons within each detection's bbox."""

    def add_negatives(group_df):
        all_kps = np.array(list(group_df.keypoints_xyc))  # (N, 17, 3)
        results = []
        for i in range(len(group_df)):
            other_kps = np.delete(all_kps, i, axis=0)  # (N-1, 17, 3)
            if len(other_kps) == 0:
                results.append(np.empty((0, 17, 3)))
                continue
            # Convert to bbox coords
            bbox = group_df.iloc[i].bbox_ltwh
            l, t, w, h = bbox
            neg = other_kps.copy()
            neg[:, :, 0] -= l
            neg[:, :, 1] -= t
            # Mark OOB keypoints as invisible
            for j in range(len(neg)):
                oob = (neg[j, :, 0] < 0) | (neg[j, :, 0] >= w) | \
                      (neg[j, :, 1] < 0) | (neg[j, :, 1] >= h)
                neg[j, oob, 2] = 0
            # Remove completely invisible skeletons
            visible_mask = neg[:, :, 2].sum(axis=1) > 0
            neg = neg[visible_mask]
            results.append(neg)
        group_df = group_df.copy()
        group_df["negative_kps"] = results
        return group_df

    dets_df_reid = dets_df[dets_df.split != "none"]
    result = dets_df_reid.groupby("image_id", group_keys=False).apply(add_negatives)
    dets_df.loc[result.index, "negative_kps"] = result["negative_kps"]
    # Fill NaN for non-reid detections
    dets_df["negative_kps"] = dets_df["negative_kps"].apply(
        lambda x: x if isinstance(x, np.ndarray) else np.empty((0, 17, 3))
    )


def compute_occlusion_level(row):
    """Compute occlusion level: ratio of negative to positive keypoint visibility."""
    pos_vis = row.keypoints_xyc[:, 2].sum()
    neg_vis = row.negative_kps[:, :, 2].sum() if len(row.negative_kps) > 0 else 0
    if pos_vis == 0:
        return neg_vis * 2
    return neg_vis / pos_vis


def query_gallery_split(dets_df, ratio=0.2):
    """Split test detections into query/gallery based on occlusion level."""
    dets_df.loc[dets_df.split != "none", "split"] = "gallery"

    def occlusion_sampling(group):
        group = group.sort_values(by="occ_level", ascending=False)
        n_query = max(1, int(len(group) * ratio))
        return group.iloc[:n_query]

    reid_dets = dets_df[dets_df.split != "none"]
    np.random.seed(0)
    queries = reid_dets.groupby("person_id", group_keys=False).apply(occlusion_sampling)
    dets_df.loc[queries.index, "split"] = "query"


def save_crops(dets_df, images_df, output_dir, split_name, max_crop_size=(384, 128)):
    """Extract and save person crops from original images."""
    max_h, max_w = max_crop_size
    reid_dets = dets_df[dets_df.split != "none"].copy()

    if len(reid_dets) == 0:
        print(f"  No detections for {split_name}")
        return

    # Assign 0-based PIDs
    pid_map = {pid: i for i, pid in enumerate(sorted(reid_dets.person_id.unique()))}
    reid_dets["pid"] = reid_dets.person_id.map(pid_map)

    # Assign camera IDs from video_id
    vid_map = {vid: i for i, vid in enumerate(sorted(reid_dets.video_id.unique()))}
    reid_dets["camid"] = reid_dets.video_id.map(vid_map)

    # Group by image for efficient image loading
    grouped = reid_dets.groupby("image_id")

    saved_files = defaultdict(list)  # split -> list of filenames

    for image_id, group in tqdm(grouped, desc=f"Extracting {split_name} crops"):
        # Get image path
        if image_id not in images_df.index:
            continue
        img_meta = images_df.loc[image_id]
        if isinstance(img_meta, pd.DataFrame):
            img_meta = img_meta.iloc[0]

        img = cv2.imread(img_meta.file_path)
        if img is None:
            print(f"  WARNING: Cannot read {img_meta.file_path}")
            continue

        img_h, img_w = img.shape[:2]

        for _, det in group.iterrows():
            bbox = det.bbox_ltwh.astype(int)
            l, t, w, h = bbox

            # Clip to image bounds
            l = max(0, l)
            t = max(0, t)
            w = min(w, img_w - l)
            h = min(h, img_h - t)
            if w <= 0 or h <= 0:
                continue

            crop = img[t:t+h, l:l+w]

            # Track actual crop size before resize
            crop_h, crop_w = crop.shape[:2]

            if crop_h > max_h or crop_w > max_w:
                crop = cv2.resize(crop, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
                crop_h, crop_w = max_h, max_w

            # Determine output split directory
            det_split = det.split  # "train", "query", or "gallery"
            if det_split == "train":
                out_subdir = "bounding_box_train"
            elif det_split == "query":
                out_subdir = "query"
            else:
                out_subdir = "bounding_box_test"

            out_dir = os.path.join(output_dir, out_subdir)
            os.makedirs(out_dir, exist_ok=True)

            # Naming convention: {pid:04d}_c{camid+1}_{det_id}.jpg
            # Use detection ID directly for uniqueness
            det_id = int(det.id) if isinstance(det.id, (int, np.integer)) else hash(str(det.id))
            filename = f"{det.pid:04d}_c{det.camid + 1}_{det_id}.jpg"

            out_path = os.path.join(out_dir, filename)
            cv2.imwrite(out_path, crop)
            saved_files[det_split].append(filename)

    return saved_files


def write_list_files(output_dir, saved_files):
    """Write train.list, query.list, gallery.list files."""
    for split_name, filenames in saved_files.items():
        if split_name == "train":
            list_file = os.path.join(output_dir, "train.list")
        elif split_name == "query":
            list_file = os.path.join(output_dir, "query.list")
        else:
            list_file = os.path.join(output_dir, "gallery.list")

        filenames_sorted = sorted(filenames)
        with open(list_file, "w") as f:
            for fn in filenames_sorted:
                f.write(fn + "\n")
        print(f"  Wrote {list_file}: {len(filenames_sorted)} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Build Occluded-PoseTrack-ReID from PoseTrack21")
    parser.add_argument("--posetrack-root", type=str, required=True,
                        help="Path to PoseTrack21 dataset root")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for ReID dataset")
    parser.add_argument("--kpr-annotations", type=str, default=None,
                        help="Path to KPR's occluded_posetrack_reid dir (with sampling JSONs)")
    parser.add_argument("--max-crop-h", type=int, default=384)
    parser.add_argument("--max-crop-w", type=int, default=128)
    parser.add_argument("--query-ratio", type=float, default=0.2)
    args = parser.parse_args()

    pt_root = Path(args.posetrack_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Annotation paths
    ann_path = pt_root / "posetrack_data"
    assert ann_path.exists(), f"Annotations not found: {ann_path}"

    # KPR sampling annotations (optional)
    kpr_dir = None
    if args.kpr_annotations:
        kpr_dir = Path(args.kpr_annotations)
    else:
        # Try default location inside PoseTrack21
        kpr_default = pt_root / "occluded_posetrack_reid"
        if kpr_default.exists():
            kpr_dir = kpr_default

    # Process train and val splits
    all_saved_files = defaultdict(list)

    for pt_split, reid_splits in [("train", ["train"]), ("val", ["query", "gallery"])]:
        print(f"\n{'='*60}")
        print(f"Processing PoseTrack21 {pt_split} split -> ReID {reid_splits}")
        print(f"{'='*60}")

        split_ann_path = ann_path / pt_split
        if not split_ann_path.exists():
            print(f"  WARNING: {split_ann_path} not found, skipping")
            continue

        # Load annotations
        images_raw, dets_raw = load_posetrack21_annotations(split_ann_path)
        print(f"  Loaded {len(images_raw)} images, {len(dets_raw)} detections")

        # Build DataFrames
        images_df, dets_df = build_dataframes(
            images_raw, dets_raw, str(pt_root), pt_split)
        print(f"  Built DataFrames: {len(images_df)} images, {len(dets_df)} detections")

        # Load or compute sampling
        if kpr_dir:
            sampling_file = kpr_dir / f"{pt_split}_dataset_sampling.json"
            if sampling_file.exists():
                print(f"  Loading KPR sampling: {sampling_file}")
                load_dataset_sampling(dets_df, str(sampling_file))
            else:
                print(f"  WARNING: KPR sampling not found at {sampling_file}")
                print(f"  Will use default sampling (all detections with vis>0.3)")
                dets_df["split"] = "none"
                mask = dets_df.visibility >= 0.3
                mask &= dets_df.bbox_ltwh.apply(lambda x: x[2] > 10 and x[3] > 10)
                dets_df.loc[mask, "split"] = pt_split
        else:
            print("  No KPR annotations, using default sampling")
            dets_df["split"] = "none"
            mask = dets_df.visibility >= 0.3
            mask &= dets_df.bbox_ltwh.apply(lambda x: x[2] > 10 and x[3] > 10)
            dets_df.loc[mask, "split"] = pt_split

        n_reid = (dets_df.split != "none").sum()
        print(f"  ReID detections: {n_reid} / {len(dets_df)}")

        if n_reid == 0:
            print("  No detections selected, skipping")
            continue

        # For val split: check if KPR already provides query/gallery assignments
        if pt_split == "val":
            n_query = (dets_df.split == "query").sum()
            n_gallery = (dets_df.split == "gallery").sum()
            if n_query > 0 and n_gallery > 0:
                print(f"  KPR query/gallery split loaded: {n_query} queries, {n_gallery} gallery")
            else:
                # Fall back to computing query/gallery from occlusion
                print("  Computing negative keypoints for occlusion scoring...")
                compute_negative_keypoints(dets_df)
                dets_df["occ_level"] = dets_df.apply(compute_occlusion_level, axis=1)
                query_gallery_split(dets_df, ratio=args.query_ratio)
                n_query = (dets_df.split == "query").sum()
                n_gallery = (dets_df.split == "gallery").sum()
                print(f"  Computed query/gallery split: {n_query} queries, {n_gallery} gallery")

        # Save crops
        print("  Extracting image crops...")
        saved = save_crops(
            dets_df, images_df, str(output_dir), pt_split,
            max_crop_size=(args.max_crop_h, args.max_crop_w))

        if saved:
            for split, files in saved.items():
                all_saved_files[split].extend(files)

    # Write list files
    print(f"\n{'='*60}")
    print("Writing list files...")
    write_list_files(str(output_dir), all_saved_files)

    # Print statistics
    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    for split in ["train", "query", "gallery"]:
        n = len(all_saved_files.get(split, []))
        print(f"  {split}: {n} images")

    # Count unique PIDs per split
    for split_name, subdir in [("train", "bounding_box_train"),
                                ("query", "query"),
                                ("gallery", "bounding_box_test")]:
        dir_path = output_dir / subdir
        if dir_path.exists():
            pids = set()
            for f in dir_path.iterdir():
                if f.suffix == ".jpg":
                    pid = int(f.stem.split("_")[0])
                    pids.add(pid)
            print(f"  {split_name}: {len(pids)} unique PIDs")

    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
