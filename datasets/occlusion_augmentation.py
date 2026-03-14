"""
Realistic Occlusion Augmentation (ROA)

Loads RGBA object patches from Pascal VOC 2012 and pastes them onto
training images with alpha blending. Replaces random erasing with
semantically meaningful occlusion simulation.

Based on: https://github.com/isarandi/synthetic-occlusion
"""

import math
import os
import random
import xml.etree.ElementTree
import numpy as np
import cv2
import PIL.Image


def load_occluders(pascal_voc_root_path, classes_filter=None):
    """Load RGBA occluder patches from Pascal VOC segmentation data.

    Returns a list of RGBA numpy arrays (H, W, 4) with soft alpha edges.
    """
    occluders = []
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    if classes_filter is None:
        classes_filter = ["person", "bicycle", "boat", "bus", "car", "motorbike", "train"]

    ann_dir = os.path.join(pascal_voc_root_path, 'Annotations')
    annotation_paths = sorted([
        os.path.join(ann_dir, f) for f in os.listdir(ann_dir)
        if f.endswith('.xml')
    ])

    for annotation_path in annotation_paths:
        xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
        is_segmented = (xml_root.find('segmented').text != '0')
        if not is_segmented:
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall('object')):
            is_authorized_class = (obj.find('name').text in classes_filter)
            is_difficult = (obj.find('difficult').text != '0')
            is_truncated = (obj.find('truncated').text != '0')
            if is_authorized_class and not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))

        if not boxes:
            continue

        im_filename = xml_root.find('filename').text
        seg_filename = im_filename.replace('jpg', 'png')
        im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
        seg_path = os.path.join(pascal_voc_root_path, 'SegmentationObject', seg_filename)

        if not os.path.exists(seg_path):
            continue

        im = np.asarray(PIL.Image.open(im_path))
        labels = np.asarray(PIL.Image.open(seg_path))

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8) * 255
            object_image = im[ymin:ymax, xmin:xmax]
            if cv2.countNonZero(object_mask) < 500:
                continue

            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            object_with_mask = np.concatenate(
                [object_image, object_mask[..., np.newaxis]], axis=-1)

            # Downscale for efficiency
            object_with_mask = _resize_by_factor(object_with_mask, 0.5)
            occluders.append(object_with_mask)

    return occluders


def occlude_with_objects(im, occluders, n=1, min_overlap=0.2, max_overlap=0.6):
    """Paste random occluders onto image with alpha blending.

    Args:
        im: numpy array (H, W, 3), will be modified in-place
        occluders: list of RGBA patches from load_occluders()
        n: max number of occluders to paste
        min_overlap/max_overlap: fraction of image area covered by occluder

    Returns:
        im: modified image (same object, modified in-place)
    """
    result = im.copy()
    width_height = np.asarray([im.shape[1], im.shape[0]])
    im_area = im.shape[1] * im.shape[0]
    count = np.random.randint(1, n + 1)

    for _ in range(count):
        occluder = random.choice(occluders)
        occluder_area = occluder.shape[1] * occluder.shape[0]
        if occluder_area < 1:
            continue
        overlap = random.uniform(min_overlap, max_overlap)
        scale_factor = math.sqrt(overlap * im_area / occluder_area)
        occluder = _resize_by_factor(occluder, scale_factor)
        center = np.random.uniform([0, 0], width_height)
        _paste_over(im_src=occluder, im_dst=result, center=center)

    return result


def _paste_over(im_src, im_dst, center):
    """Paste RGBA image onto RGB image with alpha blending."""
    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)

    if (end_dst - start_dst).min() <= 0:
        return

    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]

    if region_src.shape[0] == 0 or region_src.shape[1] == 0:
        return

    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32) / 255

    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
        alpha * color_src + (1 - alpha) * region_dst).astype(np.uint8)


def _resize_by_factor(im, factor):
    """Resize image by factor."""
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    if new_size[0] < 1 or new_size[1] < 1:
        return im
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)
