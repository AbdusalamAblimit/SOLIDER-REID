#!/usr/bin/env python3
"""Static contracts for the exp392 Phase 0B2-S teacher audit.

This module contains deterministic counterfactual builders shared by the
teacher-only audit.  Its executable contract uses synthetic tensors only: it
does not read ReID data, load CLIP, construct an optimizer, or access CUDA.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


REGIONS = 5
LEVELS = (0.25, 0.50, 0.75)
Y_BINS = 8
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def seed_from_text(text):
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _generator(key):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_from_text(key))
    return generator


def deterministic_balanced_targets(valid, paths, seed):
    """Choose one valid slot per image without reading any model output."""
    valid = torch.as_tensor(valid, dtype=torch.bool)
    if valid.ndim != 2 or valid.shape[1] != REGIONS:
        raise ValueError("valid must have shape [N, 5]")
    if len(paths) != len(valid):
        raise ValueError("paths and valid length mismatch")
    ordered = sorted(
        range(len(paths)),
        key=lambda index: hashlib.sha256(
            (str(paths[index]) + "\0" + str(seed)).encode("utf-8")
        ).hexdigest(),
    )
    counts = [0] * REGIONS
    targets = torch.full((len(paths),), -1, dtype=torch.long)
    for index in ordered:
        choices = [
            region for region in range(REGIONS)
            if bool(valid[index, region])
        ]
        if not choices:
            continue
        start = seed_from_text("%s\0%s" % (paths[index], seed)) % REGIONS
        cyclic_rank = {
            region: (region - start) % REGIONS for region in choices
        }
        target = min(
            choices, key=lambda region: (counts[region], cyclic_rank[region])
        )
        targets[index] = target
        counts[target] += 1
    return targets, torch.as_tensor(counts, dtype=torch.long)


# Historical salt-and-pepper helpers are kept private only to preserve an
# executable negative contract during review.  B2-S must not import them;
# static_contract() exclusively exercises the connected-rectangle path below.
def _legacy_ordered_coordinates(mask, key):
    coordinates = torch.as_tensor(mask, dtype=torch.bool).nonzero(
        as_tuple=False
    )
    if len(coordinates) == 0:
        raise ValueError("support mask is empty")
    order = torch.randperm(len(coordinates), generator=_generator(key))
    return coordinates.index_select(0, order)


def _legacy_nested_support_sets(mask, levels, key):
    mask = torch.as_tensor(mask, dtype=torch.bool)
    coordinates = _legacy_ordered_coordinates(mask, key)
    count = len(coordinates)
    sets = []
    realized = []
    errors = []
    for level in levels:
        selected = int(round(float(level) * count))
        current = torch.zeros_like(mask)
        if selected:
            chosen = coordinates[:selected]
            current[chosen[:, 0], chosen[:, 1]] = True
        sets.append(current)
        value = selected / float(count)
        realized.append(value)
        errors.append(abs(value - float(level)))
    return {
        "sets": torch.stack(sets),
        "ordered_coordinates": coordinates,
        "realized": torch.as_tensor(realized, dtype=torch.float64),
        "errors": torch.as_tensor(errors, dtype=torch.float64),
        "support_pixels": int(count),
    }


def _dilate(mask, radius):
    mask = torch.as_tensor(mask, dtype=torch.float32)[None, None]
    return F.max_pool2d(
        mask,
        kernel_size=2 * int(radius) + 1,
        stride=1,
        padding=int(radius),
    )[0, 0].bool()


def _legacy_matched_nonoverlap_sets(
    target_mask,
    content_mask,
    target_order,
    levels,
    key,
    exclusion_radius=24,
    y_bins=Y_BINS,
):
    """Map every selected target pixel to a zero-overlap, y-matched control."""
    target_mask = torch.as_tensor(target_mask, dtype=torch.bool)
    content_mask = torch.as_tensor(content_mask, dtype=torch.bool)
    if target_mask.shape != content_mask.shape:
        raise ValueError("target/content shape mismatch")
    maximum = int(round(max(float(level) for level in levels) * len(target_order)))
    target_order = target_order[:maximum]
    eligible = content_mask & (~_dilate(target_mask, exclusion_radius))
    candidates = eligible.nonzero(as_tuple=False)
    if len(candidates) < maximum:
        raise ValueError("insufficient zero-overlap control pixels")
    candidate_order = torch.randperm(
        len(candidates), generator=_generator(str(key) + "\0candidate")
    )
    candidates = candidates.index_select(0, candidate_order)
    height = target_mask.shape[0]
    target_bins = torch.clamp(
        target_order[:, 0] * int(y_bins) // max(height, 1),
        max=int(y_bins) - 1,
    )
    candidate_bins = torch.clamp(
        candidates[:, 0] * int(y_bins) // max(height, 1),
        max=int(y_bins) - 1,
    )
    mapping = torch.full_like(target_order, -1)
    used = torch.zeros(len(candidates), dtype=torch.bool)
    for bin_index in range(int(y_bins)):
        target_positions = (target_bins == bin_index).nonzero(
            as_tuple=False
        ).flatten()
        candidate_positions = (candidate_bins == bin_index).nonzero(
            as_tuple=False
        ).flatten()
        take = min(len(target_positions), len(candidate_positions))
        if take:
            mapping[target_positions[:take]] = candidates[
                candidate_positions[:take]
            ]
            used[candidate_positions[:take]] = True
    missing = (mapping[:, 0] < 0).nonzero(as_tuple=False).flatten()
    remaining = (~used).nonzero(as_tuple=False).flatten()
    if len(remaining) < len(missing):
        raise RuntimeError("control assignment exhausted unexpectedly")
    if len(missing):
        mapping[missing] = candidates[remaining[:len(missing)]]
    if bool((mapping < 0).any()):
        raise RuntimeError("control assignment incomplete")

    sets = []
    for level in levels:
        selected = int(round(float(level) * len(_legacy_ordered_coordinates(
            target_mask, key
        ))))
        current = torch.zeros_like(target_mask)
        chosen = mapping[:selected]
        if len(chosen):
            current[chosen[:, 0], chosen[:, 1]] = True
        sets.append(current)
    sets = torch.stack(sets)
    target_y = target_order[:, 0].float() / max(height - 1, 1)
    control_y = mapping[:, 0].float() / max(height - 1, 1)
    return {
        "sets": sets,
        "ordered_coordinates": mapping,
        "eligible_pixels": int(len(candidates)),
        "mean_abs_y_error": float((target_y - control_y).abs().mean()),
    }


def _box_mask(shape, box):
    y0, x0, y1, x1 = (int(value) for value in box)
    result = torch.zeros(shape, dtype=torch.bool)
    result[y0:y1, x0:x1] = True
    return result


def connected_occlusion_rectangles(mask, levels, key):
    """Grow one connected rectangle from a deterministic bbox side."""
    mask = torch.as_tensor(mask, dtype=torch.bool)
    y0, x0, y1, x1 = _bbox(mask)
    directions = ("top", "bottom", "left", "right")
    direction = directions[seed_from_text(str(key) + "\0direction") % 4]
    vertical = direction in ("top", "bottom")
    extent = (y1 - y0) if vertical else (x1 - x0)
    support_pixels = int(mask.sum())
    if support_pixels <= 0 or extent <= 0:
        raise ValueError("invalid connected occlusion support")

    candidates = []
    immediately_previous_overlap = 0
    for step in range(1, extent + 1):
        if direction == "top":
            box = (y0, x0, y0 + step, x1)
        elif direction == "bottom":
            box = (y1 - step, x0, y1, x1)
        elif direction == "left":
            box = (y0, x0, y1, x0 + step)
        else:
            box = (y0, x1 - step, y1, x1)
        current = _box_mask(mask.shape, box)
        overlap = int((current & mask).sum())
        if overlap > immediately_previous_overlap:
            candidates.append((
                step,
                box,
                current,
                overlap,
                overlap - immediately_previous_overlap,
            ))
        immediately_previous_overlap = overlap
    boxes = []
    sets = []
    realized = []
    increments = []
    previous_step = 0
    previous_overlap = 0
    for level in levels:
        chosen = None
        for step, box, current, overlap, last_strip_pixels in candidates:
            if step <= previous_step or overlap <= previous_overlap:
                continue
            if overlap / float(support_pixels) >= float(level):
                chosen = (
                    step, box, current, overlap, last_strip_pixels
                )
                break
        if chosen is None:
            raise ValueError("could not realize connected overlap level")
        step, box, current, overlap, last_strip_pixels = chosen
        boxes.append(box)
        sets.append(current)
        realized.append(overlap / float(support_pixels))
        increments.append(last_strip_pixels / float(support_pixels))
        previous_step = step
        previous_overlap = overlap
    return {
        "direction": direction,
        "boxes": boxes,
        "sets": torch.stack(sets),
        "realized": torch.as_tensor(realized, dtype=torch.float64),
        "last_increment_fraction": torch.as_tensor(
            increments, dtype=torch.float64
        ),
        "support_pixels": support_pixels,
        "support_box": (y0, x0, y1, x1),
    }


def translated_nonoverlap_rectangles(
    target_mask,
    overlap_boxes,
    direction,
    exclusion_radius=24,
):
    """Translate the largest rectangle once, then preserve nested alignment."""
    target_mask = torch.as_tensor(target_mask, dtype=torch.bool)
    maximum = overlap_boxes[-1]
    max_height = int(maximum[2] - maximum[0])
    max_width = int(maximum[3] - maximum[1])
    forbidden = _dilate(target_mask, exclusion_radius).to(torch.int64)
    integral = torch.zeros(
        forbidden.shape[0] + 1,
        forbidden.shape[1] + 1,
        dtype=torch.int64,
    )
    integral[1:, 1:] = forbidden.cumsum(0).cumsum(1)
    intersections = (
        integral[max_height:, max_width:]
        - integral[:-max_height, max_width:]
        - integral[max_height:, :-max_width]
        + integral[:-max_height, :-max_width]
    )
    candidates = (intersections == 0).nonzero(as_tuple=False)
    if len(candidates) == 0:
        raise ValueError("no translated non-overlap rectangle exists")
    target_center_y = 0.5 * (maximum[0] + maximum[2] - 1)
    height = target_mask.shape[0]
    ranked = sorted(
        (tuple(int(value) for value in coordinate.tolist()) for coordinate in candidates),
        key=lambda coordinate: (
            abs(
                (coordinate[0] + 0.5 * (max_height - 1) - target_center_y)
                / max(height - 1, 1)
            ),
            coordinate[0],
            coordinate[1],
        ),
    )
    anchor_y, anchor_x = ranked[0]
    boxes = []
    sets = []
    for box in overlap_boxes:
        current_height = int(box[2] - box[0])
        current_width = int(box[3] - box[1])
        if direction == "top":
            y0 = anchor_y
            x0 = anchor_x
        elif direction == "bottom":
            y0 = anchor_y + max_height - current_height
            x0 = anchor_x
        elif direction == "left":
            y0 = anchor_y
            x0 = anchor_x
        elif direction == "right":
            y0 = anchor_y
            x0 = anchor_x + max_width - current_width
        else:
            raise ValueError("unknown sweep direction")
        translated = (y0, x0, y0 + current_height, x0 + current_width)
        boxes.append(translated)
        sets.append(_box_mask(target_mask.shape, translated))
    sets = torch.stack(sets)
    dilated = _dilate(target_mask, exclusion_radius)
    if bool((sets & dilated[None]).any()):
        raise RuntimeError("translated rectangle intersects target dilation")
    control_center_y = 0.5 * (boxes[-1][0] + boxes[-1][2] - 1)
    return {
        "boxes": boxes,
        "sets": sets,
        "normalized_y_error": abs(control_center_y - target_center_y)
        / max(height - 1, 1),
        "exclusion_radius": int(exclusion_radius),
    }


def _gaussian_kernel(sigma, dtype):
    radius = int(math.ceil(3.0 * float(sigma)))
    axis = torch.arange(-radius, radius + 1, dtype=dtype)
    kernel = torch.exp(-0.5 * (axis / float(sigma)).square())
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)


def _legacy_deterministic_random_texture(
    image, support_mask, key, sigma=1.5
):
    image = torch.as_tensor(image, dtype=torch.float32)
    support_mask = torch.as_tensor(support_mask, dtype=torch.bool)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be [3,H,W]")
    if not bool(support_mask.any()):
        raise ValueError("support mask is empty")
    noise = torch.randn(
        image.shape, generator=_generator(str(key) + "\0texture")
    )
    kernel = _gaussian_kernel(sigma, noise.dtype)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)
    radius = kernel.shape[-1] // 2
    noise = F.conv2d(
        noise[None], kernel, padding=radius, groups=3
    )[0]
    result = torch.empty_like(noise)
    for channel in range(3):
        source = image[channel][support_mask]
        generated = noise[channel][support_mask]
        source_mean = source.mean()
        source_std = source.std(unbiased=False).clamp_min(0.02)
        generated_mean = generated.mean()
        generated_std = generated.std(unbiased=False).clamp_min(1e-6)
        result[channel] = (
            (noise[channel] - generated_mean)
            / generated_std
            * source_std
            + source_mean
        )
    return result.clamp(0.0, 1.0)


def random_occluder_texture(image, statistics_mask, size, key, sigma=1.5):
    """Create one frozen texture tensor shared by all overlap levels."""
    image = torch.as_tensor(image, dtype=torch.float32)
    statistics_mask = torch.as_tensor(statistics_mask, dtype=torch.bool)
    height, width = (int(value) for value in size)
    if height <= 0 or width <= 0 or not bool(statistics_mask.any()):
        raise ValueError("invalid random occluder shape/statistics mask")
    noise = torch.randn(
        3, height, width,
        generator=_generator(str(key) + "\0occluder-texture"),
    )
    kernel = _gaussian_kernel(sigma, noise.dtype)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)
    radius = kernel.shape[-1] // 2
    noise = F.conv2d(noise[None], kernel, padding=radius, groups=3)[0]
    source = image[:, statistics_mask]
    result = torch.empty_like(noise)
    for channel in range(3):
        source_mean = source[channel].mean()
        source_std = source[channel].std(unbiased=False).clamp_min(0.02)
        generated = noise[channel]
        result[channel] = (
            (generated - generated.mean())
            / generated.std(unbiased=False).clamp_min(1e-6)
            * source_std
            + source_mean
        )
    return result.clamp(0.0, 1.0)


def _bbox(mask):
    coordinates = torch.as_tensor(mask, dtype=torch.bool).nonzero(
        as_tuple=False
    )
    if len(coordinates) == 0:
        raise ValueError("bbox mask is empty")
    lower = coordinates.amin(0)
    upper = coordinates.amax(0) + 1
    return tuple(int(value) for value in (*lower.tolist(), *upper.tolist()))


def _legacy_donor_slot_texture(donor_image, donor_mask, recipient_mask):
    donor_image = torch.as_tensor(donor_image, dtype=torch.float32)
    dy0, dx0, dy1, dx1 = _bbox(donor_mask)
    ry0, rx0, ry1, rx1 = _bbox(recipient_mask)
    crop = donor_image[:, dy0:dy1, dx0:dx1][None]
    target_height = ry1 - ry0
    target_width = rx1 - rx0
    scale = min(
        target_height / float(max(dy1 - dy0, 1)),
        target_width / float(max(dx1 - dx0, 1)),
    )
    resized_height = max(int(round((dy1 - dy0) * scale)), 1)
    resized_width = max(int(round((dx1 - dx0) * scale)), 1)
    resized = F.interpolate(
        crop,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0].clamp(0.0, 1.0)
    texture = torch.as_tensor(CLIP_MEAN, dtype=torch.float32).view(
        3, 1, 1
    ).expand_as(donor_image).clone()
    top = ry0 + (target_height - resized_height) // 2
    left = rx0 + (target_width - resized_width) // 2
    texture[:, top:top + resized_height, left:left + resized_width] = resized
    return texture


def donor_slot_occluder_texture(donor_image, donor_mask, size):
    donor_image = torch.as_tensor(donor_image, dtype=torch.float32)
    y0, x0, y1, x1 = _bbox(donor_mask)
    target_height, target_width = (int(value) for value in size)
    crop = donor_image[:, y0:y1, x0:x1][None]
    scale = min(
        target_height / float(max(y1 - y0, 1)),
        target_width / float(max(x1 - x0, 1)),
    )
    resized_height = max(int(round((y1 - y0) * scale)), 1)
    resized_width = max(int(round((x1 - x0) * scale)), 1)
    resized = F.interpolate(
        crop,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0].clamp(0.0, 1.0)
    texture = torch.as_tensor(CLIP_MEAN, dtype=torch.float32).view(
        3, 1, 1
    ).expand(3, target_height, target_width).clone()
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    texture[:, top:top + resized_height, left:left + resized_width] = resized
    return texture


def _aligned_texture_crop(texture, size, direction):
    height, width = (int(value) for value in size)
    if direction in ("top", "left"):
        return texture[:, :height, :width]
    if direction == "bottom":
        return texture[:, -height:, :width]
    if direction == "right":
        return texture[:, :height, -width:]
    raise ValueError("unknown sweep direction")


def apply_rectangle_texture(image, box, texture, direction):
    image = torch.as_tensor(image, dtype=torch.float32)
    y0, x0, y1, x1 = (int(value) for value in box)
    current = _aligned_texture_crop(
        torch.as_tensor(texture, dtype=torch.float32),
        (y1 - y0, x1 - x0),
        direction,
    )
    result = image.clone()
    result[:, y0:y1, x0:x1] = current
    return result


def _legacy_apply_texture(image, selected, texture):
    image = torch.as_tensor(image, dtype=torch.float32)
    selected = torch.as_tensor(selected, dtype=torch.bool)
    texture = torch.as_tensor(texture, dtype=torch.float32)
    if image.shape != texture.shape or image.shape[-2:] != selected.shape:
        raise ValueError("texture application shape mismatch")
    result = image.clone()
    result[:, selected] = texture[:, selected]
    return result


def _legacy_apply_mapped_texture(
    image, destination_coordinates, source_coordinates, texture, count
):
    """Write the same ordered material values at a counterfactual location."""
    image = torch.as_tensor(image, dtype=torch.float32)
    destination_coordinates = torch.as_tensor(
        destination_coordinates, dtype=torch.long
    )
    source_coordinates = torch.as_tensor(source_coordinates, dtype=torch.long)
    texture = torch.as_tensor(texture, dtype=torch.float32)
    count = int(count)
    if count < 0 or count > len(destination_coordinates):
        raise ValueError("mapped texture count outside destination range")
    if count > len(source_coordinates):
        raise ValueError("mapped texture count outside source range")
    result = image.clone()
    if count:
        destination = destination_coordinates[:count]
        source = source_coordinates[:count]
        result[:, destination[:, 0], destination[:, 1]] = texture[
            :, source[:, 0], source[:, 1]
        ]
    return result


def select_matched_donor(metadata, recipient_index, target):
    recipient = metadata[recipient_index]
    candidates = []
    for index, candidate in enumerate(metadata):
        if index == recipient_index:
            continue
        if candidate["path"] == recipient["path"]:
            continue
        if candidate["pid"] == recipient["pid"]:
            continue
        if not bool(candidate["valid"][target]):
            continue
        candidates.append(index)
    if not candidates:
        return None
    same_camera = [
        index for index in candidates
        if metadata[index]["camid"] == recipient["camid"]
    ]
    if same_camera:
        candidates = same_camera

    recipient_area = max(float(recipient["area"][target]), 1e-8)

    def score(index):
        candidate = metadata[index]
        distance = (
            abs(float(candidate["area"][target]) - recipient_area)
            / recipient_area
            + abs(float(candidate["y_center"][target])
                  - float(recipient["y_center"][target]))
            + abs(float(candidate["confidence"][target])
                  - float(recipient["confidence"][target]))
        )
        return distance, str(candidate["path"])

    return min(candidates, key=score)


def select_wrong_slot_occluder_donor(metadata, recipient_index, target):
    recipient = metadata[recipient_index]
    for offset in range(1, REGIONS):
        donor_slot = (int(target) + offset) % REGIONS
        candidates = []
        for index, candidate in enumerate(metadata):
            if index == recipient_index:
                continue
            if candidate["path"] == recipient["path"]:
                continue
            if candidate["pid"] == recipient["pid"]:
                continue
            if not bool(candidate["valid"][donor_slot]):
                continue
            candidates.append(index)
        if not candidates:
            continue
        same_camera = [
            index for index in candidates
            if metadata[index]["camid"] == recipient["camid"]
        ]
        if same_camera:
            candidates = same_camera
        recipient_area = max(float(recipient["area"][target]), 1e-8)

        def score(index):
            candidate = metadata[index]
            distance = (
                abs(float(candidate["area"][donor_slot]) - recipient_area)
                / recipient_area
                + abs(float(candidate["y_center"][donor_slot])
                      - float(recipient["y_center"][target]))
                + abs(float(candidate["confidence"][donor_slot])
                      - float(recipient["confidence"][target]))
            )
            return distance, str(candidate["path"])

        return min(candidates, key=score), donor_slot
    return None


def mask_iou(left, right):
    left = torch.as_tensor(left, dtype=torch.bool)
    right = torch.as_tensor(right, dtype=torch.bool)
    union = (left | right).sum()
    if int(union) == 0:
        return 0.0
    return float((left & right).sum().double() / union.double())


def fixed_feature_text_logits(features, text_bank):
    """Change only the text bank while the regional visual features stay fixed."""
    features = torch.as_tensor(features)
    text_bank = torch.as_tensor(text_bank)
    if features.shape[-2] != REGIONS:
        raise ValueError("features must end in [5,D]")
    if text_bank.shape[:2] != (REGIONS, 2):
        raise ValueError("text_bank must be [5,2,D]")
    if features.shape[-1] != text_bank.shape[-1]:
        raise ValueError("feature/text width mismatch")
    correct = torch.einsum("...rd,rsd->...rs", features, text_bank)
    return {
        "correct": correct,
        "slot_cycle": torch.einsum(
            "...rd,rsd->...rs", features, text_bank.roll(1, dims=0)
        ),
        "state_inversion": torch.einsum(
            "...rd,rsd->...rs", features, text_bank.flip(1)
        ),
    }


def _nested_exact(sets):
    return all(
        bool((sets[index] & (~sets[index + 1])).sum() == 0)
        for index in range(len(sets) - 1)
    )


def _legacy_pixel_contract(seed):
    height, width = 64, 32
    content = torch.ones(height, width, dtype=torch.bool)
    support = torch.zeros(height, width, dtype=torch.bool)
    support[18:46, 10:22] = True
    nested_a = _legacy_nested_support_sets(
        support, LEVELS, "target-%d" % seed
    )
    nested_b = _legacy_nested_support_sets(
        support, LEVELS, "target-%d" % seed
    )
    controls_a = _legacy_matched_nonoverlap_sets(
        support,
        content,
        nested_a["ordered_coordinates"],
        LEVELS,
        "control-%d" % seed,
        exclusion_radius=3,
    )
    controls_b = _legacy_matched_nonoverlap_sets(
        support,
        content,
        nested_b["ordered_coordinates"],
        LEVELS,
        "control-%d" % seed,
        exclusion_radius=3,
    )

    image = torch.linspace(0.05, 0.95, steps=height * width).view(
        1, height, width
    ).repeat(3, 1, 1)
    image[1] = image[1].flip(0)
    image[2] = image[2].flip(1)
    donor = 1.0 - image
    donor_mask = torch.zeros_like(support)
    donor_mask[8:32, 2:9] = True
    random_a = _legacy_deterministic_random_texture(
        image, support, "texture-%d" % seed
    )
    random_b = _legacy_deterministic_random_texture(
        image, support, "texture-%d" % seed
    )
    random_other = _legacy_deterministic_random_texture(
        image, support, "texture-other-%d" % seed
    )
    donor_texture = _legacy_donor_slot_texture(donor, donor_mask, support)
    mean_texture = torch.as_tensor(CLIP_MEAN).view(3, 1, 1).expand_as(
        image
    )
    donor_content = (
        donor_texture - mean_texture
    ).abs().amax(0) > 1e-6
    donor_content_box = _bbox(donor_content)
    donor_source_box = _bbox(donor_mask)
    donor_source_ratio = (
        (donor_source_box[2] - donor_source_box[0])
        / float(donor_source_box[3] - donor_source_box[1])
    )
    donor_content_ratio = (
        (donor_content_box[2] - donor_content_box[0])
        / float(donor_content_box[3] - donor_content_box[1])
    )
    selected = nested_a["sets"][-1]
    material_outputs = {
        "clip_mean": _legacy_apply_texture(image, selected, mean_texture),
        "random_texture": _legacy_apply_texture(image, selected, random_a),
        "different_pid_cutmix": _legacy_apply_texture(
            image, selected, donor_texture
        ),
    }
    valid = torch.ones(17, REGIONS, dtype=torch.bool)
    valid[-1] = False
    paths = ["image_%02d.jpg" % index for index in range(len(valid))]
    targets_a, target_counts_a = deterministic_balanced_targets(
        valid, paths, seed
    )
    targets_b, target_counts_b = deterministic_balanced_targets(
        valid, paths, seed
    )
    metadata = [
        {
            "path": "recipient.jpg",
            "pid": 0,
            "camid": 1,
            "valid": [True] * REGIONS,
            "area": [0.20] * REGIONS,
            "y_center": [0.50] * REGIONS,
            "confidence": [0.80] * REGIONS,
        },
        {
            "path": "cross_camera_near.jpg",
            "pid": 1,
            "camid": 2,
            "valid": [True] * REGIONS,
            "area": [0.20] * REGIONS,
            "y_center": [0.50] * REGIONS,
            "confidence": [0.80] * REGIONS,
        },
        {
            "path": "same_camera_far.jpg",
            "pid": 2,
            "camid": 1,
            "valid": [True] * REGIONS,
            "area": [0.30] * REGIONS,
            "y_center": [0.60] * REGIONS,
            "confidence": [0.70] * REGIONS,
        },
        {
            "path": "same_pid.jpg",
            "pid": 0,
            "camid": 1,
            "valid": [True] * REGIONS,
            "area": [0.20] * REGIONS,
            "y_center": [0.50] * REGIONS,
            "confidence": [0.80] * REGIONS,
        },
    ]
    donor_index = select_matched_donor(metadata, 0, 2)

    features = torch.arange(30, dtype=torch.float32).view(2, REGIONS, 3)
    text_bank = torch.arange(30, 60, dtype=torch.float32).view(
        REGIONS, 2, 3
    )
    wrong_text = fixed_feature_text_logits(features, text_bank)
    flipped_support = torch.flip(support, dims=(-1,))
    restored_support = torch.flip(flipped_support, dims=(-1,))

    target_counts = [int(current.sum()) for current in nested_a["sets"]]
    control_counts = [int(current.sum()) for current in controls_a["sets"]]
    mapped_random = _legacy_apply_mapped_texture(
        image,
        controls_a["ordered_coordinates"],
        nested_a["ordered_coordinates"],
        random_a,
        target_counts[-1],
    )
    mapped_destination = controls_a["ordered_coordinates"][:target_counts[-1]]
    mapped_source = nested_a["ordered_coordinates"][:target_counts[-1]]
    gates = {
        "balanced_target_repeat_exact": bool(
            torch.equal(targets_a, targets_b)
            and torch.equal(target_counts_a, target_counts_b)
        ),
        "balanced_target_ignores_invalid": int(targets_a[-1]) == -1,
        "balanced_target_spread_at_most_one": int(
            target_counts_a.max() - target_counts_a.min()
        ) <= 1,
        "target_nested_exact": _nested_exact(nested_a["sets"]),
        "target_repeat_exact": bool(torch.equal(
            nested_a["sets"], nested_b["sets"]
        )),
        "target_level_error_within_one_pixel": bool(
            nested_a["errors"].max()
            <= 1.0 / nested_a["support_pixels"] + 1e-12
        ),
        "control_nested_exact": _nested_exact(controls_a["sets"]),
        "control_repeat_exact": bool(torch.equal(
            controls_a["sets"], controls_b["sets"]
        )),
        "control_count_exact": target_counts == control_counts,
        "control_zero_overlap_exact": all(
            int((current & support).sum()) == 0
            for current in controls_a["sets"]
        ),
        "random_texture_repeat_exact": bool(torch.equal(random_a, random_b)),
        "random_texture_seed_sensitive": not bool(torch.equal(
            random_a, random_other
        )),
        "random_texture_finite": bool(torch.isfinite(random_a).all()),
        "all_materials_change_selected": all(
            not bool(torch.equal(output[:, selected], image[:, selected]))
            for output in material_outputs.values()
        ),
        "all_materials_preserve_unselected_exact": all(
            bool(torch.equal(output[:, ~selected], image[:, ~selected]))
            for output in material_outputs.values()
        ),
        "nonoverlap_material_values_exact": bool(torch.equal(
            mapped_random[
                :,
                mapped_destination[:, 0],
                mapped_destination[:, 1],
            ],
            random_a[:, mapped_source[:, 0], mapped_source[:, 1]],
        )),
        "donor_aspect_ratio_preserved_with_rounding": abs(
            donor_content_ratio - donor_source_ratio
        ) <= 1.0 / min(
            donor_content_box[2] - donor_content_box[0],
            donor_content_box[3] - donor_content_box[1],
        ),
        "donor_different_pid_path": bool(
            donor_index is not None
            and metadata[donor_index]["pid"] != metadata[0]["pid"]
            and metadata[donor_index]["path"] != metadata[0]["path"]
        ),
        "donor_same_camera_preferred": donor_index == 2,
        "wrong_mask_low_iou": mask_iou(support, donor_mask) == 0.0,
        "flip_inverse_exact": bool(torch.equal(support, restored_support)),
        "slot_cycle_exact": bool(torch.equal(
            wrong_text["slot_cycle"],
            torch.einsum(
                "...rd,rsd->...rs",
                features,
                text_bank.roll(1, dims=0),
            ),
        )),
        "slot_cycle_keeps_visual_slot_fixed": not bool(torch.equal(
            wrong_text["slot_cycle"],
            wrong_text["correct"].roll(1, dims=-2),
        )),
        "state_inversion_exact": bool(torch.equal(
            wrong_text["state_inversion"], wrong_text["correct"].flip(-1)
        )),
    }
    return {
        "status": "EXP392_PHASE0B2_FULL_TEACHER_STATIC_COMPLETE",
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "formal_training_authorized": False,
        "gpu_authorized": False,
        "gates": gates,
        "measurements": {
            "target_counts": target_counts,
            "control_counts": control_counts,
            "target_realized": nested_a["realized"].tolist(),
            "target_max_error": float(nested_a["errors"].max()),
            "control_mean_abs_y_error": controls_a["mean_abs_y_error"],
            "donor_source_ratio": donor_source_ratio,
            "donor_content_ratio": donor_content_ratio,
            "balanced_slot_counts": target_counts_a.tolist(),
            "selected_donor_path": (
                metadata[donor_index]["path"]
                if donor_index is not None else None
            ),
        },
    }


def static_contract(seed):
    height, width = 128, 256
    support = torch.zeros(height, width, dtype=torch.bool)
    support[40:88, 108:148] = True
    support[52:76, 98:108] = True
    support[52:76, 148:158] = True
    connected_a = connected_occlusion_rectangles(
        support, LEVELS, "connected-%d" % seed
    )
    connected_b = connected_occlusion_rectangles(
        support, LEVELS, "connected-%d" % seed
    )
    controls_a = translated_nonoverlap_rectangles(
        support,
        connected_a["boxes"],
        connected_a["direction"],
        exclusion_radius=24,
    )
    controls_b = translated_nonoverlap_rectangles(
        support,
        connected_b["boxes"],
        connected_b["direction"],
        exclusion_radius=24,
    )

    image = torch.linspace(0.05, 0.95, steps=height * width).view(
        1, height, width
    ).repeat(3, 1, 1)
    image[1] = image[1].flip(0)
    image[2] = image[2].flip(1)
    donor = 1.0 - image
    donor_mask = torch.zeros_like(support)
    donor_mask[20:68, 16:40] = True
    maximum_box = connected_a["boxes"][-1]
    maximum_size = (
        maximum_box[2] - maximum_box[0],
        maximum_box[3] - maximum_box[1],
    )
    support_bbox_mask = _box_mask(
        support.shape, connected_a["support_box"]
    )
    random_a = random_occluder_texture(
        image, support_bbox_mask, maximum_size, "texture-%d" % seed
    )
    random_b = random_occluder_texture(
        image, support_bbox_mask, maximum_size, "texture-%d" % seed
    )
    random_other = random_occluder_texture(
        image,
        support_bbox_mask,
        maximum_size,
        "texture-other-%d" % seed,
    )
    mean_texture = torch.as_tensor(CLIP_MEAN, dtype=torch.float32).view(
        3, 1, 1
    ).expand(3, maximum_size[0], maximum_size[1])
    cutmix_texture = donor_slot_occluder_texture(
        donor, donor_mask, maximum_size
    )
    textures = {
        "clip_mean": mean_texture,
        "random_texture": random_a,
        "different_pid_wrong_slot_cutmix": cutmix_texture,
    }
    material_contract = {}
    for material, texture in textures.items():
        overlap_outputs = []
        control_outputs = []
        values_exact = []
        outside_exact = []
        lower_values_preserved = []
        for level_index, (overlap_box, control_box) in enumerate(zip(
            connected_a["boxes"], controls_a["boxes"]
        )):
            overlap = apply_rectangle_texture(
                image, overlap_box, texture, connected_a["direction"]
            )
            control = apply_rectangle_texture(
                image, control_box, texture, connected_a["direction"]
            )
            overlap_outputs.append(overlap)
            control_outputs.append(control)
            oy0, ox0, oy1, ox1 = overlap_box
            cy0, cx0, cy1, cx1 = control_box
            values_exact.append(bool(torch.equal(
                overlap[:, oy0:oy1, ox0:ox1],
                control[:, cy0:cy1, cx0:cx1],
            )))
            outside = ~(connected_a["sets"][level_index])
            outside_exact.append(bool(torch.equal(
                overlap[:, outside], image[:, outside]
            )))
            if level_index:
                previous = connected_a["sets"][level_index - 1]
                lower_values_preserved.append(bool(torch.equal(
                    overlap[:, previous], overlap_outputs[level_index - 1][:, previous]
                )))
        material_contract[material] = {
            "target_control_values_exact_all_levels": all(values_exact),
            "outside_exact_all_levels": all(outside_exact),
            "lower_level_values_preserved": all(lower_values_preserved),
            "finite": all(torch.isfinite(value).all().item()
                          for value in overlap_outputs + control_outputs),
        }

    source_values = image[:, support_bbox_mask]
    random_mean_error = float(
        (random_a.mean((1, 2)) - source_values.mean(1)).abs().max()
    )
    random_std_error = float(
        (
            random_a.std((1, 2), unbiased=False)
            - source_values.std(1, unbiased=False)
        ).abs().max()
    )

    valid = torch.ones(17, REGIONS, dtype=torch.bool)
    valid[-1] = False
    paths = ["image_%02d.jpg" % index for index in range(len(valid))]
    targets_a, target_counts_a = deterministic_balanced_targets(
        valid, paths, seed
    )
    targets_b, target_counts_b = deterministic_balanced_targets(
        valid, paths, seed
    )
    metadata = [
        {
            "path": "recipient.jpg",
            "pid": 0,
            "camid": 1,
            "valid": [True] * REGIONS,
            "area": [0.20, 0.30, 0.18, 0.22, 0.16],
            "y_center": [0.15, 0.40, 0.42, 0.68, 0.86],
            "confidence": [0.80] * REGIONS,
        },
        {
            "path": "cross_camera_near.jpg",
            "pid": 1,
            "camid": 2,
            "valid": [True] * REGIONS,
            "area": [0.20, 0.30, 0.18, 0.22, 0.16],
            "y_center": [0.15, 0.40, 0.42, 0.68, 0.86],
            "confidence": [0.80] * REGIONS,
        },
        {
            "path": "same_camera_far.jpg",
            "pid": 2,
            "camid": 1,
            "valid": [True] * REGIONS,
            "area": [0.30, 0.40, 0.28, 0.32, 0.26],
            "y_center": [0.20, 0.45, 0.47, 0.73, 0.91],
            "confidence": [0.70] * REGIONS,
        },
        {
            "path": "same_pid.jpg",
            "pid": 0,
            "camid": 1,
            "valid": [True] * REGIONS,
            "area": [0.20, 0.30, 0.18, 0.22, 0.16],
            "y_center": [0.15, 0.40, 0.42, 0.68, 0.86],
            "confidence": [0.80] * REGIONS,
        },
    ]
    same_slot_donor = select_matched_donor(metadata, 0, 2)
    wrong_slot_donor = select_wrong_slot_occluder_donor(metadata, 0, 2)

    features = torch.arange(30, dtype=torch.float32).view(2, REGIONS, 3)
    text_bank = torch.arange(30, 60, dtype=torch.float32).view(
        REGIONS, 2, 3
    )
    wrong_text = fixed_feature_text_logits(features, text_bank)
    restored_support = torch.flip(torch.flip(support, (-1,)), (-1,))
    gates = {
        "balanced_target_repeat_exact": bool(
            torch.equal(targets_a, targets_b)
            and torch.equal(target_counts_a, target_counts_b)
        ),
        "balanced_target_ignores_invalid": int(targets_a[-1]) == -1,
        "balanced_target_spread_at_most_one": int(
            target_counts_a.max() - target_counts_a.min()
        ) <= 1,
        "connected_direction_repeat_exact": (
            connected_a["direction"] == connected_b["direction"]
        ),
        "connected_boxes_repeat_exact": connected_a["boxes"] == connected_b["boxes"],
        "connected_masks_repeat_exact": bool(torch.equal(
            connected_a["sets"], connected_b["sets"]
        )),
        "connected_masks_strictly_nested": _nested_exact(
            connected_a["sets"]
        ),
        "connected_realized_strictly_increasing": bool(
            (connected_a["realized"][1:] > connected_a["realized"][:-1]).all()
        ),
        "connected_realized_reaches_levels": bool(
            (connected_a["realized"] >= torch.as_tensor(LEVELS)).all()
        ),
        "connected_overshoot_within_last_strip": bool(
            (
                connected_a["realized"] - torch.as_tensor(LEVELS)
                <= connected_a["last_increment_fraction"] + 1e-12
            ).all()
        ),
        "control_boxes_repeat_exact": controls_a["boxes"] == controls_b["boxes"],
        "control_masks_repeat_exact": bool(torch.equal(
            controls_a["sets"], controls_b["sets"]
        )),
        "control_masks_strictly_nested": _nested_exact(controls_a["sets"]),
        "control_24px_dilation_zero_exact": not bool((
            controls_a["sets"] & _dilate(support, 24)[None]
        ).any()),
        "control_y_error_within_one_bin": (
            controls_a["normalized_y_error"] <= 1.0 / Y_BINS
        ),
        "all_material_contracts": all(
            all(values.values()) for values in material_contract.values()
        ),
        "random_texture_repeat_exact": bool(torch.equal(random_a, random_b)),
        "random_texture_seed_sensitive": not bool(torch.equal(
            random_a, random_other
        )),
        "random_texture_mean_match": random_mean_error <= 0.03,
        "random_texture_std_match": random_std_error <= 0.03,
        "same_slot_donor_no_fixed_point": bool(
            same_slot_donor is not None
            and metadata[same_slot_donor]["pid"] != metadata[0]["pid"]
            and metadata[same_slot_donor]["path"] != metadata[0]["path"]
        ),
        "same_slot_donor_prefers_same_camera": same_slot_donor == 2,
        "wrong_slot_donor_no_fixed_point": bool(
            wrong_slot_donor is not None
            and metadata[wrong_slot_donor[0]]["pid"] != metadata[0]["pid"]
            and metadata[wrong_slot_donor[0]]["path"] != metadata[0]["path"]
            and wrong_slot_donor[1] != 2
        ),
        "wrong_slot_donor_prefers_same_camera": (
            wrong_slot_donor is not None and wrong_slot_donor[0] == 2
        ),
        "flip_inverse_exact": bool(torch.equal(support, restored_support)),
        "slot_cycle_exact": bool(torch.equal(
            wrong_text["slot_cycle"],
            torch.einsum(
                "...rd,rsd->...rs", features, text_bank.roll(1, dims=0)
            ),
        )),
        "slot_cycle_keeps_visual_slot_fixed": not bool(torch.equal(
            wrong_text["slot_cycle"], wrong_text["correct"].roll(1, dims=-2)
        )),
        "state_inversion_exact": bool(torch.equal(
            wrong_text["state_inversion"], wrong_text["correct"].flip(-1)
        )),
    }
    return {
        "status": "EXP392_PHASE0B2_FULL_TEACHER_STATIC_COMPLETE",
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "formal_training_authorized": False,
        "pose_feasibility_authorized": all(gates.values()),
        "gpu_authorized": False,
        "gates": gates,
        "measurements": {
            "sweep_direction": connected_a["direction"],
            "overlap_boxes": connected_a["boxes"],
            "control_boxes": controls_a["boxes"],
            "realized_overlap": connected_a["realized"].tolist(),
            "last_increment_fraction": connected_a[
                "last_increment_fraction"
            ].tolist(),
            "control_normalized_y_error": controls_a[
                "normalized_y_error"
            ],
            "random_texture_mean_max_error": random_mean_error,
            "random_texture_std_max_error": random_std_error,
            "balanced_slot_counts": target_counts_a.tolist(),
            "same_slot_donor_path": (
                metadata[same_slot_donor]["path"]
                if same_slot_donor is not None else None
            ),
            "wrong_slot_donor": (
                {
                    "path": metadata[wrong_slot_donor[0]]["path"],
                    "slot": wrong_slot_donor[1],
                }
                if wrong_slot_donor is not None else None
            ),
            "material_contract": material_contract,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def main():
    args = parse_args()
    script_path = Path(__file__).resolve()
    result = static_contract(args.seed)
    result["execution"] = {
        "audit_script_sha256": sha256_file(script_path),
        "device": "cpu",
        "reads_reid_data": False,
        "loads_clip": False,
        "seed": int(args.seed),
    }
    write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "verdict": result["verdict"],
        "gates": result["gates"],
        "measurements": result["measurements"],
        "output_sha256": sha256_file(args.output),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if result["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
