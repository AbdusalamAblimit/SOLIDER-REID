"""Pure helpers for the PACIT revision-3 asset oracle.

The CLIP selector and the blind evaluator are intentionally separated:
``select_clip_candidate`` accepts only an already-computed score matrix, while
``blind_evaluate`` accepts pixels, masks, pose fields, and D0 boolean gates but
has no CLIP input.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F


IMAGE_HEIGHT = 384
IMAGE_WIDTH = 128
FIELD_HEIGHT = 96
FIELD_WIDTH = 32
ORACLE_COUNT = 512
ANCHOR_COUNT = 5
ASPECT_RATIOS = (0.40, 0.60, 0.80, 1.00, 1.25, 1.67, 2.50)
PROPOSALS_PER_POOL = ANCHOR_COUNT * len(ASPECT_RATIOS)
ACTIVE_PROPOSALS_PER_IMAGE = len(ASPECT_RATIOS)
ROA_COUNT = 8
AREA_FRACTION = 0.06
AREA_RELATIVE_TOLERANCE = 0.01
ALPHA = 1.0
SELECTION_SALT = "exp415-pacit-oracle-v3"
CALIPER_BLIND_SALT = "exp415-v3-caliper-blind"

FIXED_ANCHORS = (
    (0.10, 0.50),
    (0.30, 0.50),
    (0.48, 0.50),
    (0.67, 0.50),
    (0.86, 0.50),
)

COLOR_NAMES = (
    "black",
    "white",
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "brown",
)

# Fixed sRGB prototypes. Neutral gray is deliberately absent.
COLOR_PROTOTYPE_RGB = (
    (0.06, 0.06, 0.06),
    (0.94, 0.94, 0.94),
    (0.78, 0.08, 0.08),
    (0.88, 0.34, 0.04),
    (0.88, 0.78, 0.05),
    (0.08, 0.58, 0.14),
    (0.04, 0.68, 0.68),
    (0.08, 0.18, 0.78),
    (0.52, 0.12, 0.66),
    (0.42, 0.20, 0.07),
)
COLOR_LAB_RADIUS = (25.0, 22.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 38.0)

ANATOMY_TARGET_MIN = 0.25
ANATOMY_NON_TARGET_MEAN_MAX = 0.10
ANATOMY_NON_TARGET_SINGLE_MAX = 0.25
COLOR_PRESENCE_MIN = 0.10
COLOR_CAPTURE_MIN = 0.25
COLOR_PURITY_MIN = 0.20
COLOR_ABSOLUTE_DROP_MIN = 0.15
COLOR_RELATIVE_DROP_MIN = 0.80
COLOR_COMPONENT_PIXELS_MIN = 32
COLOR_COMPONENT_RATIO_MIN = 0.60


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_text(value):
    return sha256_bytes(str(value).encode("utf-8"))


def ordered_digest(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def select_oracle_rows(rows, count=ORACLE_COUNT):
    """Select a record-order-independent fixed oracle sample."""
    if len(rows) < int(count):
        raise ValueError("not enough rows for the frozen oracle")
    keyed = []
    seen = set()
    for row in rows:
        relative_path = str(row["relative_path"])
        if relative_path in seen:
            raise RuntimeError("duplicate relative path")
        seen.add(relative_path)
        key = sha256_text(SELECTION_SALT + "\0" + relative_path)
        keyed.append((key, relative_path, row))
    keyed.sort(key=lambda item: (item[0], item[1]))
    output = []
    for oracle_index, (_, _, row) in enumerate(keyed[: int(count)]):
        copied = dict(row)
        copied["oracle_index"] = int(oracle_index)
        output.append(copied)
    return output


def percentile(values, q):
    """Frozen NumPy-linear percentile."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("invalid percentile input")
    return float(np.quantile(array, float(q), method="linear"))


def median_mad(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("invalid MAD input")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median, mad


def _best_dimensions(target_area, aspect):
    best = None
    for height in range(1, IMAGE_HEIGHT + 1):
        approximate = int(round(float(target_area) / float(height)))
        for width in (approximate - 1, approximate, approximate + 1):
            if width < 1 or width > IMAGE_WIDTH:
                continue
            area = height * width
            ratio = float(width) / float(height)
            cost = (
                abs(area - int(target_area)) / float(max(int(target_area), 1))
                + 0.02 * abs(math.log(max(ratio, 1e-12) / float(aspect)))
            )
            item = (cost, abs(area - int(target_area)), height, width)
            if best is None or item < best:
                best = item
    if best is None:
        raise RuntimeError("could not fit proposal rectangle")
    if best[1] / float(target_area) > AREA_RELATIVE_TOLERANCE:
        raise RuntimeError("proposal area tolerance is mathematically unreachable")
    return int(best[2]), int(best[3])


def render_full_pose_fields(keypoints, valid):
    if keypoints.shape != (17, 2) or valid.shape != (17,):
        raise ValueError("unexpected pose shape")
    # Keep the rest of this module importable for CPU contracts without
    # importing the full SOLIDER model (and its OpenCV/runtime dependencies).
    from model.pose_clip_relation import render_pose_indexed_regions

    masks, region_valid = render_pose_indexed_regions(
        keypoints.float().unsqueeze(0),
        valid.bool().unsqueeze(0),
        image_hw=(IMAGE_HEIGHT, IMAGE_WIDTH),
        field_hw=(FIELD_HEIGHT, FIELD_WIDTH),
        sigma=1.5,
    )
    fields = F.interpolate(
        masks.float(),
        size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        mode="bilinear",
        align_corners=False,
    )[0]
    return fields, region_valid[0].bool()


def _weighted_centroid(field):
    mass = float(field.sum())
    if mass <= 0.0:
        return None
    yy = torch.arange(IMAGE_HEIGHT, dtype=torch.float32, device=field.device).view(
        IMAGE_HEIGHT, 1
    )
    xx = torch.arange(IMAGE_WIDTH, dtype=torch.float32, device=field.device).view(
        1, IMAGE_WIDTH
    )
    center_y = float((field * yy).sum() / field.sum())
    center_x = float((field * xx).sum() / field.sum())
    return center_y, center_x


def _rectangle_mask(center_y, center_x, height, width):
    top = int(round(float(center_y) - float(height) / 2.0))
    left = int(round(float(center_x) - float(width) / 2.0))
    top = min(max(top, 0), IMAGE_HEIGHT - int(height))
    left = min(max(left, 0), IMAGE_WIDTH - int(width))
    mask = torch.zeros(IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.bool)
    mask[top : top + int(height), left : left + int(width)] = True
    return mask, top, left


def _generate_indexed_proposals(anchors, anchor_valid):
    if len(anchors) != ANCHOR_COUNT or len(anchor_valid) != ANCHOR_COUNT:
        raise ValueError("unexpected proposal anchor count")
    target_area = int(round(AREA_FRACTION * IMAGE_HEIGHT * IMAGE_WIDTH))
    proposals = []
    for anchor_index in range(ANCHOR_COUNT):
        center_y, center_x = anchors[anchor_index]
        for aspect_index, aspect in enumerate(ASPECT_RATIOS):
            height, width = _best_dimensions(target_area, aspect)
            mask, top, left = _rectangle_mask(
                center_y, center_x, height, width
            )
            proposal_index = anchor_index * len(ASPECT_RATIOS) + aspect_index
            proposals.append(
                {
                    "candidate_index": int(proposal_index),
                    "anchor_index": int(anchor_index),
                    "aspect_index": int(aspect_index),
                    "aspect": float(aspect),
                    "anchor_valid": bool(anchor_valid[anchor_index]),
                    "top": int(top),
                    "left": int(left),
                    "height": int(height),
                    "width": int(width),
                    "area_pixels": int(mask.sum()),
                    "area_fraction": float(mask.sum())
                    / float(IMAGE_HEIGHT * IMAGE_WIDTH),
                    "centroid_y": float(top + (height - 1) / 2.0)
                    / float(IMAGE_HEIGHT - 1),
                    "centroid_x": float(left + (width - 1) / 2.0)
                    / float(IMAGE_WIDTH - 1),
                    "mask_sha256": sha256_bytes(
                        mask.numpy().astype(np.uint8, copy=False).tobytes()
                    ),
                    "mask": mask,
                }
            )
    if [row["candidate_index"] for row in proposals] != list(
        range(PROPOSALS_PER_POOL)
    ):
        raise RuntimeError("proposal index contract failed")
    return proposals


def generate_fixed_proposals():
    """Generate the P=0 pool without accepting or reading pose inputs."""
    anchors = [
        (
            normalized_y * (IMAGE_HEIGHT - 1),
            normalized_x * (IMAGE_WIDTH - 1),
        )
        for normalized_y, normalized_x in FIXED_ANCHORS
    ]
    return _generate_indexed_proposals(anchors, [True] * ANCHOR_COUNT)


def generate_pose_proposals(keypoints, valid):
    """Generate the P=1 pool; no semantic score filters or reorders it."""
    fields, region_valid = render_full_pose_fields(keypoints, valid)
    anchors = []
    anchor_valid = []
    for anchor_index in range(ANCHOR_COUNT):
        fixed_y = FIXED_ANCHORS[anchor_index][0] * (IMAGE_HEIGHT - 1)
        fixed_x = FIXED_ANCHORS[anchor_index][1] * (IMAGE_WIDTH - 1)
        centroid = (
            _weighted_centroid(fields[anchor_index])
            if bool(region_valid[anchor_index])
            else None
        )
        anchor_valid.append(bool(centroid is not None))
        anchors.append(centroid if centroid is not None else (fixed_y, fixed_x))
    proposals = _generate_indexed_proposals(anchors, anchor_valid)
    return proposals, fields.cpu(), region_valid.cpu()


def active_anchor_index(oracle_index):
    return int(oracle_index) % ANCHOR_COUNT


def active_proposals(proposals, oracle_index):
    if len(proposals) != PROPOSALS_PER_POOL:
        raise ValueError("unexpected complete proposal pool")
    anchor_index = active_anchor_index(oracle_index)
    selected = [
        row for row in proposals if int(row["anchor_index"]) == anchor_index
    ]
    selected.sort(key=lambda row: int(row["aspect_index"]))
    if len(selected) != ACTIVE_PROPOSALS_PER_IMAGE:
        raise RuntimeError("active proposal count mismatch")
    if [row["aspect_index"] for row in selected] != list(
        range(ACTIVE_PROPOSALS_PER_IMAGE)
    ):
        raise RuntimeError("active aspect index mismatch")
    return selected


def deterministic_fill(relative_path):
    seed = int(
        sha256_text(SELECTION_SALT + "\0fill\0" + str(relative_path))[:16], 16
    ) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gray = (
        0.50
        + 0.08
        * torch.randn(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            generator=generator,
            dtype=torch.float32,
        )
    ).clamp(0.38, 0.62)
    return gray.unsqueeze(0).expand(3, -1, -1).clone()


def apply_candidate(rgb, mask, fill):
    if rgb.shape != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError("unexpected RGB shape")
    if mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or mask.dtype != torch.bool:
        raise ValueError("unexpected mask")
    if fill.shape != rgb.shape:
        raise ValueError("fill shape mismatch")
    output = rgb.clone()
    output[:, mask] = fill[:, mask]
    if not torch.equal(output[:, ~mask], rgb[:, ~mask]):
        raise RuntimeError("mask-exterior RGB changed")
    return output


def compute_centered_color_drop(
    original_image_feature,
    edited_image_features,
    color_text_features,
):
    """Frozen CLIP scorer with no pose, slot, blind-color, or D0 inputs."""
    original = torch.as_tensor(original_image_feature, dtype=torch.float64)
    edited = torch.as_tensor(edited_image_features, dtype=torch.float64)
    text = torch.as_tensor(color_text_features, dtype=torch.float64)
    if original.ndim != 1:
        raise ValueError("original feature must be one-dimensional")
    if edited.shape != (ACTIVE_PROPOSALS_PER_IMAGE, original.shape[0]):
        raise ValueError("unexpected edited feature shape")
    if text.shape != (len(COLOR_NAMES), original.shape[0]):
        raise ValueError("unexpected color text feature shape")
    if not (
        torch.isfinite(original).all()
        and torch.isfinite(edited).all()
        and torch.isfinite(text).all()
    ):
        raise ValueError("nonfinite CLIP feature")
    original = F.normalize(original, dim=0)
    edited = F.normalize(edited, dim=1)
    text = F.normalize(text, dim=1)
    original_similarity = text @ original
    edited_similarity = edited @ text.transpose(0, 1)
    color_count = len(COLOR_NAMES)
    original_margin = original_similarity - (
        original_similarity.sum() - original_similarity
    ) / float(color_count - 1)
    edited_margin = edited_similarity - (
        edited_similarity.sum(dim=1, keepdim=True) - edited_similarity
    ) / float(color_count - 1)
    return original_margin.unsqueeze(0) - edited_margin


def select_clip_candidate(clip_drop):
    """Select from a [7, 10] score matrix; no pose input is accepted."""
    scores = torch.as_tensor(clip_drop, dtype=torch.float64)
    if scores.shape != (ACTIVE_PROPOSALS_PER_IMAGE, len(COLOR_NAMES)):
        raise ValueError("unexpected CLIP drop matrix")
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite CLIP selector score")
    flat_index = int(torch.argmax(scores.reshape(-1)))
    aspect_index = flat_index // len(COLOR_NAMES)
    color_index = flat_index % len(COLOR_NAMES)
    return {
        "aspect_index": int(aspect_index),
        "selector_color_index": int(color_index),
        "selector_score": float(scores[aspect_index, color_index]),
    }


def caliper_eligible(
    reference,
    candidates,
    reference_d0_shift,
    candidate_d0_shifts,
    reference_d0_ce_change,
    candidate_d0_ce_changes,
    clean_top5,
    candidate_top5,
    *,
    require_centroid,
    allow_reference=False,
):
    """Return a per-candidate nuisance-match bitmap.

    This function never reads RGB, CLIP scores/labels, or blind-color values.
    """
    if len(candidates) != ACTIVE_PROPOSALS_PER_IMAGE:
        raise ValueError("unexpected caliper candidate count")
    shifts = np.asarray(candidate_d0_shifts, dtype=np.float64)
    ce_changes = np.asarray(candidate_d0_ce_changes, dtype=np.float64)
    top5 = np.asarray(candidate_top5, dtype=np.bool_)
    if shifts.shape != (ACTIVE_PROPOSALS_PER_IMAGE,):
        raise ValueError("unexpected D0 shift shape")
    if ce_changes.shape != shifts.shape or top5.shape != shifts.shape:
        raise ValueError("unexpected D0 caliper shape")
    if not np.isfinite(shifts).all() or not np.isfinite(ce_changes).all():
        raise ValueError("nonfinite D0 caliper input")
    output = []
    for candidate_index, candidate in enumerate(candidates):
        area_ok = abs(
            int(candidate["area_pixels"]) - int(reference["area_pixels"])
        ) <= 1
        aspect_ok = abs(
            math.log(
                float(candidate["aspect"]) / float(reference["aspect"])
            )
        ) <= math.log(1.25)
        centroid_ok = True
        if bool(require_centroid):
            centroid_ok = max(
                abs(
                    float(candidate["centroid_y"])
                    - float(reference["centroid_y"])
                ),
                abs(
                    float(candidate["centroid_x"])
                    - float(reference["centroid_x"])
                ),
            ) <= 0.01
        d0_ok = (
            abs(float(shifts[candidate_index]) - float(reference_d0_shift))
            <= 0.010
        )
        ce_ok = (
            abs(
                float(ce_changes[candidate_index])
                - float(reference_d0_ce_change)
            )
            <= 0.25
        )
        different = candidate["mask_sha256"] != reference["mask_sha256"]
        mask_ok = bool(different or allow_reference)
        output.append(
            bool(
                clean_top5
                and top5[candidate_index]
                and area_ok
                and aspect_ok
                and centroid_ok
                and d0_ok
                and ce_ok
                and mask_ok
            )
        )
    return np.asarray(output, dtype=np.bool_)


def select_caliper_hash_candidate(relative_path, candidates, eligible):
    eligible = np.asarray(eligible, dtype=np.bool_)
    if eligible.shape != (len(candidates),):
        raise ValueError("caliper selection shape mismatch")
    indices = np.flatnonzero(eligible)
    if indices.size == 0:
        return None
    ranked = sorted(
        indices.tolist(),
        key=lambda index: (
            caliper_blind_key(
                relative_path, candidates[index]["candidate_index"]
            ),
            int(candidates[index]["candidate_index"]),
        ),
    )
    return int(ranked[0])


def caliper_blind_key(relative_path, candidate_index):
    return sha256_text(
        CALIPER_BLIND_SALT
        + "\0"
        + str(relative_path)
        + "\0"
        + str(int(candidate_index))
    )


def _srgb_to_linear(rgb):
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )


def srgb_to_cielab(rgb):
    """Convert channel-first sRGB in [0,1] to channel-first CIELAB (D65)."""
    tensor = torch.as_tensor(rgb, dtype=torch.float64)
    if tensor.shape[0] != 3:
        raise ValueError("expected channel-first RGB")
    if not torch.isfinite(tensor).all():
        raise ValueError("nonfinite RGB")
    tensor = tensor.clamp(0.0, 1.0)
    linear = _srgb_to_linear(tensor)
    red, green, blue = linear[0], linear[1], linear[2]
    x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
    y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z = (0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(value):
        return torch.where(
            value > epsilon,
            value.pow(1.0 / 3.0),
            (kappa * value + 16.0) / 116.0,
        )

    fx, fy, fz = f(x), f(y), f(z)
    return torch.stack(
        (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)),
        dim=0,
    )


def classify_color_bins(rgb):
    """Return fixed CIELAB bins 0..9 and -1 for neutral/unclassified pixels."""
    lab = srgb_to_cielab(rgb)
    prototypes = torch.tensor(
        COLOR_PROTOTYPE_RGB, dtype=torch.float64
    ).transpose(0, 1)
    prototype_lab = srgb_to_cielab(prototypes).transpose(0, 1)
    flat = lab.reshape(3, -1).transpose(0, 1)
    distances = torch.cdist(flat, prototype_lab)
    nearest_distance, nearest_index = distances.min(dim=1)
    radii = torch.tensor(COLOR_LAB_RADIUS, dtype=torch.float64)[nearest_index]
    chroma = torch.linalg.vector_norm(flat[:, 1:3], dim=1)
    chroma_valid = (nearest_index < 2) | (chroma >= 15.0)
    labels = torch.where(
        (nearest_distance <= radii) & chroma_valid,
        nearest_index,
        torch.full_like(nearest_index, -1),
    )
    return labels.reshape(lab.shape[1:]).to(torch.int64)


def _pose_coverage(fields, region_valid, mask):
    field_mass = fields.flatten(1).sum(dim=1)
    valid = region_valid.bool() & (field_mass > 0.0)
    covered = torch.zeros(ANCHOR_COUNT, dtype=torch.float64)
    if bool(valid.any()):
        numerator = (
            fields.double() * mask.double().unsqueeze(0)
        ).flatten(1).sum(dim=1)
        covered[valid] = numerator[valid] / field_mass.double()[valid]
    return covered, valid


def _largest_component_size(binary):
    array = torch.as_tensor(binary, dtype=torch.bool).cpu().numpy()
    if array.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    visited = np.zeros_like(array, dtype=np.bool_)
    best = 0
    height, width = array.shape
    for top in range(height):
        for left in range(width):
            if not array[top, left] or visited[top, left]:
                continue
            stack = [(top, left)]
            visited[top, left] = True
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for next_y, next_x in (
                    (y - 1, x),
                    (y + 1, x),
                    (y, x - 1),
                    (y, x + 1),
                ):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and array[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
            best = max(best, size)
    return int(best)


def blind_evaluate(
    original_rgb,
    edited_rgb,
    mask,
    pose_fields,
    region_valid,
    *,
    expected_anchor_index,
    identity_safe,
):
    """Evaluate an already-selected intervention without CLIP inputs."""
    if mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError("unexpected mask shape")
    fields = torch.as_tensor(pose_fields, dtype=torch.float64)
    valid = torch.as_tensor(region_valid, dtype=torch.bool)
    if fields.shape != (ANCHOR_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError("unexpected pose field shape")
    coverage, valid = _pose_coverage(fields, valid, mask)
    if bool(valid.any()):
        masked_coverage = coverage.clone()
        masked_coverage[~valid] = -1.0
        target_slot = int(torch.argmax(masked_coverage))
        non_target = coverage[
            torch.tensor(
                [index != target_slot for index in range(ANCHOR_COUNT)],
                dtype=torch.bool,
            )
            & valid
        ]
        non_target_mean = float(non_target.mean()) if len(non_target) else 0.0
        non_target_max = float(non_target.max()) if len(non_target) else 0.0
        target_coverage = float(coverage[target_slot])
    else:
        target_slot = -1
        target_coverage = 0.0
        non_target_mean = 1.0
        non_target_max = 1.0

    anatomy_valid = bool(
        target_slot >= 0
        and target_slot == int(expected_anchor_index)
        and target_coverage >= ANATOMY_TARGET_MIN
        and non_target_mean <= ANATOMY_NON_TARGET_MEAN_MAX
        and non_target_max <= ANATOMY_NON_TARGET_SINGLE_MAX
    )

    original_labels = classify_color_bins(original_rgb)
    edited_labels = classify_color_bins(edited_rgb)
    best = None
    if target_slot >= 0:
        field = fields[target_slot]
        hard_support = field >= (0.25 * float(field.max()))
        slot_mass = float(field.sum())
        masked_slot_mass = float((field * mask.double()).sum())
        for color_index in range(len(COLOR_NAMES)):
            original_color = original_labels == color_index
            edited_color = edited_labels == color_index
            total_color_mass = float((field * original_color.double()).sum())
            captured_original = float(
                (field * mask.double() * original_color.double()).sum()
            )
            captured_edited = float(
                (field * mask.double() * edited_color.double()).sum()
            )
            presence = total_color_mass / max(slot_mass, 1e-12)
            capture = captured_original / max(total_color_mass, 1e-12)
            purity = captured_original / max(masked_slot_mass, 1e-12)
            edited_fraction = captured_edited / max(masked_slot_mass, 1e-12)
            absolute_drop = purity - edited_fraction
            relative_drop = absolute_drop / max(purity, 1e-12)
            connected_mask = mask & hard_support & original_color
            captured_pixels = int(connected_mask.sum())
            component_pixels = _largest_component_size(connected_mask)
            component_ratio = component_pixels / float(max(captured_pixels, 1))
            normalized_min = min(
                presence / COLOR_PRESENCE_MIN,
                capture / COLOR_CAPTURE_MIN,
                purity / COLOR_PURITY_MIN,
                component_pixels / float(COLOR_COMPONENT_PIXELS_MIN),
                component_ratio / COLOR_COMPONENT_RATIO_MIN,
            )
            item = (
                normalized_min,
                presence,
                capture,
                purity,
                component_ratio,
                component_pixels,
                -color_index,
                color_index,
                absolute_drop,
                relative_drop,
                edited_fraction,
            )
            if best is None or item[:7] > best[:7]:
                best = item
    if best is None:
        best = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0,
            -1,
            0.0,
            0.0,
            0.0,
        )

    (
        blind_score,
        presence,
        capture,
        purity,
        component_ratio,
        component_pixels,
        _,
        blind_color_index,
        absolute_drop,
        relative_drop,
        edited_fraction,
    ) = best
    coherent_color_removal = bool(
        presence >= COLOR_PRESENCE_MIN
        and capture >= COLOR_CAPTURE_MIN
        and purity >= COLOR_PURITY_MIN
        and component_pixels >= COLOR_COMPONENT_PIXELS_MIN
        and component_ratio >= COLOR_COMPONENT_RATIO_MIN
        and absolute_drop >= COLOR_ABSOLUTE_DROP_MIN
        and relative_drop >= COLOR_RELATIVE_DROP_MIN
    )
    outcome = bool(
        anatomy_valid
        and coherent_color_removal
        and identity_safe
    )
    return {
        "target_slot": int(target_slot),
        "expected_anchor_index": int(expected_anchor_index),
        "target_coverage": float(target_coverage),
        "non_target_coverage_mean": float(non_target_mean),
        "non_target_coverage_max": float(non_target_max),
        "blind_color_index": int(blind_color_index),
        "blind_color_name": (
            COLOR_NAMES[blind_color_index] if blind_color_index >= 0 else None
        ),
        "blind_score": float(blind_score),
        "presence": float(presence),
        "capture": float(capture),
        "purity": float(purity),
        "component_pixels": int(component_pixels),
        "component_ratio": float(component_ratio),
        "edited_fraction": float(edited_fraction),
        "absolute_drop": float(absolute_drop),
        "relative_drop": float(relative_drop),
        "anatomy_valid": bool(anatomy_valid),
        "coherent_color_removal": bool(coherent_color_removal),
        "identity_safe": bool(identity_safe),
        "Y": int(outcome),
    }


def select_raw_color_candidate(evaluations, eligible):
    """Strong non-CLIP selector used only as a matched control."""
    if len(evaluations) != ACTIVE_PROPOSALS_PER_IMAGE:
        raise ValueError("unexpected raw-color candidate count")
    eligible = np.asarray(eligible, dtype=np.bool_)
    if eligible.shape != (ACTIVE_PROPOSALS_PER_IMAGE,):
        raise ValueError("unexpected raw-color eligibility")
    indices = np.flatnonzero(eligible)
    if indices.size == 0:
        return None
    ranked = sorted(
        indices.tolist(),
        key=lambda index: (
            -float(evaluations[index]["blind_score"]),
            -float(evaluations[index]["absolute_drop"]),
            index,
        ),
    )
    return int(ranked[0])


def select_d0_hard_candidate(d0_displacements, eligible):
    shifts = np.asarray(d0_displacements, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=np.bool_)
    if (
        shifts.shape != (ACTIVE_PROPOSALS_PER_IMAGE,)
        or eligible.shape != shifts.shape
    ):
        raise ValueError("unexpected D0-hard inputs")
    if not np.isfinite(shifts).all():
        raise ValueError("nonfinite D0 displacement")
    indices = np.flatnonzero(eligible)
    if indices.size == 0:
        return None
    ranked = sorted(indices.tolist(), key=lambda index: (-shifts[index], index))
    return int(ranked[0])


def strong_control_eligible(base_eligible, candidate_identity_safe):
    """Intersect the nuisance caliper with the full ROA/top-5 identity gate."""
    base = np.asarray(base_eligible, dtype=np.bool_)
    identity = np.asarray(candidate_identity_safe, dtype=np.bool_)
    expected = (ACTIVE_PROPOSALS_PER_IMAGE,)
    if base.shape != expected or identity.shape != expected:
        raise ValueError("strong-control eligibility shape mismatch")
    return base & identity


def d0_identity_gate(clean_top5, edited_top5, displacement, roa_displacements):
    roa = np.asarray(roa_displacements, dtype=np.float64)
    if roa.shape != (ROA_COUNT,) or not np.isfinite(roa).all():
        raise ValueError("unexpected ROA displacement vector")
    if not np.isfinite(float(displacement)):
        raise ValueError("nonfinite D0 displacement")
    p50 = percentile(roa, 0.50)
    p90 = percentile(roa, 0.90)
    identity_safe = bool(
        clean_top5
        and edited_top5
        and float(displacement) >= p50
        and float(displacement) <= p90
    )
    return {
        "clean_top5": bool(clean_top5),
        "edited_top5": bool(edited_top5),
        "displacement": float(displacement),
        "roa_p50": float(p50),
        "roa_p90": float(p90),
        "identity_safe": bool(identity_safe),
    }


def deterministic_roa_masks(relative_path, count=ROA_COUNT):
    target_area = int(round(AREA_FRACTION * IMAGE_HEIGHT * IMAGE_WIDTH))
    seed = int(
        sha256_text(SELECTION_SALT + "\0roa\0" + str(relative_path))[:16], 16
    ) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    masks = []
    seen = set()
    attempts = 0
    while len(masks) < int(count) and attempts < 4096:
        attempts += 1
        aspect = ASPECT_RATIOS[int(rng.randint(0, len(ASPECT_RATIOS)))]
        height, width = _best_dimensions(target_area, aspect)
        top = int(rng.randint(0, IMAGE_HEIGHT - height + 1))
        left = int(rng.randint(0, IMAGE_WIDTH - width + 1))
        key = (top, left, height, width)
        if key in seen:
            continue
        seen.add(key)
        mask = torch.zeros(IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.bool)
        mask[top : top + height, left : left + width] = True
        masks.append(mask)
    if len(masks) != int(count):
        raise RuntimeError("could not construct frozen ROA controls")
    return masks


FACTORIAL_ARM_NAMES = ("pc", "pose_only", "clip_only", "neither")
TRAINING_INTERVENTION_ARM_NAMES = (
    "pc",
    "pose_only",
    "clip_only",
    "neither",
    "d0_hard",
    "raw_color",
)
QUARTET_EDGE_NAMES = ("c_given_p1", "c_given_p0", "p_given_c1", "p_given_c0")


def quartet_match_decision(edge_flags):
    if tuple(sorted(edge_flags)) != tuple(sorted(QUARTET_EDGE_NAMES)):
        raise ValueError("quartet requires all four direct match edges")
    return bool(all(bool(edge_flags[name]) for name in QUARTET_EDGE_NAMES))


def finalize_factorial_rows(arm_rows):
    """Atomically propagate any single-arm record/match failure to all arms."""
    if tuple(sorted(arm_rows)) != tuple(sorted(FACTORIAL_ARM_NAMES)):
        raise ValueError("unexpected factorial arm names")
    for arm_name in FACTORIAL_ARM_NAMES:
        if len(arm_rows[arm_name]) != ORACLE_COUNT:
            raise ValueError("factorial arm must contain exactly 512 rows")
    reference_ids = [
        str(row["row_id"]) for row in arm_rows[FACTORIAL_ARM_NAMES[0]]
    ]
    if len(set(reference_ids)) != ORACLE_COUNT:
        raise ValueError("factorial row ids must be unique")
    common_match = []
    for row_index in range(ORACLE_COUNT):
        complete = True
        reference_edges = None
        for arm_name in FACTORIAL_ARM_NAMES:
            row = arm_rows[arm_name][row_index]
            edge_flags = row.get("match_edges")
            if edge_flags is None:
                edge_match = False
            else:
                try:
                    edge_match = quartet_match_decision(edge_flags)
                except ValueError:
                    edge_match = False
            if reference_edges is None:
                reference_edges = edge_flags
            elif edge_flags != reference_edges:
                edge_match = False
                complete = False
            complete = bool(
                complete
                and row.get("arm_complete", False)
                and edge_match
                and row.get("Y", None) in (0, 1, False, True)
            )
        common_match.append(complete)
    output = {}
    for arm_name in FACTORIAL_ARM_NAMES:
        rows = arm_rows[arm_name]
        row_ids = [str(row["row_id"]) for row in rows]
        if row_ids != reference_ids:
            raise ValueError("factorial row ids/order mismatch")
        values = []
        for row, matched in zip(rows, common_match):
            raw = row.get("Y", 0)
            value = int(raw) if raw in (0, 1, False, True) else 0
            values.append(value if matched else 0)
        output[arm_name] = np.asarray(values, dtype=np.float64)
    return {
        "row_ids": reference_ids,
        "quartet_matched": np.asarray(common_match, dtype=np.bool_),
        "outcomes": output,
    }


def finalize_paired_control_rows(reference_rows, control_rows):
    """Atomically zero both sides when either matched-control record fails."""
    if len(reference_rows) != ORACLE_COUNT or len(control_rows) != ORACLE_COUNT:
        raise ValueError("paired control must contain exactly 512 rows")
    reference_ids = [str(row["row_id"]) for row in reference_rows]
    control_ids = [str(row["row_id"]) for row in control_rows]
    if reference_ids != control_ids or len(set(reference_ids)) != ORACLE_COUNT:
        raise ValueError("paired control row id mismatch")
    matched = []
    left = []
    right = []
    for reference, control in zip(reference_rows, control_rows):
        pair_ok = bool(
            reference.get("arm_complete", False)
            and control.get("arm_complete", False)
            and reference.get("pair_match_ok", False)
            and control.get("pair_match_ok", False)
            and reference.get("Y", None) in (0, 1, False, True)
            and control.get("Y", None) in (0, 1, False, True)
        )
        matched.append(pair_ok)
        left.append(int(reference.get("Y", 0)) if pair_ok else 0)
        right.append(int(control.get("Y", 0)) if pair_ok else 0)
    return {
        "row_ids": reference_ids,
        "pair_matched": np.asarray(matched, dtype=np.bool_),
        "reference": np.asarray(left, dtype=np.float64),
        "control": np.asarray(right, dtype=np.float64),
    }


def factorial_interaction(y_pc, y_pose, y_clip, y_neither):
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (y_pc, y_pose, y_clip, y_neither)
    ]
    if any(array.shape != (ORACLE_COUNT,) for array in arrays):
        raise ValueError("factorial outcomes must use the fixed 512 denominator")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("nonfinite factorial outcome")
    paired = arrays[0] - arrays[1] - arrays[2] + arrays[3]
    return float(paired.mean()), paired


def paired_bootstrap_interaction(
    y_pc,
    y_pose,
    y_clip,
    y_neither,
    *,
    repetitions=10000,
):
    estimate, paired = factorial_interaction(
        y_pc, y_pose, y_clip, y_neither
    )
    seed = int(
        sha256_text(SELECTION_SALT + "\0paired-bootstrap")[:16], 16
    ) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    samples = np.empty(int(repetitions), dtype=np.float64)
    for start in range(0, int(repetitions), 256):
        batch = min(256, int(repetitions) - start)
        indices = rng.randint(0, len(paired), size=(batch, len(paired)))
        samples[start : start + batch] = paired[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.05, 0.95), method="linear")
    return {
        "estimate": float(estimate),
        "one_sided_95_lower": float(low),
        "one_sided_95_upper": float(high),
        "repetitions": int(repetitions),
        "seed": int(seed),
    }


def paired_bootstrap_difference(left, right, *, repetitions=10000, salt):
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != (ORACLE_COUNT,) or right_array.shape != (
        ORACLE_COUNT,
    ):
        raise ValueError("paired difference must use fixed 512 denominator")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("nonfinite paired difference")
    paired = left_array - right_array
    seed = int(
        sha256_text(SELECTION_SALT + "\0paired-difference\0" + str(salt))[:16],
        16,
    ) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    samples = np.empty(int(repetitions), dtype=np.float64)
    for start in range(0, int(repetitions), 256):
        batch = min(256, int(repetitions) - start)
        indices = rng.randint(0, ORACLE_COUNT, size=(batch, ORACLE_COUNT))
        samples[start : start + batch] = paired[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.05, 0.95), method="linear")
    return {
        "estimate": float(paired.mean()),
        "one_sided_95_lower": float(low),
        "one_sided_95_upper": float(high),
        "repetitions": int(repetitions),
        "seed": int(seed),
    }


def aspect_total_variation(left_indices, right_indices):
    if len(left_indices) != len(right_indices) or not left_indices:
        raise ValueError("aspect histogram input mismatch")
    left = np.bincount(
        [int(index) % len(ASPECT_RATIOS) for index in left_indices],
        minlength=len(ASPECT_RATIOS),
    ).astype(np.float64)
    right = np.bincount(
        [int(index) % len(ASPECT_RATIOS) for index in right_indices],
        minlength=len(ASPECT_RATIOS),
    ).astype(np.float64)
    left /= left.sum()
    right /= right.sum()
    return float(0.5 * np.abs(left - right).sum())


def paired_difficulty_summary(reference_shifts, control_shifts):
    reference = np.asarray(reference_shifts, dtype=np.float64)
    control = np.asarray(control_shifts, dtype=np.float64)
    if reference.shape != (ORACLE_COUNT,) or control.shape != (
        ORACLE_COUNT,
    ):
        raise ValueError("paired difficulty shape mismatch")
    if reference.size == 0 or not np.isfinite(reference).all() or not np.isfinite(
        control
    ).all():
        raise ValueError("invalid paired difficulty values")
    difference = np.abs(reference - control)
    return {
        "median_absolute_difference": float(np.median(difference)),
        "p90_absolute_difference": percentile(difference, 0.90),
        "pass": bool(
            np.median(difference) <= 0.010
            and percentile(difference, 0.90) <= 0.020
        ),
    }


def common_training_view_modes(arm_valid):
    """Return a common edited/clean-noop decision without dropping a sample."""
    if tuple(sorted(arm_valid)) != tuple(
        sorted(TRAINING_INTERVENTION_ARM_NAMES)
    ):
        raise ValueError("unexpected training arm names")
    values = [
        bool(arm_valid[name]) for name in TRAINING_INTERVENTION_ARM_NAMES
    ]
    quartet_valid = bool(all(values))
    mode = "edited" if quartet_valid else "clean_noop"
    return {
        "quartet_valid": quartet_valid,
        "view_modes": {
            name: mode for name in TRAINING_INTERVENTION_ARM_NAMES
        },
        "drop_sample": False,
    }
