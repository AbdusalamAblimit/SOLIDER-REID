#!/usr/bin/env python3
"""Pure tensor utilities for the exp405 Phase-0 measurement contract.

This module intentionally has no data, filesystem, CUDA, model, or CLIP loading
side effects.  It defines only the operations that the later teacher-only
measurement is required to use and audit.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Real
import operator
import unicodedata
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F


def _safe_l2_normalize(value: torch.Tensor, *, dim: int) -> torch.Tensor:
    """L2-normalize finite tensors without overflow on finite extreme values."""
    scale = value.abs().amax(dim=dim, keepdim=True)
    if not torch.isfinite(scale).all():
        raise ValueError("non-finite normalization scale")
    tiny = torch.finfo(value.dtype).tiny
    scaled = torch.where(scale > 0, value / scale.clamp_min(tiny), value)
    norm = (scaled * scaled).sum(dim=dim, keepdim=True).sqrt()
    output = torch.where(norm > 0, scaled / norm.clamp_min(tiny), scaled)
    if not torch.isfinite(output).all():
        raise ValueError("non-finite normalized output")
    return output


def _mix_int64(value: torch.Tensor) -> torch.Tensor:
    """Deterministic avalanche mixer over signed int64 tensor words."""
    value = value ^ (value >> 30)
    value = value * -4658895280553007687
    value = value ^ (value >> 27)
    value = value * -7723592293110705685
    return value ^ (value >> 31)


def stable_sample_keys(relative_paths: Iterable[str]) -> torch.Tensor:
    """Map canonical relative sample paths to stable int64 deletion keys."""
    paths = list(relative_paths)
    if not paths:
        raise ValueError("relative_paths must be non-empty")
    keys = []
    for path in paths:
        if not isinstance(path, str) or not path or "\\" in path:
            raise ValueError("sample paths must be non-empty canonical strings")
        if unicodedata.normalize("NFC", path) != path:
            raise ValueError("sample paths must already use Unicode NFC")
        if any(unicodedata.category(character).startswith("C") for character in path):
            raise ValueError("sample paths must not contain control characters")
        parts = path.split("/")
        if path.startswith("/") or any(part in ("", ".", "..") for part in parts):
            raise ValueError("sample paths must be normalized and relative")
        digest = hashlib.sha256(("exp405/sample/" + path).encode("utf-8")).digest()
        keys.append(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
    if len(set(paths)) != len(paths):
        raise ValueError("sample paths must be unique within an intervention batch")
    if len(set(keys)) != len(keys):
        raise ValueError("stable sample-key collision")
    return torch.tensor(keys, dtype=torch.int64)


def mass_normalized_pool(
    feature: torch.Tensor,
    masks: torch.Tensor,
    *,
    min_mass: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool BxCxHxW features with BxKxHxW masks without slot mixing."""
    if isinstance(min_mass, bool) or not isinstance(min_mass, Real):
        raise ValueError("min_mass must be a real scalar")
    min_mass = float(min_mass)
    if not math.isfinite(min_mass) or min_mass <= 0:
        raise ValueError("min_mass must be positive and finite")
    if feature.ndim != 4 or masks.ndim != 4:
        raise ValueError("feature and masks must be rank four")
    if not feature.is_floating_point() or not masks.is_floating_point():
        raise ValueError("pool tensors must use floating dtypes")
    if feature.dtype != masks.dtype or feature.device != masks.device:
        raise ValueError("pool feature/masks must share exact dtype and device")
    if min(feature.shape) <= 0 or masks.shape[1] <= 0:
        raise ValueError("pool dimensions must be positive")
    if feature.shape[0] != masks.shape[0] or feature.shape[-2:] != masks.shape[-2:]:
        raise ValueError("feature/mask geometry mismatch")
    if not torch.isfinite(feature).all() or not torch.isfinite(masks).all():
        raise ValueError("non-finite feature or mask")
    if bool(((masks < 0) | (masks > 1)).any()):
        raise ValueError("masks must lie in [0, 1]")

    mass = masks.sum(dim=(-2, -1))
    geometry_valid = mass >= float(min_mass)
    pooled = torch.einsum("bkhw,bchw->bkc", masks, feature)
    pooled = pooled / mass.clamp_min(1e-12).unsqueeze(-1)
    pooled = torch.where(geometry_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))
    if not torch.isfinite(pooled).all() or not torch.isfinite(mass).all():
        raise ValueError("non-finite pooled output")
    return pooled, geometry_valid, mass


def deterministic_slot_delete(
    feature: torch.Tensor,
    mask: torch.Tensor,
    sample_key: torch.Tensor,
    slot_key: torch.Tensor,
    *,
    fraction: float,
    fill: float = 0.0,
    ordering_seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Delete a hashed fraction of each binary slot and return its exact support."""
    if feature.ndim != 4 or mask.ndim != 3:
        raise ValueError("feature must be BxCxHxW and mask BxHxW")
    if not feature.is_floating_point() or mask.dtype != torch.bool:
        raise ValueError("deletion requires floating feature and boolean mask")
    if min(feature.shape) <= 0:
        raise ValueError("deletion dimensions must be positive")
    if feature.shape[0] != mask.shape[0] or feature.shape[-2:] != mask.shape[-2:]:
        raise ValueError("feature/mask geometry mismatch")
    if mask.device != feature.device:
        raise ValueError("deletion feature/mask must share device")
    if sample_key.shape != (feature.shape[0],) or slot_key.shape != sample_key.shape:
        raise ValueError("sample/slot keys must have one value per sample")
    if sample_key.dtype not in (torch.int32, torch.int64) or slot_key.dtype not in (
        torch.int32, torch.int64
    ):
        raise ValueError("sample/slot keys must use an integer dtype")
    if sample_key.device != feature.device or slot_key.device != feature.device:
        raise ValueError("sample/slot keys must share the feature device")
    if isinstance(fraction, bool) or not isinstance(fraction, Real):
        raise ValueError("fraction must be a real scalar")
    if not math.isfinite(float(fraction)) or not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("fraction outside [0, 1]")
    if isinstance(fill, bool) or not isinstance(fill, Real):
        raise ValueError("fill must be a real scalar")
    if not math.isfinite(float(fill)):
        raise ValueError("fill must be finite")
    if isinstance(ordering_seed, bool):
        raise ValueError("ordering_seed must be an integer")
    try:
        ordering_seed = operator.index(ordering_seed)
    except TypeError as error:
        raise ValueError("ordering_seed must be an integer") from error
    ordering_seed %= 2147483647
    if not torch.isfinite(feature).all() or not torch.isfinite(mask).all():
        raise ValueError("non-finite feature or mask")
    output = feature.clone()
    erased_mask = torch.zeros_like(mask, dtype=torch.bool)
    realized = []
    for row in range(feature.shape[0]):
        indices = torch.nonzero(mask[row].reshape(-1) > 0, as_tuple=False).flatten()
        seed_word = torch.tensor(
            ordering_seed, dtype=torch.int64, device=feature.device
        )
        joint = (
            indices.to(torch.int64)
            ^ sample_key[row].to(torch.int64)
            ^ (slot_key[row].to(torch.int64) * 2246822519)
            ^ (seed_word * 2654435761)
        )
        key = _mix_int64(joint)
        indices = indices[torch.argsort(key, stable=True)]
        count = int(indices.numel())
        erase = int(math.floor(float(fraction) * count + 0.5))
        flat = output[row].reshape(output.shape[1], -1)
        if erase:
            flat[:, indices[:erase]] = float(fill)
            erased_mask[row].reshape(-1)[indices[:erase]] = True
        realized.append(erase / float(count) if count else 0.0)
    if not torch.isfinite(output).all():
        raise ValueError("non-finite deletion output")
    return output, erased_mask, torch.tensor(
        realized, dtype=torch.float64, device=feature.device
    )


def scatter_replace(
    base: torch.Tensor,
    mask: torch.Tensor,
    state: torch.Tensor,
    *,
    budget: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Replace a slot by a BxC state; zero budget is exact tensor identity."""
    if base.ndim != 4 or mask.ndim != 3 or state.ndim != 2:
        raise ValueError("base/mask/state ranks are invalid")
    if base.shape[:2] != state.shape or base.shape[0] != mask.shape[0]:
        raise ValueError("batch/channel mismatch")
    if base.shape[-2:] != mask.shape[-2:]:
        raise ValueError("spatial mismatch")
    if mask.dtype != torch.bool:
        raise ValueError("scatter mask must be the boolean erased support")
    if not base.is_floating_point() or not state.is_floating_point():
        raise ValueError("scatter base/state must use floating dtypes")
    if mask.device != base.device or state.device != base.device:
        raise ValueError("scatter tensors must share device")
    if state.dtype != base.dtype:
        raise ValueError("scatter base/state must share exact dtype")
    if not torch.isfinite(base).all() or not torch.isfinite(mask).all() or not torch.isfinite(state).all():
        raise ValueError("non-finite scatter input")
    if bool(((mask < 0) | (mask > 1)).any()):
        raise ValueError("scatter mask must lie in [0, 1]")

    if not torch.is_tensor(budget):
        if isinstance(budget, bool) or not isinstance(budget, Real):
            raise ValueError("scalar budget must be a real number")
        budget = torch.full(
            (base.shape[0],), float(budget), dtype=base.dtype, device=base.device
        )
    elif (
        not budget.is_floating_point()
        or budget.dtype != base.dtype
        or budget.device != base.device
    ):
        raise ValueError("tensor budget must share the base floating dtype and device")
    if budget.ndim != 1 or budget.shape != (base.shape[0],):
        raise ValueError("budget must have one value per sample")
    if not torch.isfinite(budget).all():
        raise ValueError("budget must be finite")
    if bool(((budget < 0) | (budget > 1)).any()):
        raise ValueError("budget outside [0, 1]")

    alpha = mask.to(dtype=base.dtype).unsqueeze(1) * budget[:, None, None, None]
    proposal = state[:, :, None, None].expand_as(base)
    output = base * (1.0 - alpha) + proposal * alpha
    if not torch.isfinite(output).all():
        raise ValueError("non-finite scatter output")
    return output


def clip_slot_state(
    visual: torch.Tensor,
    text_prototypes: torch.Tensor,
    visible_text: torch.Tensor,
    occluded_text: torch.Tensor,
    geometry_valid: torch.Tensor,
    *,
    logit_scale: float,
) -> Dict[str, torch.Tensor]:
    """Return sample-specific p/q/v without converting geometry into support."""
    if visual.ndim != 3 or text_prototypes.ndim != 2:
        raise ValueError("visual must be BxKxD and text prototypes KxD")
    if min(visual.shape) <= 0:
        raise ValueError("CLIP state dimensions must be positive")
    if visual.shape[1:] != text_prototypes.shape:
        raise ValueError("visual/text slot shape mismatch")
    if visible_text.shape != text_prototypes.shape or occluded_text.shape != text_prototypes.shape:
        raise ValueError("support prototype shape mismatch")
    if geometry_valid.shape != visual.shape[:2] or geometry_valid.dtype != torch.bool:
        raise ValueError("geometry_valid must be a BxK boolean tensor")
    if geometry_valid.device != visual.device:
        raise ValueError("geometry_valid and visual must share device")
    tensors = (visual, text_prototypes, visible_text, occluded_text)
    if not all(value.is_floating_point() for value in tensors):
        raise ValueError("CLIP state tensors must use floating dtypes")
    if any(value.dtype != visual.dtype or value.device != visual.device for value in tensors[1:]):
        raise ValueError("CLIP state tensors must share exact dtype and device")
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("non-finite CLIP state input")
    if isinstance(logit_scale, bool) or not isinstance(logit_scale, Real):
        raise ValueError("logit_scale must be a real scalar")
    logit_scale = float(logit_scale)
    if not math.isfinite(logit_scale) or logit_scale <= 0:
        raise ValueError("logit_scale must be positive and finite")

    converted = tuple(value.float() for value in tensors)
    if not all(bool(torch.isfinite(value).all()) for value in converted):
        raise ValueError("CLIP state overflow during float32 conversion")
    visual32, text32, visible32, occluded32 = converted
    if bool((visual32.abs().amax(dim=-1)[geometry_valid] == 0).any()):
        raise ValueError("valid slots require non-zero visual vectors")
    if any(bool((value.abs().amax(dim=-1) == 0).any()) for value in (
        text32, visible32, occluded32
    )):
        raise ValueError("text prototypes must have non-zero norm")
    v = _safe_l2_normalize(visual32, dim=-1)
    text = _safe_l2_normalize(text32, dim=-1)
    logits = float(logit_scale) * torch.einsum("bkd,jd->bkj", v, text)
    distribution = logits.softmax(dim=-1)
    visible = _safe_l2_normalize(visible32, dim=-1)
    occluded = _safe_l2_normalize(occluded32, dim=-1)
    visible_logit = float(logit_scale) * (v * visible.unsqueeze(0)).sum(-1)
    occluded_logit = float(logit_scale) * (v * occluded.unsqueeze(0)).sum(-1)
    support = torch.sigmoid(visible_logit - occluded_logit)
    support = torch.where(geometry_valid, support, torch.zeros_like(support))
    if not torch.isfinite(distribution).all() or not torch.isfinite(support).all():
        raise ValueError("non-finite CLIP state output")
    return {
        "visual": v,
        "distribution": distribution,
        "support": support,
        "geometry_valid": geometry_valid,
    }


def semantic_transport_budget(
    recipient_distribution: torch.Tensor,
    donor_distribution: torch.Tensor,
    donor_support: torch.Tensor,
    recipient_valid: torch.Tensor,
    donor_valid: torch.Tensor,
    slot: torch.Tensor,
) -> torch.Tensor:
    """CLIP-dependent budget for a chosen anatomical slot."""
    if recipient_distribution.ndim != 3 or donor_distribution.ndim != 3:
        raise ValueError("distributions must be BxKxK")
    if min(recipient_distribution.shape) <= 0:
        raise ValueError("semantic budget dimensions must be positive")
    if recipient_distribution.shape != donor_distribution.shape:
        raise ValueError("distribution shape mismatch")
    if recipient_distribution.shape[1] != recipient_distribution.shape[2]:
        raise ValueError("slot distribution must be square")
    if donor_support.shape != recipient_distribution.shape[:2]:
        raise ValueError("support shape mismatch")
    if recipient_valid.shape != recipient_distribution.shape[:2] or donor_valid.shape != recipient_valid.shape:
        raise ValueError("geometry validity shape mismatch")
    if recipient_valid.dtype != torch.bool or donor_valid.dtype != torch.bool:
        raise ValueError("geometry validity must be boolean")
    batch = recipient_distribution.shape[0]
    if slot.shape != (batch,):
        raise ValueError("slot must have one index per sample")
    if slot.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot must use an integer dtype")
    if any(value.device != recipient_distribution.device for value in (
        donor_distribution, donor_support, recipient_valid, donor_valid, slot
    )):
        raise ValueError("semantic budget tensors must share device")
    if any(value.dtype != recipient_distribution.dtype for value in (
        donor_distribution, donor_support
    )) or not recipient_distribution.is_floating_point():
        raise ValueError("semantic distributions/support must share a floating dtype")
    if bool(((slot < 0) | (slot >= recipient_distribution.shape[1])).any()):
        raise ValueError("slot index out of range")
    distributions = (recipient_distribution, donor_distribution)
    if not all(bool(torch.isfinite(value).all()) for value in (*distributions, donor_support)):
        raise ValueError("non-finite semantic budget input")
    if any(bool((value < 0).any()) for value in distributions):
        raise ValueError("semantic distributions must be non-negative")
    if not all(bool(torch.allclose(
        value.sum(dim=-1), torch.ones_like(value[..., 0]), atol=1e-5, rtol=1e-5
    )) for value in distributions):
        raise ValueError("semantic distributions must sum to one")
    if bool(((donor_support < 0) | (donor_support > 1)).any()):
        raise ValueError("donor support must lie in [0, 1]")
    index = torch.arange(batch, device=slot.device)
    recipient = recipient_distribution[index, slot]
    donor = donor_distribution[index, slot]
    agreement = F.cosine_similarity(recipient, donor, dim=-1).clamp(0.0, 1.0)
    support = donor_support[index, slot]
    valid = recipient_valid[index, slot] & donor_valid[index, slot]
    budget = torch.where(valid, agreement * support, torch.zeros_like(support))
    if not torch.isfinite(budget).all():
        raise ValueError("non-finite semantic budget output")
    return budget


def contextual_transport_state(
    donor_slot: torch.Tensor,
    donor_context: torch.Tensor,
    recipient_context: torch.Tensor,
) -> torch.Tensor:
    """Zero-parameter teacher probe; not a proposed learnable operator."""
    if donor_slot.ndim != 2 or min(donor_slot.shape) <= 0:
        raise ValueError("transport tensors must be non-empty BxC matrices")
    if donor_slot.shape != donor_context.shape or donor_slot.shape != recipient_context.shape:
        raise ValueError("transport tensors must share shape")
    if not donor_slot.is_floating_point() or any(
        value.dtype != donor_slot.dtype or value.device != donor_slot.device
        for value in (donor_context, recipient_context)
    ):
        raise ValueError("transport tensors must share a floating dtype and device")
    if not all(bool(torch.isfinite(value).all()) for value in (
        donor_slot, donor_context, recipient_context
    )):
        raise ValueError("non-finite transport input")
    output = donor_slot - donor_context.detach() + recipient_context.detach()
    if not torch.isfinite(output).all():
        raise ValueError("non-finite transport output")
    return output


def pid_disjoint_split(
    pids: torch.Tensor,
    *,
    seed: int,
    fit_fraction: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministically split whole identities; no PID can cross the boundary."""
    if pids.ndim != 1 or len(pids) == 0:
        raise ValueError("pids must be a non-empty vector")
    if pids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("pids must use an integer dtype")
    if isinstance(fit_fraction, bool) or not isinstance(fit_fraction, Real):
        raise ValueError("fit_fraction must be a real scalar")
    fit_fraction = float(fit_fraction)
    if not math.isfinite(fit_fraction) or not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in (0, 1)")
    unique = torch.unique(pids.cpu(), sorted=True)
    if len(unique) < 2:
        raise ValueError("at least two identities are required")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    try:
        seed = operator.index(seed)
    except TypeError as error:
        raise ValueError("seed must be an integer") from error
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = unique[torch.randperm(len(unique), generator=generator)]
    fit_count = max(1, min(len(order) - 1, int(math.floor(len(order) * fit_fraction))))
    fit_pids = set(int(value) for value in order[:fit_count].tolist())
    fit = torch.tensor([int(value) in fit_pids for value in pids.cpu().tolist()])
    return fit, ~fit


def recipient_not_k_features(
    recipient_slots: torch.Tensor,
    target_slot: torch.Tensor,
) -> torch.Tensor:
    """Keep fixed anatomical columns, zero target content, and append its address."""
    if recipient_slots.ndim != 3 or recipient_slots.shape[1] < 2:
        raise ValueError("recipient_slots must be BxKxC with K >= 2")
    if not recipient_slots.is_floating_point() or not torch.isfinite(recipient_slots).all():
        raise ValueError("recipient slots must be finite floating tensors")
    batch, slots, channels = recipient_slots.shape
    if min(batch, channels) <= 0:
        raise ValueError("recipient slot dimensions must be positive")
    if target_slot.shape != (batch,) or target_slot.dtype not in (torch.int32, torch.int64):
        raise ValueError("target_slot must be one integer per sample")
    if target_slot.device != recipient_slots.device:
        raise ValueError("target_slot and recipient_slots must share device")
    if bool(((target_slot < 0) | (target_slot >= slots)).any()):
        raise ValueError("target_slot is out of range")
    keep = (
        torch.arange(slots, device=recipient_slots.device).unsqueeze(0)
        != target_slot[:, None]
    )
    masked = recipient_slots * keep.unsqueeze(-1).to(recipient_slots.dtype)
    address = F.one_hot(target_slot.to(torch.long), num_classes=slots).to(
        recipient_slots.dtype
    )
    output = torch.cat((masked.flatten(1), address), dim=1)
    if not torch.isfinite(output).all():
        raise ValueError("non-finite recipient-only probe output")
    return output


def ridge_fit_predict(
    fit_x: torch.Tensor,
    fit_y: torch.Tensor,
    test_x: torch.Tensor,
    *,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """Small deterministic CPU probe used only for donor-free realizability."""
    if fit_x.ndim != 2 or fit_y.ndim != 2 or test_x.ndim != 2:
        raise ValueError("ridge tensors must be matrices")
    if min(fit_x.shape) <= 0 or fit_y.shape[1] <= 0 or min(test_x.shape) <= 0:
        raise ValueError("ridge tensor dimensions must be positive")
    if len(fit_x) == 0 or len(test_x) == 0 or len(fit_x) != len(fit_y):
        raise ValueError("ridge sample dimensions are invalid")
    if fit_x.shape[1] != test_x.shape[1]:
        raise ValueError("ridge feature dimensions differ")
    if any(value.device.type != "cpu" for value in (fit_x, fit_y, test_x)):
        raise ValueError("ridge probe is CPU-only")
    if not all(value.is_floating_point() for value in (fit_x, fit_y, test_x)):
        raise ValueError("ridge tensors must use floating dtypes")
    if any(value.dtype != fit_x.dtype for value in (fit_y, test_x)):
        raise ValueError("ridge tensors must share exact dtype")
    if isinstance(ridge, bool) or not isinstance(ridge, Real):
        raise ValueError("ridge must be a real scalar")
    ridge = float(ridge)
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive and finite")
    if not all(bool(torch.isfinite(value).all()) for value in (fit_x, fit_y, test_x)):
        raise ValueError("non-finite ridge input")
    x = torch.cat((fit_x.double(), torch.ones(len(fit_x), 1)), dim=1)
    test = torch.cat((test_x.double(), torch.ones(len(test_x), 1)), dim=1)
    gram = x.T @ x
    identity = torch.eye(gram.shape[0], dtype=torch.float64)
    identity[-1, -1] = 0.0
    weights = torch.linalg.solve(gram + float(ridge) * identity, x.T @ fit_y.double())
    output = (test @ weights).to(dtype=fit_y.dtype)
    if not torch.isfinite(output).all():
        raise ValueError("non-finite ridge output")
    return output


def regression_scores(target: torch.Tensor, prediction: torch.Tensor) -> Dict[str, float]:
    if target.shape != prediction.shape:
        raise ValueError("target/prediction mismatch")
    if target.ndim != 2 or len(target) == 0:
        raise ValueError("regression tensors must be non-empty matrices")
    if target.shape[1] <= 0 or not target.is_floating_point() or not prediction.is_floating_point():
        raise ValueError("regression tensors must have positive floating width")
    if not torch.isfinite(target).all() or not torch.isfinite(prediction).all():
        raise ValueError("non-finite regression input")
    if target.device != prediction.device or target.dtype != prediction.dtype:
        raise ValueError("regression tensors must share exact dtype and device")
    target64 = target.double()
    prediction64 = prediction.double()
    common_scale = torch.stack((target64.abs().max(), prediction64.abs().max())).max()
    if not torch.isfinite(common_scale):
        raise ValueError("non-finite regression scale")
    common_scale = common_scale.clamp_min(1e-300)
    target_scaled = target64 / common_scale
    prediction_scaled = prediction64 / common_scale
    cosine = (
        _safe_l2_normalize(target_scaled, dim=-1)
        * _safe_l2_normalize(prediction_scaled, dim=-1)
    ).sum(dim=-1).mean()
    residual = ((target_scaled - prediction_scaled) ** 2).sum()
    centered = target_scaled - target_scaled.mean(dim=0, keepdim=True)
    denominator = (centered ** 2).sum().clamp_min(1e-12)
    r2 = 1.0 - residual / denominator
    if not torch.isfinite(cosine) or not torch.isfinite(r2):
        raise ValueError("non-finite regression score")
    return {"cosine": float(cosine), "r2": float(r2)}


def reid_map_r1(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_pids: Iterable[int],
    gallery_pids: Iterable[int],
    query_camids: Iterable[int],
    gallery_camids: Iterable[int],
    gallery_sample_keys: Iterable[int],
    *,
    junk_pid: int = -1,
) -> Dict[str, float]:
    """Camera-aware cosine mAP/R1 with same-PID/same-camera and junk removal."""
    if query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != gallery.shape[1]:
        raise ValueError("query/gallery descriptor shape mismatch")
    if query.shape[1] <= 0 or not query.is_floating_point() or not gallery.is_floating_point():
        raise ValueError("query/gallery descriptors must have positive floating width")
    if not torch.isfinite(query).all() or not torch.isfinite(gallery).all():
        raise ValueError("non-finite descriptor")
    if len(query) == 0 or len(gallery) == 0:
        raise ValueError("query/gallery must be non-empty")
    query = query.detach().cpu().double()
    gallery = gallery.detach().cpu().double()
    if bool((query.abs().amax(dim=-1) == 0).any()) or bool((gallery.abs().amax(dim=-1) == 0).any()):
        raise ValueError("zero-norm descriptor")
    query = _safe_l2_normalize(query, dim=-1)
    gallery = _safe_l2_normalize(gallery, dim=-1)
    def strict_integers(values: Iterable[int], name: str) -> list[int]:
        result = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError(f"{name} must not contain booleans")
            try:
                result.append(operator.index(value))
            except TypeError as error:
                raise ValueError(f"{name} must contain integer scalars") from error
        return result

    q_pids = strict_integers(query_pids, "query_pids")
    g_pid_values = strict_integers(gallery_pids, "gallery_pids")
    q_camids = strict_integers(query_camids, "query_camids")
    g_camid_values = strict_integers(gallery_camids, "gallery_camids")
    if isinstance(junk_pid, bool):
        raise ValueError("junk_pid must be an integer")
    try:
        junk_pid = operator.index(junk_pid)
    except TypeError as error:
        raise ValueError("junk_pid must be an integer") from error
    g_key_values = strict_integers(gallery_sample_keys, "gallery_sample_keys")
    g_pids = torch.tensor(g_pid_values, dtype=torch.long)
    g_camids = torch.tensor(g_camid_values, dtype=torch.long)
    g_keys = torch.tensor(g_key_values, dtype=torch.long)
    if len(q_pids) != len(query) or len(q_camids) != len(query):
        raise ValueError("query metadata length mismatch")
    if len(g_pids) != len(gallery) or len(g_camids) != len(gallery):
        raise ValueError("gallery metadata length mismatch")
    if len(g_keys) != len(gallery) or len(set(g_key_values)) != len(gallery):
        raise ValueError("gallery sample keys must be unique and complete")
    scores = query @ gallery.T
    average_precisions = []
    rank1 = []
    for row, pid in enumerate(q_pids):
        keep = (g_pids != int(junk_pid)) & ~(
            (g_pids == pid) & (g_camids == q_camids[row])
        )
        kept_scores = scores[row, keep]
        kept_pids = g_pids[keep]
        kept_keys = g_keys[keep]
        key_order = torch.argsort(kept_keys, stable=True)
        score_order = torch.argsort(
            kept_scores[key_order], descending=True, stable=True
        )
        matches = (kept_pids[key_order][score_order] == pid).double()
        if not bool(matches.any()):
            continue
        cumulative = matches.cumsum(0)
        ranks = torch.arange(1, len(matches) + 1, dtype=torch.float64)
        average_precisions.append(float(((cumulative / ranks) * matches).sum() / matches.sum()))
        rank1.append(float(matches[0]))
    if not average_precisions:
        raise ValueError("no query has a cross-camera gallery positive")
    return {
        "mAP": 100.0 * sum(average_precisions) / len(average_precisions),
        "R1": 100.0 * sum(rank1) / len(rank1),
        "valid_queries": len(average_precisions),
    }
