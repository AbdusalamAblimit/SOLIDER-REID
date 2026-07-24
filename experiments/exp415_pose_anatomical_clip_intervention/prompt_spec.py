"""Frozen color-only prompt specification for the PACIT revision-3 selector."""

from __future__ import annotations

import hashlib
import json


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

TEMPLATES = (
    "a cropped photo of a person wearing {} clothing",
    "a person whose visible clothing is {}",
    "a pedestrian wearing a {} garment",
)


def prompt_payload():
    prompts = [
        [template.format(color) for template in TEMPLATES]
        for color in COLOR_NAMES
    ]
    return {
        "schema": "exp415-pacit-color-prompt-v3",
        "color_names": list(COLOR_NAMES),
        "templates": list(TEMPLATES),
        "prompts": prompts,
        "selector_scope": "standard whole-image original/edited encoder",
        "margin": "cos(v,t_k)-mean_{j!=k}(cos(v,t_j))",
        "forbidden_inputs": [
            "pose",
            "slot_name",
            "pose_crop",
            "pose_region_feature",
            "d0_feature",
        ],
    }


def prompt_spec_sha256():
    encoded = json.dumps(
        prompt_payload(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def flattened_prompts():
    prompts = []
    layout = []
    for color_prompts in prompt_payload()["prompts"]:
        indices = []
        for prompt in color_prompts:
            indices.append(len(prompts))
            prompts.append(prompt)
        layout.append(indices)
    return prompts, layout
