#!/usr/bin/env python3
"""Build the once-only frozen CLIP visible/occluded text axes for exp412."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REGION_NAMES = (
    "head",
    "upper_torso_arms",
    "lower_torso",
    "upper_legs",
    "lower_legs_feet",
)
REGION_PHRASES = (
    "head, face, and hair",
    "shoulders, chest, upper torso, arms, and hands",
    "abdomen, waist, hips, lower torso, and pelvis",
    "thighs and upper legs between the hips and knees",
    "lower legs and feet below the knees",
)
SUPPORT_PROMPT_PAIRS = (
    (
        "a photo of a person with clearly visible {}",
        "a photo of a person with occluded or obstructed {}",
    ),
    (
        "the person's {} is clearly visible and unobstructed",
        "the person's {} is hidden or obstructed",
    ),
    (
        "clear visual evidence of the person's {}",
        "weak visual evidence of the person's {}",
    ),
    (
        "a fully observable human {}",
        "a heavily obscured human {}",
    ),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--clip-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parents[2]
    output = Path(args.output).expanduser()
    checkpoint = Path(args.clip_checkpoint).expanduser().resolve(strict=True)
    if not output.is_absolute():
        raise ValueError("output path must be absolute")
    if output.exists() or output.parent.exists():
        raise FileExistsError("PSGC text asset path must be fresh")
    if sha256_file(checkpoint) != args.clip_checkpoint_sha256:
        raise RuntimeError("CLIP checkpoint SHA mismatch")
    output.parent.mkdir(parents=False, exist_ok=False)

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=str(checkpoint), device="cpu"
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    prompts = []
    for phrase in REGION_PHRASES:
        for visible, occluded in SUPPORT_PROMPT_PAIRS:
            prompts.extend((visible.format(phrase), occluded.format(phrase)))
    with torch.inference_mode():
        text = model.encode_text(tokenizer(prompts))
    text = F.normalize(text.float(), dim=-1)
    text = text.view(5, len(SUPPORT_PROMPT_PAIRS), 2, -1)
    prototypes = F.normalize(text.mean(dim=1), dim=-1).cpu().numpy()
    if prototypes.shape != (5, 2, 768):
        raise RuntimeError("unexpected CLIP text embedding shape")

    prompt_spec = json.dumps(
        {
            "region_names": REGION_NAMES,
            "region_phrases": REGION_PHRASES,
            "support_prompt_pairs": SUPPORT_PROMPT_PAIRS,
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    prompt_spec_sha256 = hashlib.sha256(prompt_spec).hexdigest()
    builder_sha256 = sha256_file(Path(__file__).resolve())
    source_head = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez(
                handle,
                schema=np.asarray("exp412-psgc-text-axes-v1"),
                region_names=np.asarray(REGION_NAMES),
                visible_prototypes=prototypes[:, 0].astype(np.float32),
                occluded_prototypes=prototypes[:, 1].astype(np.float32),
                clip_checkpoint_sha256=np.asarray(
                    args.clip_checkpoint_sha256
                ),
                prompt_spec_sha256=np.asarray(prompt_spec_sha256),
                builder_sha256=np.asarray(builder_sha256),
                source_head=np.asarray(source_head),
            )
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "prompt_spec_sha256": prompt_spec_sha256,
                "builder_sha256": builder_sha256,
                "source_head": source_head,
                "prototype_shape": list(prototypes.shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
