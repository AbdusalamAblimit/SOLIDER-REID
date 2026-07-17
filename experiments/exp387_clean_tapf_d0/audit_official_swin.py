"""Executable audit of official SOLIDER semantic-weight and with_cp paths."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from config import cfg as default_cfg
from model.backbones.swin_transformer import SwinTransformer


OFFICIAL_COMMIT = "8c08e1c3255e8e1e51e006bf189e52cc57b009ed"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_source(path):
    return subprocess.check_output(
        ["git", "show", "{}:{}".format(OFFICIAL_COMMIT, path)], text=True
    )


def small_swin(with_cp=False, semantic_weight=-1.0):
    return SwinTransformer(
        pretrain_img_size=(64, 32),
        patch_size=4,
        window_size=4,
        embed_dims=16,
        depths=(1, 1, 1, 1),
        num_heads=(1, 2, 4, 8),
        strides=(4, 2, 2, 2),
        out_indices=(3,),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cp=with_cp,
        semantic_weight=semantic_weight,
    )


def audit_static_wiring():
    defaults = official_source("config/defaults.py")
    make_model = official_source("model/make_model.py")
    swin = official_source("model/backbones/swin_transformer.py")
    has_config_key = "MODEL.WITH_CP" in defaults
    wrapper_passes_flag = "with_cp=cfg.MODEL.WITH_CP" in make_model
    backbone_supports_flag = "with_cp=False" in swin and "cp.checkpoint" in swin
    semantic_uses_hardcoded_cuda = "semantic_weight = w.cuda()" in swin
    if has_config_key or wrapper_passes_flag or not backbone_supports_flag:
        raise RuntimeError("Unexpected official with_cp static wiring")
    if not semantic_uses_hardcoded_cuda:
        raise RuntimeError("Expected official hard-coded semantic CUDA placement")

    config = default_cfg.clone()
    config.defrost()
    merge_rejected = False
    rejection = None
    try:
        config.merge_from_list(["MODEL.WITH_CP", "True"])
    except (KeyError, AssertionError) as error:
        merge_rejected = True
        rejection = "{}: {}".format(type(error).__name__, error)
    if not merge_rejected:
        raise RuntimeError("Official config unexpectedly accepts MODEL.WITH_CP")
    return {
        "official_commit": OFFICIAL_COMMIT,
        "defaults_has_model_with_cp": has_config_key,
        "make_model_passes_with_cp": wrapper_passes_flag,
        "backbone_supports_with_cp": backbone_supports_flag,
        "semantic_uses_hardcoded_cuda": semantic_uses_hardcoded_cuda,
        "config_merge_rejected": merge_rejected,
        "config_merge_error": rejection,
    }


def audit_terminal_semantic_weight():
    torch.manual_seed(1234)
    model = small_swin(with_cp=False, semantic_weight=0.2)
    eval_return = model.eval()
    if eval_return is not None:
        raise RuntimeError("Expected the official backbone eval return bug")
    image = torch.randn(2, 3, 64, 32)
    semantic = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    with torch.no_grad():
        reference, _ = model(image, semantic_weight=semantic)

        terminal_w = model.semantic_embed_w[3]
        terminal_b = model.semantic_embed_b[3]
        terminal_w.weight.copy_(
            torch.linspace(-50.0, 50.0, terminal_w.weight.numel()).view_as(
                terminal_w.weight
            )
        )
        terminal_w.bias.copy_(
            torch.linspace(-10.0, 10.0, terminal_w.bias.numel())
        )
        terminal_b.weight.copy_(
            torch.linspace(40.0, -40.0, terminal_b.weight.numel()).view_as(
                terminal_b.weight
            )
        )
        terminal_b.bias.copy_(
            torch.linspace(7.0, -7.0, terminal_b.bias.numel())
        )
        terminal_changed, _ = model(image, semantic_weight=semantic)
        terminal_exact = torch.equal(reference, terminal_changed)

        live_w = model.semantic_embed_w[2]
        live_b = model.semantic_embed_b[2]
        live_w.weight.copy_(
            torch.linspace(-2.0, 2.0, live_w.weight.numel()).view_as(
                live_w.weight
            )
        )
        live_w.bias.copy_(torch.linspace(-1.0, 1.0, live_w.bias.numel()))
        live_b.weight.copy_(
            torch.linspace(1.5, -1.5, live_b.weight.numel()).view_as(
                live_b.weight
            )
        )
        live_b.bias.copy_(torch.linspace(0.5, -0.5, live_b.bias.numel()))
        live_changed, _ = model(image, semantic_weight=semantic)

    live_max_abs = float((reference - live_changed).abs().max().item())
    if not terminal_exact:
        raise RuntimeError("Terminal semantic branch unexpectedly affects descriptor")
    if live_max_abs <= 0.0:
        raise RuntimeError("Earlier semantic branch unexpectedly has no effect")
    return {
        "terminal_index": 3,
        "terminal_extreme_mutation_descriptor_exact": terminal_exact,
        "earlier_live_index": 2,
        "earlier_mutation_descriptor_max_abs": live_max_abs,
        "backbone_eval_returns_none": eval_return is None,
        "interpretation": (
            "semantic stages 0-2 can affect later stages; terminal stage-3 "
            "output is computed after the last consumer and is dead for descriptor"
        ),
    }


def audit_direct_checkpoint_core():
    torch.manual_seed(4321)
    plain = small_swin(with_cp=False, semantic_weight=-1.0)
    checkpointed = small_swin(with_cp=True, semantic_weight=-1.0)
    checkpointed.load_state_dict(plain.state_dict(), strict=True)
    plain_train_return = plain.train()
    checkpointed_train_return = checkpointed.train()
    if plain_train_return is not None or checkpointed_train_return is not None:
        raise RuntimeError("Expected the official backbone train return bug")
    image_plain = torch.randn(2, 3, 64, 32, requires_grad=True)
    image_checkpoint = image_plain.detach().clone().requires_grad_(True)

    plain_global, plain_outs = plain(image_plain)
    checkpoint_global, checkpoint_outs = checkpointed(image_checkpoint)
    forward_max_abs = float(
        (plain_global - checkpoint_global).abs().max().item()
    )
    if len(plain_outs) != len(checkpoint_outs):
        raise RuntimeError("Checkpoint output count mismatch")
    for left, right in zip(plain_outs, checkpoint_outs):
        forward_max_abs = max(
            forward_max_abs, float((left - right).abs().max().item())
        )

    plain_loss = plain_global.square().mean() + sum(
        output.square().mean() for output in plain_outs
    )
    checkpoint_loss = checkpoint_global.square().mean() + sum(
        output.square().mean() for output in checkpoint_outs
    )
    plain_loss.backward()
    checkpoint_loss.backward()
    input_grad_max_abs = float(
        (image_plain.grad - image_checkpoint.grad).abs().max().item()
    )
    parameter_grad_max_abs = 0.0
    compared_gradients = 0
    for (plain_name, plain_parameter), (cp_name, cp_parameter) in zip(
        plain.named_parameters(), checkpointed.named_parameters()
    ):
        if plain_name != cp_name:
            raise RuntimeError("Checkpoint parameter order mismatch")
        if (plain_parameter.grad is None) != (cp_parameter.grad is None):
            raise RuntimeError("Checkpoint gradient ownership mismatch")
        if plain_parameter.grad is not None:
            compared_gradients += 1
            parameter_grad_max_abs = max(
                parameter_grad_max_abs,
                float(
                    (plain_parameter.grad - cp_parameter.grad).abs().max().item()
                ),
            )

    direct_flags = [
        block.with_cp for stage in checkpointed.stages for block in stage.blocks
    ]
    if not all(direct_flags):
        raise RuntimeError("Direct with_cp did not reach every Swin block")
    tolerance = 1e-6
    if max(forward_max_abs, input_grad_max_abs, parameter_grad_max_abs) > tolerance:
        raise RuntimeError("Direct with_cp numerical parity failed")
    return {
        "all_block_flags_true": all(direct_flags),
        "block_count": len(direct_flags),
        "forward_max_abs": forward_max_abs,
        "input_grad_max_abs": input_grad_max_abs,
        "parameter_grad_max_abs": parameter_grad_max_abs,
        "compared_parameter_gradients": compared_gradients,
        "backbone_train_returns_none": True,
        "tolerance": tolerance,
        "interpretation": "checkpoint core works when wired directly",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        "status": "EXP387_OFFICIAL_SWIN_AUDIT_PASS",
        "script_sha256": sha256_file(__file__),
        "static_wiring": audit_static_wiring(),
        "semantic_weight": audit_terminal_semantic_weight(),
        "direct_with_cp": audit_direct_checkpoint_core(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
