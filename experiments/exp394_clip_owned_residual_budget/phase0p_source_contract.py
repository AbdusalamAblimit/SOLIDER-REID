#!/usr/bin/env python3
"""Static source-seam contract before exp394 production implementation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


EXPECTED_SHA256 = {
    "model/tapf.py": "559b75f1aad9973828f7298789f50d6b8e7fd536d648423d3468ee5903f0f1ba",
    "model/make_model.py": "87603a7eb2f26d599d0d3e755fe9997ae168197351b2823bd5eff0b823e9f4b0",
    "model/backbones/swin_transformer.py": "b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef",
    "processor/processor.py": "5b0886cb16ec0e9020d39ed14bc119e8e35c88661148b7af8b1208c9edda4904",
    "config/defaults.py": "b67365bd7f238a3263abf165e863386dcde0766cfa38c7f89e885eb856f63005",
    "datasets/pose_dataset.py": "d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc",
    "model/clip_semantic_teacher.py": "50c2607394f81573788ade6c1337f173753763cd35d69925a4645dbee695de79",
    "configs/occluded_duke/swin_tiny_tapf_semantic_rz_c0.yml": "f409cc069b6f3500e009e6d40681e8baf9547bb77b864e9f35a7ea02ca11d1a6",
}
EXPECTED_EXECUTION_HEAD = "09340f76f84502f9018bee3c8eec005961b0a8cb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dotted_name(node: ast.AST) -> str | None:
    names = []
    current = node
    while isinstance(current, ast.Attribute):
        names.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        names.append(current.id)
        return ".".join(reversed(names))
    return None


def assignment_values(tree: ast.AST) -> dict[str, object]:
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not isinstance(value, ast.Constant):
            continue
        for target in targets:
            name = dotted_name(target)
            if name is not None:
                result[name] = value.value
    return result


def find_named(tree: ast.AST, kind: type, name: str) -> ast.AST:
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, kind) and getattr(node, "name", None) == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one %s named %s, found %d"
            % (kind.__name__, name, len(matches))
        )
    return matches[0]


def node_source(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise RuntimeError("Could not recover AST source segment")
    return segment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    sources = {}
    trees = {}
    actual_sha = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha[relative] = sha256_file(path)
        if path.suffix == ".py":
            sources[relative] = path.read_text(encoding="utf-8")
            trees[relative] = ast.parse(
                sources[relative], filename=str(path)
            )

    defaults = assignment_values(trees["config/defaults.py"])
    tapf_source = sources["model/tapf.py"]
    make_source = sources["model/make_model.py"]
    swin_source = sources["model/backbones/swin_transformer.py"]
    processor_source = sources["processor/processor.py"]
    pose_source = sources["datasets/pose_dataset.py"]
    teacher_source = sources["model/clip_semantic_teacher.py"]

    semantic_class = find_named(
        trees["model/tapf.py"], ast.ClassDef, "CleanSemanticTapfC0"
    )
    router_class = find_named(
        trees["model/tapf.py"], ast.ClassDef, "SemanticSpatialRouter"
    )
    router_forward = find_named(router_class, ast.FunctionDef, "forward")
    semantic_prepare = find_named(semantic_class, ast.FunctionDef, "prepare")
    semantic_apply = find_named(semantic_class, ast.FunctionDef, "apply_gate")
    do_train = find_named(
        trees["processor/processor.py"], ast.FunctionDef, "do_train"
    )
    do_inference = find_named(
        trees["processor/processor.py"], ast.FunctionDef, "do_inference"
    )

    semantic_text = node_source(tapf_source, semantic_class)
    router_text = node_source(tapf_source, router_class)
    router_forward_text = node_source(tapf_source, router_forward)
    semantic_prepare_text = node_source(tapf_source, semantic_prepare)
    semantic_apply_text = node_source(tapf_source, semantic_apply)
    train_text = node_source(processor_source, do_train)
    inference_text = node_source(processor_source, do_inference)

    router_args = [argument.arg for argument in router_forward.args.args]
    checks = {
        "sealed_source_sha_exact": actual_sha == EXPECTED_SHA256,
        "tapf_default_off": defaults.get("_C.MODEL.TAPF.ENABLED") is False,
        "semantic_default_off": (
            defaults.get("_C.MODEL.TAPF.SEMANTIC_ENABLED") is False
        ),
        "production_flag_absent_before_implementation": (
            "EVIDENCE_BUDGET" not in sources["config/defaults.py"]
            and "CleanEvidenceBudget" not in tapf_source
        ),
        "production_config_absent_before_implementation": not (
            repo_root
            / "configs/occluded_duke/swin_tiny_tapf_evidence_budget_c0.yml"
        ).exists(),
        "swin_two_live_stage3_consumers": (
            "i == 3" in swin_source
            and "for bank_index, block in enumerate(stage.blocks)" in swin_source
            and "x = tapf.apply_gate(" in swin_source
            and "len(stage.blocks) != 2" in swin_source
        ),
        "semantic_two_independent_routers": (
            "self.psg_bank = nn.ModuleList" in semantic_text
            and "for _ in range(2)" in semantic_text
        ),
        "router_production_signature_exact": router_args == [
            "self", "tokens", "hw_shape", "mask", "support"
        ],
        "router_has_token_context_expert": all(
            marker in router_text
            for marker in (
                "self.token_projection",
                "self.context_projection",
                "self.expert",
                "region_delta",
            )
        ),
        "semantic_anchor_source_detached": (
            "self.anchor(source_feature.detach())" in semantic_prepare_text
        ),
        "semantic_state_detached_before_router": (
            "consumer_mask = (" in semantic_prepare_text
            and ").detach()" in semantic_prepare_text
            and "state[\"consumer_mask\"]" in semantic_apply_text
            and "state[\"consumer_support\"]" in semantic_apply_text
        ),
        "teacher_constructed_only_in_train": (
            "FrozenClipSlotTeacher" in train_text
            and "FrozenClipSlotTeacher" not in inference_text
        ),
        "teacher_targets_no_grad_detached": (
            "with torch.no_grad(), amp.autocast" in train_text
            and "semantic_targets[" in train_text
            and "].detach().clone()" in train_text
        ),
        "checkpoint_saves_model_state_only": (
            "torch.save(model.state_dict()" in train_text
            and "semantic_teacher.state_dict" not in train_text
        ),
        "eval_has_no_pose_or_teacher_construction": (
            "semantic_teacher" not in inference_text
            and "FrozenClip" not in inference_text
        ),
        "pose_dataset_provides_pre_re_teacher_rgb": (
            'pose_batch["teacher_rgb"] = torch.stack' in pose_source
            and "Mixed teacher-RGB availability" in pose_source
        ),
        "teacher_is_frozen_visual_only_seam": (
            "parameter.requires_grad_(False)" in teacher_source
            and "self.visual = model.visual" in teacher_source
            and "@torch.inference_mode()" in teacher_source
        ),
        "existing_semantic_selection_isolated": (
            "if cfg.MODEL.TAPF.SEMANTIC_ENABLED" in make_source
            and "tapf_class = CleanSemanticTapfC0" in make_source
            and "Semantic fast-track is single-stage only" in make_source
        ),
        "current_router_has_no_rho_budget": (
            "rho_star" not in router_forward_text
            and "EVIDENCE_BUDGET" not in make_source
        ),
    }

    payload = {
        "scope": "EXP394_PHASE0P_STATIC_SOURCE_SEAM",
        "verdict": (
            "PHASE0P_SOURCE_PASS" if all(checks.values())
            else "PHASE0P_SOURCE_FAIL"
        ),
        "expected_execution_head": EXPECTED_EXECUTION_HEAD,
        "checks": checks,
        "source_sha256": actual_sha,
        "production_absent": True,
        "authorization": (
            "PASS authorizes only a fresh-repo production implementation; "
            "CUDA and formal training remain NO-START"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("output_sha256=%s" % sha256_file(args.output))
    return 0 if payload["verdict"] == "PHASE0P_SOURCE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
