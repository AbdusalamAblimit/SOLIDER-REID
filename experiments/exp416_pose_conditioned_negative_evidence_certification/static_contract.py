#!/usr/bin/env python3
"""Static and synthetic contract gate for the exp416 fuel audit."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
for import_root in (REPOSITORY_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import build_candidate_manifest
import build_fuel_cache
import clip_crop_encoder
import d0_feature_extractor
import fuel_audit
import fuel_core
import fuel_io
import geometry_census


SCHEMA = "exp416-pcnec-static-contract-v1"
EXPECTED_DESIGN_SHA256 = (
    "6d062b2782abb1bd5c9fa36a1a7500ff3105c203d70e5e9e10c6d8e03622ef77"
)
EXPECTED_INTERPRETER = Path("/usr/local/anaconda3/envs/mmpose-abu/bin/python")
FIXED_REPOSITORY_ROOT = Path("/home/afr/SOLIDER-REID-exp416-pcnec-formal-v1")
FORMAL_AUDIT_AUTHORIZED = False
SOURCE_NAMES = (
    "build_candidate_manifest.py",
    "build_fuel_cache.py",
    "clip_crop_encoder.py",
    "d0_feature_extractor.py",
    "fuel_audit.py",
    "fuel_core.py",
    "fuel_io.py",
    "geometry_census.py",
    "static_contract.py",
)


def _import_roots(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module)
    return roots


def _assert_no_imports(filename, forbidden):
    imports = _import_roots(SCRIPT_DIR / filename)
    observed = sorted(
        name
        for name in imports
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
    )
    if observed:
        raise RuntimeError(
            "{} imports forbidden modules: {}".format(
                filename, ",".join(observed)
            )
        )


def _assert_source_isolation():
    _assert_no_imports(
        "build_candidate_manifest.py",
        ("open_clip", "datasets", "mmpose"),
    )
    candidate = (SCRIPT_DIR / "build_candidate_manifest.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "OccludedDuke(",
        "PoseTargetStore(",
        ".query",
        ".gallery",
    ):
        if token in candidate:
            raise RuntimeError("candidate builder contains forbidden token: " + token)
    _assert_no_imports("geometry_census.py", ("open_clip",))
    _assert_no_imports(
        "fuel_audit.py", ("torch", "open_clip", "datasets", "model")
    )
    clip_source = (SCRIPT_DIR / "clip_crop_encoder.py").read_text(
        encoding="utf-8"
    )
    for token in ("encode_text", "get_tokenizer", "PoseTargetStore"):
        if token in clip_source:
            raise RuntimeError("crop encoder contains forbidden API: " + token)
    stage3 = (SCRIPT_DIR / "build_fuel_cache.py").read_text(encoding="utf-8")
    d0_close = stage3.find('event_log.append("d0_close")')
    clip_construct = stage3.find('event_log.append("clip_construct")')
    if min(d0_close, clip_construct) < 0 or d0_close >= clip_construct:
        raise RuntimeError("stage3 source does not close D0 before CLIP construct")


def _assert_cross_file_contracts():
    if fuel_io.sha256_file(SCRIPT_DIR / "design.md") != EXPECTED_DESIGN_SHA256:
        raise RuntimeError("exp416 design SHA changed")
    if build_candidate_manifest.SCHEMA != geometry_census.BANK_SCHEMA:
        raise RuntimeError("candidate/geometry schema mismatch")
    if build_candidate_manifest.SCHEMA != fuel_audit.BANK_SCHEMA:
        raise RuntimeError("candidate/audit schema mismatch")
    if build_fuel_cache.BANK_SCHEMA != build_candidate_manifest.SCHEMA:
        raise RuntimeError("stage3 candidate schema mismatch")
    if build_fuel_cache.GEOMETRY_SCHEMA != geometry_census.SCHEMA:
        raise RuntimeError("stage3 geometry schema mismatch")
    if build_fuel_cache.SCHEMA != fuel_audit.FUEL_SCHEMA:
        raise RuntimeError("stage3/audit fuel schema mismatch")
    if tuple(build_fuel_cache.CACHE_FIELDS) != tuple(fuel_audit.FUEL_FIELDS):
        if set(build_fuel_cache.CACHE_FIELDS) != set(fuel_audit.FUEL_FIELDS):
            raise RuntimeError("stage3/audit fuel fields mismatch")
    if tuple(fuel_core.ARM_NAMES) != (
        "correct",
        "pose_only_raw_color",
        "pose_only_student_part",
        "canonical_location_clip",
        "neither",
        "slot_shuffle",
        "wrong_rgb",
        "global_clip",
        "d0_only",
    ):
        raise RuntimeError("frozen arm order changed")
    if geometry_census.JOINT_SCORE_MIN != 0.30:
        raise RuntimeError("geometry visibility threshold changed")
    if geometry_census.MIN_VISIBLE_JOINTS != 2:
        raise RuntimeError("geometry minimum joint count changed")
    if geometry_census.MIN_QUERY_COVERAGE != 0.80:
        raise RuntimeError("geometry query coverage gate changed")
    if (
        geometry_census.MIN_COMMON_PAIRS_PER_SLOT != 100000
        or geometry_census.MIN_QUERY_PIDS_PER_SLOT != 300
    ):
        raise RuntimeError("geometry per-slot coverage gates changed")
    if (
        fuel_audit.AUROC_AP_MIN_DELTA != 0.03
        or fuel_audit.D0_MAP_R1_MIN_DELTA != 0.01
        or fuel_audit.CONTROL_MAP_R1_MIN_DELTA != 0.005
    ):
        raise RuntimeError("fuel effect gates changed")
    candidate_microbatch = build_candidate_manifest.D0_MICROBATCH
    cache_microbatch = build_fuel_cache.D0_MICROBATCH
    if (
        candidate_microbatch != 8
        or cache_microbatch != 8
        or candidate_microbatch != cache_microbatch
    ):
        raise RuntimeError("candidate/stage3 D0 microbatch mismatch")
    if build_candidate_manifest.IMPOSTOR_TOPK != 20:
        raise RuntimeError("candidate impostor top-K changed")
    if fuel_core.CAMERA_MATCHED_IMPOSTORS is not True:
        raise RuntimeError("candidate bank camera matching contract is absent")


def run_self_tests():
    fuel_io.run_self_test()
    core_result = fuel_core.run_self_test()
    if core_result.get("status") != "PASS":
        raise RuntimeError("fuel_core self-test failed")
    clip_crop_encoder.run_self_test()
    d0_feature_extractor.run_self_test()
    geometry_census.run_self_test()
    fuel_audit.run_self_test()
    build_candidate_manifest.run_self_test()
    build_fuel_cache.run_self_test()


def run_contract():
    _assert_source_isolation()
    _assert_cross_file_contracts()
    run_self_tests()
    source_sha = {
        name: fuel_io.sha256_file(SCRIPT_DIR / name) for name in SOURCE_NAMES
    }
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "source_sha256": source_sha,
        "candidate_train_only": True,
        "d0_before_clip": True,
        "cpu_adjudicator": True,
        "consumer_aligned_gate_implemented": False,
        "formal_audit_authorized": FORMAL_AUDIT_AUTHORIZED,
        "training_authorized": False,
    }


def validate_formal(args):
    if not FORMAL_AUDIT_AUTHORIZED:
        raise RuntimeError(
            "formal audit blocked: consumer-aligned residual gate is not implemented"
        )
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("formal static requires PYTHONDONTWRITEBYTECODE=1")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("formal static requires PYTHONHASHSEED=0")
    if Path(sys.executable).resolve() != EXPECTED_INTERPRETER.resolve(strict=True):
        raise RuntimeError("formal static interpreter mismatch")
    if REPOSITORY_ROOT.resolve(strict=True) != FIXED_REPOSITORY_ROOT:
        raise RuntimeError("formal static repository path mismatch")
    head = fuel_io.git_head(REPOSITORY_ROOT)
    if not args.expected_head or str(args.expected_head) != head:
        raise RuntimeError("formal static HEAD mismatch")
    if fuel_io.git_tracked_status(REPOSITORY_ROOT):
        raise RuntimeError("formal static tracked worktree is dirty")
    if fuel_io.git_index_status(REPOSITORY_ROOT):
        raise RuntimeError("formal static index is dirty")
    for name in SOURCE_NAMES:
        result = subprocess.run(
            (
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "ls-files",
                "--error-unmatch",
                str((SCRIPT_DIR / name).relative_to(REPOSITORY_ROOT)),
            ),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError("formal static source is untracked: " + name)
    fuel_io.assert_no_cuda_compute_processes()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--expected-head")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.self_test and not args.formal:
        raise ValueError("choose --self-test or --formal")
    if args.formal:
        validate_formal(args)
    result = run_contract()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
