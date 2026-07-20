#!/usr/bin/env python3
"""Reviewed trust root and sole publisher for the exp405 Phase-0 contract."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import MappingProxyType


EXPECTED_CORE_SHA256 = "29ddd00ce03ed73b6d1c7ab722de88490e2490638bc83b192e215c6ab4bb0f8b"
EXPECTED_CONTRACT_SHA256 = "13aff524a64c3341ee3d51a5d998bd29e08c3ba2131e1739c63695ed02f92f60"
OUTPUT_NAME = re.compile(r"phase0_static_[A-Za-z0-9][A-Za-z0-9_.-]*\.json\Z")
LAUNCHER_MODE = "same-source-reexec-v4"
BOOTSTRAP_PATH_ENV = "_EXP405_BOOTSTRAP_PATH"
EXPECTED_PYVENV_SHA256 = "39fe3064980027fed1216d0b9ced4da9e270652270d84f2bcfb74d694711cf48"
EXPECTED_SITE_TREE_SHA256 = "b3428d0e161d8bb6b2f98cc75301cf7ef4f67dac2cb3b5fea348b91b15faccf2"
EXPECTED_SITE_TREE_FILE_COUNT = 18763
EXPECTED_SITE_TREE_BYTE_COUNT = 585021614
EXPECTED_TORCH_RECORD_VERIFIED_FILES = 12713
DEPENDENCY_DIGESTS = {
    "torch_init_sha256": "cf40c075c95864036e835795756d69b8cccfafa76f3bcde5eba9d06065ccd3d1",
    "torch_record_sha256": "b5e76f2212a8b17cac6bf771887c4d8a647502d3e33bf7e61d720bbab1f89367",
    "torch_c_sha256": "06b303bc0e60a65552970fa2e2ca395a6f32a70ea167e3dbba7be82d2d6cbc4f",
    "libtorch_python_sha256": "eb3f31be95527c2d9ff816ddce5f282f2031f71586628d008855a315c5683bb1",
    "libtorch_cpu_sha256": "f1584a65a2a09b5ddbe90a4e195ba824087b430a9f21cb1df9b8894177b99987",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest(), size
            digest.update(chunk)
            size += len(chunk)


def hash_site_tree(root: Path) -> tuple[str, int, int, dict[str, tuple[str, int]]]:
    manifest = hashlib.sha256()
    entries = {}
    total_size = 0
    paths = sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix())
    for path in paths:
        if path.is_symlink():
            fail("frozen site-packages tree must not contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest, size = digest_file(path)
        entries[relative] = (digest, size)
        manifest.update(relative.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(str(size).encode("ascii"))
        manifest.update(b"\0")
        manifest.update(digest.encode("ascii"))
        manifest.update(b"\n")
        total_size += size
    return manifest.hexdigest(), len(entries), total_size, entries


def verify_torch_record(
    record_path: Path,
    site_packages: Path,
    venv_dir: Path,
    tree_entries: dict[str, tuple[str, int]],
) -> int:
    verified = 0
    with record_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    for row in rows:
        if len(row) != 3 or not row[0]:
            fail("invalid frozen torch RECORD row")
        candidate = (site_packages / row[0]).resolve()
        try:
            candidate.relative_to(venv_dir.resolve())
        except ValueError as error:
            raise RuntimeError("torch RECORD path escapes the frozen venv") from error
        if not row[1]:
            if candidate != record_path.resolve() or row[2]:
                fail("only torch RECORD itself may omit its digest")
            continue
        algorithm, encoded_digest = row[1].split("=", 1)
        if algorithm != "sha256" or not row[2].isdigit():
            fail("unsupported torch RECORD digest or size")
        if candidate.is_symlink() or not candidate.is_file():
            fail("torch RECORD artifact is not a regular file")
        try:
            relative = candidate.relative_to(site_packages.resolve()).as_posix()
        except ValueError:
            actual_digest, actual_size = digest_file(candidate)
        else:
            if relative not in tree_entries:
                fail("torch RECORD artifact missing from frozen site tree")
            actual_digest, actual_size = tree_entries[relative]
        padding = "=" * (-len(encoded_digest) % 4)
        expected_digest = base64.urlsafe_b64decode(encoded_digest + padding).hex()
        if actual_digest != expected_digest or actual_size != int(row[2]):
            fail("torch RECORD content digest or size mismatch")
        verified += 1
    return verified


def fail(message: str) -> None:
    raise RuntimeError(message)


def same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def unlink_if_owned(path: Path, owner: os.stat_result) -> None:
    try:
        observed = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return
    if same_inode(owner, observed):
        os.unlink(path)


def read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def publish_payload(path: Path, payload: dict) -> tuple[os.stat_result, str]:
    encoded = (
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    owner = os.fstat(descriptor)
    published = False
    try:
        view = memoryview(encoded)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                fail("short write while creating frozen payload")
            written += count
        os.fsync(descriptor)
        if not same_inode(owner, os.stat(temporary, follow_symlinks=False)):
            fail("temporary output inode ownership changed")
        if read_descriptor(descriptor) != encoded:
            fail("temporary output bytes differ from encoded payload")

        os.link(temporary, path, follow_symlinks=False)
        published = True
        final_stat = os.stat(path, follow_symlinks=False)
        if not same_inode(owner, final_stat):
            fail("published output inode differs from the open temporary file")
        final_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            final_flags |= os.O_NOFOLLOW
        final_descriptor = os.open(path, final_flags)
        try:
            if not same_inode(owner, os.fstat(final_descriptor)):
                fail("published output descriptor has the wrong inode")
            if read_descriptor(final_descriptor) != encoded:
                fail("published output bytes differ from encoded payload")
        finally:
            os.close(final_descriptor)

        unlink_if_owned(temporary, owner)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if published:
            unlink_if_owned(path, owner)
        unlink_if_owned(temporary, owner)
        raise
    finally:
        os.close(descriptor)
    return owner, sha256_bytes(encoded)


def main() -> int:
    if len(sys.argv) != 2 or OUTPUT_NAME.fullmatch(sys.argv[1]) is None:
        fail("usage: .venv/bin/python -I -S frozen_bootstrap.py phase0_static_<name>.json")
    output_name = sys.argv[1]
    if "/" in output_name or "\\" in output_name:
        fail("output must be a basename")

    executed_bootstrap_bytes = globals().get("_EXP405_EXECUTED_BOOTSTRAP_BYTES")
    reexecuted = isinstance(executed_bootstrap_bytes, bytes)
    if reexecuted:
        if not (
            sys.argv[0] == "-c"
            and sys.flags.isolated
            and sys.flags.no_site
            and sys.flags.safe_path
            and sys.flags.no_user_site
            and sys.dont_write_bytecode
        ):
            fail("frozen re-execution requires isolated no-site bytecode-free mode")
        bootstrap_path = Path(os.environ[BOOTSTRAP_PATH_ENV])
    else:
        if not (
            sys.flags.isolated
            and sys.flags.no_site
            and sys.flags.safe_path
            and sys.flags.no_user_site
        ):
            fail("initial frozen bootstrap entry requires Python -I -S")
        bootstrap_path = Path(__file__).absolute()

    script_dir = bootstrap_path.parent.resolve()
    repo_dir = script_dir.parent.parent
    venv_dir = repo_dir / ".venv"
    venv_python = venv_dir / "bin" / "python"
    if (
        not venv_python.is_file()
        or Path(sys.executable).absolute() != venv_python.absolute()
        or Path(sys.executable).resolve() != venv_python.resolve()
    ):
        fail("the repository uv-managed .venv Python is required")

    if not reexecuted:
        if bootstrap_path.is_symlink() or not bootstrap_path.is_file():
            fail("bootstrap must be a regular non-symlink file")
        launcher_bytes = bootstrap_path.read_bytes()
        wrapper = (
            "_EXP405_EXECUTED_BOOTSTRAP_BYTES = " + repr(launcher_bytes) + "\n"
            "exec(compile(_EXP405_EXECUTED_BOOTSTRAP_BYTES, "
            + repr(str(bootstrap_path))
            + ", 'exec', dont_inherit=True), globals())"
        )
        allowed_environment = {
            name: os.environ[name]
            for name in ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR")
            if name in os.environ
        }
        allowed_environment.update({
            BOOTSTRAP_PATH_ENV: str(bootstrap_path),
            "CUDA_VISIBLE_DEVICES": "",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        })
        os.execve(
            venv_python,
            [str(venv_python), "-B", "-I", "-S", "-c", wrapper, output_name],
            allowed_environment,
        )
        fail("same-source re-execution unexpectedly returned")

    required_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    if any(os.environ.get(name) != value for name, value in required_environment.items()):
        fail("deterministic frozen environment was not preserved")
    if any(name.startswith("PYTHON") for name in os.environ):
        fail("unapproved Python import environment")
    bootstrap_sha256 = sha256_bytes(executed_bootstrap_bytes)

    site_packages_path = venv_dir / "lib" / "python3.11" / "site-packages"
    if site_packages_path.is_symlink() or not site_packages_path.is_dir():
        fail("canonical venv site-packages path is unavailable")
    if any("site-packages" in entry or str(repo_dir) in entry for entry in sys.path):
        fail("no-site interpreter unexpectedly contains a non-standard import path")
    sys.path.append(str(site_packages_path))
    import_path = tuple(sys.path)

    core_path = script_dir / "phase0_core.py"
    contract_path = script_dir / "phase0_static_contract.py"
    output_path = script_dir / output_name
    receipt_path = output_path.with_suffix(".receipt.json")
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    receipt_temporary_path = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    for source_path in (core_path, contract_path):
        if source_path.is_symlink() or not source_path.is_file():
            fail("frozen sources must be regular non-symlink files")
    if output_path.exists() or output_path.is_symlink():
        fail("output path must be fresh")
    if temporary_path.exists() or temporary_path.is_symlink():
        fail("temporary output path must be fresh")
    if receipt_path.exists() or receipt_path.is_symlink():
        fail("receipt path must be fresh")
    if receipt_temporary_path.exists() or receipt_temporary_path.is_symlink():
        fail("temporary receipt path must be fresh")

    dependency_paths = {
        "torch_init_sha256": site_packages_path / "torch" / "__init__.py",
        "torch_record_sha256": (
            site_packages_path / "torch-2.13.0.dist-info" / "RECORD"
        ),
        "torch_c_sha256": (
            site_packages_path / "torch" / "_C.cpython-311-darwin.so"
        ),
        "libtorch_python_sha256": (
            site_packages_path / "torch" / "lib" / "libtorch_python.dylib"
        ),
        "libtorch_cpu_sha256": (
            site_packages_path / "torch" / "lib" / "libtorch_cpu.dylib"
        ),
    }
    pyvenv_path = venv_dir / "pyvenv.cfg"
    for dependency_path in (pyvenv_path, *dependency_paths.values()):
        if dependency_path.is_symlink() or not dependency_path.is_file():
            fail("frozen dependency artifact must be a regular non-symlink file")
    pyvenv_sha256 = sha256_bytes(pyvenv_path.read_bytes())
    site_tree_sha256, site_tree_file_count, site_tree_byte_count, tree_entries = (
        hash_site_tree(site_packages_path)
    )
    dependency_digests = {}
    for name, path in dependency_paths.items():
        relative = path.relative_to(site_packages_path).as_posix()
        dependency_digests[name] = tree_entries[relative][0]
    torch_record_verified_files = verify_torch_record(
        dependency_paths["torch_record_sha256"], site_packages_path, venv_dir,
        tree_entries,
    )
    if pyvenv_sha256 != EXPECTED_PYVENV_SHA256:
        fail("frozen pyvenv.cfg digest mismatch")
    if (
        site_tree_sha256 != EXPECTED_SITE_TREE_SHA256
        or site_tree_file_count != EXPECTED_SITE_TREE_FILE_COUNT
        or site_tree_byte_count != EXPECTED_SITE_TREE_BYTE_COUNT
        or torch_record_verified_files != EXPECTED_TORCH_RECORD_VERIFIED_FILES
    ):
        fail("frozen site-packages dependency closure mismatch")
    if dependency_digests != DEPENDENCY_DIGESTS:
        fail("frozen PyTorch artifact digest mismatch")

    core_bytes = core_path.read_bytes()
    contract_bytes = contract_path.read_bytes()
    core_sha256 = sha256_bytes(core_bytes)
    contract_sha256 = sha256_bytes(contract_bytes)
    if core_sha256 != EXPECTED_CORE_SHA256:
        fail("frozen core SHA256 mismatch")
    if contract_sha256 != EXPECTED_CONTRACT_SHA256:
        fail("frozen contract SHA256 mismatch")

    execution_sentinel = object()
    context = MappingProxyType({
        "bootstrap_path": str(bootstrap_path),
        "bootstrap_sha256": bootstrap_sha256,
        "contract_bytes": contract_bytes,
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha256,
        "core_bytes": core_bytes,
        "core_path": str(core_path),
        "core_sha256": core_sha256,
        "execution_sentinel": execution_sentinel,
        "import_path": import_path,
        "launcher_mode": LAUNCHER_MODE,
        "pyvenv_sha256": pyvenv_sha256,
        "site_packages_path": str(site_packages_path),
        "site_tree_byte_count": site_tree_byte_count,
        "site_tree_file_count": site_tree_file_count,
        "site_tree_sha256": site_tree_sha256,
        "torch_record_verified_files": torch_record_verified_files,
        **dependency_digests,
    })
    namespace = {
        "__file__": str(contract_path),
        "__name__": "exp405_phase0_static_contract",
        "_EXP405_BOOTSTRAP_CONTEXT": context,
        "_EXP405_EXECUTED_CONTRACT_BYTES": contract_bytes,
        "_EXP405_EXECUTION_SENTINEL": execution_sentinel,
    }
    code = compile(contract_bytes, str(contract_path), "exec", dont_inherit=True)
    exec(code, namespace)
    payload = namespace["main"](output_name)
    if not isinstance(payload, dict):
        fail("contract returned a non-dictionary payload")
    gates = payload.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or not all(type(value) is bool for value in gates.values())
        or payload.get("passed") != sum(gates.values())
        or payload.get("total") != len(gates)
        or payload.get("status") != ("PASS" if all(gates.values()) else "FAIL")
    ):
        fail("contract payload status is internally inconsistent")
    post_tree_sha256, post_tree_files, post_tree_bytes, _ = hash_site_tree(
        site_packages_path
    )
    if (
        post_tree_sha256 != site_tree_sha256
        or post_tree_files != site_tree_file_count
        or post_tree_bytes != site_tree_byte_count
    ):
        fail("site-packages dependency closure changed during contract execution")
    exit_code = 0 if payload["status"] == "PASS" else 1
    payload_owner = None
    try:
        payload_owner, payload_sha256 = publish_payload(output_path, payload)
        receipt = {
            "bootstrap_sha256": bootstrap_sha256,
            "canonical_payload_sha256": payload_sha256,
            "contract_sha256": contract_sha256,
            "core_sha256": core_sha256,
            "execution_receipt_sha256": sha256_bytes(
                f"{output_name}\0{payload_sha256}\0{bootstrap_sha256}".encode("utf-8")
            ),
            "output_name": output_name,
            "site_tree_sha256": site_tree_sha256,
            "status": payload["status"],
        }
        publish_payload(receipt_path, receipt)
    except BaseException:
        if payload_owner is not None:
            unlink_if_owned(output_path, payload_owner)
        directory = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        raise
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
